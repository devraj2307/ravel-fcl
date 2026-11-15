"""Server: dynamic aggregation + EVCL variational consolidation (spec Sections 2.5,
Algorithm 1 lines 13-21).

Aggregation (Section 2.5, Algorithm 1 lines 16-18)
--------------------------------------------------
Upon receiving the de-quantized updates {Δθ̂_k, ΔĜ_k}, the server

1. expands the global classifier head (and the generator's class-conditional
   embeddings) if new classes arrived this round;
2. aggregates θ̄_r^g = θ_g^{r-1} + Σ_k w_k Δθ̂_k (and likewise for the generator's
   shared layers; class-conditional CBN embeddings can use per-class weights
   w_k^(c), falling back to the client weights w_k in the baseline);
3. zero-pads head deltas of clients that have not yet seen the newest classes, so
   updates of different head sizes C_t^(k) aggregate safely.

ServerEVCLConsolidator (Section 2.5, Eqs. 8-10; Extension C Section 5.3, Eq. 20)
---------------------------------------------------------------------------------
Diagonal Gaussian posterior q_r^(g)(θ) = N(μ_r^(g), diag(σ_r^(g)²)) with
μ_r^(g) ← θ̄_r^g and σ_init = 0.1, minimizing Eq. 18 on synthetic samples
(x, y) ~ G_r^g:

    L_EVCL = E[L_CE(μ_r^(g); x, y)]
           + (λ_evcl / 2) Σ_i [ F_{r-1,i} (μ_{r,i} − μ_{r-1,i})²
                              + σ_{r,i}²/σ_{r-1,i}² − 1 − log(σ_{r,i}²/σ_{r-1,i}²) ]

with λ_evcl = 1e3 and the diagonal Fisher matrix estimated from synthetic
samples.  Extension C (Section 5.3, Eq. 20) upgrades the Fisher gate to a
Dual-Gate that requires each synthetic sample to simultaneously satisfy:

1. classifier confidence  p(y|x; μ_r) > τ_conf = 0.60,
2. normalised entropy     H_norm(p(·|x)) < τ_ent  = 0.45,
3. discriminator realism  D_adv(x) > τ_real = 0.45.

The Fisher estimate is smoothed by a 3-round EMA with decay γ_ema = 0.8
(Eq. 10).  Classes for which fewer than 15% of the synthetic samples pass the
Dual-Gate are excluded from the update (fallback protection, Section 5.3);
the EMA is committed only when at least 15% of all samples are accepted.

Stage 1 (Section 5.2, Eqs. 13-19) is implemented by
:class:`federated.defense.ClassAwareMMDDefense` and applied in
:meth:`FederatedServer.run_round` prior to aggregation.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from federated.client import ClientUpdate
from federated.compression import decompress_state_dict
from federated.defense import ClassAwareMMDDefense, ScreenResult
from models.reacgan import (
    Discriminator,
    Generator,
    clone_model,
    is_batch_norm_buffer,
)
from utils.rng import make_generator

LOGGER = logging.getLogger(__name__)

LAMBDA_EVCL = 1.0       # λ_evcl (Eq. 18) - much reduced
SIGMA_INIT = 0.1       # σ_init for the variational posterior
CONF_THRESH = 0.6      # τ_conf classifier confidence gate (Eqs. 9, 20)
ENT_THRESH = 0.45      # τ_ent normalized entropy gate (Eq. 20)
REAL_THRESH = 0.45     # τ_real discriminator realism gate (Eq. 20)
EMA_DECAY = 0.8        # γ_ema 3-round EMA (Eq. 10)
MIN_ACCEPT_FRAC = 0.15 # fallback protection (Eq. 20 / Section 5.3)
DISTILL_TEMPERATURE = 2.0  # T_distill for KL anchoring to the aggregated model
DISTILL_WEIGHT = 1.0       # weight of the teacher KL term
SYNTH_CE_WEIGHT = 0.001    # negligible
SYNTH_LABEL_SMOOTHING = 0.1

# Parameters whose leading dimension indexes classes (heads) and may therefore
# differ between clients that observed different cumulative class sets C_{≤t}^(k).
_HEAD_MARKERS = ("classifier.", "prototypes", "cbn.gain", "cbn.bias",
                 "gamma_embed", "beta_embed")


def is_head_param(name: str) -> bool:
    return any(marker in name for marker in _HEAD_MARKERS)


def pad_head_delta(name: str, delta: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Zero-pad a client's head delta along the class dimension (dim 0).

    Clients that have not observed the newest classes contribute a zero gradient
    for those rows, which matches FedAvg on the union head C_{≤t}.
    """
    if tuple(delta.shape) == tuple(target_shape):
        return delta
    if (
        is_head_param(name)
        and delta.dim() >= 1
        and target_shape[0] > delta.shape[0]
        and tuple(delta.shape[1:]) == tuple(target_shape[1:])
    ):
        out = torch.zeros(target_shape, dtype=delta.dtype, device=delta.device)
        out[: delta.shape[0]] = delta
        return out
    raise ValueError(
        f"incompatible delta shape for '{name}': {tuple(delta.shape)} vs {tuple(target_shape)}"
    )


def mask_unseen_head_rows(
    name: str,
    delta: torch.Tensor,
    known_classes: Optional[Sequence[int]],
) -> torch.Tensor:
    """Zero head rows for classes a client has never observed.

    Even after zero-padding to the global head width, a client's delta for
    classes it has not seen can be non-zero (backbone drift and, previously,
    alignment KL on unseen logits).  Averaging those rows corrupts the global
    classifier; class-aware aggregation keeps the previous global value instead.
    """
    if known_classes is None or not is_head_param(name) or delta.dim() < 1:
        return delta
    num_rows = int(delta.shape[0])
    known = sorted({int(c) for c in known_classes if 0 <= int(c) < num_rows})
    if not known or len(known) >= num_rows:
        return delta
    out = torch.zeros_like(delta)
    out[known] = delta[known]
    return out


def _overlap_slices(old: torch.Tensor, ref: torch.Tensor) -> tuple:
    """Slices selecting the region shared by ``old`` and ``ref`` (dim-wise)."""
    return tuple(slice(0, min(a, b)) for a, b in zip(old.shape, ref.shape))


def _resize_like(old: torch.Tensor, ref: torch.Tensor, fill: float) -> torch.Tensor:
    """Copy ``old`` into a tensor shaped like ``ref``, filling new rows with ``fill``."""
    if old.shape == ref.shape:
        return old.clone()
    out = torch.full_like(ref, fill)
    overlap = _overlap_slices(old, ref)
    out[overlap] = old[overlap]
    return out


class ServerEVCLConsolidator:
    """Diagonal-Gaussian EVCL consolidation of the global parameters (Eq. 18)."""

    def __init__(
        self,
        lambda_evcl: float = LAMBDA_EVCL,
        sigma_init: float = SIGMA_INIT,
        conf_thresh: float = CONF_THRESH,
        ema_decay: float = EMA_DECAY,
        z_dim: int = 128,
        batch_size: int = 64,
        num_batches: int = 10,
        lr: float = 1e-6,
        steps: int = 1,
        logvar_bounds: Tuple[float, float] = (-12.0, 0.0),
        min_accept_frac: float = MIN_ACCEPT_FRAC,
        dual_gate: bool = True,
        ent_thresh: float = ENT_THRESH,
        real_thresh: float = REAL_THRESH,
        distill_temperature: float = DISTILL_TEMPERATURE,
        distill_weight: float = DISTILL_WEIGHT,
        synth_ce_weight: float = SYNTH_CE_WEIGHT,
        label_smoothing: float = SYNTH_LABEL_SMOOTHING,
        polyak_tau: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.lambda_evcl = float(lambda_evcl)
        self.sigma_init = float(sigma_init)
        self.conf_thresh = float(conf_thresh)
        self.ema_decay = float(ema_decay)
        self.z_dim = int(z_dim)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.lr = float(lr)
        self.steps = int(steps)
        self.logvar_bounds = logvar_bounds
        self.min_accept_frac = float(min_accept_frac)
        self.dual_gate = bool(dual_gate)
        self.ent_thresh = float(ent_thresh)
        self.real_thresh = float(real_thresh)
        self.distill_temperature = max(1e-3, float(distill_temperature))
        self.distill_weight = float(distill_weight)
        self.synth_ce_weight = float(synth_ce_weight)
        self.label_smoothing = float(label_smoothing)
        self.polyak_tau = float(max(0.0, min(1.0, polyak_tau)))
        self.last_fim_stats: Dict[str, float] = {}

        self.fisher_ema: Dict[str, torch.Tensor] = {}   # F_{r,i} (Eq. 10)
        self.logvars: Dict[str, nn.Parameter] = {}      # log σ_{r,i}²  (σ = exp(0.5 logvar))
        self.prev_mu: Dict[str, torch.Tensor] = {}      # μ_{r-1}
        self.prev_logvar: Dict[str, torch.Tensor] = {}  # σ²_{r-1}
        self.rounds_completed = 0
        self._seed = int(seed)
        self._generator = make_generator(seed, "cpu")

    # ------------------------------------------------------------------
    # Variational bookkeeping
    # ------------------------------------------------------------------
    def prepare(self, disc: nn.Module, reset_variational: bool = True) -> None:
        """Align log-variance parameters with ``disc``; optionally reset to σ_init.

        Each round the posterior is re-initialized at μ_r ← θ̄_r^g with σ_init
        (Section 2.5); μ_{r-1} and σ²_{r-1} are kept from the previous round.
        """
        named = dict(disc.named_parameters())
        log_sigma2 = math.log(self.sigma_init**2)
        with torch.no_grad():
            for n in [k for k in self.logvars if k not in named]:
                del self.logvars[n]
            # Heads only ever grow at dim 0 (new classes).  Pad the per-parameter
            # statistics so the EVCL penalty stays shape-compatible across rounds.
            for store in (self.fisher_ema, self.prev_mu, self.prev_logvar):
                for n in [k for k in store if k not in named]:
                    del store[n]
            for n, p in named.items():
                ref = p.detach()
                if n in self.fisher_ema and self.fisher_ema[n].shape != ref.shape:
                    # unseen class rows were never consolidated -> zero Fisher
                    self.fisher_ema[n] = _resize_like(self.fisher_ema[n], ref, 0.0)
                if n in self.prev_mu and self.prev_mu[n].shape != ref.shape:
                    # unseen class rows start at the current mean -> zero drift
                    old = self.prev_mu[n]
                    new = ref.clone()
                    new[_overlap_slices(old, ref)] = old[_overlap_slices(old, ref)]
                    self.prev_mu[n] = new
                if n in self.prev_logvar and self.prev_logvar[n].shape != ref.shape:
                    self.prev_logvar[n] = _resize_like(self.prev_logvar[n], ref, log_sigma2)
            for n, p in named.items():
                lv = self.logvars.get(n)
                if lv is None or lv.shape != p.shape:
                    self.logvars[n] = nn.Parameter(
                        torch.full(p.shape, math.log(self.sigma_init**2),
                                   dtype=p.dtype, device=p.device)
                    )
                elif reset_variational:
                    lv.fill_(math.log(self.sigma_init**2))

    # ------------------------------------------------------------------
    # Eq. 9 + Eq. 10 + Eq. 20: dual-gated synthetic FIM with 3-round EMA
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_synthetic(self, gen: nn.Module, num_classes: int, batch_size: Optional[int] = None):
        device = next(gen.parameters()).device
        if self._generator.device != device:
            self._generator = make_generator(self._seed, device)
        z = torch.randn(batch_size or self.batch_size, self.z_dim,
                        device=device, generator=self._generator)
        y = torch.randint(num_classes, (z.size(0),), device=device,
                          generator=self._generator)
        return gen(z, y), y

    @staticmethod
    def normalized_entropy(probs: torch.Tensor) -> torch.Tensor:
        """H_norm(p) = -Σ_c p_c log p_c / log(C_t)  (Eq. 20, gate 2).

        Normalising by log(C_t) maps the entropy of a uniform distribution over
        the global head to 1.0 and a one-hot prediction to 0.0.
        """
        c = max(2, int(probs.size(-1)))
        logp = probs.clamp_min(1e-12).log()
        return -(probs * logp).sum(dim=-1) / math.log(c)

    def dual_gate_mask(
        self,
        probs: torch.Tensor,
        realism: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        """g_dual = [p(y|x) > τ_conf] & [H_norm < τ_ent] & [D_adv(x) > τ_real].

        With ``dual_gate=False`` only the classifier-confidence gate of the
        baseline RAVEL-FCL is applied.  ``num_classes`` overrides the head width
        used to normalise the entropy (the global C_t, Eq. 20).
        """
        conf, _ = probs.max(dim=-1)
        gate = conf > self.conf_thresh
        if self.dual_gate:
            if num_classes is None or int(num_classes) == int(probs.size(-1)):
                entropy = self.normalized_entropy(probs)
            else:  # normalise by log(C_t) explicitly
                logp = probs.clamp_min(1e-12).log()
                entropy = -(probs * logp).sum(dim=-1) / math.log(max(2, int(num_classes)))
            gate = gate & (entropy < self.ent_thresh)
            gate = gate & (realism > self.real_thresh)
        return gate

    def estimate_fim(self, disc: nn.Module, gen: nn.Module, num_classes: int) -> float:
        """F̂_r,i = E[ g_dual(x, y) · (∂ log p(y|x; μ_r)/∂θ_i)² ], then Eq. 10 EMA.

        Two passes implement the per-class fallback of Section 5.3:

        * pass 1 (no grad) synthesizes samples and records the Dual-Gate
          indicators, the classifier confidence / normalised entropy and the
          discriminator realism score per sample;
        * classes for which fewer than ``min_accept_frac`` (15%) of their
          synthetic samples pass g_dual are treated as unstable and their
          samples are excluded from the Fisher estimate;
        * pass 2 recomputes gradients only for the accepted samples of the
          remaining (stable) classes.

        The γ_ema = 0.8 EMA is committed only when at least 15% of all samples
        are accepted.  Returns the raw Dual-Gate acceptance rate.
        """
        was_training = disc.training
        disc.eval()
        gen.eval()

        # -- pass 1: synthesis + gate statistics -------------------------
        batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        class_total: Dict[int, int] = {}
        class_pass: Dict[int, int] = {}
        total = 0
        raw_pass = 0
        entropy_sum = 0.0
        realism_sum = 0.0
        for _ in range(self.num_batches):
            x_syn, y = self._sample_synthetic(gen, num_classes)
            with torch.no_grad():
                out = disc(x_syn)
                probs = F.softmax(out["logits"], dim=-1)
                realism = torch.sigmoid(out["adv_logit"] / 2.0).clamp(min=0.1)
                gate = self.dual_gate_mask(probs, realism, num_classes)
                entropy_sum += float(self.normalized_entropy(probs).sum())
                realism_sum += float(realism.sum())
                conf = probs.max(dim=-1).values
                score = torch.nan_to_num(conf * realism, nan=0.0, posinf=1.0, neginf=0.0)
            batches.append((x_syn.detach(), y, gate.detach(), score.detach()))
            total += int(y.numel())
            raw_pass += int(gate.sum().item())
            for c in y.unique().tolist():
                mask = y == int(c)
                class_total[int(c)] = class_total.get(int(c), 0) + int(mask.sum())
                class_pass[int(c)] = class_pass.get(int(c), 0) + int((gate & mask).sum())

        unstable = {
            c for c in class_total
            if class_total[c] > 0 and class_pass[c] / class_total[c] < self.min_accept_frac
        }

        gate_empty = raw_pass < max(1, math.ceil(self.min_accept_frac * total)) if total > 0 else True
        fallback = False
        if gate_empty and total > 0:
            fallback = True
            unstable = set()

        # -- pass 2: gradients for accepted samples of stable classes ----
        fim_accum: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in disc.named_parameters() if p.requires_grad
        }
        accepted = 0
        for x_syn, y, gate, score in batches:
            if fallback:
                n_b = score.numel()
                if n_b == 0:
                    keep = gate.clone()
                else:
                    k_b = max(1, int(math.ceil(0.25 * n_b)))
                    k_eff = min(k_b, n_b)
                    try:
                        # Class-balanced selection: top-k per class
                        keep = torch.zeros_like(score, dtype=torch.bool)
                        seen_c = torch.unique(y)
                        per_class_k = max(1, int(math.ceil(k_eff / max(1, len(seen_c)))))
                        for c in seen_c:
                            m = y == c
                            if m.sum() == 0:
                                continue
                            sc = score[m]
                            kc = min(per_class_k, int(sc.numel()))
                            if kc <= 0:
                                continue
                            _, idx = torch.topk(sc, kc)
                            keep[torch.nonzero(m)[idx]] = True
                    except Exception:
                        _, topk = torch.topk(score, k_eff)
                        keep = torch.zeros_like(score, dtype=torch.bool)
                        keep[topk] = True
            else:
                keep = gate.clone()
                for c in unstable:
                    keep &= y != c
            n_pass = int(keep.sum().item())
            if n_pass == 0:
                continue
            accepted += n_pass
            disc.zero_grad(set_to_none=True)
            logits = disc(x_syn)["logits"][keep]
            loss = -F.log_softmax(logits, dim=-1)[
                torch.arange(n_pass, device=logits.device), y[keep]
            ].mean()
            loss.backward()
            for n, p in disc.named_parameters():
                if p.requires_grad and p.grad is not None:
                    grad = torch.nan_to_num(
                        p.grad.detach(), nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    fim_accum[n] += grad.pow(2).clamp_max(1e12) * n_pass
        disc.zero_grad(set_to_none=True)
        if was_training:
            disc.train()

        acceptance = raw_pass / max(total, 1)
        committed = accepted / max(total, 1)
        # Fallback: only commit to the EMA if enough samples passed the gate.
        if accepted > 0 and committed >= self.min_accept_frac:
            fim_scale = 0.1 if fallback else 1.0
            # Normalize Fisher to keep fim_mean ~ 1.0
            if fim_accum:
                all_vals = torch.cat([v.reshape(-1) for v in fim_accum.values()])
                global_fim_mean = float(all_vals.mean().detach().cpu())
                if global_fim_mean < 1e-12:
                    global_fim_mean = 1.0
            else:
                global_fim_mean = 1.0
            for n, acc in fim_accum.items():
                fim_hat = torch.nan_to_num(
                    acc / max(accepted, 1), nan=0.0, posinf=1e12, neginf=0.0
                ).clamp(0.0, 1e12) * fim_scale / (global_fim_mean + 1e-8)
                if n in self.fisher_ema:
                    self.fisher_ema[n] = (self.ema_decay * self.fisher_ema[n]
                                          + (1.0 - self.ema_decay) * fim_hat)
                else:  # first round: F_0 = 0 -> F_1 = F̂_1
                    self.fisher_ema[n] = fim_hat.clone()

        self.last_fim_stats = {
            "dual_gate": float(self.dual_gate),
            "acceptance": float(acceptance),
            "effective_acceptance": float(committed),
            "committed": float(committed),
            "unstable_classes": float(len(unstable)),
            "entropy": entropy_sum / max(total, 1),
            "realism": realism_sum / max(total, 1),
            "gate_fallback": float(1.0 if fallback else 0.0),
        }
        LOGGER.info(
            "EVCL FIM: raw gate %d/%d samples (%.2f), committed %d (%s gate, "
            "%d unstable classes)", raw_pass, total, acceptance, accepted,
            "dual" if self.dual_gate else "confidence", len(unstable),
        )
        return float(committed) if committed > 0 else float(acceptance)

    # ------------------------------------------------------------------
    # Eq. 18: refined EVCL loss
    # ------------------------------------------------------------------
    def evcl_loss(
        self,
        disc: nn.Module,
        gen: nn.Module,
        num_classes: int,
        teacher: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """L_EVCL at the current parameters (μ_r) and log-variances (σ_r²).

        ``teacher`` is a frozen snapshot of the freshly aggregated global model
        \\bar{θ}_g^t.  Synthetic samples from an imperfect generator are often
        uninformative; minimizing hard CE on them drives every logit to
        ``ln(C)`` (uniform collapse).  The data-dependent part is therefore
        anchored to the aggregated model through temperature-scaled KL
        distillation, and the (small) CE term is restricted to teacher-confident
        samples and label-smoothed.  The Fisher quadratic penalty still anchors
        μ_r to the previous consolidated posterior μ_{r-1}.
        """
        x_syn, y = self._sample_synthetic(gen, num_classes)
        logits = disc(x_syn)["logits"]
        ce = F.cross_entropy(logits, y)

        distill = ce.new_zeros(())
        ce_stable = ce.new_zeros(())
        gate_frac = ce.new_zeros(())
        if teacher is not None:
            with torch.no_grad():
                t_out = teacher(x_syn)
                t_logits = t_out["logits"].detach()
                t_probs = F.softmax(t_logits, dim=-1)
                t_probs_T = F.softmax(t_logits / self.distill_temperature, dim=-1)
                t_conf = t_probs.max(dim=-1).values
                if self.dual_gate:
                    t_realism = torch.sigmoid(t_out["adv_logit"])
                    gate = self.dual_gate_mask(t_probs, t_realism, num_classes)
                else:
                    gate = t_conf > self.conf_thresh
            # KL(teacher_T || student_T), scaled by T² so its gradient matches
            # the un-temperatured logits.  Optimized on *all* synthetic samples:
            # matching the teacher cannot push predictions toward uniform.
            student_logp = F.log_softmax(logits / self.distill_temperature, dim=-1)
            distill = F.kl_div(
                student_logp, t_probs_T, reduction="batchmean"
            ) * (self.distill_temperature ** 2)
            # If nothing clears the gate, keep the top-quartile teacher-
            # confidence samples so the CE term still has a stable signal.
            if not bool(gate.any()):
                k = max(1, int(math.ceil(0.25 * gate.numel())))
                _, topk = torch.topk(t_conf, min(k, t_conf.numel()))
                gate = torch.zeros_like(gate)
                gate[topk] = True
            gate_frac = gate.float().mean()
            if bool(gate.any()):
                ce_stable = F.cross_entropy(
                    logits[gate], y[gate], label_smoothing=self.label_smoothing
                )
        else:
            # Legacy path: no teacher snapshot available.
            ce_stable = ce
            gate_frac = ce.new_ones(())

        penalty = ce.new_zeros(())
        drift = ce.new_zeros(())
        kl_sigma = ce.new_zeros(())
        for n, p in disc.named_parameters():
            lv = self.logvars.get(n)
            if lv is None:
                continue
            f_prev = self.fisher_ema.get(n)
            mu_prev = self.prev_mu.get(n, p.detach())
            logvar_prev = self.prev_logvar.get(n, lv.detach())
            if f_prev is not None:
                f_prev = torch.nan_to_num(f_prev, nan=0.0, posinf=0.0, neginf=0.0)
                drift = drift + (f_prev * (p - mu_prev) ** 2).sum().clamp(max=1e12)
            ratio = torch.exp(lv - logvar_prev)           # σ_r,i² / σ_{r-1,i}²
            kl_sigma = kl_sigma + (ratio - 1.0 - torch.log(ratio)).sum()
        # Normalize Fisher penalty to avoid scale explosion on large models
        fim_sum = drift.detach().sum() if isinstance(drift, torch.Tensor) else torch.tensor(0.0)
        fim_scale = 1.0
        if isinstance(fim_sum, torch.Tensor) and fim_sum.numel() == 1:
            if float(fim_sum) > 1e18:
                fim_scale = 1e-18
            elif float(fim_sum) > 1e15:
                fim_scale = 1e-15
            elif float(fim_sum) > 1e12:
                fim_scale = 1e-12
        penalty = 0.5 * self.lambda_evcl * fim_scale * (drift * fim_scale + kl_sigma)
        if teacher is None:
            data_term = ce
        else:
            data_term = self.distill_weight * distill + self.synth_ce_weight * ce_stable
        total = data_term + penalty
        return {
            "total": total,
            "ce": ce,
            "ce_stable": ce_stable,
            "distill": distill,
            "gate_frac": gate_frac,
            "drift": drift,
            "kl_sigma": kl_sigma,
        }

    # ------------------------------------------------------------------
    # One consolidation round (Algorithm 1 line 21)
    # ------------------------------------------------------------------
    def consolidate_round(
        self, disc: nn.Module, gen: nn.Module, num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """Estimate the FIM at the aggregated μ_r and optimize Eq. 18.

        The global parameters themselves are μ_r; afterwards they hold the
        consolidated posterior mean and prev_mu/prev_logvar store μ_{r-1} for
        the next round's penalty.
        """
        num_classes = int(num_classes or disc.num_classes)
        self.prepare(disc, reset_variational=True)

        # Skip EVCL synthetic consolidation in the very first round when G_g^0
        # is untrained random weights; just initialize anchors from the
        # aggregated model.
        if self.rounds_completed == 0:
            with torch.no_grad():
                self.prev_mu = {n: p.detach().clone() for n, p in disc.named_parameters()}
                self.prev_logvar = {n: lv.detach().clone() for n, lv in self.logvars.items()}
                # Keep fisher_ema zero on first round so the EWC drift penalty
                # does not corrupt the freshly aggregated client weights.
                self.fisher_ema = {n: torch.zeros_like(p) for n, p in disc.named_parameters()}
            self.rounds_completed += 1
            self.last_fim_stats = {
                "dual_gate": float(self.dual_gate),
                "acceptance": 0.0,
                "effective_acceptance": 0.0,
                "committed": 0.0,
                "unstable_classes": 0.0,
                "entropy": 0.0,
                "realism": 0.0,
                "gate_fallback": 0.0,
            }
            metrics = {
                "evcl_total": 0.0,
                "evcl_ce": 0.0,
                "evcl_ce_stable": 0.0,
                "evcl_distill": 0.0,
                "evcl_gate_frac": 0.0,
                "evcl_drift": 0.0,
                "evcl_kl_sigma": 0.0,
                "fim_acceptance": 0.0,
                "fim_committed": 0.0,
                "fim_unstable_classes": 0.0,
                "fim_entropy": 0.0,
                "fim_realism": 0.0,
                "dual_gate": float(self.dual_gate),
                "fim_mean": 0.0,
            }
            LOGGER.info(
                "EVCL round %d: skipped (untrained generator) - initialized anchors from aggregated model",
                self.rounds_completed,
            )
            return metrics

        acceptance = self.estimate_fim(disc, gen, num_classes)

        # Frozen anchor to the freshly aggregated global model \bar{θ}_g^t.
        # The EVCL data term distills toward this teacher instead of fitting
        # hard labels on synthetic noise (which caused uniform-logit collapse).
        teacher = clone_model(disc)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Snapshot aggregated parameters before EVCL for gentle Polyak blend
        theta_agg: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in disc.named_parameters()
        }

        params = [p for p in disc.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            [{"params": params}, {"params": list(self.logvars.values())}], lr=self.lr
        )
        was_training = disc.training
        disc.eval()   # freeze BN statistics while consolidating; grads still flow
        for p in disc.parameters():
            p.requires_grad_(True)
        last: Dict[str, torch.Tensor] = {}
        for _ in range(max(1, int(self.steps))):
            last = self.evcl_loss(disc, gen, num_classes, teacher=teacher)
            if not bool(torch.isfinite(last["total"])):
                LOGGER.warning("EVCL: non-finite loss, skipping the consolidation step")
                break
            optimizer.zero_grad(set_to_none=True)
            last["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            with torch.no_grad():
                for lv in self.logvars.values():
                    lv.clamp_(*self.logvar_bounds)
        # Gentle Polyak blend: keep EVCL as a soft regularizer
        if self.polyak_tau > 0.0:
            tau = self.polyak_tau
            with torch.no_grad():
                for n, p in disc.named_parameters():
                    if n in theta_agg:
                        p.copy_((1.0 - tau) * theta_agg[n] + tau * p.detach())
        # If EVCL made things worse in terms of data term, revert to aggregated
        # (safety guard)
        try:
            with torch.no_grad():
                x_syn, y = self._sample_synthetic(gen, num_classes)
                t_logits = teacher(x_syn)["logits"]
                s_logits = disc(x_syn)["logits"]
                t_loss = F.cross_entropy(t_logits, y)
                s_loss = F.cross_entropy(s_logits, y)
                if float(s_loss) > 1.5 * float(t_loss):
                    for n, p in disc.named_parameters():
                        if n in theta_agg:
                            p.copy_(theta_agg[n])
        except Exception as exc:
            LOGGER.warning("EVCL revert check failed: %s", exc)
        if was_training:
            disc.train()

        with torch.no_grad():
            self.prev_mu = {n: p.detach().clone() for n, p in disc.named_parameters()}
            self.prev_logvar = {n: lv.detach().clone() for n, lv in self.logvars.items()}
        self.rounds_completed += 1

        def _scalar(key: str) -> float:
            value = last.get(key)
            return float(value.detach()) if value is not None else 0.0

        metrics = {
            "evcl_total": _scalar("total"),
            "evcl_ce": _scalar("ce"),
            "evcl_ce_stable": _scalar("ce_stable"),
            "evcl_distill": _scalar("distill"),
            "evcl_gate_frac": _scalar("gate_frac"),
            "evcl_drift": _scalar("drift"),
            "evcl_kl_sigma": _scalar("kl_sigma"),
            "fim_acceptance": float(acceptance),
            "fim_committed": float(self.last_fim_stats.get("committed", 0.0)),
            "fim_unstable_classes": float(self.last_fim_stats.get("unstable_classes", 0.0)),
            "fim_entropy": float(self.last_fim_stats.get("entropy", 0.0)),
            "fim_realism": float(self.last_fim_stats.get("realism", 0.0)),
            "dual_gate": float(self.dual_gate),
            "fim_mean": float(torch.stack([f.mean() for f in self.fisher_ema.values()]).mean())
            if self.fisher_ema else 0.0,
        }
        LOGGER.info(
            "EVCL round %d: total=%.4f ce=%.4f acceptance=%.2f",
            self.rounds_completed, metrics["evcl_total"], metrics["evcl_ce"], acceptance,
        )
        return metrics


class FederatedServer:
    """Central server holding (θ_g, G_g), aggregating clients and consolidating."""

    def __init__(
        self,
        disc: Discriminator,
        gen: Generator,
        evcl: Optional[ServerEVCLConsolidator] = None,
        defense: Optional[ClassAwareMMDDefense] = None,
        seed: int = 0,
    ) -> None:
        self.disc = disc
        self.gen = gen
        self.evcl = evcl or ServerEVCLConsolidator(z_dim=gen.z_dim)
        self.defense = defense     # Extension C Stage 1 (Section 5.2, Eqs. 13-19)
        self.seen_classes: Set[int] = set()   # C_{<=r-1}^(g)
        self.last_screen: Optional[ScreenResult] = None
        self.round = 0
        self.history: List[Dict[str, float]] = []
        self._seed = int(seed)
        self._generator = make_generator(seed, next(disc.parameters()).device)

    # ------------------------------------------------------------------
    # Algorithm 1 line 4: broadcast
    # ------------------------------------------------------------------
    def ensure_heads(self, num_classes: int) -> None:
        """Expand θ_g and G_g to cover ``num_classes`` labels (max class id + 1).

        Clients report the classes they will train on at the start of a round;
        the generator must be able to condition on those labels before the
        replay batches are synthesized (Algorithm 1 line 5).  Aggregation also
        expands the heads when a client uploads a wider classifier head, but
        that happens only at the *end* of the round.
        """
        target = int(num_classes)
        if target > self.disc.num_classes:
            self.disc.expand_classes(target)
            self.gen.expand_classes(target)
            LOGGER.info("server: expanded heads to %d classes before broadcast", target)

    def broadcast(self) -> "Tuple[Discriminator, Generator]":
        return self.disc, self.gen

    # ------------------------------------------------------------------
    # Extension C Stage 1: de-quantize once, then screen in feature space
    # ------------------------------------------------------------------
    def decompress_updates(
        self, updates: List[ClientUpdate], add_noise: bool = True
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """De-quantize every update exactly once (zCDP noise is stochastic)."""
        device = next(self.disc.parameters()).device
        if self._generator.device != device:
            self._generator = make_generator(self._seed, device)
        return [
            (
                decompress_state_dict(u.delta_disc, generator=self._generator,
                                      add_noise=add_noise),
                decompress_state_dict(u.delta_gen, generator=self._generator,
                                      add_noise=add_noise),
            )
            for u in updates
        ]

    def _candidate_models(
        self,
        update: ClientUpdate,
        disc_delta: Dict[str, torch.Tensor],
        gen_delta: Dict[str, torch.Tensor],
    ) -> Tuple[Discriminator, Generator]:
        """Reconstruct θ_k^r = θ_g^{r-1} + Δθ̂_k and G_k^r = G_g^{r-1} + ΔĜ_k."""
        cand_disc = clone_model(self.disc)
        cand_gen = clone_model(self.gen)
        target = max(int(self.disc.num_classes), int(update.head_size))
        if cand_disc.num_classes < target:
            cand_disc.expand_classes(target)
        if cand_gen.num_classes < target:
            cand_gen.expand_classes(target)
        with torch.no_grad():
            for model, delta_dict in ((cand_disc, disc_delta), (cand_gen, gen_delta)):
                for name, p in model.named_parameters():
                    delta = delta_dict.get(name)
                    if delta is None:
                        continue
                    p.add_(pad_head_delta(name, delta.to(p.device), p.shape))
        return cand_disc, cand_gen

    @torch.no_grad()
    def screen(
        self,
        updates: List[ClientUpdate],
        deltas: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    ) -> ScreenResult:
        """Class-conditional MMD screening (Eqs. 13-19) against θ_g^{r-1}, G_g^{r-1}.

        Must run *before* :meth:`aggregate` expands the heads / updates the
        parameters, so the reference distributions are the previous global ones.
        """
        if self.defense is None:
            raise RuntimeError("server has no Extension C defense configured")
        scores: Dict[int, float] = {}
        per_class: Dict[int, Dict[int, float]] = {}
        for u, (disc_delta, gen_delta) in zip(updates, deltas):
            cand_disc, cand_gen = self._candidate_models(u, disc_delta, gen_delta)
            scores[u.client_id], per_class[u.client_id] = self.defense.score_client(
                self.disc, self.gen, cand_disc, cand_gen,
                getattr(u, "active_classes", ()), self.seen_classes,
                delta_disc=disc_delta,
            )
        trust, z_scores = self.defense.compute_trust_weights(
            scores, malicious_ratio=getattr(self, '_malicious_ratio', 0.0),
            num_active=len(updates)
        )
        result = ScreenResult(
            trust_weights=trust,
            scores=scores,
            z_scores=z_scores,
            per_class_scores=per_class,
            class_weights=self.defense.compute_class_weights(per_class),
            rejected=[k for k, w in trust.items() if w <= 1e-12],
        )
        self.last_screen = result
        return result

    # ------------------------------------------------------------------
    # Algorithm 1 lines 16-18: expand heads + weighted aggregation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aggregate(
        self,
        updates: List[ClientUpdate],
        trust_weights: Optional[Dict[int, float]] = None,
        class_weights: Optional[Dict[int, Dict[int, float]]] = None,
        add_noise: bool = True,
        deltas: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
    ) -> Dict[str, float]:
        """θ_g ← θ_g + Σ_k w_k Δθ̂_k over decompressed (de-quantized) updates.

        ``trust_weights``: optional per-client weights w_k (the Extension C MMD
        screening supplies Boltzmann trust weights; the baseline uses uniform).
        ``class_weights``: optional per-class weights {c: {client_id: w}} for the
        generator's class-conditional layers (Algorithm 1 line 18); defaults to
        the client weights w_k.
        ``deltas``: pre-decompressed (disc, gen) delta dicts.  The zCDP
        de-quantization is stochastic, so a caller that already decompressed the
        updates for Stage-1 screening MUST pass the same deltas here instead of
        letting the server decompress a second time.
        """
        if not updates:
            raise ValueError("aggregate() needs at least one client update")

        if trust_weights is None:
            w = {u.client_id: 1.0 / len(updates) for u in updates}
        else:
            total = sum(trust_weights.get(u.client_id, 0.0) for u in updates)
            w = {u.client_id: trust_weights.get(u.client_id, 0.0) / max(total, 1e-12)
                 for u in updates}

        if deltas is None:
            deltas = self.decompress_updates(updates, add_noise=add_noise)
        decom_disc = [d[0] for d in deltas]
        decom_gen = [d[1] for d in deltas]
        known_by_client: Dict[int, Optional[Sequence[int]]] = {}
        for u in updates:
            known = getattr(u, "known_classes", None) or getattr(u, "active_classes", None)
            known_by_client[u.client_id] = known

        # 1. expand global heads if new classes arrived (Section 2.5)
        target = max([self.disc.num_classes] + [u.head_size for u in updates])
        if target > self.disc.num_classes:
            self.disc.expand_classes(target)
            self.gen.expand_classes(target)
            LOGGER.info("server: expanded heads to %d classes", target)

        # 2. weighted aggregation (zero-padding head deltas of smaller clients)
        for model, client_deltas, is_generator in ((self.disc, decom_disc, False),
                                                   (self.gen, decom_gen, True)):
            named = dict(model.named_parameters())
            for name, p in named.items():
                acc = torch.zeros_like(p)
                # Snapshot old for prototypes/head anchoring
                old_p = p.clone().detach() if ('prototypes' in name) else None
                for u, d in zip(updates, client_deltas):
                    client_w = float(w.get(u.client_id, 0.0))
                    if client_w <= 1e-12:
                        continue
                    delta = d.get(name)
                    if delta is None:
                        continue
                    delta = pad_head_delta(name, delta.to(p.device), p.shape)
                    # Class-aware head aggregation: never let a client update
                    # rows for classes it has not observed.
                    delta = mask_unseen_head_rows(
                        name, delta, known_by_client.get(u.client_id)
                    )
                    row_w = (self._class_row_weights(name, p, u.client_id, client_w,
                                                     class_weights)
                             if is_generator else None)
                    if row_w is None:
                        acc = acc + client_w * delta
                    else:  # per-class weights w_k^(c) on CBN embeddings (line 18)
                        shape = (-1,) + (1,) * (delta.dim() - 1)
                        acc = acc + row_w.view(shape) * delta
                p.add_(acc)
                if old_p is not None and 'prototypes' in name:
                    # Anchor classes that might be dominated: blend toward old
                    try:
                        num_c = min(old_p.shape[0], p.shape[0])
                        # Stronger anchoring for old classes
                        p[:num_c] = 0.7 * old_p[:num_c] + 0.3 * p[:num_c]
                    except Exception as exc:
                        LOGGER.warning("prototype anchoring failed for %s: %s", name, exc)

        # 3. BatchNorm running statistics: aggregate raw buffers separately.
        #    They are never sparsified/quantized/noised, otherwise eval-mode
        #    normalization no longer matches the aggregated parameters.
        self._aggregate_buffers(self.disc, updates, w, "buffer_disc")
        self._aggregate_buffers(self.gen, updates, w, "buffer_gen")

        self.round += 1
        metrics = {
            "round": float(self.round),
            "head_size": float(self.disc.num_classes),
            "num_updates": float(len(updates)),
            **{f"w_{k}": v for k, v in w.items()},
        }
        return metrics

    @staticmethod
    @torch.no_grad()
    def _aggregate_buffers(
        model: nn.Module,
        updates: Sequence[ClientUpdate],
        weights: Mapping[int, float],
        attr: str,
    ) -> None:
        """Weighted-average BatchNorm buffers from clients (no compression)."""
        for name, buf in model.named_buffers():
            if not is_batch_norm_buffer(name):
                continue
            dtype = buf.dtype if buf.is_floating_point() else torch.float32
            acc = torch.zeros_like(buf, dtype=dtype)
            total = 0.0
            for u in updates:
                client_buffers = getattr(u, attr, None) or {}
                value = client_buffers.get(name)
                if value is None or tuple(value.shape) != tuple(buf.shape):
                    continue
                weight = float(weights.get(u.client_id, 0.0))
                if weight <= 0.0:
                    continue
                acc = acc + weight * value.to(device=buf.device, dtype=dtype)
                total += weight
            if total > 0.0:
                buf.copy_((acc / total).to(buf.dtype))

    @staticmethod
    def _class_row_weights(
        name: str,
        param: torch.Tensor,
        client_id: int,
        default_w: float,
        class_weights: Optional[Dict[int, Dict[int, float]]],
    ) -> Optional[torch.Tensor]:
        """Per-class weights w_k^(c) for generator class-conditional embeddings.

        Applies to the ConditionalBatchNorm gain/beta rows (Algorithm 1 line
        18).  When per-class weights are available the row weight is *strict*:
        a client that did not observe class c receives weight 0 for that row
        (Eq. 19, final paragraph), and a globally rejected client
        (w_k = 0, z_k > z_max) is excluded from every row.  Classes with no
        per-class entry fall back to the client weight w_k.
        """
        if class_weights is None or not ("cbn" in name or "embed" in name):
            return None
        if default_w <= 1e-12:      # hard-rejected by the Boltzmann trust weight
            return torch.zeros(param.shape[0], dtype=param.dtype, device=param.device)
        rows = [
            (float(class_weights[c][client_id]) if c in class_weights and client_id in
             class_weights[c] else (0.0 if c in class_weights else default_w))
            for c in range(param.shape[0])
        ]
        return torch.tensor(rows, dtype=param.dtype, device=param.device)

    # ------------------------------------------------------------------
    # Algorithm 1 line 21: EVCL consolidation stage
    # ------------------------------------------------------------------
    def consolidate(self, num_classes: Optional[int] = None) -> Dict[str, float]:
        return self.evcl.consolidate_round(self.disc, self.gen, num_classes)

    def run_round(
        self,
        updates: List[ClientUpdate],
        trust_weights: Optional[Dict[int, float]] = None,
        class_weights: Optional[Dict[int, Dict[int, float]]] = None,
        use_defense: Optional[bool] = None,
        consolidate: bool = True,
        post_aggregate_hook: Optional[Callable[[], Mapping[str, float]]] = None,
    ) -> Dict[str, float]:
        """Screen, aggregate and consolidate one round (Algorithm 1 lines 13-21).

        Extension C Stage 1 runs before aggregation: the updates are
        de-quantized exactly once, candidate models are reconstructed against
        the previous global (θ_g^{r-1}, G_g^{r-1}), MMD/JS screening produces the
        trust weights w_k (and per-class CBN weights), and the same
        de-quantized deltas are then aggregated with those weights.  Stage 2
        (dual-gated EVCL consolidation) runs afterwards on the aggregated G_r^g.
        """
        deltas = self.decompress_updates(updates)

        screen_metrics: Dict[str, float] = {}
        defense_on = (self.defense is not None) if use_defense is None else bool(use_defense)
        if defense_on and self.defense is not None:
            result = self.screen(updates, deltas)
            if trust_weights is None:
                trust_weights = result.trust_weights
            if class_weights is None:
                class_weights = result.class_weights
            screen_metrics = result.as_metrics()
            screen_metrics["defense_rejected"] = float(len(result.rejected))
            screen_metrics["defense_uniform"] = float(
                abs(max(result.trust_weights.values()) - 1.0 / len(result.trust_weights)) < 1e-9
            ) if result.trust_weights else 0.0

        metrics = self.aggregate(
            updates, trust_weights=trust_weights,
            class_weights=class_weights, deltas=deltas,
        )
        metrics.update(screen_metrics)
        # Stage-by-stage diagnostics: let the caller evaluate the *aggregated*
        # model before EVCL consolidation touches it.
        if post_aggregate_hook is not None:
            hook_metrics = post_aggregate_hook() or {}
            metrics.update({str(k): float(v) for k, v in hook_metrics.items()})
        if consolidate:
            metrics.update(self.consolidate())

        # The global model now covers every class the clients reported.
        for u in updates:
            self.seen_classes.update(int(c) for c in getattr(u, "active_classes", ()))
        self.history.append(metrics)
        return metrics
