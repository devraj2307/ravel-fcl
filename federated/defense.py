"""Extension C, Stage 1: class-conditional multi-kernel RBF-MMD screening.

Spec Section 5.2 / Algorithm 1 lines 13-16.

Before aggregation the server reconstructs every participating client's
candidate models::

    theta_k^r = theta_g^{r-1} + Delta_theta_hat_k
    G_k^r     = G_g^{r-1}     + Delta_G_hat_k

and, for each class c in the client's active set C_active^(k), draws a small
validation batch {z_b}_{b=1..M} ~ N(0, I_128) to synthesize
X_k^(c) = {G_k^r(z_b, c)}.  Two cases are screened:

Case 1 -- recurring classes c in C_{<=r-1} (Eqs. 13-16)
    * multi-bandwidth Gaussian RBF MMD^2 between the candidate and the
      reference G_g^{r-1}(z_b, c), both embedded by the *previous* global
      feature projector f_{theta_g^{r-1}} on the unit hypersphere;
    * symmetric Jensen-Shannon divergence between the candidate and the
      previous global classifier predictions on the reference batch::

          s_k^(c) = d_MMD^2(Phi_k^(c), Phi_g^(c)) + eta_js * d_JS^(k,c)

Case 2 -- newly introduced classes c not in C_{<=r-1} (Eq. 17)
    No reference distribution exists, so the class must not collapse onto an
    already-learned class centroid::

          s_k^(c) = max(0, max_y mean_u u^T v_y^(g,r-1) - m_n)^2

Client anomaly scores S_k = mean_c s_k^(c) are robustly normalized with the
median / MAD statistic and mapped to clipped Boltzmann trust weights::

    z_k  = (S_k - median(S)) / (1.4826 * (MAD(S) + eps))
    w~_k = exp(-gamma_trust * max(0, z_k))   if z_k <= z_max else 0
    w_k  = w~_k / sum_j w~_j

Per-class CBN aggregation weights use the per-class trust scores directly::

    w_k^(c) propto exp(-gamma_trust * s_k^(c))

over clients that observed class c (Eq. 19, final paragraph).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from losses.ravel_losses import symmetric_divergence
from utils.rng import make_generator

LOGGER = logging.getLogger(__name__)

# Section 5.2 defaults
SIGMAS = (0.2, 0.5, 1.0, 2.0)   # RBF bandwidths (Eq. 14)
ETA_JS = 0.5                    # eta_js (Eq. 16)
T_ALIGN = 2.0                   # temperature for the JS comparison (Section 2.3)
M_N_MARGIN = 0.25               # inter-class margin m_n (Eq. 17)
Z_MAX = 2.5                     # hard rejection threshold (Eq. 19)
GAMMA_TRUST = 2.0               # Boltzmann temperature (Eq. 19)
MAD_SCALE = 1.4826              # consistency constant for the MAD
EPS = 1e-6


def compute_multikernel_mmd(
    x: torch.Tensor, y: torch.Tensor, sigmas: Sequence[float] = SIGMAS
) -> torch.Tensor:
    """Unbiased multi-bandwidth RBF-kernel MMD^2 on unit-hypersphere features.

    Implements Eq. 14 with K(u, v) = sum_l exp(-||u - v||^2 / (2 sigma_l^2)):

        d^2 = 1/(M(M-1)) sum_{i != j} K(u_i, u_j)
            + 1/(M(M-1)) sum_{i != j} K(v_i, v_j)
            - 2/M^2 sum_{i,j} K(u_i, v_j)
    """
    m = int(x.size(0))
    if m < 2 or y.size(0) < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    dist_xx = torch.cdist(x, x, p=2).pow(2)
    dist_yy = torch.cdist(y, y, p=2).pow(2)
    dist_xy = torch.cdist(x, y, p=2).pow(2)
    k_xx = sum(torch.exp(-dist_xx / (2.0 * s ** 2)) for s in sigmas)
    k_yy = sum(torch.exp(-dist_yy / (2.0 * s ** 2)) for s in sigmas)
    k_xy = sum(torch.exp(-dist_xy / (2.0 * s ** 2)) for s in sigmas)
    mask = ~torch.eye(m, dtype=torch.bool, device=x.device)
    mmd_sq = k_xx[mask].mean() + k_yy[mask].mean() - 2.0 * k_xy.mean()
    return torch.clamp(mmd_sq, min=0.0)


@dataclass
class ScreenResult:
    """Output of one Stage-1 screening round (Algorithm 1 lines 13-16)."""

    trust_weights: Dict[int, float]                 # w_k (Eq. 19)
    scores: Dict[int, float]                        # S_k
    z_scores: Dict[int, float]                      # z_k (Eq. 18)
    per_class_scores: Dict[int, Dict[int, float]]   # s_k^(c)
    class_weights: Dict[int, Dict[int, float]]      # w_k^(c) for CBN rows
    rejected: List[int] = field(default_factory=list)

    def as_metrics(self, prefix: str = "") -> Dict[str, float]:
        """Flatten for logging / history."""
        out: Dict[str, float] = {}
        for k, w in self.trust_weights.items():
            out[f"{prefix}trust_w_{k}"] = float(w)
            out[f"{prefix}score_{k}"] = float(self.scores.get(k, 0.0))
            out[f"{prefix}z_{k}"] = float(self.z_scores.get(k, 0.0))
        return out


class ClassAwareMMDDefense:
    """Stage 1: class-conditional multi-kernel MMD & classifier screening."""

    def __init__(
        self,
        z_dim: int = 128,
        val_batch_size: int = 64,
        temperature: float = T_ALIGN,
        eta_js: float = ETA_JS,
        margin_n: float = M_N_MARGIN,
        z_max: float = Z_MAX,
        gamma_trust: float = GAMMA_TRUST,
        sigmas: Sequence[float] = SIGMAS,
        seed: int = 0,
    ) -> None:
        self.z_dim = int(z_dim)
        self.M = int(val_batch_size)
        self.temperature = float(temperature)
        self.eta_js = float(eta_js)
        self.margin_n = float(margin_n)
        self.z_max = float(z_max)
        self.gamma_trust = float(gamma_trust)
        self.sigmas = tuple(float(s) for s in sigmas)
        self._seed = int(seed)
        self._generator = make_generator(seed, "cpu")
        self._ema_score: Dict[int, float] = {}
        self._ema_alpha = 0.7

    # ------------------------------------------------------------------
    # Per-client class-conditional anomaly score (Eqs. 13-17)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_client(
        self,
        global_disc: torch.nn.Module,
        global_gen: torch.nn.Module,
        candidate_disc: torch.nn.Module,
        candidate_gen: torch.nn.Module,
        active_classes: Iterable[int],
        seen_global_classes: Iterable[int],
        delta_disc: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, Dict[int, float]]:
        """Return (S_k, {c: s_k^(c)}) for one candidate client."""
        device = next(global_disc.parameters()).device
        if self._generator.device != device:
            self._generator = make_generator(self._seed, device)
        global_disc.eval()
        global_gen.eval()
        candidate_disc.eval()
        candidate_gen.eval()

        seen = {int(c) for c in seen_global_classes}
        c_prev = int(global_disc.num_classes)
        v_glob = global_disc.normalized_prototypes().detach()
        old_classes = sorted(c for c in seen if 0 <= c < v_glob.shape[0])
        v_old = v_glob[torch.tensor(old_classes, dtype=torch.long, device=device)] \
            if old_classes else None

        per_class: Dict[int, float] = {}
        for c in sorted({int(c) for c in active_classes}):
            if c < 0 or c >= candidate_gen.num_classes:
                continue  # the candidate cannot condition on this class
            z = torch.randn(self.M, self.z_dim, device=device,
                            generator=self._generator)
            y_cond = torch.full((self.M,), c, dtype=torch.long, device=device)

            # Candidate class-conditional sample bank X_k^(c) and its features
            phi_k = global_disc(candidate_gen(z, y_cond))["proj"]      # [M, 128]

            if c in seen and c < c_prev:
                # Case 1: recurring class -> MMD^2 + eta_js * JS (Eqs. 13-16)
                x_g = global_gen(z, y_cond)
                out_g = global_disc(x_g)
                phi_g = out_g["proj"]
                mmd = compute_multikernel_mmd(phi_k, phi_g, self.sigmas)

                n_cls = min(int(out_g["logits"].shape[-1]), int(candidate_disc.num_classes))
                p_g = F.softmax(out_g["logits"][:, :n_cls] / self.temperature, dim=-1)
                p_k = F.softmax(
                    candidate_disc(x_g)["logits"][:, :n_cls] / self.temperature, dim=-1
                )
                js = symmetric_divergence(p_k, p_g, kind="js").mean()
                score = float(mmd) + self.eta_js * float(js)
            else:
                # Case 2: new class -> inter-class margin violation (Eq. 17)
                if v_old is not None and v_old.numel() > 0:
                    cos_sims = (phi_k @ v_old.t()).mean(dim=0)          # [n_old]
                    violation = torch.clamp(cos_sims.max() - self.margin_n, min=0.0)
                    score = float(violation ** 2)
                else:
                    score = 0.0
            per_class[c] = score

        mean_score = float(sum(per_class.values()) / len(per_class)) if per_class else 0.0
        # Update-norm / classifier-row anomaly (catches attack_boost in round 0)
        if delta_disc is not None:
            try:
                total_norm = 0.0
                for v in delta_disc.values():
                    if isinstance(v, torch.Tensor):
                        total_norm += float(v.norm().cpu())
                mean_score += 0.01 * total_norm
                # classifier head row norms
                for name in ['classifier.weight', 'classifier.linear.weight']:
                    if name in delta_disc and isinstance(delta_disc[name], torch.Tensor):
                        wdelta = delta_disc[name]
                        mean_score += 0.05 * float(wdelta.norm(dim=1).max().cpu())
                        break
            except Exception as exc:
                LOGGER.debug("class score computation failed: %s", exc)
        return mean_score, per_class

    # ------------------------------------------------------------------
    # Robust MAD normalization + Boltzmann trust weights (Eqs. 18-19)
    # ------------------------------------------------------------------
    def compute_trust_weights(
        self, scores: Mapping[int, float], malicious_ratio: float = 0.0, num_active: int = 0
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        if not scores:
            return {}, {}
        # Update EMA
        for k, v in scores.items():
            if k in self._ema_score:
                self._ema_score[k] = self._ema_alpha * self._ema_score[k] + (1.0 - self._ema_alpha) * float(v)
            else:
                self._ema_score[k] = float(v)
        ids = sorted(scores)
        use_scores = {k: self._ema_score.get(k, scores[k]) for k in ids}
        vals = [float(use_scores[k]) for k in ids]
        med = float(median(vals))
        mad = float(median([abs(v - med) for v in vals])) + EPS
        z = {k: (float(use_scores[k]) - med) / (MAD_SCALE * mad) for k in ids}

        s_rel = {k: max(0.0, float(use_scores[k]) - med) for k in ids}
        raw = {k: math.exp(-self.gamma_trust * s_rel[k]) for k in ids}
        threshold = med + self.z_max * MAD_SCALE * mad
        # Cap hard rejections
        candidates = [k for k in ids if float(use_scores[k]) > threshold]
        max_reject = max(1, int(round(float(malicious_ratio) * float(num_active or len(ids)))))
        if len(candidates) > max_reject:
            # keep only top max_reject by score
            candidates_sorted = sorted(candidates, key=lambda kk: float(use_scores[kk]), reverse=True)
            for k in candidates_sorted[max_reject:]:
                candidates.remove(k)
        rejected_all = len(candidates) >= len(ids)
        if rejected_all:
            for k in candidates:
                raw[k] = 0.0
            total = sum(raw.values())
            if total < 1e-8:
                weights = {k: 1.0 / len(ids) for k in ids}
            else:
                weights = {k: v / total for k, v in raw.items()}
            return weights, z
        for k in candidates:
            raw[k] = 0.0
        total = sum(raw.values())
        if total < 1e-8:
            LOGGER.warning("MMD screening rejected every client; falling back to uniform")
            weights = {k: 1.0 / len(ids) for k in ids}
        else:
            weights = {k: v / total for k, v in raw.items()}
        return weights, z

    def compute_class_weights(
        self, per_class_scores: Mapping[int, Mapping[int, float]]
    ) -> Dict[int, Dict[int, float]]:
        """w_k^(c) propto exp(-gamma_trust * s_k^(c)) over observers of class c."""
        classes = sorted({c for scores in per_class_scores.values() for c in scores})
        out: Dict[int, Dict[int, float]] = {}
        for c in classes:
            raw = {
                k: math.exp(-self.gamma_trust * float(scores[c]))
                for k, scores in per_class_scores.items()
                if c in scores
            }
            total = sum(raw.values())
            if total < 1e-12:
                out[c] = {k: 1.0 / len(raw) for k in raw}
            else:
                out[c] = {k: v / total for k, v in raw.items()}
        return out

    # ------------------------------------------------------------------
    # Convenience wrapper (reference API, spec Section 6.2)
    # ------------------------------------------------------------------
    def screen_clients(
        self,
        global_disc: torch.nn.Module,
        global_gen: torch.nn.Module,
        candidates: Mapping[int, Tuple[torch.nn.Module, torch.nn.Module]],
        client_active_classes: Mapping[int, Sequence[int]],
        seen_global_classes: Iterable[int],
    ) -> ScreenResult:
        """Score every candidate client and return trust / class weights."""
        seen = {int(c) for c in seen_global_classes}
        scores: Dict[int, float] = {}
        per_class: Dict[int, Dict[int, float]] = {}
        for k, (cand_disc, cand_gen) in candidates.items():
            scores[k], per_class[k] = self.score_client(
                global_disc, global_gen, cand_disc, cand_gen,
                client_active_classes.get(k, ()), seen,
            )
        trust, z = self.compute_trust_weights(scores)
        class_weights = self.compute_class_weights(per_class)
        rejected = [k for k, w in trust.items() if w <= 1e-12]
        return ScreenResult(
            trust_weights=trust, scores=scores, z_scores=z,
            per_class_scores=per_class, class_weights=class_weights,
            rejected=rejected,
        )


# Alias matching the spec's reference class name (Section 6.2)
ExtensionCDefense = ClassAwareMMDDefense
