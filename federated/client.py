"""Federated client local training (spec Section 2.2/2.3, Algorithm 1 lines 3-12).

Per communication round r, client k:

* receives the broadcast globals (θ_g^{r-1}, G_g^{r-1}) and snapshots its own
  generator as G_k^{r-1} (the "previous local generator" replay stream);
* synthesizes two replay batches with dataperturbation noise
  η_dp ~ N(0, 0.01^2 I)  (Section 2.3 / 5.2.1):

      X_glob ~ G_g^{r-1}(z, y_syn) + η_dp        (global generator stream)
      X_prev ~ G_k^{r-1}(z, y_syn) + η_dp        (previous local generator stream)

  and unions them with the real local data D_k^t into D_train;
* for E local epochs minimizes (Eq. 5, Algorithm 1 line 10):

      L_D = L_adv^D + λ_cls · L_D2D-CE + λ_align · L_align       (discriminator θ_k)
      L_G = L_adv^G + λ_cls · L_D2D-CE                           (generator G_k)

  where the alignment loss couples the classifier logits of the three streams
  (prev, glob, real) that share the same label conditioning y (Section 2.3);
* returns the compressed updates (Δθ_k, ΔG_k) plus the active class list
  (Algorithm 1 lines 11-12, via federated/compression.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from federated.compression import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DEFAULT_KEEP_RATIO,
    DEFAULT_L2_CLIP,
    CompressedUpdate,
    compress_state_dict,
)
from attacks.poisoning import PoisoningAttack
from utils.rng import make_generator
from losses.ravel_losses import (
    LAMBDA_ALIGN,
    LAMBDA_CLS,
    AdaptiveD2DCE,
    discriminator_objective,
    generator_objective,
    ravel_alignment_loss,
)
from models.reacgan import (
    Discriminator,
    Generator,
    clone_model,
    is_batch_norm_buffer,
)

LOGGER = logging.getLogger(__name__)

ETA_DP = 0.01  # η_dp ~ N(0, 0.01^2 I) replay perturbation (Section 2.3)


@dataclass
class ClientUpdate:
    """Algorithm 1 line 12: compressed updates + active class list."""

    client_id: int
    round: int
    task: int
    delta_disc: Dict[str, CompressedUpdate]
    delta_gen: Dict[str, CompressedUpdate]
    active_classes: List[int]
    head_size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    # Cumulative classes this client has ever observed.  The server uses this
    # set to mask classifier/prototype head rows it never trained, so a client
    # cannot corrupt the logits of classes it has not seen yet.
    known_classes: List[int] = field(default_factory=list)
    # BatchNorm running statistics, transmitted raw (never compressed or
    # noised -- otherwise the eval-mode forward pass is destroyed).
    buffer_disc: Dict[str, torch.Tensor] = field(default_factory=dict)
    buffer_gen: Dict[str, torch.Tensor] = field(default_factory=dict)


def _masked_logits(
    logits: torch.Tensor, allowed_classes: Iterable[int]
) -> torch.Tensor:
    """Mask logits of classes the client has not observed (``-inf``).

    The global classifier head spans ``C_{<=t}`` but a client only ever trains
    on its own cumulative set.  Leaving unseen logits active lets the alignment
    KL push their weights negative (each client has a different unseen set),
    which is then averaged into the global head and collapses accuracy right
    after aggregation.
    """
    num_classes = int(logits.size(-1))
    allowed = sorted({int(c) for c in allowed_classes if 0 <= int(c) < num_classes})
    if not allowed or len(allowed) >= num_classes:
        return logits
    keep = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)
    keep[allowed] = True
    return logits.masked_fill(~keep.unsqueeze(0), float("-inf"))


def _bn_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot BatchNorm running statistics (buffers, not parameters)."""
    return {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if is_batch_norm_buffer(name)
    }


def _synth_batch(
    generator: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    eta_dp: float,
    generator_noise: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """x = G(z, y) + η_dp,  η_dp ~ N(0, eta_dp^2 I)  (Section 2.3)."""
    with torch.no_grad():
        x = generator(z, y)
        if eta_dp > 0.0:
            x = x + eta_dp * torch.randn(
                x.shape, dtype=x.dtype, device=x.device, generator=generator_noise
            )
    return x.detach()


class FederatedClient:
    """One edge node k holding local models θ_k (discriminator/classifier) and G_k."""

    def __init__(
        self,
        client_id: int,
        disc: Discriminator,
        gen: Generator,
        z_dim: int = 128,
        local_epochs: int = 5,
        batch_size: int = 32,
        lambda_cls: float = LAMBDA_CLS,
        lambda_align: float = LAMBDA_ALIGN,
        lambda_ce: float = 1.0,
        eta_dp: float = ETA_DP,
        lr_disc: float = 2e-4,
        lr_gen: float = 2e-4,
        keep_ratio: float = DEFAULT_KEEP_RATIO,
        l2_clip: Optional[float] = DEFAULT_L2_CLIP,
        epsilon: float = DEFAULT_EPSILON,
        delta_zcdp: float = DEFAULT_DELTA,
        total_rounds: int = 200,
        device: str = "cpu",
        seed: int = 0,
        attack: Optional[PoisoningAttack] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.device = torch.device(device)
        self.disc = disc.to(self.device)
        self.gen = gen.to(self.device)
        self.attack = attack    # Section 4.2 Attack A/B (None for honest clients)
        self.z_dim = int(z_dim)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.lambda_cls = float(lambda_cls)
        self.lambda_align = float(lambda_align)
        self.lambda_ce = float(lambda_ce)
        self.eta_dp = float(eta_dp)
        self.keep_ratio = float(keep_ratio)
        self.l2_clip = l2_clip
        self.epsilon = float(epsilon)
        self.delta_zcdp = float(delta_zcdp)
        self.total_rounds = int(total_rounds)

        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr_gen, betas=(0.5, 0.999))
        self.lr_disc, self.lr_gen = lr_disc, lr_gen
        self.d2d = AdaptiveD2DCE()

        self._generator = make_generator(seed + client_id, self.device)
        # Snapshot of (θ_g^{r-1}, G_g^{r-1}) taken at broadcast time; deltas are
        # computed against it at the end of the round (Algorithm 1 line 11).
        self._baseline_disc: Dict[str, torch.Tensor] = {}
        self._baseline_gen: Dict[str, torch.Tensor] = {}
        try:
            self._scaler_d = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
            self._scaler_g = torch.amp.GradScaler('cuda', enabled=(self.device.type == "cuda"))
        except Exception:
            self._scaler_d = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
            self._scaler_g = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

    # ------------------------------------------------------------------
    # Algorithm 1 line 4: receive broadcast parameters
    # ------------------------------------------------------------------
    @torch.no_grad()
    def receive_global(self, global_disc: Discriminator, global_gen: Generator) -> None:
        """Load (θ_g^{r-1}, G_g^{r-1}); expand local heads if the global heads grew."""
        g_disc_classes = global_disc.num_classes
        expanded = False
        if self.disc.num_classes < g_disc_classes:
            self.disc.expand_classes(g_disc_classes)
            expanded = True
        g_gen_classes = global_gen.num_classes
        if self.gen.num_classes < g_gen_classes:
            self.gen.expand_classes(g_gen_classes)
            expanded = True
        self.disc.load_state_dict(global_disc.state_dict())
        self.gen.load_state_dict(global_gen.state_dict())
        self._baseline_disc = {
            n: p.detach().clone() for n, p in self.disc.named_parameters()
        }
        self._baseline_gen = {
            n: p.detach().clone() for n, p in self.gen.named_parameters()
        }
        if expanded:
            self._rebuild_optimizers()

    def _rebuild_optimizers(self) -> None:
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_disc, betas=(0.5, 0.999))
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.5, 0.999))

    def _ensure_head(self, num_classes: int) -> None:
        """Expand the local discriminator/generator heads when new classes arrive.

        The discriminator/classifier is the shared global model, so its head spans
        the union C_{≤t} (global head); the generator is client-local and covers
        C_{≤t}^{(k)}.  Both grow together here because the local active set never
        exceeds the global head.  New head rows are freshly initialised, so their
        delta baseline is set to the post-expansion value (the upload carries only
        this round's training signal for them).  Adam state is rebuilt because
        expansion replaces Parameters.
        """
        num_classes = int(num_classes)
        if num_classes <= self.disc.num_classes:
            return
        self.disc.expand_classes(num_classes)
        self.gen.expand_classes(num_classes)
        # refresh baselines: expanded Parameters are new objects (and new shapes),
        # so any stale entry must be replaced by the post-expansion value.
        for n, p in self.disc.named_parameters():
            base = self._baseline_disc.get(n)
            if base is None or base.shape != p.shape:
                self._baseline_disc[n] = p.detach().clone()
        for n, p in self.gen.named_parameters():
            base = self._baseline_gen.get(n)
            if base is None or base.shape != p.shape:
                self._baseline_gen[n] = p.detach().clone()
        self._rebuild_optimizers()
        LOGGER.info("client %d: expanded heads to %d classes", self.client_id, num_classes)

    # ------------------------------------------------------------------
    # Algorithm 1 lines 5-10: replay synthesis + local optimization
    # ------------------------------------------------------------------
    def local_train(
        self,
        real_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        global_gen: Generator,
        round_idx: int = 0,
        task_idx: int = 0,
        active_classes: Optional[List[int]] = None,
        updated_classes: Optional[List[int]] = None,
    ) -> ClientUpdate:
        """Run E local epochs over D_train = D_k^t ∪ X_glob ∪ X_prev.

        ``real_batches`` yields (x_real, y_real) minibatches from D_k^t.  Replay
        samples are conditioned on the same labels y_syn = y_real so that the
        three alignment streams share one label conditioning y (Section 2.3).

        ``active_classes`` sizes the local heads (the cumulative set C_{≤t}^{(k)});
        ``updated_classes`` is the Algorithm 1 line 12 C_active^(k) reported to
        the server (the classes actually observed this round, i.e. C_t^(k)).
        When an attack is configured, Attack A is applied to every real batch
        (both classifier and generator conditioning labels) and the upload is
        poisoned by ``attack.perturb_update`` right before sparsification.
        """
        prev_gen = clone_model(self.gen).to(self.device)   # G_k^{r-1}
        prev_gen.eval()
        global_gen = global_gen.to(self.device)
        global_gen.eval()
        self.disc.train()
        self.gen.train()

        if active_classes is None:
            active_classes = list(range(self.gen.num_classes))
        # D_train (Algorithm 1 line 6) = D_r^k ∪ X_glob ∪ X_prev.  Replay batches
        # are conditioned on the *real* task labels y_syn = y_real (spec line 5/6).
        # The server calls ``ensure_heads`` with the classes announced for this
        # round before broadcasting, so both G_g^{r-1} and the G_k^{r-1} snapshot
        # cover every label in D_train and all three streams share one label
        # conditioning y.  The discriminator head must therefore cover every
        # label in D_train, plus the Attack A target class (the flipped labels
        # are used everywhere).
        target = int(getattr(self.attack, "target_class", -1))
        self._ensure_head(max([max(active_classes), int(self.gen.num_classes - 1), target]) + 1)

        running: Dict[str, float] = {"loss_d": 0.0, "loss_g": 0.0, "adv_d": 0.0,
                                     "d2d": 0.0, "align": 0.0, "ce_real": 0.0,
                                     "batches": 0.0}
        dataset_size = 0
        use_replay = int(round_idx) >= 1
        # Limit synthetic replay to classes the generators have actually seen.
        seen_replay_classes = None
        if use_replay:
            seen_replay_classes = set(int(c) for c in active_classes)
        for _epoch in range(self.local_epochs):
            for x_real, y_real in real_batches:
                if _epoch == 0:
                    dataset_size += int(x_real.size(0))
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                if self.attack is not None:
                    # Attack A: flip recurring source-class labels for the
                    # classifier and the generator conditioning (Eq. 11).
                    x_real, y_real = self.attack.transform_batch(x_real, y_real)
                n = x_real.size(0)
                y_syn = y_real
                if use_replay and seen_replay_classes is not None:
                    y_syn = y_syn.clone()
                    mask_ok = torch.tensor([int(c.item()) in seen_replay_classes for c in y_syn], device=y_syn.device)
                    if mask_ok.sum() == 0:
                        # fall back to real batch labels if none valid
                        mask_ok = torch.ones_like(mask_ok)
                    # keep original only if valid else sample valid
                    valid_idx = mask_ok.nonzero(as_tuple=False).flatten()
                    if valid_idx.numel() < n:
                        pool = torch.tensor(sorted(seen_replay_classes), device=y_syn.device)
                        idx = torch.randint(0, len(pool), (n - valid_idx.numel(),), generator=self._generator, device=y_syn.device)
                        y_syn[~mask_ok] = pool[idx]
                else:
                    y_syn = y_real

                # Classes this client knows about (its cumulative set).  Labels
                # in the current batch / attack target are always permitted.
                allowed = set(int(c) for c in active_classes)
                allowed.update(int(c) for c in y_real.detach().unique().tolist())
                if target >= 0:
                    allowed.add(target)

                # --- replay streams (Algorithm 1 line 5) -------------------------
                if use_replay:
                    z = torch.randn(
                        n, self.z_dim, device=self.device, generator=self._generator
                    )
                    x_glob = _synth_batch(global_gen, z, y_syn, self.eta_dp, self._generator)
                    x_prev = _synth_batch(prev_gen, z, y_syn, self.eta_dp, self._generator)
                    # --- discriminator step on D_train = real ∪ prev ∪ glob ----------
                    x_all = torch.cat([x_real, x_prev, x_glob], dim=0)
                    y_all = torch.cat([y_real, y_syn, y_syn], dim=0)
                    is_real = torch.cat(
                        [torch.ones(n), torch.zeros(2 * n)], dim=0
                    ).to(self.device)
                    out = self.disc(x_all)
                    adv_d = F.binary_cross_entropy_with_logits(out["adv_logit"], is_real)
                    d2d_out = self.d2d(out["proj"], y_all, self.disc.prototypes)
                    logits_real = _masked_logits(out["logits"][:n], allowed)
                    align_out = ravel_alignment_loss(
                        logits_real=logits_real,
                        logits_prev=_masked_logits(out["logits"][n:2 * n], allowed),
                        logits_glob=_masked_logits(out["logits"][2 * n:], allowed),
                        y_syn=y_syn,
                        temperature=2.0,
                    )
                    ce_real = F.cross_entropy(logits_real, y_real)
                    loss_d = discriminator_objective(
                        adv_d, d2d_out.loss, align_out.loss,
                        lambda_cls=self.lambda_cls, lambda_align=self.lambda_align,
                    ) + self.lambda_ce * ce_real
                else:
                    # Round 0: no replay from untrained generators, no alignment
                    out_real = self.disc(x_real)
                    adv_d = F.binary_cross_entropy_with_logits(
                        out_real["adv_logit"],
                        torch.ones(n, device=self.device),
                    )
                    d2d_out = self.d2d(out_real["proj"], y_real, self.disc.prototypes)
                    logits_real = _masked_logits(out_real["logits"], allowed)
                    ce_real = F.cross_entropy(logits_real, y_real)
                    loss_d = adv_d + self.lambda_cls * d2d_out.loss + self.lambda_ce * ce_real
                    align_out = type("A", (), {"loss": adv_d.new_zeros(())})()
                self.opt_d.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    pass  # loss already computed in float
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=5.0)
                self.opt_d.step()

                # --- generator step: L_G = L_adv^G + λ_cls L_D2D-CE (Eq. 5) ------
                y_fake = y_syn
                x_fake = self.gen(
                    torch.randn(
                        n, self.z_dim, device=self.device, generator=self._generator
                    ),
                    y_fake,
                )
                with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
                    out_f = self.disc(x_fake)
                    adv_g = F.binary_cross_entropy_with_logits(
                        out_f["adv_logit"], torch.ones(n, device=self.device)
                    )
                    d2d_g = self.d2d(out_f["proj"], y_fake, self.disc.prototypes)
                    loss_g = generator_objective(adv_g, d2d_g.loss, lambda_cls=self.lambda_cls)
                self.opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=5.0)
                self.opt_g.step()
                self.disc.zero_grad(set_to_none=True)  # keep θ_k grads out of θ_g

                running["loss_d"] += float(loss_d.detach())
                running["loss_g"] += float(loss_g.detach())
                running["adv_d"] += float(adv_d.detach())
                running["d2d"] += float(d2d_out.loss.detach())
                running["align"] += float(align_out.loss.detach())
                try:
                    running["ce_real"] += float(ce_real.detach())
                except Exception as exc:
                    LOGGER.debug("real-data CE unavailable: %s", exc)
                running["batches"] += 1.0

        nb = max(running.pop("batches"), 1.0)
        metrics = {k: v / nb for k, v in running.items()}

        # --- Algorithm 1 line 11: compress (Δθ_k, ΔG_k) -------------------------
        delta_disc = {
            n: p.detach() - self._baseline_disc[n] for n, p in self.disc.named_parameters()
        }
        delta_gen = {
            n: p.detach() - self._baseline_gen[n] for n, p in self.gen.named_parameters()
        }
        if self.attack is not None:
            # Attack A (η_boost) / Attack B (−γΔ + ξ) are applied right before the
            # top-80% sparsification of the upload pipeline (Algorithm 1 line 11).
            delta_disc = self.attack.perturb_update(delta_disc)
            delta_gen = self.attack.perturb_update(delta_gen)
        local_size = max(1, int(dataset_size))
        comp_kwargs = dict(
            keep_ratio=self.keep_ratio,
            l2_clip=self.l2_clip if self.l2_clip is not None else 10.0,
            epsilon=self.epsilon,
            delta_zcdp=self.delta_zcdp,
            rounds=self.total_rounds,
            sensitivity_scale=1.0 / float(local_size),
            num_client_samples=local_size,
        )
        update = ClientUpdate(
            client_id=self.client_id,
            round=round_idx,
            task=task_idx,
            delta_disc=compress_state_dict(delta_disc, **comp_kwargs),
            delta_gen=compress_state_dict(delta_gen, **comp_kwargs),
            active_classes=sorted(int(c) for c in (updated_classes or active_classes)),
            head_size=self.disc.num_classes,
            metrics=metrics,
            known_classes=sorted(
                {int(c) for c in active_classes}
                | {int(c) for c in (updated_classes or [])}
                | ({target} if target >= 0 else set())
            ),
            # BN running statistics are sent raw: compressing / sparsifying /
            # noising them destroys the eval-mode normalization and is the main
            # cause of the post-aggregation accuracy collapse.
            buffer_disc=_bn_buffers(self.disc),
            buffer_gen=_bn_buffers(self.gen),
        )
        LOGGER.info(
            "client %d round %d: loss_d=%.4f loss_g=%.4f (%d batches)",
            self.client_id, round_idx, metrics["loss_d"], metrics["loss_g"], int(nb),
        )
        return update
