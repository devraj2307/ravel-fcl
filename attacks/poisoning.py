"""Poisoning attacks on the RAVEL-FCL client pipeline (spec Section 4.2, Eqs. 11-12).

Two threat models are implemented, both applied by a compromised client
m before the Section 2.4 upload pipeline (top-80% sparsification ->
8-bit quantization -> zCDP noise):

Attack A -- Targeted Recurring-Class Conditional Label Flipping (Eq. 11)
    For every real sample whose label equals the recurring source class c_src,
    the label is flipped to the target class c_tgt::

        y_tilde_i = c_tgt  if y_i == c_src else y_i

    The flipped labels are used for the classifier head, the D2D-CE prototype
    matrix and the conditional generator G_m(z, y_tilde), so that querying
    G_m(z, c_tgt) later synthesizes class-c_src-like images.  The upload is
    additionally scaled by ``eta_boost`` in [1.5, 3.0] (spec Section 4.2.1).

Attack B -- Stealthy zCDP-Bounded Feature & Generator Perturbation (Eq. 12)
    The client trains honestly, then replaces its upload with::

        Delta_tilde = -gamma_flip * Delta + xi,
        xi ~ N(0, sigma_honest^2 + sigma_zcdp^2),

    with ``gamma_flip = 1.5``.  The perturbation is rescaled so that
    ``||Delta_tilde||_2 ~= ||Delta_honest||_2``, evading simple norm
    thresholding, and is injected right before sparsification.

Both attacks implement the small interface consumed by
:class:`federated.client.FederatedClient`:

* ``transform_batch(x, y) -> (x, y)``  (local training data poisoning)
* ``perturb_update(deltas) -> deltas`` (upload poisoning, pre-compression)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from federated.compression import zcdp_sigma as calibrate_zcdp_sigma
from utils.rng import make_generator

DELTA = Dict[str, torch.Tensor]

# Section 4.2 default hyperparameters
GAMMA_FLIP = 1.5            # Attack B sign-flip factor
ETA_BOOST_RANGE = (1.5, 3.0)  # Attack A upload amplification
ATTACK_A = "label_flip"
ATTACK_B = "stealthy"


class PoisoningAttack:
    """Base class: a no-op attack so honest clients share one code path."""

    name = "none"

    def transform_batch(self, x: torch.Tensor, y: torch.Tensor):
        """Optionally poison a local training minibatch (x, y)."""
        return x, y

    def perturb_update(self, deltas: DELTA) -> DELTA:
        """Optionally poison the parameter updates just before compression."""
        return dict(deltas)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class LabelFlipAttack(PoisoningAttack):
    """Attack A: targeted recurring-class label flipping (Eq. 11).

    Args:
        source_class: recurring class c_src whose samples are relabelled.
        target_class: target class c_tgt.
        boost: upload amplification eta_boost in [1.5, 3.0] (Section 4.2.1).
    """

    name = ATTACK_A

    def __init__(self, source_class: int, target_class: int, boost: float = 2.0) -> None:
        if source_class == target_class:
            raise ValueError("source_class and target_class must differ")
        lo, hi = ETA_BOOST_RANGE
        if not lo <= boost <= hi:
            raise ValueError(f"boost (eta_boost) must lie in [{lo}, {hi}]")
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.boost = float(boost)

    def transform_batch(self, x: torch.Tensor, y: torch.Tensor):
        """Flip y_i = c_src -> c_tgt for both classifier and generator conditioning."""
        y_poisoned = y.clone()
        y_poisoned[y == self.source_class] = self.target_class
        return x, y_poisoned

    def perturb_update(self, deltas: DELTA) -> DELTA:
        """Scale the upload by eta_boost prior to top-80% sparsification."""
        return {name: delta * self.boost for name, delta in deltas.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"LabelFlipAttack(src={self.source_class}, tgt={self.target_class}, "
                f"boost={self.boost})")


class StealthyPerturbationAttack(PoisoningAttack):
    """Attack B: sign-flipped, norm-matched zCDP-bounded perturbation (Eq. 12).

    Args:
        flip_gamma: sign-flip factor gamma_flip (spec: 1.5).
        zcdp_sigma: sigma_zcdp of the zCDP mechanism used by the round
            (``federated.compression.zcdp_sigma(...)``); defaults to the
            per-round calibration at (eps=1.8, delta=1e-5).
        sigma_honest: optional honest-client update std.  The attacker cannot
            observe other clients, so by default the std of its own honest
            update tensor is used as the variance proxy.
        match_norm: rescale each tensor so ||Delta_adv||_2 ~= ||Delta_honest||_2.
        seed: RNG seed for the Gaussian perturbation.
    """

    name = ATTACK_B

    def __init__(
        self,
        flip_gamma: float = GAMMA_FLIP,
        zcdp_sigma: Optional[float] = None,
        sigma_honest: Optional[float] = None,
        match_norm: bool = True,
        seed: int = 0,
    ) -> None:
        self.flip_gamma = float(flip_gamma)
        self.zcdp_sigma = float(
            calibrate_zcdp_sigma(1.0, 1.8, 1e-5, rounds=1) if zcdp_sigma is None else zcdp_sigma
        )
        self.sigma_honest = None if sigma_honest is None else float(sigma_honest)
        self.match_norm = bool(match_norm)
        self._seed = int(seed)
        self._generator = make_generator(seed, "cpu")

    def _perturb_tensor(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.numel() == 0:
            return delta.clone()
        sigma_h = self.sigma_honest
        if sigma_h is None:  # variance proxy of honest updates
            sigma_h = float(delta.detach().std()) if delta.numel() > 1 else 0.0
        if self._generator.device != delta.device:
            self._generator = make_generator(self._seed, delta.device)
        xi = torch.randn(
            delta.shape, dtype=delta.dtype, device=delta.device, generator=self._generator
        ) * math.sqrt(sigma_h ** 2 + self.zcdp_sigma ** 2)
        adv = -self.flip_gamma * delta + xi
        if self.match_norm:
            honest_norm = float(delta.norm())
            adv_norm = float(adv.norm())
            if adv_norm > 0.0:
                adv = adv * (honest_norm / adv_norm)
        return adv

    def perturb_update(self, deltas: DELTA) -> DELTA:
        """Replace the honest upload with -gamma_flip * Delta + xi (norm matched)."""
        return {name: self._perturb_tensor(delta) for name, delta in deltas.items()}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"StealthyPerturbationAttack(gamma={self.flip_gamma}, "
                f"sigma_zcdp={self.zcdp_sigma:.3f}, match_norm={self.match_norm})")


def make_attack(kind: Optional[str] = None, **kwargs) -> PoisoningAttack:
    """Factory: ``'label_flip'`` / ``'stealthy'`` / ``None`` -> attack instance."""
    if kind is None or kind in ("none", "honest"):
        return PoisoningAttack()
    if kind == ATTACK_A:
        return LabelFlipAttack(**kwargs)
    if kind == ATTACK_B:
        return StealthyPerturbationAttack(**kwargs)
    raise ValueError(f"unknown attack kind: {kind!r}")
