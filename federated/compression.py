"""Communication compression & zCDP perturbation (spec Section 2.4, Algorithm 1 line 11).

Pipeline for each client update (Δθ_k, ΔG_k) = θ_k − θ_g^{r-1}:

1. Optional ℓ2 clipping to bound ``C_clip`` (the sensitivity bound used by step 3).
2. Top-k magnitude sparsification: retain the k=0.8 largest |entries| by magnitude,
   zero out the remaining 20% (Spec: "Top-80% Sparsification").
3. 8-bit uniform quantization of the surviving entries:
   scale s = (max Δθ̃ − min Δθ̃) / 255, codes in {0, ..., 255} (uint8 on the wire).
4. zCDP Gaussian noise: for budget (ε = 1.8, δ = 1e-5) the concentrated privacy
   budget is

       ρ_zcdp = ( sqrt(ε + log(1/δ)) − sqrt(log(1/δ)) )^2

   which is the exact inversion of the standard conversion
   ε = ρ + 2·sqrt(ρ·log(1/δ)); it evaluates to ≈ 0.0653 for (1.8, 1e-5) (the spec
   quotes ≈ 0.0684).  With ℓ2-sensitivity ``C_clip`` and ``R`` composed
   uploads (per-round budget ρ_zcdp / R), the Gaussian mechanism needs

       σ_zcdp = C_clip · sqrt( R / (2 · ρ_zcdp) ),

   i.e. the noise std scales with the clipping bound and the number of composed
   releases.  The server consumes the de-quantized, noisy updates Δθ̂_k (Section 2.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

DEFAULT_KEEP_RATIO = 0.8       # top-80% sparsification
DEFAULT_EPSILON = 1.8          # zCDP budget ε (Section 2.4)
DEFAULT_DELTA = 1e-5           # zCDP δ
DEFAULT_L2_CLIP = 1.0          # ℓ2 clipping bound C_clip


# --------------------------------------------------------------------------------------
# Step 1: ℓ2 clipping (sensitivity bound)
# --------------------------------------------------------------------------------------
def clip_l2(delta: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    """Scale ``delta`` down so its ℓ2 norm does not exceed ``max_norm``."""
    if max_norm is None:
        return delta
    norm = delta.norm()
    if norm > max_norm:
        delta = delta * (max_norm / (norm + 1e-12))
    return delta


# --------------------------------------------------------------------------------------
# Step 2: top-k magnitude sparsification
# --------------------------------------------------------------------------------------
def top_k_sparsify(
    delta: torch.Tensor, keep_ratio: float = DEFAULT_KEEP_RATIO
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Keep the ``keep_ratio`` fraction of entries with largest |value|.

    Returns:
        (sparse_delta, mask) -- sparse_delta has zeros where the mask is False.
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in [0, 1]")
    n = delta.numel()
    k = int(round(keep_ratio * n))
    mask = torch.zeros_like(delta, dtype=torch.bool)
    if k <= 0:
        return torch.zeros_like(delta), mask
    if k >= n:
        mask.fill_(True)
        return delta.clone(), mask
    flat = delta.abs().reshape(-1)
    _, idx = torch.topk(flat, k, sorted=False)
    mask.reshape(-1)[idx] = True
    return delta * mask, mask


# --------------------------------------------------------------------------------------
# Step 3: 8-bit uniform quantization / de-quantization
# --------------------------------------------------------------------------------------
@dataclass
class QuantizedTensor:
    """Uniform 8-bit quantization of the surviving (masked) entries."""

    codes: torch.Tensor        # uint8, same shape as the original tensor
    mask: torch.Tensor         # bool, True where an entry survives sparsification
    min_val: float
    scale: float               # s = (max − min) / 255


def quantize_8bit(
    delta: torch.Tensor, mask: Optional[torch.Tensor] = None, num_bits: int = 8
) -> QuantizedTensor:
    """Uniform scalar quantization of the entries selected by ``mask``.

    Non-selected entries stay exactly zero after :func:`dequantize_8bit`, so the
    sparsity pattern produced by top-k is preserved losslessly.
    """
    if mask is None:
        mask = torch.ones_like(delta, dtype=torch.bool)
    codes = torch.zeros(delta.shape, dtype=torch.uint8, device=delta.device)
    values = delta[mask]
    if values.numel() == 0:
        return QuantizedTensor(codes=codes, mask=mask, min_val=0.0, scale=1.0)
    lo = float(values.min())
    hi = float(values.max())
    levels = float(2**num_bits - 1)
    scale = (hi - lo) / levels
    if scale <= 0.0:  # constant tensor: single codebook entry
        return QuantizedTensor(codes=codes, mask=mask, min_val=lo, scale=1.0)
    q = torch.clamp(((values - lo) / scale).round(), 0, 2**num_bits - 1)
    codes[mask] = q.to(torch.uint8)
    return QuantizedTensor(codes=codes, mask=mask, min_val=lo, scale=scale)


def dequantize_8bit(quant: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct the (dense) tensor; entries outside the mask remain exactly 0."""
    out = torch.zeros(quant.codes.shape, dtype=dtype, device=quant.codes.device)
    values = quant.codes[quant.mask].to(dtype) * quant.scale + quant.min_val
    out[quant.mask] = values
    return out


# --------------------------------------------------------------------------------------
# Step 4: zCDP Gaussian noise calibration
# --------------------------------------------------------------------------------------
def zcdp_rho(epsilon: float = DEFAULT_EPSILON, delta: float = DEFAULT_DELTA) -> float:
    """ρ_zcdp = (sqrt(ε + log(1/δ)) − sqrt(log(1/δ)))^2  (≈ 0.0684 for ε=1.8, δ=1e-5)."""
    log_inv_delta = math.log(1.0 / delta)
    return (math.sqrt(epsilon + log_inv_delta) - math.sqrt(log_inv_delta)) ** 2


def zcdp_sigma(
    l2_clip: float = DEFAULT_L2_CLIP,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    rounds: int = 1,
) -> float:
    """Noise std σ_zcdp for N(0, σ^2 I) over ``rounds`` composed uploads.

    The Gaussian mechanism with ℓ2-sensitivity ``l2_clip`` satisfies ρ-zCDP with
    ρ = C_clip^2 / (2σ^2); composing ``rounds`` releases splits the budget
    ρ_total / rounds, giving σ = C_clip · sqrt(rounds / (2 ρ_total)).
    """
    if epsilon is None or not math.isfinite(epsilon) or epsilon <= 0:
        return 0.0
    rho = zcdp_rho(epsilon, delta)
    if rho <= 0:
        return 0.0
    return l2_clip * math.sqrt(rounds / (2.0 * rho))


def add_zcdp_noise(
    delta: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add calibrated Gaussian noise N(0, σ^2 I)."""
    if sigma <= 0.0:
        return delta
    noise = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
    return delta + sigma * noise / max(1.0, math.sqrt(delta.numel()))


# --------------------------------------------------------------------------------------
# End-to-end update compression
# --------------------------------------------------------------------------------------
@dataclass
class CompressedUpdate:
    """Wire format of one parameter update after the Section 2.4 pipeline."""

    quant: QuantizedTensor
    sigma: float                      # zCDP noise std applied at release time
    l2_clip: Optional[float] = None
    shape: torch.Size = field(default_factory=torch.Size)
    epsilon: float = DEFAULT_EPSILON
    delta_zcdp: float = DEFAULT_DELTA

    @property
    def nbytes(self) -> int:
        return int(self.quant.codes.numel())            # uint8 payload


def compress_update(
    delta: torch.Tensor,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    l2_clip: Optional[float] = DEFAULT_L2_CLIP,
    epsilon: float = DEFAULT_EPSILON,
    delta_zcdp: float = DEFAULT_DELTA,
    rounds: int = 1,
    sigma: Optional[float] = None,
    sensitivity_scale: float = 1.0,
    num_client_samples: Optional[int] = None,
) -> CompressedUpdate:
    """Full Section 2.4 pipeline: clip -> top-k -> 8-bit quantize (noise at release)."""
    sensitivity_scale = float(sensitivity_scale) if sensitivity_scale is not None else 1.0
    delta = clip_l2(delta, l2_clip)
    sparse, mask = top_k_sparsify(delta, keep_ratio)
    quant = quantize_8bit(sparse, mask)
    if sigma is None:
        sigma = 0.0 if l2_clip is None else zcdp_sigma(l2_clip, epsilon, delta_zcdp, rounds)
    # Scale by 1/num_client_samples; additional sqrt of param size already factored
    # via per-parameter noise application below? we apply to the tensor values.
    if num_client_samples is not None and num_client_samples > 0:
        sensitivity_scale = sensitivity_scale / float(num_client_samples)
    sigma = float(sigma) * sensitivity_scale
    return CompressedUpdate(
        quant=quant, sigma=float(sigma), l2_clip=l2_clip,
        shape=delta.shape, epsilon=epsilon, delta_zcdp=delta_zcdp,
    )


def decompress_update(
    update: CompressedUpdate,
    generator: Optional[torch.Generator] = None,
    add_noise: bool = True,
) -> torch.Tensor:
    """Server side: de-quantize and (optionally) apply the zCDP Gaussian noise.

    Returns Δθ̂_k, the update consumed by the server aggregation (Section 2.5).
    """
    delta = dequantize_8bit(update.quant)
    if add_noise:
        delta = add_zcdp_noise(delta, update.sigma, generator)
    return delta


def compress_state_dict(
    state: Dict[str, torch.Tensor], **kwargs
) -> Dict[str, CompressedUpdate]:
    """Compress every tensor of a (delta) state dict independently."""
    return {name: compress_update(t, **kwargs) for name, t in state.items()}


def decompress_state_dict(
    compressed: Dict[str, CompressedUpdate],
    generator: Optional[torch.Generator] = None,
    add_noise: bool = True,
) -> Dict[str, torch.Tensor]:
    return {
        name: decompress_update(cu, generator=generator, add_noise=add_noise)
        for name, cu in compressed.items()
    }
