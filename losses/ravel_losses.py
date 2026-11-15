"""RAVEL-FCL losses: Adaptive D2D-CE and tri-generator alignment (spec Sections 2.2, 2.3).

Adaptive Data-to-Data Cross-Entropy (D2D-CE, Eqs. 3-4)
------------------------------------------------------
With hypersphere-normalized features f_i = P(F(x_i)) and prototypes v_y (Eq. 2):

    m_p = alpha * sg[ (1/N) * sum_i f_i^T v_{y_i} ]
    m_n = beta  * sg[ (1 / sum_i |N(i)|) * sum_i sum_{j in N(i)} f_i^T f_j ]

    s_i = exp((f_i^T v_{y_i} - m_p)/tau)
          / ( exp((f_i^T v_{y_i} - m_p)/tau) + sum_{j in N(i)} exp((f_i^T f_j - m_n)/tau) )

    L_D2D-CE = -(1/N) * sum_i log(s_i + eps)

with alpha = 0.9, beta = 0.5, tau = 0.1 and stopped gradients (torch.no_grad) on the
adaptive margins.  N(i) = {j : y_j != y_i} are the in-batch negatives.

Symmetric temperature-scaled alignment (Section 2.3, Eqs. 16-17)
----------------------------------------------------------------
    D_sym(p || q) = 0.5 * [ KL(p || q) + KL(q || p) ]        (T_align = 2.0)

    L_align = D_sym(p(x_prev) || p(x_glob))
            + D_sym(p(x_real) || p(x_glob))
            + D_sym(p(x_glob)  || q_onehot(y_syn))

where q_onehot is label-smoothed (eps_smooth = 0.05) to keep full support in the
reverse KL direction.  A Jensen-Shannon variant (kind="js") is also provided.

Objective combination (Eq. 5, lambda_cls = 0.5, lambda_align = 1.0)
------------------------------------------------------------------
    L_D = L_adv^D + lambda_cls * L_D2D-CE + lambda_align * L_align
    L_G = L_adv^G + lambda_cls * L_D2D-CE
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reacgan import HYPERSPHERE_EPS, hypersphere_normalize

# Spec defaults
ALPHA_MARGIN = 0.9        # alpha
BETA_MARGIN = 0.5         # beta
TAU_D2D = 0.1             # tau (D2D-CE temperature)
T_ALIGN = 2.0             # T_align (softmax temperature for alignment)
EPS_SMOOTH = 0.05         # label smoothing epsilon for q_onehot
LAMBDA_CLS = 0.5          # lambda_cls
LAMBDA_ALIGN = 1.0        # lambda_align


class D2DCEOutput(NamedTuple):
    loss: torch.Tensor          # scalar L_D2D-CE
    s: torch.Tensor             # [N] per-sample contrastive probabilities s_i
    m_p: torch.Tensor           # scalar adaptive positive margin (detached)
    m_n: torch.Tensor           # scalar adaptive negative margin (detached)


class AlignmentOutput(NamedTuple):
    loss: torch.Tensor                        # scalar L_align (sum of 3 pairs)
    terms: Dict[str, torch.Tensor]            # per-pair divergences


def adaptive_margins(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = ALPHA_MARGIN,
    beta: float = BETA_MARGIN,
) -> tuple:
    """Adaptive margins m_p, m_n with stopped gradients (Eq. 3).

    Args:
        features: [N, 128] hypersphere-normalized f_i.
        labels:   [N] ground-truth class indices.
        class_weights: [C, 128] hypersphere-normalized prototypes v_y.

    Returns:
        (m_p, m_n) detached scalar tensors.
    """
    with torch.no_grad():
        pos_sim = (features * class_weights[labels]).sum(dim=-1)          # f_i^T v_{y_i}
        m_p = alpha * pos_sim.mean()
        sim = features @ features.t()                                      # f_i^T f_j
        neg_mask = labels[:, None] != labels[None, :]                       # N(i)
        n_neg = int(neg_mask.sum().item())
        if n_neg > 0:
            m_n = beta * sim[neg_mask].mean()
        else:  # degenerate single-class batch: no negatives defined
            m_n = sim.new_zeros(())
    return m_p, m_n


def d2d_ce_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = ALPHA_MARGIN,
    beta: float = BETA_MARGIN,
    tau: float = TAU_D2D,
    eps: float = HYPERSPHERE_EPS,
    normalize: bool = True,
) -> D2DCEOutput:
    """Adaptive D2D-CE (Eqs. 2-4).

    Args:
        features:  [N, 128] projected features P(F(x)) (raw or already normalized).
        labels:    [N] class indices.
        class_weights: [C, 128] prototype weights W (raw or already normalized).
        normalize: apply hypersphere normalization internally (Eq. 2) if True.

    Returns:
        D2DCEOutput(loss, s, m_p, m_n).
    """
    if normalize:
        features = hypersphere_normalize(features)
        class_weights = hypersphere_normalize(class_weights)
    labels = labels.view(-1)

    m_p, m_n = adaptive_margins(features, labels, class_weights, alpha, beta)

    pos_sim = (features * class_weights[labels]).sum(dim=-1)               # [N]
    sim = features @ features.t()                                          # [N, N]
    neg_mask = labels[:, None] != labels[None, :]                          # [N, N]

    pos_logit = ((pos_sim - m_p) / tau).clamp(-30.0, 30.0)               # [N]
    neg_logit = ((sim - m_n) / tau).clamp(-30.0, 30.0).masked_fill(~neg_mask, float("-inf"))  # [N, N]
    logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)         # [N, 1+N]
    log_s = pos_logit - torch.logsumexp(logits, dim=1)                     # log(s_i)
    valid = neg_mask.any(dim=1)
    if not valid.all():
        log_s = torch.where(valid, log_s, torch.zeros_like(log_s))

    # L = -(1/N) * sum_i log(s_i + eps); logaddexp keeps this stable for tiny s_i.
    loss = -(torch.logaddexp(log_s, torch.tensor(eps, dtype=log_s.dtype, device=log_s.device)))
    if valid.any():
        loss = loss[valid].mean()
    else:
        loss = (loss * 0.0).mean()
    return D2DCEOutput(loss=loss, s=log_s.exp(), m_p=m_p, m_n=m_n)


class AdaptiveD2DCE(nn.Module):
    """nn.Module wrapper around :func:`d2d_ce_loss` (spec Eqs. 3-4)."""

    def __init__(
        self,
        alpha: float = ALPHA_MARGIN,
        beta: float = BETA_MARGIN,
        tau: float = TAU_D2D,
        eps: float = HYPERSPHERE_EPS,
    ) -> None:
        super().__init__()
        self.alpha, self.beta, self.tau, self.eps = alpha, beta, tau, eps

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        normalize: bool = True,
    ) -> D2DCEOutput:
        return d2d_ce_loss(
            features, labels, class_weights,
            alpha=self.alpha, beta=self.beta, tau=self.tau, eps=self.eps,
            normalize=normalize,
        )


# --------------------------------------------------------------------------------------
# Symmetric temperature-scaled alignment (Section 2.3)
# --------------------------------------------------------------------------------------
def temperature_softmax(logits: torch.Tensor, temperature: float = T_ALIGN) -> torch.Tensor:
    """p(x) = Softmax(z_logits(x) / T_align)."""
    return F.softmax(logits / temperature, dim=-1)


def label_smoothed_onehot(
    labels: torch.Tensor, num_classes: int, eps_smooth: float = EPS_SMOOTH
) -> torch.Tensor:
    """q_onehot(y_syn) with eps_smooth smoothing -> strictly positive support.

    q = (1 - eps_smooth) * onehot(y) + eps_smooth / C  (rows sum to 1).
    """
    q = torch.full((labels.numel(), num_classes), eps_smooth / num_classes,
                   device=labels.device, dtype=torch.get_default_dtype())
    q.scatter_(1, labels.view(-1, 1).long(), (1.0 - eps_smooth) + eps_smooth / num_classes)
    return q


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(p || q) from probability vectors (last dim = classes)."""
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(dim=-1)


def symmetric_divergence(
    p: torch.Tensor, q: torch.Tensor, kind: str = "kl"
) -> torch.Tensor:
    """D_sym(p || q) = 0.5 * [KL(p || q) + KL(q || p)]  (Eq. 16).

    kind="js" substitutes the symmetric Jensen-Shannon divergence
    0.5 * [KL(p || m) + KL(q || m)], m = 0.5 (p + q).
    """
    if kind == "kl":
        return 0.5 * (_kl(p, q) + _kl(q, p))
    if kind == "js":
        m = 0.5 * (p + q)
        return 0.5 * (_kl(p, m) + _kl(q, m))
    raise ValueError(f"unknown divergence kind: {kind!r}")


def ravel_alignment_loss(
    logits_real: torch.Tensor,
    logits_prev: torch.Tensor,
    logits_glob: torch.Tensor,
    y_syn: torch.Tensor,
    temperature: float = T_ALIGN,
    eps_smooth: float = EPS_SMOOTH,
    kind: str = "kl",
) -> AlignmentOutput:
    """L_align over the three distribution pairs (Eq. 17).

    L_align = D_sym(p(x_prev) || p(x_glob))
            + D_sym(p(x_real) || p(x_glob))
            + D_sym(p(x_glob)  || q_onehot(y_syn))

    All logits are [N, C] classifier outputs C_cls(x) of the three streams
    (previous local generator, global generator, real data) sharing the same
    synthetic conditioning labels ``y_syn``.
    """
    p_real = temperature_softmax(logits_real, temperature)
    p_prev = temperature_softmax(logits_prev, temperature)
    p_glob = temperature_softmax(logits_glob, temperature)
    q_syn = label_smoothed_onehot(y_syn, logits_glob.size(-1), eps_smooth)

    d_prev_glob = symmetric_divergence(p_prev, p_glob, kind).mean()
    d_real_glob = symmetric_divergence(p_real, p_glob, kind).mean()
    d_glob_syn = symmetric_divergence(p_glob, q_syn, kind).mean()
    loss = d_prev_glob + d_real_glob + d_glob_syn
    return AlignmentOutput(
        loss=loss,
        terms={"prev_glob": d_prev_glob, "real_glob": d_real_glob, "glob_syn": d_glob_syn},
    )


class SymmetricAlignmentLoss(nn.Module):
    """nn.Module wrapper around :func:`ravel_alignment_loss` (T_align = 2.0)."""

    def __init__(
        self,
        temperature: float = T_ALIGN,
        eps_smooth: float = EPS_SMOOTH,
        kind: str = "kl",
    ) -> None:
        super().__init__()
        self.temperature, self.eps_smooth, self.kind = temperature, eps_smooth, kind

    def forward(
        self,
        logits_real: torch.Tensor,
        logits_prev: torch.Tensor,
        logits_glob: torch.Tensor,
        y_syn: torch.Tensor,
    ) -> AlignmentOutput:
        return ravel_alignment_loss(
            logits_real, logits_prev, logits_glob, y_syn,
            temperature=self.temperature, eps_smooth=self.eps_smooth, kind=self.kind,
        )


# --------------------------------------------------------------------------------------
# Total objectives (Eq. 5)
# --------------------------------------------------------------------------------------
def discriminator_objective(
    adv_loss: torch.Tensor,
    d2d_loss: torch.Tensor,
    align_loss: Optional[torch.Tensor] = None,
    lambda_cls: float = LAMBDA_CLS,
    lambda_align: float = LAMBDA_ALIGN,
) -> torch.Tensor:
    """L_D = L_adv^D + lambda_cls * L_D2D-CE + lambda_align * L_align."""
    loss = adv_loss + lambda_cls * d2d_loss
    if align_loss is not None:
        loss = loss + lambda_align * align_loss
    return loss


def generator_objective(
    adv_loss: torch.Tensor, d2d_loss: torch.Tensor, lambda_cls: float = LAMBDA_CLS
) -> torch.Tensor:
    """L_G = L_adv^G + lambda_cls * L_D2D-CE."""
    return adv_loss + lambda_cls * d2d_loss
