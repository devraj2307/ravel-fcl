"""ReACGAN building blocks for RAVEL-FCL (spec Sections 1.2 and 2.2).

* ``Generator``            -- 5-layer conditional generator G: Z x C_{<=t} -> X with
                              z ~ N(0, I_128) and Conditional Batch Normalization.
* ``Discriminator``        -- ResNet-18 backbone F: X -> R^512 (initial 7x7 stride-2
                              conv + max-pool replaced by a 3x3 stride-1 conv for
                              28x28 / 32x32 inputs), binary adversarial head D_adv,
                              2-layer MLP projector P: R^512 -> R^256 -> R^128 with
                              ReLU, learnable class prototypes W (R^128 x C_t), and
                              linear classification head C_cls: R^512 -> R^{C_t}.
* ``hypersphere_normalize`` -- Eq. 2: f = P(F(x)) / (||P(F(x))||_2 + eps) and
                              v_y = w_y / (||w_y||_2 + eps).
* ``expand_classes``        -- dynamically grows the classifier head, class
                              prototypes and generator class embeddings when new
                              classes emerge in a communication round (FedCIL-style).
* Spectral normalization is applied to conv/linear layers of both networks.
"""

from __future__ import annotations

from typing import Dict, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

HYPERSPHERE_EPS = 1e-8  # eps_eps of Eq. 2
WEIGHT_STD = 0.02       # StudioGAN-style N(0, 0.02) initialization

# BatchNorm statistics are *buffers*, not parameters.  They carry the running
# mean/variance needed for correct eval-mode forward passes, so they must be
# aggregated separately from the (compressed, noised) parameter deltas.
BN_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def is_batch_norm_buffer(name: str) -> bool:
    """True for BatchNorm running statistics (safe to average, never compress)."""
    return name.endswith(BN_BUFFER_SUFFIXES)

def clone_model(module: nn.Module) -> nn.Module:
    """Deep-copy a model that may use legacy (hook-based) spectral norm.

    ``torch.nn.utils.spectral_norm`` caches the normalized weight as a plain,
    non-leaf attribute ``module.weight`` after every forward pass; deep-copying
    such a model raises "Only Tensors created explicitly by the user ... support
    the deepcopy protocol".  Dropping the cached attribute first is safe: the
    forward pre-hook recomputes it on the next call, and it is not part of
    ``state_dict`` (which stores ``weight_orig`` / ``weight_u`` / ``weight_v``).
    """
    for m in module.modules():
        if hasattr(m, "weight_orig") and "weight" in m.__dict__:
            del m.__dict__["weight"]
    return copy.deepcopy(module)


def calibrate_spectral_norm(module: nn.Module, n_iter: int = 10) -> None:
    """Refine the legacy spectral-norm singular-vector estimates at init time.

    ``torch.nn.utils.spectral_norm`` seeds ``weight_u`` / ``weight_v`` randomly
    and only runs power iteration during *training* forward passes.  A freshly
    constructed (or server-side, eval-only) model therefore divides by a poor
    sigma estimate and its effective weights can be tens of times too large,
    which compounds across the ResNet blocks.  Ten power iterations make
    ``||W_SN||_2 ~= 1`` immediately; ``u`` / ``v`` are buffers and stay
    consistent with the weights thereafter.
    """
    try:
        from torch.nn.utils.spectral_norm import SpectralNorm
    except ImportError:  # pragma: no cover - defensive
        return
    was_training = module.training
    module.train()
    with torch.no_grad():
        for m in module.modules():
            for hook in m._forward_pre_hooks.values():
                if isinstance(hook, SpectralNorm):
                    for _ in range(n_iter):
                        hook(m, None)
    module.train(was_training)

def hypersphere_normalize(features: torch.Tensor, eps: float = HYPERSPHERE_EPS) -> torch.Tensor:
    """Project features (or class weights) onto the unit hypersphere (Eq. 2)."""
    return F.normalize(features, p=2, dim=-1, eps=eps)


def _init_weight(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, WEIGHT_STD)
            if getattr(m, "bias", None) is not None and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, WEIGHT_STD)


def _sn_linear(in_features: int, out_features: int, sn: bool) -> nn.Linear:
    lin = nn.Linear(in_features, out_features)
    return spectral_norm(lin) if sn else lin


def _sn_conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, sn: bool = False) -> nn.Conv2d:
    conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
    return spectral_norm(conv) if sn else conv


def _expand_linear_out(module: nn.Linear, new_out: int, std: float = WEIGHT_STD) -> None:
    """Grow a Linear's output dimension in place, preserving old weights.

    Handles plain linears, legacy hook-based spectral norm (``weight_orig`` /
    ``weight_u``) and parametrization-based spectral norm.  New rows of the SN
    ``u`` buffer are zero-padded, so the estimated sigma -- and hence every old
    row of the layer -- is exactly preserved.
    """
    old_out = module.out_features
    if new_out == old_out:
        return
    if new_out < old_out:
        raise ValueError("expand_classes expects new_num_classes >= current")
    in_features = module.in_features
    device, dtype = module.weight.device, module.weight.dtype

    has_legacy_sn = hasattr(module, "weight_orig")           # torch.nn.utils.spectral_norm
    has_param_sn = hasattr(module, "parametrizations") and "weight" in module.parametrizations

    with torch.no_grad():
        new_w = torch.empty(new_out, in_features, device=device, dtype=dtype)
        nn.init.normal_(new_w, 0.0, std)
        new_b = torch.zeros(new_out, device=device, dtype=dtype)
        if has_legacy_sn:
            new_w[:old_out] = module.weight_orig.detach()
            module.weight_orig = nn.Parameter(new_w)
            new_u = torch.zeros(new_out, device=device, dtype=dtype)
            new_u[:old_out] = module.weight_u.detach()
            module.weight_u = new_u
        elif has_param_sn:
            plist = module.parametrizations.weight
            new_w[:old_out] = plist.original.detach()
            plist.original = nn.Parameter(new_w)
            sn_mod = plist[0]
            new_u = torch.zeros(new_out, device=device, dtype=dtype)
            new_u[:old_out] = sn_mod.u.detach()
            sn_mod.u = new_u
        else:
            new_w[:old_out] = module.weight.detach()
            module.weight = nn.Parameter(new_w)
        if module.bias is not None:
            new_b[:old_out] = module.bias.detach()
            module.bias = nn.Parameter(new_b)
    module.out_features = new_out


def _expand_rows(param: nn.Parameter, new_rows: int, std: float = WEIGHT_STD) -> nn.Parameter:
    """Grow a [rows, cols] parameter (class prototypes / embeddings) in place."""
    old = param.detach()
    if new_rows < old.shape[0]:
        raise ValueError("expand_classes expects new_num_classes >= current")
    new = torch.empty(new_rows, *old.shape[1:], device=old.device, dtype=old.dtype)
    nn.init.normal_(new, 0.0, std)
    new[: old.shape[0]] = old
    return nn.Parameter(new)


# --------------------------------------------------------------------------------------
# Conditional Batch Normalization (spectral-norm-free, class-conditional affine params)
# --------------------------------------------------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    """cBN: class-conditional gain/beta embeddings modulating a BatchNorm2d (StudioGAN style).

    gamma, beta come from per-class embeddings, so ``expand_classes`` grows them
    together with the rest of the class-conditional parameters.
    """

    def __init__(self, num_classes: int, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False)
        self.gain = nn.Embedding(num_classes, num_features)  # gamma
        self.bias = nn.Embedding(num_classes, num_features)  # beta
        nn.init.zeros_(self.gain.weight)  # start as plain BN: (1 + 0) * bn(x) + 0
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1:
            y = y.view(-1)
        gamma = self.gain(y).unsqueeze(-1).unsqueeze(-1)
        beta = self.bias(y).unsqueeze(-1).unsqueeze(-1)
        return self.bn(x) * (1.0 + gamma) + beta

    def expand_classes(self, new_num_classes: int) -> None:
        self.gain.weight = _expand_rows(self.gain.weight, new_num_classes, std=0.0)
        self.bias.weight = _expand_rows(self.bias.weight, new_num_classes, std=0.0)
        self.num_classes = new_num_classes


# --------------------------------------------------------------------------------------
# Generator: 5 layers (1 linear + 3 cBN blocks + final conv), z in R^128
# --------------------------------------------------------------------------------------
class GenBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, up_size: int, num_classes: int, sn: bool):
        super().__init__()
        self.up_size = up_size
        self.conv = _sn_conv(in_ch, out_ch, 3, 1, sn=sn)
        self.cbn = ConditionalBatchNorm2d(num_classes, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.up_size, mode="nearest")
        return self.act(self.cbn(self.conv(x), y))

    def expand_classes(self, new_num_classes: int) -> None:
        self.cbn.expand_classes(new_num_classes)


class Generator(nn.Module):
    """Conditional generator G_t^k: Z x C_{<=t} -> X (spec Section 2.2).

    28x28: 4x4 -> 7x7 -> 14x14 -> 28x28;  32x32: 4x4 -> 8x8 -> 16x16 -> 32x32.
    """

    def __init__(
        self,
        num_classes: int,
        z_dim: int = 128,
        img_size: int = 28,
        img_channels: int = 1,
        base_channels: int = 256,
        spectral_norm_gen: bool = True,
    ) -> None:
        super().__init__()
        if img_size not in (28, 32):
            raise ValueError("img_size must be 28 or 32")
        self.num_classes = int(num_classes)
        self.z_dim = int(z_dim)
        self.img_size = int(img_size)
        self.img_channels = int(img_channels)

        chans = (base_channels, 128, 64, 32)
        ups = (7, 14, 28) if img_size == 28 else (8, 16, 32)

        self.fc = _sn_linear(z_dim, 4 * 4 * chans[0], spectral_norm_gen)
        self.blocks = nn.ModuleList(
            [
                GenBlock(chans[i], chans[i + 1], ups[i], num_classes, spectral_norm_gen)
                for i in range(len(ups))
            ]
        )
        self.final = _sn_conv(chans[-1], img_channels, 3, 1, sn=spectral_norm_gen)
        self.out_act = nn.Tanh()
        self.apply(_init_weight)
        for blk in self.blocks:  # cBN starts as plain BN: (1 + 0) * bn(x) + 0
            nn.init.zeros_(blk.cbn.gain.weight)
            nn.init.zeros_(blk.cbn.bias.weight)
        calibrate_spectral_norm(self)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dim() > 1:
            y = y.view(-1)
        if int(y.max()) >= self.num_classes or int(y.min()) < 0:
            raise IndexError(
                f"label {int(y.max())} out of range for generator with {self.num_classes} classes"
            )
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        for blk in self.blocks:
            x = blk(x, y)
        return self.out_act(self.final(x))

    def expand_classes(self, new_num_classes: int) -> None:
        """Grow every class-conditional generator embedding to ``new_num_classes``."""
        new_num_classes = int(new_num_classes)
        if new_num_classes < self.num_classes:
            raise ValueError("expand_classes expects new_num_classes >= current")
        for blk in self.blocks:
            blk.expand_classes(new_num_classes)
        self.num_classes = new_num_classes


# --------------------------------------------------------------------------------------
# Discriminator / classifier with ResNet-18 backbone
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, sn: bool = False):
        super().__init__()
        self.conv1 = _sn_conv(in_ch, out_ch, 3, stride, sn=sn)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _sn_conv(out_ch, out_ch, 3, 1, sn=sn)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))
                if sn
                else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNet18Backbone(nn.Module):
    """F: X -> R^512.  Initial 7x7 stride-2 conv + max-pool replaced by 3x3 stride-1
    conv (no pooling) so 28x28 / 32x32 inputs work unchanged (spec Section 2.2)."""

    LAYERS = (2, 2, 2, 2)

    def __init__(self, img_channels: int = 1, sn: bool = True):
        super().__init__()
        self.in_ch = 64
        self.conv1 = _sn_conv(img_channels, 64, 3, 1, sn=sn)  # adapted stem
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(inplace=True)
        self.layer1 = self._make_stage(64, self.LAYERS[0], stride=1, sn=sn)
        self.layer2 = self._make_stage(128, self.LAYERS[1], stride=2, sn=sn)
        self.layer3 = self._make_stage(256, self.LAYERS[2], stride=2, sn=sn)
        self.layer4 = self._make_stage(512, self.LAYERS[3], stride=2, sn=sn)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

    def _make_stage(self, out_ch: int, blocks: int, stride: int, sn: bool) -> nn.Sequential:
        stages = [BasicBlock(self.in_ch, out_ch, stride, sn=sn)]
        self.in_ch = out_ch
        for _ in range(blocks - 1):
            stages.append(BasicBlock(out_ch, out_ch, 1, sn=sn))
        return nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.gap(x).flatten(1)  # [N, 512]


class Discriminator(nn.Module):
    """Shared-backbone discriminator/classifier theta_k (spec Section 2.2).

    forward() returns a dict:
        features    [N, 512]   F(x)
        adv_logit   [N]        pre-sigmoid D_adv(F(x)); ``adv_prob`` in (0, 1)
        proj_raw    [N, 128]   P(F(x)) before normalization
        proj        [N, 128]   hypersphere-normalized f_i (Eq. 2)
        logits      [N, C_t]   unnormalized C_cls logits z_logits(x)
    """

    def __init__(
        self,
        num_classes: int,
        img_channels: int = 1,
        feature_dim: int = 512,
        proj_hidden: int = 256,
        proj_dim: int = 128,
        spectral_norm_d: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.proj_dim = int(proj_dim)
        self.backbone = ResNet18Backbone(img_channels=img_channels, sn=spectral_norm_d)
        self.adv_head = _sn_linear(feature_dim, 1, spectral_norm_d)
        # 2-layer MLP projector P: R^512 -> R^256 -> R^128 with ReLU
        self.projector = nn.Sequential(
            _sn_linear(feature_dim, proj_hidden, spectral_norm_d),
            nn.ReLU(inplace=True),
            _sn_linear(proj_hidden, proj_dim, spectral_norm_d),
        )
        self.classifier = _sn_linear(feature_dim, self.num_classes, spectral_norm_d)
        # Learnable class prototype embeddings W = [w_1, ..., w_Ct] in R^{128 x C_t}
        self.prototypes = nn.Parameter(torch.empty(self.num_classes, proj_dim))
        nn.init.normal_(self.prototypes, 0.0, WEIGHT_STD)
        self.apply(_init_weight)
        for m in self.backbone.modules():  # keep BN statistics sane
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        calibrate_spectral_norm(self)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def normalized_prototypes(self) -> torch.Tensor:
        """v_y = w_y / (||w_y||_2 + eps), shape [C_t, 128] (Eq. 2)."""
        return hypersphere_normalize(self.prototypes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        adv_logit = self.adv_head(f).squeeze(-1)
        proj_raw = self.projector(f)
        # Use cosine-similarity logits bounded to [-10, 10]
        f_norm = F.normalize(f, p=2, dim=-1)
        if hasattr(self.classifier, "weight"):
            w = self.classifier.weight
        elif hasattr(self.classifier, "weight_orig"):
            w = self.classifier.weight_orig
        else:
            w = self.classifier.parametrizations.weight.original
        v_norm = F.normalize(w, p=2, dim=-1)
        logits = torch.matmul(f_norm, v_norm.t()) / 0.1
        if getattr(self.classifier, "bias", None) is not None:
            logits = logits + self.classifier.bias
        return {
            "features": f,
            "adv_logit": adv_logit,
            "adv_prob": torch.sigmoid(adv_logit),
            "proj_raw": proj_raw,
            "proj": hypersphere_normalize(proj_raw),
            "logits": logits,
        }

    def expand_classes(self, new_num_classes: int) -> None:
        """Grow classifier head C_cls and prototypes W when new classes appear."""
        new_num_classes = int(new_num_classes)
        if new_num_classes < self.num_classes:
            raise ValueError("expand_classes expects new_num_classes >= current")
        if new_num_classes > self.num_classes:
            _expand_linear_out(self.classifier, new_num_classes)
            self.prototypes = _expand_rows(self.prototypes, new_num_classes)
        self.num_classes = new_num_classes
