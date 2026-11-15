"""Evaluation metrics for RAVEL-FCL (spec Sections 3, 5.7 and 5.10.1).

The suite tracks the four diagnostics used throughout the stress-testing matrix
(Table 1) and the adversarial evaluation:

* **Test accuracy** -- overall, old-class and new-class top-1 accuracy on the
  real test split, plus a per-class breakdown.
* **Average forgetting rate** -- the standard continual-learning definition
  over the task-accuracy matrix ``a[t, u]`` (accuracy on task ``u`` after
  finishing task ``t``)::

      AF = 1/(T-1) * sum_{u=1}^{T-1} ( max_{t<u} a[t, u] - a[T-1, u] )

* **Feature-space MMD** -- unbiased multi-bandwidth RBF-kernel MMD^2 between
  real and synthetic class-conditional samples embedded by the discriminator's
  hypersphere projector (the same kernel used by Extension C, Eq. 14).
* **Class centroid drift** -- mean L2 displacement of the per-class feature
  centroids, both round-over-round and relative to the first observation of
  the class.

All functions operate on CPU/CUDA ``torch.Tensor`` objects and never require
NumPy.  Evaluation is batched and optionally sub-sampled so that it can run
after every communication round on a laptop CPU.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from federated.defense import SIGMAS, compute_multikernel_mmd

Number = Union[int, float]


# --------------------------------------------------------------------------------------
# Classifier evaluation
# --------------------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    """Result of one classifier evaluation pass."""

    accuracy: float
    loss: float
    per_class_accuracy: Dict[int, float]
    per_class_count: Dict[int, int]
    num_samples: int

    def accuracy_on(self, classes: Iterable[int]) -> Optional[float]:
        """Class-frequency-weighted accuracy over ``classes`` (None if empty)."""
        correct = 0.0
        total = 0.0
        for c in classes:
            c = int(c)
            if c not in self.per_class_count:
                continue
            correct += self.per_class_accuracy[c] * self.per_class_count[c]
            total += self.per_class_count[c]
        return correct / total if total > 0 else None


def _select_indices(labels: torch.Tensor, classes: Optional[Iterable[int]],
                    max_samples: Optional[int]) -> torch.Tensor:
    """Indices of ``labels`` belonging to ``classes`` (optionally capped)."""
    if classes is None:
        idx = torch.arange(labels.numel())
    else:
        keep = torch.zeros(labels.numel(), dtype=torch.bool)
        for c in classes:
            keep |= labels == int(c)
        idx = keep.nonzero(as_tuple=False).flatten()
    if max_samples is not None and idx.numel() > int(max_samples):
        # Deterministic stride keeps every class represented on average.
        step = max(1, idx.numel() // int(max_samples))
        idx = idx[::step][: int(max_samples)]
    return idx


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    classes: Optional[Iterable[int]] = None,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    attack_src: Optional[int] = None,
    attack_tgt: Optional[int] = None,
) -> ClassificationMetrics:
    """Top-1 accuracy / cross-entropy of ``model`` on ``(x, y)``.

    ``classes`` restricts the evaluation to a class subset (e.g. the classes of
    one task).  ``max_samples`` sub-samples the subset to bound evaluation cost.
    """
    idx = _select_indices(y, classes, max_samples)
    was_training = model.training
    model.eval()
    correct: Dict[int, int] = {}
    total: Dict[int, int] = {}
    total_loss = 0.0
    total_n = 0

    for start in range(0, idx.numel(), int(batch_size)):
        sel = idx[start:start + int(batch_size)]
        xb = x[sel].to(device)
        yb = y[sel].to(device)
        logits = model(xb)["logits"]
        if logits.size(-1) <= int(yb.max()):
            # The head has not grown to this label yet; count it as wrong.
            pred = torch.full_like(yb, -1)
            loss = F.cross_entropy(logits, yb.clamp_max(logits.size(-1) - 1))
        else:
            loss = F.cross_entropy(logits, yb)
            pred = logits.argmax(dim=-1)
        total_loss += float(loss.detach()) * yb.numel()
        total_n += yb.numel()
        for c in yb.unique().tolist():
            mask = yb == int(c)
            total[int(c)] = total.get(int(c), 0) + int(mask.sum())
            correct[int(c)] = correct.get(int(c), 0) + int((pred[mask] == int(c)).sum())

    if was_training:
        model.train()
    per_class_accuracy = {
        c: (correct.get(c, 0) / total[c] if total.get(c, 0) else 0.0)
        for c in sorted(total)
    }
    n = max(1, total_n)
    return ClassificationMetrics(
        accuracy=sum(correct.values()) / n,
        loss=total_loss / n,
        per_class_accuracy=per_class_accuracy,
        per_class_count=dict(sorted(total.items())),
        num_samples=total_n,
    )


@torch.no_grad()
def targeted_attack_success_rate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    source_class: int,
    target_class: int,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> Optional[float]:
    """Fraction of true ``source_class`` test samples predicted as ``target_class``."""
    idx = (y == int(source_class)).nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    was_training = model.training
    model.eval()
    hits = 0
    for start in range(0, idx.numel(), int(batch_size)):
        sel = idx[start:start + int(batch_size)]
        logits = model(x[sel].to(device))["logits"]
        if logits.size(-1) <= int(target_class):
            continue
        hits += int((logits.argmax(dim=-1) == int(target_class)).sum())
    if was_training:
        model.train()
    return hits / idx.numel()


# --------------------------------------------------------------------------------------
# Feature-space metrics (Section 5.10.1)
# --------------------------------------------------------------------------------------
@dataclass
class FeatureMetricResult:
    mmd: Optional[float] = None
    per_class_mmd: Dict[int, float] = field(default_factory=dict)
    centroid_drift: Optional[float] = None
    centroid_drift_anchor: Optional[float] = None
    per_class_centroid_drift: Dict[int, float] = field(default_factory=dict)
    per_class_centroid_drift_anchor: Dict[int, float] = field(default_factory=dict)


@torch.no_grad()
def projected_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Hypersphere-normalized projector features ``P(F(x))`` (Eq. 2)."""
    outs: List[torch.Tensor] = []
    was_training = model.training
    model.eval()
    for start in range(0, x.size(0), int(batch_size)):
        outs.append(model(x[start:start + int(batch_size)].to(device))["proj"].detach().cpu())
    if was_training:
        model.train()
    return torch.cat(outs, dim=0) if outs else torch.empty(0)


@torch.no_grad()
def class_feature_centroids(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    classes: Iterable[int],
    samples_per_class: int = 128,
    batch_size: int = 256,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """Mean projected feature per class, shape {c: [proj_dim]}."""
    was_training = model.training
    model.eval()
    centroids: Dict[int, torch.Tensor] = {}
    for c in sorted({int(c) for c in classes}):
        idx = (y == c).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        if samples_per_class is not None and idx.numel() > int(samples_per_class):
            if generator is None:
                pick = torch.randperm(idx.numel())[: int(samples_per_class)]
            else:
                pick = torch.randperm(idx.numel(), generator=generator)[: int(samples_per_class)]
            idx = idx[pick]
        feats = projected_features(model, x[idx], batch_size=batch_size, device=device)
        if not torch.isfinite(feats).all():
            continue
        centroids[c] = feats.mean(dim=0)
    if was_training:
        model.train()
    return centroids


@torch.no_grad()
def feature_space_mmd(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    x_real: torch.Tensor,
    y_real: torch.Tensor,
    classes: Iterable[int],
    samples_per_class: int = 64,
    batch_size: int = 128,
    device: Union[str, torch.device] = "cpu",
    sigmas: Sequence[float] = SIGMAS,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Optional[float], Dict[int, float]]:
    """Real-vs-synthetic multi-kernel MMD^2 per class and averaged.

    For every class the real test samples are embedded with the current
    discriminator and compared against ``G(z, c)`` samples embedded by the same
    network.  The kernel is the unbiased RBF sum of Eq. 14.
    """
    rng = rng or torch.Generator().manual_seed(0)
    was_training_model = model.training
    was_training_gen = generator.training
    model.eval()
    generator.eval()
    z_dim = int(getattr(generator, "z_dim", 128))
    per_class: Dict[int, float] = {}
    for c in sorted({int(c) for c in classes}):
        idx = (y_real == c).nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        if idx.numel() > int(samples_per_class):
            pick = torch.randperm(idx.numel(), generator=rng)[: int(samples_per_class)]
            idx = idx[pick]
        n = int(idx.numel())
        real = projected_features(model, x_real[idx], batch_size=batch_size, device=device)
        z = torch.randn(n, z_dim, generator=rng).to(device)
        y = torch.full((n,), c, dtype=torch.long, device=device)
        fake_x = generator(z, y).detach().cpu()
        fake = projected_features(model, fake_x, batch_size=batch_size, device=device)
        if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
            continue
        value = float(compute_multikernel_mmd(
            real.to(device), fake.to(device), tuple(float(s) for s in sigmas)
        ).detach().cpu())
        if math.isfinite(value):
            per_class[c] = value
    if was_training_model:
        model.train()
    if was_training_gen:
        generator.train()
    mean = sum(per_class.values()) / len(per_class) if per_class else None
    return mean, per_class


@torch.no_grad()
def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 16,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Sliced Wasserstein-1 distance between two feature clouds.

    Exact W1 in a 128-D feature space is expensive; we use the standard sliced
    approximation: project both clouds onto ``n_projections`` random unit
    directions, sort each 1-D projection, and average the absolute quantile
    differences.  This is the same diagnostic family as the generator FID/IS
    probes in Section 3.4 and is cheap enough for per-round logging.
    """
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    n = min(x.size(0), y.size(0))
    if n < 1:
        return 0.0
    if x.size(0) != n:
        x = x[torch.randperm(x.size(0), generator=generator)[:n]]
    if y.size(0) != n:
        y = y[torch.randperm(y.size(0), generator=generator)[:n]]
    g = generator or torch.Generator().manual_seed(0)
    directions = torch.randn(x.size(1), int(n_projections), generator=g)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    px = (x @ directions).sort(dim=0).values
    py = (y @ directions).sort(dim=0).values
    return float((px - py).abs().mean())


@torch.no_grad()
def feature_space_wasserstein(
    model: torch.nn.Module,
    generator: torch.nn.Module,
    x_real: torch.Tensor,
    y_real: torch.Tensor,
    classes: Iterable[int],
    samples_per_class: int = 64,
    batch_size: int = 128,
    device: Union[str, torch.device] = "cpu",
    n_projections: int = 16,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Optional[float], Dict[int, float]]:
    """Sliced W1 between real and synthetic projected features per class."""
    rng = rng or torch.Generator().manual_seed(0)
    was_training_model = model.training
    was_training_gen = generator.training
    model.eval()
    generator.eval()
    z_dim = int(getattr(generator, "z_dim", 128))
    per_class: Dict[int, float] = {}
    for c in sorted({int(c) for c in classes}):
        idx = (y_real == c).nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        if idx.numel() > int(samples_per_class):
            pick = torch.randperm(idx.numel(), generator=rng)[: int(samples_per_class)]
            idx = idx[pick]
        n = int(idx.numel())
        real = projected_features(model, x_real[idx], batch_size=batch_size, device=device)
        z = torch.randn(n, z_dim, generator=rng).to(device)
        y = torch.full((n,), c, dtype=torch.long, device=device)
        fake_x = generator(z, y).detach().cpu()
        fake = projected_features(model, fake_x, batch_size=batch_size, device=device)
        if not torch.isfinite(real).all() or not torch.isfinite(fake).all():
            continue
        value = sliced_wasserstein_distance(
            real, fake, n_projections=n_projections, generator=rng
        )
        if math.isfinite(value):
            per_class[c] = value
    if was_training_model:
        model.train()
    if was_training_gen:
        generator.train()
    mean = sum(per_class.values()) / len(per_class) if per_class else None
    return mean, per_class


class CentroidDriftTracker:
    """Tracks per-class feature centroids across rounds / tasks.

    ``drift`` is the mean L2 displacement relative to the previous observation
    (a round-to-round stability signal); ``drift_anchor`` is measured against
    the first centroid ever observed for the class (cumulative drift).
    """

    def __init__(self) -> None:
        self.previous: Dict[int, torch.Tensor] = {}
        self.anchor: Dict[int, torch.Tensor] = {}

    def update(self, centroids: Mapping[int, torch.Tensor]) -> FeatureMetricResult:
        round_d: Dict[int, float] = {}
        anchor_d: Dict[int, float] = {}
        for c, mu in centroids.items():
            mu = mu.detach().float().cpu()
            if c in self.previous:
                round_d[c] = float(torch.linalg.vector_norm(mu - self.previous[c]))
            if c in self.anchor:
                anchor_d[c] = float(torch.linalg.vector_norm(mu - self.anchor[c]))
            else:
                self.anchor[c] = mu.clone()
            self.previous[c] = mu.clone()
        return FeatureMetricResult(
            centroid_drift=(sum(round_d.values()) / len(round_d)) if round_d else None,
            centroid_drift_anchor=(sum(anchor_d.values()) / len(anchor_d)) if anchor_d else None,
            per_class_centroid_drift=round_d,
            per_class_centroid_drift_anchor=anchor_d,
        )


# --------------------------------------------------------------------------------------
# Task matrix, average forgetting rate, and experiment logging
# --------------------------------------------------------------------------------------
class MetricsTracker:
    """Accumulates per-round metrics and the task-accuracy matrix.

    Args:
        task_classes: non-overlapping class list introduced by each task
            (``C_t \\ C_{<=t-1}``), used to index the task-accuracy matrix.
        num_tasks: number of sequential tasks T.
    """

    def __init__(self, task_classes: Sequence[Sequence[int]], num_tasks: Optional[int] = None):
        self.task_classes: List[Tuple[int, ...]] = [tuple(int(c) for c in cs) for cs in task_classes]
        self.num_tasks = int(num_tasks if num_tasks is not None else len(self.task_classes))
        self.rounds: List[Dict[str, object]] = []
        # a[t][u] = accuracy on task u after finishing task t (NaN if not evaluated)
        self.task_matrix: List[List[Optional[float]]] = [
            [None] * self.num_tasks for _ in range(self.num_tasks)
        ]
        self.last_recorded_task: int = -1

    # ------------------------------------------------------------------
    # Per-round logging
    # ------------------------------------------------------------------
    def log_round(self, record: Mapping[str, object]) -> Dict[str, object]:
        row = {str(k): _to_builtin(v) for k, v in record.items()}
        row.setdefault("round", len(self.rounds))
        self.rounds.append(row)
        return row

    # ------------------------------------------------------------------
    # Task matrix / forgetting
    # ------------------------------------------------------------------
    def task_accuracy_from_per_class(
        self, task_idx: int, per_class_accuracy: Mapping[int, float],
        per_class_count: Optional[Mapping[int, int]] = None,
    ) -> Optional[float]:
        classes = self.task_classes[task_idx] if task_idx < len(self.task_classes) else ()
        correct = 0.0
        total = 0.0
        for c in classes:
            if c not in per_class_accuracy:
                continue
            n = float(per_class_count[c]) if per_class_count and c in per_class_count else 1.0
            correct += float(per_class_accuracy[c]) * n
            total += n
        return correct / total if total > 0 else None

    def record_task_eval(
        self,
        task_idx: int,
        per_class_accuracy: Mapping[int, float],
        per_class_count: Optional[Mapping[int, int]] = None,
    ) -> None:
        """Record a snapshot after finishing ``task_idx`` for every task <= it."""
        task_idx = int(task_idx)
        if task_idx < 0:
            return
        for u in range(min(task_idx + 1, self.num_tasks)):
            self.task_matrix[task_idx][u] = self.task_accuracy_from_per_class(
                u, per_class_accuracy, per_class_count
            )
        self.last_recorded_task = max(self.last_recorded_task, task_idx)

    def average_forgetting(
        self, final_task: Optional[int] = None, include_reference: Optional[bool] = None
    ) -> Optional[float]:
        """Average forgetting rate (Section 5.7).

        The standard definition excludes the first task (the reference task):
        for each task u in 1..T-2, the best accuracy obtained on u at any
        earlier snapshot is compared with the final accuracy on u.  Set
        ``include_reference=True`` to also average the reference task's
        forgetting, which is useful for short runs with only two tasks.  When
        unspecified, reference inclusion is enabled automatically for a
        two-task run (so the metric is always defined at the end of Task 1).
        """
        last = self.last_recorded_task if final_task is None else int(final_task)
        if last <= 0:
            return None
        if include_reference is None:
            include_reference = last <= 1
        start = 0 if include_reference else 1
        forgets: List[float] = []
        for u in range(start, last):
            final = self.task_matrix[last][u]
            if final is None:
                continue
            earlier = [
                self.task_matrix[t][u] for t in range(u, last)
                if self.task_matrix[t][u] is not None
            ]
            if not earlier:
                continue
            forgets.append(max(earlier) - final)
        return sum(forgets) / len(forgets) if forgets else None

    def per_task_forgetting(
        self, final_task: Optional[int] = None, include_reference: Optional[bool] = None
    ) -> Dict[int, float]:
        last = self.last_recorded_task if final_task is None else int(final_task)
        out: Dict[int, float] = {}
        if last <= 0:
            return out
        if include_reference is None:
            include_reference = last <= 1
        start = 0 if include_reference else 1
        for u in range(start, last):
            final = self.task_matrix[last][u]
            if final is None:
                continue
            earlier = [
                self.task_matrix[t][u] for t in range(u, last)
                if self.task_matrix[t][u] is not None
            ]
            if earlier:
                out[u] = max(earlier) - final
        return out

    # ------------------------------------------------------------------
    # Summary / export
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        last = self.last_recorded_task
        final_accs = self.task_matrix[last] if last >= 0 else []
        finite = [a for a in final_accs if a is not None]
        return {
            "num_rounds_logged": len(self.rounds),
            "last_recorded_task": last,
            "task_classes": {t: list(cs) for t, cs in enumerate(self.task_classes)},
            "task_accuracy_matrix": self.task_matrix,
            "final_task_accuracies": {
                str(t): final_accs[t] for t in range(len(final_accs))
            },
            "final_average_accuracy": (sum(finite) / len(finite)) if finite else None,
            "average_forgetting_rate": self.average_forgetting(),
            "per_task_forgetting": self.per_task_forgetting(),
            # The reference task is normally excluded; this companion value
            # makes short two-task runs interpretable as well.
            "average_forgetting_rate_including_reference": self.average_forgetting(
                include_reference=True
            ),
            "per_task_forgetting_including_reference": self.per_task_forgetting(
                include_reference=True
            ),
            "last_round": self.rounds[-1] if self.rounds else None,
        }


# --------------------------------------------------------------------------------------
# JSON / CSV export (no NumPy / pandas dependency)
# --------------------------------------------------------------------------------------
def _to_builtin(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def save_json(payload: Mapping[str, object], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, indent=2, sort_keys=True)
    return path


def save_csv(rows: Sequence[Mapping[str, object]], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                fields.append(key)
    # Put the round index first for readability.
    if "round" in fields:
        fields.remove("round")
        fields.insert(0, "round")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(row.get(k)) for k in fields})
    return path


def _csv_cell(value: object) -> object:
    value = _to_builtin(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value


def export_metrics(
    tracker: MetricsTracker,
    output_dir: Union[str, Path],
    config: Optional[Mapping[str, object]] = None,
    summary_extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, Path]:
    """Write ``metrics.json`` (full record) and ``metrics.csv`` (per round)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = tracker.summary()
    if summary_extra:
        summary.update(_to_builtin(summary_extra))
    payload = {
        "config": _to_builtin(config or {}),
        "summary": summary,
        "rounds": tracker.rounds,
    }
    return {
        "json": save_json(payload, out / "metrics.json"),
        "csv": save_csv(tracker.rounds, out / "metrics.csv"),
        "summary": save_json(summary, out / "summary.json"),
    }


def old_new_split(
    seen_before: Iterable[int], seen_now: Iterable[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Partition the currently evaluated classes into old and newly introduced."""
    before = {int(c) for c in seen_before}
    now = {int(c) for c in seen_now}
    old = tuple(sorted(now & before))
    new = tuple(sorted(now - before))
    return old, new
