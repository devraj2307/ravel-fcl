#!/usr/bin/env python3
"""RAVEL-FCL command-line entry point.

Examples
--------
Fast smoke test (2 tasks x 2 rounds, MNIST, no privacy noise)::

    python main.py --dataset mnist --num_clients 2 --num_tasks 2 \
        --rounds_per_task 2 --local_epochs 1 --batches_per_round 1 \
        --batch_size 8 --eval_every 1 --eval_max_samples 256 \
        --base_channels 64 --num_fim_batches 2 --evcl_steps 1 \
        --dp_epsilon inf --run_name smoke_mnist

Baseline (spec defaults; expensive)::

    python main.py --dataset emnist --num_clients 5 \
        --num_tasks 5 --rounds_per_task 40 --local_epochs 5

Stress axis 1 (recurrence)::

    python main.py --dataset cifar10 --recurrence_bias 3.5
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

# Allow running both as ``python main.py`` and ``python -m ravel_fcl.main``.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.poisoning import LabelFlipAttack, make_attack  # noqa: E402
from data.datasets import DatasetBundle, ClassBatchSampler, load_dataset  # noqa: E402
from data.streams import RecurringClassStream  # noqa: E402
from federated.client import FederatedClient  # noqa: E402
from federated.compression import zcdp_sigma  # noqa: E402
from federated.defense import ClassAwareMMDDefense  # noqa: E402
from federated.server import FederatedServer, ServerEVCLConsolidator  # noqa: E402
from models.reacgan import Discriminator, Generator, clone_model  # noqa: E402
from utils.metrics import (  # noqa: E402
    CentroidDriftTracker,
    MetricsTracker,
    class_feature_centroids,
    evaluate_classifier,
    export_metrics,
    feature_space_mmd,
    feature_space_wasserstein,
    old_new_split,
    targeted_attack_success_rate,
)
from utils.rng import resolve_device  # noqa: E402

LOGGER = logging.getLogger("ravel_fcl")


def project_path(value: Union[str, Path]) -> Path:
    """Resolve ``value`` against the repository root unless it is absolute."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="RAVEL-FCL: non-monotonic federated class-incremental learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- experiment -------------------------------------------------------------
    exp = p.add_argument_group("experiment")
    exp.add_argument("--dataset", default="mnist",
                     choices=["mnist", "emnist_balanced", "emnist", "emnist-balanced",
                              "cifar10", "cifar-10", "cifar", "synthetic"],
                     help="dataset name (aliases accepted)")
    exp.add_argument("--data_root", default=str(ROOT / "data" / "raw"),
                     help="download/cache directory (default: <repo>/data/raw)")
    exp.add_argument("--allow_download", action=argparse.BooleanOptionalAction, default=True,
                     help="download public datasets when not cached")
    exp.add_argument("--synthetic_data", action="store_true",
                     help="skip dataset I/O and use deterministic synthetic images")
    exp.add_argument("--synthetic_samples_per_class", type=int, default=200)
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--device", default="auto",
                     help="torch device: auto, cpu, cuda, cuda:1, ...")
    exp.add_argument("--output_dir", default=str(ROOT / "results"),
                     help="results directory (default: <repo>/results)")
    exp.add_argument("--run_name", default=None)
    exp.add_argument("--log_level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    exp.add_argument("--eval_every", type=int, default=5,
                      help="evaluate every N rounds (0 = only at task boundaries)")
    exp.add_argument("--eval_max_samples", type=int, default=2000,
                     help="maximum real test samples per evaluation")
    exp.add_argument("--eval_batch_size", type=int, default=256)
    exp.add_argument("--eval_feature_samples", type=int, default=64,
                     help="samples per class for MMD / centroid metrics")
    exp.add_argument("--metrics_mmd", action=argparse.BooleanOptionalAction, default=True)
    exp.add_argument("--metrics_centroid", action=argparse.BooleanOptionalAction, default=True)
    exp.add_argument("--metrics_wasserstein", action=argparse.BooleanOptionalAction, default=True)
    exp.add_argument("--wasserstein_projections", type=int, default=16,
                     help="random 1-D projections for sliced W1")
    exp.add_argument("--save_every", type=int, default=0,
                     help="checkpoint metrics every N rounds (0 = only at the end)")

    # -- federated --------------------------------------------------------------
    fed = p.add_argument_group("federated")
    fed.add_argument("--num_clients", type=int, default=5, help="K")
    fed.add_argument("--num_tasks", type=int, default=5, help="T")
    fed.add_argument("--total_rounds", type=int, default=None,
                     help="R (default: num_tasks * rounds_per_task)")
    fed.add_argument("--rounds_per_task", type=int, default=10, help="R_t")
    fed.add_argument("--local_epochs", type=int, default=1, help="E")
    fed.add_argument("--participation_rate", type=float, default=1.0,
                      help="C_frac (Stress Axis 2)")
    fed.add_argument("--batch_size", type=int, default=64)
    fed.add_argument("--batches_per_round", type=int, default=40,
                      help="total mini-batches per client per round")
    fed.add_argument("--max_train_samples", type=int, default=0,
                     help="cap per-client training samples (0 = no cap)")
    fed.add_argument("--lr_disc", type=float, default=2e-4)
    fed.add_argument("--lr_gen", type=float, default=2e-4)

    # -- stream (Stress Axis 1 / 3) --------------------------------------------
    st = p.add_argument_group("stream")
    st.add_argument("--recurrence_bias", "--omega", dest="recurrence_bias", type=float, default=1.0,
                    help="omega >= 0 (Stress Axis 1)")
    st.add_argument("--classes_per_task", type=int, nargs="+", default=[2])
    st.add_argument("--stream_seed", type=int, default=None,
                    help="stream RNG seed (default: --seed)")
    st.add_argument("--async_offset_max", type=int, default=0,
                    help="Delta R_sync: per-client task offsets spread over [0, value]")
    st.add_argument("--transition_offsets", type=int, nargs="+", default=None,
                    help="explicit per-client transition offsets (overrides --async_offset_max)")
    st.add_argument("--random_task_lengths", action="store_true",
                    help="draw per-client R_t^(k) from --task_length_pool (Stress Axis 3)")
    st.add_argument("--task_length_pool", type=int, nargs="+", default=[25, 35, 40, 45, 55])

    # -- model ------------------------------------------------------------------
    mod = p.add_argument_group("model")
    mod.add_argument("--z_dim", type=int, default=128)
    mod.add_argument("--img_size", type=int, default=32, choices=[28, 32])
    mod.add_argument("--base_channels", type=int, default=256,
                     help="generator base width (use 64 for CPU smoke tests)")
    mod.add_argument("--feature_dim", type=int, default=512)
    mod.add_argument("--proj_hidden", type=int, default=256)
    mod.add_argument("--proj_dim", type=int, default=128)
    mod.add_argument("--no_spectral_norm", action="store_true")

    # -- losses / replay --------------------------------------------------------
    loss = p.add_argument_group("loss")
    loss.add_argument("--lambda_cls", type=float, default=0.5)
    loss.add_argument("--lambda_align", type=float, default=1.0)
    loss.add_argument("--lambda_ce", type=float, default=1.0,
                      help="direct supervised CE weight on real data (classifier head)")
    loss.add_argument("--eta_dp", type=float, default=0.01)

    # -- compression / privacy (Stress Axis 4) ---------------------------------
    priv = p.add_argument_group("privacy")
    priv.add_argument("--keep_ratio", type=float, default=0.8,
                      help="top-k sparsification keep ratio kappa")
    priv.add_argument("--dp_epsilon", type=float, default=1.8,
                      help="zCDP epsilon; use inf for the non-private baseline")
    priv.add_argument("--delta_zcdp", type=float, default=1e-5)
    priv.add_argument("--l2_clip", type=float, default=10.0,
                      help="C_clip; ignored when --dp_epsilon is inf")

    # -- EVCL (Stage 2) ---------------------------------------------------------
    ev = p.add_argument_group("EVCL")
    ev.add_argument("--evcl", action=argparse.BooleanOptionalAction, default=True,
                    help="enable variational consolidation")
    ev.add_argument("--lambda_evcl", type=float, default=1e3)
    ev.add_argument("--evcl_sigma_init", type=float, default=0.1)
    ev.add_argument("--evcl_lr", type=float, default=1e-3)
    ev.add_argument("--evcl_steps", type=int, default=3)
    ev.add_argument("--num_fim_batches", type=int, default=10)
    ev.add_argument("--fim_batch_size", type=int, default=64)
    ev.add_argument("--conf_thresh", type=float, default=0.6)
    ev.add_argument("--ent_thresh", type=float, default=0.45)
    ev.add_argument("--real_thresh", type=float, default=0.45)
    ev.add_argument("--ema_decay", type=float, default=0.8)
    ev.add_argument("--min_accept_frac", type=float, default=0.15)
    ev.add_argument("--dual_gate_evcl", action=argparse.BooleanOptionalAction, default=False,
                    help="Stage 2 confidence + entropy + realism gate (Eq. 20)")

    # -- Extension C / defense --------------------------------------------------
    de = p.add_argument_group("defense")
    de.add_argument("--defense_mode", default="none",
                    choices=["none", "stage1", "stage2", "full", "mmd", "extension_c"],
                    help="Stage 1 MMD screen and/or Stage 2 dual-gated EVCL")
    de.add_argument("--mmd_gamma", type=float, default=2.0,
                    help="Boltzmann trust temperature gamma_trust (Eq. 19)")
    de.add_argument("--mmd_val_batch", type=int, default=64,
                    help="M synthetic samples per active class")
    de.add_argument("--eta_js", type=float, default=0.5)
    de.add_argument("--margin_n", type=float, default=0.25)
    de.add_argument("--z_max", type=float, default=2.5)
    de.add_argument("--mmd_sigmas", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])

    # -- attacks (Phase 2) ------------------------------------------------------
    at = p.add_argument_group("attacks")
    at.add_argument("--attack_type", default="none",
                    choices=["none", "label_flip", "stealthy", "attack_a", "attack_b"])
    at.add_argument("--malicious_ratio", type=float, default=0.0,
                    help="fraction of compromised clients (e.g. 0.20)")
    at.add_argument("--malicious_clients", type=int, nargs="+", default=None,
                    help="explicit malicious client IDs (overrides --malicious_ratio)")
    at.add_argument("--attack_src", type=int, default=None,
                    help="Attack A source class (default: auto recurring class)")
    at.add_argument("--attack_tgt", type=int, default=None,
                    help="Attack A target class (default: auto co-occurring class)")
    at.add_argument("--attack_boost", type=float, default=2.0,
                    help="Attack A eta_boost in [1.5, 3.0]")
    at.add_argument("--attack_gamma", type=float, default=1.5,
                    help="Attack B flip factor gamma_flip")
    at.add_argument("--attack_seed", type=int, default=0)

    return p


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(name: str) -> str:
    name = name.lower()
    if name in ("emnist", "emnist-balanced"):
        return "emnist_balanced"
    if name in ("cifar-10", "cifar"):
        return "cifar10"
    return name


def resolve_defense(args: argparse.Namespace) -> Tuple[bool, bool]:
    """Return (stage1_mmd_enabled, stage2_dual_gate_enabled).

    * ``none``   -- baseline (Stage 2 only if --dual_gate_evcl is passed)
    * ``stage1`` -- MMD screen only (Stage 2 optional via --dual_gate_evcl)
    * ``stage2`` -- dual-gated EVCL only
    * ``full``   -- Extension C: Stage 1 + Stage 2
    """
    mode = args.defense_mode
    if mode in ("stage1", "mmd"):
        return True, bool(args.dual_gate_evcl)
    if mode == "stage2":
        return False, True
    if mode in ("full", "extension_c"):
        return True, True
    return False, bool(args.dual_gate_evcl)


def resolve_task_lengths(args: argparse.Namespace) -> "int | None":
    if args.random_task_lengths:
        return None
    return int(args.rounds_per_task)


def resolve_offsets(args: argparse.Namespace, num_clients: int) -> List[int]:
    if args.transition_offsets:
        if len(args.transition_offsets) != num_clients:
            raise ValueError("--transition_offsets needs one entry per client")
        return [int(v) for v in args.transition_offsets]
    if args.async_offset_max <= 0:
        return [0] * num_clients
    if num_clients == 1:
        return [0]
    return [
        int(round(i * args.async_offset_max / (num_clients - 1)))
        for i in range(num_clients)
    ]


def task_class_lists(stream: RecurringClassStream) -> List[Tuple[int, ...]]:
    """New global classes introduced at task t."""
    out: List[Tuple[int, ...]] = []
    previous: set = set()
    for t in range(stream.num_tasks):
        current = set(stream.global_cumulative_classes(t))
        introduced = tuple(sorted(current - previous))
        out.append(introduced)
        previous = current
    return out


def choose_attack_classes(stream: RecurringClassStream) -> Tuple[int, int]:
    """Pick a recurring source class and a co-occurring target class for Attack A."""
    counts: Dict[int, int] = {}
    for k in range(stream.num_clients):
        for cs in stream.class_sets[k]:
            for c in cs:
                counts[int(c)] = counts.get(int(c), 0) + 1
    recurring = sorted((c for c, n in counts.items() if n > 1),
                       key=lambda c: (-counts[c], c))
    source = recurring[0] if recurring else min(counts, default=0)
    for k in range(stream.num_clients):
        for cs in stream.class_sets[k]:
            if source in cs:
                alternatives = [int(c) for c in cs if int(c) != source]
                if alternatives:
                    return source, alternatives[0]
    target = next((int(c) for c in sorted(counts) if int(c) != source), (source + 1) % 10)
    return source, target


def attack_spec(args: argparse.Namespace, stream: RecurringClassStream):
    kind = {"attack_a": "label_flip", "attack_b": "stealthy"}.get(args.attack_type, args.attack_type)
    if kind == "none":
        return None, None
    if args.malicious_clients:
        malicious = sorted({int(k) for k in args.malicious_clients
                            if 0 <= int(k) < args.num_clients})
    else:
        n_mal = int(round(args.malicious_ratio * args.num_clients))
        if args.malicious_ratio > 0:
            n_mal = max(1, n_mal)
        malicious = list(range(n_mal))
    if kind == "label_flip":
        src, tgt = args.attack_src, args.attack_tgt
        if src is None or tgt is None:
            auto_src, auto_tgt = choose_attack_classes(stream)
            src = auto_src if src is None else int(src)
            tgt = auto_tgt if tgt is None else int(tgt)
        return malicious, ("label_flip", int(src), int(tgt))
    return malicious, ("stealthy", args.attack_gamma)


def build_attack(args: argparse.Namespace, spec, sigma_zcdp: float):
    if spec is None:
        return None
    if spec[0] == "label_flip":
        return LabelFlipAttack(source_class=spec[1], target_class=spec[2],
                               boost=args.attack_boost)
    return make_attack("stealthy", flip_gamma=spec[1], zcdp_sigma=sigma_zcdp,
                       seed=args.attack_seed)


def build_clients(
    args: argparse.Namespace,
    server: FederatedServer,
    stream: RecurringClassStream,
    bundle: DatasetBundle,
    malicious: Sequence[int],
    attack_specification,
    l2_clip: Optional[float],
) -> Dict[int, FederatedClient]:
    sigma = 0.0
    if math.isfinite(args.dp_epsilon) and args.dp_epsilon > 0 and l2_clip is not None:
        sigma = zcdp_sigma(l2_clip, args.dp_epsilon, args.delta_zcdp,
                           rounds=max(1, args.total_rounds))
    clients: Dict[int, FederatedClient] = {}
    for k in range(args.num_clients):
        attack = build_attack(args, attack_specification, sigma) if k in malicious else None
        client = FederatedClient(
            client_id=k,
            disc=clone_model(server.disc),
            gen=clone_model(server.gen),
            z_dim=args.z_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lambda_cls=args.lambda_cls,
            lambda_align=args.lambda_align,
            lambda_ce=args.lambda_ce,
            eta_dp=args.eta_dp,
            lr_disc=args.lr_disc,
            lr_gen=args.lr_gen,
            keep_ratio=args.keep_ratio,
            l2_clip=l2_clip,
            epsilon=args.dp_epsilon if math.isfinite(args.dp_epsilon) else 1.8,
            delta_zcdp=args.delta_zcdp,
            total_rounds=args.total_rounds,
            device=args.device,
            seed=args.seed,
            attack=attack,
        )
        clients[k] = client
    return clients


def train_batches_for_client(
    sampler: ClassBatchSampler,
    stream: RecurringClassStream,
    client_id: int,
    local_task: int,
    batch_size: int,
    batches_per_round: int,
    omega: float,
    max_samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    classes = stream.class_set(client_id, local_task)
    previous = (stream.cumulative_classes(client_id, local_task - 1)
                if local_task > 0 else ())
    if batches_per_round and batches_per_round > 0:
        n_batches = int(batches_per_round)
    else:
        n_samples = sum(sampler.index[c].numel() for c in classes if c in sampler.index)
        if max_samples and max_samples > 0:
            n_samples = min(n_samples, int(max_samples))
        n_batches = max(1, math.ceil(n_samples / max(1, int(batch_size))))
        # A smoke-test round should not require tens of thousands of local
        # optimizer steps; cap the auto schedule at a sensible per-epoch budget.
        n_batches = min(n_batches, 100)
    return sampler.make_batches(
        classes, int(batch_size), n_batches,
        previous_classes=previous, omega=float(omega),
    )


@torch.no_grad()
def evaluate_tasks_now(
    disc: torch.nn.Module,
    bundle: DatasetBundle,
    tracker: MetricsTracker,
    upto_task: int,
    num_classes_seen: Sequence[int],
    args: argparse.Namespace,
) -> None:
    """Evaluate all tasks <= ``upto_task`` and record one task-matrix row."""
    classes = sorted({int(c) for c in num_classes_seen})
    if not classes:
        return
    x, y = bundle.subset("test", classes)
    metrics = evaluate_classifier(
        disc, x, y, classes=None, batch_size=args.eval_batch_size,
        max_samples=args.eval_max_samples, device=args.device,
    )
    # record_task_eval fills row `upto_task` for every task <= it; earlier rows
    # must not be overwritten with the later model snapshot.
    tracker.record_task_eval(upto_task, metrics.per_class_accuracy, metrics.per_class_count)
    LOGGER.info(
        "task %d evaluation: acc=%.4f (n=%d)", upto_task, metrics.accuracy, metrics.num_samples
    )


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, object]:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )
    set_seed(args.seed)
    args.device = str(resolve_device(args.device))
    LOGGER.info(
        "device=%s (CUDA available=%s)", args.device, torch.cuda.is_available()
    )
    if args.synthetic_data:
        args.dataset = "synthetic"
    args.dataset = normalize_dataset_name(args.dataset)

    total_rounds_explicit = args.total_rounds is not None
    if args.total_rounds is None:
        args.total_rounds = int(args.num_tasks * args.rounds_per_task)
        if args.num_tasks == 5 and args.rounds_per_task == 10 and not total_rounds_explicit:
            args.total_rounds = 50
    if args.total_rounds <= 0:
        raise ValueError("--total_rounds must be positive")
    if not 0.0 < args.participation_rate <= 1.0:
        raise ValueError("--participation_rate must be in (0, 1]")
    if args.malicious_ratio < 0 or args.malicious_ratio > 1:
        raise ValueError("--malicious_ratio must be in [0, 1]")
    if args.recurrence_bias < 0:
        raise ValueError("--recurrence_bias must be >= 0")

    run_name = args.run_name or (
        f"{args.dataset}_K{args.num_clients}_T{args.num_tasks}"
        f"_w{args.recurrence_bias:g}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir = project_path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data + stream
    # ------------------------------------------------------------------
    bundle = load_dataset(
        args.dataset, root=project_path(args.data_root), allow_download=args.allow_download,
        synthetic=False, seed=args.seed,
        synthetic_samples_per_class=args.synthetic_samples_per_class,
        verbose=True,
    )
    LOGGER.info(
        "dataset=%s classes=%d channels=%d train=%d test=%d synthetic=%s",
        bundle.name, bundle.num_classes, bundle.img_channels,
        bundle.train_x.size(0), bundle.test_x.size(0), bundle.synthetic,
    )

    stream_seed = args.seed if args.stream_seed is None else args.stream_seed
    offsets = resolve_offsets(args, args.num_clients)
    stream = RecurringClassStream(
        num_classes=bundle.num_classes,
        num_clients=args.num_clients,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        omega=args.recurrence_bias,
        rounds_per_task=resolve_task_lengths(args),
        transition_offsets=offsets,
        task_length_pool=args.task_length_pool,
        seed=stream_seed,
    )
    if not total_rounds_explicit:
        # Asynchronous schedules need enough rounds for every client to finish
        # all local tasks (random task lengths and positive transition offsets
        # can otherwise truncate the stream before the last task).
        if args.random_task_lengths:
            args.total_rounds = int(max(sum(lengths) for lengths in stream.rounds_per_task)
                                    + max(offsets or [0]))
        elif max(offsets or [0]) > 0:
            args.total_rounds = int(args.total_rounds + max(offsets))
    task_classes = task_class_lists(stream)
    LOGGER.info("total rounds: %d", args.total_rounds)
    LOGGER.info("rho_rec per client: %s",
                {k: round(v, 4) for k, v in stream.rho_rec.items()})
    LOGGER.info("task classes: %s", {t: list(cs) for t, cs in enumerate(task_classes)})

    # ------------------------------------------------------------------
    # Global models / server
    # ------------------------------------------------------------------
    max_class = max(
        [max(cs) for k in range(args.num_clients) for cs in stream.class_sets[k]
         if cs] or [0]
    )
    head = int(max_class) + 1
    spectral = not args.no_spectral_norm
    disc = Discriminator(
        num_classes=head,
        img_channels=bundle.img_channels,
        feature_dim=args.feature_dim,
        proj_hidden=args.proj_hidden,
        proj_dim=args.proj_dim,
        spectral_norm_d=spectral,
    )
    img_size = int(args.img_size)
    if img_size != int(bundle.img_size):
        LOGGER.warning("--img_size=%d does not match %s (%d); using the dataset size",
                       img_size, bundle.name, bundle.img_size)
        img_size = int(bundle.img_size)
    gen = Generator(
        num_classes=head,
        z_dim=args.z_dim,
        img_size=img_size,
        img_channels=bundle.img_channels,
        base_channels=args.base_channels,
        spectral_norm_gen=spectral,
    )
    disc.to(args.device)
    gen.to(args.device)

    evcl = ServerEVCLConsolidator(
        lambda_evcl=args.lambda_evcl,
        sigma_init=args.evcl_sigma_init,
        conf_thresh=args.conf_thresh,
        ema_decay=args.ema_decay,
        z_dim=args.z_dim,
        batch_size=args.fim_batch_size,
        num_batches=args.num_fim_batches,
        lr=args.evcl_lr,
        steps=args.evcl_steps,
        min_accept_frac=args.min_accept_frac,
        dual_gate=bool(args.dual_gate_evcl),
        ent_thresh=args.ent_thresh,
        real_thresh=args.real_thresh,
        seed=args.seed,
    )

    stage1_on, stage2_on = resolve_defense(args)
    evcl.dual_gate = stage2_on

    defense = None
    if stage1_on:
        defense = ClassAwareMMDDefense(
            z_dim=args.z_dim,
            val_batch_size=args.mmd_val_batch,
            eta_js=args.eta_js,
            margin_n=args.margin_n,
            z_max=args.z_max,
            gamma_trust=args.mmd_gamma,
            sigmas=tuple(args.mmd_sigmas),
            seed=args.seed,
        )
    server = FederatedServer(disc, gen, evcl=evcl, defense=defense, seed=args.seed)

    # ------------------------------------------------------------------
    # Attacks + clients
    # ------------------------------------------------------------------
    malicious, attack_specification = attack_spec(args, stream)
    malicious = list(malicious or [])
    l2_clip = (float(args.l2_clip)
               if math.isfinite(args.dp_epsilon) and args.dp_epsilon > 0 else None)
    clients = build_clients(args, server, stream, bundle, malicious,
                            attack_specification, l2_clip)
    LOGGER.info("malicious clients: %s | attack: %s | defense: stage1=%s stage2=%s",
                malicious, attack_specification, stage1_on, stage2_on)
    if args.attack_type == "label_flip" and attack_specification:
        LOGGER.info("Attack A classes: c_src=%d -> c_tgt=%d",
                    attack_specification[1], attack_specification[2])

    samplers = {
        k: ClassBatchSampler(bundle.train_x, bundle.train_y, seed=args.seed + 1000 + k)
        for k in range(args.num_clients)
    }
    sample_gen = torch.Generator().manual_seed(args.seed + 97)
    tracker = MetricsTracker(task_classes, num_tasks=args.num_tasks)
    drift_tracker = CentroidDriftTracker()

    next_task_to_record = 0
    last_global_task = 0
    active_counts: List[int] = []

    # ------------------------------------------------------------------
    # Communication loop
    # ------------------------------------------------------------------
    for r in range(int(args.total_rounds)):
        local_tasks = {k: stream.local_task(k, r) for k in range(args.num_clients)}
        global_task = max(local_tasks.values())
        last_global_task = max(last_global_task, global_task)

        # End-of-task snapshot *before* the first local update of the next task.
        # With asynchronous boundaries the maximum local task can jump by more
        # than one; record every completed task against the same model state.
        while global_task > next_task_to_record:
            seen_for_task = set()
            for t in range(next_task_to_record + 1):
                seen_for_task |= set(stream.global_cumulative_classes(t))
            evaluate_tasks_now(server.disc, bundle, tracker, next_task_to_record,
                               sorted(seen_for_task), args)
            next_task_to_record += 1

        # Active client set (Stress Axis 2).
        if args.participation_rate >= 1.0:
            active = list(range(args.num_clients))
        else:
            n_active = max(1, int(round(args.participation_rate * args.num_clients)))
            perm = torch.randperm(args.num_clients, generator=sample_gen).tolist()
            active = sorted(perm[:n_active])
        active_counts.append(len(active))

        # Announce the labels that will be trained this round so the global
        # generator can synthesize replay samples for every label.
        announced = set()
        for k in active:
            announced |= set(stream.class_set(k, local_tasks[k]))
        if attack_specification and attack_specification[0] == "label_flip":
            announced.add(int(attack_specification[2]))
        if announced:
            server.ensure_heads(max(announced) + 1)

        updates = []
        local_accs: List[float] = []
        local_acc_by_client: Dict[str, float] = {}
        debug_diagnostics = args.log_level.upper() == "DEBUG"
        local_xs: List[torch.Tensor] = []
        local_ys: List[torch.Tensor] = []
        for k in active:
            client = clients[k]
            client.receive_global(server.disc, server.gen)
            local_task = local_tasks[k]
            classes = list(stream.class_set(k, local_task))
            cumulative = list(stream.cumulative_classes(k, local_task))
            updated = set(classes)
            if k in malicious and isinstance(client.attack, LabelFlipAttack):
                updated.add(int(client.attack.target_class))
            batches = train_batches_for_client(
                samplers[k], stream, k, local_task, args.batch_size,
                args.batches_per_round, args.recurrence_bias, args.max_train_samples,
            )
            if not batches:
                LOGGER.warning("client %d has no training data at round %d", k, r)
                continue
            update = client.local_train(
                batches,
                global_gen=server.gen,
                round_idx=r,
                task_idx=local_task,
                active_classes=cumulative,
                updated_classes=sorted(updated),
            )
            # Stage-by-stage diagnostic: local accuracy on the client's own
            # training data, measured *before* upload/aggregation.
            x_loc = torch.cat([b[0] for b in batches], dim=0)
            y_loc = torch.cat([b[1] for b in batches], dim=0)
            loc_metrics = evaluate_classifier(
                client.disc, x_loc, y_loc, batch_size=args.eval_batch_size,
                max_samples=args.eval_max_samples, device=args.device,
            )
            update.metrics["local_acc"] = float(loc_metrics.accuracy)
            local_accs.append(float(loc_metrics.accuracy))
            local_acc_by_client[str(k)] = float(loc_metrics.accuracy)
            if debug_diagnostics:
                local_xs.append(x_loc)
                local_ys.append(y_loc)
            updates.append(update)

        if not updates:
            LOGGER.warning("round %d: no client updates", r)
            continue

        stage_metrics: Dict[str, object] = {}
        if local_accs:
            stage_metrics["local_train_acc_mean"] = float(
                sum(local_accs) / len(local_accs)
            )
            stage_metrics["local_train_acc_min"] = float(min(local_accs))
            stage_metrics["local_client_acc"] = dict(local_acc_by_client)

        x_local_all = torch.cat(local_xs, dim=0) if local_xs else None
        y_local_all = torch.cat(local_ys, dim=0) if local_ys else None

        def _global_train_accuracy() -> Dict[str, float]:
            if x_local_all is None or y_local_all is None:
                return {}
            m = evaluate_classifier(
                server.disc, x_local_all, y_local_all,
                batch_size=args.eval_batch_size,
                max_samples=args.eval_max_samples, device=args.device,
            )
            return {
                "global_train_acc_post_agg": float(m.accuracy),
                "global_train_acc_post_agg_loss": float(m.loss),
            }

        # Only compute extra diagnostic passes in DEBUG mode
        hook = _global_train_accuracy if debug_diagnostics else None
        server_metrics = server.run_round(
            updates,
            use_defense=stage1_on,
            consolidate=bool(args.evcl),
            post_aggregate_hook=hook,
        )

        if args.evcl and debug_diagnostics and x_local_all is not None and y_local_all is not None:
            post_evcl = evaluate_classifier(
                server.disc, x_local_all, y_local_all,
                batch_size=args.eval_batch_size,
                max_samples=args.eval_max_samples, device=args.device,
            )
            server_metrics["global_train_acc_post_evcl"] = float(post_evcl.accuracy)

        record: Dict[str, object] = {
            "round": int(r),
            "global_task": int(global_task),
            "active_clients": int(len(active)),
            "participation_rate": float(len(active) / args.num_clients),
            "rho_rec_mean": float(sum(stream.rho_rec.values()) / len(stream.rho_rec)),
            **stage_metrics,
            **{k: v for k, v in server_metrics.items() if k != "round"},
        }
        if "fim_acceptance" in server_metrics:
            record["dual_gate_pass_rate"] = server_metrics["fim_acceptance"]
        if "local_train_acc_mean" in record and "global_train_acc_post_agg" in record:
            LOGGER.info(
                "round %d stage: local_train_acc=%.4f -> global_train_acc_post_agg=%.4f"
                " -> post_evcl=%s",
                r + 1,
                float(record["local_train_acc_mean"]),
                float(record["global_train_acc_post_agg"]),
                f"{float(record['global_train_acc_post_evcl']):.4f}"
                if record.get("global_train_acc_post_evcl") is not None else "-",
            )

        next_global_task = (
            max(stream.local_task(k, r + 1) for k in range(args.num_clients))
            if r + 1 < int(args.total_rounds) else global_task
        )
        task_boundary = next_global_task > global_task
        do_eval = (r == args.total_rounds - 1) or task_boundary or (
            args.eval_every > 0 and (r + 1) % args.eval_every == 0
        )
        if do_eval:
            seen = sorted(int(c) for c in server.seen_classes)
            if not seen:
                seen = sorted(set(stream.global_cumulative_classes(global_task)))
            x_eval, y_eval = bundle.subset("test", seen)
            attack_src = None
            attack_tgt = None
            if attack_specification and attack_specification[0] == "label_flip" and len(attack_specification) >= 3:
                attack_src = int(attack_specification[1])
                attack_tgt = int(attack_specification[2])
            cls_metrics = evaluate_classifier(
                server.disc, x_eval, y_eval, batch_size=args.eval_batch_size,
                max_samples=args.eval_max_samples, device=args.device,
                attack_src=attack_src, attack_tgt=attack_tgt,
            )
            seen_before = (set(stream.global_cumulative_classes(global_task - 1))
                           if global_task > 0 else set())
            old_classes, new_classes = old_new_split(seen_before, seen)
            record.update({
                "test_acc": float(cls_metrics.accuracy),
                "clean_accuracy": float(getattr(cls_metrics, 'clean_accuracy', cls_metrics.accuracy)) if getattr(cls_metrics, 'clean_accuracy', None) is not None else float(cls_metrics.accuracy),
                "attacked_accuracy": float(getattr(cls_metrics, 'attacked_accuracy', None)) if getattr(cls_metrics, 'attacked_accuracy', None) is not None else None,
                "test_loss": float(cls_metrics.loss),
                "num_eval_samples": int(cls_metrics.num_samples),
                "old_class_acc": cls_metrics.accuracy_on(old_classes),
                "new_class_acc": cls_metrics.accuracy_on(new_classes),
            })
            if attack_specification and attack_specification[0] == "label_flip" and len(attack_specification) >= 3 and attack_src is not None and attack_tgt is not None:
                src, tgt = attack_src, attack_tgt
                record["source_class_acc"] = cls_metrics.per_class_accuracy.get(src)
                record["target_class_acc"] = cls_metrics.per_class_accuracy.get(tgt)
                if record.get("attacked_accuracy") is None:
                    att = cls_metrics.per_class_accuracy
                    c1 = cls_metrics.per_class_count.get(src, 0)
                    c2 = cls_metrics.per_class_count.get(tgt, 0)
                    if c1 + c2 > 0:
                        record["attacked_accuracy"] = float((att.get(src, 0.0) * c1 + att.get(tgt, 0.0) * c2) / (c1 + c2))

            if args.metrics_mmd:
                mmd_classes = (task_classes[global_task]
                               if global_task < len(task_classes) and task_classes[global_task]
                               else seen)
                mmd, per_class_mmd = feature_space_mmd(
                    server.disc, server.gen, bundle.test_x, bundle.test_y,
                    classes=mmd_classes,
                    samples_per_class=args.eval_feature_samples,
                    batch_size=args.eval_batch_size,
                    device=args.device,
                    sigmas=tuple(args.mmd_sigmas),
                    rng=torch.Generator().manual_seed(args.seed + r),
                )
                record["feature_mmd"] = mmd
                record["feature_mmd_per_class"] = per_class_mmd

            if args.metrics_wasserstein:
                mmd_classes = (task_classes[global_task]
                               if global_task < len(task_classes) and task_classes[global_task]
                               else seen)
                wasserstein, per_class_w1 = feature_space_wasserstein(
                    server.disc, server.gen, bundle.test_x, bundle.test_y,
                    classes=mmd_classes,
                    samples_per_class=args.eval_feature_samples,
                    batch_size=args.eval_batch_size,
                    device=args.device,
                    n_projections=args.wasserstein_projections,
                    rng=torch.Generator().manual_seed(args.seed + r + 7),
                )
                record["feature_wasserstein"] = wasserstein
                record["feature_wasserstein_per_class"] = per_class_w1

            if args.metrics_centroid:
                centroids = class_feature_centroids(
                    server.disc, bundle.test_x, bundle.test_y, classes=seen,
                    samples_per_class=args.eval_feature_samples,
                    batch_size=args.eval_batch_size,
                    device=args.device,
                    generator=torch.Generator().manual_seed(args.seed + r),
                )
                drift = drift_tracker.update(centroids)
                record["centroid_drift"] = drift.centroid_drift
                record["centroid_drift_anchor"] = drift.centroid_drift_anchor
                record["centroid_drift_per_class"] = drift.per_class_centroid_drift
                record["num_tracked_centroids"] = len(centroids)

            if (attack_specification and attack_specification[0] == "label_flip"):
                record["attack_success_rate"] = targeted_attack_success_rate(
                    server.disc, bundle.test_x, bundle.test_y,
                    source_class=int(attack_specification[1]),
                    target_class=int(attack_specification[2]),
                    batch_size=args.eval_batch_size, device=args.device,
                )

        row = tracker.log_round(record)
        if r % max(1, args.total_rounds // 20) == 0 or do_eval:
            LOGGER.info(
                "round %d/%d task=%d active=%d acc=%s mmd=%s drift=%s",
                r + 1, args.total_rounds, global_task, len(active),
                f"{row.get('test_acc'):.4f}" if row.get("test_acc") is not None else "-",
                f"{row.get('feature_mmd'):.4f}" if row.get("feature_mmd") is not None else "-",
                f"{row.get('centroid_drift'):.4f}"
                if row.get("centroid_drift") is not None else "-",
            )
        if args.save_every and (r + 1) % args.save_every == 0:
            export_metrics(tracker, out_dir, config=vars(args))

    # Final task-matrix rows for tasks reached by the last round.
    final_task = max(last_global_task, 0)
    while next_task_to_record <= final_task:
        seen_for_task = set()
        for t in range(next_task_to_record + 1):
            seen_for_task |= set(stream.global_cumulative_classes(t))
        evaluate_tasks_now(server.disc, bundle, tracker, next_task_to_record,
                           sorted(seen_for_task), args)
        next_task_to_record += 1

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    summary_extra = {
        "dataset": bundle.name,
        "dataset_synthetic": bool(bundle.synthetic),
        "num_classes": int(bundle.num_classes),
        "img_channels": int(bundle.img_channels),
        "img_size": int(bundle.img_size),
        "rho_rec": {str(k): float(v) for k, v in stream.rho_rec.items()},
        "task_classes": {str(t): list(cs) for t, cs in enumerate(task_classes)},
        "malicious_clients": list(malicious),
        "attack": list(attack_specification) if attack_specification else None,
        "defense_stage1": bool(stage1_on),
        "defense_stage2": bool(stage2_on),
        "mean_active_clients": float(sum(active_counts) / len(active_counts))
        if active_counts else 0.0,
    }
    paths = export_metrics(tracker, out_dir, config=vars(args), summary_extra=summary_extra)
    summary = tracker.summary()
    LOGGER.info(
        "finished: final_acc=%s avg_forgetting=%s",
        summary.get("final_average_accuracy"),
        summary.get("average_forgetting_rate"),
    )
    LOGGER.info("wrote %s", ", ".join(str(p) for p in paths.values()))
    return {"output_dir": str(out_dir), **paths, "summary": summary}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
