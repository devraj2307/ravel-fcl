"""Non-monotonic recurring class stream protocol (spec Sections 2.1 and 3.1, Eq. 1).

K clients train across T sequential tasks.  At local task ``t`` client ``k`` holds a
class set ``C_t^(k) ⊂ C`` with ``|C_t^(k)| ∈ {2, 3, 4}``.  Unlike strictly disjoint
class-incremental learning, previously observed classes may *reappear* in later tasks
(non-monotonic recurrence).  The exact recurrence ratio of client ``k`` is (Eq. 1)::

    r_c^(k)   = |{u ∈ {1..T} : c ∈ C_u^(k)}|          (recurrence count of class c)
    rho_rec^(k) = (1 / |C|) * Σ_{c ∈ C} I[r_c^(k) > 1]

Recurrence bias omega (spec Section 3.1, Stress Axis 1)
-------------------------------------------------------
When client ``k`` samples m ∈ {2, 3, 4} classes for task ``t``, every *previously seen*
class (c ∈ C_{≤t-1}^(k)) is assigned sampling weight ``omega`` and every *unseen* class
weight ``1.0``.  Classes are drawn **with replacement** and de-duplicated per task, so:

* ``omega = 0.0`` -> strictly disjoint monotonic CIL, ``rho_rec = 0.0``.
* ``omega = 1.0`` -> uniform random sampling with replacement (paper default).
* ``omega >> 1``  -> skewed revisitation; early classes dominate later tasks.

Asynchronous task boundaries (spec Section 3.3, Stress Axis 3)
--------------------------------------------------------------
The task index at client ``k`` need not align with the task index at client ``k'``.
Two mechanisms are supported:

* ``transition_offsets`` -- per-client round offsets ΔR (stagger ∈ {0, 10, 25}); a
  positive offset delays the client's task transitions.
* ``rounds_per_task`` -- per-client task lengths R_t^(k) (e.g. drawn from
  {25, 35, 40, 45, 55}) so clients cross task boundaries at staggered rounds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)

DEFAULT_CLASSES_PER_TASK: Tuple[int, ...] = (2, 3, 4)
DEFAULT_TASK_LENGTH_POOL: Tuple[int, ...] = (25, 35, 40, 45, 55)


class RecurringClassStream:
    """Generates and queries a non-monotonic recurring class stream.

    Args:
        num_classes: Size |C| of the global label set.
        num_clients: Number of federated clients K.
        num_tasks: Number of sequential tasks T.
        classes_per_task: Candidate sizes m for |C_t^(k)| (spec: {2, 3, 4}).
        omega: Recurrence bias (sampling weight of previously seen classes; unseen
            classes always get weight 1.0).  omega=0 -> disjoint monotonic CIL.
        rounds_per_task: R_t per client.  An int uses the same length for every
            client/task (40 rounds for the T=5, R=200 baseline).  A nested
            sequence ``[[R_t^(k) for t] for k]`` sets per-client task lengths.
            ``None`` draws each R_t^(k) from ``task_length_pool`` (Stress Axis 3).
        transition_offsets: Per-client round offsets ΔR applied to task boundaries
            (asynchronous task transitions; Stress Axis 3).  Default: synchronous.
        task_length_pool: Pool sampled from when ``rounds_per_task is None``.
        seed: RNG seed for reproducible streams.
    """

    def __init__(
        self,
        num_classes: int,
        num_clients: int = 5,
        num_tasks: int = 5,
        classes_per_task: Sequence[int] = DEFAULT_CLASSES_PER_TASK,
        omega: float = 1.0,
        rounds_per_task: "int | Sequence[Sequence[int]] | None" = 40,
        transition_offsets: Optional[Sequence[int]] = None,
        task_length_pool: Sequence[int] = DEFAULT_TASK_LENGTH_POOL,
        seed: int = 0,
    ) -> None:
        if num_classes < max(classes_per_task):
            raise ValueError("num_classes must be >= max(classes_per_task)")
        if omega < 0:
            raise ValueError("omega (recurrence bias) must be >= 0")

        self.num_classes = int(num_classes)
        self.num_clients = int(num_clients)
        self.num_tasks = int(num_tasks)
        self.classes_per_task = tuple(int(m) for m in classes_per_task)
        self.omega = float(omega)
        self.task_length_pool = tuple(int(r) for r in task_length_pool)
        self.seed = int(seed)

        self._generator = torch.Generator().manual_seed(self.seed)

        # ------------------------------------------------------------------
        # Per-client task lengths R_t^(k) and transition offsets ΔR^(k) (rounds).
        # ------------------------------------------------------------------
        self.rounds_per_task = self._resolve_rounds_per_task(rounds_per_task)
        if transition_offsets is None:
            offsets = [0] * self.num_clients
        else:
            offsets = [int(d) for d in transition_offsets]
            if len(offsets) != self.num_clients:
                raise ValueError("transition_offsets must have one entry per client")
        self.transition_offsets = offsets

        # ------------------------------------------------------------------
        # Class schedules C_t^(k), recurrence counts r_c^(k) and rho_rec^(k).
        # ------------------------------------------------------------------
        self.class_sets: List[List[Tuple[int, ...]]] = []
        self._recurrence_counts: List[torch.Tensor] = []
        self.rho_rec: Dict[int, float] = {}
        for k in range(self.num_clients):
            sets, counts = self._sample_client_schedule(k)
            self.class_sets.append(sets)
            self._recurrence_counts.append(counts)
            self.rho_rec[k] = self._exact_recurrence_ratio(counts)
            self._log_recurrence(k)

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------
    def _resolve_rounds_per_task(
        self, rounds_per_task: "int | Sequence[Sequence[int]] | None"
    ) -> List[List[int]]:
        if rounds_per_task is None:
            return [
                [
                    int(self.task_length_pool[m % len(self.task_length_pool)])
                    for m in torch.randint(
                        len(self.task_length_pool), (self.num_tasks,),
                        generator=self._generator,
                    ).tolist()
                ]
                for _ in range(self.num_clients)
            ]
        if isinstance(rounds_per_task, int):
            return [
                [int(rounds_per_task)] * self.num_tasks
                for _ in range(self.num_clients)
            ]
        per_client = [list(map(int, rts)) for rts in rounds_per_task]
        if len(per_client) != self.num_clients or any(
            len(rts) != self.num_tasks for rts in per_client
        ):
            raise ValueError(
                "rounds_per_task must be [num_clients][num_tasks] when given as a sequence"
            )
        return per_client

    def _sample_client_schedule(self, k: int) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
        """Sample C_t^(k) for all t with disjoint new-class pools per task."""
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        seen: List[int] = []
        sets: List[Tuple[int, ...]] = []
        # Partition classes into disjoint pools U_0,...,U_{T-1} (equal split, shuffled)
        g_global = torch.Generator().manual_seed(self.seed)
        perm_global = torch.randperm(self.num_classes, generator=g_global).tolist()
        g = torch.Generator().manual_seed(self.seed + k * 131542391)
        per_task = max(1, self.num_classes // max(1, self.num_tasks))
        pools: List[List[int]] = []
        idx = 0
        for t in range(self.num_tasks):
            pool = perm_global[idx : idx + per_task]
            idx += per_task
            pools.append(sorted([int(x) for x in pool]))
        # ensure disjoint and cover all
        all_assigned = set()
        for p in pools:
            for c in p:
                all_assigned.add(c)
        if len(all_assigned) < self.num_classes:
            remaining = [c for c in range(self.num_classes) if c not in all_assigned]
            j = 0
            for t in range(self.num_tasks):
                while len(pools[t]) < per_task and j < len(remaining):
                    pools[t].append(remaining[j])
                    j += 1
                pools[t] = sorted(pools[t])
        for t in range(self.num_tasks):
            new_pool = list(pools[t])  # ONLY from U_t
            old_pool = []
            for u in range(t):
                old_pool.extend(pools[u])
            task_classes_list: List[int] = list(new_pool)
            # Add recurring old classes according to omega from U_{<t}
            if old_pool and self.omega > 0 and t >= 1:
                weights_old = torch.ones(len(old_pool), dtype=torch.float32) * self.omega
                want_old = min(len(old_pool), max(1, per_task))
                draws_old = torch.multinomial(
                    weights_old, num_samples=want_old, replacement=False, generator=g
                ).tolist()
                for d in draws_old:
                    c = old_pool[d]
                    if c not in task_classes_list:
                        task_classes_list.append(c)
            task_set = tuple(sorted(set(task_classes_list)))
            sets.append(task_set)
            for c in task_set:
                counts[c] += 1
                if c not in seen:
                    seen.append(c)
        return sets, counts

    @staticmethod
    def _exact_recurrence_ratio(counts: torch.Tensor) -> float:
        """rho_rec^(k) = (1/|C|) * Σ_c I[r_c^(k) > 1]  (exact, per Eq. 1)."""
        return float((counts > 1).sum().item()) / float(counts.numel())

    def _log_recurrence(self, k: int) -> None:
        LOGGER.info(
            "client %d: rho_rec^(k) = %.6f (omega=%.3f, T=%d)",
            k, self.rho_rec[k], self.omega, self.num_tasks,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def class_set(self, k: int, local_task: int) -> Tuple[int, ...]:
        """C_t^(k) for local task index t (0-based)."""
        return self.class_sets[k][int(local_task)]

    def cumulative_classes(self, k: int, local_task: int) -> Tuple[int, ...]:
        """C_{≤t}^(k) = ∪_{u≤t} C_u^(k) for client k (0-based local task)."""
        seen: List[int] = []
        for t in range(int(local_task) + 1):
            for c in self.class_sets[k][t]:
                if c not in seen:
                    seen.append(c)
        return tuple(sorted(seen))

    def global_cumulative_classes(self, local_task: int) -> Tuple[int, ...]:
        """Global cumulative class set C_{≤t} = ∪_k C_{≤t}^(k); size is C_t."""
        seen: set = set()
        for k in range(self.num_clients):
            seen |= set(self.cumulative_classes(k, local_task))
        return tuple(sorted(seen))

    def recurrence_counts(self, k: int) -> torch.Tensor:
        """r_c^(k) for every class c (LongTensor of shape [num_classes])."""
        return self._recurrence_counts[k].clone()

    def boundary_rounds(self, k: int) -> List[int]:
        """Rounds at which client k transitions to tasks 1..T (with offset ΔR)."""
        lengths = self.rounds_per_task[k]
        bounds = [0]
        for r in lengths[:-1]:
            bounds.append(bounds[-1] + r)
        offset = self.transition_offsets[k]
        return [b + offset for b in bounds]

    def local_task(self, k: int, global_round: int) -> int:
        """Local task index of client k at (0-based) global round ``global_round``.

        Per-client task lengths R_t^(k) and the per-client transition offset ΔR^(k)
        make these boundaries asynchronous across clients.
        """
        bounds = self.boundary_rounds(k)
        t = 0
        for i, b in enumerate(bounds):
            if global_round >= b:
                t = i
        return min(t, self.num_tasks - 1)

    def classes_at_round(self, k: int, global_round: int) -> Tuple[int, ...]:
        """Class set C_t^(k) of client k at (0-based) global round ``global_round``."""
        return self.class_set(k, self.local_task(k, global_round))

    def sample_labels(
        self, k: int, global_round: int, n: int, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Draw n labels with replacement from client k's current class set."""
        classes = self.classes_at_round(k, global_round)
        g = torch.Generator()
        g.manual_seed(self.seed if seed is None else seed)
        idx = torch.randint(len(classes), (n,), generator=g)
        return torch.as_tensor(classes, dtype=torch.long)[idx]

    def summary(self) -> Dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "num_tasks": self.num_tasks,
            "omega": self.omega,
            "rho_rec": dict(self.rho_rec),
            "transition_offsets": list(self.transition_offsets),
        }
