"""
solver_nsga.py - NSGA-II Multi-Objective Schedule Optimiser

CP-SAT is the hard-constraint safety layer.
NSGA-II searches a gene space that controls a priority-based heuristic
decoder. Every decoded schedule is validated; invalid ones get a large
penalty so NSGA-II learns to avoid them.

Gene encoding (N tasks + 4 global biases, all in [0, 1]):
    gene[i]      - task-order bias for task i (higher = schedule earlier)
    gene[N + 0]  - fatigue_spread_bias
    gene[N + 1]  - deadline_bias
    gene[N + 2]  - compactness_bias
    gene[N + 3]  - switch_bias

Requires: pip install pymoo
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
except ImportError as _pymoo_err:
    raise ImportError(
        "pymoo is required for NSGA scheduling.\n"
        "Install it with:  pip install pymoo\n"
        f"Original error: {_pymoo_err}"
    ) from _pymoo_err

from context import UserContext, apply_context
from models import (
    Resource,
    ScheduledTask,
    ScheduleResult,
    Task,
    UnscheduledTask,
)
from objectives import (
    INVALID_PENALTY,
    ObjectiveVector,
    evaluate_objectives,
)
from solver_cpsat import solve_cpsat
from validate import validate_solution


@dataclass
class NSGAConfig:
    """Tuning knobs for the NSGA-II optimisation run."""

    population_size: int = 80
    generations: int = 80
    seed: int = 42
    time_limit_seconds: Optional[float] = None
    max_invalid_penalty: float = INVALID_PENALTY
    exact_horizon_days: int = 14
    use_cpsat_fallback: bool = True


@dataclass
class CandidateSchedule:
    """A decoded and evaluated candidate from the NSGA population."""

    result: ScheduleResult
    objectives: ObjectiveVector
    raw_x: List[float]
    valid: bool
    note: str = ""
    calendar: Optional[List[Resource]] = None


def _available_hours_on_day(
    day_idx: int,
    day: Resource,
    occupied: Set[int],
) -> List[int]:
    """Absolute hours on this day that are inside work slots and not occupied."""
    base = day_idx * 24
    result: List[int] = []
    for slot_start, slot_end in day.work_slots:
        h_lo = int(math.ceil(base + slot_start))
        h_hi = int(math.floor(base + slot_end))
        for h in range(h_lo, h_hi):
            if h not in occupied:
                result.append(h)
    return result


def decode_candidate(
    x: np.ndarray,
    tasks: List[Task],
    calendar: List[Resource],
    user_context: UserContext,
) -> ScheduleResult:
    """Decode gene vector into a ScheduleResult using a priority heuristic.

    Respects work slots, capacity, no-overlap, deadlines, and dependencies.
    Returns one ScheduledTask per allocated hour.
    """
    n = len(tasks)
    fatigue_spread = float(x[n])
    deadline_b = float(x[n + 1])
    compactness_b = float(x[n + 2])
    switch_b = float(x[n + 3])

    occupied: Set[int] = set()
    day_used: Dict[int, float] = {i: 0.0 for i in range(len(calendar))}
    task_finish_hour: Dict[str, int] = {}
    completed: Set[str] = set()

    scheduled: List[ScheduledTask] = []
    resource_usage: Dict[str, float] = {r.id: 0.0 for r in calendar}

    def _ready(t: Task) -> bool:
        return all(dep in completed for dep in t.dependencies)

    def _earliest_start(t: Task) -> int:
        if not t.dependencies:
            return 0
        dep_finishes = [task_finish_hour.get(dep, -1) for dep in t.dependencies]
        return max(dep_finishes) + 1 if dep_finishes else 0

    def _score_task(t: Task, idx: int) -> float:
        gene = float(x[idx])
        horizon = len(calendar) * 24
        urgency = max(0.0, min(1.0, 1.0 - (t.deadline / (horizon + 1))))
        weight_penalty = (t.cognitive_weight / 10.0) * fatigue_spread * 0.2

        return (
            gene * 0.4
            + t.priority / 10.0 * 0.3
            + urgency * deadline_b * 0.3
            - weight_penalty
        )

    def _try_schedule_task(task: Task) -> bool:
        """Greedily allocate hours for a task across the calendar."""
        needed = int(math.ceil(task.duration))
        earliest = _earliest_start(task)
        deadline_h = int(math.floor(task.deadline))
        start_day = max(0, earliest // 24)
        allocated_count = 0
        last_hour_used = -1

        for day_idx in range(start_day, len(calendar)):
            if allocated_count >= needed:
                break

            day = calendar[day_idx]
            cap_left = int(math.floor(day.capacity)) - day_used[day_idx]
            if cap_left <= 0:
                continue

            avail = _available_hours_on_day(day_idx, day, occupied)
            avail = [h for h in avail if h >= earliest and h < deadline_h]
            if not avail:
                continue

            if compactness_b > 0.5 and last_hour_used >= 0:
                avail.sort(key=lambda h: (abs(h - last_hour_used - 1), h))

            if switch_b > 0.5:
                my_hours_on_day = {
                    e.start_time for e in scheduled
                    if e.task_id == task.id and e.resource_id == day.id
                }
                if my_hours_on_day:
                    avail.sort(key=lambda h: min(abs(h - mh) for mh in my_hours_on_day))

            hours_this_day = min(needed - allocated_count, int(cap_left), len(avail))
            selected = avail[:hours_this_day]

            for h in selected:
                occupied.add(h)
                day_used[day_idx] += 1.0
                resource_usage[day.id] = resource_usage.get(day.id, 0.0) + 1.0
                scheduled.append(
                    ScheduledTask(
                        task_id=task.id,
                        resource_id=day.id,
                        allocated_hours=1.0,
                        start_time=float(h),
                    )
                )
                last_hour_used = h
                task_finish_hour[task.id] = h
                allocated_count += 1

        return allocated_count >= needed

    unscheduled: List[UnscheduledTask] = []
    task_indices = {t.id: i for i, t in enumerate(tasks)}

    remaining = list(tasks)
    max_passes = len(tasks) * 2
    passes = 0

    while remaining and passes < max_passes:
        passes += 1
        ready = [t for t in remaining if _ready(t)]
        if not ready:
            for t in remaining:
                unscheduled.append(
                    UnscheduledTask(
                        task_id=t.id, task_name=t.name,
                        remaining_hours=t.duration,
                        reason="dependency cycle or unresolvable dependency",
                        deadline=t.deadline,
                    )
                )
            remaining = []
            break

        ready.sort(key=lambda t: _score_task(t, task_indices[t.id]), reverse=True)

        for task in ready:
            ok = _try_schedule_task(task)
            if not ok:
                unscheduled.append(
                    UnscheduledTask(
                        task_id=task.id, task_name=task.name,
                        remaining_hours=task.duration,
                        reason="decoder could not place within constraints",
                        deadline=task.deadline,
                    )
                )
            completed.add(task.id)

        remaining = [t for t in remaining if t.id not in completed]

    return ScheduleResult(
        scheduled=scheduled,
        unscheduled=unscheduled,
        deadline_misses=[],
        dependency_violations=[],
        resource_usage=resource_usage,
    )


class SchedulingNSGAProblem(ElementwiseProblem):
    """pymoo problem that wraps the heuristic decoder.

    Each individual is a gene vector of length len(tasks) + 4.
    Invalid schedules get INVALID_PENALTY objectives.
    """

    def __init__(
        self,
        tasks: List[Task],
        calendar: List[Resource],
        user_context: UserContext,
        max_penalty: float = INVALID_PENALTY,
    ) -> None:
        n_var = len(tasks) + 4
        super().__init__(
            n_var=n_var, n_obj=4, n_constr=0,
            xl=np.zeros(n_var), xu=np.ones(n_var),
        )
        self.tasks = tasks
        self.calendar = calendar
        self.user_context = user_context
        self.max_penalty = max_penalty
        self._cache: Dict[Tuple[float, ...], CandidateSchedule] = {}

    def _make_key(self, x: np.ndarray) -> Tuple[float, ...]:
        return tuple(round(float(v), 4) for v in x)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        key = self._make_key(x)

        if key in self._cache:
            out["F"] = self._cache[key].objectives.as_list()
            return

        try:
            result = decode_candidate(x, self.tasks, self.calendar, self.user_context)
        except Exception as exc:
            warnings.warn(f"solver_nsga: decoder exception: {exc}", stacklevel=2)
            penalty_vec = [self.max_penalty] * 4
            candidate = CandidateSchedule(
                result=ScheduleResult(
                    scheduled=[], unscheduled=[], deadline_misses=[],
                    dependency_violations=[], resource_usage={}
                ),
                objectives=ObjectiveVector(*penalty_vec),
                raw_x=list(x), valid=False,
                note=f"decoder exception: {exc}",
            )
            self._cache[key] = candidate
            out["F"] = penalty_vec
            return

        valid = validate_solution(result, self.tasks, debug=False, calendar=self.calendar)

        if valid:
            objectives = evaluate_objectives(result, self.tasks, self.calendar, self.user_context)
            note = ""
        else:
            objectives = ObjectiveVector(*([self.max_penalty] * 4))
            note = "failed validation"

        candidate = CandidateSchedule(
            result=result, objectives=objectives,
            raw_x=list(x), valid=valid, note=note,
        )
        self._cache[key] = candidate
        out["F"] = objectives.as_list()


def select_pareto_solution(
    candidates: List[CandidateSchedule],
    user_context: UserContext,
    preference_profile=None,
) -> Optional[CandidateSchedule]:
    """Pick the best Pareto candidate via weighted normalised scoring.

    Weights adjust for energy level, deadline pressure, and learned
    preference profile. Hard constraints are never modified.
    """
    valid = [c for c in candidates if c.valid]
    if not valid:
        return None

    w_fatigue = 1.0
    w_switches = 1.0
    w_deadline = 1.2
    w_frag = 0.8

    if user_context.energy_level is not None:
        energy = user_context.energy_level
        if energy <= 3:
            w_fatigue *= 1.8
        elif energy <= 6:
            w_fatigue *= 1.3

    if user_context.deadline_pressure_mode:
        intensity = user_context.deadline_pressure_intensity
        if intensity == "extreme":
            w_deadline *= 2.0
        elif intensity == "moderate":
            w_deadline *= 1.5
        else:
            w_deadline *= 1.2

    if preference_profile is not None:
        w_fatigue *= preference_profile.fatigue_weight
        w_switches *= preference_profile.context_switch_weight
        w_deadline *= preference_profile.deadline_risk_weight
        w_frag *= preference_profile.fragmentation_weight

    weights = np.array([w_fatigue, w_switches, w_deadline, w_frag])

    obj_matrix = np.array([c.objectives.as_list() for c in valid], dtype=float)
    col_min = obj_matrix.min(axis=0)
    col_max = obj_matrix.max(axis=0)
    col_range = np.where(col_max - col_min > 1e-9, col_max - col_min, 1.0)
    normalised = (obj_matrix - col_min) / col_range

    scores = (normalised * weights).sum(axis=1)
    return valid[int(np.argmin(scores))]


def run_nsga(
    tasks: List[Task],
    base_calendar: List[Resource],
    user_context: UserContext,
    config: Optional[NSGAConfig] = None,
    preference_profile=None,
) -> Tuple[Optional[ScheduleResult], str, List[CandidateSchedule], List[Resource]]:
    """Run NSGA-II and return the best schedule.

    Returns (best_schedule, note, all_candidates, effective_calendar).
    Falls back to CP-SAT if NSGA produces no valid schedules.
    """
    if config is None:
        config = NSGAConfig()

    effective_calendar = apply_context(base_calendar, user_context)

    # CP-SAT preflight: confirm feasibility exists
    cpsat_result = cpsat_note = cpsat_cal = None
    if config.use_cpsat_fallback:
        cpsat_result, cpsat_note, cpsat_cal = solve_cpsat(
            tasks, base_calendar, user_context, debug=False, time_limit=10.0,
        )
        if cpsat_result is None:
            note = (
                "CP-SAT preflight: no feasible schedule found. "
                + (cpsat_note or "Check deadlines and capacity.")
            )
            return None, note, [], effective_calendar

    # Run NSGA-II
    problem = SchedulingNSGAProblem(
        tasks=tasks, calendar=effective_calendar,
        user_context=user_context, max_penalty=config.max_invalid_penalty,
    )
    algorithm = NSGA2(pop_size=config.population_size)
    termination = get_termination("n_gen", config.generations)

    try:
        pymoo_minimize(problem, algorithm, termination, seed=config.seed, verbose=False)
    except Exception as exc:
        warnings.warn(f"solver_nsga: NSGA-II exception: {exc}", stacklevel=2)

    all_candidates: List[CandidateSchedule] = list(problem._cache.values())
    valid_candidates = [c for c in all_candidates if c.valid]

    # Fallback to CP-SAT if no valid NSGA candidates
    if not valid_candidates:
        if config.use_cpsat_fallback and cpsat_result is not None:
            note = (
                "NSGA-II produced no valid schedules. "
                "Falling back to CP-SAT feasibility solution. "
                + (cpsat_note or "")
            )
            cpsat_objectives = evaluate_objectives(cpsat_result, tasks, cpsat_cal, user_context)
            fallback = CandidateSchedule(
                result=cpsat_result, objectives=cpsat_objectives,
                raw_x=[], valid=True, note="CP-SAT fallback",
                calendar=cpsat_cal,
            )
            return cpsat_result, note, [fallback], cpsat_cal

        return None, "NSGA-II produced no valid schedules.", [], effective_calendar

    # Select best from Pareto front
    best = select_pareto_solution(valid_candidates, user_context, preference_profile)
    if best is None:
        return None, "Pareto selection found no valid candidate.", all_candidates, effective_calendar

    # Final validation
    final_valid = validate_solution(
        best.result, tasks, debug=False, calendar=effective_calendar
    )
    if not final_valid:
        note = "Selected Pareto candidate failed validation. Returning CP-SAT fallback."
        if config.use_cpsat_fallback and cpsat_result is not None:
            return cpsat_result, note, all_candidates, cpsat_cal
        return None, note, all_candidates, effective_calendar

    n_valid = len(valid_candidates)
    n_total = len(all_candidates)
    note = (
        f"NSGA-II: {n_valid} valid from {n_total} evaluated. "
        f"Objectives: fatigue={best.objectives.fatigue:.1f}, "
        f"switches={best.objectives.context_switches:.0f}, "
        f"deadline_risk={best.objectives.deadline_risk:.2f}, "
        f"fragmentation={best.objectives.fragmentation:.0f}."
    )
    return best.result, note, all_candidates, effective_calendar
