"""
objectives.py - Schedule Quality Objective Functions

Hard constraints (deadlines, no-overlap, dependencies, capacity, work slots)
are enforced by CP-SAT and the decoder in solver_nsga.py. These functions
ONLY compare valid schedules -- they measure quality, not feasibility.

Intended for NSGA-II multi-objective optimisation and Pareto selection.

All functions accept a ScheduleResult plus the original task list and
calendar. Unknown task IDs are warned and skipped, never crash.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from context import UserContext
from models import Resource, ScheduleResult, Task

INVALID_PENALTY: float = 1e6


@dataclass(frozen=True)
class ObjectiveVector:
    """Four-dimensional objective vector for Pareto comparison.
    Lower is better for all objectives.
    """

    fatigue: float
    context_switches: float
    deadline_risk: float
    fragmentation: float

    def as_list(self) -> List[float]:
        return [self.fatigue, self.context_switches, self.deadline_risk, self.fragmentation]

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.fatigue, self.context_switches, self.deadline_risk, self.fragmentation)


def compute_task_time_bounds(result: ScheduleResult) -> Dict[str, Dict]:
    """Per-task scheduling bounds from a ScheduleResult.

    Handles both CP-SAT one-hour entries and greedy multi-hour chunks.

    Returns dict mapping task_id to:
        start  - earliest start_time across all entries
        finish - latest (start_time + allocated_hours)
        hours  - total allocated hours
        days   - set of resource_ids used
    """
    bounds: Dict[str, Dict] = {}
    for entry in result.scheduled:
        tid = entry.task_id
        s = float(entry.start_time or 0.0)
        alloc = float(entry.allocated_hours or 0.0)
        f = s + alloc

        if tid not in bounds:
            bounds[tid] = {"start": s, "finish": f, "hours": alloc, "days": {entry.resource_id}}
        else:
            rec = bounds[tid]
            rec["start"] = min(rec["start"], s)
            rec["finish"] = max(rec["finish"], f)
            rec["hours"] = rec["hours"] + alloc
            rec["days"].add(entry.resource_id)

    return bounds


def compute_daily_fatigue(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
) -> Dict[str, float]:
    """Cognitive fatigue per calendar day.

    fatigue[resource_id] = sum(allocated_hours * task.cognitive_weight)
    Unknown task IDs are warned once and skipped.
    """
    task_map = {t.id: t for t in tasks}
    fatigue: Dict[str, float] = {r.id: 0.0 for r in calendar}
    warned_ids: Set[str] = set()

    for entry in result.scheduled:
        task = task_map.get(entry.task_id)
        if task is None:
            if entry.task_id not in warned_ids:
                warnings.warn(
                    f"compute_daily_fatigue: unknown task_id '{entry.task_id}', skipping",
                    stacklevel=2,
                )
                warned_ids.add(entry.task_id)
            continue
        rid = entry.resource_id
        alloc = float(entry.allocated_hours or 0.0)
        fatigue[rid] = fatigue.get(rid, 0.0) + alloc * task.cognitive_weight

    return fatigue


def compute_fatigue_objective(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
    user_context: Optional[UserContext] = None,
) -> float:
    """Scalar fatigue penalty.

    threshold = 20 + 2 * energy_level  (if set), else 35.
    objective = sum(max(0, fatigue[d] - threshold)^2) + 0.01 * max_fatigue

    Low-energy users get a lower threshold so moderate cognitive load is
    penalised. High-energy users tolerate more. This is a soft preference,
    not a hard constraint.
    """
    daily_fatigue = compute_daily_fatigue(result, tasks, calendar)

    energy = None
    if user_context is not None:
        energy = user_context.energy_level

    if energy is not None:
        threshold = 20.0 + 2.0 * float(energy)
    else:
        threshold = 35.0

    overload_sum = 0.0
    max_fatigue = 0.0
    for fat in daily_fatigue.values():
        max_fatigue = max(max_fatigue, fat)
        overload_sum += max(0.0, fat - threshold) ** 2

    return overload_sum + 0.01 * max_fatigue


def compute_context_switch_objective(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
) -> float:
    """Count task-changes within the same calendar day.

    Entries sorted by (resource_id, start_time). Each consecutive block
    belonging to a different task on the same day counts as one switch.
    Overnight transitions are not counted.
    """
    day_entries: Dict[str, List] = {}
    for entry in result.scheduled:
        day_entries.setdefault(entry.resource_id, []).append(entry)

    switches = 0
    for rid, entries in day_entries.items():
        entries_sorted = sorted(entries, key=lambda e: float(e.start_time or 0.0))
        prev_task_id: Optional[str] = None
        for entry in entries_sorted:
            if prev_task_id is not None and entry.task_id != prev_task_id:
                switches += 1
            prev_task_id = entry.task_id

    return float(switches)


def compute_deadline_risk_objective(
    result: ScheduleResult,
    tasks: List[Task],
) -> float:
    """Priority-weighted deadline breathing room.

    For each scheduled task: contribution = priority / (slack + 1)
    where slack = deadline - finish_time. Negative slack (should not happen
    post-validation) gets INVALID_PENALTY.
    """
    bounds = compute_task_time_bounds(result)

    risk = 0.0
    for task in tasks:
        b = bounds.get(task.id)
        if b is None:
            continue
        finish = b["finish"]
        slack = task.deadline - finish
        if slack < 0:
            risk += INVALID_PENALTY
        else:
            risk += task.priority / (slack + 1.0)

    return risk


def compute_fragmentation_objective(
    result: ScheduleResult,
    tasks: List[Task],
) -> float:
    """Penalise spreading a task across many distinct days.

    fragmentation = sum(distinct days each task is worked on)
    Same-day multi-block allocation is not penalised here.
    """
    days_used: Dict[str, Set[str]] = {}
    for entry in result.scheduled:
        days_used.setdefault(entry.task_id, set()).add(entry.resource_id)

    return float(sum(len(days) for days in days_used.values()))


def evaluate_objectives(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
    user_context: Optional[UserContext] = None,
) -> ObjectiveVector:
    """Compute all four schedule-quality objectives.

    Hard constraints are NOT checked here -- call validate_solution() separately.
    """
    return ObjectiveVector(
        fatigue=compute_fatigue_objective(result, tasks, calendar, user_context),
        context_switches=compute_context_switch_objective(result, tasks, calendar),
        deadline_risk=compute_deadline_risk_objective(result, tasks),
        fragmentation=compute_fragmentation_objective(result, tasks),
    )
