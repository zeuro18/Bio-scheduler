"""
evaluate.py - Schedule Evaluation
Provides:
  • EvaluationResult — multi-metric evaluation of a schedule
  • evaluate_schedule() — compute metrics for any ScheduleResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from models import Resource, ScheduleResult, Task




@dataclass
class EvaluationResult:
    """Multi-metric evaluation of a schedule."""

    total_scheduled_tasks: int = 0
    total_unscheduled_tasks: int = 0
    deadline_miss_count: int = 0
    dependency_violation_count: int = 0
    peak_daily_fatigue: float = 0.0
    critical_path_pressure: float = 0.0
    feasibility_score: float = 0.0


def evaluate_schedule(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
) -> EvaluationResult:
    """
    Compute evaluation metrics for a ScheduleResult.

    Metrics:
      • total_scheduled / unscheduled tasks
      • deadline misses (last active hour + 1 > deadline)
      • dependency violations (child starts before parent finishes)
      • peak daily fatigue — max over all days of
            Σ (allocated_hours × task.cognitive_weight)
        cognitive_weight is on a 1-10 scale, so a 3h deep-learning block
        (weight 9) scores 27 whereas 3h of admin (weight 2) scores only 6.
      • critical path pressure (fraction of deadline consumed)
      • feasibility score (0–1, higher is better)
    """
    task_map = {t.id: t for t in tasks}

    # Per-task hour sets & cognitive-load accumulation
    task_hours: Dict[str, List[float]] = {}
    # day_fatigue: Σ(allocated_hours × cognitive_weight) per resource/day
    day_fatigue: Dict[str, float] = {r.id: 0.0 for r in calendar}

    for entry in result.scheduled:
        task_hours.setdefault(entry.task_id, []).append(entry.start_time or 0.0)
        task = task_map.get(entry.task_id)
        cog = task.cognitive_weight if task is not None else 1
        day_fatigue[entry.resource_id] = (
            day_fatigue.get(entry.resource_id, 0.0) + entry.allocated_hours * cog
        )

    scheduled_ids = set(task_hours.keys())

    # Deadline misses────
    deadline_misses: List[str] = []
    for tid, hours in task_hours.items():
        if tid not in task_map:
            continue
        last_h = max(hours)
        if last_h + 1 > task_map[tid].deadline:
            deadline_misses.append(tid)

    # Dependency violations────
    dep_violations: List[str] = []
    task_finish = {tid: max(hrs) for tid, hrs in task_hours.items()}
    task_start = {tid: min(hrs) for tid, hrs in task_hours.items()}

    for t in tasks:
        for dep_id in t.dependencies:
            dep_end = task_finish.get(dep_id, -1)
            child_begin = task_start.get(t.id, 999_999)
            if child_begin <= dep_end:
                dep_violations.append(f"{t.id} before {dep_id}")

    # Peak daily fatigue (cognitive-load weighted)────
    peak_fatigue = max(day_fatigue.values()) if day_fatigue else 0.0

    # Critical path pressure────
    pressures: List[float] = []
    for t in tasks:
        if t.id in task_hours and task_hours[t.id]:
            last_h = max(task_hours[t.id])
            if t.deadline > 0:
                pressures.append((last_h + 1) / t.deadline)
    cp_pressure = max(pressures) if pressures else 0.0

    # Feasibility score (0-1 heuristic)────
    n = len(tasks)
    if n == 0:
        feas = 1.0
    else:
        sched_ratio = len(scheduled_ids) / n
        miss_penalty = len(deadline_misses) / n
        dep_penalty = len(dep_violations) / n
        feas = max(0.0, sched_ratio - miss_penalty - dep_penalty)

    return EvaluationResult(
        total_scheduled_tasks=len(scheduled_ids),
        total_unscheduled_tasks=len(result.unscheduled),
        deadline_miss_count=len(deadline_misses),
        dependency_violation_count=len(dep_violations),
        peak_daily_fatigue=peak_fatigue,
        critical_path_pressure=cp_pressure,
        feasibility_score=feas,
    )
