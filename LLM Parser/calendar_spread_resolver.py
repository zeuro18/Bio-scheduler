"""
Spread resolver: maps spread_days -> effort_hours using calendar capacity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from models import Resource


def resolve_spread(
    tasks: List[Dict[str, Any]],
    infeasible: List[Dict[str, Any]],
    effective_calendar: List[Resource],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Fill ``effort_hours`` for tasks that only have ``spread_days`` set.

    Deadlines are not modified. Tasks whose resolved effort exceeds the
    deadline are appended to ``infeasible`` and omitted from the feasible list.
    """
    warnings: List[str] = []
    DEFAULT_HOURS_PER_DAY = 5.0
    daily_capacity = [r.capacity for r in effective_calendar]

    still_feasible: List[Dict[str, Any]] = []

    for task in tasks:
        spread = task.get("spread_days")

        if spread is None or task.get("effort_hours") is not None:
            task["spread_days"] = None
            still_feasible.append(task)
            continue

        spread_n = int(spread)
        deadline = float(task["deadline_hours"])
        max_days = max(1, int(deadline // 24))

        if spread_n > max_days:
            warnings.append(
                f"Task '{task['name']}': spread_days={spread_n} exceeds "
                f"days before deadline ({max_days}). Clamped to {max_days}."
            )
            spread_n = max_days

        effort = 0.0
        fallback_days = 0
        for day_idx in range(spread_n):
            if day_idx < len(daily_capacity):
                effort += daily_capacity[day_idx]
            else:
                effort += DEFAULT_HOURS_PER_DAY
                fallback_days += 1

        if fallback_days:
            warnings.append(
                f"Task '{task['name']}': calendar has only {len(daily_capacity)} days "
                f"but spread_days={spread_n}. Used {DEFAULT_HOURS_PER_DAY}h/day "
                f"fallback for {fallback_days} day(s)."
            )

        task["effort_hours"] = round(effort, 2)
        task["spread_days"] = None
        a = task.get("assumptions")
        if isinstance(a, list):
            task["assumptions"] = a
        elif a is None:
            task["assumptions"] = []
        else:
            task["assumptions"] = [str(a)]
        task["assumptions"].append(
            f"effort_hours={task['effort_hours']}h resolved from "
            f"spread_days={spread} using calendar capacity"
        )
        task["confidence"] = min(float(task.get("confidence", 1.0)), 0.8)

        if effort > deadline:
            reason = (
                f"resolved effort_hours={effort}h (from spread_days={spread}) "
                f"exceeds deadline_hours={deadline}h. "
                f"Deadline is immutable — reduce spread or split the task."
            )
            task["infeasibility_reason"] = reason
            infeasible.append(task)
            warnings.append(
                f"Task '{task['name']}' is INFEASIBLE after spread resolution: {reason}"
            )
        else:
            still_feasible.append(task)

    return still_feasible, infeasible, warnings
