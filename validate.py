"""
validate.py  -  Independent Solution Validator
================================================
Re-checks every hard constraint independently of the CP-SAT solver.
Use this to catch bugs in the extraction step or to verify any
ScheduleResult (greedy, CP-SAT, NSGA-II, etc.).

Checks performed:
  1. Duration   - each task got ceil(duration) hours
  2. No overlap - at most 1 task per absolute hour
  3. Dependencies - child starts strictly after parent finishes
  4. Deadlines  - last active hour + 1 <= deadline
  5. Capacity   - daily usage <= floor(capacity)   (requires calendar)
  6. Work slots - each hour falls inside a valid work window (requires calendar)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from models import Resource, Task, ScheduleResult
from hour_index import _local_hour_in_work_slots


def validate_solution(
    result: ScheduleResult,
    tasks: List[Task],
    debug: bool = True,
    calendar: Optional[List[Resource]] = None,
) -> bool:
    """
    Re-checks every hard constraint independently of the solver.
    Use this to catch bugs in the extraction step.

    Args:
        result: Output of :func:`_extract_solution` or any compatible structure.
        tasks: Tasks that were supposed to be scheduled.
        debug: Print [VALIDATE] lines.
        calendar: Effective calendar matching the solve; omit to skip capacity/slot checks.

    Returns:
        ``True`` if no errors accumulated.
    """
    errors: List[str] = []
    day_by_id: Dict[str, int] = {}
    if calendar is not None:
        day_by_id = {r.id: i for i, r in enumerate(calendar)}

    # Reconstruct sparse schedule: absolute hour -> who is there (for overlap + deps).
    task_hours_used: Dict[str, Set[int]] = {t.id: set() for t in tasks}
    hour_occupancy: Dict[int, List[str]] = {}  # hour -> [task_ids]
    usage_by_day: Dict[int, float] = {}

    for entry in result.scheduled:
        h_abs = int(entry.start_time or 0)

        if calendar is not None:
            day_idx = day_by_id.get(entry.resource_id)
            if day_idx is None:
                errors.append(
                    f"Unknown resource_id '{entry.resource_id}' for task '{entry.task_id}'"
                )
            else:
                # ScheduledTask packs both resource and absolute hour; they must agree.
                expected_day_idx = h_abs // 24
                if expected_day_idx != day_idx:
                    errors.append(
                        f"resource_id '{entry.resource_id}' maps to day {day_idx}, "
                        f"but absolute hour {h_abs} maps to day {expected_day_idx} "
                        f"for task '{entry.task_id}'"
                    )
                usage_by_day[day_idx] = usage_by_day.get(day_idx, 0.0) + float(
                    entry.allocated_hours or 0.0
                )

        task_hours_used[entry.task_id].add(h_abs)
        hour_occupancy.setdefault(h_abs, []).append(entry.task_id)

    # CHECK 1: Duration
    for task in tasks:
        got = len(task_hours_used[task.id])
        expected = int(math.ceil(task.duration))
        if got != expected:
            errors.append(
                f"Duration mismatch '{task.id}': got {got}h, expected {expected}h"
            )

    # CHECK 2: No overlap
    for h, occupants in hour_occupancy.items():
        if len(occupants) > 1:
            errors.append(f"Overlap at hour {h}: {occupants}")

    # CHECK 3: Dependencies
    task_finish: Dict[str, int] = {
        t_id: (max(hrs) if hrs else -1) for t_id, hrs in task_hours_used.items()
    }
    task_start: Dict[str, int] = {
        t_id: (min(hrs) if hrs else 999_999) for t_id, hrs in task_hours_used.items()
    }
    for task in tasks:
        for dep_id in task.dependencies:
            dep_finish = task_finish.get(dep_id, -1)
            child_start = task_start.get(task.id, 999_999)
            if child_start <= dep_finish:
                errors.append(
                    f"Dependency violation: '{task.id}' starts at h{child_start} "
                    f"but '{dep_id}' finishes at h{dep_finish}"
                )

    # CHECK 4: Deadlines
    for task in tasks:
        if not task_hours_used[task.id]:
            continue
        last_h = max(task_hours_used[task.id])
        # Activity in hour last_h occupies [last_h, last_h+1) on continuous timeline.
        if last_h + 1 > task.deadline:
            errors.append(
                f"Deadline miss '{task.id}': last active hour {last_h} "
                f"ends at {last_h + 1}, deadline is {task.deadline}"
            )

    # CHECK 5-6: Capacity & work slots (optional, requires calendar)
    if calendar is not None:
        for day_idx, used in usage_by_day.items():
            if day_idx < 0 or day_idx >= len(calendar):
                errors.append(f"Scheduled entries on unknown day index {day_idx}")
                continue
            cap = int(math.floor(calendar[day_idx].capacity))
            if used > cap + 1e-6:
                errors.append(
                    f"Capacity violation day_{day_idx}: used {used:.1f}h, "
                    f"limit {cap}h"
                )

        for entry in result.scheduled:
            h_abs = int(entry.start_time or 0)
            day_idx = day_by_id.get(entry.resource_id, -1)
            if day_idx < 0:
                continue
            day = calendar[day_idx]
            local_h = h_abs % 24
            if not _local_hour_in_work_slots(local_h, day.work_slots):
                errors.append(
                    f"Work-slot violation: task '{entry.task_id}' at absolute h{h_abs} "
                    f"(local {local_h}) not in slots {day.work_slots} on {day.id}"
                )

    all_ok = len(errors) == 0

    if debug:
        print("\n[VALIDATE] Independent constraint check:")
        if all_ok:
            print("  [OK] All hard constraints satisfied")
        else:
            for err in errors:
                print(f"  [FAIL] {err}")

    return all_ok
