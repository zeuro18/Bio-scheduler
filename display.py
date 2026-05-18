"""
display.py - Schedule Display Utilities

Shared pretty-printer for any ScheduleResult (greedy, CP-SAT, NSGA-II).

Usage:
    from display import print_schedule
    print_schedule(result, tasks, calendar, title="CP-SAT")
"""

from __future__ import annotations
from typing import List, Tuple
from models import Resource, ScheduleResult, Task


def format_hour_24(hour_value: float) -> str:
    """Convert a fractional hour (e.g. 14.5) to 'HH:MM' string."""
    total_minutes = int(round(hour_value * 60))
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def describe_time_slot(
    day: Resource, start_hour_abs: float, end_hour_abs: float
) -> str:
    """Return slot label for an allocation inside a day."""
    start_local = start_hour_abs % 24
    end_local = end_hour_abs % 24

    for slot_start, slot_end in day.work_slots:
        if start_local >= slot_start and end_local <= slot_end:
            return f"{format_hour_24(slot_start)}-{format_hour_24(slot_end)}"

    return f"{format_hour_24(start_local)}-{format_hour_24(end_local)}"


def _day_sort_key(resource_id: str) -> int:
    """Extract numeric index from 'day_2', 'day_10' etc. for chronological sorting."""
    try:
        return int(resource_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


_Block = Tuple[str, str, float, float, float]


def _merge_consecutive_blocks(entries: list) -> List[_Block]:
    """Collapse adjacent 1-hour entries for the same task/day into one display row.

    Returns list of (task_id, resource_id, total_hours, start_abs, end_abs).
    """
    if not entries:
        return []

    sorted_entries = sorted(
        entries,
        key=lambda e: (_day_sort_key(e.resource_id), e.start_time or 0.0, e.task_id),
    )

    blocks: List[_Block] = []

    for entry in sorted_entries:
        start = entry.start_time or 0.0
        end = start + entry.allocated_hours

        if (
            blocks
            and blocks[-1][0] == entry.task_id
            and blocks[-1][1] == entry.resource_id
            and abs(blocks[-1][4] - start) < 1e-6
        ):
            tid, rid, hrs, blk_start, _ = blocks[-1]
            blocks[-1] = (tid, rid, hrs + entry.allocated_hours, blk_start, end)
        else:
            blocks.append(
                (entry.task_id, entry.resource_id, entry.allocated_hours, start, end)
            )

    return blocks


def print_schedule(
    result: ScheduleResult,
    tasks: List[Task],
    calendar: List[Resource],
    title: str = "Schedule",
) -> None:
    """Pretty-print a ScheduleResult as a day-by-day timetable.

    Consecutive same-task entries on the same day are merged into one row.
    """
    task_map = {t.id: t.name for t in tasks}
    resource_map = {r.id: r for r in calendar}

    print(f"  WEEKLY SCHEDULE ({title})\n")

    if not result.scheduled:
        print("  (no tasks scheduled)")
        print()
        return

    blocks = _merge_consecutive_blocks(result.scheduled)

    current_rid = None
    for task_id, resource_id, total_hours, start_abs, end_abs in blocks:
        day = resource_map.get(resource_id)
        if day is None:
            continue

        if resource_id != current_rid:
            current_rid = resource_id
            print(f"\n  {day.name}")
            print("  " + "-" * 42)

        start_local = start_abs % 24
        end_local = end_abs % 24
        time_range = f"{format_hour_24(start_local)}-{format_hour_24(end_local)}"
        slot_label = describe_time_slot(day, start_abs, end_abs)

        task_name = task_map.get(task_id, task_id)
        print(
            f"   - {task_name:.<25s} "
            f"{total_hours:>4.1f}h  "
            f"{time_range}  "
            f"[slot {slot_label}]"
        )

    if result.unscheduled:
        print(f"\n  UNSCHEDULED TASKS")
        for u in result.unscheduled:
            print(
                f"\n   X {u.task_name:.<25s} {u.remaining_hours:>4.1f}h remaining  ({u.reason})"
            )

    print(f"\n  DAY-BY-DAY UTILISATION")

    sorted_calendar = sorted(calendar, key=lambda r: _day_sort_key(r.id))
    for day in sorted_calendar:
        hours_used = result.resource_usage.get(day.id, 0.0)
        cap = day.capacity
        bar_filled = int(hours_used)
        bar_empty = max(0, int(cap) - bar_filled)
        bar = "#" * bar_filled + "-" * bar_empty
        pct = (hours_used / cap * 100) if cap > 0 else 0
        print(f"  {day.name:<20s}  {bar}  {hours_used:.1f}/{cap:.0f}h ({pct:.0f}%)")

    print()
