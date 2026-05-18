"""
hour_index.py - Calendar Hour Indexing

Maps calendar days and work slots to discrete absolute hours.

'Absolute hour' means: day_idx * 24 + local_hour_within_day
Example:  day_0 slot 9-12  -> absolute hours 9, 10, 11
          day_1 slot 9-12  -> absolute hours 33, 34, 35

Only hours that fall inside a Resource's work_slots are indexed.
Hours beyond any task's deadline are excluded too (pruning).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from models import Resource, Task


def _local_hour_in_work_slots(
    local_h: int,
    work_slots: Tuple[Tuple[float, float], ...],
) -> bool:
    # True if integer local_h is a valid discrete hour start inside some slot.
    for slot_start, slot_end in work_slots:
        lo = int(math.ceil(slot_start))
        hi = int(math.floor(slot_end))
        if lo <= local_h < hi:
            return True
    return False


@dataclass
class HourIndex:
    """
    Maps every valid work hour to a unique integer.

    'Absolute hour' means: day_idx * 24 + local_hour_within_day
    Example:  day_0 slot 9-12  →  absolute hours 9, 10, 11
              day_1 slot 9-12  →  absolute hours 33, 34, 35

    Only hours that fall inside a Resource's work_slots are indexed.
    Hours beyond any task's deadline are excluded too (pruning).
    """

    valid_hours: List[int]  # sorted list of all valid absolute hours
    hour_to_day: Dict[int, int]  # absolute_hour → day_idx
    day_to_hours: Dict[int, List[int]]  # day_idx → [absolute_hours on that day]
    max_hour: int  # largest valid absolute hour

    @classmethod
    def build(cls, calendar: List[Resource], tasks: List[Task]) -> "HourIndex":
        """Scan ``calendar`` work slots and build the hour index shared by model and extraction.

        Uses ``ceil`` for slot starts and ``floor`` for slot ends so fractional boundaries
        (e.g. 14.5) do not create an illegal hour **before** the slot opens.
        """
        max_deadline = max(t.deadline for t in tasks)

        valid_hours: List[int] = []
        hour_to_day: Dict[int, int] = {}
        day_to_hours: Dict[int, List[int]] = {}

        for day_idx, day in enumerate(calendar):
            day_start = day_idx * 24
            day_to_hours[day_idx] = []

            for slot_start, slot_end in day.work_slots:
                h_lo = int(math.ceil(day_start + slot_start))
                h_hi = int(math.floor(day_start + slot_end))
                for h in range(h_lo, h_hi):
                    if h >= max_deadline:  # prune: no variable would satisfy deadlines
                        continue
                    valid_hours.append(h)
                    hour_to_day[h] = day_idx
                    day_to_hours[day_idx].append(h)

        valid_hours.sort()
        return cls(
            valid_hours=valid_hours,
            hour_to_day=hour_to_day,
            day_to_hours=day_to_hours,
            max_hour=max(valid_hours) if valid_hours else 0,
        )

    def hours_for_task(self, task: Task) -> List[int]:
        """Return ``valid_hours`` that this ``task`` may use (subset by per-task deadline).

        Variables ``x[task, h]`` are omitted for ``h >= floor(deadline)`` so deadlines are
        enforced by domain restriction: the task cannot occupy an hour that would finish
        at or past ``task.deadline`` under the evaluator's hour-end convention.
        """
        deadline_h = int(math.floor(task.deadline))
        return [h for h in self.valid_hours if h < deadline_h]
