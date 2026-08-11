from models import Resource
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Circadian fitness curves: local_hour → cognitive fitness (0.0–1.0)
# Based on chronobiology research on alertness across the day.
# ---------------------------------------------------------------------------

CIRCADIAN_CURVES: Dict[str, Dict[int, float]] = {
    "lark": {
        7: 0.6, 8: 0.9, 9: 1.0, 10: 1.0, 11: 0.9, 12: 0.7,
        13: 0.4, 14: 0.3, 15: 0.5, 16: 0.7, 17: 0.6,
        18: 0.4, 19: 0.3, 20: 0.2,
    },
    "neutral": {
        9: 0.8, 10: 1.0, 11: 1.0, 12: 0.7, 13: 0.4, 14: 0.3,
        15: 0.5, 16: 0.8, 17: 0.9, 18: 0.7, 19: 0.5,
        20: 0.4, 21: 0.3,
    },
    "owl": {
        9: 0.3, 10: 0.4, 11: 0.5, 12: 0.6, 13: 0.5, 14: 0.4,
        15: 0.5, 16: 0.7, 17: 0.8, 18: 0.9, 19: 1.0,
        20: 1.0, 21: 0.9, 22: 0.8, 23: 0.6,
    },
}

VALID_CHRONOTYPES = {"lark", "neutral", "owl"}


def circadian_score(chronotype: str, local_hour: int) -> float:
    """Return cognitive fitness (0.0–1.0) for a chronotype at a given hour."""
    curve = CIRCADIAN_CURVES.get(chronotype, CIRCADIAN_CURVES["neutral"])
    return curve.get(local_hour, 0.5)


@dataclass
class UserContext:
    # time availability
    capacity_override: Dict[str, float] = field(default_factory=dict)
    blocked_hours: List[tuple] = field(default_factory=list)

    energy_level: Optional[int] = None  # 1 to 10 scale. 1 being the most fatigued
    chronotype: str = "neutral"  # "lark", "neutral", or "owl"
    deadline_pressure_mode: bool = False
    deadline_pressure_intensity: str = "moderate"
    preferred_task_ids: Set[str] = field(default_factory=set)
    avoided_task_ids: Set[str] = field(default_factory=set)

    notes: str = ""  # exact words of the user, so it can be kept for reference
    date: str = ""  # date of the context


def apply_context(base_cal: List[Resource], context: UserContext) -> List[Resource]:
    effective = []
    for resource in base_cal:
        r_id = resource.id
        # Start with base values
        new_capacity = resource.capacity
        new_work_slots = resource.work_slots

        # capacity override
        if r_id in context.capacity_override:
            new_capacity = context.capacity_override[r_id]

        # Apply deadline pressure (extends capacity/slots)
        if context.deadline_pressure_mode:
            # Extend evening work slot based on intensity
            intensity_map = {
                "mild": 1.0,
                "moderate": 2.0,
                "extreme": 4.0,
            }
            extension = intensity_map.get(context.deadline_pressure_intensity, 2.0)

            # Extend the last work slot's end time
            if new_work_slots:
                extended_slots = list(new_work_slots)
                last_start, last_end = extended_slots[-1]
                extended_slots[-1] = (last_start, min(24.0, last_end + extension))
                new_work_slots = tuple(extended_slots)

                # Also increase capacity to match the extension
                new_capacity += extension

        # Subtract blocked hours from work_slots

        blocked_on_this_day = [
            (start, end)
            for day_id, start, end in context.blocked_hours
            if day_id == r_id
        ]
        final_slots = new_work_slots
        if blocked_on_this_day:
            final_slots = carve_out_blocked(new_work_slots, blocked_on_this_day)
            new_capacity = min(
                new_capacity, sum(end - start for start, end in final_slots)
            )

        effective.append(
            Resource(
                id=r_id,
                name=resource.name,
                capacity=new_capacity,
                work_slots=final_slots,
                weekday=resource.weekday,
            )
        )

    return effective


def apply_profile(
    base_cal: List[Resource],
    profile,  # UserProfile (imported lazily to avoid circular import)
) -> List[Resource]:
    """Apply persistent profile to base calendar.

    - Carves out recurring weekly blocks (gym, lab, lectures)
    - Replaces default work slots with custom per-weekday slots if defined

    Runs BEFORE apply_context (profile = permanent, context = transient overlay).
    """
    effective = []
    for resource in base_cal:
        wd = resource.weekday
        new_slots = resource.work_slots
        new_capacity = resource.capacity

        # Apply custom work slots for this weekday
        wd_key = str(wd)
        if wd_key in profile.custom_work_slots:
            custom = profile.custom_work_slots[wd_key]
            if custom:
                new_slots = tuple(tuple(s) for s in custom)
                new_capacity = sum(e - s for s, e in new_slots)

        # Carve out recurring blocks for this weekday
        blocks_today = [
            (b.start_hour, b.end_hour)
            for b in profile.recurring_blocks
            if b.weekday == wd
        ]
        if blocks_today:
            new_slots = carve_out_blocked(new_slots, blocks_today)
            new_capacity = min(
                new_capacity, sum(e - s for s, e in new_slots)
            )

        effective.append(
            Resource(
                id=resource.id,
                name=resource.name,
                capacity=new_capacity,
                work_slots=new_slots,
                weekday=wd,
            )
        )

    return effective


def default_context() -> UserContext:
    return UserContext()


def tired_context() -> UserContext:
    return UserContext(
        capacity_override={"day_0": 3.0},
        energy_level=3,
        notes="I'm exhausted today, only have 3 hours in me",
    )


def crunch_time_context() -> UserContext:
    return UserContext(
        deadline_pressure_mode=True,
        deadline_pressure_intensity="extreme",
        notes="Thesis is due in 2 days, I'll pull an all-nighter if needed",
    )


def meeting_context() -> UserContext:
    return UserContext(
        blocked_hours=[
            ("day_0", 14.0, 16.0),
            ("day_1", 10.0, 11.5),
        ],
        notes="I have a 2-4pm meeting today and a 10:30am meeting tomorrow",
    )


def carve_out_blocked(
    work_slots: Tuple[Tuple[float, float], ...], blocked: List[Tuple[float, float]]
) -> Tuple[Tuple[float, float], ...]:
    current = list(work_slots)

    for b_start, b_end in blocked:
        if b_start >= b_end:
            continue
        next_slots = []
        for s_start, s_end in current:
            # Case 1: no overlap
            if b_end <= s_start or b_start >= s_end:
                next_slots.append((s_start, s_end))

            # Case 2: block fully covers this slot
            elif b_start <= s_start and b_end >= s_end:
                pass

            # Case 3: block cuts left side of slot
            elif b_start <= s_start < b_end < s_end:
                next_slots.append((b_end, s_end))

            # Case 4: block cuts right side of slot
            elif s_start < b_start < s_end <= b_end:
                next_slots.append((s_start, b_start))

            # Case 5: block is inside slot, split into two
            else:
                next_slots.append((s_start, b_start))
                next_slots.append((b_end, s_end))

        current = next_slots

    return tuple(current)


if __name__ == "__main__":
    from models import generate_test_data

    tasks, base_calendar = generate_test_data()
    print("BASE CALENDAR (first 3 days)")

    for i, day in enumerate(base_calendar[:3]):
        print(f"{day.name:<25} capacity={day.capacity:.1f}h  slots={day.work_slots}")

    print("CONTEXT 1: Tired (reduced capacity on day 0)")

    context1 = tired_context()
    effective1 = apply_context(base_calendar, context1)
    for i, day in enumerate(effective1[:3]):
        print(f"{day.name:<25} capacity={day.capacity:.1f}h  slots={day.work_slots}")

    print("CONTEXT 2: Crunch time (deadline pressure, extended evening)")
    context2 = crunch_time_context()
    effective2 = apply_context(base_calendar, context2)
    for i, day in enumerate(effective2[:3]):
        print(f"{day.name:<25} capacity={day.capacity:.1f}h  slots={day.work_slots}")

    print("CONTEXT 3: Meetings (blocked hours reduce capacity)")

    context3 = meeting_context()
    effective3 = apply_context(base_calendar, context3)
    for i, day in enumerate(effective3[:3]):
        print(f"{day.name:<25} capacity={day.capacity:.1f}h  slots={day.work_slots}")

    print("\n All contexts applied successfully")
