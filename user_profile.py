"""
user_profile.py - Persistent User Profile

Stores permanent facts about the user that don't change session-to-session:
  - chronotype (lark / neutral / owl)
  - recurring weekly commitments (gym, lab, lectures)
  - custom per-weekday work slots

Separate from:
  - UserContext  (transient: today's mood, energy, ad-hoc blocked hours)
  - PreferenceProfile  (learned: Pareto weight biases from feedback)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

from context import VALID_CHRONOTYPES


@dataclass
class RecurringBlock:
    """A weekly recurring commitment (blocked time)."""

    weekday: int       # 0=Mon, 1=Tue, ..., 6=Sun
    start_hour: float  # 14.0 = 2pm
    end_hour: float    # 17.0 = 5pm
    label: str = ""    # "gym", "lab", "lecture"


@dataclass
class UserProfile:
    """Persistent user facts, loaded from user_profile.json."""

    chronotype: str = "neutral"
    recurring_blocks: List[RecurringBlock] = field(default_factory=list)
    # weekday (int) → list of (start, end) work windows
    # If empty for a weekday, DEFAULT_WORK_SLOTS is used.
    custom_work_slots: Dict[str, List[List[float]]] = field(default_factory=dict)
    notes: str = ""


def load_profile(path: str = "user_profile.json") -> UserProfile:
    """Load from JSON.  Returns defaults if file is missing or malformed."""
    if not os.path.exists(path):
        return UserProfile()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return UserProfile()

    chrono = data.get("chronotype", "neutral")
    if chrono not in VALID_CHRONOTYPES:
        chrono = "neutral"

    blocks: List[RecurringBlock] = []
    for b in data.get("recurring_blocks") or []:
        try:
            blocks.append(RecurringBlock(
                weekday=int(b["weekday"]),
                start_hour=float(b["start_hour"]),
                end_hour=float(b["end_hour"]),
                label=str(b.get("label", "")),
            ))
        except (KeyError, TypeError, ValueError):
            pass

    custom_slots: Dict[str, List[List[float]]] = {}
    for wd_str, slots in (data.get("custom_work_slots") or {}).items():
        try:
            parsed = [[float(s), float(e)] for s, e in slots]
            custom_slots[str(wd_str)] = parsed
        except (TypeError, ValueError):
            pass

    return UserProfile(
        chronotype=chrono,
        recurring_blocks=blocks,
        custom_work_slots=custom_slots,
        notes=str(data.get("notes", "")),
    )


def save_profile(profile: UserProfile, path: str = "user_profile.json") -> None:
    """Write profile to JSON."""
    data = {
        "chronotype": profile.chronotype,
        "recurring_blocks": [asdict(b) for b in profile.recurring_blocks],
        "custom_work_slots": profile.custom_work_slots,
        "notes": profile.notes,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
