"""
preferences.py - Preference Learning (v1 keyword heuristic)

Stores per-objective weight biases learned from user feedback.
These weights influence Pareto candidate selection in solver_nsga.py.
They do NOT change tasks, deadlines, dependencies, or any hard constraint.

Workflow:
    1. Load PreferenceProfile from JSON (or start with defaults).
    2. If the user provides feedback, call update_preferences_from_feedback().
    3. Save the updated profile to JSON.
    4. Pass the profile to select_pareto_solution().

Over multiple sessions the profile adapts to the user's preferences.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

_WEIGHT_MIN: float = 0.5
_WEIGHT_MAX: float = 2.5


@dataclass
class PreferenceProfile:
    """Learned preference weights and flags for schedule selection.

    *_weight fields scale the corresponding objective during Pareto
    selection (higher weight = care more about that objective).
    Boolean flags encode structural preferences the decoder may use.
    version is incremented on each meaningful update.
    """

    fatigue_weight: float = 1.0
    context_switch_weight: float = 1.0
    deadline_risk_weight: float = 1.0
    fragmentation_weight: float = 1.0

    prefer_long_blocks: bool = False
    avoid_heavy_evening: bool = False
    prefer_compact_schedule: bool = False

    version: int = 1


def load_preferences(path: str = "preferences.json") -> PreferenceProfile:
    """Load from JSON. Returns defaults if file is missing or malformed."""
    if not os.path.exists(path):
        return PreferenceProfile()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return PreferenceProfile()

    profile = PreferenceProfile()
    for key, value in data.items():
        if hasattr(profile, key):
            try:
                setattr(profile, key, type(getattr(profile, key))(value))
            except (ValueError, TypeError):
                pass
    return profile


def save_preferences(profile: PreferenceProfile, path: str = "preferences.json") -> None:
    """Write profile to JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(profile), fh, indent=2)


# Keyword rules: (trigger_phrases, field_name, delta, explanation)
_RULES: List[Tuple[List[str], str, float, str]] = [
    (
        ["too tiring", "burnt out", "too heavy", "exhausting", "worn out", "too draining"],
        "fatigue_weight",
        +0.20,
        "Increased fatigue weight: schedule will prefer lower cognitive load per day.",
    ),
    (
        ["too much switching", "couldn't focus", "jumping around", "too fragmented mentally",
         "hard to focus", "constant interruptions", "kept switching"],
        "context_switch_weight",
        +0.20,
        "Increased context-switch weight: schedule will prefer longer uninterrupted blocks.",
    ),
    (
        ["too close to deadline", "last minute", "risky", "cutting it close",
         "almost missed", "not enough buffer", "deadline stress"],
        "deadline_risk_weight",
        +0.20,
        "Increased deadline-risk weight: schedule will leave more breathing room.",
    ),
    (
        ["too scattered", "fragmented", "spread out", "spread across too many days",
         "all over the place", "not consolidated"],
        "fragmentation_weight",
        +0.20,
        "Increased fragmentation weight: schedule will consolidate tasks into fewer days.",
    ),
    (
        ["too slow", "not enough done", "too relaxed", "too easy", "too light",
         "underloaded", "I can handle more"],
        "fatigue_weight",
        -0.10,
        "Slightly decreased fatigue weight: schedule will tolerate more work per day.",
    ),
    (
        ["prefer long blocks", "want longer sessions", "like long blocks",
         "fewer shorter sessions", "want deep work", "deep work"],
        "context_switch_weight",
        +0.15,
        "Increased context-switch weight and enabled long-block preference.",
    ),
]

_FLAG_RULES: List[Tuple[List[str], str, bool, str]] = [
    (
        ["prefer long blocks", "want longer sessions", "like long blocks",
         "deep work", "fewer shorter sessions"],
        "prefer_long_blocks", True, "Set prefer_long_blocks = True.",
    ),
    (
        ["compact", "pack it in", "prefer compact", "fit it all in one go"],
        "prefer_compact_schedule", True, "Set prefer_compact_schedule = True.",
    ),
    (
        ["avoid evening", "no evening work", "don't schedule evenings",
         "keep evenings free", "lighter evenings"],
        "avoid_heavy_evening", True, "Set avoid_heavy_evening = True.",
    ),
]

_FLAG_RESET_RULES: List[Tuple[List[str], str, bool, str]] = [
    (
        ["can work evenings", "evenings are fine", "okay to schedule evenings"],
        "avoid_heavy_evening", False, "Reset avoid_heavy_evening = False.",
    ),
    (
        ["short sessions are fine", "prefer shorter sessions", "break it up"],
        "prefer_long_blocks", False, "Reset prefer_long_blocks = False.",
    ),
]


def _clamp(value: float) -> float:
    return max(_WEIGHT_MIN, min(_WEIGHT_MAX, value))


def update_preferences_from_feedback(
    profile: PreferenceProfile,
    feedback_text: str,
) -> Tuple[PreferenceProfile, List[str]]:
    """Update profile weights from keyword analysis of feedback_text.

    Multiple rules can fire for a single message. All weight changes
    are clamped to [0.5, 2.5]. Returns (updated_profile, explanations).
    """
    text = feedback_text.lower()
    explanations: List[str] = []

    for phrases, field_name, delta, explanation in _RULES:
        if any(phrase in text for phrase in phrases):
            current = float(getattr(profile, field_name))
            setattr(profile, field_name, _clamp(current + delta))
            explanations.append(explanation)

    # "too slow" / "not enough done" also increases deadline pressure
    relax_phrases = ["too slow", "not enough done", "too relaxed", "too easy",
                     "too light", "underloaded", "I can handle more"]
    if any(phrase in text for phrase in relax_phrases):
        profile.deadline_risk_weight = _clamp(profile.deadline_risk_weight + 0.10)
        explanations.append(
            "Increased deadline-risk weight: schedule will prioritise completing tasks sooner."
        )

    for phrases, flag_name, flag_value, explanation in _FLAG_RULES:
        if any(phrase in text for phrase in phrases):
            if getattr(profile, flag_name) != flag_value:
                setattr(profile, flag_name, flag_value)
                explanations.append(explanation)

    for phrases, flag_name, flag_value, explanation in _FLAG_RESET_RULES:
        if any(phrase in text for phrase in phrases):
            if getattr(profile, flag_name) != flag_value:
                setattr(profile, flag_name, flag_value)
                explanations.append(explanation)

    if explanations:
        profile.version += 1

    return profile, explanations
