"""
preferences.py - Preference Learning (v2 LLM-based extraction)

Stores per-objective weight biases learned from user feedback.
These weights influence Pareto candidate selection in solver_nsga.py.
They do NOT change tasks, deadlines, dependencies, or any hard constraint.

v2: Uses structured LLM extraction for feedback instead of keyword matching.
    The LLM receives the previous schedule context + feedback text and
    returns structured weight adjustments with reasoning.

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
import re
import warnings
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

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


def _clamp(value: float) -> float:
    return max(_WEIGHT_MIN, min(_WEIGHT_MAX, value))


# ---------------------------------------------------------------------------
# LLM-based feedback extraction
# ---------------------------------------------------------------------------

_FEEDBACK_SYSTEM_PROMPT = """\
You are a scheduling preference analyzer. The user previously received a schedule
and is giving feedback about it. You must analyze their feedback and return
structured JSON preference adjustments.

Return ONLY a JSON object with these keys:

{
  "weight_adjustments": {
    "fatigue_weight": 0.0,
    "context_switch_weight": 0.0,
    "deadline_risk_weight": 0.0,
    "fragmentation_weight": 0.0
  },
  "flag_changes": {},
  "reasoning": []
}

RULES:
- weight_adjustments: float deltas in range [-0.3, +0.3]. Positive = care more about this.
  - fatigue_weight: cognitive load per day. Increase if user says it was too tiring/heavy.
  - context_switch_weight: task changes per day. Increase if too much jumping between tasks.
  - deadline_risk_weight: how close tasks finish to deadline. Increase if too risky/last minute.
  - fragmentation_weight: how spread out tasks are. Increase if tasks too scattered.
- flag_changes: optional overrides for boolean flags. Only include flags that should change.
  Valid flags: "prefer_long_blocks" (bool), "avoid_heavy_evening" (bool), "prefer_compact_schedule" (bool)
- reasoning: array of short explanations for each adjustment you made.

Use 0.0 for weights you don't want to change. Be conservative — small deltas (0.1-0.2)
unless the feedback is very strong.

Examples:
- "too tiring" → fatigue_weight: +0.2
- "too much switching" → context_switch_weight: +0.2
- "I want deep work blocks" → context_switch_weight: +0.15, prefer_long_blocks: true
- "evenings should be lighter" → avoid_heavy_evening: true, fatigue_weight: +0.1
- "too relaxed, I can handle more" → fatigue_weight: -0.15
- "tasks too scattered" → fragmentation_weight: +0.2, prefer_compact_schedule: true
"""


def _extract_json_from_llm(text: str) -> dict:
    """Pull JSON object from LLM response, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.I).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    return {}


def _call_feedback_llm(
    feedback_text: str,
    schedule_summary: str = "",
) -> dict:
    """Call the LLM to extract structured preference updates from feedback."""
    try:
        from dotenv import load_dotenv
        from groq import Groq

        _root = os.path.dirname(os.path.abspath(__file__))
        _parser_dir = os.path.join(_root, "LLM Parser")
        load_dotenv(os.path.join(_root, ".env"))
        load_dotenv(os.path.join(_parser_dir, ".env"), override=True)

        key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
        if not key:
            return {}

        client = Groq(api_key=key)

        user_content = ""
        if schedule_summary:
            user_content += f"Previous schedule:\n{schedule_summary}\n\n"
        user_content += f"User feedback: {feedback_text}\n\nReturn ONLY the JSON object."

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = response.choices[0].message.content
        if not content:
            return {}
        return _extract_json_from_llm(content.strip())

    except Exception as exc:
        warnings.warn(f"preferences: LLM feedback call failed: {exc}", stacklevel=2)
        return {}


# ---------------------------------------------------------------------------
# Keyword fallback (used when LLM is unavailable)
# ---------------------------------------------------------------------------

_KEYWORD_RULES: List[Tuple[List[str], str, float]] = [
    (["too tiring", "burnt out", "too heavy", "exhausting", "worn out"], "fatigue_weight", +0.20),
    (["too much switching", "couldn't focus", "jumping around", "hard to focus"], "context_switch_weight", +0.20),
    (["too close to deadline", "last minute", "risky", "cutting it close"], "deadline_risk_weight", +0.20),
    (["too scattered", "fragmented", "spread out", "all over the place"], "fragmentation_weight", +0.20),
    (["too relaxed", "too easy", "too light", "I can handle more"], "fatigue_weight", -0.10),
]


def _keyword_fallback(
    profile: PreferenceProfile,
    feedback_text: str,
) -> Tuple[PreferenceProfile, List[str]]:
    """Simple keyword matching fallback when LLM is unavailable."""
    text = feedback_text.lower()
    explanations: List[str] = []

    for phrases, field_name, delta in _KEYWORD_RULES:
        if any(phrase in text for phrase in phrases):
            current = float(getattr(profile, field_name))
            setattr(profile, field_name, _clamp(current + delta))
            direction = "Increased" if delta > 0 else "Decreased"
            explanations.append(f"{direction} {field_name} by {abs(delta):.2f}")

    if explanations:
        profile.version += 1

    return profile, explanations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_preferences_from_feedback(
    profile: PreferenceProfile,
    feedback_text: str,
    schedule_summary: str = "",
) -> Tuple[PreferenceProfile, List[str]]:
    """Update profile from user feedback using LLM extraction.

    Falls back to keyword matching if the LLM call fails.
    Returns (updated_profile, explanations).
    """
    llm_result = _call_feedback_llm(feedback_text, schedule_summary)

    if not llm_result:
        return _keyword_fallback(profile, feedback_text)

    explanations: List[str] = []

    # Apply weight adjustments
    adjustments = llm_result.get("weight_adjustments") or {}
    for field_name in ("fatigue_weight", "context_switch_weight",
                       "deadline_risk_weight", "fragmentation_weight"):
        delta = adjustments.get(field_name, 0.0)
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            continue
        if abs(delta) > 0.01:
            current = float(getattr(profile, field_name))
            setattr(profile, field_name, _clamp(current + delta))

    # Apply flag changes
    flag_changes = llm_result.get("flag_changes") or {}
    for flag_name in ("prefer_long_blocks", "avoid_heavy_evening", "prefer_compact_schedule"):
        if flag_name in flag_changes:
            try:
                setattr(profile, flag_name, bool(flag_changes[flag_name]))
            except (TypeError, ValueError):
                pass

    # Collect reasoning
    reasoning = llm_result.get("reasoning") or []
    if isinstance(reasoning, list):
        explanations.extend(str(r) for r in reasoning)

    if explanations:
        profile.version += 1

    return profile, explanations
