"""
Converts the raw dict returned by the LLM into:
  - List[dict]   normalised task dicts  (effort_hours may still be None for spread tasks)
  - UserContext  extracted from the user_context block
  - List[str]    clarification questions surfaced by the LLM
  - List[str]    warnings about skipped/adjusted fields

DOES NOT build Task objects yet, that happens in llm_parser.py after
calendar_spread_resolver has filled in any missing effort_hours.

DOES NOT mutate deadlines ever. If effort > deadline the task is
flagged as INFEASIBLE and excluded from the returned task list.
The caller receives it in a separate infeasible list.

NORMALISED TASK DICT SCHEMA
{
  "id"              : str,
  "name"            : str,
  "effort_hours"    : float | None,   # None = spread_days not yet resolved
  "spread_days"     : float | None,
  "deadline_hours"  : float,          # immutable — never changed here
  "priority"        : int,            # 1-10
  "cognitive_weight": int,            # 1-10
  "dependencies"    : List[str],      # resolved task IDs
  "confidence"      : float,
  "assumptions"     : List[str],
  "notes"           : str,
}
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Set, Tuple, TypeVar
from context import UserContext


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


T = TypeVar("T")


def _clamp(value: Any, lo: int, hi: int, default: T) -> int | T:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_match(query: str, candidates: List[str]) -> str | None:
    if not candidates:
        return None
    scored = [
        (c, SequenceMatcher(None, query.lower(), c.lower()).ratio()) for c in candidates
    ]
    best_name, best_score = max(scored, key=lambda x: x[1])
    return best_name if best_score >= 0.7 else None


def _parse_user_context(raw_ctx: Dict[str, Any]) -> UserContext:
    """
    Convert the raw user_context dict from the LLM into a UserContext object.
    Every field is optional, missing or null fields use UserContext defaults.
    """
    capacity_override: Dict[str, float] = {}
    raw_cap = raw_ctx.get("capacity_override") or {}
    if isinstance(raw_cap, dict):
        for day_id, hours in raw_cap.items():
            try:
                capacity_override[str(day_id)] = float(hours)
            except (TypeError, ValueError):
                pass

    blocked_hours: List[tuple] = []
    raw_blocked = raw_ctx.get("blocked_hours") or []
    if isinstance(raw_blocked, list):
        for entry in raw_blocked:
            # Entry is [day_id, start, end]
            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                try:
                    blocked_hours.append(
                        (str(entry[0]), float(entry[1]), float(entry[2]))
                    )
                except (TypeError, ValueError):
                    pass

    energy_raw = raw_ctx.get("energy_level")
    energy_level = _clamp(energy_raw, 1, 10, None) if energy_raw is not None else None

    dpm = raw_ctx.get("deadline_pressure_mode")
    deadline_pressure_mode = bool(dpm) if dpm is not None else False

    dpi = raw_ctx.get("deadline_pressure_intensity")
    valid_intensities = {"mild", "moderate", "extreme"}
    deadline_pressure_intensity = str(dpi) if dpi in valid_intensities else "moderate"

    preferred_raw = raw_ctx.get("preferred_task_ids") or []
    preferred_task_ids: Set[str] = (
        set(preferred_raw) if isinstance(preferred_raw, list) else set()
    )

    avoided_raw = raw_ctx.get("avoided_task_ids") or []
    avoided_task_ids: Set[str] = (
        set(avoided_raw) if isinstance(avoided_raw, list) else set()
    )

    notes = str(raw_ctx.get("notes") or "").strip()
    date = str(raw_ctx.get("date") or "").strip()

    return UserContext(
        capacity_override=capacity_override,
        blocked_hours=blocked_hours,
        energy_level=energy_level,
        deadline_pressure_mode=deadline_pressure_mode,
        deadline_pressure_intensity=deadline_pressure_intensity,
        preferred_task_ids=preferred_task_ids,
        avoided_task_ids=avoided_task_ids,
        notes=notes,
        date=date,
    )


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class PostprocessResult:
    """Structured output of postprocess()."""

    __slots__ = (
        "tasks",
        "infeasible",
        "user_context",
        "clarification_questions",
        "warnings",
    )

    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        infeasible: List[Dict[str, Any]],
        user_context: UserContext,
        clarification_questions: List[str],
        warnings: List[str],
    ):
        self.tasks = tasks
        self.infeasible = infeasible  # effort > deadline
        self.user_context = user_context
        self.clarification_questions = clarification_questions
        self.warnings = warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def postprocess(raw: Dict[str, Any]) -> PostprocessResult:
    """
    Normalise the full LLM output dict.

    Parameters
    ----------
    raw : the dict from json.loads() of the LLM response.
          Expected keys: "tasks", "user_context", "clarification_questions"

    Returns
    -------
    PostprocessResult with fields:
      .tasks                   — normalised task dicts, feasible only
      .infeasible              — task dicts where effort_hours > deadline_hours
                                 (deadline NOT mutated — caller decides what to do)
      .user_context            — UserContext object
      .clarification_questions — list of strings from LLM
      .warnings                — processing warnings
    """
    warnings: List[str] = []
    normalised: List[Dict[str, Any]] = []
    infeasible: List[Dict[str, Any]] = []
    name_to_id: Dict[str, str] = {}

    raw_tasks = raw.get("tasks") or []
    raw_ctx = raw.get("user_context") or {}
    clarifications = [str(q) for q in (raw.get("clarification_questions") or [])]

    # ------------------------------------------------------------------
    # Pass 1 — normalise every task dict, build name->id map
    # ------------------------------------------------------------------
    for i, task in enumerate(raw_tasks):
        name = str(task.get("name") or f"task_{i}").strip()
        task_id = _slugify(name)
        if task_id in name_to_id.values():
            task_id = f"{task_id}_{i}"
        name_to_id[name] = task_id

        effort = _to_float_or_none(task.get("effort_hours"))
        spread = _to_float_or_none(task.get("spread_days"))
        deadline = _to_float(task.get("deadline_hours") or task.get("deadline"), 0.0)
        priority = _clamp(task.get("priority", 5), 1, 10, 5)
        cog = _clamp(task.get("cognitive_weight", 5), 1, 10, 5)
        deps_raw = task.get("dependencies") or []
        conf = max(0.0, min(1.0, _to_float(task.get("confidence", 0.8), 0.8)))
        assump = list(task.get("assumptions") or [])
        notes = str(task.get("notes") or "").strip()

        # Enforce mutual exclusivity
        if effort is not None and spread is not None:
            warnings.append(
                f"Task '{name}': LLM set both effort_hours and spread_days. "
                f"Keeping effort_hours={effort}, clearing spread_days."
            )
            spread = None

        if deadline <= 0:
            warnings.append(f"Task '{name}': deadline={deadline} is invalid — skipped.")
            continue

        normalised.append(
            {
                "id": task_id,
                "name": name,
                "effort_hours": effort,
                "spread_days": spread,
                "deadline_hours": deadline,  # immutable from here on
                "priority": priority,
                "cognitive_weight": cog,
                "deps_raw": deps_raw,
                "confidence": conf,
                "assumptions": assump,
                "notes": notes,
            }
        )

    all_names = list(name_to_id.keys())

    # ------------------------------------------------------------------
    # Pass 2 — resolve dependency names -> IDs, split feasible/infeasible
    # ------------------------------------------------------------------
    feasible: List[Dict[str, Any]] = []

    for nd in normalised:
        # Resolve deps
        resolved: List[str] = []
        deps_raw = nd.pop("deps_raw", [])
        if not isinstance(deps_raw, list):
            deps_raw = []
        for dep_name in deps_raw:
            match = _best_match(str(dep_name), all_names)
            if match:
                resolved.append(name_to_id[match])
            else:
                warnings.append(
                    f"Task '{nd['name']}': dependency '{dep_name}' "
                    f"matched nothing in this batch — dropped."
                )
        nd["dependencies"] = resolved

        # Feasibility check — deadlines are IMMUTABLE
        effort = nd["effort_hours"]
        if effort is not None and effort > nd["deadline_hours"]:
            nd["infeasibility_reason"] = (
                f"effort_hours={effort}h exceeds deadline_hours={nd['deadline_hours']}h. "
                f"Deadline is immutable — reduce scope or split the task."
            )
            infeasible.append(nd)
            warnings.append(
                f"Task '{nd['name']}' is INFEASIBLE: {nd['infeasibility_reason']}"
            )
        else:
            feasible.append(nd)

    # ------------------------------------------------------------------
    # Parse user context
    # ------------------------------------------------------------------
    user_context = _parse_user_context(raw_ctx)

    return PostprocessResult(
        tasks=feasible,
        infeasible=infeasible,
        user_context=user_context,
        clarification_questions=clarifications,
        warnings=warnings,
    )
