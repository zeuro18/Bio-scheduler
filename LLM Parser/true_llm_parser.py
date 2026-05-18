from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import time
import os
import sys
from dotenv import load_dotenv
from groq import Groq

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))
load_dotenv(os.path.join(current_dir, ".env"), override=True)
from models import Resource, Task
from context import UserContext, apply_context
from postprocessing import postprocess
from BUILDPROMPT import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, build_user_message
from calendar_spread_resolver import resolve_spread


@dataclass
class ParsedTask:
    id: str
    name: str
    effort_hours: float | None
    spread_days: float | None
    deadline_hours: float
    priority: int
    cognitive_weight: int
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 1.0
    assumptions: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedTask":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            effort_hours=(
                None
                if data.get("effort_hours") is None
                else float(data["effort_hours"])
            ),
            spread_days=(
                None if data.get("spread_days") is None else float(data["spread_days"])
            ),
            deadline_hours=float(data["deadline_hours"]),
            priority=int(data.get("priority", 5)),
            cognitive_weight=int(data.get("cognitive_weight", 5)),
            dependencies=list(data.get("dependencies") or []),
            confidence=float(data.get("confidence", 1.0)),
            assumptions=list(data.get("assumptions") or []),
            notes=str(data.get("notes") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "effort_hours": self.effort_hours,
            "spread_days": self.spread_days,
            "deadline_hours": self.deadline_hours,
            "priority": self.priority,
            "cognitive_weight": self.cognitive_weight,
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "notes": self.notes,
        }

    def needs_spread_resolution(self) -> bool:
        return self.effort_hours is None and self.spread_days is not None

    def to_task(self) -> Task | None:
        if self.effort_hours is None:
            return None
        return Task(
            id=self.id,
            name=self.name,
            duration=self.effort_hours,
            deadline=self.deadline_hours,
            priority=self.priority,
            cognitive_weight=self.cognitive_weight,
            dependencies=list(self.dependencies),
        )


@dataclass
class ParseResult:
    tasks: List[Task] = field(default_factory=list)
    parsed_tasks: List[ParsedTask] = field(default_factory=list)
    infeasible_tasks: List[Dict[str, Any]] = field(default_factory=list)
    user_context: UserContext = field(default_factory=UserContext)
    clarification_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def task_dicts(self) -> List[Dict[str, Any]]:
        return [task.to_dict() for task in self.parsed_tasks]


DEFAULT_MODEL = "llama-3.3-70b-versatile"


class _GroqClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = (
            api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
        )
        if not key:
            raise ValueError(
                "Set GROQ_API_KEY or groq_api_key in .env (project root or LLM Parser) "
                "or pass api_key=."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def complete(self, user_text: str) -> str:
        user_message = build_user_message(user_text)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n" + FEW_SHOT_EXAMPLES,
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


def _extract_json_array(raw_text: str) -> list:
    """Pull the first top-level JSON array from model text (handles ```json fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.I).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM output")
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("Unclosed JSON array in LLM output")


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.I).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise ValueError("Unclosed JSON object in LLM output")


def _normalize_llm_payload(data: Any) -> Dict[str, Any]:
    if isinstance(data, list):
        return {
            "tasks": data,
            "user_context": {},
            "clarification_questions": [],
        }
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if tasks is None:
            tasks = []
        if not isinstance(tasks, list):
            tasks = [tasks]
        return {
            "tasks": tasks,
            "user_context": data.get("user_context") or {},
            "clarification_questions": list(data.get("clarification_questions") or []),
        }
    raise TypeError(f"LLM JSON must be list or dict, got {type(data).__name__}")


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.I).strip()
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    if cleaned.startswith("["):
        return _normalize_llm_payload(_extract_json_array(raw_text))
    if cleaned.startswith("{"):
        return _normalize_llm_payload(_extract_json_object(raw_text))
    raise ValueError("LLM output must start with '[' or '{' after stripping fences")


def _dicts_to_parsed_tasks(
    feasible: List[Dict[str, Any]],
) -> Tuple[List[ParsedTask], List[str]]:
    parsed_tasks: List[ParsedTask] = []
    task_warnings: List[str] = []
    for data in feasible:
        try:
            parsed_tasks.append(ParsedTask.from_dict(data))
        except (KeyError, TypeError, ValueError) as exc:
            task_warnings.append(
                f"Task '{data.get('name', '?')}': could not build ParsedTask ({exc})"
            )
    return parsed_tasks, task_warnings


def _parsed_tasks_to_tasks(
    parsed_tasks: List[ParsedTask],
) -> Tuple[List[Task], List[str]]:
    tasks: List[Task] = []
    task_warnings: List[str] = []
    for parsed_task in parsed_tasks:
        task = parsed_task.to_task()
        if task is None:
            if parsed_task.needs_spread_resolution():
                continue
            task_warnings.append(
                f"Task '{parsed_task.name}': missing effort_hours - skipped."
            )
            continue
        tasks.append(task)
    return tasks, task_warnings


def resolve_spread_on_result(
    result: ParseResult,
    base_calendar: List[Resource],
) -> ParseResult:
    """
    Resolve spread-based task effort using the caller-provided calendar.

    ``NLTaskParser.parse()`` deliberately stops before spread resolution when no
    calendar is supplied, so callers can size the real calendar from parsed
    deadlines first and then call this helper.
    """
    all_warnings = [
        w
        for w in result.warnings
        if w != "Spread-based tasks are pending calendar-based effort resolution."
    ]
    effective_calendar = apply_context(base_calendar, result.user_context)

    feasible, infeasible, spread_warnings = resolve_spread(
        [task.to_dict() for task in result.parsed_tasks],
        [dict(t) for t in result.infeasible_tasks],
        effective_calendar,
    )
    all_warnings.extend(spread_warnings)

    for nd in feasible:
        conf = float(nd.get("confidence", 1.0))
        for assumption in nd.get("assumptions") or []:
            msg = f"[conf={conf:.2f}] '{nd['name']}': {assumption}"
            if msg not in all_warnings:
                all_warnings.append(msg)

    parsed_tasks, parse_warnings = _dicts_to_parsed_tasks(feasible)
    all_warnings.extend(parse_warnings)
    tasks, task_warnings = _parsed_tasks_to_tasks(parsed_tasks)
    all_warnings.extend(task_warnings)

    return ParseResult(
        tasks=tasks,
        parsed_tasks=parsed_tasks,
        infeasible_tasks=infeasible,
        user_context=result.user_context,
        clarification_questions=result.clarification_questions,
        warnings=all_warnings,
    )


class NLTaskParser:
    """
    Parse plain-English input into a ``ParseResult`` (tasks, context, warnings).

    Parameters
    ----------
    api_key
        Groq API key; if empty, uses ``GROQ_API_KEY`` or ``groq_api_key`` from the environment.
    model
        Groq model id (default Mixtral 8x7B).
    max_retries
        Retries on bad JSON or transient failures.
    verbose
        If True, prints raw Groq output to stdout.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        max_retries: int = 2,
        verbose: bool = False,
    ):
        key = (
            api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
        )
        if not key:
            raise ValueError(
                "api_key is required (or set GROQ_API_KEY / groq_api_key in .env).\n"
                "Get a free key at https://console.groq.com"
            )
        self.max_retries = max_retries
        self.verbose = verbose
        self._client = _GroqClient(api_key=key, model=model or DEFAULT_MODEL)

    def parse(
        self,
        user_text: str,
        base_calendar: List[Resource] | None = None,
    ) -> ParseResult:
        all_warnings: List[str] = []

        raw_object = self._call_with_retry(user_text)
        pp = postprocess(raw_object)
        all_warnings.extend(pp.warnings)
        parsed_tasks, parse_warnings = _dicts_to_parsed_tasks(pp.tasks)
        all_warnings.extend(parse_warnings)
        tasks, task_warnings = _parsed_tasks_to_tasks(parsed_tasks)
        all_warnings.extend(task_warnings)

        parse_result = ParseResult(
            tasks=tasks,
            parsed_tasks=parsed_tasks,
            infeasible_tasks=pp.infeasible,
            user_context=pp.user_context,
            clarification_questions=pp.clarification_questions,
            warnings=all_warnings,
        )

        if base_calendar is not None:
            return resolve_spread_on_result(parse_result, base_calendar)

        for parsed_task in parsed_tasks:
            for assumption in parsed_task.assumptions:
                all_warnings.append(
                    f"[conf={parsed_task.confidence:.2f}] "
                    f"'{parsed_task.name}': {assumption}"
                )

        if any(task.needs_spread_resolution() for task in parsed_tasks):
            all_warnings.append(
                "Spread-based tasks are pending calendar-based effort resolution."
            )

        parse_result.warnings = all_warnings
        return parse_result

    def _call_with_retry(self, user_text: str) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw_text = self._client.complete(user_text)
                if self.verbose:
                    print(f"[Groq raw — attempt {attempt}]\n{raw_text}\n")
                return _parse_llm_json(raw_text)
            except Exception as exc:
                last_exc = exc
                if self.verbose:
                    print(f"[Attempt {attempt} failed: {exc}]")
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        raise RuntimeError(
            f"Groq parsing failed after {self.max_retries} attempts: {last_exc}"
        )
