"""
test_parser.py
--------------
Manual test suite for the LLM task parser.
No external test framework needed — just run:

    python test_parser.py

You must have GROQ_API_KEY set in your environment:
    export GROQ_API_KEY="your_key_here"

Or pass it directly (see bottom of file).
"""

import os
import sys
from typing import List

from dotenv import load_dotenv

# LLM Parser dir (local imports) + project root (models, context, …)
_parser_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_parser_dir, ".."))
for _p in (_parser_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local .env wins over a stale machine-wide GROQ_API_KEY during dev runs
load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv(os.path.join(_parser_dir, ".env"), override=True)

from models import Task
from true_llm_parser import NLTaskParser


def _load_api_key() -> str:
    raw = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key") or ""
    return raw.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ------------------------------------------------------------------
    # 1. Single task — all fields explicit
    # ------------------------------------------------------------------
    {
        "id": "single_explicit",
        "description": "Single task with all fields stated clearly",
        "input": (
            "I need to write my thesis. It will take 8 hours. "
            "The deadline is in 100 hours from now. Priority is 9 out of 10. "
            "Cognitive load is 8 out of 10."
        ),
        "expect": {
            "count": 1,
            "tasks": [
                {
                    "name_contains": "thesis",
                    "duration_hours": 8.0,
                    "deadline_hours": 100.0,
                    "priority": 9,
                    "cognitive_weight": 8,
                    "dependency_count": 0,
                }
            ],
        },
    },
    # ------------------------------------------------------------------
    # 2. Two tasks with a dependency
    # ------------------------------------------------------------------
    {
        "id": "two_tasks_dependency",
        "description": "Two tasks where the second depends on the first",
        "input": (
            "First, complete the SMA paper. It takes 4 hours, deadline in 72 hours, "
            "priority 7, cognitive weight 3. "
            "Second, work on the Design Lab project — 5 hours, deadline in 120 hours, "
            "priority 6, cognitive weight 6. I can only start it after the SMA paper is done."
        ),
        "expect": {
            "count": 2,
            "tasks": [
                {
                    "name_contains": "sma",
                    "duration_hours": 4.0,
                    "deadline_hours": 72.0,
                    "priority": 7,
                    "cognitive_weight": 3,
                    "dependency_count": 0,
                },
                {
                    "name_contains": "design lab",
                    "duration_hours": 5.0,
                    "deadline_hours": 120.0,
                    "priority": 6,
                    "cognitive_weight": 6,
                    "dependency_count": 1,  # depends on SMA paper
                },
            ],
        },
    },
    # ------------------------------------------------------------------
    # 3. Days → hours conversion
    # ------------------------------------------------------------------
    {
        "id": "days_to_hours",
        "description": "Deadline given in days — parser must convert to hours",
        "input": (
            "Prepare for the deep learning test. "
            "It needs 10 hours of study. The exam is in 8 days. "
            "This is my top priority (10/10) and very cognitively demanding (9/10)."
        ),
        "expect": {
            "count": 1,
            "tasks": [
                {
                    "name_contains": "deep learning",
                    "duration_hours": 10.0,
                    "deadline_hours": 192.0,  # 8 × 24
                    "priority": 10,
                    "cognitive_weight": 9,
                    "dependency_count": 0,
                }
            ],
        },
    },
    # ------------------------------------------------------------------
    # 4. Three tasks, fan of dependencies, low cog weight
    # ------------------------------------------------------------------
    {
        "id": "three_tasks_chain",
        "description": "Three tasks: two independents + one that needs both done first",
        "input": (
            "Task 1: Magnetism term paper. 6 hours, deadline in 200 hours, "
            "priority 5, cognitive weight 4. "
            "Task 2: Literature review. 3 hours, deadline in 80 hours, "
            "priority 6, cognitive weight 3. "
            "Task 3: Final report. 4 hours, deadline in 250 hours, priority 8, "
            "cognitive weight 5. It depends on both the magnetism term paper "
            "and the literature review."
        ),
        "expect": {
            "count": 3,
            "tasks": [
                {
                    "name_contains": "magnetism",
                    "dependency_count": 0,
                },
                {
                    "name_contains": "literature",
                    "dependency_count": 0,
                },
                {
                    "name_contains": "final report",
                    "dependency_count": 2,
                },
            ],
        },
    },
    # ------------------------------------------------------------------
    # 5. Very short task, max priority, no dependencies
    # ------------------------------------------------------------------
    {
        "id": "quick_high_priority",
        "description": "A short urgent task with no dependencies",
        "input": (
            "Email the professor about the deadline extension. "
            "Takes 0.5 hours. Must be done within 2 hours. "
            "Priority 10. Cognitive weight 1."
        ),
        "expect": {
            "count": 1,
            "tasks": [
                {
                    "name_contains": "email",
                    "duration_hours": 0.5,
                    "deadline_hours": 2.0,
                    "priority": 10,
                    "cognitive_weight": 1,
                    "dependency_count": 0,
                }
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TOLERANCE = 0.10  # 10% relative tolerance for numeric fields


def _close(actual: float, expected: float, tol: float = TOLERANCE) -> bool:
    if expected == 0:
        return abs(actual) < 0.5
    return abs(actual - expected) / abs(expected) <= tol


def run_one_test(parser: NLTaskParser, case: dict) -> bool:
    """Return True if the test passes."""
    print(f"\n{'─'*60}")
    print(f"TEST: {case['id']}")
    print(f"  {case['description']}")
    print(f"  Input: {case['input'][:80]}{'...' if len(case['input'])>80 else ''}")

    try:
        result = parser.parse(case["input"])
        tasks = result.tasks
        errors = result.warnings
    except Exception as exc:
        msg = str(exc)
        print(f"  ✗ EXCEPTION: {exc}")
        if "invalid_api_key" in msg or "Invalid API Key" in msg:
            print(
                "  → Groq rejected the API key. Update LLM Parser/.env with a fresh key from\n"
                "    https://console.groq.com/keys  (no quotes). If tests still fail, check\n"
                "    whether Windows has an old key:  echo $env:GROQ_API_KEY"
            )
        return False

    if errors:
        for e in errors:
            print(f"  ⚠  {e}")

    exp = case["expect"]
    passed = True

    # Check task count
    if len(tasks) != exp["count"]:
        print(f"  ✗ Got {len(tasks)} task(s), expected {exp['count']}")
        passed = False
    else:
        print(f"  ✓ Task count = {len(tasks)}")

    # Check individual task fields (by position)
    for i, exp_task in enumerate(exp.get("tasks", [])):
        if i >= len(tasks):
            print(f"  ✗ Task {i} missing")
            passed = False
            continue

        t: Task = tasks[i]
        ok = True

        if "name_contains" in exp_task:
            if exp_task["name_contains"].lower() not in t.name.lower():
                print(
                    f"  ✗ Task {i} name '{t.name}' should contain "
                    f"'{exp_task['name_contains']}'"
                )
                ok = False

        for field in ("duration_hours", "deadline_hours"):
            attr = field.replace("_hours", "")  # duration / deadline
            if field in exp_task:
                actual = getattr(t, attr, None)
                if actual is None or not _close(actual, exp_task[field]):
                    print(f"  ✗ Task {i} {attr}={actual}, expected ~{exp_task[field]}")
                    ok = False

        for field in ("priority", "cognitive_weight"):
            if field in exp_task:
                actual = getattr(t, field, None)
                if actual != exp_task[field]:
                    print(f"  ✗ Task {i} {field}={actual}, expected {exp_task[field]}")
                    ok = False

        if "dependency_count" in exp_task:
            actual_deps = len(t.dependencies)
            if actual_deps != exp_task["dependency_count"]:
                print(
                    f"  ✗ Task {i} has {actual_deps} dep(s), "
                    f"expected {exp_task['dependency_count']} "
                    f"(deps={t.dependencies})"
                )
                ok = False

        if ok:
            print(f"  ✓ Task {i} '{t.name}' — all checked fields match")
        else:
            passed = False

    return passed


def main():
    api_key = _load_api_key()
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY not found.\n"
            "  Create LLM Parser/.env with:\n"
            "    GROQ_API_KEY=your_key_here\n"
            "  Get a free key at https://console.groq.com/keys"
        )
        sys.exit(1)

    parser = NLTaskParser(
        api_key=api_key,
        verbose=False,  # set True to see raw LLM output
    )

    results = []
    for case in TEST_CASES:
        passed = run_one_test(parser, case)
        results.append((case["id"], passed))

    print(f"\n{'═'*60}")
    print("RESULTS")
    print(f"{'═'*60}")
    total = len(results)
    passed_n = sum(1 for _, p in results if p)
    for tid, p in results:
        icon = "✓" if p else "✗"
        print(f"  {icon}  {tid}")
    print(f"\n  {passed_n}/{total} tests passed")

    if passed_n < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
