"""
run_e2e.py - Plain English -> LLM Parse -> CP-SAT / NSGA-II Schedule

Usage:
    python run_e2e.py "Thesis 8h due in 5 days; lab tomorrow 2-5pm; I'm tired"
    python run_e2e.py --optimizer cpsat "..."
    python run_e2e.py --feedback "too tiring" "..."
    python run_e2e.py                              # interactive mode
    python run_e2e.py -v "..."                     # verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PARSER_DIR = os.path.join(_ROOT, "LLM Parser")
for _p in (_PARSER_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_PARSER_DIR, ".env"), override=True)

from context import UserContext
from display import print_schedule
from models import Task, generate_dynamic_calendar
from objectives import ObjectiveVector, evaluate_objectives
from preferences import PreferenceProfile, load_preferences, save_preferences, update_preferences_from_feedback
from solver_cpsat import solve_cpsat
from true_llm_parser import NLTaskParser, ParseResult, resolve_spread_on_result
from validate import validate_solution


def _task_to_dict(t: Task) -> Dict[str, Any]:
    return {
        "id": t.id, "name": t.name,
        "duration_hours": t.duration, "deadline_hours": t.deadline,
        "priority": t.priority, "cognitive_weight": t.cognitive_weight,
        "dependencies": t.dependencies,
    }


def _context_to_dict(ctx: UserContext) -> Dict[str, Any]:
    d = asdict(ctx)
    d["preferred_task_ids"] = sorted(ctx.preferred_task_ids)
    d["avoided_task_ids"] = sorted(ctx.avoided_task_ids)
    return d


def print_parsed(result: ParseResult) -> None:
    payload = {
        "tasks": [_task_to_dict(t) for t in result.tasks],
        "pending_task_dicts": [
            task.to_dict() for task in result.parsed_tasks if task.effort_hours is None
        ],
        "infeasible_tasks": result.infeasible_tasks,
        "user_context": _context_to_dict(result.user_context),
        "clarification_questions": result.clarification_questions,
        "warnings": result.warnings,
    }
    print("\nPARSED (sent to solver)\n")
    print(json.dumps(payload, indent=2, default=str))


def print_objective_summary(obj: ObjectiveVector) -> None:
    sep = "-" * 64
    print(f"\n{sep}")
    print("Objective Vector")
    print(sep)
    print(f"  Fatigue overload  : {obj.fatigue:.2f}  (lower = less cognitive overload)")
    print(f"  Context switches  : {obj.context_switches:.0f}  (lower = better focus)")
    print(f"  Deadline risk     : {obj.deadline_risk:.2f}  (lower = more breathing room)")
    print(f"  Fragmentation     : {obj.fragmentation:.0f}  (lower = tasks more consolidated)")
    print(sep)


def _calendar_seed_tasks(result: ParseResult) -> List[Task]:
    """Build lightweight tasks so calendar sizing can use parsed deadlines."""
    seed_tasks = list(result.tasks)
    existing_ids = {t.id for t in seed_tasks}
    for parsed_task in result.parsed_tasks:
        if parsed_task.id in existing_ids:
            continue
        seed_tasks.append(
            Task(
                id=parsed_task.id, name=parsed_task.name,
                duration=1.0, deadline=parsed_task.deadline_hours,
                priority=parsed_task.priority,
                cognitive_weight=parsed_task.cognitive_weight,
                dependencies=list(parsed_task.dependencies),
            )
        )
    return seed_tasks


def _run_cpsat_path(tasks, base_calendar, user_context, verbose):
    """Run CP-SAT only. Returns exit code."""
    print("\nSOLVING (CP-SAT)")

    schedule, relaxation_note, effective_cal = solve_cpsat(
        tasks, base_calendar, user_context, debug=verbose,
    )

    if schedule is None:
        print("\n[FAIL] No feasible schedule found (even after relaxation).")
        return 1

    if relaxation_note:
        print(f"\n{relaxation_note}")

    print_schedule(schedule, tasks, effective_cal, title="CP-SAT Schedule")
    ok = validate_solution(schedule, tasks, debug=verbose, calendar=effective_cal)
    if not ok:
        print("\n[FAIL] Validation failed -- schedule violates hard constraints.")
        return 1

    return 0


def _run_nsga_path(
    tasks, base_calendar, user_context, verbose,
    generations, population, preferences_path, feedback_text, use_preferences,
):
    """Run NSGA-II multi-objective optimisation. Returns exit code."""
    try:
        from solver_nsga import NSGAConfig, run_nsga
    except ImportError as exc:
        print(f"\n[FAIL] Cannot import solver_nsga: {exc}")
        print("  Install pymoo with:  pip install pymoo")
        return 1

    # Load and update preferences
    profile: Optional[PreferenceProfile] = None

    if use_preferences and preferences_path:
        profile = load_preferences(preferences_path)
        if verbose:
            print(f"\n[PREFS] Loaded v{profile.version} from '{preferences_path}'")

        if feedback_text:
            profile, explanations = update_preferences_from_feedback(profile, feedback_text)
            sep = "-" * 64
            print(f"\n{sep}")
            print("Preference Update from Feedback")
            print(sep)
            for exp in explanations:
                print(f"  * {exp}")
            if not explanations:
                print("  (no changes -- feedback keywords not matched)")
            print(sep)
            save_preferences(profile, preferences_path)
    elif feedback_text and not use_preferences:
        print("\n[PREFS] --feedback provided but --no-preferences is set; feedback ignored.")

    # Run NSGA-II
    print("\nSOLVING (NSGA-II)")
    config = NSGAConfig(
        population_size=population, generations=generations,
        use_cpsat_fallback=True,
    )

    best_schedule, note, all_candidates, effective_cal = run_nsga(
        tasks, base_calendar, user_context,
        config=config, preference_profile=profile,
    )

    n_valid = sum(1 for c in all_candidates if c.valid)
    n_total = len(all_candidates)
    print(f"\nNSGA-II evaluated {n_total} candidates, {n_valid} valid.")

    if best_schedule is None:
        print(f"\n[FAIL] {note}")
        return 1

    if note:
        print(f"\n{note}")

    print_schedule(best_schedule, tasks, effective_cal, title="NSGA-II Best Schedule")

    obj = evaluate_objectives(best_schedule, tasks, effective_cal, user_context)
    print_objective_summary(obj)

    # Pareto summary
    if all_candidates:
        print(f"\nPareto candidates ({n_valid} valid out of {n_total} total):")
        shown = [c for c in all_candidates if c.valid][:8]
        for i, c in enumerate(shown, 1):
            o = c.objectives
            print(
                f"  #{i}: fatigue={o.fatigue:.1f} switches={o.context_switches:.0f} "
                f"risk={o.deadline_risk:.2f} frag={o.fragmentation:.0f}"
            )
        if n_valid > 8:
            print(f"  ...and {n_valid - 8} more valid candidates.")

    # Validate (excluding explicitly unscheduled tasks)
    unscheduled_ids = {u.task_id for u in best_schedule.unscheduled}
    validatable_tasks = [t for t in tasks if t.id not in unscheduled_ids]
    ok = validate_solution(best_schedule, validatable_tasks, debug=verbose, calendar=effective_cal)
    if not ok:
        print("\n[FAIL] Final validation failed -- schedule violates hard constraints.")
        return 1

    return 0


def run_schedule(
    user_text: str,
    verbose: bool = False,
    optimizer: str = "nsga",
    generations: int = 80,
    population: int = 80,
    feedback_text: Optional[str] = None,
    preferences_path: Optional[str] = "preferences.json",
    use_preferences: bool = True,
) -> int:
    """Full pipeline: parse -> solve -> validate -> print. Returns exit code."""
    print("\nINPUT")
    print(user_text.strip())

    parser = NLTaskParser(verbose=verbose)
    parsed = parser.parse(user_text)

    if parsed.clarification_questions:
        print("\nClarifications needed:")
        for q in parsed.clarification_questions:
            print(f"  - {q}")

    calendar_seed_tasks = _calendar_seed_tasks(parsed)
    if not calendar_seed_tasks:
        print("\nNo feasible tasks to schedule.")
        return 1

    base_calendar = generate_dynamic_calendar(calendar_seed_tasks)
    parsed = resolve_spread_on_result(parsed, base_calendar)
    print_parsed(parsed)

    if not parsed.tasks:
        print("\nNo feasible tasks to schedule.")
        return 1

    tasks = parsed.tasks
    user_context = parsed.user_context

    if optimizer == "cpsat":
        return _run_cpsat_path(tasks, base_calendar, user_context, verbose)
    else:
        return _run_nsga_path(
            tasks, base_calendar, user_context,
            verbose=verbose, generations=generations, population=population,
            preferences_path=preferences_path, feedback_text=feedback_text,
            use_preferences=use_preferences,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bio-aware task scheduler: plain English -> optimised schedule.",
    )

    ap.add_argument("text", nargs="?",
                    help="Task description in plain English (omit for interactive input).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show raw LLM JSON and solver debug output.")

    opt_group = ap.add_mutually_exclusive_group()
    opt_group.add_argument("--optimizer", choices=["cpsat", "nsga"], default="nsga",
                           metavar="{cpsat,nsga}",
                           help="Optimizer: 'nsga' (default) or 'cpsat' (feasibility only).")
    opt_group.add_argument("--cpsat-only", action="store_true",
                           help="Alias for --optimizer cpsat.")

    ap.add_argument("--generations", type=int, default=80,
                    help="NSGA-II generation count (default: 80).")
    ap.add_argument("--population", type=int, default=80,
                    help="NSGA-II population size (default: 80).")

    ap.add_argument("--feedback", metavar="TEXT", default=None,
                    help="Free-text feedback about the previous schedule.")
    ap.add_argument("--preferences", metavar="PATH", default="preferences.json",
                    help="Path to preference profile JSON (default: preferences.json).")
    ap.add_argument("--no-preferences", action="store_true",
                    help="Disable preference loading/saving for this run.")

    args = ap.parse_args()

    optimizer = "cpsat" if args.cpsat_only else args.optimizer

    if args.no_preferences:
        preferences_path = None
        use_preferences = False
    else:
        preferences_path = args.preferences
        use_preferences = True

    user_text = args.text
    if not user_text:
        print("Enter your tasks / context (end with a blank line):\n")
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip() and lines:
                break
            lines.append(line)
        user_text = "\n".join(lines).strip()

    if not user_text:
        print("No input provided.")
        sys.exit(1)

    sys.exit(
        run_schedule(
            user_text, verbose=args.verbose, optimizer=optimizer,
            generations=args.generations, population=args.population,
            feedback_text=args.feedback, preferences_path=preferences_path,
            use_preferences=use_preferences,
        )
    )


if __name__ == "__main__":
    main()
