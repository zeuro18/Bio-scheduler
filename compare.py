"""
compare.py  —  Solver Comparison Utility
==========================================
Runs both greedy and CP-SAT solvers on the same input and prints
a side-by-side metric comparison using evaluate.py.

Useful for verifying CP-SAT improves on greedy, and for catching regressions.
"""

from __future__ import annotations

from typing import List

from context import UserContext, apply_context
from evaluate import evaluate_schedule
from models import Resource, Task, run_greedy_scheduler_structured
from solver_cpsat import solve_cpsat


def compare_with_greedy(
    tasks: List[Task],
    base_calendar: List[Resource],
    ctx: UserContext,
    debug: bool = True,
) -> None:
    """
    Run both solvers and print a side-by-side comparison using evaluate.py.
    Useful for verifying CP-SAT improves on greedy, and for catching regressions.
    """
    print("SOLVER COMPARISON: Greedy  vs  CP-SAT\n")

    effective_cal = apply_context(base_calendar, ctx)

    # Greedy
    greedy_result = run_greedy_scheduler_structured(tasks, effective_cal)
    greedy_eval = evaluate_schedule(greedy_result, tasks, effective_cal)

    # CP-SAT
    cpsat_result, note, cpsat_cal = solve_cpsat(tasks, base_calendar, ctx, debug=False)
    if cpsat_result:
        cpsat_eval = evaluate_schedule(cpsat_result, tasks, cpsat_cal)
    else:
        print("CP-SAT: no feasible solution found.")
        return

    # Print comparison
    metrics = [
        (
            "Scheduled tasks",
            greedy_eval.total_scheduled_tasks,
            cpsat_eval.total_scheduled_tasks,
        ),
        (
            "Unscheduled tasks",
            greedy_eval.total_unscheduled_tasks,
            cpsat_eval.total_unscheduled_tasks,
        ),
        (
            "Deadline misses",
            greedy_eval.deadline_miss_count,
            cpsat_eval.deadline_miss_count,
        ),
        (
            "Dependency violations",
            greedy_eval.dependency_violation_count,
            cpsat_eval.dependency_violation_count,
        ),
        (
            "Peak fatigue",
            round(greedy_eval.peak_daily_fatigue, 1),
            round(cpsat_eval.peak_daily_fatigue, 1),
        ),
        (
            "CP pressure",
            round(greedy_eval.critical_path_pressure, 2),
            round(cpsat_eval.critical_path_pressure, 2),
        ),
        (
            "Feasibility score",
            round(greedy_eval.feasibility_score, 3),
            round(cpsat_eval.feasibility_score, 3),
        ),
    ]

    print(f"\n  {'Metric':<28} {'Greedy':>10} {'CP-SAT':>10}")
    print("  " + "-" * 50)
    for label, g_val, c_val in metrics:
        better = ""
        if isinstance(g_val, float) and isinstance(c_val, float):
            if c_val < g_val and label not in ("Scheduled tasks", "Feasibility score"):
                better = "  <- OK"
            elif c_val > g_val and label in ("Scheduled tasks", "Feasibility score"):
                better = "  <- OK"
        elif isinstance(g_val, int) and isinstance(c_val, int):
            if c_val < g_val and label not in ("Scheduled tasks",):
                better = "  <- OK"
            elif c_val > g_val and label == "Scheduled tasks":
                better = "  <- OK"
        print(f"  {label:<28} {str(g_val):>10} {str(c_val):>10}{better}")

    if note:
        print(f"\n  CP-SAT note: {note}")


if __name__ == "__main__":
    from models import generate_test_data
    from context import default_context

    tasks, base_calendar = generate_test_data()
    compare_with_greedy(tasks, base_calendar, default_context(), debug=True)
