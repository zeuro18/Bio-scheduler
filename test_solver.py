"""
test_solver.py - Smoke Tests for CP-SAT Solver
Quick sanity checks: runs CP-SAT under a few contexts, prints the
resulting schedule, validates constraints, and compares with greedy.

Run with:  python test_solver.py
"""

from context import default_context, tired_context, crunch_time_context
from display import print_schedule
from models import generate_test_data
from solver_cpsat import solve_cpsat
from validate import validate_solution
from compare import compare_with_greedy


def main():
    tasks, base_calendar = generate_test_data()

    # TEST 1: Default context
    print("TEST 1: Default context (no overrides)")
    ctx1 = default_context()
    result1, note1, cal1 = solve_cpsat(tasks, base_calendar, ctx1, debug=True)

    if result1:
        print_schedule(result1, tasks, cal1, title="CP-SAT - Default Context")
        validate_solution(result1, tasks, debug=True, calendar=cal1)
    else:
        print("  X No feasible solution found.")

    if note1:
        print(f"\n  Relaxation note: {note1}")

    # TEST 2: Tired context
    print("\n\nTEST 2: Tired context (reduced capacity day 0)")
    ctx2 = tired_context()
    result2, note2, cal2 = solve_cpsat(tasks, base_calendar, ctx2, debug=True)

    if result2:
        print_schedule(result2, tasks, cal2, title="CP-SAT - Tired Context")
        validate_solution(result2, tasks, debug=True, calendar=cal2)
    else:
        print("  X No feasible solution found.")

    if note2:
        print(f"\n  Relaxation note: {note2}")

    # TEST 3: Solver comparison
    print("\n\nTEST 3: Solver comparison (greedy vs CP-SAT)")
    compare_with_greedy(tasks, base_calendar, default_context(), debug=False)


if __name__ == "__main__":
    main()
