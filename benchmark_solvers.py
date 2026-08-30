"""
benchmark_solvers.py - Greedy vs. CP-SAT Scaling Benchmark
============================================================
Generates synthetic multi-day scheduling problems at increasing size
(with dependencies, at a realistic ~65% calendar utilization) and
compares the greedy baseline against the CP-SAT solver on:

  - % of tasks successfully scheduled
  - feasibility score (see evaluate.py)
  - CP-SAT wall-clock solve time

Averages results over multiple random seeds per problem size so the
numbers are stable enough to quote (e.g. in a README or resume).

Run with:  python benchmark_solvers.py
Optional:  python benchmark_solvers.py --sizes 10 20 30 50 --seeds 1 7 42 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import time
from typing import List, Tuple

from context import apply_context, default_context
from evaluate import evaluate_schedule
from models import Task, generate_dynamic_calendar, run_greedy_scheduler_structured
from solver_cpsat import solve_cpsat

DEFAULT_SIZES = [10, 20, 30, 50]
DEFAULT_SEEDS = [1, 7, 42]


def make_tasks(
    n: int,
    seed: int,
    target_utilization: float = 0.65,
    weekly_capacity: float = 41.0,
) -> List[Task]:
    """Generate n synthetic tasks whose total workload targets a given
    fraction of calendar capacity (avoids trivially-easy or
    trivially-infeasible problems)."""
    rnd = random.Random(seed)
    durations = [rnd.choice([1, 2, 3, 4, 5, 6]) for _ in range(n)]
    total_work = sum(durations)
    horizon_days = max(7, round((total_work / weekly_capacity) * 7 / target_utilization))
    horizon_hours = horizon_days * 24

    tasks: List[Task] = []
    for i in range(n):
        dur = durations[i]
        deadline = rnd.randint(max(dur * 24, 48), horizon_hours)
        deps = []
        if i > 3 and rnd.random() < 0.2:
            deps = [f"t{rnd.randint(0, i - 1)}"]
        tasks.append(
            Task(
                id=f"t{i}",
                name=f"Task {i}",
                duration=dur,
                deadline=deadline,
                priority=rnd.randint(1, 10),
                cognitive_weight=rnd.randint(1, 10),
                dependencies=deps,
            )
        )
    return tasks


def run_one(n: int, seed: int) -> Tuple[float, float, float, float, float]:
    """Returns (greedy_sched_pct, greedy_feas, cpsat_sched_pct, cpsat_feas, cpsat_ms)."""
    tasks = make_tasks(n, seed)
    calendar = generate_dynamic_calendar(tasks)
    ctx = default_context()
    eff_cal = apply_context(calendar, ctx)

    greedy_result = run_greedy_scheduler_structured(tasks, eff_cal)
    greedy_eval = evaluate_schedule(greedy_result, tasks, eff_cal)

    t0 = time.time()
    cpsat_result, _note, cpsat_cal = solve_cpsat(tasks, calendar, ctx, debug=False)
    elapsed_ms = (time.time() - t0) * 1000

    if cpsat_result:
        cpsat_eval = evaluate_schedule(cpsat_result, tasks, cpsat_cal)
        return (
            100 * greedy_eval.total_scheduled_tasks / n,
            greedy_eval.feasibility_score,
            100 * cpsat_eval.total_scheduled_tasks / n,
            cpsat_eval.feasibility_score,
            elapsed_ms,
        )
    return (100 * greedy_eval.total_scheduled_tasks / n, greedy_eval.feasibility_score, 0.0, 0.0, elapsed_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark greedy vs CP-SAT scheduling")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--csv", type=str, default=None, help="optional path to write raw results")
    args = parser.parse_args()

    rows = []
    header = (
        f"{'N':>4} | {'greedy sched%':>13} {'greedy feas':>12} | "
        f"{'cpsat sched%':>13} {'cpsat feas':>11} {'cpsat ms':>9}"
    )
    print(header)
    print("-" * len(header))

    for n in args.sizes:
        g_sched, g_feas, c_sched, c_feas, c_ms = [], [], [], [], []
        for seed in args.seeds:
            gs, gf, cs, cf, ms = run_one(n, seed)
            g_sched.append(gs)
            g_feas.append(gf)
            c_sched.append(cs)
            c_feas.append(cf)
            c_ms.append(ms)
            rows.append({"n": n, "seed": seed, "greedy_sched_pct": gs, "greedy_feas": gf,
                         "cpsat_sched_pct": cs, "cpsat_feas": cf, "cpsat_ms": ms})

        print(
            f"{n:4d} | {st.mean(g_sched):12.1f}% {st.mean(g_feas):12.2f} | "
            f"{st.mean(c_sched):12.1f}% {st.mean(c_feas):11.2f} {st.mean(c_ms):9.0f}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nRaw per-seed results written to {args.csv}")


if __name__ == "__main__":
    main()