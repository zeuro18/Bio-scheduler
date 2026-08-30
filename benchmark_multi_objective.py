"""
benchmark_multi_objective.py - End-to-End Comparative Benchmark
================================================================
Compares Greedy Baseline, CP-SAT (Hard Constraint Solver), and
NSGA-II (Multi-Objective Biological Optimizer) on realistic multi-day
scheduling workloads (~65% calendar utilization).

Evaluates:
  1. Feasibility Metrics (evaluate.py):
     - Task Completion %
     - Deadline Misses
     - Dependency Violations
     - Feasibility Score (0-1)
  2. Biological Objective Scores (objectives.py):
     - Cognitive Fatigue Penalty (variance & peak daily load)
     - Context Switches (task interruptions within a day)
     - Deadline Risk (safety buffer before deadline)
     - Task Fragmentation (scattering across multiple days)
  3. Execution Runtime (ms)

Run with:
  python benchmark_multi_objective.py
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from typing import List

from benchmark_solvers import make_tasks
from context import apply_context, default_context
from evaluate import evaluate_schedule
from models import generate_dynamic_calendar, run_greedy_scheduler_structured
from objectives import evaluate_objectives
from solver_cpsat import solve_cpsat
from solver_nsga import NSGAConfig, run_nsga


def run_benchmark(sizes: List[int], seeds: List[int]) -> None:
    print("=" * 88)
    print("BIO-SCHEDULER: END-TO-END MULTI-SOLVER BENCHMARK")
    print("Engines: Greedy Baseline | CP-SAT (OR-Tools) | NSGA-II (pymoo)")
    print("=" * 88)

    for n in sizes:
        print(f"\nEvaluating Scale: N = {n} Tasks (Averaged over seeds {seeds})...")

        # Metrics accumulators
        g_sched, c_sched, n_sched = [], [], []
        g_dl, c_dl, n_dl = [], [], []
        g_feas, c_feas, n_feas = [], [], []
        g_fatigue, c_fatigue, n_fatigue = [], [], []
        g_switch, c_switch, n_switch = [], [], []
        g_risk, c_risk, n_risk = [], [], []
        g_frag, c_frag, n_frag = [], [], []
        g_time, c_time, n_time = [], [], []

        for seed in seeds:
            tasks = make_tasks(n, seed)
            calendar = generate_dynamic_calendar(tasks)
            ctx = default_context()
            eff_cal = apply_context(calendar, ctx)

            # 1. Greedy Baseline
            t0 = time.time()
            g_res = run_greedy_scheduler_structured(tasks, eff_cal)
            g_time.append((time.time() - t0) * 1000)
            g_eval = evaluate_schedule(g_res, tasks, eff_cal)
            g_obj = evaluate_objectives(g_res, tasks, eff_cal, ctx)

            g_sched.append(100 * g_eval.total_scheduled_tasks / n)
            g_dl.append(g_eval.deadline_miss_count)
            g_feas.append(g_eval.feasibility_score)
            g_fatigue.append(g_obj.fatigue)
            g_switch.append(g_obj.context_switches)
            g_risk.append(g_obj.deadline_risk)
            g_frag.append(g_obj.fragmentation)

            # 2. CP-SAT Solver
            t0 = time.time()
            c_res, _, c_cal = solve_cpsat(tasks, calendar, ctx, debug=False)
            c_time.append((time.time() - t0) * 1000)
            if c_res:
                c_eval = evaluate_schedule(c_res, tasks, c_cal)
                c_obj = evaluate_objectives(c_res, tasks, c_cal, ctx)
                c_sched.append(100 * c_eval.total_scheduled_tasks / n)
                c_dl.append(c_eval.deadline_miss_count)
                c_feas.append(c_eval.feasibility_score)
                c_fatigue.append(c_obj.fatigue)
                c_switch.append(c_obj.context_switches)
                c_risk.append(c_obj.deadline_risk)
                c_frag.append(c_obj.fragmentation)

            # 3. NSGA-II Multi-Objective Optimizer
            t0 = time.time()
            cfg = NSGAConfig(population_size=40, generations=25, seed=seed)
            n_res, _, _, n_cal = run_nsga(tasks, calendar, ctx, config=cfg)
            n_time.append((time.time() - t0) * 1000)
            if n_res:
                n_eval = evaluate_schedule(n_res, tasks, n_cal)
                n_obj = evaluate_objectives(n_res, tasks, n_cal, ctx)
                n_sched.append(100 * n_eval.total_scheduled_tasks / n)
                n_dl.append(n_eval.deadline_miss_count)
                n_feas.append(n_eval.feasibility_score)
                n_fatigue.append(n_obj.fatigue)
                n_switch.append(n_obj.context_switches)
                n_risk.append(n_obj.deadline_risk)
                n_frag.append(n_obj.fragmentation)

        # Output formatted comparison table
        print(f"\n{'Metric':<32} | {'Greedy':<10} | {'CP-SAT':<10} | {'NSGA-II (Bio)':<14}")
        print("-" * 74)
        print(f"{'Task Scheduled %':<32} | {st.mean(g_sched):>9.1f}% | {st.mean(c_sched):>9.1f}% | {st.mean(n_sched):>13.1f}%")
        print(f"{'Deadline Miss Count (avg)':<32} | {st.mean(g_dl):>10.2f} | {st.mean(c_dl):>10.2f} | {st.mean(n_dl):>14.2f}")
        print(f"{'Feasibility Score (0-1.0)':<32} | {st.mean(g_feas):>10.2f} | {st.mean(c_feas):>10.2f} | {st.mean(n_feas):>14.2f}")
        print(f"{'Cognitive Fatigue Penalty':<32} | {st.mean(g_fatigue):>10.2f} | {st.mean(c_fatigue):>10.2f} | {st.mean(n_fatigue):>14.2f}")
        print(f"{'Context Switches (interruptions)':<32} | {st.mean(g_switch):>10.2f} | {st.mean(c_switch):>10.2f} | {st.mean(n_switch):>14.2f}")
        print(f"{'Deadline Risk Score':<32} | {st.mean(g_risk):>10.2f} | {st.mean(c_risk):>10.2f} | {st.mean(n_risk):>14.2f}")
        print(f"{'Task Fragmentation (days)':<32} | {st.mean(g_frag):>10.2f} | {st.mean(c_frag):>10.2f} | {st.mean(n_frag):>14.2f}")
        print(f"{'Execution Time (ms)':<32} | {st.mean(g_time):>10.1f} | {st.mean(c_time):>10.1f} | {st.mean(n_time):>14.1f}")
        print("-" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-way multi-solver benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42])
    args = parser.parse_args()
    run_benchmark(args.sizes, args.seeds)
