# pyright: reportAttributeAccessIssue=false
"""
solver_cpsat.py  —  CP-SAT Exact Constraint Solver
====================================================
Phase 3 of the scheduling engine.

PROBLEM FORMULATION
-------------------
Decision variables : boolean matrix  x[task_idx, absolute_hour] ∈ {0, 1}
                     x[t][h] = 1  ⟺  task t is worked on during hour h

Hard constraints
  1. Duration      — task t gets exactly ceil(duration[t]) hours assigned
  2. No overlap    — at most 1 task active per absolute hour (fixes the greedy bug)
  3. Daily capacity— per calendar day, total assigned hours ≤ floor(day.capacity)
  4. Dependencies  — child task cannot start until every parent task is complete
  5. Deadlines     — task t's last active hour must be ≤ deadline[t]
  6. Work slots    — tasks can only be assigned to hours inside defined work windows
                     (enforced implicitly: variables are only created for valid hours)

Soft-as-hard (Phase 3 only)
  7. Max day-chunks — a task may be spread across at most N distinct days
                      prevents the solver from scattering a task across every day

Objective
  Feasibility only.  Find ANY valid solution — NSGA-II handles multi-objective in Phase 4.

HIERARCHICAL RELAXATION  (deadlines in the Task model stay fixed in the formulation)
  Tier 1 → user context applied once; solve with default knobs
  Tier 2 → extend each day's last work-slot by +1 h  (mild overtime)
  Tier 3 → extend each day's last work-slot by +2 h  (moderate overtime)
  Tier 4 → extend each day's last work-slot by +4 h  (extreme overtime / all-nighter)
  Tier 5 → raise max_chunks (allow tasks to fragment across more days)
  Tier 6 → remove lowest-priority task(s) recursively (dependents of dropped roots too)

  IMPORTANT: the relaxation loop NEVER writes to UserContext or deadline_pressure_mode.
  The user's context is applied exactly once (Tier 1) and treated as a hard ceiling.
  Subsequent tiers directly mutate the *calendar copy* to give the solver more room.

HOW TO READ DEBUG OUTPUT
  [SETUP]   — problem dimensions, variable counts
  [MODEL]   — each constraint group as it is added
  [SOLVE]   — solver status + wall time
  [SOLUTION]— solution decoded back into human-readable schedule
  [VALIDATE]— independent constraint checker (double-checks solver output)
  [RELAX]   — which relaxation tier was needed, if any

INSTALL
  pip install ortools
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Set, Tuple
from ortools.sat.python import cp_model
from context import UserContext, apply_context
from hour_index import HourIndex
from models import Resource, ScheduledTask, Task, ScheduleResult, UnscheduledTask
from validate import validate_solution

DEFAULT_MAX_CHUNKS = 7  # max distinct days a task may be spread across
DEFAULT_SOLVE_LIMIT = 30.0  # seconds before solver gives up


def _get_descendants(dropped_task_id: str, remaining_tasks: List[Task]) -> List[Task]:
    """Collect every task in ``remaining_tasks`` that (transitively) depends on ``dropped_task_id``.

    Used in Tier 4 relaxation: if we remove a task from the model, any task that still lists it
    as a dependency would break the CP-SAT formulation (missing ``finish_hour`` variable) or
    encode impossible constraints. So we drop the whole dependent subtree in one step.

    The graph is built **only from edges present in** ``remaining_tasks`` (child → parent links),
    then we DFS/BFS from ``dropped_task_id`` along reversed edges (parent → children).

    Args:
        dropped_task_id: Task id that was just removed from the active set.
        remaining_tasks: Tasks still candidates for scheduling (after the pop).

    Returns:
        List of :class:`Task` objects in ``remaining_tasks`` whose dependency chain reaches
        ``dropped_task_id``. Does **not** include ``dropped_task_id`` itself.
    """
    # dep_id -> [task ids that list dep_id in .dependencies]
    children_map: Dict[str, List[str]] = {}
    for t in remaining_tasks:
        for dep in t.dependencies:
            children_map.setdefault(dep, []).append(t.id)

    # Walk forward along "someone depends on me" edges starting at the dropped root.
    seen: Set[str] = set()
    stack = [dropped_task_id]
    while stack:
        parent = stack.pop()
        for cid in children_map.get(parent, []):
            if cid not in seen:
                seen.add(cid)
                stack.append(cid)

    return [t for t in remaining_tasks if t.id in seen]


def _attach_quality_objective(
    model: cp_model.CpModel,
    tasks: List[Task],
    idx: HourIndex,
    x: Dict[Tuple[int, int], cp_model.IntVar],
    finish_hour_for,
    day_activity_by_task: Dict[int, List[cp_model.IntVar]],
    objective_weights: Dict[str, float],
) -> None:
    terms = []
    scale = 100

    def weight(name: str) -> int:
        return max(0, int(round(float(objective_weights.get(name, 0.0)) * scale)))

    w_fatigue = weight("fatigue")
    w_switch = weight("switches")
    w_risk = weight("deadline_risk")
    w_fragment = weight("fragmentation")

    if w_fatigue:
        total_fatigue_cap = max(
            1,
            sum(int(math.ceil(t.duration)) * int(t.cognitive_weight) for t in tasks),
        )
        daily_fatigue = []
        for day_idx, day_hours in idx.day_to_hours.items():
            expr = sum(
                int(tasks[t_idx].cognitive_weight) * x[t_idx, h]
                for h in day_hours
                for t_idx in range(len(tasks))
                if (t_idx, h) in x
            )
            fatigue = model.NewIntVar(0, total_fatigue_cap, f"fatigue_d{day_idx}")
            model.Add(fatigue == expr)
            daily_fatigue.append(fatigue)
        if daily_fatigue:
            max_fatigue = model.NewIntVar(0, total_fatigue_cap, "max_fatigue")
            model.AddMaxEquality(max_fatigue, daily_fatigue)
            terms.append(w_fatigue * max_fatigue)

    if w_fragment:
        fragments = [v for vals in day_activity_by_task.values() for v in vals]
        if fragments:
            terms.append(w_fragment * sum(fragments))

    if w_risk:
        finish_terms = []
        for t_idx, task in enumerate(tasks):
            fh = finish_hour_for(t_idx)
            if fh is not None:
                finish_terms.append(int(task.priority) * fh)
        if finish_terms:
            terms.append(w_risk * sum(finish_terms))

    if w_switch:
        switch_vars = []
        for day_idx, day_hours in idx.day_to_hours.items():
            ordered = sorted(day_hours)
            for h1, h2 in zip(ordered, ordered[1:]):
                if h2 != h1 + 1:
                    continue
                for a in range(len(tasks)):
                    if (a, h1) not in x:
                        continue
                    for b in range(len(tasks)):
                        if a == b or (b, h2) not in x:
                            continue
                        sw = model.NewBoolVar(f"switch_d{day_idx}_h{h1}_{a}_{b}")
                        model.Add(sw <= x[a, h1])
                        model.Add(sw <= x[b, h2])
                        model.Add(sw >= x[a, h1] + x[b, h2] - 1)
                        switch_vars.append(sw)
        if switch_vars:
            terms.append(w_switch * sum(switch_vars))

    if terms:
        model.Minimize(sum(terms))


def _build_model(
    tasks: List[Task],
    calendar: List[Resource],
    idx: HourIndex,
    max_chunks: int,
    debug: bool,
    objective_weights: Optional[Dict[str, float]] = None,
) -> Tuple[cp_model.CpModel, Dict]:
    """
    Declare all variables and hard constraints.
    Returns (model, vars_dict) where vars_dict contains everything
    needed to extract the solution later.
    """
    model = cp_model.CpModel()
    task_map = {t.id: i for i, t in enumerate(tasks)}

    # One boolean per (task, hour) the task is allowed to use/ sparse matrix via dict.
    x: Dict[Tuple[int, int], cp_model.IntVar] = {}

    day_activity_by_task: Dict[int, List[cp_model.IntVar]] = {}
    for t_idx, task in enumerate(tasks):
        task_hours = idx.hours_for_task(task)
        for h in task_hours:
            x[t_idx, h] = model.NewBoolVar(f"x_{task.id}_h{h}")

    if debug:
        print(f"\n[SETUP] Tasks            : {len(tasks)}")
        print(f"[SETUP] Valid hours total : {len(idx.valid_hours)}")
        print(f"[SETUP] Decision vars     : {len(x)}")
        print(f"[SETUP] Max chunks/task   : {max_chunks}")

    # CONSTRAINT 1 DURATION
    # Σ x[t][h]  ==  ceil(duration[t])   (always enough hours to cover fractional need)

    if debug:
        print("\n[MODEL] Adding Constraint 1: Duration")

    for t_idx, task in enumerate(tasks):
        task_hours = [h for h in idx.hours_for_task(task)]
        duration_int = int(math.ceil(task.duration))

        if len(task_hours) < duration_int:
            if debug:
                print(
                    f"Task '{task.id}': only {len(task_hours)} valid hours "
                    f"but needs {duration_int}h : will be infeasible"
                )

        model.Add(sum(x[t_idx, h] for h in task_hours) == duration_int)

    # CONSTRAINT 2 NO OVERLAP
    # At most 1 task active per absolute hour.

    if debug:
        print("[MODEL] Adding Constraint 2: No Overlap")

    for h in idx.valid_hours:
        active_at_h = [x[t_idx, h] for t_idx in range(len(tasks)) if (t_idx, h) in x]
        # Single-task hours need no constraint; two+ tasks could share h → forbid that.
        if len(active_at_h) > 1:
            model.Add(sum(active_at_h) <= 1)

    # CONSTRAINT 2b DAILY CAPACITY (resource)
    # Σ_{h on day d} Σ_t x[t][h] ≤ floor(capacity[d])  {independent of no-overlap}

    if debug:
        print("[MODEL] Adding Constraint 2b: Daily capacity")

    for day_idx, day in enumerate(calendar):
        day_hours = idx.day_to_hours.get(day_idx, [])
        cap = int(math.floor(day.capacity))
        if cap < 0:
            cap = 0
        terms = [
            x[t_idx, h]
            for h in day_hours
            for t_idx in range(len(tasks))
            if (t_idx, h) in x
        ]
        if terms:
            model.Add(sum(terms) <= cap)

    # CONSTRAINT 4 DEPENDENCIES
    # Child task cannot start until every parent task has fully completed.

    #   finish_hour[dep]  =  max( h  where  x[dep][h] = 1 )
    #   For every child hour h_c:  x[child][h_c] = 1  →  h_c ≥ finish_hour[dep] + 1

    # We build finish_hour[dep] using AddMaxEquality over auxiliary terms:
    #   term[dep][h]  =  h   if x[dep][h] = 1
    #                 =  0   otherwise
    #   finish_hour[dep] = max(all terms)

    if debug:
        print("[MODEL] Adding Constraint 4: Dependencies")

    # Reused across children: last hour index where dependency task is active.
    finish_hour: Dict[int, cp_model.IntVar] = {}  # dep t_idx -> IntVar

    for t_idx, task in enumerate(tasks):
        if not task.dependencies:
            continue

        for dep_id in task.dependencies:
            if dep_id not in task_map:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown or unavailable task '{dep_id}'"
                )
            dep_idx = task_map[dep_id]

            # finish_hour[dep] = max { h | x[dep,h]=1 }; build once, link many children
            if dep_idx not in finish_hour:
                dep_hours = [h for h in idx.valid_hours if (dep_idx, h) in x]

                if not dep_hours:
                    if debug:
                        print(f"Dep '{dep_id}' has no valid hours, skip")
                    continue

                fh = model.NewIntVar(0, idx.max_hour + 1, f"finish_{tasks[dep_idx].id}")

                # Piecewise: if x[dep,h]=1 then term=h else term=0 → max(term)=last active h
                terms = []
                for h in dep_hours:
                    term = model.NewIntVar(
                        0, idx.max_hour, f"term_{tasks[dep_idx].id}_h{h}"
                    )
                    model.Add(term == h).OnlyEnforceIf(x[dep_idx, h])
                    model.Add(term == 0).OnlyEnforceIf(x[dep_idx, h].Not())
                    terms.append(term)

                model.AddMaxEquality(fh, terms)
                finish_hour[dep_idx] = fh

                if debug:
                    print(f"  finish_hour[{dep_id}] built over {len(dep_hours)} hours")

            if dep_idx not in finish_hour:
                continue  # dep had no valid hours; skip

            # If child uses hour h_c, that hour must be strictly after dep's last hour.
            child_hours = [h for h in idx.hours_for_task(task)]
            for h_c in child_hours:
                model.Add(h_c >= finish_hour[dep_idx] + 1).OnlyEnforceIf(x[t_idx, h_c])

    def _ensure_finish_hour(t_idx: int) -> Optional[cp_model.IntVar]:
        if t_idx in finish_hour:
            return finish_hour[t_idx]
        task_hours = [h for h in idx.valid_hours if (t_idx, h) in x]
        if not task_hours:
            return None
        fh = model.NewIntVar(0, idx.max_hour + 1, f"finish_obj_{tasks[t_idx].id}")
        terms = []
        for h in task_hours:
            term = model.NewIntVar(0, idx.max_hour, f"term_obj_{tasks[t_idx].id}_h{h}")
            model.Add(term == h).OnlyEnforceIf(x[t_idx, h])
            model.Add(term == 0).OnlyEnforceIf(x[t_idx, h].Not())
            terms.append(term)
        model.AddMaxEquality(fh, terms)
        finish_hour[t_idx] = fh
        return fh

    # CONSTRAINT 5 DEADLINES
    # Enforced implicitly: x[t][h] only exists for h < deadline[t].
    # Nothing extra to add here.

    if debug:
        print("[MODEL] Constraint 5: Deadlines (enforced via variable scope)")

    # CONSTRAINT 6 MAX DAY-CHUNKS PER TASK
    # A "chunk" = a distinct day on which the task has at least 1 active hour.
    # Prevents the solver from scattering thesis across 20 different days.
    #
    # is_active_on_day[t][d] = 1 iff task t has any active hour on day d
    #                        = max( x[t][h]  for h in day d's hours )
    # Constraint: Σ_d is_active_on_day[t][d] ≤ max_chunks

    if debug:
        print(f"[MODEL] Adding Constraint 6: Max {max_chunks} day-chunks per task")

    for t_idx, task in enumerate(tasks):
        day_activity: List[cp_model.IntVar] = []

        for day_idx, day_hours in idx.day_to_hours.items():
            active_on_day = [x[t_idx, h] for h in day_hours if (t_idx, h) in x]
            if not active_on_day:
                continue

            is_active = model.NewBoolVar(f"active_{task.id}_d{day_idx}")
            # is_active = max(active_on_day) — OR over all booleans on this day
            model.AddMaxEquality(is_active, active_on_day)
            day_activity.append(is_active)

        day_activity_by_task[t_idx] = day_activity

        # If the task could touch more than max_chunks days, force it to use fewer.
        if len(day_activity) > max_chunks:
            model.Add(sum(day_activity) <= max_chunks)

    # Feasibility only — later phases may attach a linear objective to the same variables.
    if objective_weights:
        _attach_quality_objective(
            model=model,
            tasks=tasks,
            idx=idx,
            x=x,
            finish_hour_for=_ensure_finish_hour,
            day_activity_by_task=day_activity_by_task,
            objective_weights=objective_weights,
        )

    vars_dict = {"x": x, "finish_hour": finish_hour}
    return model, vars_dict


# 2) EXTRACT SOLUTION -> ScheduleResult


def _extract_solution(
    solver: cp_model.CpSolver,
    tasks: List[Task],
    calendar: List[Resource],
    idx: HourIndex,
    vars_dict: Dict,
    debug: bool,
) -> ScheduleResult:
    """
    Convert CP-SAT solution values back into a ScheduleResult.
    One ScheduledTask entry per active (task, hour) pair.
    """
    x = vars_dict["x"]

    scheduled: List[ScheduledTask] = []
    resource_usage: Dict[str, float] = {r.id: 0.0 for r in calendar}

    for t_idx, task in enumerate(tasks):
        for h in idx.hours_for_task(task):
            if (t_idx, h) not in x:
                continue
            if solver.value(x[t_idx, h]) == 1:
                day_idx = idx.hour_to_day[h]
                day = calendar[day_idx]

                scheduled.append(
                    ScheduledTask(
                        task_id=task.id,
                        resource_id=day.id,
                        allocated_hours=1.0,
                        # Absolute hour index (not local): matches HourIndex.valid_hours entries.
                        start_time=float(h),
                    )
                )
                resource_usage[day.id] += 1.0

    # identify unscheduled tasks
    allocated: Dict[str, float] = {}
    for entry in scheduled:
        allocated[entry.task_id] = allocated.get(entry.task_id, 0.0) + 1.0

    unscheduled: List[UnscheduledTask] = []
    for task in tasks:
        got = allocated.get(task.id, 0.0)
        remaining = int(math.ceil(task.duration)) - got
        if remaining > 0:  # float tolerance
            unscheduled.append(
                UnscheduledTask(
                    task_id=task.id,
                    task_name=task.name,
                    remaining_hours=remaining,
                    reason="CP-SAT could not place within constraints",
                    deadline=task.deadline,
                )
            )

    if debug:
        print(f"\n[SOLUTION] Scheduled entries : {len(scheduled)}")
        print(f"[SOLUTION] Unscheduled tasks : {len(unscheduled)}")
        for entry in unscheduled:
            print(f"  {entry.task_name}: {entry.remaining_hours:.0f}h missing")

    return ScheduleResult(
        scheduled=scheduled,
        unscheduled=unscheduled,
        deadline_misses=[],
        dependency_violations=[],  # derive precisely in evaluate_schedule()
        resource_usage=resource_usage,
    )


# 3) SINGLE SOLVE ATTEMPT


def _solve_tier(
    tasks: List[Task],
    calendar: List[Resource],
    max_chunks: int,
    time_limit: float,
    debug: bool,
    tier_label: str,
    objective_weights: Optional[Dict[str, float]] = None,
) -> Optional[ScheduleResult]:

    # One solve attempt. Returns ScheduleResult on success, None on infeasibility.

    if debug:
        print(f"[SOLVE] {tier_label}\n")

    idx = HourIndex.build(calendar, tasks)
    try:
        model, vars_dict = _build_model(
            tasks, calendar, idx, max_chunks, debug, objective_weights
        )
    except ValueError as exc:
        if debug:
            print(f"[MODEL] Invalid task graph: {exc}")
        return None

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False

    t0 = time.time()
    status = solver.solve(model)
    elapsed = time.time() - t0

    status_name = solver.status_name(status)

    if debug:
        print(f"\n[SOLVE] Status  : {status_name}")
        print(f"[SOLVE] Wall time: {elapsed:.2f}s")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result = _extract_solution(solver, tasks, calendar, idx, vars_dict, debug)
        ok = validate_solution(result, tasks, debug=debug, calendar=calendar)
        if not ok:
            print("CP-SAT Solution Validation Failed")
            return None
        return result

    if status == cp_model.INFEASIBLE:
        if debug:
            print("[SOLVE] Problem is provably infeasible with current constraints.")
    elif status == cp_model.UNKNOWN:
        if debug:
            print("[SOLVE] Time limit reached -> no solution found.")

    return None


# 4.  MAIN, hierarchical relaxation


def solve_cpsat(
    tasks: List[Task],
    base_calendar: List[Resource],
    ctx: UserContext,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    time_limit: float = DEFAULT_SOLVE_LIMIT,
    debug: bool = True,
    objective_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[ScheduleResult], str, List[Resource]]:
    """
    Main solver entry point.

    Returns (ScheduleResult, relaxation_note, effective_calendar_used).
    relaxation_note describes what was relaxed (empty string = solved as-is).
    effective_calendar_used is the post-context calendar the solver actually used
    (important when Tier 2+ extends work slots).

    Hierarchical relaxation:
      Tier 1: original constraints, user context as-is
      Tier 2: extend each day's last slot by +1 h  (mild overtime)
      Tier 3: extend each day's last slot by +2 h  (moderate overtime)
      Tier 4: extend each day's last slot by +4 h  (extreme overtime)
      Tier 5: increase max_chunks by 3 (allow more day-spreading / fragmentation)
      Tier 6: drop lowest-priority tasks one at a time until feasible

    The user's UserContext (including deadline_pressure_mode / intensity) is applied
    once at Tier 1 via apply_context() and is NEVER mutated by the relaxation loop.
    This means a user who is already in 'extreme' pressure mode never gets downgraded.
    """

    # Apply user context once — never touched again by the relaxation loop.
    effective_cal = apply_context(base_calendar, ctx)

    # PRE-FLIGHT: identify tasks that are structurally impossible before any
    # solving attempt.  A task is structurally infeasible if HourIndex finds
    # zero valid work-slot hours for it, most commonly because its deadline
    # falls before the first work slot of the day (e.g. deadline=4h but work
    # starts at hour 9).  Including such tasks in the CP-SAT model makes the
    # duration constraint trivially unsatisfiable, which causes the solver to
    # cascade-fail through all relaxation tiers and incorrectly drop every
    # other task later
    _preflight_idx = HourIndex.build(effective_cal, tasks)
    structurally_infeasible: List[Task] = []
    tasks_to_solve: List[Task] = []
    preflight_removed: List[Tuple[Task, str]] = []
    for _t in tasks:
        if _preflight_idx.hours_for_task(_t):
            tasks_to_solve.append(_t)
        else:
            structurally_infeasible.append(_t)
            preflight_removed.append((_t, "no valid work-slot hours before deadline"))
            if debug:
                print(
                    f"[PREFLIGHT] '{_t.name}' has no valid work-slot hours "
                    f"before deadline={_t.deadline}h: removed before solving. "
                    f"Either the deadline is earlier than the first work slot "
                    f"or there is no calendar capacity within the deadline window."
                )
    if structurally_infeasible:
        cascade_ids: Set[str] = set()
        for root in structurally_infeasible:
            cascade_ids.update(t.id for t in _get_descendants(root.id, tasks_to_solve))

        if cascade_ids:
            cascaded = [t for t in tasks_to_solve if t.id in cascade_ids]
            tasks_to_solve = [t for t in tasks_to_solve if t.id not in cascade_ids]
            for t in cascaded:
                preflight_removed.append(
                    (
                        t,
                        "dependency unavailable because prerequisite is structurally infeasible",
                    )
                )
            if debug:
                names = [t.name for t in cascaded]
                print(
                    f"[PREFLIGHT] Removing dependent task(s): {names} "
                    f"because prerequisite task(s) cannot be scheduled."
                )
    tasks = tasks_to_solve

    def _preflight_unscheduled() -> List[UnscheduledTask]:
        return [
            UnscheduledTask(
                task_id=t.id,
                task_name=t.name,
                remaining_hours=t.duration,
                reason=reason,
                deadline=t.deadline,
            )
            for t, reason in preflight_removed
        ]

    if not tasks:
        # Every task was structurally infeasible — nothing to solve.
        unscheduled_all = _preflight_unscheduled()
        empty_result = ScheduleResult(
            scheduled=[],
            unscheduled=unscheduled_all,
            deadline_misses=[],
            dependency_violations=[],
            resource_usage={r.id: 0.0 for r in effective_cal},
        )
        note = (
            "All tasks were removed before solving because they were structurally "
            "infeasible or depended on a structurally infeasible task. Check that "
            "deadlines are far enough in the future to overlap with work slots."
        )
        return empty_result, note, effective_cal

    # Tier 1: solve as-is
    result = _solve_tier(
        tasks,
        effective_cal,
        max_chunks,
        time_limit,
        debug,
        "Tier 1: Original constraints",
        objective_weights,
    )
    if result:
        result.unscheduled.extend(_preflight_unscheduled())
        return result, "", effective_cal

    # extend the last slot of every day in a calendar copy
    def _extend_slots(cal: List[Resource], extra_hours: float) -> List[Resource]:
        """Return a new calendar where each day's last work-slot end is pushed
        forward by ``extra_hours`` (capped at 24 h).  Capacity is bumped to
        match, but only if the slot actually grew (i.e. wasn't already at 24h).
        UserContext is never touched."""
        out = []
        for day in cal:
            slots = list(day.work_slots) if day.work_slots else []
            added = 0.0
            if slots:
                s, e = slots[-1]
                new_e = min(24.0, e + extra_hours)
                added = new_e - e
                slots[-1] = (s, new_e)
            out.append(
                Resource(
                    id=day.id,
                    name=day.name,
                    capacity=day.capacity + added,
                    work_slots=tuple(slots),
                )
            )
        return out

    # Tier 2: mild overtime (+1 h/day)
    if debug:
        print("\n[RELAX] Tier 1 failed → Tier 2: mild overtime (+1 h/day)")

    # Helper: attach tasks removed during preflight to any successful result.
    def _attach_infeasible(res: ScheduleResult) -> ScheduleResult:
        res.unscheduled.extend(_preflight_unscheduled())
        return res

    effective_cal2 = _extend_slots(effective_cal, 1.0)
    result = _solve_tier(
        tasks,
        effective_cal2,
        max_chunks,
        time_limit,
        debug,
        "Tier 2: Mild overtime (+1 h/day)",
        objective_weights,
    )
    if result:
        note = "Note: schedule required mild overtime (+1 h/day past normal hours)."
        return _attach_infeasible(result), note, effective_cal2

    #  Tier 3: moderate overtime (+2 h/day)
    if debug:
        print("\n[RELAX] Tier 2 failed → Tier 3: moderate overtime (+2 h/day)")

    effective_cal3 = _extend_slots(effective_cal, 2.0)
    result = _solve_tier(
        tasks,
        effective_cal3,
        max_chunks,
        time_limit,
        debug,
        "Tier 3: Moderate overtime (+2 h/day)",
        objective_weights,
    )
    if result:
        note = "Note: schedule required moderate overtime (+2 h/day past normal hours)."
        return _attach_infeasible(result), note, effective_cal3

    # Tier 4: extreme overtime (+4 h/day)
    if debug:
        print("\n[RELAX] Tier 3 failed → Tier 4: extreme overtime (+4 h/day)")

    effective_cal4 = _extend_slots(effective_cal, 4.0)
    result = _solve_tier(
        tasks,
        effective_cal4,
        max_chunks,
        time_limit,
        debug,
        "Tier 4: Extreme overtime (+4 h/day)",
        objective_weights,
    )
    if result:
        note = "Note: schedule required extreme overtime (+4 h/day). Consider reducing workload."
        return _attach_infeasible(result), note, effective_cal4

    # Tier 5: increase chunk limit (using extreme-OT calendar)
    relaxed_chunks = max_chunks + 3
    if debug:
        print(
            f"\n[RELAX] Tier 4 failed → Tier 5: relaxed chunks ({relaxed_chunks} days/task)"
        )

    result = _solve_tier(
        tasks,
        effective_cal4,
        relaxed_chunks,
        time_limit,
        debug,
        f"Tier 5: Relaxed chunks ({relaxed_chunks})",
        objective_weights,
    )
    if result:
        note = (
            f"Note: tasks are more fragmented than usual (up to {relaxed_chunks} days each). "
            f"Also required extreme overtime. Consider reducing total workload."
        )
        return _attach_infeasible(result), note, effective_cal4

    # Tier 6: drop lowest-priority tasks
    if debug:
        print("\n[RELAX] Tier 5 failed → Tier 6: dropping lowest-priority tasks")

    remaining_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
    dropped: List[Task] = []

    while len(remaining_tasks) > 1:
        # List is descending priority → pop = worst priority among those still in play.
        dropped_task = remaining_tasks.pop()
        dropped.append(dropped_task)

        # Any task that still depends (directly or transitively) on dropped_task must go too.
        cascade = _get_descendants(dropped_task.id, remaining_tasks)
        cascade_ids = {t.id for t in cascade}
        remaining_tasks = sorted(
            [t for t in remaining_tasks if t.id not in cascade_ids],
            key=lambda t: t.priority,
            reverse=True,
        )
        dropped.extend(cascade)
        if not remaining_tasks:
            break

        if debug:
            names = [dropped_task.name] + [t.name for t in cascade]
            print(
                f"[RELAX] Dropping task(s): {names} " f"(priority cascade / dependents)"
            )

        result = _solve_tier(
            remaining_tasks,
            effective_cal4,
            relaxed_chunks,
            time_limit,
            False,  # suppress inner debug on retries
            f"Tier 6: Dropped {[t.id for t in dropped]}",
            objective_weights,
        )
        if result:
            # Add dropped tasks as unscheduled
            for dt in dropped:
                result.unscheduled.append(
                    UnscheduledTask(
                        task_id=dt.id,
                        task_name=dt.name,
                        remaining_hours=dt.duration,
                        reason="dropped during relaxation (lowest priority)",
                        deadline=dt.deadline,
                    )
                )
            dropped_names = [t.name for t in dropped]
            note = (
                f"Warning: the following tasks were dropped to find a feasible "
                f"schedule: {dropped_names}. Add more calendar capacity or "
                f"reduce total workload."
            )
            if debug:
                print(f"\n[RELAX] Feasible after dropping: {dropped_names}")
            return _attach_infeasible(result), note, effective_cal4

    #  complete failure
    if debug:
        print("\n[RELAX] All 6 relaxation tiers exhausted. No feasible schedule found.")
        print("        Possible causes:")
        print(
            "         Total task duration exceeds total calendar capacity (even with overtime)"
        )
        print("         Dependency chain is longer than available time before deadline")
        print("         Work slot hours are too restrictive")

    return (
        None,
        "No feasible schedule found after all relaxation attempts.",
        effective_cal4,
    )
