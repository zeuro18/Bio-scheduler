import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# Scoring Weights
W = 10  # max urgency boost the panic term can add
K = 0.05  # decay rate controls how sharply urgency ramps up
DEFAULT_WORK_SLOTS: Tuple[Tuple[float, float], ...] = (
    (9.0, 12.0),
    (14.0, 17.0),
    (18.0, 21.0),
)


@dataclass
class Task:
    id: str
    name: str
    duration: float
    deadline: float
    priority: int
    cognitive_weight: int = 5
    dependencies: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        # Check basic feasibility
        return (
            self.duration > 0
            and self.deadline > 0
            and self.duration <= self.deadline
            and 1 <= self.priority <= 10
            and 1 <= self.cognitive_weight <= 10
        )


@dataclass(frozen=True)
class Resource:
    # single calendar day with explicit work windows.
    id: str
    name: str
    capacity: float
    work_slots: Tuple[Tuple[float, float], ...] = ((0.0, 24.0),)


@dataclass
class ScheduledTask:
    # task X gets Y hours on resource Z.

    task_id: str
    resource_id: str
    allocated_hours: float
    start_time: Optional[float] = None  # hour at which this task begins (Phase 3+)


@dataclass
class UnscheduledTask:
    """A task (or remainder) that could not be placed."""

    task_id: str
    task_name: str
    remaining_hours: float
    reason: str
    deadline: float


@dataclass
class ScheduleResult:
    """Unified output of any solver (greedy, CP-SAT, NSGA-II)."""

    scheduled: List[ScheduledTask]
    unscheduled: List[UnscheduledTask] = field(default_factory=list)
    deadline_misses: List[str] = field(default_factory=list)
    dependency_violations: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)


def calculate_hybrid_score(task: Task) -> float:
    return float(task.priority + (W * math.exp(-K * task.deadline)))


def find_earliest_slot(
    from_hour: float,
    duration: float,
    work_slots: Tuple[Tuple[float, float], ...],
    day_start_hour: float = 0.0,
) -> Optional[float]:
    # finds earliest absolute hour within work_slots where a block of duration hours fits, starting no earlier than from_hour.

    for slot_start, slot_end in work_slots:
        abs_start = day_start_hour + slot_start
        abs_end = day_start_hour + slot_end

        # Can't begin before from_hour
        actual_start = max(from_hour, abs_start)

        # Does the block fit before this slot closes?
        if actual_start + duration <= abs_end:
            return actual_start

    return None  # doesn't fit in any slot today


def find_earliest_slot_chunk(
    from_hour: float,
    max_duration: float,
    work_slots: Tuple[Tuple[float, float], ...],
    day_start_hour: float = 0.0,
) -> Optional[Tuple[float, float]]:
    # Return earliest slot start and allocatable chunk length(allows partial allocation)

    for slot_start, slot_end in work_slots:
        abs_start = day_start_hour + slot_start
        abs_end = day_start_hour + slot_end
        actual_start = max(from_hour, abs_start)
        available = abs_end - actual_start

        if available > 0:
            return actual_start, min(max_duration, available)

    return None


def generate_dynamic_calendar(tasks: List[Task]) -> List[Resource]:
    # Generate a calendar whose days carry realistic work-slots.
    if not tasks:
        return []

    max_deadline = max(task.deadline for task in tasks)
    total_days_needed = math.ceil(max_deadline / 24.0)
    weekly_capacities = [5, 5, 6, 6, 5, 7, 7]
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    return [
        Resource(
            id=f"day_{i}",
            name=f"Day {i + 1} ({day_names[i % 7]})",
            capacity=weekly_capacities[i % 7],
            work_slots=DEFAULT_WORK_SLOTS,
        )
        for i in range(total_days_needed)
    ]


def generate_test_data() -> Tuple[List[Task], List[Resource]]:
    tasks: List[Task] = [
        Task(
            id="dl_test",
            name="Deep Learning Class test",
            duration=10,
            deadline=200,
            priority=10,
            cognitive_weight=9,
            dependencies=["thesis"],
        ),
        Task(
            id="thesis",
            name="thesis",
            duration=8,
            deadline=100,
            priority=8,
            cognitive_weight=8,
        ),
        Task(
            id="design_lab",
            name="Design Lab",
            duration=5,
            deadline=120,
            priority=6,
            cognitive_weight=6,
            dependencies=["thesis"],
        ),
        Task(
            id="sma_paper",
            name="SMA Paper",
            duration=4,
            deadline=72,
            priority=7,
            cognitive_weight=3,
            dependencies=[],
        ),
        Task(
            id="magnetism_paper",
            name="Magnetism term paper",
            duration=6,
            deadline=200,
            priority=5,
            cognitive_weight=4,
        ),
    ]
    return tasks, generate_dynamic_calendar(tasks)


# Greedy Scheduler (DAG-aware)


def run_greedy_scheduler(
    tasks: List[Task], calendar: List[Resource]
) -> Tuple[List[ScheduledTask], dict]:

    from dag import DAGengine

    # Validate every task
    for t in tasks:
        if not t.is_valid():
            raise ValueError(
                f"Task '{t.name}' (id={t.id}) failed validation: "
                f"duration={t.duration}, deadline={t.deadline}"
            )

    dag = DAGengine(tasks)

    # Track resource usage without mutating the immutable Resource objects
    resource_usage = {r.id: 0.0 for r in calendar}
    final_ledger: List[ScheduledTask] = []

    task_map = {t.id: t for t in tasks}
    completed: set = set()
    task_finish_day: dict = {}  # task_id -> last day index with allocated hours
    task_finish_hour: dict = {}  # task_id -> absolute end hour of final chunk

    # Seed the ready queue with tasks that have no dependencies (in_degree == 0)
    ready = [t for t in tasks if dag.in_degree[t.id] == 0]
    ready.sort(key=calculate_hybrid_score, reverse=True)

    while ready:
        # Pick the highest-scored ready task
        task = ready.pop(0)

        remaining = task.duration
        deadline_day_idx = max(0, math.ceil(task.deadline / 24.0) - 1)

        # Child task cannot start before the absolute finish of all dependencies.
        earliest_start_hour = max(
            (
                task_finish_hour[dep]
                for dep in task.dependencies
                if dep in task_finish_hour
            ),
            default=0.0,
        )
        earliest_day = int(earliest_start_hour // 24)
        last_end_hour = earliest_start_hour

        # Allocate forward from earliest_day, stopping at deadline
        for day_idx in range(earliest_day, min(len(calendar), deadline_day_idx + 1)):
            if remaining <= 0:
                break

            day = calendar[day_idx]
            available = day.capacity - resource_usage[day.id]

            if available <= 0:
                continue

            day_hour_offset = day_idx * 24.0
            from_hour = max(last_end_hour, day_hour_offset)

            # Fill this day using as many slot chunks as needed.
            while remaining > 0 and available > 0:
                max_chunk = min(remaining, available)
                chunk = find_earliest_slot_chunk(
                    from_hour=from_hour,
                    max_duration=max_chunk,
                    work_slots=day.work_slots,
                    day_start_hour=day_hour_offset,
                )
                if chunk is None:
                    break

                start_hour, allocated = chunk
                resource_usage[day.id] += allocated
                remaining -= allocated
                available -= allocated
                from_hour = start_hour + allocated
                last_end_hour = from_hour
                task_finish_day[task.id] = day_idx
                task_finish_hour[task.id] = from_hour

                final_ledger.append(
                    ScheduledTask(
                        task_id=task.id,
                        resource_id=day.id,
                        allocated_hours=allocated,
                        start_time=start_hour,
                    )
                )

            if remaining > 0:
                # Move to next day midnight if work remains.
                last_end_hour = (day_idx + 1) * 24.0

        if remaining > 0:
            print(
                f" Warning: Task '{task.name}' has {remaining:.1f}h "
                f"still unscheduled (ran out of calendar capacity)."
            )

        # Only fully completed tasks can unlock dependents.
        if remaining <= 0:
            completed.add(task.id)

        for child_id in dag.graph.get(task.id, []):
            child = task_map[child_id]
            if child_id not in completed and all(
                d in completed for d in child.dependencies
            ):
                ready.append(child)

        # sort so the highest scored ready task is always picked next
        ready.sort(key=calculate_hybrid_score, reverse=True)

    return final_ledger, resource_usage


def run_greedy_scheduler_structured(
    tasks: List[Task],
    calendar: List[Resource],
) -> ScheduleResult:
    """Run the greedy scheduler and wrap its output in a ScheduleResult."""
    ledger, resource_usage = run_greedy_scheduler(tasks, calendar)

    # Figure out which tasks are fully scheduled vs partially/unscheduled
    allocated: Dict[str, float] = {}
    for entry in ledger:
        allocated[entry.task_id] = (
            allocated.get(entry.task_id, 0.0) + entry.allocated_hours
        )

    unscheduled: List[UnscheduledTask] = []
    for t in tasks:
        got = allocated.get(t.id, 0.0)
        remaining = t.duration - got
        if remaining > 0.5:
            unscheduled.append(
                UnscheduledTask(
                    task_id=t.id,
                    task_name=t.name,
                    remaining_hours=remaining,
                    reason="greedy scheduler ran out of capacity",
                    deadline=t.deadline,
                )
            )

    # Deadline misses
    task_hours: Dict[str, List[float]] = {}
    for entry in ledger:
        task_hours.setdefault(entry.task_id, []).append(
            (entry.start_time or 0.0) + entry.allocated_hours
        )
    deadline_misses = []
    task_map = {t.id: t for t in tasks}
    for tid, ends in task_hours.items():
        if tid in task_map and max(ends) > task_map[tid].deadline:
            deadline_misses.append(tid)

    return ScheduleResult(
        scheduled=ledger,
        unscheduled=unscheduled,
        deadline_misses=deadline_misses,
        dependency_violations=[],
        resource_usage=resource_usage,
    )


if __name__ == "__main__":
    from display import print_schedule

    tasks, calendar = generate_test_data()
    greedy_result = run_greedy_scheduler_structured(tasks, calendar)
    print_schedule(greedy_result, tasks, calendar, title="Greedy")
