import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Scoring Weights
W = 10      # max urgency boost the panic term can add
K = 0.05    # decay rate controls how sharply urgency ramps up
DEFAULT_WORK_SLOTS: Tuple[Tuple[float, float], ...] = (
    (9.0, 12.0),   (14.0, 17.0),
    (18.0, 21.0),  
)


@dataclass
class Task:
    """Blueprint for a schedulable task.
    
     Use string IDs instead of int ,easier to debug
     Add cognitive_weight (1-10) for fatigue modeling
    """
    id: str                              
    name: str                            
    duration: float                     
    deadline: float                      
    priority: int                       
    cognitive_weight: int = 5            
    dependencies: List[str] = field(default_factory=list)  

    def is_valid(self) -> bool:
        #Check basic feasibility
        return (
            self.duration > 0
            and self.deadline > 0
            and self.duration <= self.deadline
            and 1 <= self.priority <= 10
            and 1 <= self.cognitive_weight <= 10
        )


@dataclass(frozen=True)
class Resource:
   #single calendar day with explicit work windows.
    id: str
    name: str
    capacity: float
    work_slots: Tuple[Tuple[float, float], ...] = ((0.0, 24.0),) 


@dataclass
class ScheduledTask:
    #task X gets Y hours on resource Z.
    
    task_id: str
    resource_id: str
    allocated_hours: float
    start_time: Optional[float] = None  # hour at which this task begins (Phase 3+)


def calculate_hybrid_score(task: Task) -> float:
    return float(task.priority + (W * math.exp(-K * task.deadline)))


# Time-Slot Utility

def find_earliest_slot(
    from_hour: float,
    duration: float,
    work_slots: Tuple[Tuple[float, float], ...],
    day_start_hour: float = 0.0,
) -> Optional[float]:
    #finds earliest absolute hour within work_slots where a block of duration hours fits, starting no earlier than from_hour.

   
    for slot_start, slot_end in work_slots:
        abs_start = day_start_hour + slot_start
        abs_end   = day_start_hour + slot_end

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
    day_start_hour: float = 0.0,) -> Optional[Tuple[float, float]]:
    #Return earliest slot start and allocatable chunk length.(allows partial allocation)

    for slot_start, slot_end in work_slots:
        abs_start = day_start_hour + slot_start
        abs_end = day_start_hour + slot_end
        actual_start = max(from_hour, abs_start)
        available = abs_end - actual_start
        if available > 0:
            return actual_start, min(max_duration, available)
    return None




def generate_dynamic_calendar(tasks: List[Task]) -> List[Resource]:
    #Generate a calendar whose days carry realistic work-slots.
    if not tasks:
        return []

    max_deadline = max(task.deadline for task in tasks)
    total_days_needed = math.ceil(max_deadline / 24.0)
    weekly_capacities = [5, 5, 6, 6, 5, 7, 7]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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
            dependencies=["thesis"]   
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
            dependencies=["thesis"]  
        ),
        Task(
            id="sma_paper",
            name="SMA Paper",
            duration=4,         
            deadline=72,         
            priority=7,          
            cognitive_weight=3,   
            dependencies=[]
        ),
        Task(
            id="magnetism_paper",
            name="Magnetism term paper",
            duration=6,          
            deadline=200,      
            priority=5,
            cognitive_weight=4,  
        )
    ]
    return tasks, generate_dynamic_calendar(tasks)


# Greedy Scheduler (DAG-aware)

def run_greedy_scheduler(tasks: List[Task], calendar: List[Resource]) -> Tuple[List[ScheduledTask], dict]:
   
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
            (task_finish_hour[dep] for dep in task.dependencies if dep in task_finish_hour),
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
            if child_id not in completed and all(d in completed for d in child.dependencies):
                ready.append(child)

        #sort so the highest scored ready task is always picked next
        ready.sort(key=calculate_hybrid_score, reverse=True)

    return final_ledger, resource_usage


def format_hour_24(hour_value: float) -> str:
    total_minutes = int(round(hour_value * 60))
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def describe_time_slot(day: Resource, start_hour_abs: float, end_hour_abs: float) -> str:
    #Return slot label for an allocation inside a day.
    day_start = 0.0
    start_local = (start_hour_abs - day_start) % 24
    end_local = (end_hour_abs - day_start) % 24

    for slot_start, slot_end in day.work_slots:
        if start_local >= slot_start and end_local <= slot_end:
            return f"{format_hour_24(slot_start)}-{format_hour_24(slot_end)}"

    # Fallback for partial/edge placements that don't map cleanly to one slot
    return f"{format_hour_24(start_local)}-{format_hour_24(end_local)}"


if __name__ == "__main__":
    tasks, calendar = generate_test_data()
    ledger, resource_usage = run_greedy_scheduler(tasks, calendar)

    # Build quick lookups for prettier output
    task_map = {t.id: t.name for t in tasks}
    resource_map = {r.id: r for r in calendar}
    print("WEEKLY SCHEDULE (Greedy Allocation)")
    
    # Group by day
    current_day = None
    for entry in sorted(ledger, key=lambda e: (e.resource_id, e.start_time or 0.0, e.task_id)):
        day = resource_map[entry.resource_id]
        day_name = day.name
        if day_name != current_day:
            current_day = day_name
            print(f"\n  {current_day}")
            print("  " + "-" * 36)

        if entry.start_time is None:
            time_range = "time-na"
            slot_label = "slot-na"
        else:
            start_local = entry.start_time % 24
            end_local = (entry.start_time + entry.allocated_hours) % 24
            time_range = f"{format_hour_24(start_local)}-{format_hour_24(end_local)}"
            slot_label = describe_time_slot(day, entry.start_time, entry.start_time + entry.allocated_hours)

        print(
            f"   - {task_map[entry.task_id]:.<25s} "
            f"{entry.allocated_hours:>4.1f}h  "
            f"{time_range}  "
            f"[slot {slot_label}]"
        )

    # Summary per day
  
    print("\nDAY-BY-DAY UTILISATION")
    
    for day in calendar:
        hours_used = resource_usage[day.id]
        bar = "#" * int(hours_used) + "-" * int(day.capacity - hours_used)
        pct = (hours_used / day.capacity * 100) if day.capacity > 0 else 0
        print(f"  {day.name:<20s}  {bar}  {hours_used:.1f}/{day.capacity:.0f}h ({pct:.0f}%)")