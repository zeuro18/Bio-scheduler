import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Global Constants (Scoring Weights) ───────────────────────────────
W = 10      # max urgency boost the panic term can add
K = 0.05    # decay rate — controls how sharply urgency ramps up

# ──────────────────────────────────────────────────────────────────────
# 2. Data Models
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """Blueprint for a schedulable task.
    
     Use string IDs instead of int ,easier to debug
     Add cognitive_weight (1-10) for fatigue modeling in Phase 2+
    """
    id: str                              # unique task identifier
    name: str                            # human-readable name
    duration: float                      # total hours of work required
    deadline: float                      # hours from now (hard deadline)
    priority: int                        # 1 (low) – 10 (high)
    cognitive_weight: int = 5            # 1-10 scale; how mentally draining (Phase 2+)
    dependencies: List[str] = field(default_factory=list)  # task IDs that must finish first

    def is_valid(self) -> bool:
        """Check basic feasibility: positive duration/deadline and
        enough time window to finish. Also validate new fields."""
        return (
            self.duration > 0
            and self.deadline > 0
            and self.duration <= self.deadline
            and 1 <= self.priority <= 10
            and 1 <= self.cognitive_weight <= 10
        )


@dataclass(frozen=True)
class Resource:
    """A single calendar slot (e.g. one day of the week).
    
    FIX 3: Make Resource frozen (immutable) — prevents side-effect bugs in Phase 3+
           The scheduler returns a new Schedule object instead of mutating Resources
    """
    id: str                  # unique resource identifier
    name: str                # human label (e.g. "Day 1 (Monday)")
    capacity: float          # total available hours


@dataclass
class ScheduledTask:
    """One row in the output ledger: 'task X gets Y hours on resource Z'.
    
      Add start_time so we know exactly WHEN the task happens 
           Without start_time, CP-SAT can't encode ordering or interval constraints
    """
    task_id: str
    resource_id: str
    allocated_hours: float
    start_time: Optional[float] = None  # hour at which this task begins (Phase 3+)


# ──────────────────────────────────────────────────────────────────────
# 3. Math Helper — Sorting Heuristic
# ──────────────────────────────────────────────────────────────────────

def calculate_hybrid_score(task: Task) -> float:
    """ 'panic' multiplier: it stays near-zero
      while the deadline is far away, then spikes sharply as the deadline
      closes in
    """
    return float(task.priority + (W*math.exp(-K*task.deadline)))
def generate_dynamic_calendar(tasks: List[Task]) -> List[Resource]:
    """Generate a calendar with resource IDs matching the new string ID scheme."""
    if not tasks:
        return []

    # 1. Find the furthest deadline in hours
    max_deadline = max(task.deadline for task in tasks)
    
    # 2. Convert that into days 
    # If max deadline is 336 hours (2 weeks), this gives 14 days
    total_days_needed = math.ceil(max_deadline / 24.0)
    
    # 3. Define your weekly repeating template
    # Realistic capacity: you can work 5-6h on weekdays, more on weekends
    weekly_capacities = [5, 5, 6, 6, 5, 7, 7] 
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    calendar = []
    
    # 4. Generate the exact number of days needed
    for day_index in range(total_days_needed):
        day_of_week = day_index % 7  # Loops back to 0 every 7 days
        
        calendar.append(
            Resource(
                id=f"day_{day_index}",  # string ID
                # Output e.g., "Day 1 (Monday)", "Day 8 (Monday)"
                name=f"Day {day_index + 1} ({day_names[day_of_week]})", 
                capacity=weekly_capacities[day_of_week]
            )
        )
        
    return calendar

# ──────────────────────────────────────────────────────────────────────
# 4. Test Data Generator
# ──────────────────────────────────────────────────────────────────────

def generate_test_data() -> Tuple[List[Task], List[Resource]]:
    """Return a realistic (tasks, calendar) pair modeled on an actual
    week of work.
    
   Use string IDs and assign cognitive_weight to each task
    """

    #Inbox of tasks (real workload) 
    tasks: List[Task] = [
        Task(
            id="dl_test",
            name="Deep Learning Class test",
            duration=10,         # reduced from 12
            deadline=200,       
            priority=10,
            cognitive_weight=9,   # very mentally demanding
        ),
        Task(
            id="thesis",
            name="thesis",
            duration=8,          # reduced from 10
            deadline=100,      
            priority=8,
            cognitive_weight=8,   # intellectually heavy
        ),
        Task(
            id="design_lab",
            name="Design Lab",
            duration=5,
            deadline=120,      
            priority=6,
            cognitive_weight=6,   # moderate
        ),
        Task(
            id="sma_paper",
            name="SMA Paper",
            duration=4,         
            deadline=72,         # 3 days
            priority=7,          
            cognitive_weight=3,   # lighter work
        ),
        Task(
            id="magnetism_paper",
            name="Magnetism term paper",
            duration=6,          
            deadline=200,      
            priority=5,
            cognitive_weight=4,   # fairly easy course
        )
    ]
    return tasks, generate_dynamic_calendar(tasks)


# Greedy Scheduler

def run_greedy_scheduler(tasks: List[Task], calendar: List[Resource]) -> Tuple[List[ScheduledTask], dict]:
    """Greedy earliest-deadline-first scheduler.
    
    Returns (ledger, resource_usage) instead of mutating calendar
           for Phase 3+ where CP-SAT manages all state
    """
    
    # Validate every task
    for t in tasks:
        if not t.is_valid():
            raise ValueError(
                f"Task '{t.name}' (id={t.id}) failed validation: "
                f"duration={t.duration}, deadline={t.deadline}"
            )

    # Sort by hybrid score (highest first) 
    sorted_tasks = sorted(tasks, key=calculate_hybrid_score, reverse=True)

    # Track resource usage without mutating the immutable Resource objects
    resource_usage = {r.id: 0.0 for r in calendar}
    final_ledger: List[ScheduledTask] = []

    for task in sorted_tasks:
        remaining = task.duration
        deadline_day_idx = int(task.deadline // 24)
        
        # allocate FORWARD from day 0, stopping at deadline
        for day_idx in range(len(calendar)):
            if remaining <= 0:
                break
            
            # Don't schedule past the deadline
            if day_idx > deadline_day_idx:
                break
            
            day = calendar[day_idx]
            available = day.capacity - resource_usage[day.id]
            
            if available <= 0:
                continue

            allocated = min(remaining, available)
            resource_usage[day.id] += allocated
            remaining -= allocated

            final_ledger.append(
                ScheduledTask(
                    task_id=task.id,
                    resource_id=day.id,
                    allocated_hours=allocated,
                    start_time=None,  # greedy doesn't compute exact times
                )
            )

        if remaining > 0:
            print(
                f" Warning: Task '{task.name}' has {remaining:.1f}h "
                f"still unscheduled (ran out of calendar capacity)."
            )

    return final_ledger, resource_usage


if __name__ == "__main__":
    tasks, calendar = generate_test_data()
    ledger, resource_usage = run_greedy_scheduler(tasks, calendar)

    # Build quick lookups for prettier output
    task_map = {t.id: t.name for t in tasks}
    day_map = {r.id: r.name for r in calendar}   
    print("WEEKLY SCHEDULE (Greedy Allocation)")
    
    # Group by day
    current_day = None
    for entry in sorted(ledger, key=lambda e: (e.resource_id, e.task_id)):
        day_name = day_map[entry.resource_id]
        if day_name != current_day:
            current_day = day_name
            print(f"\n  {current_day}")
            print("  " + "-" * 36)
        print(f"   • {task_map[entry.task_id]:.<35s} {entry.allocated_hours:.1f}h")

    # Summary per day
  
    print("\nDAY-BY-DAY UTILISATION")
    
    for day in calendar:
        hours_used = resource_usage[day.id]
        bar = "█" * int(hours_used) + "░" * int(day.capacity - hours_used)
        pct = (hours_used / day.capacity * 100) if day.capacity > 0 else 0
        print(f"  {day.name:<20s}  {bar}  {hours_used:.1f}/{day.capacity:.0f}h ({pct:.0f}%)")