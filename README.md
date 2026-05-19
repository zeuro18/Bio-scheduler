# Biologically Aware Human Scheduler

Bio Scheduler turns plain English task descriptions into a calendar-aware study and work schedule. It parses tasks, deadlines, energy level, blocked time, dependencies, and schedule pressure, then solves the schedule with either CP-SAT or NSGA-II.

The project is designed for personal academic planning where not every hour is equal. A task can be urgent, cognitively heavy, dependent on another task, or blocked by meetings, labs, low energy, and limited daily capacity.

## Features

- Natural language task input through Groq.
- Structured task extraction with duration, deadline, priority, cognitive weight, and dependencies.
- Context extraction for low energy, blocked hours, capacity limits, and willingness to work late.
- Dynamic calendar generation based on parsed deadlines.
- Work slots for each day, with default morning, afternoon, and evening windows.
- CP-SAT solver using OR-Tools for hard scheduling constraints.
- NSGA-II optimizer using pymoo for multi-objective schedule quality.
- Greedy scheduler for baseline comparison.
- Dependency handling through a DAG engine.
- Calendar spread resolution for inputs like "will take 3 days" or "need 2 evenings".
- Preference learning from feedback, stored in `preferences.json`.
- Schedule validation for duration, overlap, dependencies, deadlines, capacity, and work slots.
- Objective scoring for fatigue, context switching, deadline risk, and fragmentation.
- Console schedule display with day-by-day utilization.

## How It Works

1. `run_e2e.py` accepts plain English input.
2. `LLM Parser/true_llm_parser.py` calls Groq and converts the response into structured tasks and user context.
3. `postprocessing.py` normalizes the parser output and filters infeasible tasks.
4. `calendar_spread_resolver.py` converts spread-based tasks into work hours using calendar capacity.
5. `models.py` builds a dynamic calendar from the task deadlines.
6. The selected solver creates a schedule:
   - `solver_cpsat.py` finds a feasible schedule using OR-Tools CP-SAT.
   - `solver_nsga.py` searches for better schedules with NSGA-II, then validates the result.
7. `validate.py` checks hard constraints independently.
8. `display.py` prints the final schedule.

## Requirements

- Python 3.11 or newer recommended
- Groq API key for natural language parsing
- Python packages from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root or inside `LLM Parser/`:

```env
GROQ_API_KEY=your_key_here
```

Do not commit `.env` to GitHub.

## Quick Start

Run the default NSGA-II scheduler:

```bash
python run_e2e.py "Thesis 8 hours due in 5 days, priority 9, cognitive weight 8. Lab tomorrow 2-5pm. I am tired today."
```

Run CP-SAT only:

```bash
python run_e2e.py --optimizer cpsat "Write report 3 hours due in 24 hours, priority 8."
```

Use the CP-SAT alias:

```bash
python run_e2e.py --cpsat-only "Prepare slides 4 hours due in 48 hours, priority 7."
```

Tune NSGA-II:

```bash
python run_e2e.py --optimizer nsga --generations 100 --population 100 "Study for exam 10 hours due in 8 days, priority 10, cognitive weight 9."
```

Provide feedback for preference learning:

```bash
python run_e2e.py --feedback "last schedule was too tiring and had too much switching" "Finish paper 6 hours due in 72 hours, priority 8."
```

Disable preference loading and saving:

```bash
python run_e2e.py --no-preferences "Email professor 0.5 hours due in 2 hours, priority 10, cognitive weight 1."
```

Interactive mode:

```bash
python run_e2e.py
```

## Input Format

The parser works best when the input includes:

- task name
- effort in hours, or a spread hint such as "3 days" or "2 evenings"
- deadline such as "in 48 hours" or "in 5 days"
- priority from 1 to 10
- cognitive weight from 1 to 10
- dependencies if one task must happen after another
- context like tiredness, meetings, labs, blocked time, or ability to work late

Example:

```text
Finish SMA paper, 4 hours, due in 72 hours, priority 7, cognitive weight 3.
Design Lab project, 5 hours, due in 120 hours, priority 6, cognitive weight 6.
Design Lab starts after SMA paper.
I have lab today from 2pm to 5pm and I am low energy.
```

## Main Files

| File | Purpose |
| --- | --- |
| `run_e2e.py` | Command-line entry point for parsing, solving, validation, and display |
| `models.py` | Task, resource, scheduled task, and calendar data models |
| `solver_cpsat.py` | OR-Tools CP-SAT solver with hard constraints and relaxation tiers |
| `solver_nsga.py` | NSGA-II optimizer and Pareto candidate selection |
| `objectives.py` | Fatigue, context switching, deadline risk, and fragmentation scoring |
| `validate.py` | Independent hard constraint checker |
| `context.py` | User context model and calendar adjustment logic |
| `hour_index.py` | Converts work slots into absolute schedulable hours |
| `display.py` | Console timetable and utilization printer |
| `preferences.py` | Feedback-based preference profile loading, saving, and updating |
| `compare.py` | Greedy vs CP-SAT comparison utility |
| `dag.py` | Dependency graph and topological ordering |
| `evaluate.py` | Schedule quality and feasibility metrics |
| `test_solver.py` | CP-SAT smoke tests and solver comparison |
| `LLM Parser/true_llm_parser.py` | Groq-backed natural language parser |
| `LLM Parser/postprocessing.py` | Parser output normalization and feasibility checks |
| `LLM Parser/calendar_spread_resolver.py` | Converts spread-based tasks into effort hours |
| `LLM Parser/BUILDPROMPT.py` | Prompt and few-shot examples for structured extraction |

## Solver Details

### CP-SAT

The CP-SAT solver enforces:

- exact task duration using whole-hour allocation
- no overlapping tasks
- daily capacity limits
- dependency ordering
- hard deadlines
- work-slot availability
- maximum number of active days per task

If the first solve fails, it tries controlled relaxation:

1. original constraints
2. mild overtime
3. moderate overtime
4. extreme overtime
5. more task fragmentation
6. dropping low-priority tasks when needed

### NSGA-II

The NSGA-II path searches for schedules that balance:

- lower fatigue overload
- fewer context switches
- lower deadline risk
- less fragmentation across days

It uses CP-SAT as a feasibility fallback when configured through `run_e2e.py`.

## Testing

Run the CP-SAT smoke tests:

```bash
python test_solver.py
```

Run parser tests:

```bash
python "LLM Parser/test_parser.py"
```

Parser tests require `GROQ_API_KEY`.

## Notes

- The solver currently works in discrete whole-hour blocks.
- Fractional task durations are rounded up for scheduling.
- Deadlines are represented as hours from the current moment.
- `preferences.json` is created only when preference learning is used.
- `nsga_scheduler.py` appears to be an older or alternate NSGA implementation. The main CLI uses `solver_nsga.py`.

## GitHub

Repository:

```text
https://github.com/zeuro18/Bio-scheduler
```

Recommended branch name:

```text
main
```
