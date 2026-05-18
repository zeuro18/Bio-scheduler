"""Quick manual check: parse sample text and print tasks + user_context."""

import os
import sys

from dotenv import load_dotenv

_parser_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_parser_dir, ".."))
for _p in (_parser_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv(os.path.join(_parser_dir, ".env"), override=True)

from models import generate_test_data
from true_llm_parser import NLTaskParser


def main() -> None:
    _, cal = generate_test_data()
    parser = NLTaskParser(verbose=True)
    result = parser.parse(
        "I'm exhausted today, lab from 2 to 5, but I can work late. "
        "Lab report: 3 hours, due in 24h, priority 8.",
        base_calendar=cal,
    )

    print("tasks:", [(t.name, t.duration) for t in result.tasks])
    print("energy:", result.user_context.energy_level)
    print("blocked:", result.user_context.blocked_hours)
    print("pressure:", result.user_context.deadline_pressure_mode)
    print("warnings:", result.warnings[:5])


if __name__ == "__main__":
    main()
