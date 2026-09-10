"""CI gate: the Airflow DAG graph is valid without installing Airflow."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# In CI the package is pip-installed; locally it may only exist in src/.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "airflow" / "dags"))

import retail_ml_pipeline as pipeline  # noqa: E402


def main() -> int:
    problems = pipeline.validate_graph()
    print(f"tasks: {len(pipeline.TASK_DEPENDENCIES)}")

    # Training must never be reachable without passing the leakage gate.
    def ancestors(task: str) -> set:
        found, stack = set(), list(pipeline.TASK_DEPENDENCIES.get(task, []))
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(pipeline.TASK_DEPENDENCIES.get(current, []))
        return found

    if "leakage_check" not in ancestors("backtest_models"):
        problems.append("backtest_models is reachable without the leakage gate")

    # The task callables must run standalone, outside a scheduler.
    print(pipeline.validate_data())
    print(pipeline.leakage_check())

    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
