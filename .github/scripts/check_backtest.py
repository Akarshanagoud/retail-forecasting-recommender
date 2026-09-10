"""CI gate: the feature model must still beat the seasonal naive baseline.

Run as: python .github/scripts/check_backtest.py backtest.json

A gate rather than a report. If the generator stops producing learnable
structure, or the feature pipeline stops exposing it, the build fails rather
than the README quietly becoming false.
"""
import json
import sys


def main(path: str) -> int:
    rows = {row["model"]: row for row in json.load(open(path, encoding="utf-8"))}

    missing = {"seasonal_naive", "ridge"} - set(rows)
    if missing:
        print(f"backtest output is missing models: {sorted(missing)}", file=sys.stderr)
        return 1

    baseline = rows["seasonal_naive"]["wape"]
    candidate = rows["ridge"]["wape"]
    mase = rows["ridge"]["mase"]
    print(f"seasonal_naive WAPE {baseline:.4f} vs ridge {candidate:.4f} (MASE {mase:.3f})")

    failures = []
    if candidate >= baseline:
        failures.append("ridge no longer beats the seasonal naive baseline")
    if mase is None or mase >= 1.0:
        failures.append(f"ridge MASE {mase} is not below 1.0")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "backtest.json"))
