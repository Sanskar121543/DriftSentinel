"""
DriftSentinel -- Master Benchmark Runner

Usage:
    python -m benchmarks.run_all              # all benchmarks
    python -m benchmarks.run_all --fast       # 2 trials per condition (CI)
    python -m benchmarks.run_all --mttd-only
    python -m benchmarks.run_all --strategy-only
    python -m benchmarks.run_all --skip-tests
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import time

BANNER = """
+----------------------------------------------------------+
|         DriftSentinel -- Benchmark Suite                  |
|  Validates all documented performance claims              |
+----------------------------------------------------------+
"""


def run_mttd_benchmark(n_trials: int = 5) -> dict:
    from benchmarks.drift_injection_benchmark import BenchmarkSuite
    df = BenchmarkSuite(n_trials_per_condition=n_trials).run()
    detected = df[df["alert_fired"]]
    overall  = detected["mttd_hours"].mean() if len(detected) else float("inf")
    target   = 4.0
    return {
        "name":   "Mean Time to Detect (MTTD)",
        "metric": f"{overall:.2f}h",
        "target": "<= 4.0h",
        "passed": overall <= target,
        "detail": f"{len(detected)}/{len(df)} trials detected drift",
    }


def run_strategy_benchmark() -> dict:
    from benchmarks.strategy_eval_benchmark import StrategyBenchmark
    r = StrategyBenchmark().run()
    return {
        "name":   "Strategy Selector Accuracy",
        "metric": f"{r['accuracy']:.1%}",
        "target": ">= 94.0%",
        "passed": r["passed"],
        "detail": f"{r['correct']}/{r['total']} correct",
    }


def run_unit_tests() -> dict:
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_core.py", "-q", "--tb=short"],
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    lines   = res.stdout.strip().split("\n")
    summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l), "")
    return {
        "name":   "Unit Test Suite",
        "metric": f"{elapsed:.1f}s",
        "target": "All pass",
        "passed": res.returncode == 0,
        "detail": summary.strip(),
    }


def print_results(results: list[dict]) -> bool:
    print(f"\n{'='*65}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Benchmark':<35} {'Result':<12} {'Target':<14} Status")
    print(f"  {'-'*62}")
    all_passed = True
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        if not r["passed"]:
            all_passed = False
        print(f"  {r['name']:<35} {r['metric']:<12} {r['target']:<14} {status}")
        if r.get("detail"):
            print(f"    -> {r['detail']}")
    print(f"\n{'='*65}")
    print("  ALL BENCHMARKS PASSED" if all_passed else "  SOME BENCHMARKS FAILED")
    print(f"{'='*65}\n")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",          action="store_true")
    parser.add_argument("--mttd-only",     action="store_true")
    parser.add_argument("--strategy-only", action="store_true")
    parser.add_argument("--skip-tests",    action="store_true")
    args = parser.parse_args()

    print(BANNER)
    n_trials = 2 if args.fast else 5
    results  = []

    if not args.strategy_only:
        print(f"[1/3] MTTD benchmark ({n_trials} trials per condition)...")
        results.append(run_mttd_benchmark(n_trials))

    if not args.mttd_only:
        print("[2/3] Strategy selector benchmark...")
        results.append(run_strategy_benchmark())

    if not args.skip_tests and not (args.mttd_only or args.strategy_only):
        print("[3/3] Unit tests...")
        results.append(run_unit_tests())

    ok = print_results(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
