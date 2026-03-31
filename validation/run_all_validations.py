#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VALIDATION_ROOT = Path(__file__).parent
PROJECT_ROOT = VALIDATION_ROOT.parent

CATEGORIES = [
    "acquisition_strategies",
    "learner_evaluation",
    "large_scale",
    "uncertainty",
    "performance",
    "pruning",
]


def discover_scripts(category_filter=None):
    categories_to_scan = [category_filter] if category_filter else CATEGORIES
    discovered = {}

    for category in categories_to_scan:
        scripts_dir = VALIDATION_ROOT / category / "scripts"
        if not scripts_dir.exists():
            continue

        scripts = sorted(scripts_dir.glob("validate_*.py"))

        if category == "performance" and category_filter == "performance":
            scripts = list(scripts)
            scripts += sorted(scripts_dir.glob("benchmark_*.py"))
            speedup_dir = VALIDATION_ROOT / category / "speedup" / "scripts"
            if speedup_dir.exists():
                scripts += sorted(speedup_dir.glob("benchmark_*.py"))

        if scripts:
            discovered[category] = scripts

    return discovered


def main():
    parser = argparse.ArgumentParser(description="Run LearnM8 validation scripts")
    parser.add_argument("--list", action="store_true", help="List discovered scripts and exit")
    parser.add_argument("--category", metavar="NAME", help="Run only scripts in this category")
    args = parser.parse_args()

    discovered = discover_scripts(args.category)

    if args.list:
        for category, scripts in discovered.items():
            for script in scripts:
                print(script)
        return

    print("=" * 60)
    print("RUNNING ALL VALIDATIONS")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}
    start_time = datetime.now()

    for category, scripts in discovered.items():
        print(f"\n{'=' * 60}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'=' * 60}")

        for script in scripts:
            print(f"\n  Running: {script.name}")
            try:
                subprocess.run(
                    [sys.executable, str(script)],
                    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                    check=True,
                )
                results[str(script)] = True
                print(f"  PASS: {script.name}")
            except subprocess.CalledProcessError as e:
                print(f"  FAIL: {script.name} (exit code {e.returncode})")
                results[str(script)] = False
            except Exception as e:
                print(f"  FAIL: {script.name} ({e})")
                results[str(script)] = False

    elapsed = datetime.now() - start_time

    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    for script_path, success in results.items():
        name = Path(script_path).name
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\n{passed}/{total} scripts passed")
    print(f"Total execution time: {elapsed}")
    print(f"{'=' * 60}")

    if all(results.values()) if results else True:
        print("\nAll validations passed!")
        sys.exit(0)
    else:
        failed = [Path(s).name for s, ok in results.items() if not ok]
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
