#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from datetime import datetime

STRATEGIES = ['ucb', 'ei', 'pi', 'thompson', 'entropy', 'simulated_annealing']

STRATEGY_NAMES = {
    'ucb': 'Upper Confidence Bound',
    'ei': 'Expected Improvement',
    'pi': 'Probability of Improvement',
    'thompson': 'Thompson Sampling',
    'entropy': 'Entropy-Based Selection',
    'simulated_annealing': 'Simulated Annealing'
}


def main():
    print("="*60)
    print("RUNNING ALL ACQUISITION STRATEGY VALIDATIONS")
    print("="*60)
    print(f"\nStrategies to validate: {', '.join(s.upper() for s in STRATEGIES)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}
    start_time = datetime.now()

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()} ({STRATEGY_NAMES[strategy]})")
        print(f"{'='*60}")

        script = Path(__file__).parent / f"validate_{strategy}.py"

        if not script.exists():
            print(f"  ✗ Script not found: {script}")
            results[strategy] = False
            continue

        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=False,
                check=True
            )
            results[strategy] = True
            print(f"\n  ✓ {strategy.upper()} validation completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"\n  ✗ {strategy.upper()} validation failed with exit code {e.returncode}")
            results[strategy] = False

        except Exception as e:
            print(f"\n  ✗ {strategy.upper()} validation failed: {e}")
            results[strategy] = False

    elapsed = datetime.now() - start_time

    generate_summary_index(results, elapsed)

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    for strategy, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {strategy.upper():12} ({STRATEGY_NAMES[strategy]:30}): {status}")

    print(f"\nTotal execution time: {elapsed}")
    print(f"Summary report: validation/reports/README.md")
    print(f"{'='*60}")

    if all(results.values()):
        print("\n✓ All validations passed!")
        exit(0)
    else:
        failed = [s for s, success in results.items() if not success]
        print(f"\n✗ Some validations failed: {', '.join(failed)}")
        exit(1)


def generate_summary_index(results, elapsed):
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'validation' / 'reports' / 'README.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# Acquisition Strategy Validation Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Execution Time:** {elapsed}
**Dataset:** AmpC 30K (docking scores)
**Configuration:** `validation/config/validation_strategies.yaml`

---

## Validation Results

| Strategy | Full Name | Status | Report | Plots |
|----------|-----------|--------|--------|-------|
"""

    for strategy, success in results.items():
        status = "✓ Pass" if success else "✗ Fail"
        full_name = STRATEGY_NAMES[strategy]
        report_link = f"[Report]({strategy}/validation_report.md)"
        plots_link = f"[Plots]({strategy}/plots/)"
        content += f"| {strategy.upper()} | {full_name} | {status} | {report_link} | {plots_link} |\n"

    content += f"""
---

## Quick Links

### Individual Strategy Reports

- **[UCB (Upper Confidence Bound)](ucb/validation_report.md)** - Confidence interval-based exploration
- **[EI (Expected Improvement)](ei/validation_report.md)** - Expected value of improvement
- **[PI (Probability of Improvement)](pi/validation_report.md)** - Conservative improvement probability
- **[Thompson Sampling](thompson/validation_report.md)** - Stochastic posterior sampling
- **[Entropy](entropy/validation_report.md)** - Pure uncertainty-based exploration
- **[Simulated Annealing](simulated_annealing/validation_report.md)** - Temperature-based optimization exploration

---

## Validation Framework

### Configuration

All validations use consistent settings defined in `validation/config/validation_strategies.yaml`:

- **Dataset:** AmpC 30K benchmark (30,000 compounds)
- **Cycles:** 10 active learning cycles
- **Initial Fraction:** 1% (300 compounds)
- **Batch Fraction:** 1% (300 compounds per cycle)
- **Learner:** RF Ensemble (100 trees, 3 random states)
- **Featurizer:** Morgan fingerprints (2048-bit, radius=2)
- **Mode:** Benchmark (with discovery and ranking metrics)

### Metrics Tracked

**Discovery Metrics (Category A):**
- Top-10, Top-100, Top-0.1%, Top-1% discovery rates
- Cumulative and batch enrichment factors
- Average score ratios

**Unlabeled Ranking Metrics (Category B):**
- Spearman correlation on unlabeled pool
- Top-1000 and Top-100 overlaps
- Unlabeled enrichment factors

### Execution

```bash
# Run all validations
cd validation/scripts
python run_all_validations.py

# Run individual validation
python validate_ucb.py
python validate_ei.py
# etc.
```

---

## Notes

- All validations support checkpoint/resume (skip-if-exists logic)
- Animations require FFmpeg (graceful fallback to skip if unavailable)
- Reports are Markdown files with linked PNG plots
- Raw data saved in `validation/reports/{{strategy}}/data/` directories

---

*Generated by LearnM8 validation framework v1.0.0*
"""

    output_file.write_text(content)
    print(f"\n✓ Summary index generated: {output_file}")


if __name__ == '__main__':
    main()
