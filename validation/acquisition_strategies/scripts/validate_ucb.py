#!/usr/bin/env python3

from pathlib import Path

from validation.lib import (
    ValidationRunner,
    create_comprehensive_validation_plot,
    create_animations,
    MarkdownReportGenerator
)


def main():
    strategy_name = 'ucb'

    runner = ValidationRunner(strategy_name)

    print(f"\nRunning {runner.strategy_config['full_name']} validation...")
    results = runner.run_parameter_sweep()

    output_dir = Path('validation/reports/acquisition_strategies') / strategy_name
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plots = {}
    for param_value, result in results.items():
        plot_path = plots_dir / f'validation_plot_{strategy_name}_{param_value}.png'
        plots[param_value] = create_comprehensive_validation_plot(
            result, runner.strategy_config, param_value, plot_path
        )

    print("\nGenerating animations...")
    animations = create_animations(
        results, runner.strategy_config, output_dir,
        fps=runner.global_config['animation_fps'],
        dpi=runner.global_config['animation_dpi']
    )

    print("\nGenerating report...")
    report = MarkdownReportGenerator(strategy_name, runner.strategy_config, results)
    report.generate(output_dir / 'validation_report.md', plots, animations)

    print(f"\n{'='*60}")
    print(f"✓ {strategy_name.upper()} validation complete!")
    print(f"{'='*60}")
    print(f"Report: {output_dir / 'validation_report.md'}")
    print(f"Plots: {plots_dir}")
    if animations:
        print(f"Animations: {output_dir / 'animations'}")


if __name__ == '__main__':
    main()
