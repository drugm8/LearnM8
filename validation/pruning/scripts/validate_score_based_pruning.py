#!/usr/bin/env python3
from pathlib import Path

from validation.lib.validation_runner import PruningValidationRunner
from validation.lib.plot_generator import (
    create_pruning_efficiency_timeline,
    create_discovery_efficiency_scatter,
    create_model_quality_facets,
    create_uncertainty_prediction_snapshots,
    create_score_ratio_evolution
)


def main():
    runner = PruningValidationRunner('score_based_pruning')

    print(f"\nRunning score-based pruning validation...")
    results = runner.run_parameter_sweep()

    output_dir = Path('validation/reports/pruning') / 'score_based_pruning'
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    strategy_labels = {
        0.0: 'No Pruning (0%)',
        0.15: 'Light Pruning (15%)',
        0.3: 'Heavy Pruning (30%)'
    }

    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)

    plots = {}

    print("\n[1/5] Pruning Efficiency Timeline...")
    plots['efficiency_timeline'] = create_pruning_efficiency_timeline(
        results,
        strategy_labels,
        plots_dir / 'pruning_efficiency_timeline.png'
    )

    print("\n[2/5] Discovery Efficiency Scatter...")
    plots['discovery_efficiency'] = create_discovery_efficiency_scatter(
        results,
        strategy_labels,
        plots_dir / 'discovery_efficiency_scatter.png'
    )

    print("\n[3/5] Model Quality Facets...")
    plots['model_quality'] = create_model_quality_facets(
        results,
        strategy_labels,
        plots_dir / 'model_quality_facets.png'
    )

    print("\n[4/5] Uncertainty-Prediction Snapshots (Greedy Strategy)...")
    greedy_result = results[0.15]
    plots['uncertainty_snapshots'] = create_uncertainty_prediction_snapshots(
        greedy_result,
        'Light Pruning (15%)',
        plots_dir / 'uncertainty_prediction_snapshots.png',
        cycles_to_show=[1, 3, 6, 9]
    )

    print("\n[5/5] Score Ratio Evolution...")
    plots['score_ratio'] = create_score_ratio_evolution(
        results,
        strategy_labels,
        plots_dir / 'score_ratio_evolution.png'
    )

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print(f"\n✓ Validation Complete!")
    print(f"\nOutputs:")
    print(f"  Data: {output_dir / 'data'}")
    print(f"  Plots: {plots_dir}")
    print(f"\nGenerated {len(plots)} plots:")
    for plot_name, plot_path in plots.items():
        print(f"  - {plot_path.name}")

    print(f"\nFinal Metrics Summary:")
    print(f"{'Strategy':<25} {'Final Pool':<15} {'Top-100 Disc':<15} {'Top-1000 Overlap':<20}")
    print("-" * 75)

    for pruning_frac in sorted(results.keys()):
        result = results[pruning_frac]
        cycle_metrics = result['cycle_metrics']
        final_metrics = cycle_metrics[-1]

        label = strategy_labels[pruning_frac]
        pool_size = final_metrics.get('pool_size', 'N/A')
        top_100_disc = final_metrics.get('top_100_discovery', 0.0)
        top_1000_overlap = final_metrics.get('unlabeled_top_1000_overlap', 0.0)

        print(f"{label:<25} {pool_size:<15,} {top_100_disc:<15.2f}% {top_1000_overlap:<20.2f}%")


if __name__ == '__main__':
    main()
