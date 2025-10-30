# AmpC 500K UCB Validation Script

Standalone validation script for the AmpC 500K dataset using UCB acquisition strategy.

## Overview

This script performs a complete active learning validation on the AmpC 500K screening dataset (500,000 compounds) using:

- **Strategy**: UCB (Upper Confidence Bound) with β=1.0
- **Learner**: RF Ensemble (Random Forest Ensemble)
- **Featurizer**: Morgan fingerprints (2048-bit, radius=2)
- **Cycles**: 10
- **Batch Size**: 1% per cycle (5,000 compounds)

## Requirements

- **Python**: 3.11+
- **Memory**: 8-12 GB RAM recommended
- **Runtime**: 40-60 minutes (first run with cache generation)
- **Disk Space**: ~600 MB for feature cache

## Dataset Location

The script expects the AmpC 500K dataset at:
```
/home/tony/Compound_Libraries/LearnM8_datasets/AmpC/subsampled_data/AmpC_screen_500K.csv
```

This is pre-configured in `validation/lib/dataset_config.py` as `'ampc_500k'`.

## Usage

### Run the Script

From the LearnM8 root directory:

```bash
python validation/scripts/validate_ampc_500k_ucb.py
```

Or make it executable and run directly:

```bash
chmod +x validation/scripts/validate_ampc_500k_ucb.py
./validation/scripts/validate_ampc_500k_ucb.py
```

### Output Location

Results are saved to:
```
validation/reports/ampc_500k_ucb_validation/
├── data/
│   └── ampc_500k_ucb_beta_1.0_<timestamp>/
│       ├── compounds_final.csv           # All compounds with predictions
│       ├── cycle_metrics.csv             # Per-cycle performance metrics
│       └── selection_history.csv         # Compounds selected each cycle
├── plots/
│   └── ampc_500k_ucb_validation.png     # 4-panel validation plot
└── summary.txt                           # Text summary of results
```

Each run creates a timestamped directory to avoid overwriting previous results.

## Output Files

### 1. compounds_final.csv
Contains all compounds with:
- Original data (ID, SMILES, activity)
- Predictions for each cycle (`prediction_cycle_0`, `prediction_cycle_1`, ...)
- Uncertainty estimates (`uncertainty_cycle_0`, ...)
- Selection status and cycle information

### 2. cycle_metrics.csv
Per-cycle metrics including:
- **Discovery Rates**: Top-10, Top-100, Top-0.1%, Top-1%
- **Score Ratios**: Cumulative and batch average score ratios
- **Model Quality**: Spearman correlation, Top-K overlaps
- **Pool Statistics**: Remaining unlabeled, cumulative labeled

### 3. selection_history.csv
Tracks all selected compounds with:
- Selection cycle
- Measured value
- Prediction and uncertainty at selection
- Strategy used

### 4. ampc_500k_ucb_validation.png
4-panel comprehensive validation plot:
- **Top 4 panels**: Uncertainty vs Prediction snapshots (cycles 1, 3, 6, 9)
- **Middle panel**: Discovery metrics over cycles
- **Bottom panel**: Score ratio metrics over cycles

### 5. summary.txt
Human-readable summary with:
- Configuration details
- Final performance metrics
- Runtime statistics
- File locations

## Performance Expectations

### First Run (with cache generation)
- **Feature extraction**: ~10-15 minutes
- **Per cycle**: ~3-5 minutes
- **Total time**: ~40-60 minutes

### Subsequent Runs (cached features)
- **Feature extraction**: <1 minute (cached)
- **Per cycle**: ~3-5 minutes
- **Total time**: ~30-40 minutes

### Discovery Rates (Expected)
Based on UCB β=1.0 with RF Ensemble on AmpC:
- Top-10 discovery: ~10-30% (find 1-3 of top-10 compounds)
- Top-100 discovery: ~30-50%
- Top-1% discovery: ~50-70%

## Customization

To modify parameters, edit the script's configuration section:

```python
# Configuration (lines 141-145)
BETA = 1.0                      # UCB exploration parameter
N_CYCLES = 10                   # Number of cycles
BATCH_FRACTION = 0.01           # 1% per cycle
INITIAL_BATCH_FRACTION = 0.01   # Initial random sample
```

## Troubleshooting

### Dataset Not Found
If you see "Dataset not found" error:
1. Check the dataset path in the error message
2. Verify the file exists: `ls -lh /home/tony/Compound_Libraries/LearnM8_datasets/AmpC/subsampled_data/AmpC_screen_500K.csv`
3. Update the path in `validation/lib/dataset_config.py` if needed

### Memory Errors
If the script crashes with memory errors:
1. Ensure at least 8 GB RAM is available
2. Close other memory-intensive applications
3. Consider reducing the dataset size or batch fraction

### Slow Feature Extraction
First run will be slower due to feature caching:
1. Features are cached in `.shared_cache/` directory
2. Subsequent runs will be much faster
3. Cache persists across runs and experiments

## Differences from 30K Validation

Key differences when working with 500K dataset:

| Aspect | AmpC 30K | AmpC 500K |
|--------|----------|-----------|
| Dataset size | 29,054 compounds | 500,000 compounds |
| ID column | `'ID'` | `'zincid'` |
| Runtime | ~5-10 minutes | ~40-60 minutes |
| Cache size | ~50 MB | ~500-600 MB |
| Batch size (1%) | ~290 compounds | ~5,000 compounds |

**CRITICAL**: The 500K dataset uses `'zincid'` as the ID column, not `'ID'`. The script handles this automatically via the metadata.

## Integration with Existing Validation

This script is **completely independent** of the existing validation infrastructure:

- ✅ Does NOT modify existing validation scripts
- ✅ Does NOT alter configuration files
- ✅ Uses separate output directory
- ✅ Uses shared cache (compatible)
- ✅ Can run in parallel with other validations

## Next Steps

After running the validation:

1. **Review Results**: Check `summary.txt` for quick overview
2. **Analyze Plot**: Open the validation plot to visualize performance
3. **Compare Metrics**: Compare with 30K results or other strategies
4. **Export Data**: Use CSV files for custom analysis
5. **Share Results**: All files are self-contained and shareable

## Support

For issues or questions:
- Check the main LearnM8 documentation: `README.md`
- Review validation architecture: `ARCHITECTURE_V1.md`
- Inspect the script: Comments explain each step
