#!/usr/bin/env python3
"""Generate test fixture CSV files from validation datasets.

This script creates test data fixtures used throughout the LearnM8 test suite.
All fixtures are derived from validation/ampc_30k_subsample.csv to ensure
realistic molecular structures and activity distributions.

Usage:
    python tests/data/generate_fixtures.py

Generated Files:
    tests/data/small_molecules.csv (50 compounds)
    tests/data/medium_molecules.csv (200 compounds)
    tests/data/diverse_molecules.csv (100 compounds)
    tests/data/edge_case_molecules.csv (20 compounds)

Requirements:
    - polars
    - numpy
    - rdkit

Notes:
    - Fixed random seed (42) ensures reproducibility
    - All SMILES validated with RDKit
    - Stratified sampling preserves activity distribution
    - Diversity filtering prevents overly similar compounds
"""

import polars as pl
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from typing import List
import sys

RDLogger.DisableLog('rdApp.*')

SEED = 42
SOURCE_DATA = Path(__file__).parents[2] / "validation/ampc_30k_subsample.csv"
OUTPUT_DIR = Path(__file__).parent

np.random.seed(SEED)


def load_and_validate_source_data() -> pl.DataFrame:
    """Load ampc_30k_subsample.csv and validate SMILES with RDKit.

    Returns:
        DataFrame with validated compounds (ID, SMILES, dockscore)
    """
    print(f"Loading source data from {SOURCE_DATA}...")

    if not SOURCE_DATA.exists():
        raise FileNotFoundError(
            f"Source data not found: {SOURCE_DATA}\n"
            f"Expected validation/ampc_30k_subsample.csv"
        )

    df = pl.read_csv(SOURCE_DATA)
    print(f"  Loaded {len(df)} compounds")

    print("  Validating SMILES with RDKit...")
    valid_mask = []
    for smiles in df['SMILES'].to_list():
        mol = Chem.MolFromSmiles(smiles)
        valid_mask.append(mol is not None)

    df = df.filter(pl.Series(valid_mask))
    print(f"  {len(df)} compounds with valid SMILES")

    df = df.rename({'dockscore': 'Activity'})

    return df


def compute_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints for list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Number of bits (default: 2048)

    Returns:
        Array of shape (n_compounds, n_bits)
    """
    fps = []
    print(f"    Computing Morgan fingerprints for {len(smiles_list)} compounds...", end='', flush=True)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(np.array(fp))
    print(" done")
    return np.array(fps)


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprints.

    Args:
        fp1: First fingerprint (binary array)
        fp2: Second fingerprint (binary array)

    Returns:
        Tanimoto similarity (0.0 to 1.0)
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    return intersection / union if union > 0 else 0.0


def stratified_sample(df: pl.DataFrame, n: int, column: str = 'Activity') -> pl.DataFrame:
    """Stratified sampling across activity distribution.

    Samples evenly across percentile bins to ensure representative
    activity distribution.

    Args:
        df: Input DataFrame
        n: Number of samples to draw
        column: Column to stratify by (default: 'Activity')

    Returns:
        Stratified sample DataFrame
    """
    n_bins = 5
    samples_per_bin = n // n_bins

    percentiles = [i / n_bins for i in range(n_bins + 1)]
    quantiles = [df[column].quantile(p) for p in percentiles]

    sampled_dfs = []
    for i in range(n_bins):
        bin_df = df.filter(
            (pl.col(column) >= quantiles[i]) &
            (pl.col(column) < quantiles[i + 1] if i < n_bins - 1 else True)
        )

        if len(bin_df) >= samples_per_bin:
            sampled = bin_df.sample(n=samples_per_bin, seed=SEED + i)
        else:
            sampled = bin_df

        sampled_dfs.append(sampled)

    result = pl.concat(sampled_dfs)

    if len(result) < n:
        remaining = n - len(result)
        additional = df.filter(~pl.col('ID').is_in(result['ID'])) \
                       .sample(n=remaining, seed=SEED)
        result = pl.concat([result, additional])

    return result


def ensure_diversity(df: pl.DataFrame, max_similarity: float = 0.8) -> pl.DataFrame:
    """Filter compounds to ensure structural diversity.

    Removes compounds that are too similar (Tanimoto > threshold) to
    already selected compounds.

    Args:
        df: Input DataFrame with SMILES column
        max_similarity: Maximum allowed Tanimoto similarity (default: 0.8)

    Returns:
        Filtered DataFrame with diverse compounds
    """
    print(f"  Ensuring diversity (max similarity: {max_similarity})...")

    smiles_list = df['SMILES'].to_list()
    fps = compute_morgan_fingerprints(smiles_list)

    selected_indices = [0]

    for i in range(1, len(fps)):
        similarities = [
            tanimoto_similarity(fps[i], fps[j])
            for j in selected_indices
        ]

        if max(similarities) < max_similarity:
            selected_indices.append(i)

    print(f"  Selected {len(selected_indices)}/{len(df)} compounds after diversity filtering")

    return df[selected_indices]


def maxmin_sample(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """MaxMin sampling for maximum structural diversity.

    Iteratively selects compounds that are maximally dissimilar
    to already selected compounds (Kennard-Stone algorithm).

    Args:
        df: Input DataFrame with SMILES column
        n: Number of compounds to select

    Returns:
        DataFrame with n maximally diverse compounds
    """
    print(f"  MaxMin sampling for {n} diverse compounds...")

    smiles_list = df['SMILES'].to_list()
    fps = compute_morgan_fingerprints(smiles_list)

    selected_indices = [np.random.randint(len(fps))]

    for _ in range(n - 1):
        min_distances = []
        for i in range(len(fps)):
            if i in selected_indices:
                min_distances.append(-1)
            else:
                distances = [
                    1.0 - tanimoto_similarity(fps[i], fps[j])
                    for j in selected_indices
                ]
                min_distances.append(min(distances))

        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

    print(f"  Selected {len(selected_indices)} diverse compounds")

    return df[selected_indices]


def generate_small_molecules(df: pl.DataFrame) -> pl.DataFrame:
    """Generate small_molecules.csv (50 compounds).

    Critical fixture for learner tests. Uses stratified sampling
    with diversity filtering.

    Args:
        df: Source DataFrame (validated ampc_30k data)

    Returns:
        DataFrame with 50 diverse compounds
    """
    print("\nGenerating small_molecules.csv (50 compounds)...")

    sampled = stratified_sample(df, n=200)

    diverse = ensure_diversity(sampled, max_similarity=0.7)

    if len(diverse) >= 50:
        result = diverse[:50]
    else:
        print(f"  Warning: Only {len(diverse)} diverse compounds found, taking all")
        result = diverse

    result = result.select(['ID', 'SMILES', 'Activity'])

    print(f"  Activity range: [{result['Activity'].min():.2f}, {result['Activity'].max():.2f}]")
    print(f"  Activity mean: {result['Activity'].mean():.2f}")

    return result


def generate_medium_molecules(df: pl.DataFrame) -> pl.DataFrame:
    """Generate medium_molecules.csv (200 compounds).

    For integration tests. Includes Consensus_Score column
    (Activity + noise) to simulate multi-model predictions.

    Args:
        df: Source DataFrame (validated ampc_30k data)

    Returns:
        DataFrame with 200 compounds
    """
    print("\nGenerating medium_molecules.csv (200 compounds)...")

    sampled = stratified_sample(df, n=600)

    diverse = ensure_diversity(sampled, max_similarity=0.65)

    if len(diverse) >= 200:
        result = diverse[:200]
    else:
        print(f"  Warning: Only {len(diverse)} diverse compounds found, taking all")
        result = diverse

    consensus_noise = np.random.normal(0, 0.5, len(result))
    result = result.with_columns([
        (pl.col('Activity') + pl.Series('noise', consensus_noise)).alias('Consensus_Score')
    ])

    result = result.select(['ID', 'SMILES', 'Activity', 'Consensus_Score'])

    print(f"  Activity range: [{result['Activity'].min():.2f}, {result['Activity'].max():.2f}]")
    correlation = np.corrcoef(
        result['Activity'].to_numpy(),
        result['Consensus_Score'].to_numpy()
    )[0, 1]
    print(f"  Activity-Consensus correlation: {correlation:.3f}")

    return result


def generate_diverse_molecules(df: pl.DataFrame) -> pl.DataFrame:
    """Generate diverse_molecules.csv (100 compounds).

    Maximizes structural diversity for clustering and diversity
    acquisition tests.

    Args:
        df: Source DataFrame (validated ampc_30k data)

    Returns:
        DataFrame with 100 maximally diverse compounds
    """
    print("\nGenerating diverse_molecules.csv (100 compounds)...")

    sampled = stratified_sample(df, n=500)
    result = maxmin_sample(sampled, n=100)

    result = result.select(['ID', 'SMILES', 'Activity'])

    smiles_list = result['SMILES'].to_list()
    fps = compute_morgan_fingerprints(smiles_list)

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(tanimoto_similarity(fps[i], fps[j]))

    print(f"  Pairwise similarity - mean: {np.mean(similarities):.3f}, "
          f"max: {np.max(similarities):.3f}")
    print(f"  Activity range: [{result['Activity'].min():.2f}, {result['Activity'].max():.2f}]")

    return result


def generate_edge_case_molecules() -> pl.DataFrame:
    """Generate edge_case_molecules.csv (20 compounds).

    Synthetic edge cases for error handling and robustness testing.
    Intentionally includes invalid data.

    Returns:
        DataFrame with 20 edge case compounds
    """
    print("\nGenerating edge_case_molecules.csv (20 compounds)...")

    edge_cases = []

    edge_cases.extend([
        {'ID': 'EDGE_INVALID_01', 'SMILES': 'C1CCC', 'Activity': 0.0, 'Edge_Case_Type': 'invalid_smiles_unclosed_ring'},
        {'ID': 'EDGE_INVALID_02', 'SMILES': 'C(C)(C)(C)(C)C', 'Activity': 0.0, 'Edge_Case_Type': 'invalid_smiles_valence'},
        {'ID': 'EDGE_INVALID_03', 'SMILES': 'Xyz123', 'Activity': 0.0, 'Edge_Case_Type': 'invalid_smiles_characters'},
        {'ID': 'EDGE_INVALID_04', 'SMILES': 'c1ccccc', 'Activity': 0.0, 'Edge_Case_Type': 'invalid_smiles_unclosed_aromatic'},
        {'ID': 'EDGE_INVALID_05', 'SMILES': '', 'Activity': 0.0, 'Edge_Case_Type': 'invalid_smiles_empty'},
    ])

    edge_cases.extend([
        {'ID': 'EDGE_UNUSUAL_01', 'SMILES': 'C' * 50, 'Activity': 5.0, 'Edge_Case_Type': 'unusual_long_chain'},
        {'ID': 'EDGE_UNUSUAL_02', 'SMILES': 'C', 'Activity': 1.0, 'Edge_Case_Type': 'unusual_very_small'},
        {'ID': 'EDGE_UNUSUAL_03', 'SMILES': 'C1CC2CCC3CCC4CCC5CCCC5C4C3C2C1', 'Activity': 10.0, 'Edge_Case_Type': 'unusual_many_rings'},
        {'ID': 'EDGE_UNUSUAL_04', 'SMILES': 'C[C@H]1CC[C@@H](C)[C@H](C)[C@@H]1C', 'Activity': 3.0, 'Edge_Case_Type': 'unusual_stereochemistry'},
        {'ID': 'EDGE_UNUSUAL_05', 'SMILES': 'c1ccc2c(c1)ccc1ccccc12', 'Activity': 4.0, 'Edge_Case_Type': 'unusual_fused_aromatics'},
    ])

    edge_cases.extend([
        {'ID': 'EDGE_FEATURE_01', 'SMILES': 'C1CCCCCCCCCCCCCCC1', 'Activity': 2.0, 'Edge_Case_Type': 'feature_macrocycle'},
        {'ID': 'EDGE_FEATURE_02', 'SMILES': 'C1CC2CCC1C2', 'Activity': 6.0, 'Edge_Case_Type': 'feature_bridged_bicyclic'},
        {'ID': 'EDGE_FEATURE_03', 'SMILES': 'C1CCC2(C1)CCC1(CC2)CC1', 'Activity': 7.0, 'Edge_Case_Type': 'feature_spiro'},
        {'ID': 'EDGE_FEATURE_04', 'SMILES': 'CC(=O)Oc1ccccc1C(=O)O', 'Activity': 8.0, 'Edge_Case_Type': 'feature_aspirin'},
        {'ID': 'EDGE_FEATURE_05', 'SMILES': 'C=CC=CC=CC=C', 'Activity': 9.0, 'Edge_Case_Type': 'feature_conjugated'},
    ])

    edge_cases.extend([
        {'ID': 'EDGE_BOUNDARY_01', 'SMILES': 'CCO', 'Activity': float('nan'), 'Edge_Case_Type': 'boundary_nan_activity'},
        {'ID': 'EDGE_BOUNDARY_02', 'SMILES': 'CCC', 'Activity': -999.0, 'Edge_Case_Type': 'boundary_extreme_outlier'},
        {'ID': 'EDGE_BOUNDARY_03', 'SMILES': 'CCCCCC', 'Activity': 5.5, 'Edge_Case_Type': 'boundary_duplicate_smiles'},
        {'ID': 'EDGE_BOUNDARY_04', 'SMILES': 'CCCCCC', 'Activity': 5.5, 'Edge_Case_Type': 'boundary_duplicate_smiles'},
        {'ID': '', 'SMILES': 'CCCCCCC', 'Activity': 6.0, 'Edge_Case_Type': 'boundary_missing_id'},
    ])

    df = pl.DataFrame(edge_cases)
    print(f"  Generated {len(df)} edge case compounds")
    print(f"  Invalid SMILES: 5, Unusual: 5, Features: 5, Boundary: 5")

    return df


def main():
    """Main execution function."""
    print("=" * 70)
    print("LearnM8 Test Fixture Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    df = load_and_validate_source_data()

    fixtures = {
        'small_molecules.csv': generate_small_molecules(df),
        'medium_molecules.csv': generate_medium_molecules(df),
        'diverse_molecules.csv': generate_diverse_molecules(df),
        'edge_case_molecules.csv': generate_edge_case_molecules(),
    }

    print("\n" + "=" * 70)
    print("Writing fixtures to CSV files...")
    print("=" * 70)

    for filename, fixture_df in fixtures.items():
        output_path = OUTPUT_DIR / filename
        fixture_df.write_csv(output_path)
        print(f"✓ {filename}: {len(fixture_df)} compounds -> {output_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Source dataset: {len(df)} compounds")
    print(f"Generated fixtures: {len(fixtures)} files")
    print(f"Total test compounds: {sum(len(f) for f in fixtures.values())}")
    print("\n✓ All test fixtures generated successfully!")
    print("\nNext steps:")
    print("  1. Run pytest to verify fixtures work correctly")
    print("  2. Check that learner tests now pass")
    print("  3. Commit generated CSV files to repository")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
