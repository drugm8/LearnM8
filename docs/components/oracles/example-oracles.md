# Example Oracles

`examples/oracles/` in the repository contains four oracles you can use directly or copy as a starting point. They are `Oracle` subclasses that score each compound's `SMILES` against a reference molecule (or receptor) and are covered by `tests/oracles/test_example_oracles.py`.

They are not part of the installed package. Run Python from a repository checkout and import them as `examples.oracles`. Each one writes its score into every property column you request, so `target_col` can be any name. Compounds that cannot be scored get a null value and are marked `failed` (see [Python oracle](python-oracle.md#compounds-that-cannot-be-measured)).

| Oracle | Scores | Extra dependency |
|--------|--------|------------------|
| `SimilarityOracle` | 2D fingerprint similarity to a reference | none |
| `Pharmacophore2DOracle` | Gobbi 2D pharmacophore similarity to a reference | none |
| `CDPKitPharmacophoreOracle` | 3D pharmacophore alignment to a reference | CDPKit |
| `VinaOracle` | AutoDock Vina docking affinity | Vina, Meeko |

`environment_oracles.yml` is a conda environment that includes all of them. Because the oracles are instances, pass them to the Python API as `oracle=`; they cannot be selected from the CLI `module.py:function` syntax.

## SimilarityOracle

Computes the Tanimoto or Dice similarity between each compound and a reference molecule, using Morgan, MACCS or RDKit topological fingerprints. Higher is more similar. Arguments: `reference_smiles`, `fingerprint_type` (`'morgan'`, `'maccs'`, `'topological'`), `metric` (`'tanimoto'`, `'dice'`), `radius`, `n_bits`.

```python
from learnm8 import run_active_learning
from examples.oracles import SimilarityOracle

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=SimilarityOracle(reference_smiles='c1ccccc1O', fingerprint_type='morgan'),
    target_col='similarity',
    learner='rf',
    featurizer='morgan',
    n_cycles=10,
)
```

??? example "examples/oracles/similarity.py"
    ```python
    --8<-- "examples/oracles/similarity.py"
    ```

## Pharmacophore2DOracle

Computes the Tanimoto or Dice similarity between Gobbi 2D pharmacophore fingerprints (RDKit) of each compound and a reference molecule. It uses topology only, so no conformers are generated. Arguments: `reference_smiles`, `metric` (`'tanimoto'`, `'dice'`).

```python
from examples.oracles import Pharmacophore2DOracle

oracle = Pharmacophore2DOracle(reference_smiles='c1ccc(O)cc1C(=O)O')
```

??? example "examples/oracles/pharmacophore_2d.py"
    ```python
    --8<-- "examples/oracles/pharmacophore_2d.py"
    ```

## CDPKitPharmacophoreOracle

Embeds each compound in 3D with RDKit (ETKDG, MMFF), generates a CDPKit pharmacophore, and aligns it to the reference pharmacophore. `score_type` selects the value: `'alignment_score'` is the feature match count, `'feature_overlap'` is that count divided by the larger feature count, and `'rmsd'` returns `1 / (1 + score)`; for all three, higher is a better match. Compounds with fewer than `min_feature_match` pharmacophore features get null. Requires CDPKit (`pip install cdpkit`); the class is only importable from `examples.oracles` when CDPKit is installed.

```python
from examples.oracles import CDPKitPharmacophoreOracle

oracle = CDPKitPharmacophoreOracle(
    reference_smiles='c1ccc(O)cc1C(=O)O',
    score_type='alignment_score',
    min_feature_match=3,
    max_conformers=10,
)
```

??? example "examples/oracles/cdpkit_pharmacophore.py"
    ```python
    --8<-- "examples/oracles/cdpkit_pharmacophore.py"
    ```

## VinaOracle

Docks each compound into a prepared receptor with AutoDock Vina and returns the best predicted binding affinity in kcal/mol; more negative means stronger binding, so use `score_direction='lower'`. Ligands are embedded in 3D with RDKit and converted to PDBQT with Meeko. Requires Vina and Meeko (`conda install -c conda-forge vina meeko`) and a receptor in PDBQT format. Arguments: `receptor_path`, `center`, `box_size` (default 20 x 20 x 20), `exhaustiveness` (default 8), `n_poses`, `energy_range`.

```python
from learnm8 import run_active_learning
from examples.oracles import VinaOracle

oracle = VinaOracle(
    receptor_path='receptor.pdbqt',
    center=(10.0, 15.0, 20.0),
    box_size=(20.0, 20.0, 20.0),
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle=oracle,
    target_col='binding_affinity',
    learner='rf',
    featurizer='morgan',
    score_direction='lower',
    n_cycles=10,
)
```

??? example "examples/oracles/vina.py"
    ```python
    --8<-- "examples/oracles/vina.py"
    ```
