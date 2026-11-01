# Contributing to LearnM8

## Quick Start

```bash
conda env create -f environment.yml
conda activate learnm8
pip install -e ".[test]"
```

Run the fast test suite to verify your environment:

```bash
pytest -m "not slow" tests/
```

## Development Workflow

1. Create a branch from `main`
2. Make changes following the standards below
3. Add tests for new code and update the docs
4. Run tests and linting
5. Open a pull request with a clear description of the change

### Before Committing

```bash
ruff check .          # lint
ruff format .         # format
mypy learnm8/         # type check
pytest -m "not slow" tests/   # fast tests
```

For changes to `learners/torch/`, `learners/ensemble/`, 3D featurizers in `features/`, `api.py`, or `core/cycle.py`, include the slow tests:

```bash
pytest tests/
```

Coverage must stay at or above 90%:

```bash
pytest tests/ --cov=learnm8 --cov-fail-under=90
pytest tests/ --cov=learnm8 --cov-report=html   # HTML report in htmlcov/index.html
```

## Documentation

The docs in `docs/` are built with [zensical](https://zensical.org) and configured in `zensical.toml` (navigation, theme, extensions). CI runs the strict build.

```bash
pip install zensical
zensical serve -f zensical.toml                 # live preview
zensical build -f zensical.toml --clean --strict   # as in CI
```

Update the relevant page in `docs/` and the docstrings when you change user-visible behaviour. Docs must match the code: check every parameter and default against the source.

## Adding a New Learner

1. Create a file in the appropriate subdirectory:
   - `learnm8/learners/sklearn/` — scikit-learn models
   - `learnm8/learners/torch/` — PyTorch models
   - `learnm8/learners/ensemble/` — ensemble wrappers

2. Inherit from `Learner` in `learnm8.core.interfaces` and implement:

```python
from learnm8.core.interfaces import Learner
import numpy as np

class MyLearner(Learner):
    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        smiles: list[str] | None = None
    ) -> None: ...

    def predict(
        self,
        features: np.ndarray,
        smiles: list[str] | None = None,
        *,
        compute_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]: ...

    def supports_uncertainty(self) -> bool:
        return False   # or True if your model provides uncertainty

    def get_name(self) -> str:
        return 'my_learner'
```

3. Honour the `compute_uncertainty` contract:
   - If your learner is skip-eligible (uncertainty is a separate, expensive step), return `None` as the second element when `compute_uncertainty=False`.
   - If uncertainty is inseparable from inference (e.g. MC Dropout), always run the full inference path and return `None` only as the std output when `compute_uncertainty=False`.

4. Register in `LEARNER_REGISTRY` in `learnm8/api.py`:

```python
LEARNER_REGISTRY['my_learner'] = MyLearner
```

5. Add tests in `tests/learners/`. At minimum:
   - `test_train_predict` — basic training and prediction
   - `test_compute_uncertainty_false` — verify `(predictions, None)` when `compute_uncertainty=False`
   - `test_signature` — `inspect.signature` includes `compute_uncertainty`

## Adding a New Featurizer

Featurizers are configured through the `_FEATURIZER_CONFIG` factory in
`learnm8/features/__init__.py` rather than as standalone files.

1. Add an entry to `_FEATURIZER_CONFIG` in `learnm8/features/__init__.py`,
   keyed by featurizer name, specifying the scikit-fingerprints `cls`, its
   `defaults`, `storage_dtype`, `feature_type`, and `requires_conformers`.
2. The featurizer name is exposed automatically via `FEATURIZER_REGISTRY`
   (derived from the config keys) — no separate import or `__all__` edit.
3. Add a test in `tests/features/` covering the output shape and dtype.

Binary fingerprints must use `'packed_uint8'` storage to avoid the
packed-bits corruption bug. Continuous descriptors use `'float32'`. See
`learnm8/features/base.py` and the `SkfpFeaturizer` contract for details.

## Adding a New Acquisition Strategy

1. Inherit from `AcquisitionFunction` in `learnm8.acquisition.base`.
2. Implement `select(compounds, n_select)` and `requires_uncertainty()`.
3. Register in `ACQUISITION_REGISTRY` in `learnm8/acquisition/__init__.py`.
4. Add tests in `tests/acquisition/`.
5. To stay on the memory-efficient streaming path, also implement the pointwise tier (`supports_streaming`, `score_chunk`, and if needed `shortlist_size`/`finalize`). Without it the strategy runs through `select()` only.

## Code Standards

- **No comments** unless the WHY is non-obvious (hidden constraint, workaround for a specific bug).
- **No print statements** — use `logging.getLogger(__name__)`.
- **Polars only** for internal DataFrames. All public API accepts pandas but auto-converts.
- **Pure functions** — never mutate input DataFrames (use `with_columns()` which returns a new DataFrame).
- Follow `ruff` rules as configured in `pyproject.toml`.
- All public functions and classes need type annotations.

## Architecture Notes

- `learnm8/api.py` is the only public entry point. Do not add public symbols elsewhere without discussion.
- `learnm8/exceptions.py` is the only place to define exceptions. Raise `LearnM8Error` subclasses, never built-in exceptions from library internals.
- HDF5 cache schema is versioned. If you change cache layout, bump `schema_version` and add a migration path. See `learnm8/features/cache.py`.
- The master DataFrame stays at exactly 7 columns. Per-cycle predictions go to separate parquet files.

## Commit Style

```
type(scope): short description

Longer explanation if needed.
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`. Keep subject lines under 72 characters.
