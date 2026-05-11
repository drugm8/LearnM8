"""Tests for acquisition base helpers and validation utilities."""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition.base import AcquisitionFunction, validate_uncertainty_inputs
from learnm8.exceptions import AcquisitionError


class DummyAcquisition(AcquisitionFunction):
    def __init__(self, requires_uncertainty: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._requires_uncertainty = requires_uncertainty

    def select(self, compounds: pl.DataFrame, n_select: int) -> pl.DataFrame:
        self.validate_input(compounds, n_select)
        return compounds.head(min(n_select, len(compounds)))

    def requires_uncertainty(self) -> bool:
        return self._requires_uncertainty


def _pool_with_predictions(compounds: pl.DataFrame, predictions: list[float] | np.ndarray) -> pl.DataFrame:
    return compounds.with_columns(pl.Series('prediction', predictions))


def _pool_with_predictions_and_uncertainty(
    compounds: pl.DataFrame,
    predictions: list[float] | np.ndarray,
    uncertainty: list[float] | np.ndarray,
) -> pl.DataFrame:
    return compounds.with_columns([
        pl.Series('prediction', predictions),
        pl.Series('uncertainty', uncertainty),
    ])


@pytest.mark.unit
class TestAcquisitionBaseValidation:
    def test_base_rejects_invalid_score_direction(self):
        with pytest.raises(ValueError, match="score_direction must be 'higher' or 'lower'"):
            DummyAcquisition(score_direction='sideways')

    def test_base_get_name_and_requires_uncertainty_defaults(self):
        acquisition = DummyAcquisition()

        assert acquisition.get_name() == 'DummyAcquisition'
        assert acquisition.requires_uncertainty() is False

    def test_validate_input_rejects_duplicate_ids(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(3).clone(), [0.1, 0.2, 0.3])
        duplicate_id = compounds.get_column('ID')[0]
        duplicated = compounds.with_columns(
            pl.Series('ID', [duplicate_id, duplicate_id, compounds.get_column('ID')[2]])
        )

        with pytest.raises(ValueError, match='duplicate compound IDs'):
            DummyAcquisition().validate_input(duplicated, n_select=2)

    def test_validate_input_rejects_nan_uncertainty_values(self, small_real_compounds):
        # Feature 019 FR-005: cycle.py is the canonical NaN-fail-fast path.
        # The validate_input branch is now defence-in-depth (LearnerError with
        # [secondary-guard] prefix) so it survives `python -O`.
        from learnm8.exceptions import LearnerError
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).clone(),
            [0.1, 0.2, 0.3],
            [0.1, np.nan, 0.3],
        )

        with pytest.raises(LearnerError, match=r"\[secondary-guard\] Uncertainties contain"):
            DummyAcquisition().validate_input(compounds, n_select=2)

    def test_safe_select_top_k_rejects_score_length_mismatch(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(3).clone(), [0.1, 0.2, 0.3])

        with pytest.raises(ValueError, match="scores length"):
            DummyAcquisition()._safe_select_top_k(compounds, np.array([0.1, 0.2]), n_select=2)

    def test_safe_select_top_k_replaces_invalid_scores_for_descending_selection(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(4).clone(), [0.1, 0.2, 0.3, 0.4])
        scores = np.array([0.9, np.nan, np.inf, 0.1])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=2, ascending=False)

        assert len(selected) == 2
        assert np.all(np.isfinite(selected.get_column('acquisition_score').to_numpy()))

    def test_safe_select_top_k_replaces_invalid_scores_for_ascending_selection(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(4).clone(), [0.1, 0.2, 0.3, 0.4])
        scores = np.array([0.9, np.nan, -np.inf, 0.1])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=2, ascending=True)

        assert len(selected) == 2
        assert np.all(np.isfinite(selected.get_column('acquisition_score').to_numpy()))

    def test_safe_select_top_k_ascending_selects_lowest_scores(self, small_real_compounds):
        compounds = _pool_with_predictions(
            small_real_compounds.head(5).clone(),
            [0.5, 0.9, 0.1, 0.8, 0.3],
        )
        scores = np.array([0.5, 0.9, 0.1, 0.8, 0.3])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=3, ascending=True)

        assert len(selected) == 3
        acq_scores = selected.get_column('acquisition_score').to_numpy()
        assert set(acq_scores) == {0.1, 0.3, 0.5}, "must return 3 lowest score values"
        assert list(acq_scores) == sorted(acq_scores.tolist()), "must be ordered ascending"

    def test_safe_select_top_k_all_invalid_scores_replaced_and_returned(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(3).clone(), [0.1, 0.2, 0.3])
        scores = np.array([float('nan'), float('nan'), float('nan')])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=2, ascending=False)

        assert len(selected) == 2
        acq_scores = selected.get_column('acquisition_score').to_numpy()
        assert all(np.isneginf(s) for s in acq_scores), "all-NaN scores should become -inf for descending"

    def test_safe_select_top_k_positional_alignment(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(5).clone(), [0.1] * 5)
        scores = np.array([0.5, 0.9, 0.1, 0.8, 0.3])
        all_ids = compounds.get_column('ID').to_numpy()

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=3, ascending=False)

        result_ids = set(selected.get_column('ID').to_list())
        expected_ids = {all_ids[1], all_ids[3], all_ids[0]}
        assert result_ids == expected_ids, "positional indexing must select correct compounds"
        assert set(compounds[np.array([1, 3, 0])].get_column('ID').to_list()) == expected_ids

    def test_safe_select_top_k_tiebreaking_is_deterministic(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(5).clone(), [0.1] * 5)
        scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])

        result1 = DummyAcquisition()._safe_select_top_k(compounds, scores.copy(), n_select=3, ascending=False)
        result2 = DummyAcquisition()._safe_select_top_k(compounds, scores.copy(), n_select=3, ascending=False)

        assert result1.get_column('ID').to_list() == result2.get_column('ID').to_list()
        assert result1.get_column('acquisition_score').to_list() == result2.get_column('acquisition_score').to_list()

    def test_safe_select_top_k_n_select_equals_pool_size(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(4).clone(), [0.1, 0.2, 0.3, 0.4])
        scores = np.array([0.4, 0.1, 0.3, 0.2])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=4, ascending=False)

        assert len(selected) == 4
        assert set(selected.get_column('ID').to_list()) == set(compounds.get_column('ID').to_list())

    def test_safe_select_top_k_n_select_exceeds_pool_returns_all(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(3).clone(), [0.1, 0.2, 0.3])
        scores = np.array([0.3, 0.1, 0.2])

        selected = DummyAcquisition()._safe_select_top_k(compounds, scores, n_select=10, ascending=False)

        assert len(selected) == 3

    def test_validate_input_duplicate_id_count_is_accurate(self, small_real_compounds):
        base = small_real_compounds.head(3).clone()
        predictions = [0.1, 0.2, 0.3]
        pool = _pool_with_predictions(base, predictions)
        dup_id = pool.get_column('ID')[0]
        duplicated = pool.with_columns(
            pl.Series('ID', [dup_id, pool.get_column('ID')[1], dup_id])
        )

        with pytest.raises(ValueError, match='1 duplicate'):
            DummyAcquisition().validate_input(duplicated, n_select=2)


@pytest.mark.unit
class TestNSelectExceedsPoolByStrategy:
    @pytest.mark.parametrize('strategy,extra_kwargs', [
        ('greedy', {}),
        ('ucb', {'beta': 2.0}),
        ('thompson', {'random_state': 42}),
    ])
    def test_n_select_exceeds_pool_handled_gracefully(self, strategy, extra_kwargs, small_real_compounds):
        predictions = list(np.linspace(0.1, 0.5, 5))
        uncertainties = [0.1] * 5
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(5).clone(), predictions, uncertainties
        )
        acq = get_acquisition_function(strategy)(score_direction='higher', **extra_kwargs)
        result = acq.select(compounds, n_select=20)
        assert len(result) == 5, f"{strategy}: must return all 5 compounds when n_select > pool"


@pytest.mark.unit
class TestValidateUncertaintyInputs:
    def test_validate_uncertainty_inputs_rejects_missing_prediction_column(self, small_real_compounds):
        compounds = small_real_compounds.head(3).select(['ID', 'SMILES']).with_columns(
            pl.Series('uncertainty', [0.1, 0.2, 0.3])
        )

        with pytest.raises(AcquisitionError, match="missing required 'prediction' column"):
            validate_uncertainty_inputs(compounds)

    def test_validate_uncertainty_inputs_rejects_missing_uncertainty_column(self, small_real_compounds):
        compounds = _pool_with_predictions(small_real_compounds.head(3).select(['ID', 'SMILES']).clone(), [0.1, 0.2, 0.3])

        with pytest.raises(AcquisitionError, match="missing required 'uncertainty' column"):
            validate_uncertainty_inputs(compounds)

    def test_validate_uncertainty_inputs_skips_nan_checks(self, small_real_compounds):
        # Feature 019: NaN guards are upstream (cycle.py FR-005) and at the
        # secondary-guard `validate_input` layer; validate_uncertainty_inputs
        # no longer re-checks for NaN to avoid duplicate hot-path scans.
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).select(['ID', 'SMILES']).clone(),
            [0.1, np.nan, 0.3],
            [0.1, 0.2, 0.3],
        )
        # Returns the arrays as-is — no inline raise.
        preds, uncs = validate_uncertainty_inputs(compounds)
        assert np.isnan(preds).any()
        assert not np.isnan(uncs).any()

    def test_validate_uncertainty_inputs_rejects_negative_uncertainties(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).select(['ID', 'SMILES']).clone(),
            [0.1, 0.2, 0.3],
            [0.1, -0.2, 0.3],
        )

        with pytest.raises(AcquisitionError, match='negative values'):
            validate_uncertainty_inputs(compounds)
