"""Tests for acquisition base helpers and validation utilities."""

import numpy as np
import polars as pl
import pytest

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
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).clone(),
            [0.1, 0.2, 0.3],
            [0.1, np.nan, 0.3],
        )

        with pytest.raises(ValueError, match='Uncertainties contain'):
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

    def test_validate_uncertainty_inputs_rejects_nan_predictions(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).select(['ID', 'SMILES']).clone(),
            [0.1, np.nan, 0.3],
            [0.1, 0.2, 0.3],
        )

        with pytest.raises(AcquisitionError, match='Predictions contain 1 NaN values'):
            validate_uncertainty_inputs(compounds)

    def test_validate_uncertainty_inputs_rejects_nan_uncertainties(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).select(['ID', 'SMILES']).clone(),
            [0.1, 0.2, 0.3],
            [0.1, np.nan, 0.3],
        )

        with pytest.raises(AcquisitionError, match='Uncertainties contain 1 NaN values'):
            validate_uncertainty_inputs(compounds)

    def test_validate_uncertainty_inputs_rejects_negative_uncertainties(self, small_real_compounds):
        compounds = _pool_with_predictions_and_uncertainty(
            small_real_compounds.head(3).select(['ID', 'SMILES']).clone(),
            [0.1, 0.2, 0.3],
            [0.1, -0.2, 0.3],
        )

        with pytest.raises(AcquisitionError, match='negative values'):
            validate_uncertainty_inputs(compounds)
