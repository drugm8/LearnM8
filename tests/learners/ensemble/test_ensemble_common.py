"""Common parametrized tests for all non-SMILES ensemble types.

Covers 14 test patterns shared across 6 ensemble implementations:
RFEnsemble, LREnsemble, DTEnsemble, XGBEnsemble, MixedEnsemble, EnsembleLearner.
"""

import numpy as np
import polars as pl
import pytest

from learnm8.learners.ensemble.dt_ensemble import DTEnsemble
from learnm8.learners.ensemble.ensemble import EnsembleLearner
from learnm8.learners.ensemble.lr_ensemble import LREnsemble
from learnm8.learners.ensemble.mixed_ensemble import MixedEnsemble
from learnm8.learners.ensemble.rf_ensemble import RFEnsemble
from learnm8.learners.ensemble.xgb_ensemble import XGBEnsemble
from learnm8.learners.sklearn.gaussian_process import GaussianProcessLearner
from learnm8.learners.sklearn.random_forest import RandomForestLearner

ENSEMBLE_CONFIGS = [
    pytest.param(
        ('rf_ensemble', lambda: RFEnsemble(n_estimators=10)), id='rf_ensemble'
    ),
    pytest.param(('lr_ensemble', lambda: LREnsemble()), id='lr_ensemble'),
    pytest.param(('dt_ensemble', lambda: DTEnsemble()), id='dt_ensemble'),
    pytest.param(
        ('xgb_ensemble', lambda: XGBEnsemble()),
        id='xgb_ensemble',
        marks=pytest.mark.slow,
    ),
    pytest.param(
        ('mixed_ensemble', lambda: MixedEnsemble(random_state=42)),
        id='mixed_ensemble',
        marks=pytest.mark.slow,
    ),
    pytest.param(
        (
            'ensemble',
            lambda: EnsembleLearner(
                [
                    RandomForestLearner(n_estimators=5, random_state=42),
                    GaussianProcessLearner(random_state=42),
                ]
            ),
        ),
        id='base_ensemble',
    ),
]


def _make_ensemble_with_kwargs(name, **kwargs):
    """Create an ensemble instance with custom kwargs (aggregation_method, uncertainty_method)."""
    if name == 'rf_ensemble':
        return RFEnsemble(n_estimators=10, **kwargs)
    elif name == 'lr_ensemble':
        return LREnsemble(**kwargs)
    elif name == 'dt_ensemble':
        return DTEnsemble(**kwargs)
    elif name == 'xgb_ensemble':
        return XGBEnsemble(**kwargs)
    elif name == 'mixed_ensemble':
        return MixedEnsemble(random_state=42, **kwargs)
    elif name == 'ensemble':
        return EnsembleLearner(
            [
                RandomForestLearner(n_estimators=5, random_state=42),
                GaussianProcessLearner(random_state=42),
            ],
            **kwargs,
        )
    else:
        raise ValueError(f'Unknown ensemble type: {name}')


def _prepare_targets(compounds):
    """Ensure compounds have an Activity column and return targets array."""
    if 'Activity' not in compounds.columns:
        compounds = compounds.with_columns(
            pl.Series('Activity', np.random.beta(2, 5, len(compounds)))
        )
    return compounds, compounds['Activity'].to_numpy()


@pytest.mark.slow
@pytest.mark.integration
class TestEnsembleCommon:
    """Parametrized tests shared by all non-SMILES ensemble types."""

    @pytest.fixture(params=ENSEMBLE_CONFIGS)
    def ensemble_setup(self, request):
        """Provide (name, fresh_ensemble) for each ensemble type."""
        name, factory = request.param
        return name, factory()

    def test_predict_returns_finite_values_and_uncertainty_after_training(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        _, ensemble = ensemble_setup
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        assert ensemble.is_trained

        predictions, uncertainty = ensemble.predict(features)
        assert predictions.shape[0] == len(compounds)
        assert uncertainty is not None
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)

    def test_uncertainty_estimates_are_non_negative_and_nonconstant(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        _, ensemble = ensemble_setup
        _compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert uncertainty is not None
        assert len(uncertainty) == len(predictions)
        assert np.all(uncertainty >= 0)
        assert np.std(uncertainty) > 0

    def test_supports_uncertainty_returns_true_for_all_ensemble_types(self, ensemble_setup):
        _, ensemble = ensemble_setup
        assert ensemble.supports_uncertainty() is True

    def test_train_with_empty_arrays(self, ensemble_setup):
        _, ensemble = ensemble_setup
        empty_features = np.array([]).reshape(0, 2048)
        empty_targets = np.array([])

        with pytest.raises((ValueError, RuntimeError)):
            ensemble.train(empty_features, empty_targets)

    def test_predict_without_training(self, ensemble_setup, small_real_morgan_features):
        _, ensemble = ensemble_setup
        with pytest.raises(RuntimeError, match='must be trained'):
            ensemble.predict(small_real_morgan_features)

    def test_untrained_ensemble_statistics_report_multiple_learners_and_untrained_state(self, ensemble_setup):
        _, ensemble = ensemble_setup
        stats = ensemble.get_ensemble_statistics()
        assert stats['n_learners'] >= 2
        assert stats['is_trained'] is False

    def test_trained_ensemble_statistics_include_learner_names(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        _, ensemble = ensemble_setup
        _compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        stats = ensemble.get_ensemble_statistics()
        assert stats['is_trained'] is True
        assert 'learner_names' in stats
        assert len(stats['learner_names']) >= 2

    def test_individual_predictions_return_finite_arrays_for_each_learner(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        _, ensemble = ensemble_setup
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        individual_preds = ensemble.get_individual_predictions(features)

        assert len(individual_preds) >= 1
        for _learner_name, preds in individual_preds.items():
            if preds is not None:
                assert len(preds) == len(compounds)
                assert np.all(np.isfinite(preds))

    def test_mean_aggregation_returns_prediction_and_uncertainty_per_compound(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        name, _ = ensemble_setup
        ensemble = _make_ensemble_with_kwargs(name, aggregation_method='mean')
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_median_aggregation_returns_prediction_and_uncertainty_per_compound(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        name, _ = ensemble_setup
        ensemble = _make_ensemble_with_kwargs(name, aggregation_method='median')
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(np.isfinite(predictions))

    def test_std_uncertainty_method_returns_non_negative_uncertainty(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        name, _ = ensemble_setup
        ensemble = _make_ensemble_with_kwargs(name, uncertainty_method='std')
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(uncertainty >= 0)

    def test_mad_uncertainty_method_returns_non_negative_uncertainty(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        name, _ = ensemble_setup
        ensemble = _make_ensemble_with_kwargs(name, uncertainty_method='mad')
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(uncertainty >= 0)

    def test_quantile_uncertainty_method_returns_non_negative_uncertainty(
        self, ensemble_setup, small_real_compounds, small_real_morgan_features
    ):
        name, _ = ensemble_setup
        ensemble = _make_ensemble_with_kwargs(name, uncertainty_method='quantile')
        compounds, targets = _prepare_targets(small_real_compounds)
        features = small_real_morgan_features

        ensemble.train(features, targets)
        predictions, uncertainty = ensemble.predict(features)

        assert predictions.shape[0] == len(compounds)
        assert uncertainty.shape[0] == len(compounds)
        assert np.all(uncertainty >= 0)

    def test_small_dataset_trains_and_predicts_finite_values_for_all_ensemble_types(self, ensemble_setup, tmp_path):
        _, ensemble = ensemble_setup
        small_dataset = pl.DataFrame(
            {
                'ID': ['COMP_001', 'COMP_002', 'COMP_003', 'COMP_004', 'COMP_005'],
                'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1CCNCC1'],
                'Activity': [0.5, 0.3, 0.8, 0.1, 0.6],
            }
        )

        from learnm8.features.extraction import extract_features

        features = extract_features(
            small_dataset['SMILES'].to_list(), 'morgan', tmp_path
        )
        ensemble.train(features, small_dataset['Activity'].to_numpy())
        predictions, uncertainty = ensemble.predict(features)

        assert len(predictions) == 5
        assert len(uncertainty) == 5
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)
