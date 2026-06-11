"""score_chunk parity + streaming-tier behavior across strategies.

Guards against drift between the legacy `select()` math and the streaming
`score_chunk()` math, and verifies streaming selection equals legacy selection
on tie-free data for the deterministic strategies.
"""

import numpy as np
import polars as pl
import pytest

from learnm8.acquisition import get_acquisition_function
from learnm8.acquisition.streaming import StreamingTopK

pytestmark = pytest.mark.unit

N = 200


def _pool(seed=0, with_unc=True):
    rng = np.random.default_rng(seed)
    preds = rng.permutation(np.linspace(-5, 5, N))  # distinct → tie-free
    df = {
        'ID': [f'c{i}' for i in range(N)],
        'SMILES': [f'C{"C" * (i % 4)}' for i in range(N)],
        'prediction': preds,
    }
    unc = None
    if with_unc:
        unc = rng.uniform(0.01, 2.0, size=N)
        df['uncertainty'] = unc
    return pl.DataFrame(df), preds, unc


def _streaming_select_ids(acq, df, preds, unc, k):
    scores = acq.score_chunk(preds, unc, global_offset=0, n_total=N)
    topk = StreamingTopK(k, descending=True)
    topk.update(scores, 0)
    idx, _ = topk.result()
    return [df['ID'][int(i)] for i in idx]


class TestDeterministicEquivalence:
    """Streaming selection == legacy selection on tie-free data."""

    @pytest.mark.parametrize('direction', ['higher', 'lower'])
    def test_greedy(self, direction):
        df, preds, _ = _pool(with_unc=False)
        acq = get_acquisition_function('greedy')(score_direction=direction)
        legacy = acq.select(df, 30)['ID'].to_list()
        streaming = _streaming_select_ids(acq, df, preds, None, 30)
        assert streaming == legacy

    @pytest.mark.parametrize('direction', ['higher', 'lower'])
    def test_ucb(self, direction):
        df, preds, unc = _pool()
        acq = get_acquisition_function('ucb')(score_direction=direction, beta=2.0)
        legacy = acq.select(df, 30)['ID'].to_list()
        streaming = _streaming_select_ids(acq, df, preds, unc, 30)
        assert streaming == legacy

    @pytest.mark.parametrize('strategy', ['ei', 'pi'])
    @pytest.mark.parametrize('direction', ['higher', 'lower'])
    def test_ei_pi(self, strategy, direction):
        df, preds, unc = _pool()
        cb = float(preds.max()) if direction == 'higher' else float(preds.min())
        acq = get_acquisition_function(strategy)(
            score_direction=direction, current_best=cb
        )
        legacy = acq.select(df, 30)['ID'].to_list()
        streaming = _streaming_select_ids(acq, df, preds, unc, 30)
        assert streaming == legacy

    def test_entropy(self):
        df, preds, unc = _pool()
        acq = get_acquisition_function('entropy')()
        legacy = acq.select(df, 30)['ID'].to_list()
        streaming = _streaming_select_ids(acq, df, preds, unc, 30)
        assert streaming == legacy


class TestStochasticScoreChunk:
    def test_random_uniform_in_unit_interval(self):
        _df, preds, _ = _pool(with_unc=False)
        acq = get_acquisition_function('random')(random_state=7)
        scores = acq.score_chunk(preds, None, global_offset=0, n_total=N)
        assert scores.min() >= 0.0 and scores.max() < 1.0

    def test_random_chunk_invariant(self):
        _df, preds, _ = _pool(with_unc=False)
        acq = get_acquisition_function('random')(random_state=7)
        full = acq.score_chunk(preds, None, global_offset=0, n_total=N)
        a = acq.score_chunk(preds[:50], None, global_offset=0, n_total=N)
        b = acq.score_chunk(preds[50:], None, global_offset=50, n_total=N)
        np.testing.assert_array_equal(np.concatenate([a, b]), full)

    @pytest.mark.parametrize('direction', ['higher', 'lower'])
    def test_thompson_chunk_invariant_and_signed(self, direction):
        _df, preds, unc = _pool()
        acq = get_acquisition_function('thompson')(
            random_state=11, score_direction=direction
        )
        full = acq.score_chunk(preds, unc, global_offset=0, n_total=N)
        a = acq.score_chunk(preds[:70], unc[:70], global_offset=0, n_total=N)
        b = acq.score_chunk(preds[70:], unc[70:], global_offset=70, n_total=N)
        np.testing.assert_array_equal(np.concatenate([a, b]), full)


class TestTopKShortlist:
    def test_shortlist_size_uses_k_fraction(self):
        acq = get_acquisition_function('topk')(k_fraction=0.1)
        assert acq.shortlist_size(5, 1000) == 100
        assert acq.shortlist_size(200, 1000) == 200  # never fewer than n_select
        assert acq.shortlist_size(5, 3) == 3  # clamped to pool

    def test_finalize_clamps_when_shortlist_small(self):
        acq = get_acquisition_function('topk')(k_fraction=0.1, random_state=1)
        shortlist = pl.DataFrame(
            {'ID': ['a', 'b'], 'prediction': [1.0, 2.0],
             'acquisition_score': [1.0, 2.0], 'global_index': [0, 1]}
        )
        out = acq.finalize(shortlist, 10)
        assert out.height == 2

    def test_finalize_samples_n_select(self):
        acq = get_acquisition_function('topk')(k_fraction=0.5, random_state=1)
        shortlist = pl.DataFrame(
            {'ID': [f'c{i}' for i in range(20)],
             'prediction': list(range(20)),
             'acquisition_score': list(range(20)),
             'global_index': list(range(20))}
        )
        out = acq.finalize(shortlist, 5)
        assert out.height == 5


class TestStreamingDefaults:
    def test_simulated_annealing_not_streamable(self):
        acq = get_acquisition_function('simulated_annealing')()
        assert acq.supports_streaming() is False

    def test_score_chunk_raises_without_uncertainty(self):
        from learnm8.exceptions import AcquisitionError

        _df, preds, _ = _pool(with_unc=False)
        for strat in ['ucb', 'ei', 'pi', 'entropy', 'thompson']:
            acq = get_acquisition_function(strat)(current_best=0.0)
            with pytest.raises(AcquisitionError):
                acq.score_chunk(preds, None, global_offset=0, n_total=N)
