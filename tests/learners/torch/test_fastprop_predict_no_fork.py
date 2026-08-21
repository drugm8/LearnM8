"""Regression tests for BUGFIX4: fastprop prediction must not fork worker processes.

`FastpropLearner.predict` builds its `fastpropDataLoader` with `num_workers=0`,
but PyTorch Lightning unconditionally re-instantiates the prediction dataloader
in order to wrap its batch sampler in an `_IndexBatchSamplerWrapper`
(``data_connector._prepare_dataloader``: ``... or mode == RunningStage.PREDICTING``).

The re-instantiation drops any kwarg whose value equals the *base*
``torch.utils.data.DataLoader`` default and then reconstructs the *subclass*.
Because ``fastpropDataLoader.__init__`` takes ``**torch_kwargs``, Lightning
overwrites the subclass defaults with torch's own when deciding what was
"explicitly set", so ``num_workers=0`` is indistinguishable from unset and the
subclass's declared default of ``num_workers=1, persistent_workers=True`` is
reapplied — forking a DataLoader worker from the (heavily multi-threaded)
parent, the fork-after-threads deadlock observed on the cluster.

No value of ``num_workers`` can survive that filter, so the fix is to build a
plain ``DataLoader`` in ``predict``; the subclass only supplies different
defaults and ``predict`` already uses a plain ``TensorDataset``.
"""

import os

import numpy as np
import pytest
import torch
import torch.utils.data.dataloader as torch_dataloader
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from learnm8.learners.torch.fastprop_learner import FastpropLearner


def _rebuild_as_lightning_would(loader):
    """Put a dataloader through Lightning's predict-mode re-instantiation."""
    from lightning.pytorch.trainer.states import RunningStage
    from lightning.pytorch.utilities.data import _update_dataloader

    return _update_dataloader(
        loader, SequentialSampler(loader.dataset), mode=RunningStage.PREDICTING
    )


@pytest.mark.unit
class TestLightningDataLoaderReinstantiation:
    """Pin the exact upstream mechanism, with no model training involved."""

    def test_predict_dataloader_survives_rebuild(self, monkeypatch):
        """The loader `predict` actually builds must keep num_workers=0."""
        from pytorch_lightning import Trainer

        learner = FastpropLearner(fnn_layers=1, hidden_size=4, max_epochs=1)
        learner.is_trained = True
        learner.model = object()

        captured = {}

        def fake_predict(self, model, dataloaders):
            captured['loader'] = dataloaders
            return [torch.zeros(4, 1)]

        monkeypatch.setattr(Trainer, 'predict', fake_predict)
        learner.predict(np.zeros((4, 3), dtype=np.float32))

        rebuilt = _rebuild_as_lightning_would(captured['loader'])
        assert rebuilt.num_workers == 0, (
            f'predict built a {type(captured["loader"]).__name__} that Lightning '
            f'rebuilt with num_workers={rebuilt.num_workers}; a worker will be forked.'
        )
        assert rebuilt.persistent_workers is False

    def test_fastprop_subclass_cannot_keep_num_workers_zero(self):
        """Why the subclass is banned from predict. Fails if upstream ever fixes this."""
        from fastprop.data import fastpropDataLoader

        dataset = TensorDataset(torch.zeros(8, 4))
        loader = fastpropDataLoader(
            dataset, batch_size=4, num_workers=0, persistent_workers=False
        )
        assert loader.num_workers == 0

        rebuilt = _rebuild_as_lightning_would(loader)
        assert rebuilt.num_workers == 1
        assert rebuilt.persistent_workers is True

    def test_plain_dataloader_is_unaffected(self):
        """Control: chemprop passes a plain DataLoader, whose default is 0 — safe."""
        dataset = TensorDataset(torch.zeros(8, 4))
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        rebuilt = _rebuild_as_lightning_would(loader)
        assert rebuilt.num_workers == 0
        assert rebuilt.persistent_workers is False


@pytest.mark.slow
@pytest.mark.integration
class TestFastpropPredictDoesNotFork:
    """End-to-end: a real train+predict must create no subprocess."""

    @staticmethod
    def _tiny_learner():
        return FastpropLearner(
            fnn_layers=1,
            hidden_size=8,
            max_epochs=1,
            val_fraction=0.0,
            batch_size=16,
        )

    def test_predict_creates_no_multiprocessing_iterator(self, monkeypatch):
        """No `_MultiProcessingDataLoaderIter` may be built during predict."""
        rng = np.random.default_rng(0)
        features = rng.random((32, 8)).astype(np.float32)
        targets = features[:, 0].astype(np.float32)

        learner = self._tiny_learner()
        learner.train(features, targets)

        observed: list[tuple[int, bool]] = []
        real_init = torch_dataloader._MultiProcessingDataLoaderIter.__init__

        def spy(self, loader):
            observed.append((loader.num_workers, loader.persistent_workers))
            return real_init(self, loader)

        monkeypatch.setattr(
            torch_dataloader._MultiProcessingDataLoaderIter, '__init__', spy
        )

        learner.predict(features)

        assert observed == [], (
            'FastpropLearner.predict built a multiprocessing DataLoader iterator '
            f'(num_workers, persistent_workers)={observed}; this forks a worker '
            'from a multi-threaded parent and can deadlock.'
        )

    def test_predict_does_not_call_os_fork(self, monkeypatch):
        """Direct observation of the fork itself."""
        rng = np.random.default_rng(1)
        features = rng.random((32, 8)).astype(np.float32)
        targets = features[:, 0].astype(np.float32)

        learner = self._tiny_learner()
        learner.train(features, targets)

        forked: list[int] = []
        real_fork = os.fork

        def spy_fork():
            pid = real_fork()
            if pid > 0:
                forked.append(pid)
            return pid

        monkeypatch.setattr(os, 'fork', spy_fork)

        learner.predict(features)

        assert forked == [], (
            f'FastpropLearner.predict forked {len(forked)} child process(es) '
            'from a multi-threaded parent (BUGFIX4 deadlock).'
        )
