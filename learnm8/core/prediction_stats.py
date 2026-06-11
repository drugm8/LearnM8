"""Compact summary statistics for a cycle's pool predictions.

The streaming cycle never holds the full ``(N,)`` prediction/uncertainty arrays
in the parent heap, so the per-cycle metric reductions (mean/std/min/max/median)
are computed once from the on-disk parquet and carried as this small value
object instead of the arrays.

``from_arrays`` reproduces the exact reductions the legacy path computes inline
in ``cycle._calculate_cycle_metrics`` (``np.nanmean/nanstd`` with ``dtype=
float64`` and ddof=0, ``np.nanmin/nanmax/nanmedian``) so the two paths are
value-identical. ``from_lazy`` computes the same quantities from a polars
``LazyFrame`` over the predictions parquet with the matching conventions
(``std(ddof=0)``, ``median`` = linear interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PredictionStats:
    """Scalar reductions over a cycle's pool predictions (+ optional uncertainty)."""

    count: int
    pred_mean: float | None
    pred_std: float | None
    pred_min: float | None
    pred_max: float | None
    pred_median: float | None
    has_uncertainty: bool = False
    unc_mean: float | None = None
    unc_std: float | None = None
    unc_min: float | None = None
    unc_max: float | None = None

    @staticmethod
    def _reduce_array(arr: np.ndarray | None) -> tuple:
        if arr is None or len(arr) == 0 or np.all(np.isnan(arr)):
            return (None, None, None, None, None)
        return (
            float(np.nanmean(arr, dtype=np.float64)),
            float(np.nanstd(arr, dtype=np.float64)),
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
            float(np.nanmedian(arr)),
        )

    @classmethod
    def from_arrays(
        cls,
        predictions: np.ndarray | None,
        uncertainties: np.ndarray | None,
    ) -> PredictionStats:
        """Legacy-parity reduction from in-memory arrays."""
        p_mean, p_std, p_min, p_max, p_median = cls._reduce_array(predictions)
        has_unc = uncertainties is not None and len(uncertainties) > 0
        u_mean = u_std = u_min = u_max = None
        if has_unc:
            u_mean, u_std, u_min, u_max, _ = cls._reduce_array(uncertainties)
        return cls(
            count=0 if predictions is None else len(predictions),
            pred_mean=p_mean,
            pred_std=p_std,
            pred_min=p_min,
            pred_max=p_max,
            pred_median=p_median,
            has_uncertainty=bool(has_unc),
            unc_mean=u_mean,
            unc_std=u_std,
            unc_min=u_min,
            unc_max=u_max,
        )

    @classmethod
    def from_lazy(
        cls, lf: pl.LazyFrame, *, with_uncertainty: bool
    ) -> PredictionStats:
        """Streaming reduction from a predictions ``LazyFrame``.

        One ``collect`` over the parquet; ddof=0 and linear-interpolation median
        match the numpy conventions of :meth:`from_arrays`.
        """
        exprs = [
            pl.len().alias('count'),
            pl.col('prediction').mean().alias('pred_mean'),
            pl.col('prediction').std(ddof=0).alias('pred_std'),
            pl.col('prediction').min().alias('pred_min'),
            pl.col('prediction').max().alias('pred_max'),
            pl.col('prediction')
            .quantile(0.5, interpolation='linear')
            .alias('pred_median'),
        ]
        if with_uncertainty:
            exprs += [
                pl.col('uncertainty').mean().alias('unc_mean'),
                pl.col('uncertainty').std(ddof=0).alias('unc_std'),
                pl.col('uncertainty').min().alias('unc_min'),
                pl.col('uncertainty').max().alias('unc_max'),
            ]
        row = lf.select(exprs).collect().to_dicts()[0]

        def _f(key: str) -> float | None:
            v = row.get(key)
            return None if v is None else float(v)

        count = int(row['count'])
        return cls(
            count=count,
            pred_mean=_f('pred_mean'),
            pred_std=_f('pred_std'),
            pred_min=_f('pred_min'),
            pred_max=_f('pred_max'),
            pred_median=_f('pred_median'),
            has_uncertainty=bool(with_uncertainty and count > 0),
            unc_mean=_f('unc_mean'),
            unc_std=_f('unc_std'),
            unc_min=_f('unc_min'),
            unc_max=_f('unc_max'),
        )
