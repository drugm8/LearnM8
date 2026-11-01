# Oracles

An oracle supplies the target values for the compounds a cycle selects: it is the "measurement" step, whether that is a lookup in a table, a computed score, or a call into an external tool. After each selection, the cycle calls the oracle, labels the compounds with the returned values, and retrains the learner.

## Oracle interface

Every oracle subclasses `Oracle` from `learnm8.core.interfaces`:

| Method | Required | Description |
|--------|----------|-------------|
| `measure(compounds, properties)` | yes | `compounds` is a Polars DataFrame with `ID` and `SMILES`; `properties` lists the target column names. Return a DataFrame with `ID` and one column per property, in the same row order as the input |
| `known_ids()` | no | Return the set of IDs the oracle can measure, or `None` (the default) if unknown. When a set is returned, the compound pool is filtered to those IDs |

## Oracle types

| Oracle | Created when | Mode | Page |
|--------|--------------|------|------|
| `CSVOracle` | `oracle=None` with a CSV pool, a `.csv` path, or a DataFrame | benchmark | [CSV oracle](csv-oracle.md) |
| `PythonOracle` | `oracle='module.py:function'` | run | [Python oracle](python-oracle.md) |
| Your own `Oracle` subclass instance | `oracle=MyOracle(...)` | run | [Python oracle](python-oracle.md#writing-an-oracle-subclass) |

The mode follows from the oracle type: a CSV oracle means benchmark mode (ground truth for every compound, full ranking metrics), anything else means run mode. There is no `mode` parameter. See [Benchmark mode vs run mode](../../tutorials/running-experiments.md#benchmark-mode-vs-run-mode).

Ready-made oracles you can copy are in [Example oracles](example-oracles.md). To add your own, see [Extending LearnM8](../../customization/extending.md).
