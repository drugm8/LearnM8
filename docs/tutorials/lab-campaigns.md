# Lab Campaigns (Slow Assays)

Use `learnm8 suggest` when the measurements come from a wet lab and each round takes days to months. One command per round writes the next plate as a CSV file and exits. Nothing keeps running between rounds.

## When to Use This

`run_active_learning()` expects the oracle to answer inside one running process. That works for docking or a scoring function, but not for an assay that returns results weeks later.

`suggest_next_batch()` (CLI: `learnm8 suggest`) fits that situation instead:

- Each call reads the batch files from earlier rounds, trains on every measured value so far, and writes the next batch file.
- The folder of batch files, plus the `pruned_cycle_N.parquet` files if you [prune](#pruning), is the only state. No model, session or database is kept between calls.
- LearnM8 only creates new files. It never edits a batch file the lab has touched.

## The Round Trip

1. The computational user runs `learnm8 suggest`. It writes `batch_cycle_N.csv`.
2. The lab receives that file.
3. The lab types the measured values into the target column (for example `Activity`) and saves the file in the same folder.
4. Repeat from step 1.

After four rounds (with pruning on, which can first happen in round 3) the `batch_dir` folder looks like this:

```text
campaign/
├── batch_cycle_0.csv
├── batch_cycle_1.csv
├── batch_cycle_2.csv
├── batch_cycle_3.csv
├── pruned_cycle_3.parquet        # only when pruning, see Pruning
└── .learnm8/
    ├── cache/                    # feature cache
    ├── prediction_cycle_1.parquet
    ├── prediction_cycle_2.parquet
    └── prediction_cycle_3.parquet
```

`.learnm8/` is working data. Keep it, but the lab never needs to see it. Deleting it only costs recomputation.

## Filling in a Batch File (for the Lab)

This section can be forwarded to a bench scientist on its own.

Each round you receive a file called `batch_cycle_N.csv` (N is the round number). It lists the compounds to test, one per row. Open it in Excel (or a similar spreadsheet program), type your results into the column named after the measurement (for example `Activity`), and save it as CSV in the same folder.

What to type in that column:

| What you enter | What it means |
|----------------|---------------|
| Nothing (leave it empty) | Not measured yet. The compound is not suggested again and not used for learning. You can fill the cell in later, even in an older file. It is used from the next round on. |
| A number | Measured. The value is used for learning. Type numbers as your spreadsheet normally shows them. |
| Anything else, for example `failed`, `n/a`, `precipitated`, `<0.1` | The measurement failed. The compound is never suggested again and not used for learning. |

Things that are fine:

- Adding your own columns, for example notes or plate positions.
- Sorting the rows in a different order.
- Leaving a few rows empty if some compounds could not be tested.

Things to avoid:

- Do not rename the file. The name `batch_cycle_N.csv` tells the software which round it is.
- Do not change the values in the `ID` and `SMILES` columns. Leaving a `SMILES` cell empty is fine.
- Do not delete rows. A deleted row makes that compound available for selection again. Leave the cell empty or write `failed` instead.
- Write numbers without thousands separators. If LearnM8 cannot tell what a number means, it stops and names the file and cell so you can fix it.

When you save, keep the CSV format.

## Round 0 and Later Rounds

**Round 0** has no results yet, so there is nothing to train on. The batch is a random selection from the pool (`select_initial_batch` with the `random` strategy).

**Later rounds** train the learner on every measured value from all earlier batch files, predict the remaining pool, and choose the next batch with `--strategy` (default `greedy`). Compounds that are pending or failed are removed from the pool, so each compound appears in at most one batch.

If no compound has been measured so far but the newest batch has results (for example every well failed), there is still nothing to train on. The next call draws a fresh random batch that excludes every compound already in a batch file.

Batch file columns:

| Column | Meaning |
|--------|---------|
| `rank` | Position in the batch, 1 = most preferred. |
| `ID` | Compound identifier from the pool. |
| `<target>` | Empty. The lab fills this in. |
| `SMILES` | Canonical SMILES from the pool. |
| `prediction` | Model prediction for the compound. Present in rounds that trained a model, so not in round 0 or in a random fallback round after an all-failed batch. |
| `uncertainty` | Model uncertainty. Only in rounds that trained a model, and only when the learner supports it and the strategy needs it (for example `ucb` with `rf`). |

## CLI Walkthrough

The example uses `tests/data/small_molecules.csv` (50 compounds, target column `Activity`) with batches of 5.

**Round 0:**

```bash
learnm8 suggest small_molecules.csv campaign/ \
  --target Activity --learner rf --featurizer morgan --batch-size 5
```

```text
INFO     Initialization: selecting 5 compounds using 'random' strategy
INFO     Initialization (cycle 0) complete: 5 compounds labeled
         (strategy='random')
INFO     Wrote campaign/batch_cycle_0.csv (cycle 0, 5 compounds);
         ledger: 0 measured, 0 pending, 0 failed
Wrote 5 compounds to: campaign/batch_cycle_0.csv
Next step: fill column Activity and re-run this command.
```

`campaign/batch_cycle_0.csv` (the INFO lines print the full path). The first line, `sep=,`, is a spreadsheet hint; scripts that read the file with pandas or polars should skip it:

```text
sep=,
rank,ID,Activity,SMILES
1,ZINC000126428036,"",CCOC(=O)[C@H]1CCCCN1C(=O)c1ccc(Cl)c(C)c1
2,ZINC000408478153,"",C[C@]1(C(=O)NS(=O)(=O)c2ccc(Cl)s2)CCC(=O)NC1
3,ZINC000367985531,"",COc1cccc(CCC(=O)N2CC[C@@H](n3cc4ccccc4n3)C2)c1
4,ZINC000556719669,"",CC[C@@H]1CN(C)CCN1C(=O)c1ccc(Oc2cccc(F)c2)cn1
5,ZINC000497268293,"",C[C@H](F)CCNC(=O)[C@@H](NC(=O)OC(C)(C)C)c1cscn1
```

Running the same command again before any value is entered is refused with exit code 2:

```text
Configuration error: batch_cycle_0.csv has no results entered yet; enter the
measured values (or delete the file) before suggesting the next batch.
```

**The lab fills in the values** and saves the file:

```text
sep=,
rank,ID,Activity,SMILES
1,ZINC000126428036,-38.3,CCOC(=O)[C@H]1CCCCN1C(=O)c1ccc(Cl)c(C)c1
2,ZINC000408478153,-54.51,C[C@]1(C(=O)NS(=O)(=O)c2ccc(Cl)s2)CCC(=O)NC1
3,ZINC000367985531,-36.07,COc1cccc(CCC(=O)N2CC[C@@H](n3cc4ccccc4n3)C2)c1
4,ZINC000556719669,-40.6,CC[C@@H]1CN(C)CCN1C(=O)c1ccc(Oc2cccc(F)c2)cn1
5,ZINC000497268293,-46.45,C[C@H](F)CCNC(=O)[C@@H](NC(=O)OC(C)(C)C)c1cscn1
```

**Round 1.** Use the same flags every round:

```bash
learnm8 suggest small_molecules.csv campaign/ \
  --target Activity --learner rf --featurizer morgan --batch-size 5
```

```text
INFO     Training RandomForest(n_estimators=100,unlimited_depth) on 5 compounds
         (morgan)
INFO     Training complete (0.11s)
INFO     Cycle 1 selection path: streaming
...
INFO     Cycle 1 complete: 10 labeled, 40 unlabeled remaining
INFO     Wrote campaign/batch_cycle_1.csv (cycle 1, 5 compounds);
         ledger: 5 measured, 0 pending, 0 failed
Wrote 5 compounds to: campaign/batch_cycle_1.csv
Next step: fill column Activity and re-run this command.
```

The "10 labeled" count in the cycle log includes the 5 compounds just requested, which the cycle treats as labeled for its own bookkeeping. The `ledger:` line is the real count of results read from the folder. `rf` does support uncertainty, but `greedy` does not request it, so it is not computed. The new file has a `prediction` column but no `uncertainty` column:

```text
sep=,
rank,ID,Activity,SMILES,prediction
1,ZINC000151476988,"",Cc1ccccc1[C@H]1CN(C(=O)CCSc2nnnn2C)CCO1,-40.5767
2,ZINC000286709459,"",O=C(OC[C@H]1CCCO1)N1CCN(Cc2cn3ccccc3n2)CC1,-40.7929
3,ZINC000273977507,"",CC[C@H](C)OCc1nc(-c2ccc(N3CCOC3=O)cc2)no1,-40.8965
4,ZINC000119403872,"",CN1CCN(Cc2nc(-c3ccco3)no2)CC1=O,-40.978
5,ZINC000297629495,"",C[C@@H]1CN(C(=O)C(=O)N(C)Cc2cccc(C#N)c2)[C@@H]1c1ccccc1,-41.0338
```

Fill in the values, run the command again, and so on for as many rounds as the campaign needs.

## Python API Walkthrough

```python
from learnm8 import suggest_next_batch

batch = suggest_next_batch(
    compound_pool='small_molecules.csv',
    batch_dir='campaign/',
    target_col='Activity',
    learner='rf',
    featurizer='morgan',
    batch_size=5,
)
print(batch)
```

The return value is the batch that was just written to `batch_dir/batch_cycle_N.csv`, as a Polars DataFrame with the same columns (the target column is null). Call it again after the lab has returned the filled-in file. See the [API reference](../user-guide/api-reference.md#suggest_next_batch) for all parameters.

## Pruning

Pruning removes compounds from the pool for good. In a pruning round the model predicts the unlabeled pool, the fraction with the worst predictions (lowest for `--score-direction higher`) is removed, and the batch is chosen from the rest. Pruned compounds are never suggested again. Pruning is off by default and is irreversible: lowering or turning off `--pruning-fraction` later does not bring pruned compounds back. See [Pruning](../advanced/pruning.md) for the strategy itself.

```bash
learnm8 suggest small_molecules.csv campaign/ \
  --target Activity --learner rf --featurizer morgan --batch-size 5 \
  --pruning-fraction 0.3 --pruning-start 3
```

- `--pruning-fraction` (`pruning_fraction=`): fraction of the unlabeled pool removed in each pruning round, 0.0 to 0.9. The default 0 turns pruning off.
- `--pruning-start` (`pruning_start=`): first round, by cycle number, that may prune (default 3). Early models are trained on few measurements, and pruning on their predictions removes hits. Rounds without a trained model (round 0, or the random fallback after an all-failed batch) never prune.

**Full-batch cap (`learnm8 suggest` only).** A pruning round never prunes so far that fewer than `--batch-size` unlabeled compounds remain, so the plate stays full. If the requested fraction would remove too many, a lower fraction is used and an INFO line in the log says which; if no compound can be removed while keeping a full batch, the round does not prune and writes no prune file. This cap belongs to `learnm8 suggest` / `suggest_next_batch`: pruning in `run_active_learning` does not cap, and only guarantees one surviving compound.

Each round that prunes writes `pruned_cycle_N.parquet` into the batch folder, before `batch_cycle_N.csv`. It has an `ID` and a `SMILES` column: the compounds removed in round N and their pool SMILES. Like batch files, prune files are checked against the compound pool on every call: an unknown ID or a changed SMILES raises a `ValidationError`. If a pruned compound later has a measured value in a batch file, the measurement wins: it is used for training and a warning names the ID. Like the batch files, these files are state. Do not delete them, or the pruned compounds return to the pool. The lab never needs to open them. Deleting the newest batch file and re-running still reproduces it: prune files for that round or later are ignored, and the round's prune file is rewritten.

## Rules and Safeguards

- **No duplicate plates.** If the newest batch file has no results at all, the call raises `ConfigurationError` (CLI exit code 2). Enter the values first.
- **Reproducible.** Same inputs and `random_state` give the same batch. Deleting the newest batch file and re-running reproduces it exactly, as long as the older files are unchanged.
- **Spreadsheet re-saves are handled automatically.** The separator is read from the header line and must be a comma or a semicolon (a tab-separated file is not supported). Comma-separated files use a decimal point and semicolon-separated files a decimal comma. The encoding is UTF-8 (with or without a byte order mark) or Windows-1252. Leading zeros dropped by a spreadsheet are recovered when the match is unique. Ambiguous numbers raise an error naming the file, ID and cell.
- **Unknown IDs raise an error.** The pool file's IDs are read as text, so leading zeros there are kept. If a batch ID is not in the validated pool and cannot be matched uniquely (and consistently with its SMILES), the call raises `ValidationError` listing the IDs and files. Duplicate IDs across batch files and a missing target column are also errors that name the file.
- **SMILES are cross-checked.** If a batch row's SMILES cell is non-empty and differs from the pool's canonical SMILES for that ID, the call raises `ValidationError`. This also catches a pool file that was re-sorted when IDs are generated from row position. Rows with an empty SMILES cell cannot be checked; LearnM8 logs a warning listing them.
- **Unreadable files raise an error naming the file.** This covers empty files and files that cannot be parsed as CSV. A row with values but no ID is an error. Fully blank rows are ignored.
- **Ambiguous numbers are rejected.** A numeric-looking value that could contain a thousands separator (for example `1.234` or `1,500`) raises `ValidationError` naming the file, ID and cell instead of guessing. Ask the lab for plain digits. Values that cannot be a thousands separator, such as `.5` or `1.5e3`, are read as numbers.
- **Failed cells are logged.** Every cell treated as failed is listed in a WARNING with its raw text. Check it for values a spreadsheet may have turned into text or dates.
- **Settings are not stored.** Use the same `--learner`, `--featurizer`, `--strategy` and related flags every round. Save the command in a script so rounds stay consistent.

## Limitations

- **No multi-stage schedules.** Every round uses one strategy, `--strategy`. To change strategy part-way, pass a different `--strategy` for that round.
- **Censored values count as failed.** A value like `<0.1` or `>10` is text, so the compound is treated as failed. Enter a numeric stand-in (for example the detection limit) if the compound should be used for training.
- **Replicates.** Enter one value per compound, for example the mean. Keep the individual readings in extra columns. Two rows with the same ID raise a duplicate-ID error.
