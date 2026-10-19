# Publication figures and tables

Everything needed to regenerate the figures and tables in the LearnM8 application
note. Most of them rebuild from this repository alone, because the benchmark runs
have been reduced to one tidy table (`data/benchmark_tidy.parquet`, about 650 KB)
that is tracked here. Two tiers follow from that:

- **Rebuilds from this repository alone:** Figures 2, 3, 5, 12 and S1, Figure 4's
  batch panel, and the tables.
- **Needs the Zenodo raw data:** Figure 4's pruning panel (the scored AmpC pool),
  Figure 11 (the run archive), and `--rebuild-data`. Each is called out below.

<!-- TODO: Zenodo DOI - replace both placeholders below once the deposition is minted. -->

> **Raw data.** The full run archive and the scored AmpC/D4 pools are deposited at
> Zenodo, DOI `10.5281/zenodo.XXXXXXX`. You only need it for `--rebuild-data`,
> Figure 4's pruning panel, and Figure 11.

## Environment

```bash
conda env create -f ../environment.yml
conda activate learnm8
pip install --no-deps -e ..
```

The scripts import `learnm8` itself (for scaffold and similarity helpers), so the
editable install is required, not optional. `--no-deps` is correct because the
conda environment already provides every dependency.

## Rebuilding everything

```bash
python publication/scripts/make_all.py
```

Run it from anywhere; every path is resolved relative to the script file. Figures
are written as 600 dpi PNG to `figures/png/`, tables as CSV to `tables/`. Neither
directory is tracked — they are outputs, and the repository ships the code that
produces them rather than the products.

`make_all.py` globs `fig*.py` from `scripts/`, so a fresh clone renders exactly the
tracked figure scripts. Scripts that are gitignored or untracked in an author's
working tree are not part of the published record, and a fresh clone does not have them.

A script whose raw data is absent prints one line to stderr naming the missing path
and exits with code 3 (`EXIT_RAW_DATA_MISSING` in `style.py`). `make_all.py` reports
those scripts as `SKIPPED (raw data not available)`, carries on with the rest, and
prints a built / skipped / failed summary. It exits 0 when only skips occurred, so
a fresh clone without the Zenodo data finishes cleanly; code 3 is internal to the
scripts and is never `make_all.py`'s own exit status. Any other non-zero return
code is a failure: the remaining scripts still run, and `make_all.py` exits 1 and
lists the failed scripts. Figure 4 still writes its batch panel when only the
pruning panel is skipped.

## What each script produces

| Script                           | Outputs (under `figures/png/`, `tables/`)                                             | Needs beyond this repo                                        |
| -------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `fig02_learners.py`              | `fig02_learners.png`, `fig02_learners_fixedinit.png`                                  | —                                                             |
| `fig03_acquisition.py`           | `fig03_acquisition.png`, `fig03_acquisition_fixedinit.png`                            | —                                                             |
| `fig04_batch_pruning.py`         | `fig04_batch.png` (both seed designs, one row each), `fig04_pruning.png`              | `~/LearnM8_DATA/AmpC_screen_10000K.csv` for the pruning panel |
| `fig05_scaling.py`               | `fig05_scaling.png`, `fig05b_scaling_parallel.png`, `fig05c_breakeven_by_machine.png` | —                                                             |
| `fig11_top_candidate_novelty.py` | `fig11_top_candidate_novelty.png`, `tables/fig11_selection_overlap.csv`               | raw run archive at `~/LearnM8_RESULTS_FINAL`                  |
| `fig12_target_comparison.py`     | `fig12_target_comparison.png`                                                         | —                                                             |
| `figS1_score_distributions.py`   | `figS1_score_distributions.png`                                                       | — (uses the tracked histogram cache)                          |
| `tables.py`                      | `table2_benchmark_summary{,_fixedinit}.csv`, `si_all_runs_per_cycle{,_fixedinit}.csv` | —                                                             |

Supporting modules, not run directly: `style.py` (palette, figure geometry, `save()`),
`data.py` (loads the tidy table), `analysis.py` (shared acquisition-overlap analysis
used by Figure 11).

Figure 1 is the architecture diagram. It is hand-authored SVG rather than generated,
lives at `../media/fig01_architecture.svg`, and is embedded in the repository's root
`README.md`.

Several figures render twice, once per cycle-0 seed design: `matched` (each arm seeds
with its own batch fraction) and `fixed` (a constant 1% seed, files suffixed
`_fixedinit`). They are different experiments and are never pooled into one curve.

## Rebuilding the tidy table

`data/benchmark_tidy.parquet` is the reduction of every benchmark run to one row per
run per cycle. It is tracked so the figures rebuild in seconds without downloading
anything. To regenerate it from the raw runs, download the archive from the Zenodo
DOI above, extract it to `~/LearnM8_RESULTS_FINAL`, then:

```bash
python publication/scripts/make_all.py --rebuild-data
```

Reading the run directories takes minutes; reading the parquet takes milliseconds.

## Tracked vs. generated

Tracked: the scripts, `data/benchmark_tidy.parquet`, the two small Figure S1 inputs
(`data/d4_116M_score_hist.csv`, `data/score_distributions.parquet` — the D4
histogram cannot be recomputed without the 116M scored pool), and this README.

Not tracked: `figures/`, `tables/`, `plans/`, and every other file under `data/`.
Rendered figures are regenerable output; keeping them out of history is why this
directory is small.

`plans/` holds the authors' working planning documents. It is gitignored and is not
part of the published record.
