# Publication figures and tables

Everything needed to regenerate the figures and tables in the LearnM8 application
note. Most of them rebuild from this repository alone, because the benchmark runs
have been reduced to one tidy table (`data/benchmark_tidy.parquet`, about 670 KB)
that is tracked here. Three tiers follow from that:

Figure numbers below are the manuscript's.

- **Rebuilds from this repository alone:** Figures 2 and 3, Figures S1 and S3, and
  the tables.
- **Needs the Zenodo deposit:** Figure 4 (the pruning runs and the 10M AmpC
  library), Figure S2 (the 1M acquisition runs), and `--rebuild-data`.
- **Needs files that are not deposited:** `fig04_pruning.png`, an alternative
  pruning figure from `fig04_batch_pruning.py` that the manuscript does not use,
  reads the per-cycle prediction files. With the deposit in place, the script writes
  `fig04_batch.png` (Figure 3) and then stops with `missing prediction artifact`, so
  `make_all.py` lists it as failed.

> **Raw data.** The Zenodo deposit, DOI
> [10.5281/zenodo.23941758](https://doi.org/10.5281/zenodo.23941758), holds the 209
> runs cited in the manuscript and its Supporting Information (configuration,
> per-cycle metrics and per-compound files) and the five scored AmpC and D4
> libraries. The per-cycle prediction files (81 GB) are not deposited; they are
> available from the authors on request. The deposit's README lists the files each
> figure needs.

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

| Script | Manuscript | Outputs (under `figures/png/`, `tables/`) | Needs beyond this repo |
| ------ | ---------- | ----------------------------------------- | ---------------------- |
| `fig02_learners_acquisition_b05.py` | Figure 2 | `fig02_learners_acquisition_b05.png` | — |
| `fig04_batch_pruning.py` | Figure 3 | `fig04_batch.png` (both seed designs, one row each); `fig04_pruning.png` (not in the manuscript) | for `fig04_pruning.png`: `~/LearnM8_DATA/AmpC_screen_10000K.csv` and the per-cycle prediction files, which are not deposited |
| `fig04_pruning_paper.py` | Figure 4 | `fig04_pruning_paper.png` | Zenodo `runs_metrics.zip` and `runs_10M_pruning.zip` (in `~/LearnM8_RESULTS_FINAL`), `AmpC_screen_10000K.csv.gz` (in `~/LearnM8_DATA`) |
| `figS1_score_distributions.py` | Figure S1 | `figS1_score_distributions.png` | — (uses the tracked histogram cache) |
| `fig11_top_candidate_novelty.py` | Figure S2 | `fig11_top_candidate_novelty.png`, `tables/fig11_selection_overlap.csv` | Zenodo `runs_metrics.zip` and `runs_1M_learners_acquisition.zip` (in `~/LearnM8_RESULTS_FINAL`) |
| `fig12_target_comparison.py` | Figure S3 | `fig12_target_comparison.png` | — |
| `tables.py` | SI tables | `table2_benchmark_summary{,_fixedinit}.csv`, `si_all_runs_per_cycle{,_fixedinit}.csv` | — |
| `fig02_learners_acquisition.py` | — | `fig02_learners_acquisition.png` (all three batch fractions; also provides the row drawing for Figure 2) | — |
| `fig02_learners.py` | — | `fig02_learners.png`, `fig02_learners_fixedinit.png` | — |
| `fig03_acquisition.py` | — | `fig03_acquisition.png`, `fig03_acquisition_fixedinit.png` | — |
| `fig05_scaling.py` | — | `fig05_scaling.png`, `fig05b_scaling_parallel.png`, `fig05c_breakeven_by_machine.png` | — |

Supporting modules, not run directly: `style.py` (palette, figure geometry, `save()`),
`data.py` (loads the tidy table), `analysis.py` (shared acquisition-overlap analysis
used by Figure 11).

Figure 1, the architecture overview, is not produced by any script here. It is a
hand-drawn SVG in the poster source that accompanies the manuscript, rendered to PNG
with a headless browser. The repository's root `README.md` embeds a separate
hand-authored diagram, `../media/fig01_architecture.svg`.

Several figures render twice, once per cycle-0 seed design: `matched` (each arm seeds
with its own batch fraction) and `fixed` (a constant 1% seed, files suffixed
`_fixedinit`). They are different experiments and are never pooled into one curve.

## Rebuilding the tidy table

`data/benchmark_tidy.parquet` is the reduction of every benchmark run to one row per
run per cycle. It is tracked so the figures rebuild in seconds without downloading
anything. To regenerate it from the raw runs, unzip `runs_metrics.zip` from the
Zenodo deposit into your home directory (this creates `~/LearnM8_RESULTS_FINAL`),
then:

```bash
python publication/scripts/make_all.py --rebuild-data
```

This overwrites the tracked table. The deposit holds only the 209 runs the
manuscript cites, so the rebuilt table lacks the other 128 runs of the tracked one,
and scripts for figures outside the manuscript may find no data
(`git checkout -- publication/data/benchmark_tidy.parquet` restores it).

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
