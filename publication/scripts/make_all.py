"""Rebuild every publication figure and table.

    python publication/scripts/make_all.py [--rebuild-data]

Figures read benchmark_tidy.parquet, so pass --rebuild-data only when the run
directories have changed (e.g. after new runs are added).

The fig*.py list is globbed from this directory, so a fresh clone renders
exactly the tracked figure scripts. Scripts whose status is still undecided
are gitignored (see .gitignore) and only run for someone who has them locally.

Every script runs even if an earlier one fails. A script exits with
``style.EXIT_RAW_DATA_MISSING`` (3) when its raw input, deposited on Zenodo and
not tracked in the repository, is absent: that is reported as SKIPPED and does
not affect the exit status. Any other non-zero return code is a failure. A
summary of built / skipped / failed scripts is printed at the end, and the exit
status is 1 if anything failed, else 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from style import EXIT_RAW_DATA_MISSING

HERE = Path(__file__).resolve().parent

# Scripts that render once per cycle-0 seed design. Everything else is rendered
# once, either for the matched design or (fig05) at the 1% batch, where the two
# designs are the same runs, so a second variant would be a byte-identical
# duplicate.
DESIGN_AWARE = {
    'fig02_learners.py',
    'fig03_acquisition.py',
    'tables.py',
}
DESIGNS = ('matched', 'fixed')


def run_scripts(jobs: list[tuple[str, Path, list[str]]]) -> int:
    """Run each (label, script, args) job; print a summary; return the exit status."""
    built: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for label, path, args in jobs:
        print(f'\n--- {label}', flush=True)
        code = subprocess.run(
            [sys.executable, str(path), *args], check=False
        ).returncode
        if code == 0:
            built.append(label)
        elif code == EXIT_RAW_DATA_MISSING:
            print(f'SKIPPED (raw data not available): {label}', flush=True)
            skipped.append(label)
        else:
            print(f'FAILED (exit code {code}): {label}', flush=True)
            failed.append(f'{label} (exit {code})')

    print('\n=== Summary')
    for title, names in (('built', built), ('skipped', skipped), ('failed', failed)):
        print(f'{title}: {len(names)}')
        for name in names:
            print(f'  {name}')
    return 1 if failed else 0


def main() -> int:
    scripts = ['style.py']
    if '--rebuild-data' in sys.argv:
        scripts.append('data.py')
    scripts += [
        *sorted(p.name for p in HERE.glob('fig*.py')),
        'tables.py',
    ]

    jobs: list[tuple[str, Path, list[str]]] = []
    for script in scripts:
        variants = DESIGNS if script in DESIGN_AWARE else (None,)
        for design in variants:
            args = [] if design is None else ['--design', design]
            label = script if design is None else f'{script} --design {design}'
            jobs.append((label, HERE / script, args))
    return run_scripts(jobs)


if __name__ == '__main__':
    sys.exit(main())
