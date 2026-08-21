"""Rebuild every publication figure and table.

    python publication/scripts/make_all.py [--rebuild-data]

Figures read benchmark_tidy.parquet, so pass --rebuild-data only when the run
directories have changed (e.g. after the 100M run lands).

The fig*.py list is globbed from this directory, so a fresh clone renders
exactly the tracked figure scripts. Scripts whose status is still undecided
are gitignored (see .gitignore) and only run for someone who has them locally.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The selection-history calibration figure uses a different evidence stratum.
EXCLUDED_SCRIPTS = {'fig07_uncertainty_calibration.py'}

# Scripts that render once per cycle-0 seed design. Everything else is rendered
# once: fig05, fig08 and fig09 restrict to the 1% batch, where the two designs
# are the same run, so a second variant would be a byte-identical duplicate.
DESIGN_AWARE = {
    'fig02_learners.py',
    'fig03_acquisition.py',
    'tables.py',
}
DESIGNS = ('matched', 'fixed')


def main() -> None:
    scripts = ['style.py']
    if '--rebuild-data' in sys.argv:
        scripts.append('data.py')
    scripts += [
        *sorted(p.name for p in HERE.glob('fig*.py') if p.name not in EXCLUDED_SCRIPTS),
        'tables.py',
    ]

    for script in scripts:
        variants = DESIGNS if script in DESIGN_AWARE else (None,)
        for design in variants:
            args = [] if design is None else ['--design', design]
            label = script if design is None else f'{script} --design {design}'
            print(f'\n--- {label}')
            subprocess.run([sys.executable, str(HERE / script), *args], check=True)


if __name__ == '__main__':
    main()
