"""Rebuild every publication figure and table.

    python scripts/publication/make_all.py [--rebuild-data]

Figures read benchmark_tidy.parquet, so pass --rebuild-data only when the run
directories have changed (e.g. after the 100M run lands).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    scripts = ['style.py']
    if '--rebuild-data' in sys.argv:
        scripts.append('data.py')
    scripts += [*sorted(p.name for p in HERE.glob('fig*.py')), 'tables.py']

    for script in scripts:
        print(f'\n--- {script}')
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == '__main__':
    main()
