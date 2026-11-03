"""F5 (paper) - pruning on 10M AmpC, three panels in the paper's theme and font.

The poster version of this figure is ``poster/poster_fig04_pruning.py``; both draw
through the same code, so the numbers match. This wrapper only selects the paper
sizes, which keeps the figure out of the poster build and in ``make_all.py``'s glob.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'poster'))

from poster_fig04_pruning import main_paper

if __name__ == '__main__':
    main_paper()
