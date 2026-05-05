#!/usr/bin/env python3
"""Batch-migrate all validation report directories to parquet prediction format.

Walks ``validation/reports/`` (or any other root passed via ``--root``),
finds every subdirectory that looks like an experiment output (contains
``compounds_final.csv``), and migrates each one using
``scripts.migrate_to_parquet.migrate_directory``.

Designed to be slow-and-safe:
    - Sequential or parallel processing (``--workers``, default
      ``min(os.cpu_count(), 8)``).
    - tqdm progress bar.
    - Per-directory auto-verification (the per-folder script verifies
      every parquet against the source CSV before swapping the slim CSV
      in).
    - Skip-and-log on errors: a single bad directory does not abort the
      run.
    - ``--dry-run`` prints actions but writes nothing.

Usage:
    python scripts/migrate_validation_reports.py --root validation/reports
    python scripts/migrate_validation_reports.py --root path --workers 4 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a runtime dep but degrade gracefully
    def tqdm(iterable=None, **_kwargs):  # type: ignore
        return iterable

# Allow running this script directly without installing the package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from migrate_to_parquet import migrate_directory  # noqa: E402

logger = logging.getLogger(__name__)


def find_experiment_dirs(root: Path) -> list[Path]:
    """Return every directory under ``root`` that contains compounds_final.csv."""
    dirs: list[Path] = []
    for csv_path in root.rglob('compounds_final.csv'):
        dirs.append(csv_path.parent)
    return sorted(dirs)


def _migrate_one(args: tuple) -> dict:
    """Worker entry point — migrate a single directory and return its summary."""
    output_dir, dry_run, skip_verify = args
    try:
        return migrate_directory(
            Path(output_dir), dry_run=dry_run, skip_verify=skip_verify,
        )
    except Exception as exc:  # skip-and-log policy
        return {
            'status': 'failed',
            'output_dir': str(output_dir),
            'cycles': [],
            'parquets_written': [],
            'error': repr(exc),
        }


def run(
    root: Path,
    *,
    workers: int,
    dry_run: bool,
    skip_verify: bool,
    log_path: Path | None = None,
) -> dict:
    """Migrate every experiment directory under ``root``.

    Returns aggregate summary with per-status counts and a list of failed
    directories (for re-run).
    """
    dirs = find_experiment_dirs(root)
    logger.info('Found %d experiment directories under %s', len(dirs), root)

    counters: dict[str, int] = {}
    failures: list[dict] = []

    payload = [(d, dry_run, skip_verify) for d in dirs]

    if workers <= 1:
        progress = tqdm(payload, total=len(payload), desc='migrate')
        for arg in progress:
            summary = _migrate_one(arg)
            status = summary.get('status', 'failed')
            counters[status] = counters.get(status, 0) + 1
            if status not in ('migrated', 'already_migrated', 'dry_run'):
                failures.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_migrate_one, arg): arg[0] for arg in payload}
            progress = tqdm(as_completed(futures), total=len(futures), desc='migrate')
            for fut in progress:
                summary = fut.result()
                status = summary.get('status', 'failed')
                counters[status] = counters.get(status, 0) + 1
                if status not in ('migrated', 'already_migrated', 'dry_run'):
                    failures.append(summary)

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('w') as fh:
                fh.write(f'# migration log for root={root}\n')
                for status, count in counters.items():
                    fh.write(f'# {status}: {count}\n')
                if failures:
                    fh.write('\n# failures:\n')
                    for failure in failures:
                        fh.write(
                            f"{failure.get('status')}\t{failure.get('output_dir')}\t"
                            f"{failure.get('error')}\n"
                        )
        except OSError as exc:
            logger.warning('Failed to write log to %s: %s', log_path, exc)

    return {
        'total': len(dirs),
        'counters': counters,
        'failures': failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n\n')[0])
    parser.add_argument(
        '--root', required=True, type=Path,
        help='Root directory to scan for experiment output dirs',
    )
    parser.add_argument(
        '--workers', type=int,
        default=min(os.cpu_count() or 1, 8),
        help='Number of worker processes (default: min(cpu_count, 8))',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print actions without writing files',
    )
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip auto-verification of parquet files (faster, less safe)',
    )
    parser.add_argument(
        '--log', type=Path, default=None,
        help='Optional log file path for migration summary and failures',
    )
    parser.add_argument(
        '--quiet', action='store_true', help='Suppress info logging',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    root = args.root.resolve()
    if not root.is_dir():
        logger.error('Not a directory: %s', root)
        return 2

    summary = run(
        root,
        workers=args.workers,
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
        log_path=args.log,
    )

    logger.info('Done. total=%d counters=%s', summary['total'], summary['counters'])
    if summary['failures']:
        logger.warning('Failures: %d (see log for details)', len(summary['failures']))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
