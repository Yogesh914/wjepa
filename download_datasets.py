#!/usr/bin/env python3
"""Download and prepare the six W-JEPA benchmark datasets with whar-datasets."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from config import BENCHMARK_DATASETS, WHAR_DATASET_IDS
from whar_datasets import PreProcessingPipeline, WHARDatasetID, get_dataset_cfg

# The pinned MHealth and USC-HAD parsers number subjects and sessions in the order
# the filesystem lists their raw files, so the splits could differ between machines.
# They are always parsed in the order of the original runs instead.
MHEALTH_FILES = [f"mHealth_subject{s}.log" for s in (9, 8, 7, 6, 5, 4, 3, 2, 10, 1)]
FIXED_ORDER_DATASETS = ("mhealth", "usc_had")


@contextmanager
def fixed_file_order(raw_dir: Path):
    """List MHealth subjects in MHEALTH_FILES order and walk USC-HAD in reverse
    lexicographic order; yields the list of directories the order was applied to."""
    listdir, walk, applied = os.listdir, os.walk, []

    def ordered_listdir(path="."):
        names = listdir(path)
        if Path(path).resolve() != (raw_dir / "MHEALTHDATASET").resolve():
            return names
        if sorted(n for n in names if n.endswith(".log")) != sorted(MHEALTH_FILES):
            raise RuntimeError(f"Unexpected MHealth subject files in {path}")
        applied.append(path)
        return MHEALTH_FILES + [n for n in names if not n.endswith(".log")]

    def ordered_walk(top, *args, **kwargs):
        reverse = Path(top).resolve() == (raw_dir / "USC-HAD").resolve()
        if reverse:
            applied.append(top)
        for root, dirs, files in walk(top, *args, **kwargs):
            if reverse:
                dirs.sort(reverse=True)
                files.sort(reverse=True)
            yield root, dirs, files

    os.listdir, os.walk = ordered_listdir, ordered_walk
    try:
        yield applied
    finally:
        os.listdir, os.walk = listdir, walk


def prepare_dataset(name: str, datasets_dir: Path, force_recompute: bool = False) -> None:
    """Download, extract, and convert one dataset to the common WHAR format."""
    dataset_id = WHARDatasetID(WHAR_DATASET_IDS.get(name, name))
    pipeline = PreProcessingPipeline(get_dataset_cfg(dataset_id, datasets_dir=str(datasets_dir)))
    if name not in FIXED_ORDER_DATASETS:
        pipeline.run(force_recompute=force_recompute)
        return
    # Always re-parse (and re-window) so a cache written in another order is never reused.
    with fixed_file_order(pipeline.data_dir) as applied:
        pipeline.run(force_recompute=[force_recompute, force_recompute, True, True])
    if not applied:
        raise RuntimeError(f"{name}: the parser did not use the fixed file order")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare the W-JEPA benchmark datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=BENCHMARK_DATASETS,
        default=list(BENCHMARK_DATASETS),
        help="Benchmarks to prepare (default: all six).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
        help="Output directory (default: the repository's datasets directory).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore valid cached artifacts and rerun every processing step.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue preparing other datasets after an error.",
    )
    args = parser.parse_args(argv)

    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, Exception]] = []

    for index, name in enumerate(args.datasets, start=1):
        print(f"[{index}/{len(args.datasets)}] Preparing {name}")
        try:
            prepare_dataset(name, args.datasets_dir, force_recompute=args.force_recompute)
        except Exception as error:  # report the dataset that failed before exiting
            failures.append((name, error))
            print(f"Failed to prepare {name}: {error}")
            if not args.keep_going:
                break

    if failures:
        print("Failed datasets:")
        for name, error in failures:
            print(f"  {name}: {error}")
        return 1

    print(f"Prepared {len(args.datasets)} dataset(s) in {args.datasets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
