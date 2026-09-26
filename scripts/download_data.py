"""Download the Kenneth French daily portfolio files into data/cache/.

Run once.  Everything else in the repository reads the cache, so after this the
study runs offline, and the test suite never needs it at all.

    python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimation_error.data import DATASETS, download

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=REPO / "data" / "cache")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    for key in DATASETS:
        path = download(key, args.cache, force=args.force)
        print(f"{key:<12} {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
