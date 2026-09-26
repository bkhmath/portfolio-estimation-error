"""Loading the Kenneth French daily portfolio returns.

The estimators need a real cross-section with a genuine factor structure and
enough history for a rolling protocol to roll over.  The French library gives
both, it is free and unauthenticated, and it is what the literature on this
question uses, including DeMiguel, Garlappi and Uppal, whose 1/N result is the
benchmark this project has to answer.  Using the same data as the paper you are
arguing with beats using better data.

Two datasets, picked for their aspect ratios.  The 49 industry portfolios with
a 250-day window give q = 49/249 = 0.20, a realistic working regime.  The 100
size and book-to-market portfolios give q = 0.40, the stress case where random
matrix effects are unmissable.  Running the same code on both, and seeing the
damage scale with q the way the theory says, is the point of having two.

A caveat about the 100 portfolios, better stated here than discovered in the
results: they come from sorting on two characteristics, so neighbouring
portfolios hold overlapping stocks and are much more correlated with each other
than 100 individual stocks would be.  Their true covariance is closer to low
rank than a generic cross-section, which flatters any estimator assuming low
rank.  The industry portfolios are less degenerate that way.

File format quirks, all handled in `_parse_table`:

- Several tables sit in one CSV, separated by blank lines, each introduced by a
  title line and then a header row starting with a comma.  We want the first,
  "Average Value Weighted Returns -- Daily"; the rest are equal-weighted
  returns, firm counts and average firm sizes.
- Missing values are coded `-99.99` *and* `-999`.  The file header says so.  A
  loader that only knows about -99.99 treats -999 as a -99900% return, which is
  not subtle but also does not raise.
- Returns are in percent and get divided by 100 on load.  Getting that wrong
  scales every variance by 10^4 and cancels out of any ratio, so it is the kind
  of mistake that survives to publication.
- Dates are YYYYMMDD integers.
- The file ends with a copyright line, which is not a data row.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: Codes the French library uses for missing observations.
MISSING_CODES = (-99.99, -999.0, -999.99)


@dataclass(frozen=True)
class Dataset:
    """One French library file and the table inside it that we want."""

    key: str
    zip_name: str
    table_title: str
    label: str
    n_expected: int


DATASETS = {
    "industry49": Dataset(
        key="industry49",
        zip_name="49_Industry_Portfolios_daily_CSV.zip",
        table_title="Average Value Weighted Returns -- Daily",
        label="49 industry portfolios",
        n_expected=49,
    ),
    "size_bm100": Dataset(
        key="size_bm100",
        zip_name="100_Portfolios_10x10_Daily_CSV.zip",
        table_title="Average Value Weighted Returns -- Daily",
        label="100 size / book-to-market portfolios",
        n_expected=100,
    ),
}

#: Default start of the sample.  Fixed before any result was inspected.  The
#: industry file begins in 1926, but several industries have no constituents at
#: all for long stretches of the early sample, and a covariance matrix
#: estimated over a window where a column is constant is singular for a reason
#: that has nothing to do with this project's subject.
DEFAULT_START = "1970-01-01"


def cache_path(dataset_key: str, cache_dir: Path) -> Path:
    """Where the downloaded zip for this dataset lives."""
    return Path(cache_dir) / DATASETS[dataset_key].zip_name


def download(dataset_key: str, cache_dir: Path, force: bool = False) -> Path:
    """Fetch one dataset's zip archive into the cache and return its path.

    Kept separate from loading so that the test suite and a clean clone never
    touch the network: `make data` downloads, everything else reads the cache
    or works on simulated data.
    """
    import requests  # imported here so the package imports without it

    dataset = DATASETS[dataset_key]
    destination = cache_path(dataset_key, cache_dir)
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(FRENCH_BASE_URL + dataset.zip_name, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _parse_table(text: str, table_title: str) -> pd.DataFrame:
    """Pull one blank-line-delimited table out of a French library CSV.

    Finds the title line, skips to the header row underneath it, then reads
    data rows until the first line that is not a date followed by numbers.
    """
    lines = text.split("\n")

    start = None
    for i, line in enumerate(lines):
        if line.strip() == table_title:
            start = i
            break
    if start is None:
        raise ValueError(f"table {table_title!r} not found in file")

    header_index = None
    for i in range(start + 1, min(start + 6, len(lines))):
        if lines[i].startswith(","):
            header_index = i
            break
    if header_index is None:
        raise ValueError(f"no header row found under {table_title!r}")

    columns = [c.strip() for c in lines[header_index].split(",")[1:]]

    dates: list[str] = []
    rows: list[list[float]] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        fields = stripped.split(",")
        if len(fields) != len(columns) + 1 or not fields[0].strip().isdigit():
            break
        dates.append(fields[0].strip())
        rows.append([float(v) for v in fields[1:]])

    frame = pd.DataFrame(rows, columns=columns, index=pd.to_datetime(dates, format="%Y%m%d"))
    frame.index.name = "date"
    return frame


def load_returns(
    dataset_key: str,
    cache_dir: Path,
    start: str | None = DEFAULT_START,
    end: str | None = None,
    min_coverage: float = 0.99,
) -> pd.DataFrame:
    """Load daily returns as decimals, with missing codes turned into gaps and removed.

    `min_coverage` is the fraction of the requested window a portfolio must be
    present for to be kept; portfolios below it are dropped as columns, and any
    day still carrying a gap afterwards is dropped as a row.  Both counts end up
    in the result's `attrs` and in `describe`, since silently changing N is how
    a q quoted in a README ends up wrong.

    The threshold is 0.99 rather than 1.0 for a concrete reason visible in the
    data.  Requiring a perfect record throws out the software industry over a
    single missing day in fifty-six years, which costs a whole asset to save
    one observation.  The portfolios genuinely worth dropping are the extreme
    corners of the size and book-to-market sort, empty whenever no firm happens
    to fall in them, and those are missing between 250 and 2145 days, far below
    any threshold in this region.  The gap
    between one missing day and a thousand is wide enough that the exact cutoff
    changes nothing; 0.99 sits in the middle of it.
    """
    dataset = DATASETS[dataset_key]
    archive = cache_path(dataset_key, cache_dir)
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} is not in the cache. Run `make data` (or "
            f"`python scripts/download_data.py`) once to fetch it."
        )

    with zipfile.ZipFile(archive) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1")

    frame = _parse_table(raw, dataset.table_title)
    frame = frame.replace(list(MISSING_CODES), np.nan)
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
    if frame.empty:
        raise ValueError(f"no observations for {dataset_key} in the requested window")

    coverage = frame.notna().mean(axis=0)
    kept = coverage[coverage >= min_coverage].index
    dropped = [c for c in frame.columns if c not in set(kept)]
    frame = frame[list(kept)]

    rows_before = len(frame)
    frame = frame.dropna(axis=0, how="any")

    frame = frame / 100.0
    frame.attrs["dataset"] = dataset_key
    frame.attrs["label"] = dataset.label
    frame.attrs["dropped_columns"] = dropped
    frame.attrs["dropped_rows"] = rows_before - len(frame)
    return frame


def describe(frame: pd.DataFrame) -> str:
    """A one-paragraph summary of a loaded panel, for the study log and the README."""
    n_obs, n_assets = frame.shape
    dropped = frame.attrs.get("dropped_columns", [])
    text = (
        f"{frame.attrs.get('label', 'panel')}: {n_assets} portfolios, {n_obs} daily "
        f"observations from {frame.index[0].date()} to {frame.index[-1].date()}, "
        f"annualised volatility {float(frame.std().mean()) * np.sqrt(252) * 100:.1f}% on average"
    )
    if dropped:
        text += f"; dropped {len(dropped)} incomplete portfolios ({', '.join(dropped)})"
    if frame.attrs.get("dropped_rows"):
        text += f"; dropped {frame.attrs['dropped_rows']} incomplete days"
    return text
