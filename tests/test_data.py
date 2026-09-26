"""Tests for the French library loader.

All of these run against a synthetic file written here in the library's format,
so the suite never touches the network and CI needs no data cache.  What is
being tested is the parsing of the quirks, not the contents of the real file:
the multi-table layout, both missing-value codes, the percent scaling and the
date format.  A loader that silently mishandles any of those produces a study
that is wrong in a way no downstream test would catch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from estimation_error import data

FAKE_FILE = """This file was created using a fake CRSP database.
It contains value- and equal-weighted returns for 3 fake portfolios.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Daily
,Aaa,Bbb,Ccc
19700102,   1.00,  -2.00, -99.99
19700105,   0.50,   0.25,   3.00
19700106,-999.00,   1.00,  -1.00
19700107,   2.00,   0.00,   0.50

  Average Equal Weighted Returns -- Daily
,Aaa,Bbb,Ccc
19700102,  99.00,  99.00,  99.00
19700105,  99.00,  99.00,  99.00

Copyright 2026 Nobody
"""


@pytest.fixture
def fake_cache(tmp_path: Path) -> Path:
    """A cache directory holding a zip in the French library's shape."""
    archive = tmp_path / data.DATASETS["industry49"].zip_name
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("fake_Daily.csv", FAKE_FILE)
    return tmp_path


def test_parses_the_first_table_and_ignores_the_others():
    """The equal-weighted table must not leak into the value-weighted one."""
    frame = data._parse_table(FAKE_FILE, "Average Value Weighted Returns -- Daily")
    assert list(frame.columns) == ["Aaa", "Bbb", "Ccc"]
    assert len(frame) == 4
    assert 99.0 not in frame.to_numpy()


def test_stops_at_the_copyright_line():
    frame = data._parse_table(FAKE_FILE, "Average Equal Weighted Returns -- Daily")
    assert len(frame) == 2


def test_missing_table_raises():
    with pytest.raises(ValueError):
        data._parse_table(FAKE_FILE, "A Table That Is Not There")


def test_dates_are_parsed_as_dates():
    frame = data._parse_table(FAKE_FILE, "Average Value Weighted Returns -- Daily")
    assert str(frame.index[0].date()) == "1970-01-02"
    assert frame.index.is_monotonic_increasing


def test_both_missing_codes_are_treated_as_gaps(fake_cache):
    """-99.99 and -999 both mean missing; treating -999 as a return would be catastrophic.

    With the default coverage threshold, Ccc (one gap in four days, 75%) and Aaa
    (likewise) both fall below it and are dropped, leaving the one complete
    series.  The point of the test is that nothing anywhere near -999 or -99.99
    survives into the returns.
    """
    frame = data.load_returns("industry49", fake_cache, start="1970-01-01", min_coverage=0.99)
    assert frame.to_numpy().min() > -1.0
    assert not np.isnan(frame.to_numpy()).any()
    assert list(frame.columns) == ["Bbb"]


def test_returns_are_converted_from_percent(fake_cache):
    """A 1.00 in the file is a 1% return, so 0.01 after loading."""
    frame = data.load_returns("industry49", fake_cache, start="1970-01-01", min_coverage=0.0)
    assert frame.loc["1970-01-05", "Aaa"] == pytest.approx(0.005)
    assert frame.loc["1970-01-05", "Ccc"] == pytest.approx(0.03)


def test_coverage_threshold_drops_columns_not_rows(fake_cache):
    """A portfolio with gaps is dropped whole; the days it would have cost are kept."""
    lenient = data.load_returns("industry49", fake_cache, start="1970-01-01", min_coverage=0.0)
    strict = data.load_returns("industry49", fake_cache, start="1970-01-01", min_coverage=0.99)
    assert lenient.shape[1] == 3
    assert strict.shape[1] == 1
    # Dropping the incomplete columns recovers the days they would have cost.
    assert len(strict) == 4
    assert len(lenient) == 2


def test_dropped_columns_are_reported(fake_cache):
    frame = data.load_returns("industry49", fake_cache, start="1970-01-01", min_coverage=0.99)
    assert set(frame.attrs["dropped_columns"]) == {"Aaa", "Ccc"}
    assert "dropped 2 incomplete portfolios" in data.describe(frame)


def test_date_filtering(fake_cache):
    frame = data.load_returns(
        "industry49", fake_cache, start="1970-01-05", end="1970-01-06", min_coverage=0.0
    )
    assert len(frame) == 1  # only the 5th survives, the 6th has a gap in Aaa


def test_missing_cache_raises_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load_returns("industry49", tmp_path)


def test_empty_window_raises(fake_cache):
    with pytest.raises(ValueError):
        data.load_returns("industry49", fake_cache, start="2100-01-01")


def test_dataset_registry_is_consistent():
    for key, dataset in data.DATASETS.items():
        assert dataset.key == key
        assert dataset.zip_name.endswith(".zip")
        assert dataset.n_expected > 0
