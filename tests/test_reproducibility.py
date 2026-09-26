"""Two runs of the same thing must agree.

Reproducibility is not a nice-to-have here.  The whole project is a claim that
one estimator beats another by a few percent of realised volatility, and a few
percent is exactly the size of the difference an unseeded random number
generator would produce between runs.  If the pipeline is not deterministic,
none of the comparisons mean anything.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from estimation_error import backtest, estimators, rmt
from estimation_error.simulate import (
    factor_model_covariance,
    simulate_returns,
    simulate_white_noise,
)

REPO = Path(__file__).resolve().parents[1]


def test_simulation_is_reproducible_from_a_seed():
    first = simulate_white_noise(20, 60, np.random.default_rng(42))
    second = simulate_white_noise(20, 60, np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_different_seeds_give_different_data():
    """Guards against a seeding bug that would make every draw identical."""
    first = simulate_white_noise(20, 60, np.random.default_rng(1))
    second = simulate_white_noise(20, 60, np.random.default_rng(2))
    assert not np.array_equal(first, second)


def test_true_covariance_is_reproducible():
    first = factor_model_covariance(30, 3, np.random.default_rng(7))
    second = factor_model_covariance(30, 3, np.random.default_rng(7))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", list(estimators.ESTIMATORS))
def test_estimators_are_deterministic(name):
    """No estimator may depend on hidden randomness."""
    returns = simulate_returns(
        factor_model_covariance(40, 3, np.random.default_rng(7)), 200, np.random.default_rng(1)
    )
    assert np.array_equal(
        estimators.ESTIMATORS[name](returns), estimators.ESTIMATORS[name](returns)
    )


def test_backtest_is_deterministic():
    truth = factor_model_covariance(20, 2, np.random.default_rng(1))
    values = simulate_returns(truth, 700, np.random.default_rng(4)) * 0.01
    panel = pd.DataFrame(values, index=pd.bdate_range("2000-01-03", periods=700))

    strategies = ["sample", "clipped", "linear", "nonlinear", "factor", "equal"]
    first = backtest.run_backtest(panel, strategies, window=250, step=21)
    second = backtest.run_backtest(panel, strategies, window=250, step=21)

    for name in strategies:
        assert np.array_equal(first[name].daily_returns, second[name].daily_returns)
        assert np.array_equal(first[name].predicted_variances, second[name].predicted_variances)


def test_marchenko_pastur_functions_are_pure():
    """Repeated calls must return identical values, not merely close ones."""
    x = np.linspace(0.1, 3.0, 50)
    assert np.array_equal(rmt.mp_density(x, 0.4), rmt.mp_density(x, 0.4))
    assert np.array_equal(rmt.mp_cdf(x, 0.4), rmt.mp_cdf(x, 0.4))


@pytest.mark.slow
def test_the_study_script_produces_identical_output_twice(tmp_path):
    """End-to-end: run the simulated study twice and compare every table.

    This is the test that actually protects the published numbers, because it
    exercises the same code path the reports were generated with rather than a
    reconstruction of it.  Marked slow; CI runs it, `make test-fast` skips it.
    """
    outputs = []
    for run in range(2):
        out_dir = tmp_path / f"run{run}"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "run_study.py"),
                "--simulated",
                "--quick",
                "--out",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(out_dir)

    tables = sorted(p.name for p in (outputs[0]).glob("*.csv"))
    assert tables, "the study produced no tables"
    for name in tables:
        first = pd.read_csv(outputs[0] / name)
        second = pd.read_csv(outputs[1] / name)
        pd.testing.assert_frame_equal(first, second)
