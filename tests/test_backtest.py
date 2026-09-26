"""Tests for the out-of-sample protocol and the exact finite-sample formulas.

The two formulas in `backtest` are the analytical core of the project's main
claim, so they are checked against Monte Carlo rather than against each other.
The protocol itself is checked for the one bug that would invalidate every
number in the repository: look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from estimation_error import backtest
from estimation_error.estimators import sample_covariance
from estimation_error.optimise import min_variance_weights, portfolio_variance
from estimation_error.simulate import factor_model_covariance, simulate_returns


@pytest.mark.parametrize("n_assets,n_obs", [(10, 30), (10, 50), (20, 60)])
def test_predicted_variance_formula_matches_monte_carlo(n_assets, n_obs):
    """E[predicted]/true = (T - N)/(T - 1), exactly, for Gaussian returns.

    Simulated with Sigma = I so the true minimum variance is 1/N in closed form
    and no estimate of the truth enters the comparison.
    """
    rng = np.random.default_rng(1)
    draws = 4000
    predicted = np.empty(draws)
    for i in range(draws):
        returns = rng.standard_normal((n_obs, n_assets))
        cov = sample_covariance(returns)
        predicted[i] = portfolio_variance(cov, min_variance_weights(cov))

    observed = predicted.mean() * n_assets  # divide by the true minimum variance 1/N
    expected = backtest.predicted_variance_ratio(n_assets, n_obs)
    assert observed == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("n_assets,n_obs", [(10, 30), (10, 50), (20, 60)])
def test_realised_variance_formula_matches_monte_carlo(n_assets, n_obs):
    """E[realised]/true = (T - 2)/(T - N - 1), exactly, for Gaussian returns.

    With Sigma = I the realised variance of the estimated portfolio is just the
    squared norm of its weights, so this compares a closed form against a
    quantity with no estimation in it at all.
    """
    rng = np.random.default_rng(2)
    draws = 20000
    realised = np.empty(draws)
    for i in range(draws):
        returns = rng.standard_normal((n_obs, n_assets))
        weights = min_variance_weights(sample_covariance(returns))
        realised[i] = weights @ weights

    observed = realised.mean() * n_assets
    expected = backtest.realised_variance_ratio(n_assets, n_obs)
    assert observed == pytest.approx(expected, rel=0.02)


def test_self_deception_ratio_tends_to_one_over_one_minus_q_squared():
    """The asymptotic form the finite-sample formulas must reproduce for large T."""
    for q in (0.1, 0.25, 0.4):
        n_obs = 20000
        n_assets = round(q * n_obs)
        ratio = backtest.self_deception_ratio(n_assets, n_obs)
        assert ratio == pytest.approx(1.0 / (1.0 - q) ** 2, rel=1e-3)


def test_predicted_ratio_falls_and_realised_ratio_rises_with_q():
    """The two errors point in opposite directions, which is why the gap compounds."""
    predicted = [backtest.predicted_variance_ratio(50, T) for T in (2000, 500, 200)]
    realised = [backtest.realised_variance_ratio(50, T) for T in (2000, 500, 200)]
    assert predicted[0] > predicted[1] > predicted[2]
    assert realised[0] < realised[1] < realised[2]


def test_realised_ratio_diverges_when_the_matrix_becomes_singular():
    assert np.isinf(backtest.realised_variance_ratio(100, 101))


def test_drift_weights_leaves_a_flat_period_alone():
    """No returns, no drift."""
    weights = np.array([0.3, 0.3, 0.4])
    drifted = backtest.drift_weights(weights, np.zeros((5, 3)))
    assert np.allclose(drifted, weights)


def test_drift_weights_tracks_compounded_growth():
    """A position that doubles while the other is flat must end up with twice the share.

    Worked by hand: starting from equal weights, one asset doubling gives values
    (1, 2)/3 of the original book, so the weights become (1/3, 2/3).
    """
    weights = np.array([0.5, 0.5])
    returns = np.array([[0.0, 1.0]])
    assert np.allclose(backtest.drift_weights(weights, returns), [1 / 3, 2 / 3])


def test_drift_weights_stays_normalised():
    rng = np.random.default_rng(5)
    weights = rng.dirichlet(np.ones(8))
    drifted = backtest.drift_weights(weights, rng.normal(0, 0.01, size=(21, 8)))
    assert np.sum(drifted) == pytest.approx(1.0)


def _panel(n_obs: int = 700, n_assets: int = 20, seed: int = 4) -> pd.DataFrame:
    truth = factor_model_covariance(n_assets, 2, np.random.default_rng(1))
    values = simulate_returns(truth, n_obs, np.random.default_rng(seed)) * 0.01
    index = pd.bdate_range("2000-01-03", periods=n_obs)
    return pd.DataFrame(values, index=index)


def test_backtest_does_not_look_ahead():
    """Changing data after a rebalance must not change the weights chosen before it.

    This is the one bug that would silently invalidate every out-of-sample
    number in the repository, so it gets a direct test: corrupt the tail of the
    panel and check that the predicted variances of every earlier window are
    bit-for-bit unchanged.
    """
    panel = _panel()
    window, step = 250, 21
    strategies = ["sample", "linear"]

    original = backtest.run_backtest(panel, strategies, window=window, step=step)

    corrupted = panel.copy()
    cut = window + step * 3
    corrupted.iloc[cut:] = corrupted.iloc[cut:] * 50.0
    after = backtest.run_backtest(corrupted, strategies, window=window, step=step)

    # Rebalances whose estimation window ends at or before `cut` saw no corrupted data.
    unaffected = (cut - window) // step + 1
    for name in strategies:
        assert np.array_equal(
            original[name].predicted_variances[:unaffected],
            after[name].predicted_variances[:unaffected],
        )


def test_backtest_holds_the_portfolio_for_the_whole_step():
    """Every rebalance must contribute exactly `step` daily returns, and no day twice."""
    panel = _panel()
    window, step = 250, 21
    result = backtest.run_backtest(panel, ["sample"], window=window, step=step)["sample"]
    n_rebalances = result.predicted_variances.size
    assert result.daily_returns.size == n_rebalances * step
    expected = (len(panel) - window - step) // step + 1
    assert n_rebalances == expected


def test_equal_weight_strategy_reproduces_the_panel_average():
    """A sanity check with a known answer: 1/N's return is the cross-sectional mean."""
    panel = _panel()
    window, step = 250, 21
    result = backtest.run_backtest(panel, ["equal"], window=window, step=step)["equal"]
    expected = panel.to_numpy()[window : window + result.daily_returns.size].mean(axis=1)
    assert np.allclose(result.daily_returns, expected)


def test_equal_weight_has_no_turnover_beyond_drift():
    """1/N trades only to undo drift, so its turnover must be far below an optimiser's."""
    panel = _panel()
    results = backtest.run_backtest(panel, ["equal", "sample"], window=250, step=21)
    assert (
        results["equal"].summary()["mean_turnover"]
        < 0.1 * results["sample"].summary()["mean_turnover"]
    )


def test_optimised_portfolio_realises_more_variance_than_it_predicted():
    """The central empirical claim, on simulated data where it must hold."""
    panel = _panel(n_obs=3000, n_assets=60)
    result = backtest.run_backtest(panel, ["sample"], window=250, step=21)["sample"]
    assert result.summary()["variance_ratio"] > 1.2


def test_cleaning_reduces_realised_variance_out_of_sample():
    """At a realistic q, cleaned estimators must beat the sample covariance."""
    panel = _panel(n_obs=3000, n_assets=60)
    results = backtest.run_backtest(panel, ["sample", "linear", "nonlinear"], window=250, step=21)
    baseline = results["sample"].summary()["annual_volatility"]
    assert results["linear"].summary()["annual_volatility"] < baseline
    assert results["nonlinear"].summary()["annual_volatility"] < baseline


def test_backtest_rejects_a_panel_that_is_too_short():
    with pytest.raises(ValueError):
        backtest.run_backtest(_panel(n_obs=100), ["sample"], window=250, step=21)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        backtest.run_backtest(_panel(), ["not_an_estimator"], window=250, step=21)


def test_protocol_constants_are_the_documented_ones():
    """The protocol was fixed before results; this test is what makes that checkable."""
    assert backtest.ESTIMATION_WINDOW == 250
    assert backtest.REBALANCE_EVERY == 21


def test_mean_variance_strategy_hits_its_target_in_sample():
    """The "+mv" variant must actually solve the constrained problem it claims to.

    Checked directly on one window rather than through the backtest, so a failure
    points at the weights and not at the protocol around them.
    """
    panel = _panel()
    window = panel.to_numpy()[:250]
    weights, cov = backtest._weights_for("sample+mv", window)
    mu = window.mean(axis=0)
    assert np.sum(weights) == pytest.approx(1.0)
    assert mu @ weights == pytest.approx(float(mu.mean()))
    assert cov.shape == (panel.shape[1], panel.shape[1])


def test_mean_variance_costs_variance_relative_to_minimum_variance():
    """Constraining the return can only raise in-sample variance, and does.

    Out of sample it also raises realised variance, because the target is built
    from estimated means; that is reported in the study rather than asserted
    here, since it is an empirical claim rather than an algebraic one.
    """
    panel = _panel()
    window = panel.to_numpy()[:250]
    minimum, cov = backtest._weights_for("sample", window)
    targeted, _ = backtest._weights_for("sample+mv", window)
    assert portfolio_variance(cov, targeted) >= portfolio_variance(cov, minimum) - 1e-12


def test_mean_variance_long_only_stays_feasible():
    panel = _panel()
    window = panel.to_numpy()[:250]
    weights, _ = backtest._weights_for("sample+mv_long_only", window)
    assert np.all(weights > -1e-7)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)
