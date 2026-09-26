"""Tests for portfolio construction.

Closed forms are checked against the numerical optimiser and against cases
where the answer can be written down by hand.  The Jagannathan-Ma identity is
checked as an identity, to 1e-8, because that is what it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from estimation_error import optimise
from estimation_error.estimators import sample_covariance
from estimation_error.simulate import factor_model_covariance, simulate_returns


def _sample(n_assets: int = 40, n_obs: int = 200, seed: int = 3) -> np.ndarray:
    truth = factor_model_covariance(n_assets, 3, np.random.default_rng(11))
    return sample_covariance(simulate_returns(truth, n_obs, np.random.default_rng(seed)))


def test_minimum_variance_of_the_identity_is_equal_weighting():
    """With no correlation and equal variances there is nothing to optimise."""
    weights = optimise.min_variance_weights(np.eye(25))
    assert np.allclose(weights, np.full(25, 1 / 25))


def test_minimum_variance_of_a_diagonal_matrix_is_inverse_variance_weighting():
    """Uncorrelated assets: w_i proportional to 1/sigma_i^2, which is checkable by hand."""
    variances = np.array([1.0, 4.0, 9.0, 16.0])
    weights = optimise.min_variance_weights(np.diag(variances))
    expected = (1 / variances) / np.sum(1 / variances)
    assert np.allclose(weights, expected)


def test_minimum_variance_weights_sum_to_one_and_solve_the_first_order_condition():
    """Sigma w must be proportional to the vector of ones; that is the whole condition."""
    cov = _sample()
    weights = optimise.min_variance_weights(cov)
    assert np.sum(weights) == pytest.approx(1.0)
    residual = cov @ weights
    assert np.allclose(residual, residual.mean(), rtol=1e-8)


def test_closed_form_minimum_variance_matches_the_numerical_optimiser():
    """The analytic solution and cvxpy must agree when the constraint is not binding.

    Finding a case where it is not binding takes a little care.  Any dispersion
    in the factor loadings makes the minimum-variance portfolio short the high-
    beta assets against the low-beta ones, because that is exactly the hedge it
    exists to construct.  So take identical loadings and heterogeneous
    idiosyncratic variances: Sigma = 11' + D.  By Sherman-Morrison,
    Sigma^-1 1 is proportional to D^-1 1, which is strictly positive, so the
    solution is interior and the two methods must agree to solver tolerance.
    """
    n_assets = 12
    idiosyncratic = np.linspace(0.5, 2.0, n_assets)
    cov = np.ones((n_assets, n_assets)) + np.diag(idiosyncratic)
    closed_form = optimise.min_variance_weights(cov)
    assert np.all(closed_form > 0)  # the constraint really is slack here
    assert np.allclose(closed_form, (1 / idiosyncratic) / np.sum(1 / idiosyncratic))

    numerical, _ = optimise.min_variance_long_only(cov)
    assert np.allclose(closed_form, numerical, atol=1e-7)


def test_minimum_variance_beats_every_other_portfolio_in_sample():
    """It is the minimiser, so random competitors must all do worse."""
    cov = _sample()
    best = optimise.portfolio_variance(cov, optimise.min_variance_weights(cov))
    rng = np.random.default_rng(0)
    for _ in range(50):
        challenger = rng.standard_normal(cov.shape[0])
        challenger = challenger / challenger.sum()
        assert optimise.portfolio_variance(cov, challenger) >= best - 1e-12


def test_mean_variance_hits_its_return_target():
    cov = _sample()
    mu = np.linspace(-0.01, 0.03, cov.shape[0])
    for target in (0.0, 0.01, 0.02):
        weights = optimise.mean_variance_weights(cov, mu, target)
        assert np.sum(weights) == pytest.approx(1.0)
        assert mu @ weights == pytest.approx(target)


def test_mean_variance_reduces_to_minimum_variance_at_its_own_return():
    """Targeting the minimum-variance portfolio's own return must return that portfolio.

    A consistency check on the two-fund algebra: the frontier passes through the
    minimum-variance point, so the two routes to it have to agree.
    """
    cov = _sample()
    mu = np.linspace(-0.01, 0.03, cov.shape[0])
    minimum = optimise.min_variance_weights(cov)
    weights = optimise.mean_variance_weights(cov, mu, float(mu @ minimum))
    assert np.allclose(weights, minimum, atol=1e-8)


def test_mean_variance_frontier_is_convex_in_the_target():
    """Variance is a quadratic in the target return, so it must be convex.

    Checked by second differences on an evenly spaced grid, which for an exact
    quadratic are constant and positive.
    """
    cov = _sample()
    mu = np.linspace(-0.01, 0.03, cov.shape[0])
    targets = np.linspace(0.0, 0.02, 9)
    variances = np.array(
        [
            optimise.portfolio_variance(cov, optimise.mean_variance_weights(cov, mu, t))
            for t in targets
        ]
    )
    second_difference = np.diff(variances, 2)
    assert np.all(second_difference > 0)
    assert np.std(second_difference) < 1e-8 * np.mean(second_difference) + 1e-16


def test_long_only_weights_are_feasible():
    cov = _sample()
    weights, multipliers = optimise.min_variance_long_only(cov)
    assert np.all(weights > -1e-8)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-7)
    assert np.all(multipliers > -1e-8)


def test_long_only_is_worse_in_sample_than_unconstrained():
    """It is a restriction of the feasible set, so it cannot do better in sample.

    Stated as a test because the whole interest of Jagannathan-Ma is that this
    necessary in-sample loss coexists with an out-of-sample gain.
    """
    cov = _sample()
    unconstrained = optimise.portfolio_variance(cov, optimise.min_variance_weights(cov))
    constrained, _ = optimise.min_variance_long_only(cov)
    assert optimise.portfolio_variance(cov, constrained) >= unconstrained - 1e-10


@pytest.mark.parametrize("n_assets,n_obs", [(30, 120), (49, 250), (60, 150)])
def test_jagannathan_ma_identity(n_assets, n_obs):
    """The long-only portfolio is the unconstrained portfolio of a shrunk covariance.

    Sigma~ = Sigma - (1 mu' + mu 1')/2, with mu the multipliers on the
    no-short constraints.  This is Jagannathan and Ma's theorem and it holds
    exactly, so the tolerance is numerical, not statistical.
    """
    truth = factor_model_covariance(n_assets, 3, np.random.default_rng(11))
    cov = sample_covariance(simulate_returns(truth, n_obs, np.random.default_rng(3)))

    constrained, multipliers = optimise.min_variance_long_only(cov)
    implied = optimise.jagannathan_ma_covariance(cov, multipliers)
    assert np.allclose(constrained, optimise.min_variance_weights(implied), atol=1e-8)


def test_jagannathan_ma_multipliers_vanish_where_weights_are_positive():
    """Complementary slackness, which the identity's derivation depends on."""
    cov = _sample()
    weights, multipliers = optimise.min_variance_long_only(cov)
    assert np.max(multipliers * weights) < 1e-8


def test_jagannathan_ma_correction_shrinks_the_constrained_assets():
    """The implied matrix must reduce covariances, not raise them.

    That is the sense in which the constraint acts as shrinkage: it pulls in the
    large estimated covariances belonging to the assets the optimiser wanted to
    short, which are the estimates most likely to be extreme by luck.
    """
    cov = _sample()
    _, multipliers = optimise.min_variance_long_only(cov)
    implied = optimise.jagannathan_ma_covariance(cov, multipliers)
    binding = multipliers > 1e-9
    assert binding.sum() > 0
    assert np.all(implied[binding][:, binding] <= cov[binding][:, binding] + 1e-12)


def test_condition_number_of_the_identity_is_one():
    assert optimise.condition_number(np.eye(10)) == pytest.approx(1.0)


def test_condition_number_grows_with_q():
    """More assets per observation means a worse conditioned estimate."""
    truth = factor_model_covariance(60, 3, np.random.default_rng(11))
    numbers = [
        optimise.condition_number(
            sample_covariance(simulate_returns(truth, T, np.random.default_rng(2)))
        )
        for T in (2000, 500, 120)
    ]
    assert numbers[0] < numbers[1] < numbers[2]


def test_weight_concentration_is_one_for_equal_weights():
    assert optimise.weight_concentration(optimise.equal_weights(37)) == pytest.approx(1.0)


def test_weight_concentration_falls_when_a_portfolio_concentrates():
    concentrated = np.zeros(20)
    concentrated[0] = 1.0
    assert optimise.weight_concentration(concentrated) == pytest.approx(1 / 20)


def test_gross_leverage_is_one_exactly_when_long_only():
    cov = _sample()
    constrained, _ = optimise.min_variance_long_only(cov)
    assert optimise.gross_leverage(constrained) == pytest.approx(1.0, abs=1e-6)
    assert optimise.gross_leverage(optimise.min_variance_weights(cov)) > 1.0


def test_turnover_is_zero_for_an_unchanged_portfolio():
    weights = optimise.equal_weights(10)
    assert optimise.turnover(weights, weights) == 0.0


def test_turnover_of_a_complete_switch_is_two():
    """Selling one position entirely and buying another moves 100% out and 100% in."""
    old = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    assert optimise.turnover(new, old) == pytest.approx(2.0)
