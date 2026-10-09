"""Portfolio construction, and what the constraints quietly do to the covariance.

This project optimises variance and all but ignores expected return, because
the two are not equally estimable.  The sample mean has standard error
sigma/sqrt(T); at sigma around 20% a year against a mean of maybe 6%, pinning
the mean to a percentage point needs some 400 years, and sampling the same
twenty years more finely does not help, since the drift grows with elapsed time
while its noise grows with the square root.  Quadratic variation is estimable at
high frequency, and a few hundred daily observations give a usable covariance.

So the global minimum-variance portfolio is the main object here.  Mean-variance
with a return target is implemented and reported too, but its out-of-sample
performance is dominated by error in the mean, which the study measures rather
than asserts.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights, w = Sigma^-1 1 / 1' Sigma^-1 1.

    Solved with `np.linalg.solve` rather than by forming the inverse, which is
    never needed on its own.  That is only housekeeping: the ill-conditioning
    this repository is about is statistical, not numerical.
    """
    cov = np.asarray(cov, dtype=float)
    ones = np.ones(cov.shape[0])
    z = np.linalg.solve(cov, ones)
    return z / (ones @ z)


def equal_weights(n_assets: int) -> np.ndarray:
    """The 1/N portfolio.

    DeMiguel, Garlappi and Uppal (2009) found none of fourteen optimising rules
    reliably beat it on out-of-sample Sharpe ratio over seven datasets.  It
    carries no estimation error at all, so it is the benchmark that matters.
    """
    return np.ones(n_assets) / n_assets


def mean_variance_weights(cov: np.ndarray, mu: np.ndarray, target: float) -> np.ndarray:
    """Minimum-variance weights subject to a target expected return, in closed form.

    Minimise w' Sigma w subject to 1'w = 1 and mu'w = target.  Two linear
    constraints, so the Lagrange conditions give a 2x2 system in
    a = 1'Sigma^-1 1, b = 1'Sigma^-1 mu and c = mu'Sigma^-1 mu, and the
    solution is a combination of Sigma^-1 1 and Sigma^-1 mu: two-fund
    separation.
    """
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ones = np.ones(cov.shape[0])

    inv_ones = np.linalg.solve(cov, ones)
    inv_mu = np.linalg.solve(cov, mu)
    a = ones @ inv_ones
    b = ones @ inv_mu
    c = mu @ inv_mu

    determinant = a * c - b * b
    if abs(determinant) < 1e-14 * max(abs(a * c), 1.0):
        # mu is proportional to 1: every portfolio has the same expected
        # return and the target is either unattainable or irrelevant.
        return min_variance_weights(cov)

    gamma = (c - b * target) / determinant
    delta = (a * target - b) / determinant
    return gamma * inv_ones + delta * inv_mu


def min_variance_long_only(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-variance weights with no short selling, plus the constraint multipliers.

    A quadratic program, since the inequality constraints leave no closed form.
    The multipliers are returned because they are the interesting part, see
    `jagannathan_ma_covariance`.

    CLARABEL, with the covariance symmetrised and marked PSD so cvxpy does not
    re-check it.  A numerically indefinite matrix is retried with a small ridge
    rather than failing: at q near 1 the sample covariance really does have tiny
    negative eigenvalues from round-off.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    n_assets = cov.shape[0]

    weights = cp.Variable(n_assets)
    nonneg = weights >= 0
    budget = cp.sum(weights) == 1
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(cov)))
    problem = cp.Problem(objective, [budget, nonneg])
    problem.solve(solver=cp.CLARABEL)

    if weights.value is None:
        ridge = 1e-10 * np.trace(cov) / n_assets
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(weights, cp.psd_wrap(cov + ridge * np.eye(n_assets)))),
            [budget, nonneg],
        )
        problem.solve(solver=cp.CLARABEL)
    if weights.value is None:
        raise RuntimeError(f"long-only minimum variance did not solve: {problem.status}")

    multipliers = np.asarray(nonneg.dual_value, dtype=float)
    return np.asarray(weights.value, dtype=float), multipliers


def mean_variance_long_only(cov: np.ndarray, mu: np.ndarray, target: float) -> np.ndarray:
    """Mean-variance weights with a return target and no short selling.

    A target above the largest expected return is infeasible long-only, since no
    convex combination can exceed the maximum.  It is clipped to what is
    attainable rather than raising, so one infeasible window does not stop a
    rolling backtest.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    mu = np.asarray(mu, dtype=float)
    n_assets = cov.shape[0]

    target = float(np.clip(target, mu.min(), mu.max()))

    weights = cp.Variable(n_assets)
    constraints = [cp.sum(weights) == 1, weights >= 0, mu @ weights >= target]
    problem = cp.Problem(cp.Minimize(cp.quad_form(weights, cp.psd_wrap(cov))), constraints)
    problem.solve(solver=cp.CLARABEL)

    if weights.value is None:
        raise RuntimeError(f"long-only mean-variance did not solve: {problem.status}")
    return np.asarray(weights.value, dtype=float)


def jagannathan_ma_covariance(cov: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """The covariance matrix the no-short constraint is implicitly using.

    Jagannathan and Ma (2003).  The derivation, since the result is the most
    counter-intuitive one here.

    The long-only problem is min w'Sigma w subject to 1'w = 1 and w >= 0.  Its
    Lagrangian is w'Sigma w + nu(1 - 1'w) - mu'w with mu >= 0, and stationarity
    gives 2 Sigma w = nu 1 + mu.  Now define

        Sigma~ = Sigma - (1 mu' + mu 1') / 2.

    Complementary slackness forces mu'w = 0, since a multiplier is nonzero only
    where the weight is pinned at zero, and 1'w = 1, so

        Sigma~ w = Sigma w - (1 (mu'w) + mu (1'w)) / 2 = Sigma w - mu/2 = nu 1 / 2,

    which makes Sigma~ w proportional to the vector of ones.  That is the
    first-order condition of the *unconstrained* minimum-variance problem for
    Sigma~.  The constrained solution for Sigma is therefore the unconstrained
    solution for Sigma~, and no constraint is needed to reach it.

    A multiplier mu_i is positive for the assets the optimiser wanted to short
    and could not, which are those with the largest estimated covariances
    against the rest.  In a few hundred observations an extreme covariance
    estimate is more likely extreme through estimation error than in truth, and
    subtracting (1 mu' + mu 1')/2 pulls those entries in.  So the constraint is
    a shrinkage estimator, which is how it improves out-of-sample variance while
    necessarily worsening the in-sample objective.

    `tests/test_optimise.py` asserts the identity numerically to 1e-8.
    """
    cov = np.asarray(cov, dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    ones = np.ones(cov.shape[0])
    correction = 0.5 * (np.outer(ones, multipliers) + np.outer(multipliers, ones))
    return cov - correction


def portfolio_variance(cov: np.ndarray, weights: np.ndarray) -> float:
    """w' Sigma w, the variance a covariance matrix attributes to a portfolio."""
    weights = np.asarray(weights, dtype=float)
    return float(weights @ np.asarray(cov, dtype=float) @ weights)


def condition_number(cov: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue.

    Reported against q: Marchenko-Pastur predicts it grows like
    ((1 + sqrt(q)) / (1 - sqrt(q)))^2 for pure noise, diverging as q -> 1, and
    the minimum-variance weights inherit that through the inverse.
    """
    eigenvalues = np.linalg.eigvalsh(0.5 * (np.asarray(cov) + np.asarray(cov).T))
    return float(eigenvalues[-1] / eigenvalues[0])


def turnover(new_weights: np.ndarray, old_weights: np.ndarray) -> float:
    """One-way turnover, sum |w_new - w_old|.

    `old_weights` must be the weights at the moment of rebalancing, after the
    holding period's returns have moved them, not those set at the previous
    rebalance.  The latter would charge the portfolio for drift it never traded.
    `backtest.drift_weights` does the adjustment.
    """
    return float(np.sum(np.abs(np.asarray(new_weights) - np.asarray(old_weights))))


def weight_concentration(weights: np.ndarray) -> float:
    """Effective number of positions, 1 / sum(w^2), as a fraction of N.

    The inverse Herfindahl index over N: 1 for the equally weighted portfolio,
    falling towards 1/N as it concentrates.  It collapses when the optimiser
    starts betting on noise directions, long before the realised variance makes
    the problem obvious.
    """
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights**2) / weights.size)


def gross_leverage(weights: np.ndarray) -> float:
    """Sum of absolute weights; 1 for a long-only portfolio, larger when shorting."""
    return float(np.sum(np.abs(np.asarray(weights))))
