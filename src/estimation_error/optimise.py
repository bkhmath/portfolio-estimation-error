"""Portfolio construction, and what the constraints quietly do to the covariance.

This project optimises variance and all but ignores expected return.  That is a
deliberate restriction rather than a shortcut, so here is the reason before any
result appears.

The two quantities are not equally estimable.  The sample mean of a return
series has standard error sigma/sqrt(T); for a diversified equity portfolio
sigma is around 20% a year against a mean of maybe 6%, so pinning the mean to
within a percentage point needs sigma^2/1%^2 = 400 years of data.  Only a
longer calendar sample helps.  Sampling the same twenty years more finely gives
more observations but no more information about the drift, because the drift
term in the sum grows with elapsed time while its noise grows with the square
root.  This is the standard observation that the mean of a diffusion is not
estimable from high-frequency observation of a fixed interval.

Covariance behaves differently.  Quadratic variation is estimable at high
frequency, and doubling the sampling rate really does double the information
about sigma.  A few hundred daily observations give a usable covariance matrix.
So the minimum-variance portfolio is the part of the mean-variance problem
where better estimation can win, and the tangency portfolio is not.

The global minimum-variance portfolio is therefore the main object here.
Mean-variance with a return target is implemented and reported too, since the
project would be incomplete without it, but its out-of-sample performance is
dominated by error in the mean and says little about the covariance.  That is
itself a result, and the study measures it rather than asserting it.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights, w = Sigma^-1 1 / 1' Sigma^-1 1.

    Solved with `np.linalg.solve` rather than by forming the inverse: the
    inverse is never needed on its own, and computing it costs accuracy for
    nothing.  That is housekeeping, not the point.  The ill-conditioning this
    repository is about is statistical, and no amount of care in the linear
    algebra touches it.
    """
    cov = np.asarray(cov, dtype=float)
    ones = np.ones(cov.shape[0])
    z = np.linalg.solve(cov, ones)
    return z / (ones @ z)


def equal_weights(n_assets: int) -> np.ndarray:
    """The 1/N portfolio.

    DeMiguel, Garlappi and Uppal (2009) tested fourteen optimising rules over
    seven datasets and found none of them reliably beat this one on Sharpe
    ratio out of sample: the estimation error in the optimised rules costs
    more than the optimisation gains.  It is the benchmark that matters, and
    by construction it carries no estimation error at all.
    """
    return np.ones(n_assets) / n_assets


def mean_variance_weights(cov: np.ndarray, mu: np.ndarray, target: float) -> np.ndarray:
    """Minimum-variance weights subject to a target expected return, in closed form.

    Minimise w' Sigma w subject to 1'w = 1 and mu'w = target.  Two linear
    constraints, so the Lagrange conditions give a 2x2 system in the
    multipliers built from the three scalars a = 1'Sigma^-1 1,
    b = 1'Sigma^-1 mu and c = mu'Sigma^-1 mu, and the solution is a linear
    combination of Sigma^-1 1 and Sigma^-1 mu.  This is the two-fund separation
    theorem: every frontier portfolio is a mix of the minimum-variance
    portfolio and one other.
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

    Solved as a quadratic program, since there is no closed form once the
    inequality constraints are present.  The multipliers are returned because
    they are the interesting part; see `jagannathan_ma_covariance`.

    The solver is CLARABEL, an interior-point method, and the covariance is
    symmetrised and marked positive semidefinite so that cvxpy does not spend
    time checking.  If the matrix is numerically indefinite the solve is
    retried on a slightly ridged version rather than failing, because at q near
    1 the sample covariance genuinely does have tiny negative eigenvalues from
    round-off.
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

    If the target exceeds the largest expected return the problem is infeasible
    under the long-only constraint, since no convex combination can exceed the
    maximum; in that case the target is clipped to what is attainable rather
    than raising, because in a rolling backtest an occasional infeasible window
    should not stop the study.
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

    This is the content of Jagannathan and Ma (2003), and it is the most
    counter-intuitive result in the project, so here is the derivation rather
    than a citation.

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

    So what does Sigma~ look like?  A multiplier mu_i is positive for the
    assets the optimiser wanted to short and could not, which are the ones with
    the largest estimated covariances against the rest.  In a sample of a few
    hundred observations an extreme covariance estimate is more likely to be
    extreme through estimation error than because the truth is extreme, and
    subtracting (1 mu' + mu 1')/2 pulls those entries in.  The constraint is a
    shrinkage estimator wearing a different hat.  That is how it can improve
    out-of-sample variance while, being a restriction of the feasible set,
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

    Reported against q because it is the cleanest single summary of what goes
    wrong: Marchenko-Pastur predicts it grows like
    ((1 + sqrt(q)) / (1 - sqrt(q)))^2 for pure noise, which diverges as q -> 1,
    and the minimum-variance weights inherit that divergence through the
    inverse.
    """
    eigenvalues = np.linalg.eigvalsh(0.5 * (np.asarray(cov) + np.asarray(cov).T))
    return float(eigenvalues[-1] / eigenvalues[0])


def turnover(new_weights: np.ndarray, old_weights: np.ndarray) -> float:
    """One-way turnover, sum |w_new - w_old|.

    `old_weights` must be the weights as they stand at the moment of
    rebalancing, that is after the holding period's returns have moved them,
    not the weights that were set at the previous rebalance.  Using the latter
    would charge the portfolio for drift it never traded, and would flatter
    high-turnover strategies by comparison.  `backtest.drift_weights` does the
    adjustment.
    """
    return float(np.sum(np.abs(np.asarray(new_weights) - np.asarray(old_weights))))


def weight_concentration(weights: np.ndarray) -> float:
    """Effective number of positions, 1 / sum(w^2), as a fraction of N.

    The inverse Herfindahl index divided by N.  Equals 1 for the equally
    weighted portfolio and falls towards 1/N as the portfolio concentrates.
    This is the quantity that collapses when the optimiser starts betting on
    noise directions, and it does so long before the realised variance makes
    the problem obvious.
    """
    weights = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(weights**2) / weights.size)


def gross_leverage(weights: np.ndarray) -> float:
    """Sum of absolute weights; 1 for a long-only portfolio, larger when shorting."""
    return float(np.sum(np.abs(np.asarray(weights))))
