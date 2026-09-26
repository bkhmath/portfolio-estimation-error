"""Synthetic return generators with a covariance matrix we chose ourselves.

Every claim in this project is about the distance between an estimate and the
truth.  On real returns the truth is not available, so all that can be measured
there are proxies: realised out-of-sample variance, turnover, the gap between
what the optimiser predicted and what happened.  Those are the quantities
anyone cares about, but they cannot establish that the code is correct, because
a bug and a genuinely hard estimation problem look the same in an out-of-sample
number.

The simulated layer exists to make the truth available.  Sigma is something we
wrote down, so the Frobenius and spectral-norm error of every estimator can be
computed exactly, the true minimum-variance portfolio is known in closed form,
and a claim like "cleaning reduces the error at finite T and all estimators
converge as T grows" becomes testable instead of rhetorical.

The covariance here is a factor structure, Sigma = B B^T + D.  That is the
shape equity covariance actually has, and it puts a few eigenvalues above the
Marchenko-Pastur edge with the rest inside, which is the configuration the
project is about.  The first factor gets all-positive loadings much larger than
the others: the market mode, the direction everything moves together in, and on
real data always the largest eigenvalue by a wide margin.
"""

from __future__ import annotations

import numpy as np


def factor_model_covariance(
    n_assets: int,
    n_factors: int = 3,
    rng: np.random.Generator | None = None,
    market_loading: tuple[float, float] = (0.7, 1.3),
    factor_loading_scale: float = 0.35,
    idiosyncratic_range: tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    """Build a covariance matrix Sigma = B B^T + D with a dominant market factor.

    The first column of B is drawn positive, so the first factor moves every
    asset the same way and produces one very large eigenvalue.  The remaining
    columns are sign-indefinite and smaller, producing a few eigenvalues that
    sit above the noise bulk but not dramatically so.  Those are the hard and
    interesting ones, since they are what an estimator can plausibly get wrong.
    D is diagonal with heterogeneous entries, because assets do not have equal
    idiosyncratic volatility and an estimator that quietly assumes they do
    should be penalised for it.

    The result is scaled so that the average variance is 1.  That is purely a
    convenience: it makes the Marchenko-Pastur bulk of the corresponding noise
    sit around 1 without a separate rescaling, so plots are comparable across
    parameter choices.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_factors < 1:
        raise ValueError("need at least one factor")

    loadings = np.zeros((n_assets, n_factors))
    loadings[:, 0] = rng.uniform(*market_loading, size=n_assets)
    if n_factors > 1:
        loadings[:, 1:] = factor_loading_scale * rng.standard_normal((n_assets, n_factors - 1))

    idiosyncratic = rng.uniform(*idiosyncratic_range, size=n_assets)
    cov = loadings @ loadings.T + np.diag(idiosyncratic)

    # Normalise to unit average variance.
    cov = cov / (np.trace(cov) / n_assets)
    # Force exact symmetry: the products above are symmetric in exact
    # arithmetic but not bitwise, and eigh is happier if we say so.
    return 0.5 * (cov + cov.T)


def simulate_returns(cov: np.ndarray, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_obs` observations from a centred Gaussian with covariance `cov`.

    Returns a (n_obs, n_assets) array, the orientation used everywhere in this
    repository: rows are time, columns are assets.

    Gaussian rather than heavy-tailed on purpose.  Real returns have fat tails,
    and fat tails make covariance estimation harder still, but the point of the
    simulated layer is to isolate the effect of the aspect ratio q = N/T.
    Adding tail risk here would mix two causes of the same symptom and make the
    Marchenko-Pastur comparison invalid, since the law as implemented assumes
    finite fourth moments and a variance profile the Gaussian satisfies
    exactly.
    """
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((n_obs, cov.shape[0])) @ chol.T


def simulate_white_noise(n_assets: int, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Draw iid standard normal returns, so the true covariance is the identity.

    This is the configuration Marchenko-Pastur describes exactly, and it is
    what the spectral validation tests use.  Kept as its own function rather
    than a call to `simulate_returns` with an identity argument so that the
    tests read as what they are.
    """
    return rng.standard_normal((n_obs, n_assets))
