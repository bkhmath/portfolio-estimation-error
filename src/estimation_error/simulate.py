"""Synthetic return generators with a covariance matrix we chose ourselves.

On real returns the truth is not available, so only proxies can be measured
there, and a bug and a genuinely hard estimation problem look the same in an
out-of-sample number.  With Sigma written down, the Frobenius and spectral-norm
error of every estimator is exact and the true minimum-variance portfolio is
known in closed form.

The covariance is a factor structure, Sigma = B B^T + D, which is the shape
equity covariance has and which puts a few eigenvalues above the
Marchenko-Pastur edge with the rest inside.  The first factor gets all-positive
loadings much larger than the others: the market mode.
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

    The first column of B is positive, so the first factor moves every asset
    the same way and gives one very large eigenvalue.  The rest are
    sign-indefinite and smaller, giving a few eigenvalues just above the noise
    bulk, which are the ones an estimator can plausibly get wrong.  D is
    diagonal with heterogeneous entries, since assets do not have equal
    idiosyncratic volatility.

    Scaled to unit average variance, so the Marchenko-Pastur bulk sits around 1
    and plots are comparable across parameter choices.
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

    Returns (n_obs, n_assets): rows are time, columns are assets, as
    everywhere in this repository.

    Gaussian rather than heavy-tailed, to isolate the effect of the aspect
    ratio q = N/T.  Fat tails would mix two causes of the same symptom, and the
    Marchenko-Pastur law as implemented assumes finite fourth moments.
    """
    chol = np.linalg.cholesky(cov)
    return rng.standard_normal((n_obs, cov.shape[0])) @ chol.T


def simulate_white_noise(n_assets: int, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Draw iid standard normal returns, so the true covariance is the identity.

    The configuration Marchenko-Pastur describes exactly, used by the spectral
    validation tests.
    """
    return rng.standard_normal((n_obs, n_assets))
