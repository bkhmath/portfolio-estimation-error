"""Covariance estimators: the broken baseline and the four repairs.

All five answer the same question, what matrix to hand the optimiser given T
observations of N assets, and differ in what they assume about where the
estimation error lives.

`sample` assumes nothing and is unbiased entrywise, which says nothing about
the eigenvalues, and the eigenvalues are what inversion touches.  It is the
baseline the others have to beat.

`clipped` assumes eigenvalues inside the Marchenko-Pastur bulk cannot be told
apart and replaces them by their common average, keeping the eigenvectors.

`linear_shrinkage` asks instead which convex combination of the sample
covariance and a scaled identity minimises expected squared error.  Ledoit and
Wolf showed the optimal weight can itself be estimated from the data, so there
is no tuning parameter -- an intensity hand-picked after seeing out-of-sample
results would be a fitted parameter reported as a free one.

`nonlinear_shrinkage` gives every eigenvalue its own correction, derived from
the same Stieltjes transform behind Marchenko-Pastur.  Clipping is a crude
two-level version of the smooth shrinkage function it computes.

`factor` assumes the truth really is low rank plus diagonal: a much stronger
assumption, and the one practitioners actually make.

Orientation and normalisation for all of them: returns come in as
(n_obs, n_assets), columns are demeaned, and the divisor is T - 1.  Demeaning
costs a degree of freedom, so the effective aspect ratio is q = N/(T-1), which
is the q used wherever the bulk edge is needed.
"""

from __future__ import annotations

import numpy as np

from .rmt import mp_edges, noise_variance


def _demeaned(returns: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Centre the columns and report the dimensions, since every estimator needs this."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError(f"expected a 2-d (n_obs, n_assets) array, got shape {returns.shape}")
    n_obs, n_assets = returns.shape
    if n_obs < 2:
        raise ValueError("need at least two observations")
    return returns - returns.mean(axis=0), n_obs, n_assets


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """The maximum-likelihood sample covariance, X^T X / (T - 1).

    The baseline.  When N is comparable to T it is nearly singular and its
    inverse is dominated by the smallest and worst-estimated eigenvalues; when
    N > T it is singular and the minimum-variance problem has no solution.
    """
    centred, n_obs, _ = _demeaned(returns)
    cov = centred.T @ centred / (n_obs - 1)
    return 0.5 * (cov + cov.T)


def clipped_covariance(returns: np.ndarray, n_keep: int | None = None) -> np.ndarray:
    """Eigenvalue clipping in the sense of Laloux, Cizeau, Bouchaud and Potters.

    On the correlation matrix, not the covariance: Marchenko-Pastur is about
    series with a common variance and real assets have none.  Diagonalise, fit
    sigma^2 to the bulk, replace every eigenvalue below the upper edge
    sigma^2 (1 + sqrt(q))^2 by their common mean, then put the individual
    volatilities back.  Replacing by the mean and not by sigma^2 preserves the
    trace, which for a correlation matrix is N, so the result stays one.

    The eigenvectors are untouched: the claim is that the ordering of
    eigenvalues within the bulk is noise, not that the subspace is.

    `n_keep` overrides the threshold with a fixed count, for the tests.  The
    study never passes it.
    """
    _, n_obs, n_assets = _demeaned(returns)
    cov = sample_covariance(returns)

    volatility = np.sqrt(np.diag(cov))
    if np.any(volatility <= 0):
        raise ValueError("an asset has zero sample variance; drop it before estimating")
    correlation = cov / np.outer(volatility, volatility)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    q = n_assets / (n_obs - 1)

    if n_keep is None:
        sigma2 = noise_variance(eigenvalues, q)
        _, upper_edge = mp_edges(q, sigma2)
        is_noise = eigenvalues < upper_edge
    else:
        # eigh returns ascending order, so the kept ones are at the end.
        is_noise = np.ones(n_assets, dtype=bool)
        if n_keep > 0:
            is_noise[-n_keep:] = False

    if is_noise.sum() > 0:
        eigenvalues = eigenvalues.copy()
        eigenvalues[is_noise] = eigenvalues[is_noise].mean()

    cleaned = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Clipping preserves the total trace but not each diagonal entry, so
    # renormalise to put exact ones back on the diagonal.  Without this the
    # result is not a correlation matrix and the volatilities get double
    # counted.
    scale = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(scale, scale)

    out = cleaned * np.outer(volatility, volatility)
    return 0.5 * (out + out.T)


def linear_shrinkage(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf linear shrinkage towards a scaled identity.

    (1 - d) S + d m I with m = trace(S)/N.  The squared error is quadratic in
    the intensity, so the minimiser is a ratio of the dispersion of S about its
    own scaled-identity projection and the variance of S's entries; neither is
    observable but both have consistent estimators from the same sample.

    Notation follows the 2004 paper: m the average eigenvalue, d2 the squared
    distance from S to m I, b2 the capped average squared deviation of the
    single-observation outer products from S, and the intensity is b2/d2.
    Norms are Frobenius normalised by N.  The cap b2 = min(b2, d2) keeps the
    intensity in [0, 1] and binds when T is very small relative to N.

    One convention detail that silently makes an implementation only nearly
    right: Ledoit and Wolf use the 1/T normalisation, not 1/(T - 1), and the
    intensity is not invariant to the choice, because b2 compares the
    single-observation outer products against S and only one of the two
    scales.  So the derivation below is in their convention and the answer is
    rescaled by T/(T - 1) at the end.  `tests/test_estimators.py` checks the
    result against an independent implementation.
    """
    shrunk, _ = _linear_shrinkage_with_intensity(returns)
    return shrunk


def _linear_shrinkage_with_intensity(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shared body of `linear_shrinkage` and `linear_shrinkage_intensity`."""
    centred, n_obs, n_assets = _demeaned(returns)
    # Ledoit-Wolf's own normalisation; see the note in `linear_shrinkage`.
    cov = centred.T @ centred / n_obs

    m = np.trace(cov) / n_assets
    d2 = np.sum((cov - m * np.eye(n_assets)) ** 2) / n_assets

    # b2 is the average over observations of || x_t x_t^T - S ||^2, which
    # measures how far a single observation's outer product sits from the
    # estimate; it is what the sampling variance of the entries of S is made
    # of.  The loop is O(T N^2) and there is a closed form that avoids it, but
    # at the sizes here it costs milliseconds and this version is checkable by
    # eye against the paper.
    b2 = 0.0
    for t in range(n_obs):
        outer = np.outer(centred[t], centred[t])
        b2 += np.sum((outer - cov) ** 2) / n_assets
    b2 = b2 / n_obs**2
    b2 = min(b2, d2)

    intensity = b2 / d2 if d2 > 0 else 0.0
    shrunk = (1.0 - intensity) * cov + intensity * m * np.eye(n_assets)
    # Back to the unbiased normalisation used by every other estimator here.
    shrunk = shrunk * n_obs / (n_obs - 1)
    return 0.5 * (shrunk + shrunk.T), float(intensity)


def linear_shrinkage_intensity(returns: np.ndarray) -> float:
    """The intensity `linear_shrinkage` chose, for reporting.

    Zero means the sample covariance was already the best convex combination;
    one means the estimator gave up on it entirely and returned the target.
    """
    _, intensity = _linear_shrinkage_with_intensity(returns)
    return intensity


def nonlinear_shrinkage(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf analytical nonlinear shrinkage (2020).

    The bias in the sample spectrum is not uniform -- it pushes the top
    eigenvalues up and the bottom ones down -- so one shared intensity is too
    blunt.  The oracle correction depends on the truth only through the
    Stieltjes transform of the limiting sample spectral distribution:

        d_i = lambda_i / | 1 - q - q lambda_i m(lambda_i) |^2,

    on the real axis.  Same m as behind Marchenko-Pastur, but the population
    spectrum is not the identity here so there is no closed form.  Ledoit and
    Wolf estimate m from the sample eigenvalues by kernel smoothing: its
    imaginary part on the real axis is pi times the density, taken here with an
    Epanechnikov kernel, and its real part is the Hilbert transform, which for
    that kernel is the closed form in `hilbert_term`.  The bandwidth n^(-1/3)
    is theirs and is not tuned.

    Only implemented for N < T, the regime of this study.
    """
    _, n_obs, n_assets = _demeaned(returns)
    # The demeaning costs one observation, so the effective sample size is
    # T - 1 and that is what sets the aspect ratio.
    n_eff = n_obs - 1
    if n_assets > n_eff:
        raise NotImplementedError(
            f"analytical nonlinear shrinkage here assumes N < T-1; got N={n_assets}, T-1={n_eff}"
        )
    q = n_assets / n_eff

    cov = sample_covariance(returns)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if np.any(eigenvalues <= 0):
        raise ValueError("sample covariance is singular; nonlinear shrinkage needs N < T-1")

    bandwidth = n_eff ** (-1.0 / 3.0)
    # x[i, j] = (lambda_i - lambda_j) / (h lambda_j): the distance from
    # eigenvalue i to eigenvalue j measured in units of a local bandwidth that
    # scales with lambda_j.  Scaling the bandwidth with the eigenvalue is what
    # lets one kernel handle a spectrum spanning several orders of magnitude.
    scaled_bandwidth = bandwidth * eigenvalues[None, :]
    x = (eigenvalues[:, None] - eigenvalues[None, :]) / scaled_bandwidth

    # Epanechnikov density estimate: (3 / 4 sqrt(5)) (1 - x^2/5) on |x| < sqrt(5).
    density = np.mean(np.maximum(1.0 - x**2 / 5.0, 0.0) / scaled_bandwidth, axis=1) * (
        3.0 / (4.0 * np.sqrt(5.0))
    )

    # Hilbert transform of the same kernel, in closed form.  The logarithm
    # blows up at |x| = sqrt(5); the limit there is the linear term alone.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log(np.abs((np.sqrt(5.0) - x) / (np.sqrt(5.0) + x)))
    hilbert_term = (-3.0 / (10.0 * np.pi)) * x + (3.0 / (4.0 * np.sqrt(5.0) * np.pi)) * (
        1.0 - x**2 / 5.0
    ) * log_term
    at_singularity = np.abs(np.abs(x) - np.sqrt(5.0)) < 1e-12
    hilbert_term[at_singularity] = (-3.0 / (10.0 * np.pi)) * x[at_singularity]
    hilbert = np.mean(hilbert_term / scaled_bandwidth, axis=1)

    shrunk_eigenvalues = eigenvalues / (
        (np.pi * q * eigenvalues * density) ** 2
        + (1.0 - q - np.pi * q * eigenvalues * hilbert) ** 2
    )

    out = eigenvectors @ np.diag(shrunk_eigenvalues) @ eigenvectors.T
    return 0.5 * (out + out.T)


def factor_covariance(returns: np.ndarray, n_factors: int | None = None) -> np.ndarray:
    """Principal-component factor model: low rank plus diagonal.

    Keep the k leading eigenpairs of the sample covariance as the systematic
    part and put the remainder on the diagonal, so each asset's total variance
    is exact while its covariances are described by k factors.  The POET
    construction of Fan, Liao and Mincheva, statistical rather than
    observed-factor: the factors are the directions the data says carry common
    variation.

    Statistical on purpose.  A model on the published Fama-French factors could
    only be judged out of sample, since the simulated layer contains no such
    factors, while the PCA version can be checked against a factor structure we
    built ourselves.

    With `n_factors` unset, k is the number of eigenvalues above the fitted
    Marchenko-Pastur edge, with a floor of one since the market mode is always
    present.
    """
    _, n_obs, n_assets = _demeaned(returns)
    cov = sample_covariance(returns)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    if n_factors is None:
        volatility = np.sqrt(np.diag(cov))
        correlation = cov / np.outer(volatility, volatility)
        correlation_eigenvalues = np.linalg.eigvalsh(correlation)
        q = n_assets / (n_obs - 1)
        sigma2 = noise_variance(correlation_eigenvalues, q)
        _, upper_edge = mp_edges(q, sigma2)
        n_factors = int(np.sum(correlation_eigenvalues > upper_edge))

    n_factors = int(np.clip(n_factors, 1, n_assets - 1))

    top_values = eigenvalues[-n_factors:]
    top_vectors = eigenvectors[:, -n_factors:]
    systematic = (top_vectors * top_values) @ top_vectors.T

    idiosyncratic = np.diag(cov) - np.diag(systematic)
    # Numerical slack can make a residual variance slightly negative; floor it
    # at a small positive multiple of the average variance so the result stays
    # positive definite and invertible.
    floor = 1e-8 * np.trace(cov) / n_assets
    idiosyncratic = np.maximum(idiosyncratic, floor)

    out = systematic + np.diag(idiosyncratic)
    return 0.5 * (out + out.T)


def factor_count(returns: np.ndarray) -> int:
    """The number of factors `factor_covariance` would choose, for reporting."""
    _, n_obs, n_assets = _demeaned(returns)
    cov = sample_covariance(returns)
    volatility = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(volatility, volatility)
    eigenvalues = np.linalg.eigvalsh(correlation)
    q = n_assets / (n_obs - 1)
    sigma2 = noise_variance(eigenvalues, q)
    _, upper_edge = mp_edges(q, sigma2)
    return int(max(1, np.sum(eigenvalues > upper_edge)))


#: The estimators compared throughout the study, in the order they are reported.
#: Keyed by the short name used in figures, tables and the README.
ESTIMATORS = {
    "sample": sample_covariance,
    "clipped": clipped_covariance,
    "linear": linear_shrinkage,
    "nonlinear": nonlinear_shrinkage,
    "factor": factor_covariance,
}

#: Longer labels for figures and tables.
ESTIMATOR_LABELS = {
    "sample": "Sample covariance",
    "clipped": "Eigenvalue clipping",
    "linear": "Linear shrinkage (Ledoit-Wolf)",
    "nonlinear": "Nonlinear shrinkage (Ledoit-Wolf)",
    "factor": "PCA factor model",
    "equal": "Equally weighted (1/N)",
}
