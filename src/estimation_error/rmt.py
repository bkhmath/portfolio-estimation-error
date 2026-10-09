"""Marchenko-Pastur law, and the spectral diagnosis built on it.

The eigenvalues of a sample covariance are systematically dispersed, the small
ones biased down and the large ones up, by an amount depending only on q = N/T.
Inversion scales each eigendirection by 1/lambda, so Markowitz leans hardest on
the eigenvalues that are worst estimated.

Take N uncorrelated unit-variance series, so the true spectrum is a point mass
at 1.  As N, T -> infinity with N/T -> q, the empirical spectral distribution
spreads over [(1-sqrt(q))^2, (1+sqrt(q))^2] instead: at q = 0.4, which is 100
assets and a year of daily data, that is [0.13, 2.65].  A twentyfold spread of
eigenvalues, all of it noise.

Where the law comes from.  With the Stieltjes transform
m(z) = integral dF(x)/(x - z), writing the sample covariance as a sum of T
rank-one terms and applying Sherman-Morrison to each gives, in the limit,

    q z m(z)^2 + (z + q - 1) m(z) + 1 = 0.

The branch with m(z) ~ -1/z at infinity is the one belonging to a probability
measure; its imaginary part on the real axis is `mp_density`, and the roots of
the discriminant are the edges (1 +/- sqrt(q))^2, so support and density come
out of the same equation.  docs/marchenko-pastur.md works through the algebra.

Laloux, Cizeau, Bouchaud and Potters turned this into a test: an eigenvalue
inside the bulk cannot be told apart from noise, so its direction is not worth
optimising along, while eigenvalues above the upper edge carry information.  On
equity data the largest is the market mode, far above the edge.

One caveat that changes how this is tested: eigenvalues are not an iid sample
from the limiting density.  They repel, so F_N converges faster than the
1/sqrt(N) of an iid sample, and the usual Kolmogorov-Smirnov p-value does not
apply.  The tests use KS as a distance against an absolute threshold.
"""

from __future__ import annotations

import numpy as np

# Number of quadrature points used to build the Marchenko-Pastur CDF.  The
# integrand after the substitution below is smooth and periodic-ish, so this is
# far more than enough for the ~1e-12 accuracy we want when comparing against
# empirical distributions.
_CDF_GRID = 20001


def mp_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Return the edges (lambda_minus, lambda_plus) of the Marchenko-Pastur bulk.

    The branch points of the square root in the self-consistent equation,
    hence the form sigma^2 (1 +/- sqrt(q))^2.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    root_q = np.sqrt(q)
    return sigma2 * (1.0 - root_q) ** 2, sigma2 * (1.0 + root_q) ** 2


def mp_density(x: np.ndarray | float, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density, zero outside the bulk.

    For q > 1 the sample covariance is singular and the law carries an atom of
    mass 1 - 1/q at the origin.  An atom has no density, so it appears in
    `mp_cdf` and not here.
    """
    x = np.asarray(x, dtype=float)
    lam_minus, lam_plus = mp_edges(q, sigma2)

    inside = (x > lam_minus) & (x < lam_plus)
    density = np.zeros_like(x)
    # np.where on the whole array would evaluate the sqrt at negative arguments
    # outside the bulk, so index instead and keep the warnings clean.
    xs = x[inside]
    density[inside] = np.sqrt((lam_plus - xs) * (xs - lam_minus)) / (2.0 * np.pi * q * sigma2 * xs)
    return density


def mp_cdf(x: np.ndarray | float, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur cumulative distribution function.

    The closed form involves two arctangents whose branches are easy to get
    wrong.  Substituting

        x(theta) = sigma^2 (1 + q - 2 sqrt(q) cos theta),   theta in [0, pi],

    and using (lambda_plus - x)(x - lambda_minus) = 4 q sigma^4 sin^2 theta
    with dx = 2 sqrt(q) sigma^2 sin theta dtheta turns the edge singularities
    into

        dF = (2 / pi) sin^2(theta) / (x(theta) / sigma^2) dtheta,

    bounded and smooth on the closed interval, so quadrature on a fine grid is
    accurate to round-off.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lam_minus, lam_plus = mp_edges(q, sigma2)

    theta = np.linspace(0.0, np.pi, _CDF_GRID)
    grid_x = sigma2 * (1.0 + q - 2.0 * np.sqrt(q) * np.cos(theta))
    integrand = (2.0 / np.pi) * np.sin(theta) ** 2 / (grid_x / sigma2)
    # Cumulative Simpson via cumulative trapezoid on a grid this fine is
    # already at round-off; keep it simple.
    step = theta[1] - theta[0]
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * step * (integrand[1:] + integrand[:-1]))])

    # Mass below the bulk: the atom at the origin when the matrix is singular.
    atom = max(0.0, 1.0 - 1.0 / q)

    out = np.interp(x, grid_x, cumulative, left=0.0, right=cumulative[-1]) + atom
    out[x < lam_minus] = atom
    out[x < 0.0] = 0.0
    out[x >= lam_plus] = 1.0
    return out


def mp_stieltjes(z: np.ndarray | complex, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Stieltjes transform m(z) = integral dF(x)/(x - z) of the Marchenko-Pastur law.

    Valid for Im(z) > 0, the only place it is needed.  The root of the
    self-consistent quadratic behaving like -1/z at infinity, which is what
    picks the branch.

    Written as sqrt(z - lambda_minus) * sqrt(z - lambda_plus) and not sqrt of
    the product: for Im(z) > 0 each factor is in the upper half plane, so each
    principal root is in the first quadrant and the product has the right
    branch.  The principal root of the product would cross the wrong cut.
    """
    z = np.asarray(z, dtype=complex)
    if np.any(z.imag <= 0):
        raise ValueError("mp_stieltjes is only implemented for Im(z) > 0")
    lam_minus, lam_plus = mp_edges(q, sigma2)
    root = np.sqrt(z - lam_minus) * np.sqrt(z - lam_plus)
    return (-(z + sigma2 * (q - 1.0)) + root) / (2.0 * q * sigma2 * z)


def ks_distance(eigenvalues: np.ndarray, q: float, sigma2: float = 1.0) -> float:
    """Kolmogorov-Smirnov distance between an empirical spectrum and the law.

    Sup norm between the empirical CDF and `mp_cdf`.  A distance against an
    absolute threshold, not a test statistic; see the repulsion caveat above.
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))
    n = lam.size
    theoretical = mp_cdf(lam, q, sigma2)
    # The empirical CDF jumps at each eigenvalue, so the sup norm is attained
    # immediately before or immediately after a jump; check both sides.
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - theoretical)), np.max(np.abs(lower - theoretical))))


def l1_distance(eigenvalues: np.ndarray, q: float, sigma2: float = 1.0, bins: int = 60) -> float:
    """L1 distance between a histogram of the spectrum and the density.

    A complement to `ks_distance`, which is insensitive to a density wrong
    locally but right on average.  The range is widened past the edges so
    eigenvalues leaking outside the bulk count as error rather than being
    dropped; some leakage is normal at finite N, on the N^(-2/3) scale of the
    Tracy-Widom fluctuations.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    lam_minus, lam_plus = mp_edges(q, sigma2)
    width = lam_plus - lam_minus
    lo = min(lam_minus - 0.05 * width, lam.min())
    hi = max(lam_plus + 0.05 * width, lam.max())

    counts, edges = np.histogram(lam, bins=bins, range=(lo, hi), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return float(np.sum(np.abs(counts - mp_density(centres, q, sigma2)) * np.diff(edges)))


def noise_variance(eigenvalues: np.ndarray, q: float) -> float:
    """Estimate the variance of the noise part of a real spectrum.

    On real data the total variance is not mostly noise: the market mode alone
    can carry a third of the trace, so sigma^2 = mean(all eigenvalues) puts the
    upper edge far too high and calls genuine factors noise.

    Laloux et al. fit sigma^2 to the plausibly-noise part only.  Given a
    sigma^2, call the eigenvalues below the implied upper edge the bulk and
    re-estimate sigma^2 as their mean, which is what the law says it should be;
    repeat until the bulk stops changing.  From the mean of the whole spectrum
    the top eigenvalues drop out one at a time, in a handful of steps.
    """
    lam = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    sigma2 = float(np.mean(lam))

    for _ in range(100):
        _, lam_plus = mp_edges(q, sigma2)
        bulk = lam[lam < lam_plus]
        if bulk.size < 2:
            # Everything has been declared signal; nothing sensible to fit.
            break
        new_sigma2 = float(np.mean(bulk))
        if abs(new_sigma2 - sigma2) < 1e-12 * max(sigma2, 1.0):
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2

    return sigma2


def count_signal_eigenvalues(eigenvalues: np.ndarray, q: float) -> int:
    """How many eigenvalues sit above the fitted Marchenko-Pastur upper edge.

    The number of directions where the data says something noise alone would
    not.  On the French industry portfolios it is in the low single digits,
    which is the empirical content of a handful of factors being enough.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    sigma2 = noise_variance(lam, q)
    _, lam_plus = mp_edges(q, sigma2)
    return int(np.sum(lam > lam_plus))
