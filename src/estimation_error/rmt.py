"""Marchenko-Pastur law, and the spectral diagnosis built on it.

The sample covariance of N assets from T observations is not just a slightly
noisy version of the truth.  Its eigenvalues are systematically dispersed, the
small ones biased down and the large ones biased up, by an amount depending
only on q = N/T.  Markowitz optimisation inverts the matrix, and inversion
scales each eigendirection by 1/lambda, so it leans hardest on the eigenvalues
that are worst estimated.

Marchenko-Pastur quantifies that.  Take N uncorrelated unit-variance series, so
the true covariance is the identity and the true spectrum is a point mass at 1.
As N, T -> infinity with N/T -> q, the empirical spectral distribution does not
collapse onto that point mass; it spreads over [(1-sqrt(q))^2, (1+sqrt(q))^2].
At q = 0.4, which is 100 assets and a year of daily data, that is [0.13, 2.65].
A twentyfold spread of eigenvalues, all of it noise.

Sketch of where the law comes from, since the formulas below are opaque without
it.  Work with the Stieltjes transform m(z) = integral dF(x)/(x - z) for z off
the real line.  It determines F, because the density is Im m(x + i0)/pi, and it
turns the addition of one observation into a rank-one update.  Write the sample
covariance as a sum of T rank-one terms, apply Sherman-Morrison to each, and in
the limit m satisfies

    q z m(z)^2 + (z + q - 1) m(z) + 1 = 0.

A quadratic.  The branch with m(z) ~ -1/z at infinity is the one belonging to a
probability measure; its imaginary part on the real axis is the density in
`mp_density`, and the two roots of the discriminant are the edges
(1 +/- sqrt(q))^2.  So the support falls out of the same equation as the
density and is not a separate input.  docs/marchenko-pastur.md works through
the algebra.

Laloux, Cizeau, Bouchaud and Potters turned this into a test.  An eigenvalue of
a real sample covariance lying inside the bulk cannot be told apart from noise,
so the direction it points in is not worth optimising along.  Eigenvalues above
the upper edge carry information.  On equity data the largest is the market
mode, every stock moving together, and it sits far above the edge.

One caveat, because it changes how this code is tested.  Eigenvalues are not an
iid sample from the limiting density.  They repel, which leaves them more
evenly spaced than iid points would be, so F_N converges faster than the
1/sqrt(N) an iid sample gives.  A Kolmogorov-Smirnov statistic is still a fine
distance, but its usual p-value does not apply and would be far too
conservative.  The tests here use KS as a distance against an absolute
threshold.
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

    These are the branch points of the square root in the solution of the
    self-consistent equation, which is why they take the form
    sigma^2 (1 +/- sqrt(q))^2 rather than something requiring a separate
    derivation.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    root_q = np.sqrt(q)
    return sigma2 * (1.0 - root_q) ** 2, sigma2 * (1.0 + root_q) ** 2


def mp_density(x: np.ndarray | float, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density, zero outside the bulk.

    For q > 1 there are more assets than observations, the sample covariance is
    singular, and the law carries an atom of mass 1 - 1/q at the origin on top
    of this density.  An atom has no density, so it does not appear here;
    `mp_cdf` is where it shows up.
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

    There is a closed form involving two arctangents, but it is easy to get the
    branches wrong and hard to check.  Instead substitute

        x(theta) = sigma^2 (1 + q - 2 sqrt(q) cos theta),   theta in [0, pi],

    which maps the interval onto the bulk and, because
    (lambda_plus - x)(x - lambda_minus) = 4 q sigma^4 sin^2 theta and
    dx = 2 sqrt(q) sigma^2 sin theta dtheta, turns the density's inverse square
    root singularities at the two edges into

        dF = (2 / pi) sin^2(theta) / (x(theta) / sigma^2) dtheta,

    an integrand that is bounded and smooth on the closed interval.  Simpson on
    a fine grid is then accurate to machine precision, and the result is
    obviously correct in a way the arctangent form is not.
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

    Valid for Im(z) > 0, which is the only place it is needed.  This is the
    root of the self-consistent quadratic quoted in the module docstring that
    behaves like -1/z at infinity; that condition is what picks the branch, and
    it is forced by m being the transform of a probability measure.

    Writing the square root as sqrt(z - lambda_minus) * sqrt(z - lambda_plus)
    rather than sqrt of the product matters: for Im(z) > 0 each factor lies in
    the upper half plane, so each principal square root lies in the first
    quadrant and the product has the branch we want.  The principal square root
    of the product would cross the wrong cut.
    """
    z = np.asarray(z, dtype=complex)
    if np.any(z.imag <= 0):
        raise ValueError("mp_stieltjes is only implemented for Im(z) > 0")
    lam_minus, lam_plus = mp_edges(q, sigma2)
    root = np.sqrt(z - lam_minus) * np.sqrt(z - lam_plus)
    return (-(z + sigma2 * (q - 1.0)) + root) / (2.0 * q * sigma2 * z)


def ks_distance(eigenvalues: np.ndarray, q: float, sigma2: float = 1.0) -> float:
    """Kolmogorov-Smirnov distance between an empirical spectrum and the law.

    Sup norm between the empirical CDF of the eigenvalues and `mp_cdf`.  Used
    as a distance against an absolute threshold rather than as a test
    statistic; see the caveat about eigenvalue repulsion above.
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

    A complement to `ks_distance`: the KS distance is insensitive to a density
    that is wrong locally but right on average, and this is not.  The
    integration range is widened past the edges so that eigenvalues leaking
    outside the bulk get counted as error instead of being silently dropped.
    Some leakage is normal at finite N, on the scale N^(-2/3) set by the
    Tracy-Widom fluctuations of the extreme eigenvalue.
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

    On real data the total variance is not 1, and it is not even mostly noise:
    the market mode alone can carry a third of the trace.  Fitting the bulk
    with sigma^2 = mean(all eigenvalues) therefore puts the upper edge far too
    high and declares genuine factors to be noise.

    The fix used by Laloux et al. is to fit sigma^2 only to the part of the
    spectrum that is plausibly noise.  Iterate: given a current sigma^2, call
    the eigenvalues below the implied upper edge the bulk, and re-estimate
    sigma^2 as their mean, which is what the law says it should be; repeat
    until the set of eigenvalues called bulk stops changing.  Starting from the
    mean of the whole spectrum, the top eigenvalues drop out one at a time and
    it converges in a handful of steps.
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

    This is the number of directions in which the data is saying something that
    noise alone would not have produced.  On the French industry portfolios it
    comes out in the low single digits, which is the empirical content of the
    statement that a factor model with a handful of factors is enough.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    sigma2 = noise_variance(lam, q)
    _, lam_plus = mp_edges(q, sigma2)
    return int(np.sum(lam > lam_plus))
