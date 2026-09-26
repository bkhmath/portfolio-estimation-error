"""Tests for the Marchenko-Pastur implementation.

These are the tests that matter most.  Marchenko-Pastur is a theorem, so it
either matches or the code is wrong, and every other result in the repository
is downstream of it.  Nothing here compares against a recorded output.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from estimation_error import rmt
from estimation_error.estimators import sample_covariance
from estimation_error.simulate import simulate_white_noise

QS = [0.05, 0.1, 0.25, 0.4, 0.5, 0.8, 0.95]


@pytest.mark.parametrize("q", QS)
def test_edges_are_the_branch_points(q):
    """The edges must be where the discriminant of the self-consistent quadratic vanishes.

    That discriminant is z^2 - 2z(1 + q) + (q - 1)^2, so this is an independent
    statement of where the support ends, not a restatement of the formula.
    """
    lower, upper = rmt.mp_edges(q)
    for edge in (lower, upper):
        assert abs(edge**2 - 2 * edge * (1 + q) + (q - 1) ** 2) < 1e-12


@pytest.mark.parametrize("q", [*QS, 1.5, 3.0])
def test_density_integrates_to_the_right_mass(q):
    """Mass 1 below q = 1; mass 1/q above it, the rest sitting in the atom at zero."""
    lower, upper = rmt.mp_edges(q)
    mass, _ = quad(lambda x: rmt.mp_density(x, q), lower, upper, limit=400)
    assert mass == pytest.approx(1.0 if q <= 1 else 1.0 / q, abs=1e-8)


@pytest.mark.parametrize("q", QS)
def test_first_moment_is_one(q):
    """The mean eigenvalue must equal the true variance: E tr(S)/N = tr(Sigma)/N = 1."""
    lower, upper = rmt.mp_edges(q)
    mean, _ = quad(lambda x: x * rmt.mp_density(x, q), lower, upper, limit=400)
    assert mean == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("q", [0.1, 0.4, 0.9, 2.0])
def test_cdf_agrees_with_quadrature_of_the_density(q):
    """The substitution-based CDF must equal the integral of the density it came from."""
    lower, upper = rmt.mp_edges(q)
    atom = max(0.0, 1.0 - 1.0 / q)
    for x in np.linspace(max(lower, 1e-9), upper, 9):
        expected = atom + quad(lambda t: rmt.mp_density(t, q), lower, x, limit=400)[0]
        assert rmt.mp_cdf(x, q)[0] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("q", [0.1, 0.4, 0.8])
def test_stieltjes_transform_recovers_the_density(q):
    """Im m(x + i0+)/pi is the density; this checks the branch choice was right.

    If the wrong root of the quadratic were taken, the imaginary part would come
    out negative, so this also pins down the sign convention.
    """
    lower, upper = rmt.mp_edges(q)
    x = np.linspace(lower + 1e-3, upper - 1e-3, 300)
    recovered = rmt.mp_stieltjes(x + 1e-10j, q).imag / np.pi
    assert np.max(np.abs(recovered - rmt.mp_density(x, q))) < 1e-5


def test_stieltjes_behaves_like_minus_one_over_z_at_infinity():
    """m(z) ~ -1/z is the normalisation that forces the branch; check it directly."""
    z = 1e6j
    assert rmt.mp_stieltjes(z, 0.4) == pytest.approx(-1.0 / z, rel=1e-4)


@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.9])
def test_empirical_spectrum_matches_the_law(q):
    """The mathematical core: simulate iid data and compare the spectrum to the law.

    The true covariance is the identity, so every deviation from a point mass at
    1 is estimation noise and the law predicts its distribution exactly.  The
    threshold is absolute rather than a p-value: eigenvalues are not an iid
    sample, they repel, so the empirical distribution comes out far more regular
    than iid sampling would give and a standard KS p-value means nothing here.
    """
    n_assets = 400
    n_obs = round(n_assets / q) + 1
    returns = simulate_white_noise(n_assets, n_obs, np.random.default_rng(int(100 * q)))
    eigenvalues = np.linalg.eigvalsh(sample_covariance(returns))
    effective_q = n_assets / (n_obs - 1)

    assert rmt.ks_distance(eigenvalues, effective_q) < 0.02
    assert rmt.l1_distance(eigenvalues, effective_q) < 0.20


def test_convergence_rate_is_the_rigidity_rate_not_the_iid_rate():
    """Quadrupling N must cut the KS distance by clearly more than the iid factor of 2.

    An iid sample of size N has KS distance of order N^(-1/2), so 4N would give a
    factor of 2.  Eigenvalue repulsion makes the spectrum converge like N^(-1),
    a factor of 4.  Asserting "more than 2.5" distinguishes the two regimes
    while leaving room for sampling noise, and would fail if the eigenvalues
    were being generated or compared incorrectly.
    """
    q = 0.25
    distances = []
    for n_assets in (100, 400, 1600):
        n_obs = round(n_assets / q) + 1
        returns = simulate_white_noise(n_assets, n_obs, np.random.default_rng(5 + n_assets))
        eigenvalues = np.linalg.eigvalsh(sample_covariance(returns))
        distances.append(rmt.ks_distance(eigenvalues, n_assets / (n_obs - 1)))

    assert distances[0] / distances[1] > 2.5
    assert distances[1] / distances[2] > 2.5


def test_spectrum_of_scaled_noise_is_the_scaled_law():
    """sigma^2 enters only as a scale, so a fitted sigma^2 must recover it."""
    sigma2 = 4.0
    n_assets, n_obs = 300, 1200
    returns = np.sqrt(sigma2) * simulate_white_noise(n_assets, n_obs, np.random.default_rng(2))
    eigenvalues = np.linalg.eigvalsh(sample_covariance(returns))
    fitted = rmt.noise_variance(eigenvalues, n_assets / (n_obs - 1))
    assert fitted == pytest.approx(sigma2, rel=0.05)


def test_pure_noise_has_no_signal_eigenvalues():
    """With a true identity covariance, nothing should be declared signal.

    A handful of false positives is expected at the edge, since the largest
    eigenvalue fluctuates on the Tracy-Widom scale N^(-2/3), so the assertion is
    that the count is small rather than zero.
    """
    n_assets, n_obs = 300, 1200
    returns = simulate_white_noise(n_assets, n_obs, np.random.default_rng(4))
    eigenvalues = np.linalg.eigvalsh(sample_covariance(returns))
    count = rmt.count_signal_eigenvalues(eigenvalues, n_assets / (n_obs - 1))
    assert count <= 3


def test_planted_factors_are_detected():
    """A covariance with three planted factors must show eigenvalues outside the bulk."""
    from estimation_error.simulate import factor_model_covariance, simulate_returns

    truth = factor_model_covariance(200, 3, np.random.default_rng(7))
    returns = simulate_returns(truth, 800, np.random.default_rng(8))
    cov = sample_covariance(returns)
    volatility = np.sqrt(np.diag(cov))
    correlation = cov / np.outer(volatility, volatility)
    eigenvalues = np.linalg.eigvalsh(correlation)
    assert rmt.count_signal_eigenvalues(eigenvalues, 200 / 799) >= 3


def test_density_is_zero_outside_the_bulk():
    q = 0.3
    lower, upper = rmt.mp_edges(q)
    assert rmt.mp_density(np.array([lower - 0.1, upper + 0.1, -1.0]), q).tolist() == [0.0, 0.0, 0.0]


def test_cdf_is_monotone_and_bounded():
    q = 0.4
    x = np.linspace(-0.5, 5.0, 500)
    values = rmt.mp_cdf(x, q)
    assert np.all(np.diff(values) >= -1e-12)
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_q", [0.0, -0.5])
def test_invalid_q_raises(bad_q):
    with pytest.raises(ValueError):
        rmt.mp_edges(bad_q)


def test_stieltjes_rejects_the_real_axis():
    with pytest.raises(ValueError):
        rmt.mp_stieltjes(1.0 + 0.0j, 0.4)
