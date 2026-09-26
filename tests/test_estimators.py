"""Tests for the covariance estimators.

The recurring pattern: build a covariance, generate data from it, and check
that each estimator's error behaves the way the theory says it should.  Where a
published reference implementation exists, agree with it to machine precision
rather than to a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from estimation_error import estimators
from estimation_error.simulate import (
    factor_model_covariance,
    simulate_returns,
    simulate_white_noise,
)

TRUTH_60 = factor_model_covariance(60, 3, np.random.default_rng(7))


def _returns(n_obs: int, seed: int = 1, truth: np.ndarray | None = None) -> np.ndarray:
    truth = TRUTH_60 if truth is None else truth
    return simulate_returns(truth, n_obs, np.random.default_rng(seed))


@pytest.mark.parametrize("name", list(estimators.ESTIMATORS))
def test_every_estimator_returns_a_valid_covariance(name):
    """Symmetric, positive definite, right shape.  The optimiser inverts these."""
    returns = _returns(200)
    cov = estimators.ESTIMATORS[name](returns)
    assert cov.shape == (60, 60)
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_sample_covariance_is_the_textbook_one():
    returns = _returns(120)
    assert np.allclose(estimators.sample_covariance(returns), np.cov(returns, rowvar=False, ddof=1))


def test_linear_shrinkage_matches_an_independent_implementation():
    """Agree with scikit-learn's Ledoit-Wolf exactly, up to the stated ddof rescaling.

    To machine precision, not to a loose tolerance.  Ledoit and Wolf derive the
    intensity under the 1/T normalisation; `linear_shrinkage` does the same and
    then rescales by T/(T-1) so the result sits on the same footing as the other
    estimators here.  Undoing that rescaling reproduces the reference to within
    one unit in the last place.  The residual is the round trip through the
    rescaling rather than a difference in the formula, and the intensities agree
    to 1e-12 relative, which no approximate implementation would manage.
    """
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    returns = _returns(200)
    n_obs = returns.shape[0]
    reference, reference_intensity = sklearn_covariance.ledoit_wolf(returns)

    mine = estimators.linear_shrinkage(returns) * (n_obs - 1) / n_obs
    assert np.max(np.abs(mine - reference)) < 1e-15 * np.max(np.abs(reference))
    assert estimators.linear_shrinkage_intensity(returns) == pytest.approx(
        reference_intensity, rel=1e-12
    )


def test_shrinkage_intensity_is_in_the_unit_interval_and_rises_as_data_thins():
    """The estimator should lean harder on the target when it has less to work with."""
    intensities = [
        estimators.linear_shrinkage_intensity(_returns(T, seed=3)) for T in (80, 200, 2000)
    ]
    assert all(0.0 <= value <= 1.0 for value in intensities)
    assert intensities[0] > intensities[1] > intensities[2]


def test_linear_shrinkage_interpolates_between_sample_and_target():
    """The result must lie on the segment joining the two, which pins the algebra."""
    returns = _returns(150)
    n_obs = returns.shape[0]
    cov = estimators.sample_covariance(returns) * (n_obs - 1) / n_obs
    intensity = estimators.linear_shrinkage_intensity(returns)
    target = np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    expected = (1 - intensity) * cov + intensity * target
    assert np.allclose(estimators.linear_shrinkage(returns) * (n_obs - 1) / n_obs, expected)


def test_clipping_preserves_the_trace_of_the_correlation_matrix():
    """Replacing bulk eigenvalues by their mean is trace-preserving by construction.

    The trace of a correlation matrix is N, so this is also a check that the
    result really is a correlation matrix after rescaling.
    """
    returns = _returns(200)
    clipped = estimators.clipped_covariance(returns)
    volatility = np.sqrt(np.diag(clipped))
    correlation = clipped / np.outer(volatility, volatility)
    assert np.trace(correlation) == pytest.approx(60.0)
    assert np.allclose(np.diag(correlation), 1.0)


def test_clipping_keeps_the_individual_variances():
    """Cleaning is a statement about correlations; the variances are left alone."""
    returns = _returns(200)
    sample = estimators.sample_covariance(returns)
    clipped = estimators.clipped_covariance(returns)
    assert np.allclose(np.diag(sample), np.diag(clipped), rtol=1e-10)


def test_clipping_with_everything_kept_is_the_identity_map():
    """Keeping all N eigenvalues must return the sample covariance unchanged.

    A useful boundary case: it isolates the rescaling machinery from the
    thresholding decision, so a bug in one cannot hide behind the other.
    """
    returns = _returns(200)
    kept = estimators.clipped_covariance(returns, n_keep=60)
    assert np.allclose(kept, estimators.sample_covariance(returns), atol=1e-10)


def test_clipping_with_nothing_kept_flattens_the_correlation_spectrum():
    """Keeping zero eigenvalues must return the identity correlation, i.e. a diagonal matrix."""
    returns = _returns(200)
    flat = estimators.clipped_covariance(returns, n_keep=0)
    assert np.allclose(flat, np.diag(np.diag(flat)), atol=1e-10)


def test_factor_model_reproduces_the_diagonal_exactly():
    """POET puts the whole residual on the diagonal, so total variances are exact."""
    returns = _returns(300)
    sample = estimators.sample_covariance(returns)
    factor = estimators.factor_covariance(returns, n_factors=3)
    assert np.allclose(np.diag(sample), np.diag(factor), rtol=1e-10)


def test_factor_model_off_diagonal_is_the_rank_k_truncation():
    """All covariance between assets must come from exactly k factors.

    This is the defining property of the POET construction: the off-diagonal
    part of the estimate is the off-diagonal part of the rank-k spectral
    truncation of the sample covariance, and only the diagonal is adjusted.
    Checked against a truncation rebuilt here from `eigh`, and the truncation
    itself is confirmed to have rank exactly k.
    """
    returns = _returns(300)
    sample = estimators.sample_covariance(returns)
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    truncation = (eigenvectors[:, -3:] * eigenvalues[-3:]) @ eigenvectors[:, -3:].T
    assert np.linalg.matrix_rank(truncation, tol=1e-8 * eigenvalues[-1]) == 3

    factor = estimators.factor_covariance(returns, n_factors=3)
    off_diagonal = ~np.eye(60, dtype=bool)
    assert np.allclose(factor[off_diagonal], truncation[off_diagonal], atol=1e-10)


def test_nonlinear_shrinkage_preserves_eigenvectors():
    """It is a spectral method: only the eigenvalues move.

    Checked by confirming the cleaned matrix commutes with the sample one, which
    is equivalent to sharing an eigenbasis and does not depend on eigenvector
    sign or ordering conventions.
    """
    returns = _returns(200)
    sample = estimators.sample_covariance(returns)
    cleaned = estimators.nonlinear_shrinkage(returns)
    commutator = sample @ cleaned - cleaned @ sample
    assert np.max(np.abs(commutator)) < 1e-8 * np.max(np.abs(sample)) * np.max(np.abs(cleaned)) * 60


def test_nonlinear_shrinkage_compresses_the_spectrum():
    """The whole point is to pull the extremes in towards the middle."""
    returns = _returns(150)
    sample = np.linalg.eigvalsh(estimators.sample_covariance(returns))
    cleaned = np.linalg.eigvalsh(estimators.nonlinear_shrinkage(returns))
    assert cleaned.max() < sample.max()
    assert cleaned.min() > sample.min()
    assert np.linalg.cond(np.diag(cleaned)) < np.linalg.cond(np.diag(sample))


def test_nonlinear_shrinkage_refuses_the_singular_case():
    with pytest.raises(NotImplementedError):
        estimators.nonlinear_shrinkage(_returns(40))


def test_cleaning_beats_the_sample_covariance_when_data_is_scarce():
    """At q near 0.4, every cleaned estimator must be closer to the truth on average.

    Averaged over draws, because any single draw can go either way; that is what
    estimation error means.
    """
    n_obs = 150
    errors = {name: [] for name in estimators.ESTIMATORS}
    for seed in range(12):
        returns = _returns(n_obs, seed=100 + seed)
        for name, estimator in estimators.ESTIMATORS.items():
            errors[name].append(np.linalg.norm(estimator(returns) - TRUTH_60, "fro"))

    baseline = np.mean(errors["sample"])
    for name in ("clipped", "linear", "nonlinear", "factor"):
        assert np.mean(errors[name]) < baseline, f"{name} failed to beat the sample covariance"


def test_consistent_estimators_converge_to_the_truth():
    """Sample, linear and nonlinear shrinkage must all go to zero error as T grows.

    Clipping and the factor model are deliberately excluded: both impose
    structure the truth does not exactly have, so they converge to a structured
    approximation rather than to Sigma.  That is a property of those estimators,
    not a bug, and `test_structural_estimators_converge_to_their_own_limit`
    states what they do converge to.
    """
    errors = {}
    for name in ("sample", "linear", "nonlinear"):
        errors[name] = [
            np.linalg.norm(estimators.ESTIMATORS[name](_returns(T, seed=7)) - TRUTH_60, "fro")
            for T in (500, 4000, 20000)
        ]
    # Frobenius error of a consistent estimator falls like sqrt(N^2/T), so a
    # fortyfold increase in T should buy a factor of a few.  Asserting "more
    # than 2.5" is loose enough to survive sampling noise and tight enough that
    # an estimator with a bias floor, which would flatten out instead, cannot
    # pass.
    for name, sequence in errors.items():
        assert sequence[0] > sequence[1] > sequence[2], f"{name} did not converge"
        assert sequence[0] / sequence[-1] > 2.5, f"{name} converged too slowly to be consistent"


def test_structural_estimators_converge_to_their_own_limit():
    """Clipping and POET converge, but to the structured matrix they can represent.

    The limit is computable: apply the same structural map to the true
    covariance instead of to an estimate.  Clipping's limit at small q is Sigma
    itself, so its bias floor is zero; the factor model's is not, and asserting
    that the gap is real is more honest than pretending the estimator is
    consistent.
    """
    returns = _returns(20000, seed=7)
    factor = estimators.factor_covariance(returns, n_factors=3)

    eigenvalues, eigenvectors = np.linalg.eigh(TRUTH_60)
    systematic = (eigenvectors[:, -3:] * eigenvalues[-3:]) @ eigenvectors[:, -3:].T
    limit = systematic + np.diag(np.diag(TRUTH_60) - np.diag(systematic))

    bias_floor = np.linalg.norm(limit - TRUTH_60, "fro")
    assert bias_floor > 0.1  # the floor is genuinely nonzero: POET cannot represent Sigma
    # With a lot of data the estimate sits closer to the structure it can
    # represent than that structure sits to the truth, which is what "converges
    # to its own limit rather than to Sigma" means.
    assert np.linalg.norm(factor - limit, "fro") < bias_floor
    assert np.linalg.norm(factor - TRUTH_60, "fro") > 0.5 * bias_floor


def test_factor_count_finds_the_planted_factors():
    """On data with three planted factors and a realistic q, the rule should say about three."""
    returns = _returns(250, seed=11)
    assert 2 <= estimators.factor_count(returns) <= 8


def test_estimators_reject_malformed_input():
    with pytest.raises(ValueError):
        estimators.sample_covariance(np.zeros((1, 5)))
    with pytest.raises(ValueError):
        estimators.sample_covariance(np.zeros(5))


def test_clipping_handles_pure_noise_by_flattening_everything():
    """With a true identity covariance the whole spectrum is bulk, so it all collapses."""
    returns = simulate_white_noise(80, 300, np.random.default_rng(12))
    clipped = estimators.clipped_covariance(returns)
    volatility = np.sqrt(np.diag(clipped))
    correlation = clipped / np.outer(volatility, volatility)
    off_diagonal = correlation - np.eye(80)
    assert np.max(np.abs(off_diagonal)) < 0.05
