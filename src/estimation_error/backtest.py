"""The rolling out-of-sample protocol, and the ex-ante / ex-post gap.

Every setting in this file could have been chosen after seeing results, and a
setting chosen that way is not evidence.  So the estimation window, the
rebalancing frequency and the sample start date live here as module constants,
are stated in the README, and are never passed in as tuned arguments by the
study script.  They were fixed before any out-of-sample number existed:

- 250 trading days of estimation, one calendar year.  Long enough for the
  covariance to be estimable, short enough for it to still be current, and it
  puts the two datasets at q = 49/249 = 0.20 and q = 95/249 = 0.38, either side
  of the regime the project is about.
- Rebalance every 21 trading days and hold for the following month.  Monthly is
  the standard frequency in this literature.  Daily rebalancing would give more
  out-of-sample observations and tighter variance estimates, but turnover would
  then be mostly noise trading and would stop separating the estimators, and
  turnover is one of the things being measured.
- Sample from 1970, for the coverage reason given in `data.py`.

`predicted_variance_ratio` and `realised_variance_ratio` are the interesting
functions here.  They give the variance the optimiser predicts for its own
portfolio and the variance that portfolio goes on to realise, both as exact
finite-sample expectations under Gaussian returns, and their ratio grows like
1/(1 - q)^2.  At q = 0.38 that is a factor of 2.6, so the optimiser is not
mildly optimistic; it is wrong by more than it is right and has no way of
telling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .estimators import ESTIMATORS
from .optimise import (
    equal_weights,
    gross_leverage,
    mean_variance_long_only,
    mean_variance_weights,
    min_variance_long_only,
    min_variance_weights,
    portfolio_variance,
    turnover,
    weight_concentration,
)

#: Length of the rolling estimation window, in trading days.  Fixed in advance.
ESTIMATION_WINDOW = 250

#: Trading days between rebalances, and hence the holding period.  Fixed in advance.
REBALANCE_EVERY = 21

#: Trading days per year, for annualising volatility.
TRADING_DAYS = 252


def predicted_variance_ratio(n_assets: int, n_obs: int) -> float:
    """E[predicted variance] / true minimum variance, exactly, for Gaussian returns.

    The minimum-variance portfolio built from the sample covariance S predicts
    for itself the variance 1 / (1' S^-1 1).  With demeaned Gaussian data,
    S is Wishart with n = T - 1 degrees of freedom, and the scalar 1/(1'S^-1 1)
    is distributed as the true minimum variance times a chi-squared on
    n - N + 1 degrees of freedom divided by n.  Taking expectations,

        E[predicted] / true = (n - N + 1) / n = (T - N) / (T - 1).

    The optimiser's own forecast of its variance is therefore biased *down* by
    a factor that goes to zero as N approaches T.  It does not just fail to
    notice the trouble it is in, it grows more confident as the trouble gets
    worse.
    """
    n = n_obs - 1
    if n - n_assets + 1 <= 0:
        return float("nan")
    return (n - n_assets + 1) / n


def realised_variance_ratio(n_assets: int, n_obs: int) -> float:
    """E[realised variance] / true minimum variance, exactly, for Gaussian returns.

    The same portfolio's true variance, w' Sigma w, has expectation

        E[realised] / true = (n - 1) / (n - N) = (T - 2) / (T - N - 1),

    which diverges as N approaches T - 1.  Multiplying the two ratios gives the
    factor by which the optimiser deceives itself,

        (T - 2)(T - 1) / [(T - N - 1)(T - N)]  ->  1 / (1 - q)^2.

    Both formulas are checked against Monte Carlo in `tests/test_backtest.py`.
    """
    n = n_obs - 1
    if n - n_assets <= 0:
        return float("inf")
    return (n - 1) / (n - n_assets)


def self_deception_ratio(n_assets: int, n_obs: int) -> float:
    """E[realised] / E[predicted]: how much the optimiser understates its own risk."""
    return realised_variance_ratio(n_assets, n_obs) / predicted_variance_ratio(n_assets, n_obs)


def drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Where a set of weights ends up after a holding period, before rebalancing.

    A portfolio left alone does not keep its weights: positions that went up
    become a larger share of the book.  Turnover must be measured against these
    drifted weights, not against the weights originally set, or every strategy
    is charged for trades it did not make.
    """
    growth = np.prod(1.0 + np.asarray(returns), axis=0)
    value = np.asarray(weights) * growth
    total = value.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.asarray(weights, dtype=float)
    return value / total


@dataclass
class BacktestResult:
    """Out-of-sample record for one estimator on one dataset."""

    name: str
    daily_returns: np.ndarray
    predicted_variances: np.ndarray
    realised_variances: np.ndarray
    turnovers: np.ndarray
    concentrations: np.ndarray
    leverages: np.ndarray
    dates: list = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Headline statistics.  Volatility is annualised; Sharpe is on raw returns.

        The Sharpe ratio here uses raw rather than excess returns, because no
        risk-free series is loaded and subtracting one would change every
        strategy's numerator by the same amount without changing the ranking on
        a set of fully invested portfolios.  It is reported for comparison with
        DeMiguel, Garlappi and Uppal, not as a claim about tradeable alpha.
        """
        returns = self.daily_returns
        volatility = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))
        mean = float(np.mean(returns) * TRADING_DAYS)
        predicted_volatility = float(
            np.sqrt(np.mean(self.predicted_variances)) * np.sqrt(TRADING_DAYS)
        )
        return {
            "annual_volatility": volatility,
            "annual_return": mean,
            "sharpe": mean / volatility if volatility > 0 else float("nan"),
            "predicted_volatility": predicted_volatility,
            "variance_ratio": (volatility / predicted_volatility) ** 2
            if predicted_volatility > 0
            else float("nan"),
            "mean_turnover": float(np.mean(self.turnovers)) if self.turnovers.size else 0.0,
            "mean_concentration": float(np.mean(self.concentrations)),
            "mean_leverage": float(np.mean(self.leverages)),
            "n_rebalances": int(self.predicted_variances.size),
            "n_days": int(returns.size),
        }


#: Suffix marking a strategy as the long-only version of an estimator, so that
#: "clipped" and "clipped_long_only" both work as strategy names.
LONG_ONLY_SUFFIX = "_long_only"

#: Suffix marking a strategy as mean-variance with a return target rather than
#: the global minimum-variance portfolio.
MEAN_VARIANCE_SUFFIX = "+mv"


def _weights_for(name: str, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights and the covariance used to produce them, for one estimator on one window."""
    from .estimators import sample_covariance

    n_assets = window.shape[1]
    if name == "equal":
        # The 1/N portfolio still needs a covariance to state a prediction with,
        # and the honest choice is the sample covariance: it is what a user of
        # this portfolio would have had available.
        return equal_weights(n_assets), sample_covariance(window)

    long_only = name.endswith(LONG_ONLY_SUFFIX)
    stem = name[: -len(LONG_ONLY_SUFFIX)] if long_only else name

    # A mean-variance strategy is written "<estimator>+mv": same covariance, but
    # a return target instead of the unconstrained minimum.  The target is the
    # in-sample mean return of the equally weighted portfolio over the same
    # window, which the data has just shown to be attainable and which needs no
    # tuning.  These strategies exist to show what happens when estimated
    # *means* enter the problem, so the target has to be estimated from the
    # window like everything else.
    mean_variance = stem.endswith(MEAN_VARIANCE_SUFFIX)
    if mean_variance:
        stem = stem[: -len(MEAN_VARIANCE_SUFFIX)]

    if stem not in ESTIMATORS:
        raise KeyError(f"unknown strategy {name!r}")

    cov = ESTIMATORS[stem](window)

    if mean_variance:
        mu = window.mean(axis=0)
        target = float(mu.mean())
        if long_only:
            return mean_variance_long_only(cov, mu, target), cov
        return mean_variance_weights(cov, mu, target), cov

    if long_only:
        weights, _ = min_variance_long_only(cov)
    else:
        weights = min_variance_weights(cov)
    return weights, cov


def run_backtest(
    returns: pd.DataFrame | np.ndarray,
    strategies: list[str],
    window: int = ESTIMATION_WINDOW,
    step: int = REBALANCE_EVERY,
) -> dict[str, BacktestResult]:
    """Roll the fixed protocol over a return panel and record what happened.

    At each rebalance date the covariance is estimated on the previous `window`
    days, weights are formed, and the portfolio is held for the next `step`
    days.  Nothing from the holding period is visible to the estimator, which
    is the entire point; the indices below are the place where a look-ahead bug
    would live, so they are written out rather than hidden in a slicing helper.
    """
    values = np.asarray(returns, dtype=float)
    dates = list(returns.index) if isinstance(returns, pd.DataFrame) else list(range(len(values)))
    n_obs = values.shape[0]
    if n_obs < window + step:
        raise ValueError(f"need at least {window + step} observations, got {n_obs}")

    records: dict[str, dict[str, list]] = {
        name: {
            "returns": [],
            "predicted": [],
            "realised": [],
            "turnover": [],
            "concentration": [],
            "leverage": [],
            "dates": [],
        }
        for name in strategies
    }
    previous_weights: dict[str, np.ndarray | None] = dict.fromkeys(strategies)

    for start in range(0, n_obs - window - step + 1, step):
        estimation = values[start : start + window]
        holding = values[start + window : start + window + step]

        for name in strategies:
            weights, cov = _weights_for(name, estimation)

            period_returns = holding @ weights
            record = records[name]
            record["returns"].extend(period_returns.tolist())
            record["predicted"].append(portfolio_variance(cov, weights))
            record["realised"].append(float(np.var(period_returns, ddof=1)))
            record["concentration"].append(weight_concentration(weights))
            record["leverage"].append(gross_leverage(weights))
            record["dates"].append(dates[start + window])

            if previous_weights[name] is not None:
                record["turnover"].append(turnover(weights, previous_weights[name]))
            previous_weights[name] = drift_weights(weights, holding)

    return {
        name: BacktestResult(
            name=name,
            daily_returns=np.array(r["returns"]),
            predicted_variances=np.array(r["predicted"]),
            realised_variances=np.array(r["realised"]),
            turnovers=np.array(r["turnover"]),
            concentrations=np.array(r["concentration"]),
            leverages=np.array(r["leverage"]),
            dates=r["dates"],
        )
        for name, r in records.items()
    }


def summarise(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Collect backtest summaries into one table, in the order given."""
    return pd.DataFrame({name: result.summary() for name, result in results.items()}).T


def breakeven_cost(result: BacktestResult, benchmark: BacktestResult) -> float:
    """One-way proportional cost, in basis points, at which a strategy ties the benchmark.

    Comparing against 1/N without this number would be misleading.  An optimised
    portfolio here turns over one to three times its own value a month while 1/N
    turns over about two percent, so any gross advantage it shows is bought with
    trading the benchmark never does.  The question worth answering is how
    expensive trading would have to get before the advantage disappeared.

    Model: a proportional cost c on one-way turnover, charged at each rebalance,
    so the annual drag is c times turnover times the number of rebalances per
    year.  Volatility is treated as unchanged, which is a good approximation
    because the cost is close to deterministic while the return is not.  Solving
    for the c at which the two net Sharpe ratios coincide gives a linear
    equation, since both numerators are linear in c.

    Returns infinity when the strategy cannot be caught, meaning it already has
    both a higher return and a lower volatility, and a negative number when it
    loses to the benchmark before any costs are charged at all.
    """
    a, b = result.summary(), benchmark.summary()
    per_year = TRADING_DAYS / REBALANCE_EVERY

    # Solve (a.return - c drag_a) / a.vol == (b.return - c drag_b) / b.vol for c.
    slope = (
        a["mean_turnover"] * per_year / a["annual_volatility"]
        - b["mean_turnover"] * per_year / b["annual_volatility"]
    )
    gap = a["annual_return"] / a["annual_volatility"] - b["annual_return"] / b["annual_volatility"]

    if slope <= 0:
        return float("inf") if gap > 0 else float("-inf")
    return 1e4 * gap / slope
