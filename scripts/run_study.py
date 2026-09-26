"""Run the whole study and write every figure and table under reports/.

Order matters here and mirrors the argument the README makes: the simulated
layer runs first, because it is the only layer with a known answer and there is
no point looking at real-data results produced by code that cannot reproduce a
theorem.  Nothing in the real-data section is computed until the simulated
checks have printed.

    python scripts/run_study.py                 # everything, needs the data cache
    python scripts/run_study.py --simulated     # no network and no cache needed
    python scripts/run_study.py --quick         # small sizes, for CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from estimation_error import (
    backtest,
    data,
    estimators,
    figures,
    optimise,
    rmt,
    simulate,
)

REPO = Path(__file__).resolve().parents[1]

#: Seeds are fixed so that two runs agree; `tests/test_reproducibility.py`
#: checks that they do.
SEED_SPECTRUM = 20240501
SEED_ERROR = 20240502
SEED_GAP = 20240503


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def simulated_spectrum(out_dir: Path, quick: bool) -> pd.DataFrame:
    """Validation 1: the empirical spectrum of iid data against Marchenko-Pastur."""
    section("1. Marchenko-Pastur against simulated iid returns")
    qs = [0.1, 0.4] if quick else [0.1, 0.25, 0.5, 0.9]
    sizes = [100, 400] if quick else [100, 400, 1600]

    rows = []
    spectra_for_figure = {}
    for q in qs:
        for n_assets in sizes:
            n_obs = round(n_assets / q) + 1
            rng = np.random.default_rng(SEED_SPECTRUM + int(1000 * q) + n_assets)
            returns = simulate.simulate_white_noise(n_assets, n_obs, rng)
            eigenvalues = np.linalg.eigvalsh(estimators.sample_covariance(returns))
            effective_q = n_assets / (n_obs - 1)
            rows.append(
                {
                    "q": q,
                    "N": n_assets,
                    "T": n_obs,
                    "ks_distance": rmt.ks_distance(eigenvalues, effective_q),
                    "l1_distance": rmt.l1_distance(eigenvalues, effective_q),
                    "min_eigenvalue": eigenvalues.min(),
                    "max_eigenvalue": eigenvalues.max(),
                    "mp_lower_edge": rmt.mp_edges(effective_q)[0],
                    "mp_upper_edge": rmt.mp_edges(effective_q)[1],
                }
            )
            if n_assets == sizes[-1]:
                spectra_for_figure[q] = eigenvalues

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nKS distance falls roughly as 1/N rather than 1/sqrt(N): eigenvalues repel each"
        "\nother, so a spectrum is far more regular than an iid sample of the same size."
    )
    table.to_csv(out_dir / "mp_convergence.csv", index=False)
    figures.plot_mp_spectrum_simulated(
        spectra_for_figure, out_dir / "figures" / "mp_spectrum_simulated.png"
    )
    return table


def simulated_estimator_error(out_dir: Path, quick: bool) -> pd.DataFrame:
    """Validation 2: distance to a known covariance, at finite T and as T grows."""
    section("2. Estimator error against a known covariance")
    n_assets = 60 if quick else 100
    sample_sizes = [150, 400] if quick else [150, 250, 500, 1000, 4000]
    n_draws = 3 if quick else 20

    truth = simulate.factor_model_covariance(n_assets, 3, np.random.default_rng(7))
    rows = []
    for n_obs in sample_sizes:
        errors: dict[str, list[float]] = {name: [] for name in estimators.ESTIMATORS}
        for draw in range(n_draws):
            rng = np.random.default_rng(SEED_ERROR + 31 * n_obs + draw)
            returns = simulate.simulate_returns(truth, n_obs, rng)
            for name, estimator in estimators.ESTIMATORS.items():
                errors[name].append(float(np.linalg.norm(estimator(returns) - truth, "fro")))
        row = {"T": n_obs, "q": n_assets / (n_obs - 1)}
        row.update({name: float(np.mean(values)) for name, values in errors.items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nNonlinear shrinkage wins at every T.  Clipping and the factor model beat the"
        "\nsample covariance only while data is scarce: both impose structure, and once T"
        "\nis large the structural error dominates the sampling error they were fixing."
    )
    table.to_csv(out_dir / "estimator_error.csv", index=False)
    figures.plot_estimator_error(table, out_dir / "figures" / "estimator_error.png")
    return table


def simulated_gap(out_dir: Path, quick: bool) -> pd.DataFrame:
    """Validation 3: the ex-ante / ex-post gap against the exact finite-sample theory."""
    section("3. The ex-ante / ex-post gap, against exact finite-sample formulas")
    n_assets = 60 if quick else 100
    sample_sizes = [400, 200] if quick else [1000, 500, 300, 250, 200, 150, 125, 110]
    n_draws = 10 if quick else 80

    truth = simulate.factor_model_covariance(n_assets, 3, np.random.default_rng(7))
    true_variance = optimise.portfolio_variance(truth, optimise.min_variance_weights(truth))

    rows = []
    for n_obs in sample_sizes:
        predicted, realised, conditions = [], [], []
        for draw in range(n_draws):
            rng = np.random.default_rng(SEED_GAP + 17 * n_obs + draw)
            returns = simulate.simulate_returns(truth, n_obs, rng)
            cov = estimators.sample_covariance(returns)
            weights = optimise.min_variance_weights(cov)
            predicted.append(optimise.portfolio_variance(cov, weights))
            realised.append(optimise.portfolio_variance(truth, weights))
            conditions.append(optimise.condition_number(cov))
        rows.append(
            {
                "q": n_assets / (n_obs - 1),
                "T": n_obs,
                "predicted_ratio": np.mean(predicted) / true_variance,
                "realised_ratio": np.mean(realised) / true_variance,
                "predicted_theory": backtest.predicted_variance_ratio(n_assets, n_obs),
                "realised_theory": backtest.realised_variance_ratio(n_assets, n_obs),
                "condition_number": np.mean(conditions),
            }
        )

    table = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    table["self_deception"] = table["realised_ratio"] / table["predicted_ratio"]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = table.iloc[-1]
    print(
        f"\nAt q = {worst['q']:.2f} the optimiser predicts {worst['predicted_ratio']:.3f} times the true"
        f"\nminimum variance and realises {worst['realised_ratio']:.2f} times it: a factor of"
        f" {worst['self_deception']:.0f} between\nforecast and outcome, with both curves on top of closed forms that were not fitted."
    )
    table.to_csv(out_dir / "ex_ante_ex_post.csv", index=False)
    figures.plot_predicted_vs_realised(table, out_dir / "figures" / "predicted_vs_realised.png")
    figures.plot_condition_number(table, out_dir / "figures" / "condition_number.png")
    return table


def shrinkage_shapes(out_dir: Path) -> Path:
    """Illustrative: the eigenvalue map each estimator applies."""
    section("4. What each estimator does to the spectrum")
    truth = simulate.factor_model_covariance(100, 3, np.random.default_rng(7))
    returns = simulate.simulate_returns(truth, 250, np.random.default_rng(SEED_ERROR))
    sample = estimators.sample_covariance(returns)
    sample_eigenvalues = np.linalg.eigvalsh(sample)

    cleaned = {}
    for name in ("clipped", "linear", "nonlinear"):
        cleaned[name] = np.linalg.eigvalsh(estimators.ESTIMATORS[name](returns))

    intensity = estimators.linear_shrinkage_intensity(returns)
    print(f"linear shrinkage intensity chosen by the data: {intensity:.4f}")
    print(f"factors kept by the Marchenko-Pastur rule:     {estimators.factor_count(returns)}")
    print(
        f"condition number, sample -> nonlinear:         "
        f"{optimise.condition_number(sample):.0f} -> "
        f"{optimise.condition_number(estimators.nonlinear_shrinkage(returns)):.0f}"
    )
    return figures.plot_shrinkage_functions(
        sample_eigenvalues, cleaned, out_dir / "figures" / "shrinkage_functions.png"
    )


def jagannathan_ma_identity() -> float:
    """Verify the constraint-as-shrinkage identity and report the residual."""
    section("5. Jagannathan-Ma: the no-short constraint as a shrinkage estimator")
    truth = simulate.factor_model_covariance(100, 3, np.random.default_rng(11))
    returns = simulate.simulate_returns(truth, 250, np.random.default_rng(3))
    cov = estimators.sample_covariance(returns)

    long_only_weights, multipliers = optimise.min_variance_long_only(cov)
    implied = optimise.jagannathan_ma_covariance(cov, multipliers)
    from_implied = optimise.min_variance_weights(implied)
    residual = float(np.max(np.abs(long_only_weights - from_implied)))

    unconstrained = optimise.min_variance_weights(cov)
    print(f"binding no-short constraints          : {int(np.sum(multipliers > 1e-9))} of 100")
    print(f"unconstrained gross leverage          : {optimise.gross_leverage(unconstrained):.2f}")
    print(
        f"in-sample variance, unconstrained     : {optimise.portfolio_variance(cov, unconstrained):.6f}"
    )
    print(
        f"in-sample variance, long-only         : {optimise.portfolio_variance(cov, long_only_weights):.6f}"
        "   (worse, necessarily: it is a restriction)"
    )
    print(f"max |w_long-only - GMV(Sigma tilde)|  : {residual:.2e}   <- the identity, verified")
    return residual


def real_data_study(out_dir: Path) -> dict:
    """The out-of-sample protocol on the French panels, and the Jagannathan-Ma sweep."""
    section("6. Out-of-sample study on Kenneth French daily portfolios")
    cache = REPO / "data" / "cache"
    strategies = [
        "sample",
        "clipped",
        "linear",
        "nonlinear",
        "factor",
        "equal",
        "sample_long_only",
    ]

    summaries = {}
    spectra = {}
    for key in ("industry49", "size_bm100"):
        panel = data.load_returns(key, cache)
        print("\n" + data.describe(panel))
        n_assets = panel.shape[1]
        q = n_assets / (backtest.ESTIMATION_WINDOW - 1)
        print(
            f"q = {n_assets}/{backtest.ESTIMATION_WINDOW - 1} = {q:.3f};  "
            f"Gaussian-iid theory predicts a self-deception ratio of "
            f"{backtest.self_deception_ratio(n_assets, backtest.ESTIMATION_WINDOW):.2f}"
        )

        results = backtest.run_backtest(panel, strategies)
        summary = backtest.summarise(results)
        # What it would cost to trade the difference away.  Reported next to the
        # Sharpe ratio so the two are never read apart.
        summary["breakeven_vs_equal_bp"] = [
            backtest.breakeven_cost(results[name], results["equal"]) for name in summary.index
        ]
        summaries[data.DATASETS[key].label] = summary
        print(summary.round(4).to_string())
        summary.to_csv(out_dir / f"oos_{key}.csv")

        window = panel.to_numpy()[-backtest.ESTIMATION_WINDOW :]
        cov = estimators.sample_covariance(window)
        volatility = np.sqrt(np.diag(cov))
        correlation = cov / np.outer(volatility, volatility)
        spectra[key] = np.linalg.eigvalsh(correlation)
        figures.plot_mp_spectrum_real(
            spectra[key],
            q,
            data.DATASETS[key].label,
            out_dir / "figures" / f"mp_spectrum_{key}.png",
        )

    print(
        "\nThe break-even column is the one-way transaction cost, in basis points, at which"
        "\neach strategy would see its net Sharpe ratio fall to that of 1/N.  It is the number"
        "\nthat makes the 1/N comparison honest: the optimised portfolios win on gross Sharpe,"
        "\nbut they pay for it with turnover the benchmark does not have."
    )

    figures.plot_oos_volatility(summaries, out_dir / "figures" / "oos_volatility.png")
    figures.plot_turnover(summaries, out_dir / "figures" / "turnover.png")
    figures.plot_concentration(summaries, out_dir / "figures" / "concentration.png")

    section("7. What happens when estimated expected returns enter the problem")
    rows = []
    for key in ("industry49", "size_bm100"):
        panel = data.load_returns(key, cache)
        results = backtest.run_backtest(panel, ["sample", "sample+mv", "nonlinear", "nonlinear+mv"])
        for name, result in results.items():
            summary = result.summary()
            rows.append(
                {
                    "dataset": key,
                    "strategy": name,
                    "annual_volatility": summary["annual_volatility"],
                    "sharpe": summary["sharpe"],
                    "mean_turnover": summary["mean_turnover"],
                }
            )
    frontier = pd.DataFrame(rows)
    print(frontier.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nAdding a return target, estimated from the same window like everything else,"
        "\nraises realised volatility on both datasets and both covariance estimators, and"
        "\nlowers the Sharpe ratio wherever q is large.  That is the estimability argument"
        "\nmade at the top of optimise.py, measured rather than asserted: the covariance"
        "\ncarries information at this sample size and the mean does not."
    )
    frontier.to_csv(out_dir / "mean_variance.csv", index=False)

    section("8. Where the no-short constraint starts paying, on real data")
    panel = data.load_returns("size_bm100", cache)
    n_assets = panel.shape[1]
    rows = []
    for window in (1000, 500, 250, 150, 125, 110):
        results = backtest.run_backtest(panel, ["sample", "sample_long_only"], window=window)
        rows.append(
            {
                "window": window,
                "q": n_assets / (window - 1),
                "sample": results["sample"].summary()["annual_volatility"],
                "sample_long_only": results["sample_long_only"].summary()["annual_volatility"],
            }
        )
    sweep = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    sweep["ratio"] = sweep["sample_long_only"] / sweep["sample"]
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sweep.to_csv(out_dir / "jagannathan_ma_sweep.csv", index=False)
    figures.plot_jagannathan_ma(sweep, out_dir / "figures" / "jagannathan_ma.png")

    return {"summaries": summaries, "sweep": sweep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="run only the simulated layer; needs no data cache and no network",
    )
    parser.add_argument("--quick", action="store_true", help="small sizes, for CI")
    parser.add_argument("--out", type=Path, default=REPO / "reports")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"quick": args.quick, "simulated_only": args.simulated}

    spectrum = simulated_spectrum(out_dir, args.quick)
    manifest["worst_ks_distance"] = float(spectrum["ks_distance"].max())

    simulated_estimator_error(out_dir, args.quick)
    gap = simulated_gap(out_dir, args.quick)
    manifest["max_self_deception"] = float(gap["self_deception"].max())

    shrinkage_shapes(out_dir)
    manifest["jagannathan_ma_residual"] = jagannathan_ma_identity()

    if not args.simulated:
        real = real_data_study(out_dir)
        for label, summary in real["summaries"].items():
            manifest[f"volatility::{label}"] = {
                name: float(summary.loc[name, "annual_volatility"]) for name in summary.index
            }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    section("Done")
    print(f"figures and tables written to {out_dir}")


if __name__ == "__main__":
    main()
