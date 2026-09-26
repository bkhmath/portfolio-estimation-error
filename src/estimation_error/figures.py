"""Figures.

All static PNGs, sized to stay readable once a README scales them to the width
of a browser column.

The palette is Okabe-Ito, assigned to estimators in a fixed order so an
estimator keeps its colour across every figure; a colour that moves between
plots when the series list changes is worse than no colour at all.  The
ordering was checked against deuteranopia and tritanopia simulations, and the
worst adjacent pair separates by dE 9.6 under deuteranopia.  Series are also
legended or directly labelled, so identity never rests on colour alone.

No dual axes anywhere.  Two y-scales in one frame can be set to tell whatever
story the author likes, and the reader cannot see the choice being made.  Where
two quantities of different magnitude need comparing, they get two panels or a
ratio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rmt import mp_density, mp_edges, noise_variance

#: Okabe-Ito, ordered so that adjacent pairs stay separable under simulated
#: colour vision deficiency.  Do not reorder without rechecking.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]

#: Fixed colour per strategy, so identity is stable across every figure.
STRATEGY_COLOURS = {
    "sample": "#D55E00",
    "clipped": "#0072B2",
    "linear": "#009E73",
    "nonlinear": "#E69F00",
    "factor": "#CC79A7",
    "equal": "#56B4E9",
    "sample_long_only": "#666666",
}

INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"


def _style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive grid and axes, text in ink rather than in a series colour."""
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_mp_spectrum_simulated(eigenvalues_by_q: dict[float, np.ndarray], path: Path) -> Path:
    """Empirical spectra of iid Gaussian data against the Marchenko-Pastur density.

    The validation figure.  The true covariance is the identity, so the true
    spectrum is a single point mass at 1; everything visible here is estimation
    noise, and the curve is the closed form that predicts it.
    """
    qs = sorted(eigenvalues_by_q)
    fig, axes = plt.subplots(1, len(qs), figsize=(3.4 * len(qs), 3.1), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, q in zip(axes, qs, strict=True):
        lam = eigenvalues_by_q[q]
        lo, hi = mp_edges(q)
        ax.hist(
            lam,
            bins=50,
            density=True,
            color="#BBBBBB",
            edgecolor="white",
            linewidth=0.4,
            label="empirical",
        )
        grid = np.linspace(lo, hi, 600)
        ax.plot(
            grid, mp_density(grid, q), color=PALETTE[0], linewidth=2.0, label="Marchenko-Pastur"
        )
        ax.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
        _style_axes(ax, "eigenvalue", "density" if ax is axes[0] else "", f"q = {q:g}")
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "True covariance is the identity: every eigenvalue away from 1 is estimation noise",
        fontsize=11,
        color=INK,
        y=1.03,
    )
    return _save(fig, path)


def plot_mp_spectrum_real(eigenvalues: np.ndarray, q: float, label: str, path: Path) -> Path:
    """A real correlation spectrum against the Marchenko-Pastur bulk fitted to its noise.

    The bulk is drawn with sigma^2 fitted only to the eigenvalues that plausibly
    are noise, not to the whole spectrum; see `rmt.noise_variance`.  Eigenvalues
    above the upper edge are the ones carrying information.  The largest of
    them, the market mode, sits so far outside the frame that plotting it would
    squash everything else into a single bar, so it is annotated instead.
    """
    sigma2 = noise_variance(eigenvalues, q)
    lo, hi = mp_edges(q, sigma2)
    outliers = np.sort(eigenvalues[eigenvalues > hi])[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    view = eigenvalues[eigenvalues <= hi * 1.6]
    ax.hist(
        view,
        bins=45,
        density=True,
        color="#BBBBBB",
        edgecolor="white",
        linewidth=0.4,
        label="empirical",
    )
    grid = np.linspace(lo, hi, 600)
    ax.plot(
        grid,
        mp_density(grid, q, sigma2),
        color=PALETTE[0],
        linewidth=2.0,
        label=f"Marchenko-Pastur, fitted $\\sigma^2$ = {sigma2:.2f}",
    )
    ax.axvline(hi, color=PALETTE[1], linewidth=1.4, linestyle="--", label=f"upper edge {hi:.2f}")

    note = (
        f"{len(outliers)} eigenvalues above the edge carry signal\n"
        f"largest = {outliers[0]:.1f} (market mode, off scale)\n"
        f"next: " + ", ".join(f"{v:.1f}" for v in outliers[1:5])
    )
    ax.text(
        0.98,
        0.62,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.4"},
    )
    _style_axes(ax, "eigenvalue of the correlation matrix", "density", f"{label}  (q = {q:.2f})")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return _save(fig, path)


def plot_predicted_vs_realised(table, path: Path) -> Path:
    """The ex-ante / ex-post gap against q, with the exact finite-sample theory.

    The single clearest statement of the problem: the descending line is what
    the optimiser believes, the ascending line is what happens, and the shaded
    span between them is the self-deception.  Both dashed curves are closed
    forms, not fits.
    """
    q = table["q"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.fill_between(q, table["predicted_ratio"], table["realised_ratio"], color="#EEEEEE")
    ax.plot(
        q,
        table["realised_ratio"],
        "o-",
        color=PALETTE[1],
        linewidth=2.0,
        markersize=5,
        label="realised variance / true minimum",
    )
    ax.plot(
        q,
        table["predicted_ratio"],
        "o-",
        color=PALETTE[0],
        linewidth=2.0,
        markersize=5,
        label="variance the optimiser predicts / true minimum",
    )
    ax.plot(
        q,
        table["realised_theory"],
        "--",
        color=INK,
        linewidth=1.2,
        label="exact finite-sample theory",
    )
    ax.plot(q, table["predicted_theory"], "--", color=INK, linewidth=1.2)
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")

    ax.set_yscale("log")
    _style_axes(
        ax,
        "q = N / (T - 1)",
        "ratio to the true minimum variance",
        "The optimiser becomes more confident exactly as it becomes more wrong",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def plot_condition_number(table, path: Path) -> Path:
    """Condition number of the sample covariance against q, with the noise prediction."""
    q = table["q"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(
        q,
        table["condition_number"],
        "o-",
        color=PALETTE[0],
        linewidth=2.0,
        markersize=5,
        label="sample covariance of simulated factor returns",
    )
    ax.plot(
        q,
        ((1 + np.sqrt(q)) / (1 - np.sqrt(q))) ** 2,
        "--",
        color=INK,
        linewidth=1.2,
        label=r"pure-noise prediction $\left(\frac{1+\sqrt{q}}{1-\sqrt{q}}\right)^2$",
    )
    ax.set_yscale("log")
    _style_axes(
        ax, "q = N / (T - 1)", "condition number", "Inversion gets harder faster than linearly"
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def plot_estimator_error(table, path: Path) -> Path:
    """Frobenius distance to a known covariance, against sample size.

    Simulated only, because it is the one place the truth is available.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for name in table.columns:
        if name in ("T", "q"):
            continue
        ax.plot(
            table["T"],
            table[name],
            "o-",
            color=STRATEGY_COLOURS.get(name, MUTED),
            linewidth=2.0,
            markersize=5,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(
        ax,
        "observations T (N = 100 fixed)",
        "Frobenius distance to the true covariance",
        "Structure helps when data is scarce and hurts when it is not",
    )
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return _save(fig, path)


def _bar_panel(ax, summary, column: str, ylabel: str, title: str, scale: float = 1.0) -> None:
    names = list(summary.index)
    values = [summary.loc[n, column] * scale for n in names]
    colours = [STRATEGY_COLOURS.get(n, MUTED) for n in names]
    bars = ax.bar(range(len(names)), values, color=colours, width=0.68)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )
    _style_axes(ax, "", ylabel, title)
    ax.margins(y=0.16)


def plot_oos_volatility(summaries: dict[str, object], path: Path) -> Path:
    """Realised out-of-sample annualised volatility by estimator, one panel per dataset."""
    labels = list(summaries)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.6 * len(labels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        _bar_panel(
            ax,
            summaries[label],
            "annual_volatility",
            "annualised volatility (%)",
            label,
            scale=100.0,
        )
    fig.suptitle(
        "Realised out-of-sample volatility: lower is the whole objective",
        fontsize=11,
        color=INK,
        y=1.02,
    )
    return _save(fig, path)


def plot_turnover(summaries: dict[str, object], path: Path) -> Path:
    """One-way turnover per monthly rebalance, by estimator."""
    labels = list(summaries)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.6 * len(labels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        _bar_panel(ax, summaries[label], "mean_turnover", "one-way turnover per rebalance", label)
    fig.suptitle(
        "Turnover: the cost side, which the variance objective never sees",
        fontsize=11,
        color=INK,
        y=1.02,
    )
    return _save(fig, path)


def plot_concentration(summaries: dict[str, object], path: Path) -> Path:
    """Effective number of positions as a fraction of N, by estimator."""
    labels = list(summaries)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.6 * len(labels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        _bar_panel(
            ax,
            summaries[label],
            "mean_concentration",
            "effective positions / N",
            label,
        )
    fig.suptitle(
        "Weight concentration: 1.0 is equal weighting, small means betting on few directions",
        fontsize=11,
        color=INK,
        y=1.02,
    )
    return _save(fig, path)


def plot_jagannathan_ma(table, path: Path) -> Path:
    """Where the no-short constraint starts paying for itself.

    The constrained portfolio's realised volatility is nearly flat in q while
    the unconstrained one diverges, which is the shrinkage interpretation made
    visible: the constraint stops the optimiser acting on estimation error it
    cannot distinguish from signal.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        table["q"],
        table["sample"] * 100,
        "o-",
        color=STRATEGY_COLOURS["sample"],
        linewidth=2.0,
        markersize=5,
        label="unconstrained minimum variance",
    )
    ax.plot(
        table["q"],
        table["sample_long_only"] * 100,
        "o-",
        color=STRATEGY_COLOURS["sample_long_only"],
        linewidth=2.0,
        markersize=5,
        label="long-only minimum variance",
    )

    crossing = table.loc[(table["sample_long_only"] < table["sample"])]
    if len(crossing):
        q_star = float(crossing["q"].min())
        ax.axvline(q_star, color=MUTED, linewidth=1.0, linestyle=":")
        # Anchor the note to the bottom of the frame rather than the top: the
        # legend lives in the upper left and the two series cross in the middle,
        # so the floor is the only place nothing else is drawn.
        ax.annotate(
            f"constraint starts winning\nnear q = {q_star:.2f}",
            (q_star, ax.get_ylim()[0]),
            xytext=(7, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color=INK,
        )

    _style_axes(
        ax,
        "q = N / (T - 1), varied by shortening the estimation window",
        "realised out-of-sample volatility (%)",
        "Jagannathan-Ma: the wrong constraint helps, but only once q is large enough",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def plot_shrinkage_functions(
    sample_eigenvalues: np.ndarray,
    cleaned: dict[str, np.ndarray],
    path: Path,
) -> Path:
    """What each estimator does to each eigenvalue.

    Plotting the map from sample eigenvalue to cleaned eigenvalue shows the
    family relationship directly: clipping is a step function, linear shrinkage
    is a straight line pulled towards the mean, and nonlinear shrinkage is the
    smooth curve the other two are approximating.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    order = np.argsort(sample_eigenvalues)
    x = sample_eigenvalues[order]
    ax.plot(x, x, color=MUTED, linewidth=1.0, linestyle=":", label="no cleaning (identity)")
    for name, values in cleaned.items():
        ax.plot(
            x,
            values[order],
            "-",
            color=STRATEGY_COLOURS.get(name, MUTED),
            linewidth=2.0,
            label=name,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style_axes(
        ax,
        "sample eigenvalue",
        "eigenvalue after cleaning",
        "Clipping is a step-function approximation to the nonlinear shrinkage curve",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)
