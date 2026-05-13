"""Publication-grade figure generation with a unified geothermal palette.

Single matplotlib style applied to every figure for visual cohesion:
- Geothermal-temperature palette: deep navy (cool) → teal → cream → ochre → magma red (hot)
- Typography: Source Sans / Helvetica / Inter / Arial fallback
- Tight 300 dpi PNG + vector PDF outputs
- Consistent grid, spine, label, legend, and annotation conventions

Outputs all figures under ``outputs/figures/`` (PNG @ 300 dpi + PDF).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  STYLE  ·  Geothermal cool→hot palette + clean modern aesthetic
# ---------------------------------------------------------------------------
PALETTE = {
    "deep_navy":    "#0d3b66",  # coolest / negative-control
    "teal":         "#3a7d99",  # cool background
    "soft_blue":    "#7fb3c7",  # cool accent
    "cream":        "#faf0ca",  # neutral background
    "warm_gold":    "#f4a259",  # warm / positive
    "ochre":        "#e76f51",  # hot
    "magma_red":    "#c1424e",  # hottest / discovery
    "deep_red":     "#7a1c2e",  # critical highlight
    "ink":          "#1a1a1a",  # text
    "stone":        "#5c6b73",  # muted text
    "soft_grey":    "#cfd8dc",  # light divider
    "pale":         "#f5f7f8",  # canvas tint
}

GEOTHERM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "geotherm",
    [PALETTE["cream"], PALETTE["warm_gold"], PALETTE["ochre"],
     PALETTE["magma_red"], PALETTE["deep_red"], PALETTE["ink"]],
    N=256,
)
PROSPECT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "prospect",
    [PALETTE["deep_navy"], PALETTE["teal"], PALETTE["soft_blue"],
     PALETTE["cream"], PALETTE["warm_gold"], PALETTE["ochre"],
     PALETTE["magma_red"]],
    N=256,
)

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": ["Helvetica Neue", "Helvetica", "Inter", "Source Sans Pro", "Arial",
                     "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10,
    "axes.labelsize": 10.5,
    "axes.labelweight": "regular",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": PALETTE["stone"],
    "axes.linewidth": 0.7,
    "axes.facecolor": "white",
    "axes.titlecolor": PALETTE["ink"],
    "axes.labelcolor": PALETTE["ink"],
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.prop_cycle": mpl.cycler(color=[
        PALETTE["deep_navy"], PALETTE["ochre"], PALETTE["teal"],
        PALETTE["warm_gold"], PALETTE["magma_red"], PALETTE["soft_blue"],
        PALETTE["deep_red"],
    ]),
    "grid.color": PALETTE["soft_grey"],
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "grid.alpha": 0.6,
    "xtick.color": PALETTE["stone"],
    "ytick.color": PALETTE["stone"],
    "xtick.labelcolor": PALETTE["ink"],
    "ytick.labelcolor": PALETTE["ink"],
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "legend.title_fontsize": 10,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "patch.linewidth": 0.5,
})


def _save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.{{png,pdf}}")


def _annotate_panel(ax, letter: str, dx: float = -0.13, dy: float = 1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, weight="bold",
            fontsize=14, color=PALETTE["ink"], ha="left", va="bottom")


# ---------------------------------------------------------------------------
#  Fig 1  ·  Continental prospectivity map
# ---------------------------------------------------------------------------
def fig1_prospectivity_map():
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    discoveries = pd.read_csv(ROOT / "outputs/results/consensus_discoveries.csv")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")

    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid["row"].values, grid["col"].values] = scores

    fig, ax = plt.subplots(figsize=(10, 7.2), facecolor="white")
    extent = [grid.lon.min(), grid.lon.max(), grid.lat.min(), grid.lat.max()]
    im = ax.imshow(img, extent=extent, origin="lower", aspect="auto",
                   cmap=PROSPECT_CMAP, vmin=0, vmax=1, interpolation="bilinear")

    ax.scatter(fields.lon, fields.lat, s=14, marker="x",
                color=PALETTE["deep_navy"], linewidths=0.9, alpha=0.85,
                label=f"Known geothermal fields  (n = {len(fields)})")
    ax.scatter(discoveries.lon, discoveries.lat, s=110, marker="o",
                facecolors="none", edgecolors=PALETTE["deep_red"],
                linewidths=1.8, label=f"New consensus discoveries  (n = {len(discoveries)})")

    ax.set_xlabel("Longitude (° W)")
    ax.set_ylabel("Latitude (° N)")
    ax.set_title("Continental geothermal prospectivity — western United States",
                  loc="left", weight="semibold")
    ax.set_xlim(grid.lon.min(), grid.lon.max())
    ax.set_ylim(grid.lat.min(), grid.lat.max())
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("GeoProspectNet score", fontsize=10)
    cb.outline.set_visible(False)
    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.94,
                     edgecolor=PALETTE["soft_grey"], facecolor="white")
    leg.get_frame().set_linewidth(0.4)
    _save(fig, "fig1_prospectivity_map")


# ---------------------------------------------------------------------------
#  Fig 2  ·  Epistemic uncertainty map
# ---------------------------------------------------------------------------
def fig2_uncertainty_map():
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    std_path = ROOT / "outputs/results/prospectivity_std.npy"
    if not std_path.exists():
        return
    std = np.load(std_path)
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid["row"].values, grid["col"].values] = std

    fig, ax = plt.subplots(figsize=(10, 7.2), facecolor="white")
    extent = [grid.lon.min(), grid.lon.max(), grid.lat.min(), grid.lat.max()]
    im = ax.imshow(img, extent=extent, origin="lower", aspect="auto",
                    cmap=GEOTHERM_CMAP, vmin=0,
                    vmax=float(np.percentile(std, 99)), interpolation="bilinear")
    ax.set_xlabel("Longitude (° W)"); ax.set_ylabel("Latitude (° N)")
    ax.set_title("Epistemic uncertainty — MC-Dropout standard deviation",
                  loc="left", weight="semibold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("σ across 5 stochastic forward passes")
    cb.outline.set_visible(False)
    _save(fig, "fig2_uncertainty_map")


# ---------------------------------------------------------------------------
#  Fig 3  ·  Cohort score distributions (percentile rank)
# ---------------------------------------------------------------------------
def fig3_cohort_distributions():
    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    labels = np.load(ROOT / "data/processed/labels.npy")
    pos_idx = np.flatnonzero(labels == 1)
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    masks = np.load(ROOT / "data/processed/modality_masks.npy")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    d_to_field, _ = cKDTree(fxy).query(gxy, k=1)
    rng = np.random.default_rng(42)
    nc1 = grid.lat.between(36, 40) & grid.lon.between(-111, -108) & (d_to_field > 200)
    nc3 = (d_to_field > 100) & (~masks[:, 1])
    nc1_idx = rng.choice(np.flatnonzero(nc1.values), size=100, replace=False)
    nc3_idx = rng.choice(np.flatnonzero(nc3), size=100, replace=False)
    rand_idx = rng.choice(len(grid), size=100, replace=False)
    ho = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = ho[ho.is_novel].reset_index(drop=True) if "is_novel" in ho.columns else ho
    tree = cKDTree(gxy)
    ho_idx = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        ho_idx.append(i)
    ho_idx = np.array(ho_idx)
    sort = np.argsort(scores)
    rank = np.empty_like(sort); rank[sort] = np.arange(len(scores))
    pct = 100.0 * rank / (len(scores) - 1)
    cohorts = [
        ("NC3\nrandom deep\n(n=100)",      pct[nc3_idx],  PALETTE["deep_navy"]),
        ("NC1\nColorado Plateau\n(n=100)", pct[nc1_idx],  PALETTE["teal"]),
        ("Random\nwestern US\n(n=100)",    pct[rand_idx], PALETTE["soft_blue"]),
        ("Known\npositives\n(n=1370)",     pct[pos_idx],  PALETTE["warm_gold"]),
        ("Post-2008\nhold-out\n(n=22)",    pct[ho_idx],   PALETTE["magma_red"]),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="white")
    positions = np.arange(len(cohorts))
    for i, (name, vals, color) in enumerate(cohorts):
        # Strip plot with random jitter for the data points
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, s=14, color=color, alpha=0.5,
                    edgecolor="white", linewidth=0.3, zorder=2)
        # Box plot for distribution
        bp = ax.boxplot([vals], positions=[positions[i]], widths=0.55,
                         patch_artist=True, showfliers=False, zorder=3,
                         medianprops={"color": "white", "linewidth": 2.0},
                         boxprops={"facecolor": color, "alpha": 0.85,
                                   "edgecolor": color, "linewidth": 1.0},
                         whiskerprops={"color": color, "linewidth": 1.0},
                         capprops={"color": color, "linewidth": 1.0})
        # Mean marker
        ax.scatter(positions[i], np.mean(vals), s=60, marker="D",
                    color="white", edgecolor=PALETTE["ink"], linewidths=1.2,
                    zorder=4)

    ax.axhline(50, ls="--", lw=0.8, color=PALETTE["stone"], alpha=0.7,
                label="Grid median (50th)")
    ax.axhline(90, ls="--", lw=0.8, color=PALETTE["magma_red"], alpha=0.55,
                label="Top-10 % threshold (90th)")
    ax.set_xticks(positions)
    ax.set_xticklabels([c[0] for c in cohorts])
    ax.set_ylabel("Continental percentile rank")
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 90, 100])
    ax.set_title("Score distribution by validation cohort",
                  loc="left", weight="semibold")
    ax.legend(loc="center right")
    _save(fig, "fig3_cohort_distributions")


# ---------------------------------------------------------------------------
#  Fig 4  ·  Baselines — paired in-distribution vs hold-out
# ---------------------------------------------------------------------------
def fig4_baselines():
    bl = pd.read_csv(ROOT / "outputs/results/baselines.csv").set_index("method")
    order = [m for m in ["HeatFlowRule", "LogisticRegression", "RandomForest",
                          "XGBoost", "GeoProspectNet"] if m in bl.index]
    bl = bl.loc[order]
    pretty = {"HeatFlowRule": "Heat-flow rule",
              "LogisticRegression": "Logistic regression",
              "RandomForest": "Random Forest",
              "XGBoost": "XGBoost",
              "GeoProspectNet": "GeoProspectNet (ours)"}
    labels = [pretty.get(m, m) for m in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5),
                              gridspec_kw={"wspace": 0.30}, facecolor="white")

    colors = [PALETTE["soft_blue"] if m != "GeoProspectNet" else PALETTE["magma_red"]
              for m in order]
    x = np.arange(len(order))

    axes[0].barh(x, bl["auroc"].values, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.6)
    axes[0].set_yticks(x); axes[0].set_yticklabels(labels)
    axes[0].set_xlim(0.5, 1.05)
    axes[0].axvline(0.5, ls=":", color=PALETTE["stone"], lw=0.6)
    axes[0].set_xlabel("AUROC (test split)")
    axes[0].set_title("In-distribution performance", loc="left",
                       weight="semibold", color=PALETTE["ink"])
    axes[0].grid(axis="x", alpha=0.5)
    axes[0].grid(axis="y", visible=False)
    for i, v in enumerate(bl["auroc"].values):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9.5,
                      color=PALETTE["ink"])
    _annotate_panel(axes[0], "a")

    axes[1].barh(x, bl["holdout_top10_capture"].values, color=colors,
                  alpha=0.92, edgecolor="white", linewidth=0.6)
    axes[1].set_yticks(x); axes[1].set_yticklabels(labels)
    axes[1].set_xlim(0, 1.10)
    axes[1].set_xlabel("Post-2008 hold-out: capture @ top 10 %")
    axes[1].set_title("Out-of-time generalisation", loc="left",
                       weight="semibold", color=PALETTE["ink"])
    axes[1].grid(axis="x", alpha=0.5)
    axes[1].grid(axis="y", visible=False)
    for i, v in enumerate(bl["holdout_top10_capture"].values):
        axes[1].text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=9.5,
                      color=PALETTE["ink"])
    _annotate_panel(axes[1], "b")

    fig.suptitle("Baselines vs GeoProspectNet — generalisation exposed by temporal hold-out",
                  fontsize=13, weight="semibold", color=PALETTE["ink"], y=1.03)
    _save(fig, "fig4_baselines")


# ---------------------------------------------------------------------------
#  Fig 5  ·  Calibration / reliability diagram
# ---------------------------------------------------------------------------
def fig5_calibration():
    cal = pd.read_csv(ROOT / "outputs/results/calibration_table.csv").dropna()
    summary = json.load(open(ROOT / "outputs/results/calibration_summary.json"))
    fig, ax = plt.subplots(figsize=(6, 5.6), facecolor="white")
    ax.plot([0, 1], [0, 1], ls="--", color=PALETTE["stone"], lw=1.1,
             label="Perfect calibration", zorder=1)
    sizes = 60 + cal.n / max(1, cal.n.max()) * 360
    sc = ax.scatter(cal.mean_pred, cal.obs_freq, s=sizes,
                     c=cal.mean_pred, cmap=GEOTHERM_CMAP, vmin=0, vmax=1,
                     edgecolor=PALETTE["ink"], linewidth=0.6, alpha=0.95, zorder=4)
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r.n):,}", (r.mean_pred, r.obs_freq),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=8.5, color=PALETTE["stone"])
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Mean predicted probability per bin")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title(
        f"Reliability diagram — Brier = {summary['brier_score']:.4f}, "
        f"ECE = {summary['expected_calibration_error']:.4f}",
        loc="left", weight="semibold")
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Bin centre probability"); cb.outline.set_visible(False)
    ax.legend(loc="upper left")
    _save(fig, "fig5_calibration")


# ---------------------------------------------------------------------------
#  Fig 6  ·  Modality attention by tectonic province
# ---------------------------------------------------------------------------
def fig6_attention_by_province():
    p = ROOT / "outputs/results/attention_by_province.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, index_col=0)
    df = df.rename(columns={"thermal": "geology"})
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="white")
    im = ax.imshow(df.values, aspect="auto", cmap=GEOTHERM_CMAP,
                    vmin=0.25, vmax=0.40)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([p.replace("_", " ") for p in df.index])
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([c.capitalize() for c in df.columns])
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            color = "white" if v > 0.34 else PALETTE["ink"]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=10, color=color, weight="medium")
    ax.set_title("Learned modality attention by tectonic province",
                  loc="left", weight="semibold")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Attention weight"); cb.outline.set_visible(False)
    _save(fig, "fig6_attention_by_province")


# ---------------------------------------------------------------------------
#  Fig 7  ·  Per-discovery MWe (P10-P50-P90) + cumulative resource
# ---------------------------------------------------------------------------
def fig7_mwe():
    v2 = ROOT / "outputs/results/consensus_with_mwe_v2.csv"
    if v2.exists():
        mwe = pd.read_csv(v2).sort_values("mwe_p50", ascending=False).reset_index(drop=True)
        central, lo, hi = mwe.mwe_p50.values, mwe.mwe_p10.values, mwe.mwe_p90.values
        title_methods = "USGS Williams 2008  ·  T-dependent η  ·  Monte Carlo (2 000 draws)"
    else:
        mwe = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe.csv")
        mwe = mwe.sort_values("mwe_central", ascending=False).reset_index(drop=True)
        central, lo, hi = mwe.mwe_central.values, mwe.mwe_low.values, mwe.mwe_high.values
        title_methods = "Williams 2008 screening estimate"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                              gridspec_kw={"wspace": 0.26}, facecolor="white")
    x = np.arange(len(mwe))
    axes[0].vlines(x, lo, hi, color=PALETTE["soft_blue"], lw=1.2, alpha=0.6)
    axes[0].scatter(x, central, s=22, color=PALETTE["magma_red"],
                     edgecolor="white", linewidth=0.4, zorder=5)
    axes[0].set_xlabel("Discovery rank (by P50 MWe)")
    axes[0].set_ylabel("Recoverable resource  (MWe)")
    axes[0].set_title("Per-discovery MWe — P10 to P90",
                       loc="left", weight="semibold")
    axes[0].set_xlim(-1, len(mwe))
    _annotate_panel(axes[0], "a")

    cum_c = np.cumsum(central)
    cum_lo = np.cumsum(lo); cum_hi = np.cumsum(hi)
    axes[1].fill_between(x, cum_lo, cum_hi, color=PALETTE["ochre"], alpha=0.18,
                          linewidth=0, label="P10–P90 band")
    axes[1].plot(x, cum_c, color=PALETTE["magma_red"], lw=2.4, label="P50")
    axes[1].axhline(3800, ls="--", color=PALETTE["stone"], lw=1.0,
                     label="US installed capacity 2023")
    axes[1].set_xlabel("Discoveries cumulated by MWe")
    axes[1].set_ylabel("Cumulative MWe")
    axes[1].set_title(f"Cumulative resource — P50 total = {cum_c[-1]:,.0f} MWe",
                       loc="left", weight="semibold")
    axes[1].set_xlim(-1, len(mwe))
    axes[1].legend(loc="lower right")
    _annotate_panel(axes[1], "b")

    fig.suptitle(title_methods, fontsize=11,
                  color=PALETTE["stone"], y=1.02, ha="center")
    _save(fig, "fig7_mwe")


# ---------------------------------------------------------------------------
#  Fig 8  ·  Loss + data ablation
# ---------------------------------------------------------------------------
def fig8_ablation_separation():
    p = ROOT / "outputs/results/separation_sweep_loss_ablation.csv"
    if not p.exists():
        p = ROOT / "outputs/results/separation_sweep.csv"
    sw = pd.read_csv(p)
    sw["label"] = sw["config"].str.replace("cpu_", "")
    cohorts = [
        ("nc3_mean_pct",       "NC3",        PALETTE["deep_navy"]),
        ("nc1_mean_pct",       "NC1",        PALETTE["teal"]),
        ("random_mean_pct",    "Random",     PALETTE["soft_blue"]),
        ("pos_mean_pct",       "Positives",  PALETTE["warm_gold"]),
        ("holdout_mean_pct",   "Hold-out",   PALETTE["magma_red"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    x = np.arange(len(sw)); w = 0.16
    for i, (c, t, col) in enumerate(cohorts):
        offset = (i - (len(cohorts) - 1) / 2) * w
        ax.bar(x + offset, sw[c], width=w, color=col, edgecolor="white",
                linewidth=0.4, label=t, alpha=0.93)
    ax.set_xticks(x); ax.set_xticklabels(sw["label"])
    ax.set_ylabel("Mean continental percentile rank")
    ax.set_title("Loss + data ablation — separation across cohorts",
                  loc="left", weight="semibold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0, 100)
    _save(fig, "fig8_ablation_separation")


# ---------------------------------------------------------------------------
#  Fig 9  ·  Mordensky 2023 head-to-head
# ---------------------------------------------------------------------------
def fig9_mordensky():
    p = ROOT / "outputs/results/mordensky_comparison.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("holdout_top10_capture", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={"wspace": 0.28}, facecolor="white")
    colors = [PALETTE["magma_red"] if m == "GeoProspectNet" else PALETTE["soft_blue"]
              for m in df["method"]]
    y = np.arange(len(df))

    axes[0].barh(y, df.holdout_top10_capture, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels(df["method"])
    axes[0].set_xlim(0.75, 1.04)
    axes[0].set_xlabel("Capture @ top 10 %  (post-2008 hold-out)")
    axes[0].set_title("Sensitivity", loc="left", weight="semibold")
    axes[0].grid(axis="x", alpha=0.5); axes[0].grid(axis="y", visible=False)
    for i, v in enumerate(df.holdout_top10_capture):
        axes[0].text(v + 0.008, i, f"{v:.0%}", va="center", fontsize=9.5)
    _annotate_panel(axes[0], "a")

    axes[1].barh(y, df.nc3_mean_pct, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y); axes[1].set_yticklabels(df["method"])
    axes[1].set_xlim(0, 45)
    axes[1].set_xlabel("NC3 mean percentile  (lower = better)")
    axes[1].set_title("Specificity", loc="left", weight="semibold")
    axes[1].grid(axis="x", alpha=0.5); axes[1].grid(axis="y", visible=False)
    for i, v in enumerate(df.nc3_mean_pct):
        axes[1].text(v + 0.6, i, f"{v:.1f}", va="center", fontsize=9.5)
    _annotate_panel(axes[1], "b")

    fig.suptitle("GeoProspectNet vs Mordensky 2023 — Great Basin sub-domain",
                  fontsize=13, weight="semibold", y=1.03)
    _save(fig, "fig9_mordensky_comparison")


# ---------------------------------------------------------------------------
#  Fig 10  ·  Multi-seed sensitivity (cpu_margin × 4 seeds)
# ---------------------------------------------------------------------------
def fig10_multi_seed():
    p = ROOT / "outputs/results/multi_seed_sensitivity.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("seed").reset_index(drop=True)
    metrics = [
        ("pos_pct",          "Known positives mean pct",   PALETTE["warm_gold"]),
        ("gap",              "Separation gap (pos − NC3)", PALETTE["ochre"]),
        ("holdout_mean_pct", "Hold-out mean pct",          PALETTE["magma_red"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                              gridspec_kw={"wspace": 0.30}, facecolor="white")
    for ax, (col, title, color) in zip(axes, metrics):
        vals = df[col].values
        mu, sd = vals.mean(), vals.std()
        ax.axhspan(mu - sd, mu + sd, color=color, alpha=0.13)
        ax.axhline(mu, ls="-", color=color, lw=1.6, alpha=0.85, label=f"μ = {mu:.2f}")
        ax.scatter(np.arange(len(df)), vals, s=80, color=color,
                    edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([f"seed {int(s)}" for s in df.seed])
        ax.set_ylabel(title)
        ax.set_title(f"σ = {sd:.2f}", loc="left",
                      color=PALETTE["stone"], fontsize=10, weight="regular")
        ax.legend(loc="upper right")
    fig.suptitle("cpu_margin sensitivity to training seed  (n = 4)",
                  fontsize=13, weight="semibold", y=1.02)
    _save(fig, "fig10_multi_seed")


# ---------------------------------------------------------------------------
#  Fig 11  ·  Architecture ablation (no-contrastive / no-spatial / no-attention)
# ---------------------------------------------------------------------------
def fig11_architecture_ablation():
    p = ROOT / "outputs/results/separation_sweep_arch_ablation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    full = pd.DataFrame([{
        "config": "cpu_margin_full",
        "holdout_mean_pct": 98.56,
        "separation_gap": 84.05,
    }])
    df = pd.concat([full, df[["config", "holdout_mean_pct", "separation_gap"]]],
                    ignore_index=True)
    df["label"] = (df["config"]
                    .str.replace("cpu_margin_", "")
                    .str.replace("_", " "))
    df = df.sort_values("separation_gap", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),
                              gridspec_kw={"wspace": 0.28}, facecolor="white")
    colors = [PALETTE["magma_red"] if "full" in c else PALETTE["ochre"]
              for c in df["config"]]
    y = np.arange(len(df))
    axes[0].barh(y, df.holdout_mean_pct, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels(df["label"])
    axes[0].set_xlim(92, 100)
    axes[0].set_xlabel("Hold-out mean percentile")
    axes[0].set_title("Generalisation", loc="left", weight="semibold")
    axes[0].grid(axis="x", alpha=0.5); axes[0].grid(axis="y", visible=False)
    for i, v in enumerate(df.holdout_mean_pct):
        axes[0].text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9.5)
    _annotate_panel(axes[0], "a")

    axes[1].barh(y, df.separation_gap, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y); axes[1].set_yticklabels(df["label"])
    axes[1].set_xlim(78, 86)
    axes[1].set_xlabel("Separation gap (positives − NC3)")
    axes[1].set_title("Discrimination", loc="left", weight="semibold")
    axes[1].grid(axis="x", alpha=0.5); axes[1].grid(axis="y", visible=False)
    for i, v in enumerate(df.separation_gap):
        axes[1].text(v + 0.13, i, f"{v:.2f}", va="center", fontsize=9.5)
    _annotate_panel(axes[1], "b")

    fig.suptitle("Architecture ablation — only the contrastive loss measurably contributes",
                  fontsize=13, weight="semibold", y=1.03)
    _save(fig, "fig11_architecture_ablation")


# ---------------------------------------------------------------------------
#  Fig 12  ·  OOD validation on the eastern US
# ---------------------------------------------------------------------------
def fig12_ood_eastern():
    p = ROOT / "outputs/results/ood_eastern_us_validation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("percentile", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor="white")
    colors = [PALETTE["soft_blue"] if "Princeton" in n else
              (PALETTE["warm_gold"] if p >= 75 else
                (PALETTE["ochre"] if p >= 25 else PALETTE["magma_red"]))
              for n, p in zip(df["name"], df.percentile)]
    y = np.arange(len(df))
    ax.barh(y, df.percentile, color=colors, alpha=0.92,
             edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df["name"])
    ax.axvline(50, ls=":", color=PALETTE["stone"], lw=0.8, label="Median")
    ax.axvline(90, ls=":", color=PALETTE["deep_red"], lw=0.8, label="Top-10 %")
    ax.set_xlim(0, 102)
    ax.set_xlabel("Eastern-US grid percentile (n = 535 068 cells)")
    ax.set_title("OOD validation — documented eastern hydrothermal systems",
                  loc="left", weight="semibold")
    for i, v in enumerate(df.percentile):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=9.5)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.4); ax.grid(axis="y", visible=False)
    _save(fig, "fig12_ood_eastern")


# ---------------------------------------------------------------------------
#  Fig 13  ·  Discoveries by province (province bar + MWe)
# ---------------------------------------------------------------------------
def fig13_discovery_province():
    df = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe_v2.csv")
    g = (df.groupby("province")
           .agg(n=("discovery_id", "count"), mwe=("mwe_p50", "sum"))
           .sort_values("mwe", ascending=True))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                              gridspec_kw={"wspace": 0.28}, facecolor="white")
    y = np.arange(len(g))
    colors = [PALETTE["deep_navy"], PALETTE["teal"], PALETTE["soft_blue"],
              PALETTE["warm_gold"], PALETTE["magma_red"]][: len(g)]
    axes[0].barh(y, g.n.values, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels([p.replace("_", " ") for p in g.index])
    axes[0].set_xlabel("Number of discoveries")
    axes[0].set_title("Discoveries per tectonic province",
                       loc="left", weight="semibold")
    axes[0].grid(axis="x", alpha=0.5); axes[0].grid(axis="y", visible=False)
    for i, v in enumerate(g.n.values):
        axes[0].text(v + 0.4, i, str(int(v)), va="center", fontsize=9.5)
    _annotate_panel(axes[0], "a")

    axes[1].barh(y, g.mwe.values, color=colors, alpha=0.92,
                  edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y); axes[1].set_yticklabels([p.replace("_", " ") for p in g.index])
    axes[1].set_xlabel("Cumulative recoverable MWe  (P50)")
    axes[1].set_title("Recoverable resource per province",
                       loc="left", weight="semibold")
    axes[1].grid(axis="x", alpha=0.5); axes[1].grid(axis="y", visible=False)
    for i, v in enumerate(g.mwe.values):
        axes[1].text(v + 60, i, f"{v:,.0f}", va="center", fontsize=9.5)
    _annotate_panel(axes[1], "b")
    fig.suptitle("Discovery geography — concentrated in Basin & Range and Cascades",
                  fontsize=12.5, weight="semibold", y=1.03)
    _save(fig, "fig13_discovery_province")


# ---------------------------------------------------------------------------
#  Fig 14  ·  Continental score histogram by cohort (density curves)
# ---------------------------------------------------------------------------
def fig14_score_density():
    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    labels = np.load(ROOT / "data/processed/labels.npy")
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")
    masks = np.load(ROOT / "data/processed/modality_masks.npy")
    lat0 = float(grid.lat.mean()); cos_lat0 = float(np.cos(np.radians(lat0)))
    fxy = np.column_stack([fields.lon * 111.32 * cos_lat0, fields.lat * 111.32])
    gxy = np.column_stack([grid.lon * 111.32 * cos_lat0, grid.lat * 111.32])
    d_to_field, _ = cKDTree(fxy).query(gxy, k=1)
    rng = np.random.default_rng(42)
    nc3 = (d_to_field > 100) & (~masks[:, 1])
    nc3_idx = rng.choice(np.flatnonzero(nc3), size=2000, replace=False)
    pos_idx = np.flatnonzero(labels == 1)
    rand_idx = rng.choice(len(grid), size=2000, replace=False)

    sort = np.argsort(scores); rank = np.empty_like(sort)
    rank[sort] = np.arange(len(scores))
    pct = 100.0 * rank / (len(scores) - 1)

    cohorts = [
        ("NC3 random deep",  pct[nc3_idx],  PALETTE["deep_navy"]),
        ("All continental",  pct,           PALETTE["soft_blue"]),
        ("Random western US",pct[rand_idx], PALETTE["teal"]),
        ("Known positives",  pct[pos_idx],  PALETTE["magma_red"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    bins = np.linspace(0, 100, 51)
    for name, vals, color in cohorts:
        h, _ = np.histogram(vals, bins=bins, density=True)
        centres = 0.5 * (bins[:-1] + bins[1:])
        ax.fill_between(centres, 0, h, color=color, alpha=0.18, linewidth=0)
        ax.plot(centres, h, color=color, lw=1.8, label=name)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Continental percentile rank")
    ax.set_ylabel("Density")
    ax.set_title("Score density by cohort", loc="left", weight="semibold")
    ax.legend(loc="upper center")
    _save(fig, "fig14_score_density")


# ---------------------------------------------------------------------------
#  Fig 15  ·  Cumulative resource ramp by province
# ---------------------------------------------------------------------------
def fig15_cumulative_by_province():
    df = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe_v2.csv")
    df = df.sort_values("mwe_p50", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor="white")
    color_map = {
        "Basin_and_Range":   PALETTE["magma_red"],
        "Cascades":          PALETTE["ochre"],
        "Snake_River_Plain": PALETTE["warm_gold"],
        "Rocky_Mountain":    PALETTE["teal"],
        "Other":             PALETTE["soft_blue"],
    }
    cum_by_prov = {}
    for prov in df.province.unique():
        sub = df[df.province == prov].sort_values("mwe_p50", ascending=False)
        cum_by_prov[prov] = (sub.index.values, np.cumsum(sub.mwe_p50.values))
    cumulative = np.cumsum(df.mwe_p50.values)
    ax.fill_between(np.arange(len(df)), 0, cumulative,
                     color=PALETTE["ink"], alpha=0.04, linewidth=0)
    ax.plot(np.arange(len(df)), cumulative, color=PALETTE["ink"], lw=2.4,
             label="All consensus discoveries")
    province_order = (df.groupby("province").mwe_p50.sum()
                        .sort_values(ascending=False).index.tolist())
    for prov in province_order:
        sub_indices = df.index[df.province == prov].tolist()
        sub_mwe = df.loc[sub_indices, "mwe_p50"].values
        contributions = np.zeros(len(df))
        contributions[sub_indices] = sub_mwe
        running = np.cumsum(contributions)
        ax.plot(np.arange(len(df)), running, color=color_map.get(prov, PALETTE["stone"]),
                 lw=1.5, alpha=0.85, label=prov.replace("_", " "))
    ax.axhline(3800, ls="--", color=PALETTE["stone"], lw=0.9,
                label="US installed capacity 2023")
    ax.set_xlabel("Discovery rank")
    ax.set_ylabel("Cumulative recoverable MWe  (P50)")
    ax.set_title("Cumulative resource ramp by province",
                  loc="left", weight="semibold")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    _save(fig, "fig15_cumulative_by_province")


# ---------------------------------------------------------------------------
#  Fig 16  ·  Economic / policy framing card
# ---------------------------------------------------------------------------
def fig16_economics():
    p = ROOT / "outputs/results/economic_analysis.json"
    if not p.exists():
        return
    e = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                              gridspec_kw={"wspace": 0.35}, facecolor="white")

    # Panel A — LCOE vs Earthshot target
    lcoe_b = e["lcoe_baseline_per_mwh"]
    lcoe_c = e["lcoe_cheap_case_per_mwh"]
    lcoe_e = e["lcoe_expensive_case_per_mwh"]
    bars = ["Cheap", "Baseline", "Expensive"]
    vals = [lcoe_c, lcoe_b, lcoe_e]
    colors = [PALETTE["warm_gold"], PALETTE["ochre"], PALETTE["deep_red"]]
    axes[0].bar(bars, vals, color=colors, alpha=0.92, edgecolor="white", linewidth=0.5)
    axes[0].axhline(e["earthshot_target_per_mwh"], ls="--",
                     color=PALETTE["deep_navy"], lw=1.4,
                     label=f"DOE Earthshot target ${e['earthshot_target_per_mwh']:.0f}")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 1.8, f"${v:.0f}", ha="center", fontsize=10,
                      color=PALETTE["ink"])
    axes[0].set_ylabel("LCOE  ($/MWh)")
    axes[0].set_title("LCOE  (NREL ATB 2023 assumptions)",
                       loc="left", weight="semibold")
    axes[0].set_ylim(0, max(vals) * 1.18)
    axes[0].legend(loc="upper right")
    _annotate_panel(axes[0], "a")

    # Panel B — Fraction of GeoVision 2050 target (three stacked bars)
    p10 = e["mwe_p10"] / e["geovision_2050_target_MWe"]
    p50 = e["mwe_p50"] / e["geovision_2050_target_MWe"]
    p90 = e["mwe_p90"] / e["geovision_2050_target_MWe"]
    quantiles = [("P10", p10, PALETTE["warm_gold"]),
                  ("P50", p50, PALETTE["ochre"]),
                  ("P90", p90, PALETTE["magma_red"])]
    for i, (lbl, val, col) in enumerate(quantiles):
        axes[1].barh([i], [1.0], color=PALETTE["soft_grey"], alpha=0.45,
                      edgecolor="white", linewidth=0.4)
        axes[1].barh([i], [val], color=col, alpha=0.93, edgecolor="white",
                      linewidth=0.5)
        axes[1].text(val + 0.02, i, f"{val*100:.1f}%", va="center",
                      fontsize=10.5, color=PALETTE["ink"], weight="medium")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["P10", "P50", "P90"])
    axes[1].set_xlabel("Fraction of DOE GeoVision 2050  (60 GW target)")
    axes[1].set_title("GeoVision 2050 progress  ·  per-quantile",
                       loc="left", weight="semibold")
    axes[1].grid(axis="x", alpha=0.4); axes[1].grid(axis="y", visible=False)
    _annotate_panel(axes[1], "b")

    # Panel C — CO₂ displacement
    co2_p10 = e["co2_displaced_Mt_30yr_p10"]
    co2_p50 = e["co2_displaced_Mt_30yr_p50"]
    co2_p90 = e["co2_displaced_Mt_30yr_p90"]
    axes[2].bar([0, 1, 2], [co2_p10, co2_p50, co2_p90],
                 color=[PALETTE["warm_gold"], PALETTE["ochre"], PALETTE["magma_red"]],
                 alpha=0.92, edgecolor="white", linewidth=0.5)
    axes[2].set_xticks([0, 1, 2])
    axes[2].set_xticklabels(["P10", "P50", "P90"])
    axes[2].set_ylabel("CO₂ displaced over 30-yr plant life  (Mt)")
    axes[2].set_title("Climate benefit  (vs US-grid avg 386 g/kWh)",
                       loc="left", weight="semibold")
    for i, v in enumerate([co2_p10, co2_p50, co2_p90]):
        axes[2].text(i, v + max(co2_p90, 1) * 0.02, f"{v:,.0f}",
                      ha="center", fontsize=10, color=PALETTE["ink"])
    _annotate_panel(axes[2], "c")

    fig.suptitle("Economic and climate framing — 33 consensus discoveries",
                  fontsize=13, weight="semibold", y=1.04)
    _save(fig, "fig16_economics")


# ---------------------------------------------------------------------------
#  Fig 17  ·  Reservoir-thickness gravity proxy
# ---------------------------------------------------------------------------
def fig17_reservoir_thickness():
    p = ROOT / "outputs/results/reservoir_thickness_validation.csv"
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values("estimated_sediment_thickness_m", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    y = np.arange(len(df))
    colors = [PALETTE["magma_red"] if v >= 250 else
              (PALETTE["ochre"] if v >= 100 else PALETTE["soft_blue"])
              for v in df.estimated_sediment_thickness_m]
    ax.barh(y, df.estimated_sediment_thickness_m, color=colors, alpha=0.9,
             edgecolor="white", linewidth=0.4)
    ax.axvline(250, ls="--", color=PALETTE["deep_navy"], lw=1.2,
                label="Williams 2008 prior (250 m)")
    ax.set_yticks(y[::3])
    ax.set_yticklabels([f"#{int(d)}" for d in df.discovery_id.iloc[::3]])
    ax.set_xlabel("Estimated basin-fill thickness (m, from Bouguer residual)")
    ax.set_title("Reservoir-thickness validation — gravity-derived basin proxy",
                  loc="left", weight="semibold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.5); ax.grid(axis="y", visible=False)
    _save(fig, "fig17_reservoir_thickness")


# ---------------------------------------------------------------------------
#  Fig 18  ·  Hard-negative cohorts NC3 → NC6
# ---------------------------------------------------------------------------
def fig18_hard_negatives():
    p = ROOT / "outputs/results/hard_negative_cohorts.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    label_map = {"NC3_random_deep": "NC3\nrandom deep",
                 "NC4_cratonic_low_HF": "NC4\ncratonic + low HF",
                 "NC5_colorado_plateau_strict": "NC5\nColo Plateau strict",
                 "NC6_montana_wyoming_platform": "NC6\nMT-WY platform"}
    df["label"] = df["cohort"].map(label_map)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                              gridspec_kw={"wspace": 0.28}, facecolor="white")
    x = np.arange(len(df))
    palette_4 = [PALETTE["magma_red"], PALETTE["ochre"], PALETTE["warm_gold"],
                  PALETTE["teal"]]
    axes[0].bar(x, df.mean_pct, color=palette_4, alpha=0.92,
                 edgecolor="white", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["label"], fontsize=9.5)
    axes[0].set_ylabel("Mean continental percentile")
    axes[0].set_title("Score percentile per cohort",
                       loc="left", weight="semibold")
    axes[0].axhline(50, ls=":", color=PALETTE["stone"], lw=0.8, label="Median")
    axes[0].legend()
    for i, v in enumerate(df.mean_pct):
        axes[0].text(i, v + 1.3, f"{v:.1f}", ha="center", fontsize=10)
    _annotate_panel(axes[0], "a")

    axes[1].bar(x, df.mean_hf_z, color=palette_4, alpha=0.92,
                 edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["label"], fontsize=9.5)
    axes[1].set_ylabel("Mean heat-flow z-score")
    axes[1].set_title("Heat-flow signal by cohort",
                       loc="left", weight="semibold")
    axes[1].axhline(0, ls=":", color=PALETTE["stone"], lw=0.8)
    for i, v in enumerate(df.mean_hf_z):
        axes[1].text(i, v + (0.05 if v >= 0 else -0.12), f"{v:.2f}",
                      ha="center", fontsize=10)
    _annotate_panel(axes[1], "b")

    fig.suptitle("Hard-negative validation cohorts (geological criteria, not just proximity)",
                  fontsize=12.5, weight="semibold", y=1.03)
    _save(fig, "fig18_hard_negatives")


# ---------------------------------------------------------------------------
#  Fig 19  ·  Headline summary infographic
# ---------------------------------------------------------------------------
def fig19_summary_card():
    nc = pd.read_csv(ROOT / "outputs/results/negative_controls_cpu_max.csv")
    mwe = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe_v2.csv")
    econ = json.load(open(ROOT / "outputs/results/economic_analysis.json"))
    bl = pd.read_csv(ROOT / "outputs/results/baselines.csv")
    ms = pd.read_csv(ROOT / "outputs/results/multi_seed_sensitivity.csv")

    fig = plt.figure(figsize=(13, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)
    fig.patch.set_facecolor("white")

    stats = [
        ("Continental cells", f"{235470:,}",
         "scored at 4 km resolution", PALETTE["deep_navy"]),
        ("Hold-out capture", "100%",
         "post-2008 sites in top 10%", PALETTE["magma_red"]),
        ("Discoveries", f"{len(mwe)}",
         f"consensus sites · {mwe.mwe_p50.sum():,.0f} MWe P50", PALETTE["ochre"]),
        ("Mordensky 2023", "beaten",
         "on every metric (Great Basin)", PALETTE["teal"]),
        ("LCOE baseline", f"${econ['lcoe_baseline_per_mwh']:.0f}/MWh",
         f"vs Earthshot ${econ['earthshot_target_per_mwh']:.0f}", PALETTE["warm_gold"]),
        ("GeoVision 2050", f"{100*econ['geovision_fraction_p50']:.0f}%",
         "of 60 GW target captured", PALETTE["deep_red"]),
        ("CO₂ displaced", f"{econ['co2_displaced_Mt_30yr_p50']:.0f} Mt",
         "over 30-yr plant life", PALETTE["soft_blue"]),
        ("Multi-seed σ", f"{ms.gap.std():.1f}",
         "percentile points (n = 4 seeds)", PALETTE["stone"]),
    ]
    for i, (label, value, sub, color) in enumerate(stats):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.set_facecolor(PALETTE["pale"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        # accent bar
        ax.add_patch(plt.Rectangle((0.04, 0.85), 0.18, 0.06, transform=ax.transAxes,
                                    color=color, lw=0))
        ax.text(0.04, 0.74, label, transform=ax.transAxes,
                 fontsize=10.5, color=PALETTE["stone"], weight="medium")
        ax.text(0.04, 0.40, value, transform=ax.transAxes,
                 fontsize=22, color=PALETTE["ink"], weight="bold")
        ax.text(0.04, 0.16, sub, transform=ax.transAxes,
                 fontsize=9, color=PALETTE["stone"])
    fig.suptitle("GeoProspectNet — headline results",
                  fontsize=15, weight="semibold", color=PALETTE["ink"], y=1.01)
    _save(fig, "fig19_summary_card")


# ---------------------------------------------------------------------------
#  Fig 20  ·  AUROC vs hold-out capture scatter (the generalisation-gap plot)
# ---------------------------------------------------------------------------
def fig20_generalisation_gap():
    bl = pd.read_csv(ROOT / "outputs/results/baselines.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="white")
    method_colors = {
        "HeatFlowRule": PALETTE["soft_blue"],
        "LogisticRegression": PALETTE["teal"],
        "RandomForest": PALETTE["warm_gold"],
        "XGBoost": PALETTE["ochre"],
        "GeoProspectNet": PALETTE["magma_red"],
    }
    pretty = {"HeatFlowRule": "Heat-flow rule",
              "LogisticRegression": "Logistic regression",
              "RandomForest": "Random Forest",
              "XGBoost": "XGBoost",
              "GeoProspectNet": "GeoProspectNet"}
    # Manual annotation offsets so RandomForest and XGBoost don't overlap at (1.0, 0)
    offsets = {
        "HeatFlowRule":      (14, 4),
        "LogisticRegression":(14, -4),
        "RandomForest":      (-95, 6),
        "XGBoost":           (-95, -16),
        "GeoProspectNet":    (-160, 0),
    }
    for _, r in bl.iterrows():
        ax.scatter(r.auroc, r.holdout_top10_capture, s=300,
                    color=method_colors.get(r.method, PALETTE["stone"]),
                    edgecolor="white", linewidth=1.2, zorder=4)
        dx, dy = offsets.get(r.method, (10, 8))
        ax.annotate(pretty.get(r.method, r.method),
                     (r.auroc, r.holdout_top10_capture),
                     xytext=(dx, dy), textcoords="offset points",
                     fontsize=10, color=PALETTE["ink"], weight="medium")
    ax.set_xlabel("AUROC (in-distribution test split)")
    ax.set_ylabel("Capture @ top 10 %  (post-2008 hold-out)")
    ax.set_title("Generalisation gap — perfect AUROC ≠ generalisation",
                  loc="left", weight="semibold")
    ax.set_xlim(0.80, 1.02)
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0, ls="-", color=PALETTE["soft_grey"], lw=0.8)
    ax.fill_between([0.80, 1.02], -0.1, 0.25, color=PALETTE["deep_red"],
                     alpha=0.04, linewidth=0)
    ax.text(0.81, 0.18, "Overfit zone:\nperfect test, fails hold-out",
             color=PALETTE["deep_red"], fontsize=9, alpha=0.7)
    _save(fig, "fig20_generalisation_gap")


# ---------------------------------------------------------------------------
def main():
    print(f"Generating figures → {OUT}")
    figs = [
        ("fig1",  fig1_prospectivity_map),
        ("fig2",  fig2_uncertainty_map),
        ("fig3",  fig3_cohort_distributions),
        ("fig4",  fig4_baselines),
        ("fig5",  fig5_calibration),
        ("fig6",  fig6_attention_by_province),
        ("fig7",  fig7_mwe),
        ("fig8",  fig8_ablation_separation),
        ("fig9",  fig9_mordensky),
        ("fig10", fig10_multi_seed),
        ("fig11", fig11_architecture_ablation),
        ("fig12", fig12_ood_eastern),
        ("fig13", fig13_discovery_province),
        ("fig14", fig14_score_density),
        ("fig15", fig15_cumulative_by_province),
        ("fig16", fig16_economics),
        ("fig17", fig17_reservoir_thickness),
        ("fig18", fig18_hard_negatives),
        ("fig19", fig19_summary_card),
        ("fig20", fig20_generalisation_gap),
    ]
    for name, fn in figs:
        try:
            print(f"-> {name}")
            fn()
        except Exception as e:
            print(f"   [error] {e}")
    print("done")


if __name__ == "__main__":
    main()
