"""Generate all publication figures from cached results.

Run once after the pipeline completes:
    python -m src.visualization.make_figures

Writes PNG (300 dpi) and PDF to outputs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

mpl.rcParams.update({
    "savefig.dpi": 300, "figure.dpi": 110, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path}")
    plt.close(fig)


def fig1_prospectivity_map():
    """Continental prospectivity heat-map with discoveries + known fields overlaid."""
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    discoveries = pd.read_csv(ROOT / "outputs/results/consensus_discoveries.csv")
    fields = pd.read_csv(ROOT / "data/metadata/known_fields_details.csv")

    # Build 2-D grid for imshow
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid["row"].values, grid["col"].values] = scores

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [grid.lon.min(), grid.lon.max(), grid.lat.min(), grid.lat.max()]
    im = ax.imshow(img, extent=extent, origin="lower", aspect="auto",
                   cmap="magma_r", vmin=0, vmax=1)
    ax.scatter(fields.lon, fields.lat, s=10, marker="x", color="#3a86ff",
               linewidths=0.8, label=f"Known fields ({len(fields)})")
    ax.scatter(discoveries.lon, discoveries.lat, s=80, marker="o",
               facecolors="none", edgecolors="#ffd60a", linewidths=1.6,
               label=f"Consensus discoveries ({len(discoveries)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Continental geothermal prospectivity, western United States")
    ax.legend(loc="lower left", frameon=True, framealpha=0.85)
    cb = fig.colorbar(im, ax=ax, label="GeoProspectNet score (probability)",
                      shrink=0.85)
    _save(fig, "fig1_prospectivity_map")


def fig2_uncertainty_map():
    grid = pd.read_csv(ROOT / "data/processed/grid_coordinates.csv")
    std = np.load(ROOT / "outputs/results/prospectivity_std.npy")
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    img = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    img[grid["row"].values, grid["col"].values] = std

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [grid.lon.min(), grid.lon.max(), grid.lat.min(), grid.lat.max()]
    im = ax.imshow(img, extent=extent, origin="lower", aspect="auto",
                   cmap="viridis", vmin=0, vmax=float(np.percentile(std, 99)))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Epistemic uncertainty (MC-Dropout std)")
    fig.colorbar(im, ax=ax, label="σ across MC-Dropout passes", shrink=0.85)
    _save(fig, "fig2_uncertainty_map")


def fig3_cohort_distributions():
    """Score distribution by cohort — the discrimination plot."""
    scores = np.load(ROOT / "outputs/results/scores_cpu_max.npy")
    labels = np.load(ROOT / "data/processed/labels.npy")
    pos_idx = np.flatnonzero(labels == 1)

    # Get the cohort cell indices the same way negative_controls does
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

    # Hold-out
    ho = pd.read_csv(ROOT / "data/raw/labels/nv_permits_holdout.csv")
    novel = ho[ho.is_novel].reset_index(drop=True) if "is_novel" in ho.columns else ho
    tree = cKDTree(gxy)
    ho_idx = []
    for _, r in novel.iterrows():
        _, i = tree.query([r.lon * 111.32 * cos_lat0, r.lat * 111.32], k=1)
        ho_idx.append(i)
    ho_idx = np.array(ho_idx)

    # Convert raw probabilities to continental percentile rank — clearer
    # because the model is saturated near 1.0 for most western US cells.
    sort = np.argsort(scores)
    rank = np.empty_like(sort)
    rank[sort] = np.arange(len(scores))
    pct = 100.0 * rank / (len(scores) - 1)

    cohorts = {
        "Known positives\n(n=1370)":     pct[pos_idx],
        "Post-2008 hold-out\n(n=22)":     pct[ho_idx],
        "Random western US\n(n=100)":    pct[rand_idx],
        "NC1 Colorado Plateau\n(n=100)": pct[nc1_idx],
        "NC3 random deep\n(n=100)":      pct[nc3_idx],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#0a9396", "#005f73", "#ee9b00", "#bb3e03", "#9d0208"]
    positions = np.arange(len(cohorts))
    data = list(cohorts.values())
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 1.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    # Annotate medians
    for i, vals in enumerate(data):
        med = float(np.median(vals))
        ax.annotate(f"{med:.0f}", (positions[i], med),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color="white", weight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(list(cohorts.keys()), fontsize=9)
    ax.set_ylabel("Continental percentile rank")
    ax.set_ylim(-2, 102)
    ax.axhline(50, ls=":", color="grey", alpha=0.6, label="Grid median (50th)")
    ax.axhline(90, ls=":", color="#ee9b00", alpha=0.6, label="Top 10% threshold (90th)")
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("Score distribution by validation cohort (continental percentile rank)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig3_cohort_distributions")


def fig4_baselines_bar():
    bl = pd.read_csv(ROOT / "outputs/results/baselines.csv")
    bl = bl.set_index("method")
    order = [m for m in ["HeatFlowRule", "LogisticRegression", "RandomForest",
                          "XGBoost", "GeoProspectNet"] if m in bl.index]
    bl = bl.loc[order]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(order))
    ax[0].bar(x, bl["auroc"], color="#0a9396", alpha=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=20, ha="right")
    ax[0].set_ylabel("AUROC (test split)"); ax[0].set_ylim(0.5, 1.02)
    ax[0].set_title("In-distribution performance"); ax[0].grid(axis="y", alpha=0.3)
    ax[1].bar(x, bl["holdout_top10_capture"], color="#bb3e03", alpha=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=20, ha="right")
    ax[1].set_ylabel("Post-2008 hold-out: capture @ top 10%")
    ax[1].set_ylim(0, 1.05); ax[1].grid(axis="y", alpha=0.3)
    ax[1].set_title("Out-of-time generalisation")
    fig.suptitle("Baselines vs GeoProspectNet — generalisation gap exposed by temporal hold-out", fontsize=12)
    _save(fig, "fig4_baselines")


def fig5_calibration():
    cal = pd.read_csv(ROOT / "outputs/results/calibration_table.csv")
    summary = json.load(open(ROOT / "outputs/results/calibration_summary.json"))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    df = cal.dropna()
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.scatter(df.mean_pred, df.obs_freq, s=80 + df.n / 50,
               c="#bb3e03", alpha=0.8, edgecolor="white", zorder=10)
    for _, r in df.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.mean_pred, r.obs_freq),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean predicted probability per bin")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title(f"Calibration (Brier={summary['brier_score']:.4f}, ECE={summary['expected_calibration_error']:.4f})")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    _save(fig, "fig5_calibration")


def fig6_modality_importance():
    p = ROOT / "outputs/results/permutation_importance.csv"
    if not p.exists():
        print(f"  [skip fig6] {p} not found")
        return
    df = pd.read_csv(p).sort_values("delta_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(df.modality, df.delta_auc_mean, xerr=df.delta_auc_std,
            color="#0a9396", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("ΔAUROC when modality is permuted (mean ± std, 5 repeats)")
    ax.set_title("Modality permutation importance")
    ax.axvline(0, color="grey", lw=0.8); ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig6_modality_importance")


def fig7_mwe_histogram():
    # Prefer v2 Monte-Carlo file if present
    v2 = ROOT / "outputs/results/consensus_with_mwe_v2.csv"
    if v2.exists():
        mwe = pd.read_csv(v2)
        mwe = mwe.sort_values("mwe_p50", ascending=False).reset_index(drop=True)
        central = mwe.mwe_p50.values
        low = mwe.mwe_p10.values; high = mwe.mwe_p90.values
        method_label = "USGS Williams 2008 + T-dependent η, Monte Carlo P50"
    else:
        mwe = pd.read_csv(ROOT / "outputs/results/consensus_with_mwe.csv")
        mwe = mwe.sort_values("mwe_central", ascending=False).reset_index(drop=True)
        central = mwe.mwe_central.values
        low = mwe.mwe_low.values; high = mwe.mwe_high.values
        method_label = "USGS Williams 2008 volumetric (screening)"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].bar(np.arange(len(mwe)), central, color="#005f73", alpha=0.85,
              yerr=[central - low, high - central],
              error_kw={"alpha": 0.4, "lw": 0.7})
    ax[0].set_xlabel("Discovery rank")
    ax[0].set_ylabel("Recoverable resource (MWe)")
    ax[0].set_title(f"Per-discovery MWe (P10–P50–P90)\n{method_label}")
    ax[0].grid(axis="y", alpha=0.3)

    cum_c = np.cumsum(central)
    cum_lo = np.cumsum(low)
    cum_hi = np.cumsum(high)
    ax[1].plot(np.arange(len(mwe)), cum_c, color="#bb3e03", lw=2.0, label="P50")
    ax[1].fill_between(np.arange(len(mwe)), cum_lo, cum_hi,
                       alpha=0.18, color="#bb3e03", label="P10–P90")
    ax[1].axhline(3800, color="grey", ls="--", label="US installed capacity (2023)")
    ax[1].set_xlabel("Discoveries ordered by MWe")
    ax[1].set_ylabel("Cumulative MWe")
    ax[1].set_title(f"Cumulative recoverable resource  (P50 = {cum_c[-1]:,.0f} MWe)")
    ax[1].grid(alpha=0.3); ax[1].legend(loc="lower right")
    _save(fig, "fig7_mwe")


def fig8_separation_sweep():
    sw = pd.read_csv(ROOT / "outputs/results/separation_sweep.csv")
    sw["label"] = sw["config"].str.replace("cpu_", "")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cohorts = ["nc3_mean_pct", "nc1_mean_pct", "random_mean_pct",
               "pos_mean_pct", "holdout_mean_pct"]
    titles = ["NC3", "NC1", "Random", "Positives", "Hold-out"]
    colors = ["#9d0208", "#bb3e03", "#ee9b00", "#005f73", "#0a9396"]
    x = np.arange(len(sw))
    w = 0.15
    for i, (c, t, col) in enumerate(zip(cohorts, titles, colors)):
        ax.bar(x + (i - 2) * w, sw[c], width=w, color=col, label=t, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(sw["label"])
    ax.set_ylabel("Mean percentile rank")
    ax.set_title("Loss + data ablation: separation across cohorts")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 100)
    _save(fig, "fig8_ablation_separation")


def fig13_ood_eastern():
    p = ROOT / "outputs/results/ood_eastern_us_validation.csv"
    if not p.exists():
        print(f"  [skip fig13] {p} not found")
        return
    df = pd.read_csv(p).sort_values("percentile", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#0a9396" if "Princeton" in n else
              ("#52b788" if pct >= 75 else
                ("#f4a261" if pct >= 25 else "#bb3e03"))
              for n, pct in zip(df["name"], df.percentile)]
    y = np.arange(len(df))
    ax.barh(y, df.percentile, color=colors, alpha=0.9,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(df["name"], fontsize=9)
    ax.axvline(50, ls=":", color="grey", label="Median")
    ax.axvline(90, ls=":", color="#ee9b00", label="Top-10 %")
    ax.set_xlabel("Eastern-US grid percentile (n = 535 068 cells)")
    ax.set_xlim(0, 102)
    ax.set_title("OOD validation — known eastern hydrothermal / warm-spring systems")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "fig13_ood_eastern")


def fig12_arch_ablation():
    p = ROOT / "outputs/results/separation_sweep_arch_ablation.csv"
    if not p.exists():
        print(f"  [skip fig12] {p} not found")
        return
    df = pd.read_csv(p)
    # Add the full cpu_margin reference row for comparison
    full = {"config": "cpu_margin (full)", "holdout_mean_pct": 98.56,
            "separation_gap": 84.05}
    df = pd.concat([pd.DataFrame([full]), df[["config", "holdout_mean_pct",
                                                "separation_gap"]]],
                   ignore_index=True)
    df["label"] = df["config"].str.replace("cpu_margin_", "").str.replace("_", "-")
    df = df.sort_values("separation_gap", ascending=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#0a9396" if "full" in c else "#bb3e03" for c in df["config"]]
    x = np.arange(len(df))
    ax[0].barh(x, df.holdout_mean_pct, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.4)
    ax[0].set_yticks(x); ax[0].set_yticklabels(df["label"])
    ax[0].set_xlim(90, 100)
    ax[0].set_xlabel("Hold-out mean percentile"); ax[0].grid(axis="x", alpha=0.3)
    ax[0].set_title("Generalisation (post-2008 hold-out)")
    ax[1].barh(x, df.separation_gap, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.4)
    ax[1].set_yticks(x); ax[1].set_yticklabels(df["label"])
    ax[1].set_xlim(78, 86)
    ax[1].set_xlabel("Separation gap (positives − NC3)")
    ax[1].grid(axis="x", alpha=0.3)
    ax[1].set_title("Discrimination (separation gap)")
    fig.suptitle("Architecture ablation — only the contrastive loss measurably helps",
                  fontsize=12)
    _save(fig, "fig12_architecture_ablation")


def fig11_multi_seed():
    p = ROOT / "outputs/results/multi_seed_sensitivity.csv"
    if not p.exists():
        print(f"  [skip fig11] {p} not found")
        return
    df = pd.read_csv(p).sort_values("seed").reset_index(drop=True)
    metrics = [
        ("pos_pct", "Known positives mean pct", "#0a9396"),
        ("gap", "Separation gap (pos − NC3)", "#005f73"),
        ("holdout_mean_pct", "Hold-out mean pct", "#ee9b00"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title, color) in zip(axes, metrics):
        vals = df[col].values
        mu, sd = vals.mean(), vals.std()
        ax.bar(np.arange(len(df)), vals, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.4)
        ax.axhline(mu, ls="--", color="grey")
        ax.fill_between([-0.5, len(df) - 0.5], mu - sd, mu + sd,
                        color="grey", alpha=0.15)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels([f"s={int(s)}" for s in df.seed])
        ax.set_title(f"{title}\nμ = {mu:.2f}  σ = {sd:.2f}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("cpu_margin sensitivity to training seed (4 seeds)", fontsize=12)
    _save(fig, "fig11_multi_seed_sensitivity")


def fig10_mordensky_comparison():
    p = ROOT / "outputs/results/mordensky_comparison.csv"
    if not p.exists():
        print(f"  [skip fig10] {p} not found")
        return
    df = pd.read_csv(p)
    df = df.sort_values("holdout_top10_capture", ascending=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(df))
    colors = ["#0a9396" if m == "GeoProspectNet" else "#a98467" for m in df["method"]]
    ax[0].barh(x, df.holdout_top10_capture, color=colors, alpha=0.9)
    ax[0].set_yticks(x); ax[0].set_yticklabels(df["method"])
    ax[0].set_xlim(0.7, 1.05)
    ax[0].set_xlabel("Hold-out top-10 % capture")
    ax[0].set_title("Sensitivity (post-2008 hold-out)")
    ax[0].grid(axis="x", alpha=0.3)
    ax[1].barh(x, df.nc3_mean_pct, color=colors, alpha=0.9)
    ax[1].set_yticks(x); ax[1].set_yticklabels(df["method"])
    ax[1].set_xlabel("NC3 random-deep mean percentile (lower = better)")
    ax[1].set_title("Specificity (random deep cells)")
    ax[1].grid(axis="x", alpha=0.3)
    fig.suptitle("GeoProspectNet vs Mordensky 2023 — Great Basin sub-domain", fontsize=12)
    _save(fig, "fig10_mordensky_comparison")


def fig9_attention_by_province():
    p = ROOT / "outputs/results/attention_by_province.csv"
    if not p.exists():
        print(f"  [skip fig9] {p} not found")
        return
    df = pd.read_csv(p, index_col=0)
    # Rename thermal -> geology (when use_thermal=False the third slot is geology)
    df = df.rename(columns={"thermal": "geology"})
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(df.values, aspect="auto", cmap="cividis", vmin=0.25, vmax=0.40)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i,j]:.3f}", ha="center", va="center",
                    fontsize=10, color="white" if df.values[i,j] > 0.34 else "black")
    fig.colorbar(im, ax=ax, label="Attention weight", shrink=0.85)
    ax.set_title("Modality attention by tectonic province")
    _save(fig, "fig9_attention_by_province")


def main():
    print("Generating figures →", OUT)
    figs = [
        ("fig1_prospectivity_map", fig1_prospectivity_map),
        ("fig2_uncertainty_map", fig2_uncertainty_map),
        ("fig3_cohort_distributions", fig3_cohort_distributions),
        ("fig4_baselines", fig4_baselines_bar),
        ("fig5_calibration", fig5_calibration),
        ("fig6_modality_importance", fig6_modality_importance),
        ("fig7_mwe", fig7_mwe_histogram),
        ("fig8_separation_sweep", fig8_separation_sweep),
        ("fig9_attention_by_province", fig9_attention_by_province),
        ("fig10_mordensky_comparison", fig10_mordensky_comparison),
        ("fig11_multi_seed_sensitivity", fig11_multi_seed),
        ("fig12_architecture_ablation", fig12_arch_ablation),
        ("fig13_ood_eastern", fig13_ood_eastern),
    ]
    for name, fn in figs:
        try:
            print(f"-> {name}")
            fn()
        except Exception as e:
            print(f"   [error in {name}] {e}")
    print("done")


if __name__ == "__main__":
    main()
