"""HRP Artemis II — publication-grade figures for arXiv + LinkedIn.

Produces four standalone figures plus one composite panel:
  Fig. A — Headline: per-modality LOSO accuracy with 95% CrI bars
  Fig. B — Inter-individual variability (latent trajectories per subject)
  Fig. C — Feature-efficiency frontier (n_features vs accuracy)
  Fig. D — Calibration of Bayesian predictions
  Fig. ALL — 2x2 composite panel suitable as paper Fig.1 + LinkedIn header

Colour palette: NASA-deep-space inspired
  - Mars red:    #c0392b
  - Space blue:  #2980b9
  - Crew green:  #27ae60
  - Caution:     #e67e22
  - Neutral grey: #7f8c8d
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

ROOT = Path(r"C:\Users\engim\marte\HRP_Artemis_II")
PUB = ROOT / "publicacion"
FIG = PUB / "figures_publicacion"
FIG.mkdir(exist_ok=True)

# Brand colours
MARS = "#c0392b"
SPACE = "#2980b9"
CREW = "#27ae60"
CAUTION = "#e67e22"
NEUTRAL = "#7f8c8d"
INK = "#1a1a2e"
BG = "white"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": BG,
})

demo = json.loads((PUB / "_demo_metrics_2026-05-09.json").read_text(encoding='utf-8'))
bayes = json.loads((PUB / "_bayesian_metrics_2026-05-09.json").read_text(encoding='utf-8'))
v3 = json.loads((PUB / "_v3_multimodal_metrics_2026-05-09.json").read_text(encoding='utf-8'))


# =============================================================================
# Fig. A — Headline: per-modality LOSO accuracy, 95% CrI
# =============================================================================
def fig_a_headline():
    rows = [
        ("Urine cytokines",        "Urine Alamar 200-plex (OSD-656)",     22, 406, 0.688, 0.471, 0.836, MARS),
        ("Cardiovascular",         "EvePanel cardiovascular (OSD-575)",   28, 18,  0.679, 0.492, 0.821, MARS),
        ("Serum cytokines",        "Alamar serum 200-plex (OSD-575)",     27, 406, 0.625, 0.441, 0.785, SPACE),
        ("HRV (terrestrial)",      "MMASH HRV proxy (PhysioNet, n=22)",   22, 9,   0.636, 0.427, 0.803, NEUTRAL),
        ("Metabolic (CMP)",        "Comprehensive metabolic panel",        28, 57,  0.571, 0.389, 0.736, SPACE),
        ("Immune (Eve)",           "EvePanel immune (OSD-575)",           28, 142, 0.571, 0.389, 0.736, SPACE),
        ("Complete blood count",   "CBC (OSD-569)",                        28, 60,  0.571, 0.389, 0.736, SPACE),
        ("Microbiome pathways",    "Pathway abundances CPM (OSD-572)",     189, 567, 0.406, 0.340, 0.479, CAUTION),
        ("Microbiome taxonomy",    "Metaphlan species (OSD-572)",          319, 2723, 0.404, 0.352, 0.459, CAUTION),
    ]
    rows.sort(key=lambda r: r[4])

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ys = np.arange(len(rows))
    accs = [r[4] for r in rows]
    los = [r[5] for r in rows]
    his = [r[6] for r in rows]
    cols = [r[7] for r in rows]
    labels = [f"{r[0]}\n  n={r[2]}, p={r[3]}" for r in rows]

    for y, acc, lo, hi, c in zip(ys, accs, los, his, cols):
        ax.plot([lo, hi], [y, y], color=c, linewidth=2.5, alpha=0.55)
        ax.plot(acc, y, "o", color=c, markersize=11, markeredgecolor="white", markeredgewidth=1.5, zorder=3)

    chance_2 = 0.50
    chance_3 = 0.333
    ax.axvline(chance_2, color="#bbb", linestyle=":", linewidth=1, zorder=0)
    ax.text(chance_2, len(rows)-0.3, "  chance (binary)", color="#888", fontsize=8, va="bottom")
    ax.axvline(chance_3, color="#ddd", linestyle=":", linewidth=1, zorder=0)
    ax.text(chance_3, 0.7, "  chance (3-class)", color="#aaa", fontsize=8, va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("LOSO classification accuracy with 95% Bayesian credible interval")
    ax.set_xlim(0.20, 1.00)
    ax.set_xticks(np.arange(0.2, 1.01, 0.1))
    ax.set_title("Per-modality leave-one-subject-out classification of spaceflight phase\n"
                 "Inspiration4 cohort (n=4 civilian crew); kNN baseline + Beta-Binomial 95% CrI",
                 fontsize=11, pad=10)

    legend = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=MARS, markersize=10, label="Top discriminators (binary)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=SPACE, markersize=10, label="Other binary modalities"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=NEUTRAL, markersize=10, label="Terrestrial proxy"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=CAUTION, markersize=10, label="3-class (microbiome)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.18, linestyle="-", linewidth=0.5)

    fig.text(0.99, 0.005, "Puerta Ángulo · Methodology demonstration · NASA OSDR + PhysioNet · CC BY 4.0",
             ha="right", fontsize=7, color="#999")
    fig.savefig(FIG / "fig_A_headline_accuracy.png", dpi=200)
    plt.close(fig)
    print("Wrote fig_A_headline_accuracy.png")


# =============================================================================
# Fig. B — Inter-individual variability (latent trajectories)
# =============================================================================
def fig_b_trajectories():
    master = pd.read_csv(PUB / "_v3_master_table.csv")
    common = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82"]
    subjects = ["C001", "C002", "C003", "C004"]
    subj_colors = {"C001": SPACE, "C002": CAUTION, "C003": CREW, "C004": MARS}
    subj_marker = {"C001": "o", "C002": "s", "C003": "^", "C004": "D"}

    feat_cols = [c for c in master.columns if c not in ("subject", "timepoint", "spaceflight")]
    means = master[feat_cols].mean()
    stds = master[feat_cols].std().replace(0, 1.0)
    Z = (master[feat_cols] - means) / stds
    Z = Z.fillna(0).to_numpy()

    valid_idx = master["timepoint"].isin(common) & master["subject"].isin(subjects)
    Z_v = Z[valid_idx]
    meta_v = master[valid_idx][["subject", "timepoint"]].reset_index(drop=True)
    Z_v_centered = Z_v - Z_v.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Z_v_centered, full_matrices=False)
    pc = U[:, :2] * S[:2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    ax1 = axes[0]
    for s in subjects:
        sub_idx = meta_v["subject"] == s
        if not sub_idx.any():
            continue
        sub_meta = meta_v[sub_idx].copy()
        sub_meta["tp_order"] = sub_meta["timepoint"].map(lambda t: common.index(t) if t in common else -1)
        order = sub_meta.sort_values("tp_order").index
        ax1.plot(pc[order, 0], pc[order, 1], "-",
                 color=subj_colors[s], alpha=0.4, linewidth=1.5, zorder=1)
        for i in order:
            tp = meta_v.loc[i, "timepoint"]
            phase = "pre" if tp.startswith("L-") else "post"
            ax1.scatter(pc[i, 0], pc[i, 1],
                        marker=subj_marker[s],
                        c=subj_colors[s],
                        s=130 if phase == "post" else 90,
                        edgecolor="white", linewidth=1.6, zorder=3,
                        alpha=0.95)
    ax1.set_xlabel("PC1 (multi-modal variance)")
    ax1.set_ylabel("PC2")
    ax1.set_title("Each crew member follows a distinct multi-modal trajectory", fontsize=11, pad=8)
    ax1.axhline(0, color="#ddd", linewidth=0.5, zorder=0)
    ax1.axvline(0, color="#ddd", linewidth=0.5, zorder=0)
    ax1.grid(alpha=0.15)
    legend = [Line2D([0],[0], marker=subj_marker[s], color="w", markerfacecolor=subj_colors[s],
                     markersize=10, label=s) for s in subjects]
    legend.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=7,  label="pre-flight (smaller)"))
    legend.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=11, label="post-flight (larger)"))
    ax1.legend(handles=legend, loc="best", fontsize=8, ncol=2)

    ax2 = axes[1]
    pc1_traj = {}
    for s in subjects:
        sub_idx = meta_v["subject"] == s
        sub_meta = meta_v[sub_idx].reset_index(drop=True)
        sub_pc = pc[meta_v.index[sub_idx]]
        order = [common.index(t) if t in common else -1 for t in sub_meta["timepoint"]]
        sorted_pairs = sorted(zip(order, sub_pc[:, 0]))
        xs = [p[0] for p in sorted_pairs if p[0] >= 0]
        ys = [p[1] for p in sorted_pairs if p[0] >= 0]
        pc1_traj[s] = (xs, ys)
        ax2.plot(xs, ys, marker=subj_marker[s], color=subj_colors[s], linewidth=2,
                 markersize=8, label=s, alpha=0.9, markeredgecolor="white", markeredgewidth=1.2)
    ax2.set_xticks(range(len(common)))
    ax2.set_xticklabels(common, rotation=0)
    ax2.axvline(2.5, color="#888", linestyle="--", linewidth=1)
    ax2.text(2.5, ax2.get_ylim()[1]*0.92, "  LAUNCH", color="#666", fontsize=9, va="top")
    ax2.set_xlabel("Timepoint (sols from launch)")
    ax2.set_ylabel("PC1")
    ax2.set_title("Per-subject longitudinal evolution", fontsize=11, pad=8)
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.15)

    fig.suptitle("Inter-individual variability is detectable with n=4 (multi-modal PCA)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.99, 0.005, "Inspiration4 cohort · 7 modalities × 6 timepoints · 24 obs × 1110 features",
             ha="right", fontsize=7, color="#999")
    fig.savefig(FIG / "fig_B_intersubject_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_B_intersubject_trajectories.png")


# =============================================================================
# Fig. C — Feature efficiency frontier
# =============================================================================
def fig_c_efficiency():
    pts = [
        ("Cardiovascular",       18,   0.679, MARS,    True),
        ("HRV (MMASH)",          9,    0.636, NEUTRAL, False),
        ("Metabolic CMP",        57,   0.571, SPACE,   False),
        ("CBC",                  60,   0.571, SPACE,   False),
        ("Immune EvePanel",      142,  0.571, SPACE,   False),
        ("Serum Alamar 200",     406,  0.625, SPACE,   False),
        ("Urine Alamar 200",     406,  0.688, MARS,    True),
        ("Pathway CPM",          567,  0.406, CAUTION, False),
        ("Microbiome taxa",      2723, 0.404, CAUTION, False),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    texts = []
    for name, p, acc, col, hi in pts:
        ax.scatter(p, acc, s=180 if hi else 90, c=col,
                   edgecolor="white", linewidth=1.6, zorder=3, alpha=0.95)
        texts.append(ax.text(p, acc, f"{name} (p={p})",
                             fontsize=9, color=col,
                             fontweight="bold" if hi else "normal",
                             zorder=4))

    ax.set_xscale("log")
    ax.axhline(0.50, color="#ccc", linestyle=":", linewidth=1, zorder=0)
    ax.text(2, 0.50, "  binary chance", color="#888", fontsize=8, va="bottom", zorder=0)
    ax.axhline(0.333, color="#ddd", linestyle=":", linewidth=1, zorder=0)
    ax.text(2, 0.333, "  3-class chance", color="#aaa", fontsize=8, va="bottom", zorder=0)

    ax.set_xlabel("Number of features (log scale)")
    ax.set_ylabel("LOSO classification accuracy")
    ax.set_title("More features ≠ more accuracy.\n18 cardiovascular biomarkers match 406 urinary cytokines.",
                 fontsize=11.5, pad=10)
    ax.set_xlim(5, 5500)
    ax.set_ylim(0.30, 0.80)
    ax.grid(alpha=0.18)

    adjust_text(
        texts,
        ax=ax,
        expand=(1.35, 1.7),
        force_text=(0.55, 0.95),
        force_static=(0.35, 0.55),
        force_explode=(0.25, 0.45),
        only_move={"text": "y", "static": "y", "explode": "xy"},
        arrowprops=dict(arrowstyle="-", color="#777", lw=0.6, alpha=0.8,
                        shrinkA=4, shrinkB=4),
        max_move=80,
    )

    fig.text(0.99, 0.005, "9 modalities, n=4 crew, LOSO-CV · Inspiration4 + MMASH proxy",
             ha="right", fontsize=7, color="#999")
    fig.savefig(FIG / "fig_C_feature_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_C_feature_efficiency.png")


# =============================================================================
# Fig. D — Calibration & posterior predictive
# =============================================================================
def fig_d_calibration():
    cardio_b = bayes["cardio_eve_bayesian_hierarchical"]
    folds = cardio_b["fold_details"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1 = axes[0]
    subjects_lbl = [f["held_subject"] for f in folds]
    accs_b = [f["acc"] for f in folds]
    knn_acc = 0.679
    x = np.arange(len(subjects_lbl))
    w = 0.36
    ax1.bar(x - w/2, accs_b, w, color=SPACE, label="Bayesian hierarchical", alpha=0.9)
    ax1.bar(x + w/2, [knn_acc]*len(subjects_lbl), w, color=NEUTRAL, alpha=0.7,
            label=f"kNN baseline (mean={knn_acc:.2f})")
    ax1.axhline(0.5, color="#ccc", linestyle=":", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects_lbl)
    ax1.set_ylabel("LOSO accuracy on held-out subject")
    ax1.set_title("Bayesian model accuracy is comparable to kNN…", fontsize=10.5, pad=8)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(alpha=0.15, axis="y")

    ax2 = axes[1]
    briers = [f["brier"] for f in folds]
    ci_widths = [f["mean_ci_width"] for f in folds]
    ax2.bar(x - w/2, briers, w, color=MARS, alpha=0.9, label="Brier score (lower = calibrated)")
    ax2.axhline(0.25, color="#ccc", linestyle=":", linewidth=1)
    ax2.text(len(folds)-0.3, 0.252, "uniform predictor", fontsize=8, color="#888", va="bottom", ha="right")
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects_lbl)
    ax2.set_ylabel("Brier score")
    ax2.set_ylim(0, 0.32)
    ax2.set_title("…but the Bayesian model adds calibration + 95% CI on every prediction", fontsize=10.5, pad=8)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(alpha=0.15, axis="y")
    ax2b = ax2.twinx()
    ax2b.plot(x, ci_widths, "-D", color=CREW, markeredgecolor="white", markersize=8, linewidth=1.8,
              label="Mean 95% CI width", alpha=0.95)
    ax2b.set_ylabel("Mean 95% predictive CI width", color=CREW)
    ax2b.tick_params(axis="y", labelcolor=CREW)
    ax2b.set_ylim(0, 1.0)
    ax2b.spines["top"].set_visible(False)

    fig.suptitle(f"Cardiovascular EvePanel: per-fold posterior summary (n={cardio_b['n_samples']}, p={cardio_b['n_features']})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.99, 0.005, "Hierarchical Bayesian logistic w/ horseshoe-style shrinkage · NumPyro NUTS, 400+400 draws, 2 chains",
             ha="right", fontsize=7, color="#999")
    fig.savefig(FIG / "fig_D_bayesian_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_D_bayesian_calibration.png")


# =============================================================================
# Composite 2x2 panel
# =============================================================================
def fig_composite():
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    rows = [
        ("Urine cytokines (n=22)",        22, 406, 0.688, 0.471, 0.836, MARS),
        ("Cardiovascular (n=28)",         28, 18,  0.679, 0.492, 0.821, MARS),
        ("Serum cytokines (n=27)",        27, 406, 0.625, 0.441, 0.785, SPACE),
        ("HRV terrestrial proxy (n=22)",  22, 9,   0.636, 0.427, 0.803, NEUTRAL),
        ("Metabolic CMP (n=28)",          28, 57,  0.571, 0.389, 0.736, SPACE),
        ("Immune EvePanel (n=28)",        28, 142, 0.571, 0.389, 0.736, SPACE),
        ("CBC (n=28)",                    28, 60,  0.571, 0.389, 0.736, SPACE),
        ("Pathway CPM (n=189)",           189, 567, 0.406, 0.340, 0.479, CAUTION),
        ("Microbiome species (n=319)",    319, 2723, 0.404, 0.352, 0.459, CAUTION),
    ]
    rows.sort(key=lambda r: r[3])

    ax_a = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        ax_a.plot([r[4], r[5]], [y, y], color=r[6], linewidth=2.2, alpha=0.55)
        ax_a.plot(r[3], y, "o", color=r[6], markersize=9, markeredgecolor="white", markeredgewidth=1.3)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax_a.axvline(0.5, color="#ccc", linestyle=":", linewidth=1)
    ax_a.set_xlim(0.20, 1.00)
    ax_a.set_xlabel("LOSO accuracy (95% CrI)")
    ax_a.set_title("A · Per-modality classification accuracy", fontweight="bold", loc="left", fontsize=11)
    ax_a.grid(axis="x", alpha=0.18)

    ax_b = fig.add_subplot(gs[0, 1])
    pts = [
        ("Cardio",   18,   0.679, MARS),
        ("HRV",      9,    0.636, NEUTRAL),
        ("CMP",      57,   0.571, SPACE),
        ("CBC",      60,   0.571, SPACE),
        ("Immune",   142,  0.571, SPACE),
        ("Serum",    406,  0.625, SPACE),
        ("Urine",    406,  0.688, MARS),
        ("Pathway",  567,  0.406, CAUTION),
        ("Taxa",     2723, 0.404, CAUTION),
    ]
    texts_b = []
    for name, p, acc, c in pts:
        ax_b.scatter(p, acc, s=130, c=c, edgecolor="white", linewidth=1.4, alpha=0.95)
        texts_b.append(ax_b.text(p, acc, name, fontsize=8.5, color=c, zorder=4))
    ax_b.set_xscale("log")
    ax_b.axhline(0.5, color="#ccc", linestyle=":", linewidth=1)
    ax_b.set_xlabel("Number of features (log)")
    ax_b.set_ylabel("LOSO accuracy")
    ax_b.set_title("B · Feature-efficiency frontier", fontweight="bold", loc="left", fontsize=11)
    ax_b.grid(alpha=0.18)
    ax_b.set_xlim(5, 5500)
    ax_b.set_ylim(0.30, 0.80)
    adjust_text(
        texts_b,
        ax=ax_b,
        expand=(1.3, 1.6),
        force_text=(0.5, 0.9),
        force_static=(0.3, 0.5),
        only_move={"text": "y", "static": "y", "explode": "xy"},
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, alpha=0.75,
                        shrinkA=3, shrinkB=3),
        max_move=60,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    master = pd.read_csv(PUB / "_v3_master_table.csv")
    common = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82"]
    subjects = ["C001", "C002", "C003", "C004"]
    subj_colors = {"C001": SPACE, "C002": CAUTION, "C003": CREW, "C004": MARS}
    subj_marker = {"C001": "o", "C002": "s", "C003": "^", "C004": "D"}
    feat_cols = [c for c in master.columns if c not in ("subject", "timepoint", "spaceflight")]
    means = master[feat_cols].mean(); stds = master[feat_cols].std().replace(0, 1.0)
    Z = ((master[feat_cols] - means) / stds).fillna(0).to_numpy()
    valid_idx = master["timepoint"].isin(common) & master["subject"].isin(subjects)
    Z_v = Z[valid_idx]; meta_v = master[valid_idx][["subject", "timepoint"]].reset_index(drop=True)
    Z_v -= Z_v.mean(axis=0)
    _, S, Vt = np.linalg.svd(Z_v, full_matrices=False)
    pc1 = (Z_v @ Vt[0])
    for s in subjects:
        sel = meta_v["subject"] == s
        sub = meta_v[sel].copy(); sub["o"] = sub["timepoint"].map(lambda t: common.index(t) if t in common else -1)
        idx_sorted = sub.sort_values("o").index
        ax_c.plot(range(len(idx_sorted)), pc1[idx_sorted],
                  marker=subj_marker[s], color=subj_colors[s],
                  linewidth=2, markersize=8, label=s,
                  markeredgecolor="white", markeredgewidth=1.2, alpha=0.92)
    ax_c.set_xticks(range(len(common))); ax_c.set_xticklabels(common)
    ax_c.axvline(2.5, color="#888", linestyle="--", linewidth=1)
    ax_c.text(2.5, ax_c.get_ylim()[1]*0.93, "  LAUNCH", color="#666", fontsize=8.5, va="top")
    ax_c.set_xlabel("Timepoint")
    ax_c.set_ylabel("PC1 (multi-modal latent)")
    ax_c.set_title("C · Inter-individual trajectories", fontweight="bold", loc="left", fontsize=11)
    ax_c.legend(fontsize=8.5, loc="best")
    ax_c.grid(alpha=0.18)

    ax_d = fig.add_subplot(gs[1, 1])
    cardio_b = bayes["cardio_eve_bayesian_hierarchical"]
    folds = cardio_b["fold_details"]
    sub_lbl = [f["held_subject"] for f in folds]
    accs_b = [f["acc"] for f in folds]
    briers = [f["brier"] for f in folds]
    knn_acc = 0.679
    x = np.arange(len(sub_lbl)); w = 0.30
    ax_d.bar(x - w, accs_b, w, color=SPACE, alpha=0.92, label="Bayesian acc")
    ax_d.bar(x,     [knn_acc]*len(sub_lbl), w, color=NEUTRAL, alpha=0.7, label="kNN acc")
    ax_d.bar(x + w, briers, w, color=MARS, alpha=0.92, label="Brier (cal.)")
    ax_d.axhline(0.5, color="#ccc", linestyle=":", linewidth=1)
    ax_d.axhline(0.25, color="#ddd", linestyle=":", linewidth=1)
    ax_d.set_xticks(x); ax_d.set_xticklabels(sub_lbl)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_title("D · Bayesian adds calibration on cardiovascular panel",
                   fontweight="bold", loc="left", fontsize=11)
    ax_d.legend(fontsize=8.5, loc="upper right", ncol=3)
    ax_d.grid(alpha=0.18, axis="y")
    ax_d.set_xlabel("Held-out subject (LOSO fold)")

    fig.suptitle("Multi-modal small-sample inference on the SpaceX Inspiration4 cohort\n"
                 "n=4 civilian crew · 9 modalities · 24 aligned (subject, timepoint) observations",
                 fontsize=14.5, fontweight="bold", y=0.998)
    fig.text(0.5, -0.005,
             "Puerta Ángulo (2026) · Methodology demonstration · Open-source release · CC BY 4.0",
             ha="center", fontsize=8, color="#888")
    fig.savefig(FIG / "fig_ALL_composite.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_ALL_composite.png")


if __name__ == "__main__":
    fig_a_headline()
    fig_b_trajectories()
    fig_c_efficiency()
    fig_d_calibration()
    fig_composite()
    print(f"\nAll figures in: {FIG}")
