"""HRP Artemis II — v3: Multi-modal Bayesian factor model with shared latent state.

This is the methodologically-important step. Instead of training one classifier
per modality, we infer a single low-dimensional latent state z[subject, timepoint]
that ALL the n=4 multi-modal observations are explained by:

    z[s, t] ~ Normal(0, I_K)             # K-dim latent state per subject × timepoint
    X_m[s, t] = W_m · z[s, t] + ε_m      # loadings per modality m
    y_phase[s, t] = sigmoid(α + β · z[s, t])  # spaceflight phase from latent

Modalities aligned on the common grid {C001..C004} × {L-92, L-44, L-3, R+1, R+45, R+82}
= 24 observations × multiple modalities.

Outputs:
  publicacion/_v3_multimodal_metrics_2026-05-09.json
  publicacion/_v3_latent_2d.png
  publicacion/_v3_latent_trajectories.png
  publicacion/_v3_loadings.png
  publicacion/_v3_master_table.csv
"""
from __future__ import annotations
import os, sys, json, datetime, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
demo = import_module("02_demo_pipeline")

import pymc as pm
import arviz as az
import zipfile

ROOT = Path(r"C:\Users\engim\marte\HRP_Artemis_II")
PUB = ROOT / "publicacion"
PUB.mkdir(exist_ok=True)
DATE = datetime.date.today().isoformat()

OSDR = ROOT / "dataset" / "nasa_osdr_inspiration4"

# Common timepoints across all modalities (no in-flight: cytokines/CBC/urine miss FD2/FD3)
COMMON_TPS = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82"]
SUBJECTS = ["C001", "C002", "C003", "C004"]


# ============================================================
# 1. Build aligned master table: rows = (subject, timepoint), cols = features per modality
# ============================================================
def standardize_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Drop metadata cols, prefix feature names with modality, return numeric-only.
    Deduplicates by column name (keeps first occurrence)."""
    drop = [c for c in df.columns
            if c in ("Sample ID", "Sample Name", "subject", "timepoint", "spaceflight",
                    "_modality", "Unnamed: 2") or "Unnamed" in str(c)]
    feats = df.drop(columns=drop, errors="ignore")
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.dropna(axis=1, how="all")  # drop fully-empty cols
    # Deduplicate column names (in case the source CSV has repeated headers)
    feats = feats.loc[:, ~feats.columns.duplicated()]
    feats.columns = [f"{prefix}__{c}" for c in feats.columns]
    return feats


def load_modality_aligned(osd_dir: Path, glob_pat: str, prefix: str) -> pd.DataFrame:
    """Load a flat OSD-575-style modality and pivot to (subject, timepoint) index."""
    df = demo.load_osd575_modality(osd_dir, glob_pat, prefix)
    if df.empty:
        return pd.DataFrame()
    valid = df.dropna(subset=["subject", "timepoint"]).copy()
    feats = standardize_columns(valid, prefix)
    valid_idx = valid[["subject", "timepoint"]].reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    out = pd.concat([valid_idx, feats], axis=1)
    out = out[out["subject"].isin(SUBJECTS) & out["timepoint"].isin(COMMON_TPS)]
    out = out.drop_duplicates(subset=["subject", "timepoint"], keep="first")
    return out.set_index(["subject", "timepoint"])


def load_microbiome_aggregated(osd572: Path, prefix: str = "tax") -> pd.DataFrame:
    """OSD-572 microbiome: collapse 13 sites to per-(subject, timepoint) mean abundance,
    keep top-K most-variable taxa to keep dimensionality manageable."""
    tax = demo.load_microbiome_taxonomy(osd572)
    if tax.empty:
        return pd.DataFrame()
    id_col = "clade_name" if "clade_name" in tax.columns else tax.columns[0]
    sample_cols = [c for c in tax.columns if c != id_col and demo._SAMPLE_RE.search(str(c))]
    X_wide = tax.set_index(id_col)[sample_cols].apply(pd.to_numeric, errors="coerce").T
    meta = pd.DataFrame([demo.parse_sample_col(c) for c in X_wide.index], index=X_wide.index)
    full = X_wide.join(meta)
    full = full[full["subject"].isin(SUBJECTS)]
    # Map timepoint code → common timepoint string
    tp_map = {"L-92": "L-92", "L-44": "L-44", "L-3": "L-3",
              "FD2": None, "FD3": None,
              "R+1": "R+1", "R+45": "R+45", "R+82": "R+82"}
    full["timepoint_common"] = full["timepoint_code"].map(tp_map)
    full = full[full["timepoint_common"].isin(COMMON_TPS)]
    feat_cols = [c for c in full.columns
                 if c not in ("subject", "timepoint", "timepoint_code", "site_code", "spaceflight", "timepoint_common")]
    # Aggregate across sites: mean per (subject, timepoint)
    agg = full.groupby(["subject", "timepoint_common"])[feat_cols].mean()
    agg.index.names = ["subject", "timepoint"]
    # Keep top-K variance taxa (K=80 is plenty after multi-site averaging)
    var = agg.var()
    top = var.sort_values(ascending=False).head(80).index
    agg = agg[top]
    agg.columns = [f"{prefix}__{c[:60]}" for c in agg.columns]
    return agg


def main_build_master() -> pd.DataFrame:
    print("Building aligned master table…")
    osd575 = OSDR / "OSD-575_blood_metabolic_cytokines"
    osd656 = OSDR / "OSD-656_urine_inflammation"
    osd569 = OSDR / "OSD-569_whole_blood_seq"
    osd572 = OSDR / "OSD-572_oral_nasal_skin_microbiome"

    parts = []
    for label, osd_dir, glob_pat in [
        ("alamar", osd575, "*AlamarPanel_TRANSFORMED.csv"),
        ("cmp", osd575, "*CMP_TRANSFORMED.csv"),
        ("cardio", osd575, "*cardiovascular_EvePanel_TRANSFORMED.csv"),
        ("immune_eve", osd575, "*immune_EvePanel_TRANSFORMED.csv"),
        ("urine", osd656, "*urine.immune.AlamarPanel_TRANSFORMED.csv"),
        ("cbc", osd569, "*CBC_TRANSFORMED.csv"),
    ]:
        m = load_modality_aligned(osd_dir, glob_pat, label)
        print(f"  {label}: {m.shape if not m.empty else 'empty'}")
        if not m.empty:
            parts.append(m)

    # Microbiome (aggregated across sites)
    micro = load_microbiome_aggregated(osd572, "tax")
    print(f"  taxonomy (aggregated): {micro.shape if not micro.empty else 'empty'}")
    if not micro.empty:
        parts.append(micro)

    if not parts:
        raise RuntimeError("No modalities loaded")

    # Outer join on (subject, timepoint), then deduplicate any leftover column collisions
    master = parts[0]
    for p in parts[1:]:
        master = master.join(p, how="outer", lsuffix="", rsuffix="_dup")
    master = master.loc[:, ~master.columns.str.endswith("_dup")]
    master = master.loc[:, ~master.columns.duplicated()]

    # Restrict to common grid
    grid = pd.MultiIndex.from_product([SUBJECTS, COMMON_TPS], names=["subject", "timepoint"])
    master = master.reindex(grid)
    # Spaceflight label per timepoint
    master["spaceflight"] = master.index.get_level_values("timepoint").map(
        lambda t: "pre-flight" if t.startswith("L-") else "post-flight"
    )
    print(f"\nMaster table: {master.shape} (rows = {len(master)} subject-timepoints)")
    print(f"Coverage per modality (non-NaN columns × non-NaN rows):")
    return master


# ============================================================
# 2. Bayesian factor model
# ============================================================
def fit_factor_model(X: np.ndarray, y_phase: np.ndarray, subj_idx: np.ndarray, tp_idx: np.ndarray,
                     n_subjects: int, n_timepoints: int, K: int = 2,
                     draws: int = 500, tune: int = 500, seed: int = 42):
    """Probabilistic factor model with shared latent state z[s, t].

    For observation i with (subject s, timepoint t):
      z_i = z[s, t] ∈ R^K
      X_i = W · z_i + b + ε_i,  ε_i ~ Normal(0, σ_x · I)
      y_i = Bernoulli(sigmoid(α + γ · z_i))
    """
    n_obs, p = X.shape
    obs_to_z = (subj_idx * n_timepoints + tp_idx).astype(np.int64)
    n_z = n_subjects * n_timepoints
    # NaNs are imputed to 0 AFTER z-scoring → equivalent to imputing the column mean
    X_imputed = np.nan_to_num(X, nan=0.0)

    with pm.Model() as model:
        # Latent state per (subject, timepoint) — standard Normal prior
        z = pm.Normal("z", 0.0, 1.0, shape=(n_z, K))

        # Loadings + intercepts per feature
        W = pm.Normal("W", 0.0, 1.0, shape=(p, K))
        b = pm.Normal("b", 0.0, 1.0, shape=p)
        sigma_x = pm.HalfNormal("sigma_x", 1.0)

        z_obs = z[obs_to_z]                         # (n_obs, K)
        X_hat = pm.math.dot(z_obs, W.T) + b         # (n_obs, p)
        pm.Normal("X_obs", mu=X_hat, sigma=sigma_x, observed=X_imputed)

        # Spaceflight phase from latent
        alpha = pm.Normal("alpha", 0.0, 1.0)
        gamma = pm.Normal("gamma", 0.0, 1.0, shape=K)
        eta = alpha + pm.math.dot(z_obs, gamma)
        pm.Bernoulli("y_obs", logit_p=eta, observed=y_phase)

        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          random_seed=seed, progressbar=False, target_accept=0.9,
                          return_inferencedata=True, nuts_sampler="numpyro")
    return idata, model


def main():
    print("=" * 70)
    print("HRP Artemis II — v3: Multi-modal factor model")
    print("=" * 70)

    master = main_build_master()
    csv_path = PUB / f"_v3_master_table.csv"
    master.to_csv(csv_path)
    print(f"Wrote master table: {csv_path}")

    # Z-score per feature column (across observations)
    feat_cols = list(dict.fromkeys([c for c in master.columns if c != "spaceflight"]))  # unique-preserving order
    feats = master.loc[:, feat_cols].copy()
    # Drop columns with all-NaN
    feats = feats.dropna(axis=1, how="all")
    # Also drop high-NaN columns (>50% missing)
    keep_cols = feats.columns[feats.notna().mean() >= 0.50]
    feats = feats[keep_cols]
    print(f"\nFeatures after NaN filter: {feats.shape[1]} (was {len(feat_cols)})")

    # Z-score
    means = feats.mean()
    stds = feats.std().replace(0, 1.0)
    Z = ((feats - means) / stds).to_numpy()  # shape (n_obs, p) with NaN remaining where original NaN

    # Encode subject / timepoint indices
    subj_idx = np.array([SUBJECTS.index(s) for s, _ in master.index])
    tp_idx = np.array([COMMON_TPS.index(t) for _, t in master.index])
    y = (master["spaceflight"] == "pre-flight").astype(int).to_numpy()
    print(f"y distribution: pre-flight={y.sum()}, post-flight={len(y) - y.sum()}")

    # Fit factor model with K=2 latent dims
    print(f"\nFitting factor model (K=2 latent dims) on {Z.shape[0]} obs × {Z.shape[1]} features…")
    idata, model = fit_factor_model(Z, y, subj_idx, tp_idx,
                                     n_subjects=len(SUBJECTS), n_timepoints=len(COMMON_TPS),
                                     K=2, draws=400, tune=400)

    # Posterior mean of z per (subject, timepoint)
    z_post = idata.posterior["z"].mean(("chain", "draw")).values   # (n_z, K)
    n_subj = len(SUBJECTS)
    n_tps = len(COMMON_TPS)
    z_grid = z_post.reshape(n_subj, n_tps, -1)  # (n_subj, n_tps, K)

    # Loadings posterior mean
    W_post = idata.posterior["W"].mean(("chain", "draw")).values  # (p, K)

    # In-sample predicted probabilities
    alpha_m = float(idata.posterior["alpha"].mean())
    gamma_m = idata.posterior["gamma"].mean(("chain", "draw")).values
    obs_to_z = subj_idx * n_tps + tp_idx
    eta_m = alpha_m + z_post[obs_to_z] @ gamma_m
    p_pred = 1.0 / (1.0 + np.exp(-eta_m))
    pred = (p_pred >= 0.5).astype(int)
    in_sample_acc = float((pred == y).mean())
    print(f"\nIn-sample accuracy (factor → phase): {in_sample_acc:.3f}")

    # Diagnostics
    summary = az.summary(idata, var_names=["alpha", "gamma", "sigma_x"])
    print("\nGroup-level posterior:")
    print(summary.to_string())

    # ============================================================
    # Plots
    # ============================================================
    # 1) Latent 2D scatter colored by phase, marker per subject
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"pre-flight": "#2980b9", "post-flight": "#c0392b"}
    markers = {"C001": "o", "C002": "s", "C003": "^", "C004": "D"}
    for s_i, s in enumerate(SUBJECTS):
        for t_i, t in enumerate(COMMON_TPS):
            phase = "pre-flight" if t.startswith("L-") else "post-flight"
            ax.scatter(z_grid[s_i, t_i, 0], z_grid[s_i, t_i, 1],
                       c=colors[phase], marker=markers[s], s=110, alpha=0.85,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(t, (z_grid[s_i, t_i, 0], z_grid[s_i, t_i, 1]),
                        fontsize=7, alpha=0.6, xytext=(4, 4), textcoords="offset points")
    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="C001", markerfacecolor="#888", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="C002", markerfacecolor="#888", markersize=10),
        Line2D([0], [0], marker="^", color="w", label="C003", markerfacecolor="#888", markersize=10),
        Line2D([0], [0], marker="D", color="w", label="C004", markerfacecolor="#888", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="pre-flight", markerfacecolor=colors["pre-flight"], markersize=10),
        Line2D([0], [0], marker="o", color="w", label="post-flight", markerfacecolor=colors["post-flight"], markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=9)
    ax.set_xlabel("Latent z₁ (posterior mean)")
    ax.set_ylabel("Latent z₂ (posterior mean)")
    ax.set_title(f"Multi-modal latent state — n=4 × 6 timepoints, {Z.shape[1]} features\n"
                 f"In-sample acc {in_sample_acc:.3f}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PUB / f"_v3_latent_2d.png", dpi=120, facecolor="white")
    plt.close(fig)

    # 2) Latent trajectories per subject (z over timepoints)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(2):
        for s_i, s in enumerate(SUBJECTS):
            axes[k].plot(range(n_tps), z_grid[s_i, :, k], marker=markers[s], label=s, linewidth=1.8)
        axes[k].set_xticks(range(n_tps))
        axes[k].set_xticklabels(COMMON_TPS)
        axes[k].axvline(2.5, color="gray", linestyle="--", alpha=0.5)  # between L-3 and R+1 = launch
        axes[k].text(2.5, axes[k].get_ylim()[1] * 0.92, "Launch", ha="center", color="gray", fontsize=8)
        axes[k].set_ylabel(f"Latent z_{k+1} posterior mean")
        axes[k].set_xlabel("Timepoint")
        axes[k].set_title(f"Latent dimension {k+1} trajectory per subject")
        axes[k].legend(fontsize=8)
        axes[k].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PUB / f"_v3_latent_trajectories.png", dpi=120, facecolor="white")
    plt.close(fig)

    # 3) Top loadings per latent dimension
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k in range(2):
        order = np.argsort(np.abs(W_post[:, k]))[::-1][:15]
        names = [keep_cols[i][:50] for i in order]
        vals = W_post[order, k]
        colors_bar = ["#27ae60" if v > 0 else "#c0392b" for v in vals]
        axes[k].barh(range(len(order)), vals, color=colors_bar)
        axes[k].set_yticks(range(len(order)))
        axes[k].set_yticklabels(names, fontsize=7)
        axes[k].axvline(0, color="black", linewidth=0.5)
        axes[k].set_xlabel(f"Loading on latent z_{k+1}")
        axes[k].set_title(f"Top-15 features driving latent dim {k+1}")
        axes[k].invert_yaxis()
    fig.tight_layout()
    fig.savefig(PUB / f"_v3_loadings.png", dpi=120, facecolor="white")
    plt.close(fig)

    # Persist metrics
    metrics = {
        "n_obs": int(Z.shape[0]),
        "n_features_used": int(Z.shape[1]),
        "n_latent_dims": 2,
        "in_sample_accuracy": in_sample_acc,
        "alpha_posterior_mean": alpha_m,
        "gamma_posterior_mean": gamma_m.tolist(),
        "sigma_x_posterior_mean": float(idata.posterior["sigma_x"].mean()),
        "subjects": SUBJECTS,
        "timepoints": COMMON_TPS,
        "modalities_in_master": ["alamar", "cmp", "cardio", "immune_eve", "urine", "cbc", "tax"],
    }
    metrics_path = PUB / f"_v3_multimodal_metrics_{DATE}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")
    print(f"Wrote _v3_latent_2d.png, _v3_latent_trajectories.png, _v3_loadings.png")


if __name__ == "__main__":
    main()
