"""HRP Artemis II - 09: statistical significance and permutation testing.

Two rigorous checks that strengthen the inferential claims, computed from the
committed master table only (no 12 GB download, no MCMC):

  (1) Per-feature Welch t-test (post vs pre-flight) with Benjamini-Hochberg FDR.
      Expectation in the p >> n regime: essentially no feature survives correction,
      which is the honest motivation for the Bayesian / shrinkage approach.

  (2) Label-permutation null for the COMBINED multi-modal separability:
      PCA(k) -> logistic regression under leave-one-subject-out, compared against
      a null built by shuffling the pre/post labels. Yields an empirical p-value
      for "is the multi-modal pre/post signal above chance?".

Outputs:
  publicacion/figures_publicacion/fig_I_perfeature_fdr.png
  publicacion/figures_publicacion/fig_J_permutation_null.png
  publicacion/_significance_metrics.json

Run:  python notebooks/09_significance.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publicacion"
FIGS = PUB / "figures_publicacion"
MASTER = PUB / "_v3_master_table.csv"
RED, BLUE, GREY = "#c0392b", "#2980b9", "#7f8c8d"
RNG = np.random.default_rng(42)


def load():
    df = pd.read_csv(MASTER)
    tp = df["timepoint"].astype(str)
    df["_phase"] = np.where(tp.str.startswith("L"), 0,
                            np.where(tp.str.startswith("R"), 1, np.nan))
    df = df.dropna(subset=["_phase"]).reset_index(drop=True)
    feat = [c for c in df.columns if c not in ("subject", "timepoint", "_phase")
            and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat


# ------------------------------------------------- (1) per-feature FDR
def per_feature_fdr(df, feat, q=0.05):
    pre = df[df["_phase"] == 0]
    post = df[df["_phase"] == 1]
    pvals, effect, names = [], [], []
    for c in feat:
        a, b = pre[c].dropna(), post[c].dropna()
        if len(a) < 3 or len(b) < 3 or (a.std(ddof=1) == 0 and b.std(ddof=1) == 0):
            continue
        t, p = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
        if np.isnan(p):
            continue
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or np.nan
        d = (b.mean() - a.mean()) / sd if sd and not np.isnan(sd) else 0.0
        pvals.append(float(p)); effect.append(float(d)); names.append(c)
    pvals = np.array(pvals); effect = np.array(effect)
    # Benjamini-Hochberg
    order = np.argsort(pvals)
    m = len(pvals)
    ranked = pvals[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    qcut = ranked[k - 1] if k > 0 else 0.0
    qvals = np.minimum.accumulate((pvals[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q_by_name = {names[order[i]]: float(min(1, qvals[i])) for i in range(m)}
    n_sig = int(k)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(pvals, bins=30, color=BLUE, edgecolor="white")
    ax[0].set_title(f"Per-feature p-values (m={m})"); ax[0].set_xlabel("p (Welch t, post vs pre)")
    ax[0].set_ylabel("features")
    ax[1].scatter(effect, -np.log10(pvals), s=10, c=np.where(pvals < 0.05, RED, GREY), alpha=0.6)
    ax[1].axhline(-np.log10(0.05), color="#555", ls="--", lw=1)
    ax[1].set_xlabel("standardized effect (post - pre)"); ax[1].set_ylabel("-log10 p")
    ax[1].set_title(f"Volcano - {n_sig} survive BH-FDR (q<{q})")
    fig.tight_layout(); fig.savefig(FIGS / "fig_I_perfeature_fdr.png", dpi=120); plt.close(fig)
    return {"n_features_tested": m, "n_uncorrected_p05": int((pvals < 0.05).sum()),
            "n_significant_bh_fdr": n_sig, "fdr_q": q,
            "expected_false_positives_at_p05": round(0.05 * m, 1)}


# ------------------------------------------------- (2) permutation null
def _loso_logit_acc(X, y, subj):
    accs = []
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        if tr.sum() < 4 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=0.5)
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else np.nan


def permutation_null(df, feat, n_perm=2000, k=10):
    X = df[feat].copy()
    X = X.loc[:, X.notna().mean() > 0.6]
    X = X.fillna(X.median())
    X = X.loc[:, X.std(ddof=0) > 0]
    Xs = StandardScaler().fit_transform(X)
    k = min(k, Xs.shape[0] - 1, Xs.shape[1])
    Z = PCA(n_components=k, random_state=42).fit_transform(Xs)
    y = df["_phase"].to_numpy().astype(int)
    subj = df["subject"].astype(str).to_numpy()
    obs = _loso_logit_acc(Z, y, subj)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = RNG.permutation(y)
        null[i] = _loso_logit_acc(Z, yp, subj)
    null = null[~np.isnan(null)]
    p_emp = (np.sum(null >= obs) + 1) / (len(null) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(null, bins=30, color=GREY, edgecolor="white", alpha=0.8, label="permuted-label null")
    ax.axvline(obs, color=RED, lw=2.5, label=f"observed = {obs:.3f}")
    ax.axvline(0.5, color="#333", ls=":", lw=1, label="chance")
    ax.set_xlabel("LOSO logistic accuracy (multi-modal, PCA-reduced)")
    ax.set_ylabel("permutations"); ax.set_title(f"Permutation test: empirical p = {p_emp:.3f}")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "fig_J_permutation_null.png", dpi=120); plt.close(fig)
    return {"pca_components": int(k), "observed_loso_acc": round(float(obs), 4),
            "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4),
            "n_permutations": int(len(null)), "empirical_p": round(float(p_emp), 4)}


def run():
    print("HRP Artemis II - significance & permutation testing")
    df, feat = load()
    print(f"  obs={len(df)}, features={len(feat)}")
    fdr = per_feature_fdr(df, feat)
    print(f"  FDR: {fdr['n_uncorrected_p05']} nominal p<.05 (expected {fdr['expected_false_positives_at_p05']} by chance), "
          f"{fdr['n_significant_bh_fdr']} survive BH-FDR")
    perm = permutation_null(df, feat)
    print(f"  Permutation: observed {perm['observed_loso_acc']} vs null {perm['null_mean']}+/-{perm['null_std']}, "
          f"empirical p={perm['empirical_p']}")
    out = {"per_feature_fdr": fdr, "permutation_null": perm}
    (PUB / "_significance_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("  wrote publicacion/_significance_metrics.json")


if __name__ == "__main__":
    run()
