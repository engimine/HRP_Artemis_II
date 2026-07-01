"""HRP Artemis II — Bayesian hierarchical logistic v2.

Goal: replace the kNN baseline of demo v1 with a hierarchical Bayesian model
that respects the n=4 structure of Inspiration4 — random intercept per
subject + horseshoe-style shrinkage on feature coefficients.

Applied first to OSD-575 cardiovascular EvePanel (n=28 samples, p=18 features,
2-class pre/post-flight) — the modality where kNN already showed signal
(acc=0.679) and dimensionality is tractable for full posterior inference.

LOSO-CV: hold out one of C001-C004 at a time, fit on the other 3, compute
posterior predictive accuracy on the held-out subject's samples.

Outputs:
  publicacion/_bayesian_metrics_2026-05-09.json
  publicacion/_bayesian_trace_<modality>.png        (trace + rank plot)
  publicacion/_bayesian_coef_<modality>.png         (coefficient forest plot)
  publicacion/_bayesian_pred_<modality>.png         (LOSO posterior predictive)
"""
from __future__ import annotations
import os, sys, json, datetime, warnings
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")  # silence g++ warning, fallback to Python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Reuse loaders + parser from v1 demo (but don't let it hijack stdout)
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
demo = import_module("02_demo_pipeline")  # re-use load_osd575_modality, parse_sample_flex, zscore

import pymc as pm
import arviz as az

ROOT = Path(r"C:\Users\engim\marte\HRP_Artemis_II")
PUB = ROOT / "publicacion"
PUB.mkdir(exist_ok=True)
DATE = datetime.date.today().isoformat()

OSDR = ROOT / "dataset" / "nasa_osdr_inspiration4"


# ============================================================
# Bayesian hierarchical logistic with horseshoe shrinkage
# ============================================================
def fit_hierarchical_logistic(X: np.ndarray, y: np.ndarray, subj_idx: np.ndarray,
                               n_subjects: int, draws: int = 600, tune: int = 600,
                               chains: int = 2, cores: int = 1, seed: int = 42) -> az.InferenceData:
    """Fit a hierarchical Bayesian logistic regression.

    y_i ~ Bernoulli(sigmoid(α_subj[i] + X[i] @ β))
    α_s ~ Normal(μ_α, σ_α)         # random subject intercept
    β_j ~ Normal(0, λ * τ_j)        # horseshoe-ish shrinkage prior
    λ   ~ HalfCauchy(0.5)           # global scale
    τ_j ~ HalfCauchy(0.5)           # local per-feature scale
    μ_α ~ Normal(0, 1)
    σ_α ~ HalfNormal(1)
    """
    n, p = X.shape
    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", 0, 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_subjects)

        lam = pm.HalfCauchy("lam", 0.5)
        tau = pm.HalfCauchy("tau", 0.5, shape=p)
        beta = pm.Normal("beta", 0.0, sigma=lam * tau, shape=p)

        eta = alpha[subj_idx] + pm.math.dot(X, beta)
        p_obs = pm.math.sigmoid(eta)
        pm.Bernoulli("y_obs", p=p_obs, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          random_seed=seed, progressbar=False, target_accept=0.9,
                          return_inferencedata=True, nuts_sampler="numpyro")
        idata = pm.sample_posterior_predictive(idata, var_names=["y_obs"], extend_inferencedata=True,
                                                random_seed=seed, progressbar=False)
    return idata, model


def posterior_predict(idata: az.InferenceData, model: pm.Model, X_new: np.ndarray,
                       subj_idx_new: np.ndarray) -> np.ndarray:
    """Compute posterior predictive probabilities for held-out samples.

    Uses posterior draws of alpha, beta to compute P(y=1 | new) marginalised over draws.
    """
    post = idata.posterior
    # Stack chains × draws into one axis
    alpha_draws = post["alpha"].stack(sample=("chain", "draw")).values  # (n_subjects, n_draws)
    beta_draws = post["beta"].stack(sample=("chain", "draw")).values    # (p, n_draws)
    n_draws = alpha_draws.shape[-1]
    # Per held-out sample: alpha[subj_new] + X_new @ beta -> sigmoid
    a_for_obs = alpha_draws[subj_idx_new, :]  # (n_new, n_draws)
    eta = a_for_obs + X_new @ beta_draws       # (n_new, n_draws)
    p_pred = 1.0 / (1.0 + np.exp(-eta))
    return p_pred  # (n_new, n_draws)


# ============================================================
# LOSO-CV with Bayesian model
# ============================================================
def loso_bayesian(features: pd.DataFrame, labels: pd.Series, subjects: pd.Series,
                  draws: int = 500, tune: int = 500) -> dict:
    feat_arr = features.fillna(0.0).to_numpy(dtype=float)
    y_full = (labels.astype(str) == labels.astype(str).iloc[0]).astype(int).to_numpy()
    # Encode binary: class 0 = first label, class 1 = the other
    pos_label = sorted(labels.unique())[-1]
    y_full = (labels.astype(str) == str(pos_label)).astype(int).to_numpy()
    print(f"  positive class = {pos_label} ({y_full.sum()}/{len(y_full)} positives)")
    sids = subjects.astype(str).to_numpy()
    unique_subj = sorted(set(sids))
    subj_to_idx = {s: i for i, s in enumerate(unique_subj)}

    accs, briers, p_correct_means, ci_widths = [], [], [], []
    fold_records = []
    for held in unique_subj:
        train_mask = sids != held
        test_mask = sids == held
        if train_mask.sum() < 4 or test_mask.sum() == 0:
            continue
        X_tr, y_tr = feat_arr[train_mask], y_full[train_mask]
        X_te, y_te = feat_arr[test_mask], y_full[test_mask]
        # Re-index subjects (training subjects only) — held-out goes into a NEW slot drawn from prior
        train_subj_idx = np.array([subj_to_idx[s] for s in sids[train_mask]])
        # For prediction on held-out: use mu_alpha posterior as the "new subject" intercept
        print(f"  Fold held={held}: n_train={len(X_tr)}, n_test={len(X_te)}, p={X_tr.shape[1]}")
        idata, model = fit_hierarchical_logistic(
            X_tr, y_tr, train_subj_idx, n_subjects=len(unique_subj),
            draws=draws, tune=tune, chains=2, cores=1
        )
        # Posterior predictive for held-out: use mu_alpha posterior since this subject was unseen.
        post = idata.posterior
        beta_draws = post["beta"].stack(sample=("chain", "draw")).values
        mu_alpha_draws = post["mu_alpha"].stack(sample=("chain", "draw")).values  # (n_draws,)
        eta_te = mu_alpha_draws[None, :] + X_te @ beta_draws  # (n_te, n_draws)
        p_pred = 1.0 / (1.0 + np.exp(-eta_te))
        p_mean = p_pred.mean(axis=1)
        pred = (p_mean >= 0.5).astype(int)
        acc = float((pred == y_te).mean())
        brier = float(((p_mean - y_te) ** 2).mean())
        # Confidence on the correct class
        p_correct = np.where(y_te == 1, p_mean, 1 - p_mean)
        # 95% credible interval width (per sample, then averaged)
        lo = np.quantile(p_pred, 0.025, axis=1)
        hi = np.quantile(p_pred, 0.975, axis=1)
        ci_w = float((hi - lo).mean())
        accs.append(acc); briers.append(brier)
        p_correct_means.append(float(p_correct.mean()))
        ci_widths.append(ci_w)
        fold_records.append({"held_subject": held, "n_test": int(len(y_te)),
                             "acc": acc, "brier": brier,
                             "mean_p_correct": float(p_correct.mean()),
                             "mean_ci_width": ci_w})
        print(f"    held={held}: acc={acc:.3f}, brier={brier:.3f}, mean_p_correct={p_correct.mean():.3f}, ci_w={ci_w:.3f}")

    return {
        "n_folds": len(accs),
        "accuracy_mean": float(np.mean(accs)) if accs else None,
        "accuracy_std": float(np.std(accs)) if accs else None,
        "brier_mean": float(np.mean(briers)) if briers else None,
        "mean_p_correct": float(np.mean(p_correct_means)) if p_correct_means else None,
        "mean_ci_width": float(np.mean(ci_widths)) if ci_widths else None,
        "n_features": int(feat_arr.shape[1]),
        "n_samples": int(feat_arr.shape[0]),
        "model": "hierarchical_logistic_horseshoe",
        "fold_details": fold_records,
    }


# ============================================================
# Plots
# ============================================================
def plot_coefficients(idata: az.InferenceData, feature_names: list, out_path: Path) -> None:
    post = idata.posterior["beta"].stack(sample=("chain", "draw")).values  # (p, n_draws)
    means = post.mean(axis=1)
    lo = np.quantile(post, 0.025, axis=1)
    hi = np.quantile(post, 0.975, axis=1)
    order = np.argsort(np.abs(means))[::-1][:20]  # top 20 by |mean|
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(order))
    ax.errorbar(means[order], y_pos, xerr=[means[order] - lo[order], hi[order] - means[order]],
                fmt='o', color='#c0392b', ecolor='#888')
    ax.axvline(0, color='#666', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i][:40] for i in order], fontsize=8)
    ax.set_xlabel("β posterior mean ± 95% CI")
    ax.set_title("Top-20 coefficients (full data, hierarchical model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)


def plot_loso_summary(metrics: dict, out_path: Path) -> None:
    folds = metrics.get("fold_details", [])
    if not folds:
        return
    subjects = [f["held_subject"] for f in folds]
    accs = [f["acc"] for f in folds]
    p_corrects = [f["mean_p_correct"] for f in folds]
    ci_ws = [f["mean_ci_width"] for f in folds]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].bar(subjects, accs, color="#c0392b"); axes[0].set_ylabel("LOSO accuracy"); axes[0].set_ylim(0, 1.0)
    axes[0].axhline(0.5, color="#888", linestyle="--", alpha=0.5)
    axes[1].bar(subjects, p_corrects, color="#2980b9"); axes[1].set_ylabel("Mean P(correct class)"); axes[1].set_ylim(0, 1.0)
    axes[2].bar(subjects, ci_ws, color="#e67e22"); axes[2].set_ylabel("Mean 95% CI width")
    for ax in axes:
        ax.set_xlabel("Held-out subject")
    fig.suptitle("Hierarchical Bayesian LOSO — per-fold posterior summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def run() -> None:
    print("=" * 70)
    print("HRP Artemis II — Bayesian hierarchical v2")
    print("=" * 70)

    # Cardiovascular: n=28, p=18, n_subjects=4, classes pre/post — tractable for full Bayesian fit
    osd575 = OSDR / "OSD-575_blood_metabolic_cytokines"
    cardio = demo.load_osd575_modality(osd575, "*cardiovascular_EvePanel_TRANSFORMED.csv", "cardio_eve")
    valid = cardio.dropna(subset=["subject", "spaceflight"])
    feat_cols = [c for c in valid.columns
                 if c not in ("Sample ID", "Sample Name", "subject", "timepoint", "spaceflight", "_modality")
                 and pd.api.types.is_numeric_dtype(valid[c])]
    X = demo.zscore(valid[feat_cols].fillna(0.0))
    y = valid["spaceflight"]
    groups = valid["subject"]
    print(f"\nCardiovascular: n={len(X)}, p={X.shape[1]}, subjects={groups.nunique()}, classes={sorted(y.unique())}")

    # 1. LOSO-CV with Bayesian (numpyro JAX backend — fast, no C compiler needed)
    print("\n--- LOSO-CV (Bayesian hierarchical, draws=400, JAX/numpyro) ---")
    bayes_loso = loso_bayesian(X, y, groups, draws=400, tune=400)
    print(f"\nBayesian LOSO summary: acc={bayes_loso['accuracy_mean']:.3f} ± {bayes_loso['accuracy_std']:.3f}, "
          f"brier={bayes_loso['brier_mean']:.3f}, mean P(correct)={bayes_loso['mean_p_correct']:.3f}, "
          f"mean CI width={bayes_loso['mean_ci_width']:.3f}")

    # 2. Full-data fit for coefficient interpretation
    print("\n--- Full-data fit (for coefficient inspection) ---")
    pos_label = sorted(y.unique())[-1]
    y_full = (y == pos_label).astype(int).to_numpy()
    subj_unique = sorted(groups.unique())
    subj_idx = np.array([subj_unique.index(s) for s in groups])
    idata_full, _ = fit_hierarchical_logistic(
        X.to_numpy(dtype=float), y_full, subj_idx,
        n_subjects=len(subj_unique), draws=600, tune=600, chains=2, cores=1
    )
    summary = az.summary(idata_full, var_names=["mu_alpha", "sigma_alpha", "lam"])
    print("\nGroup-level posterior summary:")
    print(summary.to_string())

    # 3. Plots
    plot_coefficients(idata_full, feat_cols, PUB / f"_bayesian_coef_cardio_{DATE}.png")
    plot_loso_summary(bayes_loso, PUB / f"_bayesian_loso_cardio_{DATE}.png")

    # 4. Persist metrics
    out = {
        "cardio_eve_bayesian_hierarchical": bayes_loso,
        "cardio_eve_group_posterior": {
            "mu_alpha_mean": float(idata_full.posterior["mu_alpha"].mean()),
            "mu_alpha_sd": float(idata_full.posterior["mu_alpha"].std()),
            "sigma_alpha_mean": float(idata_full.posterior["sigma_alpha"].mean()),
            "lam_mean": float(idata_full.posterior["lam"].mean()),
        },
    }
    metrics_path = PUB / f"_bayesian_metrics_{DATE}.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {PUB}/_bayesian_coef_cardio_{DATE}.png")
    print(f"Wrote {PUB}/_bayesian_loso_cardio_{DATE}.png")


if __name__ == "__main__":
    run()
