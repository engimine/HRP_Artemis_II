"""HRP Artemis II - 08: extended analysis, figures and dashboard.

Adds analysis and visual outputs on top of the released pipeline, WITHOUT
needing the 12 GB raw download or any MCMC: everything here is derived from the
already-committed harmonised master table (publicacion/_v3_master_table.csv,
24 x 1115) and the metrics JSONs.

Produces:
  publicacion/figures_publicacion/fig_E_classifier_comparison.png   (cited by the paper)
  publicacion/figures_publicacion/fig_F_top_movers.png
  publicacion/figures_publicacion/fig_G_pca_phase.png
  publicacion/figures_publicacion/fig_H_missingness.png
  publicacion/_extended_metrics.json
  publicacion/dashboard.html        (self-contained visibility dashboard)

Run:  python notebooks/08_extended_analysis.py
"""
from __future__ import annotations
import json, base64, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "publicacion"
FIGS = PUB / "figures_publicacion"
FIGS.mkdir(parents=True, exist_ok=True)
MASTER = PUB / "_v3_master_table.csv"
CLF_JSON = PUB / "_baseline_comparison_metrics.json"

RED, BLUE, ORANGE, GREEN = "#c0392b", "#2980b9", "#e67e22", "#27ae60"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})


# ----------------------------------------------------------------- data
def load_master():
    df = pd.read_csv(MASTER)
    # phase: L-* pre-flight (0), R+* post-flight (1)
    tp = df["timepoint"].astype(str)
    df["_phase"] = np.where(tp.str.startswith("L"), 0,
                            np.where(tp.str.startswith("R"), 1, np.nan))
    return df


def numeric_features(df):
    drop = {"subject", "timepoint", "_phase"}
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def modality_of(col: str) -> str:
    if "__" in col:
        return col.split("__", 1)[0]
    for pref in ("cardio", "cbc", "cmp", "eve", "immune", "urine", "serum", "meta"):
        if col.lower().startswith(pref):
            return pref
    return "other"


# ----------------------------------------------------------------- fig E: classifiers
def fig_classifier_comparison():
    data = json.loads(CLF_JSON.read_text(encoding="utf-8"))
    rows = []
    for _, v in data.items():
        rows.append((v["name"], v["kNN"]["acc"], v["RandomForest"]["acc"], v["SVM_RBF"]["acc"]))
    rows.sort(key=lambda r: max(r[1:]), reverse=True)
    names = [r[0] for r in rows]
    knn = [r[1] for r in rows]; rf = [r[2] for r in rows]; svm = [r[3] for r in rows]
    y = np.arange(len(names)); h = 0.26
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + h, knn, h, label="kNN (k=5)", color=BLUE)
    ax.barh(y, rf, h, label="Random Forest", color=RED)
    ax.barh(y - h, svm, h, label="SVM-RBF", color=ORANGE)
    ax.axvline(0.5, color="#555", ls="--", lw=1, label="chance")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOSO accuracy"); ax.set_xlim(0, 1)
    ax.set_title("Per-modality LOSO accuracy by classifier")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); out = FIGS / "fig_E_classifier_comparison.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return {"most_informative_knn": names[int(np.argmax(knn))],
            "most_informative_rf": rows[int(np.argmax(rf))][0],
            "rf_gain_taxonomy": None}


# ----------------------------------------------------------------- fig F: top movers
def fig_top_movers(df, feat, k=20):
    sub = df.dropna(subset=["_phase"])
    pre = sub[sub["_phase"] == 0][feat]
    post = sub[sub["_phase"] == 1][feat]
    # standardized mean difference (post - pre) in pooled-sd units
    mu_pre, mu_post = pre.mean(), post.mean()
    sd = pd.concat([pre, post]).std(ddof=0).replace(0, np.nan)
    d = ((mu_post - mu_pre) / sd).dropna()
    top = d.reindex(d.abs().sort_values(ascending=False).index)[:k]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [RED if v > 0 else BLUE for v in top.values]
    ax.barh(range(len(top))[::-1], top.values, color=colors)
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels([n[:46] for n in top.index], fontsize=8)
    ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("Standardized mean shift  post - pre  (pooled SD units)")
    ax.set_title(f"Top {k} biomarkers by pre/post-flight shift (n=24)")
    fig.tight_layout(); out = FIGS / "fig_F_top_movers.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return top


def fig_pca_phase(df, feat):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sub = df.dropna(subset=["_phase"]).copy()
    X = sub[feat].copy()
    X = X.loc[:, X.notna().mean() > 0.6]           # keep columns >60% present
    X = X.fillna(X.median())
    X = X.loc[:, X.std(ddof=0) > 0]                 # drop constant columns
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    Z = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    subs = sub["subject"].astype(str).values
    markers = {s: m for s, m in zip(sorted(set(subs)), ["o", "s", "^", "D", "P", "X"])}
    for ph, col, lab in [(0, BLUE, "pre-flight (L)"), (1, RED, "post-flight (R)")]:
        m = sub["_phase"].values == ph
        for s in sorted(set(subs)):
            sel = m & (subs == s)
            if sel.any():
                ax.scatter(Z[sel, 0], Z[sel, 1], c=col, marker=markers[s], s=90,
                           edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.scatter([], [], c=BLUE, marker="o", label="pre-flight (L)")
    ax.scatter([], [], c=RED, marker="o", label="post-flight (R)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.set_title("Multi-modal state: pre vs post-flight (PCA, 24 obs)")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout(); out = FIGS / "fig_G_pca_phase.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return {"pc1_var": float(pca.explained_variance_ratio_[0]),
            "pc2_var": float(pca.explained_variance_ratio_[1]),
            "n_features_used": int(X.shape[1])}


def fig_missingness(df, feat):
    mods = {}
    for c in feat:
        mods.setdefault(modality_of(c), []).append(c)
    names, miss = [], []
    for mod, cols in sorted(mods.items(), key=lambda kv: -len(kv[1])):
        names.append(f"{mod} (n={len(cols)})")
        miss.append(float(df[cols].isna().mean().mean()))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(names))[::-1], np.array(miss) * 100, color=ORANGE)
    ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean missing (%)"); ax.set_xlim(0, 100)
    ax.set_title("Missingness by modality group (master table)")
    fig.tight_layout(); out = FIGS / "fig_H_missingness.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return {n: round(m, 3) for n, m in zip(names, miss)}


# ----------------------------------------------------------------- dashboard
def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_dashboard(ext, clf_json):
    figs = [
        ("A · Headline LOSO accuracy", "fig_A_headline_accuracy.png"),
        ("E · Classifier sensitivity", "fig_E_classifier_comparison.png"),
        ("B · Inter-subject trajectories", "fig_B_intersubject_trajectories.png"),
        ("G · PCA pre vs post-flight", "fig_G_pca_phase.png"),
        ("C · Feature-efficiency frontier", "fig_C_feature_efficiency.png"),
        ("F · Top biomarker shifts", "fig_F_top_movers.png"),
        ("D · Bayesian calibration", "fig_D_bayesian_calibration.png"),
        ("H · Missingness by modality", "fig_H_missingness.png"),
    ]
    cards = ""
    for title, fn in figs:
        p = FIGS / fn
        if p.exists():
            cards += (f'<figure><figcaption>{title}</figcaption>'
                      f'<img src="data:image/png;base64,{b64(p)}"></figure>\n')
    date = datetime.date.today().isoformat()
    movers = ext.get("top_movers_preview", [])
    movers_rows = "".join(
        f"<tr><td>{n[:52]}</td><td class='num'>{v:+.2f}</td></tr>" for n, v in movers)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HRP Artemis II - Analysis Dashboard</title>
<style>
:root{{--bg:#0e1420;--card:#172234;--ink:#eaf0f7;--mut:#93a3ba;--red:#c0392b;--blue:#2980b9;--line:#25344c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:'Segoe UI',system-ui,Arial,sans-serif;line-height:1.5}}
header{{padding:32px 40px;border-bottom:1px solid var(--line);
background:linear-gradient(180deg,#12203a,#0e1420)}}
h1{{margin:0 0 6px;font-size:28px}}.sub{{color:var(--mut);font-size:15px}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 24px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:8px 0 28px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.kpi .v{{font-size:30px;font-weight:800;color:#ff8a5b}}.kpi .l{{color:var(--mut);font-size:13px;margin-top:4px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}
figcaption{{font-weight:700;margin-bottom:10px;color:#cfe0f5}}
figure img{{width:100%;border-radius:8px;background:#fff}}
h2{{margin:34px 0 12px;font-size:20px;border-left:4px solid var(--red);padding-left:12px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
td,th{{padding:8px 12px;border-bottom:1px solid var(--line);font-size:14px;text-align:left}}
.num{{text-align:right;font-variant-numeric:tabular-nums;color:#ffb08a}}
.note{{color:var(--mut);font-size:13px;margin-top:8px}}
footer{{color:var(--mut);font-size:13px;padding:24px 40px;border-top:1px solid var(--line);margin-top:30px}}
a{{color:#6db3ff}}
</style></head><body>
<header>
<h1>HRP Artemis II — Analysis Dashboard</h1>
<div class="sub">Bayesian multi-modal latent-state inference for n=4 deep-space crew analogues ·
SpaceX Inspiration4 (NASA OSDR) · generated {date}</div>
</header>
<div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="v">4</div><div class="l">crew subjects (n)</div></div>
    <div class="kpi"><div class="v">24 × 1115</div><div class="l">harmonised master table</div></div>
    <div class="kpi"><div class="v">9</div><div class="l">modalities fused</div></div>
    <div class="kpi"><div class="v">0.688</div><div class="l">best non-invasive LOSO acc (urine, kNN)</div></div>
    <div class="kpi"><div class="v">0.804</div><div class="l">best acc under Random Forest (serum)</div></div>
    <div class="kpi"><div class="v">0.224</div><div class="l">Bayesian Brier (calibrated)</div></div>
  </div>

  <h2>Headline finding</h2>
  <p>Under a distance-based baseline, urine cytokines and an 18-feature cardiovascular panel are the only
  credibly above-chance non-invasive surrogates. A classifier-sensitivity pass shows this ranking is
  <b>estimator-dependent</b>: a random forest lifts both microbiome modalities from chance to 0.56–0.65 and the
  serum panel to 0.804. The honest conclusion is not "microbiome has no signal" but "a distance classifier is the
  wrong tool at p≫n".</p>

  <h2>Figures</h2>
  <div class="grid">
  {cards}
  </div>

  <h2>Top biomarker shifts (pre → post-flight, standardized)</h2>
  <table><tr><th>Feature</th><th class="num">shift (SD)</th></tr>{movers_rows}</table>
  <div class="note">Post-minus-pre mean shift in pooled-SD units across the 24 subject–timepoint observations.
  Exploratory (n=4); interpret with the Bayesian credible intervals in figure D.</div>

  <h2>Reproducibility</h2>
  <p class="note">All panels regenerate from <code>publicacion/_v3_master_table.csv</code> and the metrics JSONs via
  <code>python notebooks/08_extended_analysis.py</code> — no raw download or MCMC required.
  Paper &amp; DOI: <a href="https://doi.org/10.5281/zenodo.21015202">10.5281/zenodo.21015202</a>.</p>
</div>
<footer>HRP Artemis II · open release (MIT) · María Jesús Puerta Ángulo · Inspiration4 used as a public analogue
(not Artemis II flight data).</footer>
</body></html>"""
    out = PUB / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


# ----------------------------------------------------------------- main
def run():
    print("=" * 60, "\nHRP Artemis II — extended analysis + dashboard\n", "=" * 60)
    df = load_master()
    feat = numeric_features(df)
    print(f"master table: {df.shape}, numeric features: {len(feat)}")

    clf = fig_classifier_comparison(); print("  fig_E classifier comparison -> ok")
    movers = fig_top_movers(df, feat); print("  fig_F top movers -> ok")
    pca = fig_pca_phase(df, feat); print(f"  fig_G PCA -> PC1 {pca['pc1_var']*100:.0f}% PC2 {pca['pc2_var']*100:.0f}%")
    miss = fig_missingness(df, feat); print("  fig_H missingness -> ok")

    ext = {
        "generated": datetime.date.today().isoformat(),
        "classifier_summary": clf,
        "pca": pca,
        "missingness_by_modality": miss,
        "top_movers_preview": [(str(n), float(v)) for n, v in movers.head(12).items()],
    }
    (PUB / "_extended_metrics.json").write_text(json.dumps(ext, indent=2), encoding="utf-8")
    print("  wrote publicacion/_extended_metrics.json")

    dash = build_dashboard(ext, clf)
    print(f"  wrote {dash}")
    print("\nDONE.")


if __name__ == "__main__":
    run()
