"""HRP Artemis II — Classifier baseline comparison (appendix).

Reviewer-requested robustness check: re-runs the EXACT same per-modality
leave-one-subject-out (LOSO) protocol used for the kNN baseline
(see 02_demo_pipeline.py), but compares three classifiers head-to-head:

  - kNN (k=5)                  -- the baseline reported in the main paper
  - Random Forest (300 trees)  -- ensemble, non-linear
  - SVM (RBF kernel)           -- margin-based, non-linear

Same data, same z-scoring, same LOSO folds, same Beta-Binomial 95% CrI.
Only the classifier changes. Numbers are computed, never hand-entered.

Output:
  publicacion/_baseline_comparison_metrics.json
  prints a Markdown/LaTeX-ready comparison table
"""
from __future__ import annotations
import io, sys, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\engim\marte\HRP_Artemis_II")
PUB = ROOT / "publicacion"
SEED = 2026

# ---- import the loaders/helpers from 02_demo_pipeline.py (module name has a digit) ----
spec = importlib.util.spec_from_file_location("demo02", ROOT / "notebooks" / "02_demo_pipeline.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def loso_acc(X: pd.DataFrame, y: pd.Series, groups: pd.Series, clf_factory) -> dict:
    """LOSO accuracy + Beta-Binomial 95% CrI for an arbitrary classifier factory."""
    feat = X.fillna(0.0).to_numpy()
    yy = y.astype(str).to_numpy()
    sids = groups.astype(str).to_numpy()
    accs = []
    for held in sorted(set(sids)):
        tr, te = sids != held, sids == held
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = clf_factory(len(feat[tr]))
        clf.fit(feat[tr], yy[tr])
        accs.append(accuracy_score(yy[te], clf.predict(feat[te])))
    n_total = int(feat.shape[0])
    acc = float(np.mean(accs)) if accs else None
    hits = int(round(acc * n_total)) if acc is not None else 0
    mean, lo, hi = m.bayesian_proportion_ci(hits, n_total)
    return {"acc": None if acc is None else round(acc, 3),
            "ci95": [round(lo, 3), round(hi, 3)], "folds": len(accs),
            "n": n_total, "p": int(feat.shape[1])}


CLASSIFIERS = {
    "kNN": lambda ntr: KNeighborsClassifier(n_neighbors=max(1, min(5, ntr - 1))),
    "RandomForest": lambda ntr: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "SVM_RBF": lambda ntr: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
}


def build_modalities() -> dict:
    """Replicate exactly the per-modality (X, y, groups) construction of 02_demo_pipeline.run()."""
    OSDR = m.OSDR
    mods: dict = {}

    # --- flat OSD-575/656/569 modalities ---
    flat = [
        ("urine_alamar",  OSDR / "OSD-656_urine_inflammation",        "*urine.immune.AlamarPanel_TRANSFORMED.csv"),
        ("cardio_eve",    OSDR / "OSD-575_blood_metabolic_cytokines", "*cardiovascular_EvePanel_TRANSFORMED.csv"),
        ("alamar_immune", OSDR / "OSD-575_blood_metabolic_cytokines", "*AlamarPanel_TRANSFORMED.csv"),
        ("immune_eve",    OSDR / "OSD-575_blood_metabolic_cytokines", "*immune_EvePanel_TRANSFORMED.csv"),
        ("metabolic_cmp", OSDR / "OSD-575_blood_metabolic_cytokines", "*CMP_TRANSFORMED.csv"),
        ("blood_cbc",     OSDR / "OSD-569_whole_blood_seq",           "*CBC_TRANSFORMED.csv"),
    ]
    for label, d, glob in flat:
        if not d.exists():
            continue
        t = m.load_osd575_modality(d, glob, label)
        if t.empty:
            continue
        valid = t.dropna(subset=["subject", "spaceflight"])
        if valid.empty or valid["subject"].nunique() < 2:
            continue
        feat_cols = [c for c in valid.columns
                     if c not in ("Sample ID", "Sample Name", "subject", "timepoint", "spaceflight", "_modality")
                     and pd.api.types.is_numeric_dtype(valid[c])]
        if not feat_cols:
            continue
        mods[label] = (m.zscore(valid[feat_cols].fillna(0.0)), valid["spaceflight"], valid["subject"])

    # --- microbiome taxonomy + pathway (wide) ---
    osd572 = OSDR / "OSD-572_oral_nasal_skin_microbiome"
    for name, table, id_hint in [("taxonomy", m.load_microbiome_taxonomy(osd572), "clade_name"),
                                 ("pathway", m.load_pathway_abundances(osd572), "Pathway")]:
        if table.empty:
            continue
        id_col = id_hint if id_hint in table.columns else table.columns[0]
        sample_cols = [c for c in table.columns if m._SAMPLE_RE.search(str(c))]
        if not sample_cols:
            continue
        X_wide = table.set_index(id_col)[sample_cols].apply(pd.to_numeric, errors="coerce").T
        meta_rows = pd.DataFrame([m.parse_sample_col(c) for c in X_wide.index], index=X_wide.index)
        valid = X_wide.join(meta_rows).dropna(subset=["subject", "spaceflight"])
        if valid.empty:
            continue
        feat_cols = [c for c in valid.columns
                     if c not in ("subject", "spaceflight", "timepoint", "timepoint_code", "site_code")]
        mods[name] = (m.zscore(valid[feat_cols].fillna(0.0)), valid["spaceflight"], valid["subject"])

    # --- MMASH terrestrial proxy (n=22) ---
    mmash_root = ROOT / "dataset" / "physionet_mmash" / "multilevel-monitoring-of-activity-and-sleep-in-healthy-people-1.0.0" / "DataPaper"
    if mmash_root.exists():
        mmash_df = m.load_mmash_summary(mmash_root)
        if not mmash_df.empty and "rmssd_ms" in mmash_df.columns and len(mmash_df) >= 4:
            thr = mmash_df["rmssd_ms"].median()
            labels = (mmash_df["rmssd_ms"] >= thr).map({True: "high_rmssd", False: "low_rmssd"})
            feat_cols = [c for c in mmash_df.columns if c not in ("subject", "rmssd_ms")
                         and pd.api.types.is_numeric_dtype(mmash_df[c])]
            if feat_cols:
                mods["mmash_proxy"] = (m.zscore(mmash_df[feat_cols].fillna(0.0)), labels, mmash_df["subject"])
    return mods


def main():
    mods = build_modalities()
    order = ["urine_alamar", "cardio_eve", "mmash_proxy", "alamar_immune",
             "immune_eve", "metabolic_cmp", "blood_cbc", "pathway", "taxonomy"]
    pretty = {"urine_alamar": "Urine cytokine (Alamar)", "cardio_eve": "Cardiovascular (EVE)",
              "mmash_proxy": "MMASH proxy", "alamar_immune": "Serum cytokine (Alamar)",
              "immune_eve": "Immune EVE", "metabolic_cmp": "Metabolic CMP", "blood_cbc": "Blood CBC",
              "pathway": "Microbiome pathway", "taxonomy": "Microbiome taxonomy"}

    results: dict = {}
    print(f"\n{'Modality':<26}{'p':>6}{'n':>5}   {'kNN':>20}{'RandomForest':>22}{'SVM-RBF':>20}")
    for label in order:
        if label not in mods:
            continue
        X, y, g = mods[label]
        row = {}
        for cname, factory in CLASSIFIERS.items():
            row[cname] = loso_acc(X, y, g, factory)
        results[label] = {"name": pretty.get(label, label), **row}
        def fmt(r):
            return f"{r['acc']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]" if r['acc'] is not None else "—"
        k, rf, sv = row["kNN"], row["RandomForest"], row["SVM_RBF"]
        print(f"{pretty.get(label, label):<26}{k['p']:>6}{k['n']:>5}   {fmt(k):>20}{fmt(rf):>22}{fmt(sv):>20}")

    out = PUB / "_baseline_comparison_metrics.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    # LaTeX-ready rows
    print("\n--- LaTeX rows (Modality & kNN & RF & SVM) ---")
    for label in order:
        if label not in results:
            continue
        r = results[label]
        def cell(c):
            x = r[c]
            return f"{x['acc']:.3f}" if x['acc'] is not None else "--"
        print(f"{r['name']} & {cell('kNN')} & {cell('RandomForest')} & {cell('SVM_RBF')} \\\\")


if __name__ == "__main__":
    main()
