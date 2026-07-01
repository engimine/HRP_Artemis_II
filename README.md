# Bayesian multi-modal latent-state inference for n=4 deep-space crew analogues

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21015202.svg)](https://doi.org/10.5281/zenodo.21015202)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Methodology pipeline applied to the SpaceX Inspiration4 public archive as a
surrogate analogue for the NASA Artemis II Human Research Data Methodology
Challenge. Released as an open-science contribution to the NASA HRP community.

> **Open-source release** (MIT). Published as a post-challenge open contribution
> to the NASA HRP community, after the Artemis II Methodology Challenge submission
> window closed on 2026-06-05.

---

## Author

María Jesús Puerta Ángulo · Independent researcher · Mining and AI Engineer · Spain
Correspondence: mariajesuspuertaangulo@gmail.com

## Eligibility

The Artemis II Methodology Challenge is restricted by the America COMPETES Act
to U.S. citizens, permanent residents and qualifying U.S. entities. As a Spanish
national the author was ineligible to enter, and presents the methodology as an
open contribution to the broader NASA HRP community rather than a competitive
submission. No NASA endorsement is claimed or implied.

---

## In plain language

**The problem.** Imagine learning a rule by watching only **four people**, while
measuring **a thousand things** about each (blood, sleep, heart, microbes…). With
so few people and so many measurements, ordinary statistics break down: they
return answers that *look* confident but are mostly noise. That is exactly NASA's
situation with astronauts — very few subjects, measured in enormous detail.

**What this work does.**
1. *Tidies the chaos.* The public data from a comparable real mission (four
   civilians who flew to space, *Inspiration4*) was scattered across **nine
   incompatible databases**. This work merges them into **one clean, aligned
   table** — groundwork nobody had done before.
2. *Uses a method that works with little data.* Instead of a falsely confident
   yes/no, the model reports its **honest confidence**: *"this I know; this I do
   not."* Admitting uncertainty is, in science, worth a great deal.
3. *Finds useful things.* For example: **18 cardiovascular measurements carry the
   same information as 406 urine markers** — you can measure far less and learn
   the same.

**What it is for.** When NASA releases the *real* Artemis II crew data, there is a
**tested recipe ready** to analyse it without wasting months or drawing false
conclusions. For Moon/Mars missions it helps decide **what is worth measuring**
(fewer instruments = less mass = lower cost). On Earth, the same "few subjects,
many measurements" problem appears in **rare diseases** and **personalised
medicine**.

**Why it did not exist.** The data were public but **fragmented**, and nobody had
done the unglamorous work of unifying them. Faced with only four subjects, most
analyses either give up or take shortcuts that mislead; doing it *honestly*,
reproducibly and **openly** is rare. And since the official challenge was closed
to non-US participants, no eligible competitor had an incentive to release this
openly — so it is released here, free, for the whole community.

> **In one sentence:** a problem that breaks ordinary statistics, solved honestly
> on the best available public data, with the tool left ready and open for when the
> real astronaut data arrives.

---

## What's in this repo

```
HRP_Artemis_II/
├── notebooks/                      Reproducible pipeline (Python 3.11)
│   ├── 01_inventory.py             Scan datasets, write inventory
│   ├── 02_demo_pipeline.py         Per-modality LOSO baseline (kNN + Beta-Binom)
│   ├── 03_bayesian_hierarchical.py Hierarchical logistic + horseshoe shrinkage
│   ├── 04_multimodal_factor.py     Factor model with shared latent state
│   ├── 05_append_v3_to_html.py     Aggregate results into report
│   ├── 07_baseline_comparison.py   Classifier sensitivity (kNN vs RF vs SVM)
│   ├── 08_extended_analysis.py     Extended figures + dashboard (no MCMC needed)
│   └── mcmc_diagnostics.py         Drop-in R-hat / ESS / divergence reporter
├── dataset/                        (gitignored — see Data Availability)
│   └── _download_osdr_inspiration4_v2.py  Public OSDR REST API download script
├── publicacion/
│   ├── dashboard.html              ★ Self-contained analysis dashboard (open in a browser)
│   ├── _demo_report_2026-05-09.html  Full methodology report
│   ├── _extended_metrics.json        Extended-analysis outputs (top movers, PCA)
│   ├── _baseline_comparison_metrics.json  Classifier comparison
│   ├── _bayesian_metrics_*.json      Hierarchical posterior summaries
│   ├── _v3_multimodal_metrics_*.json Factor model posteriors
│   ├── _v3_master_table.csv          Aligned 24×1115 multi-modal grid
│   ├── figures_publicacion/          fig_A … fig_H + composite
│   └── paper/
│       ├── arxiv_nora_v1/paper.pdf            Academic paper — PRIMARY
│       └── arxiv_nora_v1/paper.tex            LaTeX source (elsarticle) + references.bib
├── tests/test_smoke.py             Fast reproducibility smoke tests (pytest)
├── requirements.txt                Pinned environment
├── README.md                       This file
├── LICENSE                         MIT
└── .gitignore
```

---

## Data Availability

All data are public. Datasets used:

| Source | Identifier | Access |
|---|---|---|
| NASA OSDR | OSD-569, OSD-570, OSD-572, OSD-573, OSD-574, OSD-575, OSD-630, OSD-656, OSD-687 | osdr.nasa.gov |
| PhysioNet | MMASH (Multilevel Monitoring of Activity and Sleep in Healthy people) | DOI 10.13026/cerq-fc86 |
| CDC NCHS | NHANES 2021–2023 cycle | cdc.gov/nchs/nhanes |

The 12.4 GB of processed NASA OSDR files used here remain hosted on osdr.nasa.gov
and are NOT redistributed in this repository. The download script
`dataset/_download_osdr_inspiration4_v2.py` retrieves them via the public OSDR REST API.

---

## Reproducibility

```bash
# 1. Clone
git clone https://github.com/engimine/HRP_Artemis_II.git
cd HRP_Artemis_II

# 2. Python environment (pinned)
pip install -r requirements.txt

# 3. Fast path — reproduce the figures, significance tests + dashboard WITHOUT the 12 GB.
#    (Uses the committed master table and metrics; no MCMC.)
python notebooks/09_significance.py        # per-feature FDR + permutation test
python notebooks/08_extended_analysis.py   # extended figures + dashboard
pytest -q                                  # smoke tests
#    → open publicacion/dashboard.html in a browser for the full visual summary.

# 4. Full path — reconstruct everything from source data:
python dataset/_download_osdr_inspiration4_v2.py   # ~12 GB, ~30 min
python notebooks/01_inventory.py                   # inventory + master table
python notebooks/02_demo_pipeline.py               # per-modality LOSO baseline
python notebooks/07_baseline_comparison.py         # kNN vs RF vs SVM sensitivity
python notebooks/03_bayesian_hierarchical.py       # hierarchical Bayesian + MCMC diagnostics
python notebooks/04_multimodal_factor.py           # multi-modal factor model
python notebooks/05_append_v3_to_html.py           # aggregate into report
```

Pinned software stack (see `requirements.txt`): Python 3.11 · numpy 2.2.6 · scipy 1.16.2 ·
pandas 2.3.3 · scikit-learn 1.7.2 · PyMC 5.28.5 · NumPyro 0.21.0 · JAX 0.10.0 · ArviZ 0.23.4 ·
matplotlib 3.10.6. Hardware: consumer x86_64 laptop, single CPU core. Wall-clock < 15 minutes.

## Dashboard

`publicacion/dashboard.html` is a self-contained visual summary (KPIs + all figures +
top biomarker shifts) — open it directly in a browser for at-a-glance visibility of the
results. Regenerate it any time with `python notebooks/08_extended_analysis.py`.

## Results at a glance

- **Central result:** *no individual biomarker is significant* — 0 of 1112 features survive
  Benjamini–Hochberg correction — yet the **fused multi-modal representation separates pre- from
  post-flight state significantly above chance** (LOSO accuracy 0.71; label-permutation *p = 0.026*).
  Multi-modal fusion recovers a spaceflight signature that per-feature analysis cannot see.
- **Feature efficiency:** an 18-feature cardiovascular panel matches a 406-feature urine panel
  (0.679 vs 0.688), and this compactness is robust across classifiers.
- **Classifier sensitivity:** modality informativeness is estimator-dependent — a random forest
  lifts both microbiome modalities from chance to 0.56–0.65 and the serum panel to 0.804.
- The hierarchical Bayesian model is **honestly calibrated** (Brier 0.224; mean 95% CrI width 0.775).
- Everything is reported with the *n = 4* uncertainty intact — no over-claimed effect sizes.

---

## Citing this work

```
Puerta Ángulo, M. J. (2026). Bayesian multi-modal latent-state inference for
n=4 deep-space crew analogues: a reproducible methodology pipeline for the
NASA Artemis II Human Research Data Challenge. Zenodo.
https://doi.org/10.5281/zenodo.21015202
```

---

## License

MIT License. See `LICENSE` for the full text.

The author thanks the Inspiration4 crew (C001–C004) for the public release of
their biomedical data, the NASA Open Science Data Repository team, and the
PyMC, NumPyro and ArviZ communities for the probabilistic-programming
infrastructure on which this work depends.
