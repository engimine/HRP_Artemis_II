# Bayesian multi-modal latent-state inference for n=4 deep-space crew analogues

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

## What's in this repo

```
HRP_Artemis_II/
├── notebooks/                      Reproducible pipeline (Python 3.11)
│   ├── 01_inventory.py             Scan datasets, write _INVENTORY.md
│   ├── 02_demo_pipeline.py         Per-modality LOSO baseline (kNN + Beta-Binom)
│   ├── 03_bayesian_hierarchical.py Hierarchical logistic + horseshoe shrinkage
│   ├── 04_multimodal_factor.py     Factor model with shared latent state
│   └── 05_append_v3_to_html.py     Aggregate results into report
├── dataset/                        (gitignored — see Data Availability)
│   └── _download_osdr_inspiration4_v2.py  Public OSDR REST API download script
├── publicacion/
│   ├── _demo_report_2026-05-09.html  Full methodology report
│   ├── _demo_metrics_*.json          Per-modality LOSO metrics
│   ├── _bayesian_metrics_*.json      Hierarchical posterior summaries
│   ├── _v3_multimodal_metrics_*.json Factor model posteriors
│   ├── _v3_master_table.csv          Aligned 24×1112 multi-modal grid
│   └── paper/
│       ├── arxiv_nora_v1/paper.pdf            Academic paper — PRIMARY (arXiv-NORA preset)
│       └── arxiv_nora_v1/paper.tex            LaTeX source (elsarticle) + references.bib
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

# 2. Python environment
pip install pandas scikit-learn numpy scipy matplotlib pymc numpyro arviz pyreadstat pymupdf

# 3. Download datasets (~12 GB, ~30 minutes on a residential connection)
python dataset/_download_osdr_inspiration4_v2.py

# 4. Inventory + master table
python notebooks/01_inventory.py

# 5. Per-modality LOSO baseline
python notebooks/02_demo_pipeline.py

# 6. Hierarchical Bayesian on cardiovascular EvePanel
python notebooks/03_bayesian_hierarchical.py

# 7. Multi-modal factor model
python notebooks/04_multimodal_factor.py

# 8. Aggregate into report
python notebooks/05_append_v3_to_html.py
```

Software stack: Python 3.11.2 · PyMC 5.28.5 · NumPyro 0.21.0 · JAX 0.10.0 ·
ArviZ 0.23.4 · pandas, scikit-learn, numpy, scipy, matplotlib.
Hardware: consumer x86_64 laptop, single CPU core. Wall-clock < 15 minutes.

---

## Citing this work

```
Puerta Ángulo, M. J. (2026). Bayesian multi-modal latent-state inference for
n=4 deep-space crew analogues: a reproducible methodology pipeline for the
NASA Artemis II Human Research Data Challenge. Open-source release,
github.com/engimine/HRP_Artemis_II. DOI: to be assigned.
```

---

## License

MIT License. See `LICENSE` for the full text.

The author thanks the Inspiration4 crew (C001–C004) for the public release of
their biomedical data, the NASA Open Science Data Repository team, and the
PyMC, NumPyro and ArviZ communities for the probabilistic-programming
infrastructure on which this work depends.
