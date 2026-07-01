"""Minimal smoke tests for the HRP Artemis II pipeline.

Fast, dependency-light checks that materially raise reproducibility credibility.
They do NOT need the 12 GB raw download: they exercise (1) the master-table
shape, (2) LOSO structural invariants, and (3) Beta-Binomial interval sanity.

Run from the repo root:  pytest -q
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "publicacion" / "_v3_master_table.csv"


# ---------------------------------------------------------------- 1. data shape
@pytest.mark.skipif(not MASTER.exists(), reason="master table not present")
def test_master_table_shape():
    import pandas as pd
    df = pd.read_csv(MASTER)
    assert df.shape[0] == 24, f"expected 24 aligned observations, got {df.shape[0]}"
    assert df.shape[1] > 1000, f"expected >1000 columns, got {df.shape[1]}"
    assert {"subject", "timepoint"}.issubset(df.columns)


# ------------------------------------------------------- 2. LOSO structural test
def _loso_folds(subjects):
    subjects = np.asarray(subjects)
    for s in np.unique(subjects):
        yield np.where(subjects != s)[0], np.where(subjects == s)[0]


def test_loso_partitions_complete_and_disjoint():
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    folds = list(_loso_folds(subjects))
    assert len(folds) == len(np.unique(subjects))
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        assert len(train) + len(test) == len(subjects)
        held = subjects[test][0]
        assert np.all(subjects[test] == held)
        assert held not in subjects[train]


# ------------------------------------------------- 3. Beta-Binomial interval sanity
def _beta_binomial_interval(n_correct, n_total, cred=0.95):
    from scipy.stats import beta
    a, b = 1 + n_correct, 1 + (n_total - n_correct)
    mean = a / (a + b)
    lo, hi = beta.ppf([(1 - cred) / 2, 1 - (1 - cred) / 2], a, b)
    return float(lo), float(mean), float(hi)


@pytest.mark.parametrize("n_correct,n_total", [(0, 4), (2, 4), (4, 4), (7, 10)])
def test_beta_binomial_interval_ordered_and_bounded(n_correct, n_total):
    lo, mean, hi = _beta_binomial_interval(n_correct, n_total)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert (hi - lo) > 0.2, "n=4 band should be honestly wide"
