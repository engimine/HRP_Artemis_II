"""
mcmc_diagnostics.py — Drop-in MCMC convergence reporting for HRP Artemis II.

Closes the gap between what paper.tex promises (R-hat, ESS bulk/tail, divergences)
and what the pipeline actually persists. Call it at the end of 03_bayesian_hierarchical.py
(and 04) right after sampling, and it writes a JSON the paper can cite.

Usage (numpyro path, as in notebook 03):
    import arviz as az
    from mcmc_diagnostics import report_diagnostics
    idata = az.from_numpyro(mcmc)                      # mcmc = numpyro NUTS run
    report_diagnostics(idata, out_path="publicacion/_mcmc_diagnostics_run.json",
                       target_accept=0.9, label="hierarchical_horseshoe_full")

Usage (PyMC path):
    report_diagnostics(idata, out_path="...json", target_accept=0.9, label="...")

Thresholds follow standard practice (Vehtari et al. 2021): R-hat < 1.01,
ESS bulk/tail > 400, zero divergences => PASS.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import arviz as az


def report_diagnostics(idata,
                       out_path: str,
                       target_accept: float | None = None,
                       label: str = "",
                       rhat_max: float = 1.01,
                       ess_min: float = 400.0) -> dict:
    """Compute and persist MCMC convergence diagnostics.

    Returns the diagnostics dict and writes it to `out_path` as JSON.
    """
    summ = az.summary(idata, kind="diagnostics")  # r_hat, ess_bulk, ess_tail per param

    rhat_worst = float(np.nanmax(summ["r_hat"].values)) if "r_hat" in summ else math.nan
    ess_bulk_min = float(np.nanmin(summ["ess_bulk"].values)) if "ess_bulk" in summ else math.nan
    ess_tail_min = float(np.nanmin(summ["ess_tail"].values)) if "ess_tail" in summ else math.nan

    # Divergences live in sample_stats when available (numpyro & pymc both populate it)
    n_divergent = None
    n_draws = None
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div = idata.sample_stats["diverging"].values
        n_divergent = int(np.sum(div))
        n_draws = int(div.size)

    passed = (
        (not math.isnan(rhat_worst) and rhat_worst < rhat_max)
        and (not math.isnan(ess_bulk_min) and ess_bulk_min > ess_min)
        and (not math.isnan(ess_tail_min) and ess_tail_min > ess_min)
        and (n_divergent is None or n_divergent == 0)
    )

    diagnostics = {
        "label": label,
        "rhat_max": round(rhat_worst, 4) if not math.isnan(rhat_worst) else None,
        "rhat_threshold": rhat_max,
        "ess_bulk_min": round(ess_bulk_min, 1) if not math.isnan(ess_bulk_min) else None,
        "ess_tail_min": round(ess_tail_min, 1) if not math.isnan(ess_tail_min) else None,
        "ess_threshold": ess_min,
        "n_divergent": n_divergent,
        "n_draws_total": n_draws,
        "target_accept": target_accept,
        "n_params_checked": int(summ.shape[0]),
        "converged": bool(passed),
        "verdict": "PASS" if passed else "REVIEW — re-run with more draws / higher target_accept",
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"[mcmc_diagnostics] {label}: {diagnostics['verdict']} "
          f"(R-hat_max={diagnostics['rhat_max']}, ESS_bulk_min={diagnostics['ess_bulk_min']}, "
          f"divergences={n_divergent}) -> {out}")
    return diagnostics


if __name__ == "__main__":
    # Self-check on a trivial well-mixed toy model so the file runs standalone.
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax.random as random

    def model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, num_chains=2, progress_bar=False)
    mcmc.run(random.PRNGKey(2026))
    idata = az.from_numpyro(mcmc)
    report_diagnostics(idata, out_path="_mcmc_diagnostics_selftest.json",
                       target_accept=0.9, label="selftest_normal")
