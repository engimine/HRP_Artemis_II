"""HRP Artemis II — Inventory script.

Scans every dataset folder under ./dataset/ and produces a markdown report
mapping samples → subjects → timepoints → modalities. Auto-runs end-to-end.

Outputs:
  - dataset/_INVENTORY.md                (human-readable summary)
  - dataset/_inventory_samples.csv       (long-format sample table all OSDs)

Usage:
  python notebooks/01_inventory.py
"""
from __future__ import annotations
import io, os, sys, zipfile, json
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(r"C:\Users\engim\marte\HRP_Artemis_II")
DATASET = ROOT / "dataset"
OSDR = DATASET / "nasa_osdr_inspiration4"
OUT_MD = DATASET / "_INVENTORY.md"
OUT_CSV = DATASET / "_inventory_samples.csv"


def parse_isa_zip(zip_path: Path) -> pd.DataFrame | None:
    """Read s_OSD-XXX.txt from an ISA zip and return the sample table."""
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            sample_files = [n for n in z.namelist() if n.startswith("s_") and n.endswith(".txt")]
            if not sample_files:
                return None
            with z.open(sample_files[0]) as f:
                return pd.read_csv(f, sep="\t", encoding="latin1", low_memory=False)
    except Exception as e:
        print(f"  [ISA parse error] {zip_path.name}: {e}")
        return None


def find_isa_zip(osd_dir: Path) -> Path | None:
    for f in osd_dir.glob("*ISA*.zip"):
        return f
    return None


def osd_summary(osd_dir: Path) -> dict:
    files = list(osd_dir.glob("*"))
    n_files = len(files)
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    ext_counts: dict[str, int] = {}
    for f in files:
        if f.is_file():
            ext = f.suffix.lower().lstrip(".") or "noext"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    isa_zip = find_isa_zip(osd_dir)
    sample_df = parse_isa_zip(isa_zip) if isa_zip else None
    info: dict = {
        "name": osd_dir.name,
        "n_files": n_files,
        "size_mb": round(total_mb, 1),
        "ext_counts": ext_counts,
        "isa_present": isa_zip is not None,
        "n_samples": None,
        "subjects": None,
        "timepoints": None,
        "sites": None,
        "spaceflight": None,
    }
    if sample_df is not None and not sample_df.empty:
        info["n_samples"] = len(sample_df)
        if "Source Name" in sample_df.columns:
            info["subjects"] = sorted(sample_df["Source Name"].dropna().unique().tolist())
        for col, key in [
            ("Factor Value[Time]", "timepoints"),
            ("Factor Value[Sample Location]", "sites"),
            ("Factor Value[Spaceflight]", "spaceflight"),
        ]:
            if col in sample_df.columns:
                vals = sample_df[col].dropna().astype(str).unique().tolist()
                info[key] = sorted(vals)
    return info, sample_df


def main() -> None:
    print(f"Scanning {OSDR}")
    rows: list[dict] = []
    long_samples: list[pd.DataFrame] = []

    for osd_dir in sorted(OSDR.glob("OSD-*")):
        if not osd_dir.is_dir():
            continue
        info, sdf = osd_summary(osd_dir)
        rows.append(info)
        if sdf is not None and not sdf.empty:
            sdf2 = sdf.copy()
            sdf2.insert(0, "_OSD", osd_dir.name)
            long_samples.append(sdf2)
        print(f"  {osd_dir.name:55s} {info['n_files']:4d} files {info['size_mb']:8.1f} MB · subjects={info['subjects']}")

    # Other (non-OSDR) datasets: physionet, NHANES, exam stress
    other = []
    for sub in ("physionet_mmash", "physionet_wearable_exam_stress", "nhanes_2021-2023"):
        d = DATASET / sub
        if not d.exists():
            continue
        files = list(d.rglob("*"))
        n_files = sum(1 for f in files if f.is_file())
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
        other.append({"name": sub, "n_files": n_files, "size_mb": round(total_mb, 1)})
        print(f"  {sub:55s} {n_files:4d} files {total_mb:8.1f} MB")

    # Write CSV with long-format sample table (concat ISA tables across OSDs)
    if long_samples:
        joined = pd.concat(long_samples, ignore_index=True, sort=False)
        joined.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"\nWrote {OUT_CSV}  ({len(joined)} rows × {len(joined.columns)} cols)")

    # Markdown report
    md = ["# HRP Artemis II — Inventario de datasets",
          "",
          f"_Generated: 2026-05-09 · script `notebooks/01_inventory.py`_",
          "",
          "## NASA OSDR Inspiration4",
          "",
          "| OSD | Files | Size | ISA | n samples | subjects | timepoints | sites | spaceflight |",
          "|---|---:|---:|:-:|---:|---|---|---|---|"]
    for r in rows:
        subj = ", ".join(r["subjects"][:6]) + ("…" if r["subjects"] and len(r["subjects"]) > 6 else "") if r["subjects"] else "—"
        tps = ", ".join(r["timepoints"][:6]) + ("…" if r["timepoints"] and len(r["timepoints"]) > 6 else "") if r["timepoints"] else "—"
        sites = ", ".join(r["sites"][:6]) + ("…" if r["sites"] and len(r["sites"]) > 6 else "") if r["sites"] else "—"
        sf = ", ".join(r["spaceflight"]) if r["spaceflight"] else "—"
        md.append(f"| {r['name']} | {r['n_files']} | {r['size_mb']} MB | {'✓' if r['isa_present'] else '✗'} | "
                  f"{r['n_samples'] if r['n_samples'] else '—'} | {subj} | {tps} | {sites} | {sf} |")
    md += ["",
           "## Otros datasets (proxies)",
           "",
           "| Dataset | Files | Size |",
           "|---|---:|---:|"]
    for o in other:
        md.append(f"| {o['name']} | {o['n_files']} | {o['size_mb']} MB |")
    md += ["",
           "## Notas",
           "",
           "- The author is ineligible for the HRP Challenge (US-only eligibility); this work is released as an open post-challenge contribution.",
           "- Los OSDs con `ISA ✗` o `0 files` necesitan re-descarga (script `_download_osdr_inspiration4_v2.py`).",
           "- Subject naming: `C001-C004` = 4 Inspiration4 astronauts. Controls: `Communal`, `Laboratory`, `Swab Water`, `Open Air`, `Zymo Shield`, etc.",
           "- Timepoints clásicos: `L-92`, `L-44`, `L-3` (pre-flight) · `FD2`, `FD3` (in-flight) · `R+1`, `R+45`, `R+82` (post-flight).",
           ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
