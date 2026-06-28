"""Inspiration4 OSD downloader — v2 (2026-05-09).

Fixes vs v1:
  - Logs each step to a file, never silent.
  - Retry with exponential backoff for transient failures.
  - Skips OSD-572 (already populated 3 GB on disk).
  - Cap 100 MB per file (was 50 MB) to capture mid-size processed tables
    while still skipping raw FASTQ / BAM that are >500 MB each.
  - Parallel download (4 workers) per OSD.
  - Resume safe: skips file if already on disk with non-zero size.
  - Per-OSD summary even on failure.
"""
from __future__ import annotations
import sys, io, os, json, re, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"C:\Users\engim\marte\HRP_Artemis_II\dataset\nasa_osdr_inspiration4"
LOG_PATH = os.path.join(ROOT, "_download_log_2026-05-09.txt")
MAX_SIZE = 100_000_000  # 100 MB
WORKERS = 4
RETRIES = 3
SKIP_OSDS = {"OSD-572"}  # already complete

OSD_IDS = [
    ("OSD-572", "oral_nasal_skin_microbiome"),
    ("OSD-575", "blood_metabolic_cytokines"),
    ("OSD-630", "stool_microbiome"),
    ("OSD-656", "urine_inflammation"),
    ("OSD-569", "whole_blood_seq"),
    ("OSD-570", "pbmc_singlecell"),
    ("OSD-573", "dragon_capsule_swabs"),
    ("OSD-574", "deltoid_skin"),
    ("OSD-687", "tcell_histones"),
]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_json(url: str) -> dict:
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            log(f"  fetch_json retry {attempt+1}/{RETRIES} in {wait}s ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise last_err  # type: ignore


def download(url: str, dst: str, expected: int) -> tuple[bool, int]:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True, os.path.getsize(dst)
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            tmp = dst + ".part"
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                while True:
                    buf = r.read(64 * 1024)
                    if not buf:
                        break
                    f.write(buf)
            os.replace(tmp, dst)
            return True, os.path.getsize(dst)
        except Exception as e:
            last_err = e
            try:
                os.remove(dst + ".part")
            except OSError:
                pass
            wait = 2 ** attempt
            time.sleep(wait)
    log(f"    [FAIL after {RETRIES}] {os.path.basename(dst)}: {last_err}")
    return False, 0


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+\-]", "_", s)


def process_osd(osd_id: str, label: str) -> tuple[str, str, int, int, int]:
    if osd_id in SKIP_OSDS:
        log(f"\n=== {osd_id} ({label}) SKIPPED (already on disk) ===")
        return osd_id, label, 0, 0, 0

    log(f"\n=== {osd_id} ({label}) ===")
    num = osd_id.replace("OSD-", "")
    api = f"https://osdr.nasa.gov/osdr/data/osd/files/{num}"

    try:
        data = fetch_json(api)
    except Exception as e:
        log(f"  [API fatal] {e}")
        return osd_id, label, 0, 0, 0

    studies = data.get("studies", {})
    all_files = []
    for sk, sv in studies.items():
        all_files.extend(sv.get("study_files", []))

    small = [f for f in all_files
             if isinstance(f.get("file_size"), int) and 0 < f["file_size"] <= MAX_SIZE]
    log(f"  total files: {len(all_files)} · within {MAX_SIZE//1_000_000} MB cap: {len(small)}")

    out_dir = os.path.join(ROOT, f"{osd_id}_{label}")
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for f in small:
        rurl = f.get("remote_url", "")
        if not rurl:
            continue
        full = "https://osdr.nasa.gov" + rurl
        fname = safe_name(f.get("file_name", "file"))
        dst = os.path.join(out_dir, fname)
        tasks.append((full, dst, f.get("file_size", 0)))

    n_ok, n_fail, total_bytes = 0, 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download, url, dst, sz): (url, dst) for url, dst, sz in tasks}
        for fut in as_completed(futures):
            ok, size = fut.result()
            if ok:
                n_ok += 1
                total_bytes += size
            else:
                n_fail += 1

    log(f"  -> {n_ok}/{len(tasks)} ok ({total_bytes/1e6:.1f} MB), {n_fail} failed")
    return osd_id, label, n_ok, n_fail, total_bytes


def main() -> None:
    os.makedirs(ROOT, exist_ok=True)
    open(LOG_PATH, "w", encoding="utf-8").close()  # truncate fresh log
    log(f"Starting Inspiration4 download (v2) — cap {MAX_SIZE//1_000_000} MB, {WORKERS} workers")
    log(f"Log file: {LOG_PATH}")

    summary = []
    for osd_id, label in OSD_IDS:
        try:
            r = process_osd(osd_id, label)
            summary.append(r)
        except Exception as e:
            log(f"  [OSD-level fatal] {osd_id}: {type(e).__name__}: {e}")
            summary.append((osd_id, label, 0, 0, 0))

    log("\n" + "=" * 70)
    log(" SUMMARY ")
    log("=" * 70)
    grand_ok, grand_fail, grand_bytes = 0, 0, 0
    for osd_id, label, n_ok, n_fail, tot in summary:
        flag = " (skip)" if osd_id in SKIP_OSDS else ""
        log(f"  {osd_id:8s} {label:30s} ok={n_ok:4d} fail={n_fail:3d}  {tot/1e6:7.1f} MB{flag}")
        grand_ok += n_ok
        grand_fail += n_fail
        grand_bytes += tot
    log(f"\n  TOTAL new: {grand_ok} ok / {grand_fail} fail / {grand_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
