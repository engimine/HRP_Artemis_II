"""Download Inspiration4 processed data + metadata from NASA OSDR.
Skips raw FASTQ (huge); keeps everything <50 MB so we get TSVs, abundance tables,
metadata ISA zips and READMEs for downstream stats with PyMC + PyTorch.
"""
import sys, io, os, json, re, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"C:\Users\engim\marte\HRP_Artemis_II\dataset\nasa_osdr_inspiration4"
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
MAX_SIZE = 50_000_000  # 50 MB cap


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url, dst):
    if os.path.exists(dst):
        return os.path.getsize(dst)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            buf = r.read(64 * 1024)
            if not buf:
                break
            f.write(buf)
    return os.path.getsize(dst)


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._+\-]", "_", s)


def main():
    os.makedirs(ROOT, exist_ok=True)
    summary = []
    for osd_id, label in OSD_IDS:
        num = osd_id.replace("OSD-", "")
        api = f"https://osdr.nasa.gov/osdr/data/osd/files/{num}"
        print(f"\n=== {osd_id} ({label}) ===")
        print(f"API: {api}")
        try:
            data = fetch_json(api)
        except Exception as e:
            print(f"  [API error] {e}")
            summary.append((osd_id, "API error", 0, 0))
            continue
        studies = data.get("studies", {})
        all_files = []
        for sk, sv in studies.items():
            all_files.extend(sv.get("study_files", []))
        # Filter: small only
        small = [f for f in all_files
                 if isinstance(f.get("file_size"), int) and 0 < f["file_size"] <= MAX_SIZE]
        print(f"  total files {len(all_files)}, ≤{MAX_SIZE//1_000_000}MB: {len(small)}")

        out_dir = os.path.join(ROOT, f"{osd_id}_{label}")
        os.makedirs(out_dir, exist_ok=True)
        n_ok = 0; total_bytes = 0
        for f in small:
            rurl = f.get("remote_url", "")
            if not rurl:
                continue
            full = "https://osdr.nasa.gov" + rurl
            fname = safe_name(f.get("file_name", "file"))
            dst = os.path.join(out_dir, fname)
            try:
                size = download(full, dst)
                n_ok += 1
                total_bytes += size
            except Exception as e:
                print(f"  [skip] {fname}: {e}")
        print(f"  -> {n_ok}/{len(small)} files, {total_bytes/1e6:.1f} MB in {out_dir}")
        summary.append((osd_id, label, n_ok, total_bytes))

    print("\n" + "=" * 70)
    print(" SUMMARY ")
    print("=" * 70)
    for osd_id, label, n, tot in summary:
        print(f"  {osd_id:8s} {label:30s} {n:4d} files  {tot/1e6:7.1f} MB")
    grand = sum(t for _, _, _, t in summary)
    print(f"\n  TOTAL: {grand/1e6:.1f} MB")


if __name__ == "__main__":
    main()
