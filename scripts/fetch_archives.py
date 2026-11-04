"""Download fixed source archives with a retrieval log (URL, UTC time, HTTP status, bytes, SHA-256).

Usage:  python scripts/fetch_archives.py <group> [<group> ...]
Groups: napierone, zenodo_heic, govdocs1
Already-downloaded files whose SHA-256 matches the log are skipped, so the command can be re-run.
Downloaded files are data only: they are hashed and stored, never executed.
"""
import csv
import datetime as dt
import hashlib
import os
import sys
import time

import requests

from paths import RAW, MANIFESTS, USER_AGENT

NAPIER = "https://s3.eu-north-1.amazonaws.com/napierone.com/NapierOne/Data"
GROUPS = {
    # NapierOne (Davies, Macfarlane, Buchanan 2022, FSI:DI 40:301330); AWS Open Data bucket napierone.com
    "napierone": [
        ("napierone/archives/CSV-tiny.zip", f"{NAPIER}/CSV/CSV-tiny.zip"),
        ("napierone/archives/CSV-small.zip", f"{NAPIER}/CSV/CSV-small.zip"),
        ("napierone/archives/DOCX-tiny.zip", f"{NAPIER}/DOCX/DOCX-tiny.zip"),
        ("napierone/archives/PPTX-tiny.zip", f"{NAPIER}/PPTX/PPTX-tiny.zip"),
        ("napierone/archives/PDF-tiny.zip", f"{NAPIER}/PDF/PDF-tiny.zip"),
        ("napierone/archives/MP4-FROM-WEB-tiny.zip", f"{NAPIER}/MP4/MP4-FROM-WEB-tiny.zip"),
    ] + [(f"napierone/archives/{n}_zip_hashes.txt", f"{NAPIER}/{d}/{n}_zip_hashes.txt") for d, n in (
        ("CSV", "CSV-tiny"), ("CSV", "CSV-small"), ("DOCX", "DOCX-tiny"), ("PPTX", "PPTX-tiny"), ("PDF", "PDF-tiny"),
        ("MP4", "MP4-FROM-WEB-tiny"))],
    # Zenodo records (CC BY 4.0). Individual LEUKOS files are listed by scripts/collect_zenodo.py.
    "zenodo_heic": [
        ("zenodo/17640511/data.zip", "https://zenodo.org/records/17640511/files/data.zip?download=1"),
    ],
    # Govdocs1 (Garfinkel et al. 2009), official Digital Corpora download server
    "govdocs1": [(f"govdocs1/{t}.zip", f"https://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/{t}.zip")
                 for t in ("000", "001", "002")],
}
LOG = os.path.join(MANIFESTS, "archive_retrieval_log.csv")
FIELDS = ["local_path", "url", "retrieved_utc", "http_status", "bytes", "sha256"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_log():
    if not os.path.exists(LOG):
        return {}
    with open(LOG, newline="", encoding="utf-8") as f:
        return {r["local_path"]: r for r in csv.DictReader(f)}


def save_log(rows):
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def fetch(rel, url, log):
    dest = os.path.join(RAW, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if rel in log and os.path.exists(dest) and sha256_file(dest) == log[rel]["sha256"]:
        print("skip (verified)", rel, flush=True)
        return
    for attempt in range(5):
        try:
            tmp = dest + ".part"
            with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
                status = r.status_code
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            os.replace(tmp, dest)
            break
        except Exception as e:  # noqa: BLE001
            print(f"retry {attempt + 1} {rel}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"failed: {url}")
    log[rel] = {"local_path": rel, "url": url, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "http_status": status, "bytes": os.path.getsize(dest), "sha256": sha256_file(dest)}
    save_log(log)
    print("done", rel, log[rel]["bytes"], log[rel]["sha256"], flush=True)


def main():
    log = load_log()
    for g in sys.argv[1:]:
        for rel, url in GROUPS[g]:
            fetch(rel, url, log)
    print("FETCH COMPLETE", " ".join(sys.argv[1:]), flush=True)


if __name__ == "__main__":
    main()
