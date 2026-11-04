"""HEIC candidates from two versioned Zenodo records (CC BY 4.0).

  10.5281/zenodo.17640511  edltauSCI Multi-Device Image Dataset (Ozaman 2025): source-camera-identification photos;
                           HEIC files from iPhone 13 Pro Max and iPhone 17 Pro inside data.zip
  10.5281/zenodo.13623651  LEUKOS' dataset (Taglialatela 2024): photographs of 16th-17th c. Stamboeger, HEIC files
Record metadata is saved next to the files. Individual LEUKOS files are downloaded and checked against the MD5
published by Zenodo; data.zip (fetched by fetch_archives.py) is checked against its published MD5.

Fixed candidate order: all HEIC members of 17640511 sorted by SHA-256("heic-order-v1|" + member name), followed by
all HEIC files of 13623651 sorted the same way. Output: data/manifests/candidates_zenodo_heic.csv
"""
import csv
import datetime as dt
import hashlib
import json
import os
import zipfile

import requests

from paths import MANIFESTS, RAW, USER_AGENT

RECORDS = {"17640511": "zip", "13623651": "files"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def order_key(name):
    return hashlib.sha256(("heic-order-v1|" + name).encode("utf-8")).hexdigest()


def main():
    rows = []
    for rid, kind in RECORDS.items():
        d = os.path.join(RAW, "zenodo", rid)
        os.makedirs(os.path.join(d, "files"), exist_ok=True)
        rec = requests.get(f"https://zenodo.org/api/records/{rid}", timeout=60, headers={"User-Agent": USER_AGENT}).json()
        json.dump(rec, open(os.path.join(d, "record.json"), "w", encoding="utf-8"), indent=1)
        m = rec["metadata"]
        base = {"record_id": rid, "doi": rec.get("doi"), "concept_doi": rec.get("conceptdoi"),
                "record_title": m["title"], "record_version": m.get("version") or "", "publication_date":
                m.get("publication_date"), "creators": "; ".join(c["name"] for c in m.get("creators", [])),
                "license": (m.get("license") or {}).get("id"), "source_page": f"https://zenodo.org/records/{rid}"}
        if kind == "zip":
            zf = [f for f in rec["files"] if f["key"] == "data.zip"][0]
            path = os.path.join(RAW, "zenodo", rid, "data.zip")
            got = md5(path)
            if "md5:" + got != zf["checksum"]:
                raise RuntimeError(f"data.zip MD5 mismatch {got} vs {zf['checksum']}")
            z = zipfile.ZipFile(path)
            members = sorted((i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".heic")),
                             key=lambda i: order_key(i.filename))
            for i in members:
                rows.append({**base, "file_key": i.filename, "archive": f"data/raw/zenodo/{rid}/data.zip",
                             "archive_md5_verified": got, "direct_download_url":
                             f"https://zenodo.org/records/{rid}/files/data.zip?download=1", "size": i.file_size,
                             "published_md5": "", "local_path": ""})
        else:
            files = sorted((f for f in rec["files"] if f["key"].lower().endswith(".heic")), key=lambda f: order_key(f["key"]))
            for f in files:
                local = os.path.join(d, "files", f["key"])
                url = f"https://zenodo.org/records/{rid}/files/{f['key']}?download=1"
                if not (os.path.exists(local) and "md5:" + md5(local) == f["checksum"]):
                    r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
                    r.raise_for_status()
                    open(local, "wb").write(r.content)
                if "md5:" + md5(local) != f["checksum"]:
                    raise RuntimeError(f"MD5 mismatch for {f['key']}")
                rows.append({**base, "file_key": f["key"], "archive": "", "archive_md5_verified": "",
                             "direct_download_url": url, "size": f["size"], "published_md5": f["checksum"],
                             "local_path": os.path.relpath(local, os.path.dirname(RAW))})
        print(rid, "candidates:", sum(r["record_id"] == rid for r in rows), flush=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for k, r in enumerate(rows):
        r["candidate_order"] = k
        r["metadata_retrieved_utc"] = now
    with open(os.path.join(MANIFESTS, "candidates_zenodo_heic.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("ZENODO CANDIDATES", len(rows))


if __name__ == "__main__":
    main()
