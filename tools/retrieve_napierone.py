#!/usr/bin/env python3
"""Download the 250 NapierOne files of FileFragment500 and check every one of them.

The PDF, DOCX, PPTX, CSV and MP4 groups were taken from the NapierOne data set (Davies, Macfarlane and Buchanan, 2022),
which Edinburgh Napier University publishes through the Registry of Open Data on AWS. They are not included in this
package. This script

  1. downloads the six NapierOne archives that hold them (about 650 MB in total; no account is needed),
  2. compares each archive with the SHA-256 value that NapierOne publishes next to it,
  3. extracts the 250 files under the names used in this package, data/accepted_files/<group>/<ID>.<ext>,
  4. checks the size and SHA-256 value of every extracted file against data/manifests/source_manifest.csv.

A file is written only if its SHA-256 value matches the manifest, so the result is identical to the collection used in
the study. Files that are already present and correct are skipped, so the script can simply be run again after an
interruption.

usage:
  python tools/retrieve_napierone.py                    download the archives to data/raw/napierone/archives/
  python tools/retrieve_napierone.py --archives DIR     use archives already downloaded to DIR (original file names)
  python tools/retrieve_napierone.py --delete-archives  delete the archives in data/raw/napierone/archives/ afterwards

The NapierOne licence terms apply to the retrieved files (LICENSES/NapierOne_terms.txt).
Python 3.8 or later; standard library only.
"""
import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "manifests", "source_manifest.csv")
ARCHIVE_HASHES = os.path.join(ROOT, "data", "manifests", "napierone_archive_verification.csv")
DEFAULT_ARCHIVES = os.path.join(ROOT, "data", "raw", "napierone", "archives")
USER_AGENT = "FileFragment500-retrieval/1.0 (reconstruction of a research data set)"
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def target_path(row):
    """stored_path is recorded with Windows separators (accepted_files\\pdf\\PDF-001.pdf)."""
    return os.path.join(ROOT, "data", *row["stored_path"].replace("\\", "/").split("/"))


def is_correct(path, row):
    return (os.path.isfile(path) and os.path.getsize(path) == int(row["size_bytes"])
            and sha256_file(path) == row["sha256"])


def download(url, dest, tries=3):
    tmp = dest + ".part"
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, shown = 0, -1
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total and done * 10 // total != shown:
                        shown = done * 10 // total
                        print(f"    {done * 100 // total:3d}% of {total / 1e6:.0f} MB", flush=True)
            if total and os.path.getsize(tmp) != total:
                raise IOError(f"incomplete download ({os.path.getsize(tmp)} of {total} bytes)")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001  network errors are retried
            print(f"    attempt {attempt} failed: {e}", flush=True)
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


def main():
    ap = argparse.ArgumentParser(description="Download and verify the 250 NapierOne files of FileFragment500.")
    ap.add_argument("--archives", help="folder that already holds the NapierOne archives (original file names)")
    ap.add_argument("--delete-archives", action="store_true",
                    help="delete the archives in data/raw/napierone/archives/ after extraction")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source_name"] == "NapierOne"]
    with open(ARCHIVE_HASHES, encoding="utf-8", newline="") as f:
        published = {r["archive"]: r["published_sha256"] for r in csv.DictReader(f)}
    adir = os.path.abspath(args.archives) if args.archives else DEFAULT_ARCHIVES
    os.makedirs(adir, exist_ok=True)

    todo = [r for r in rows if not is_correct(target_path(r), r)]
    print(f"{len(rows)} NapierOne files in the manifest; {len(rows) - len(todo)} already present and correct")
    by_archive = {}
    for r in todo:
        by_archive.setdefault(r["direct_download_url"], []).append(r)

    written, used = 0, []
    for url, members in sorted(by_archive.items()):
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(adir, name)
        if not os.path.isfile(path):
            if args.archives:
                sys.exit(f"{name} is not in {adir}; download it from {url}")
            print(f"downloading {name} from {url}", flush=True)
            download(url, path)
        used.append(path)
        digest = sha256_file(path)
        if digest == published.get(name):
            print(f"  {name}: SHA-256 matches the value published by NapierOne")
        else:
            print(f"  WARNING {name}: SHA-256 {digest} differs from the published value recorded on 27 October 2026 "
                  f"({published.get(name)}). The archive may have been updated; every extracted file is still checked.")
        with zipfile.ZipFile(path) as z:
            for r in members:
                data = z.read(r["archive_member"])
                if len(data) != int(r["size_bytes"]) or hashlib.sha256(data).hexdigest() != r["sha256"]:
                    sys.exit(f"{r['collection_id']}: member {r['archive_member']} of {name} does not match the manifest")
                out = target_path(r)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out + ".part", "wb") as f:
                    f.write(data)
                os.replace(out + ".part", out)
                written += 1
        print(f"  {len(members)} files extracted and verified", flush=True)

    if args.delete_archives and not args.archives:
        for p in used:
            os.remove(p)
        print("downloaded archives deleted")
    bad = [r["collection_id"] for r in rows if not is_correct(target_path(r), r)]
    if bad:
        sys.exit(f"{len(bad)} NapierOne files are missing or differ from the manifest: {', '.join(bad[:10])}")
    print(f"done: all {len(rows)} NapierOne files are present and verified ({written} written in this run)")


if __name__ == "__main__":
    main()
