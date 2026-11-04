#!/usr/bin/env python3
"""Check the 500 source files of FileFragment500 against the frozen records.

  * every file listed in data/manifests/source_manifest.csv: present at its stored path, correct size and SHA-256;
  * when all 500 files are present: the collection digest recorded in configs/collection_freeze.json (SHA-256 of the
    500 file SHA-256 values, sorted and concatenated);
  * the frozen manifests listed in configs/collection_freeze.json (SHA-256 of each file);
  * with --package: every file listed in SHA256SUMS.txt (the files shipped in this package).

usage: python tools/verify_collection.py [--package]
Exit status 0 only if every check passes. Python 3.8 or later; standard library only.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Verify the FileFragment500 collection.")
    ap.add_argument("--package", action="store_true", help="also check every file listed in SHA256SUMS.txt")
    args = ap.parse_args()
    failures = 0

    with open(os.path.join(ROOT, "data", "manifests", "source_manifest.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    status = {}
    for r in rows:
        path = os.path.join(ROOT, "data", *r["stored_path"].replace("\\", "/").split("/"))
        if not os.path.isfile(path):
            s = "missing"
        elif os.path.getsize(path) != int(r["size_bytes"]) or sha256_file(path) != r["sha256"]:
            s = "different"
        else:
            s = "ok"
        status[r["collection_id"]] = (r["source_name"], r["extension_group"], s)

    print(f"{'source':<20}{'groups':<28}{'files':>6}{'ok':>6}{'missing':>9}{'different':>11}")
    for src in sorted({v[0] for v in status.values()}):
        v = [x for x in status.values() if x[0] == src]
        groups = ", ".join(sorted({x[1] for x in v}))
        print(f"{src:<20}{groups:<28}{len(v):>6}{sum(x[2] == 'ok' for x in v):>6}"
              f"{sum(x[2] == 'missing' for x in v):>9}{sum(x[2] == 'different' for x in v):>11}")
    n_ok = sum(x[2] == "ok" for x in status.values())
    different = [k for k, x in status.items() if x[2] == "different"]
    missing = [k for k, x in status.items() if x[2] == "missing"]
    if different:
        failures += 1
        print("files that differ from the manifest:", ", ".join(different[:20]))
    if missing:
        failures += 1
        if all(status[k][0] == "NapierOne" for k in missing):
            print(f"{len(missing)} NapierOne files are not present yet: run  python tools/retrieve_napierone.py")
        else:
            print("missing files:", ", ".join(missing[:20]))

    freeze = json.load(open(os.path.join(ROOT, "configs", "collection_freeze.json"), encoding="utf-8"))
    if n_ok == len(rows):
        digest = hashlib.sha256("".join(sorted(r["sha256"] for r in rows)).encode()).hexdigest()
        same = digest == freeze["accepted_files_sha256_digest"]
        failures += not same
        print(f"collection digest {'matches' if same else 'DIFFERS FROM'} the frozen record "
              f"({freeze['frozen_utc']}): {digest}")
    for name, expected in freeze["manifests"].items():
        p = os.path.join(ROOT, "data", "manifests", name)
        same = os.path.isfile(p) and sha256_file(p) == expected
        failures += not same
        if not same:
            print(f"frozen manifest {name} is missing or changed")
    print(f"frozen manifests checked: {len(freeze['manifests'])}")

    if args.package and not os.path.isfile(os.path.join(ROOT, "SHA256SUMS.txt")):
        print("--package: SHA256SUMS.txt is not here (it belongs to the unpacked data set package); skipped")
    elif args.package:
        bad, n = [], 0
        with open(os.path.join(ROOT, "SHA256SUMS.txt"), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                digest, rel = line.rstrip("\n").split("  ", 1)
                n += 1
                p = os.path.join(ROOT, *rel.split("/"))
                if not os.path.isfile(p) or sha256_file(p) != digest:
                    bad.append(rel)
        failures += bool(bad)
        print(f"package files checked against SHA256SUMS.txt: {n}, problems: {len(bad)}")
        for rel in bad[:20]:
            print("   ", rel)

    print(f"{n_ok} of {len(rows)} source files present and verified")
    if failures:
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
