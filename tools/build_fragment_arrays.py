#!/usr/bin/env python3
"""Build the fragment byte arrays of FileFragment500 from the 500 source files.

Writes data/fragments/{train,val,test}_frags.npy: uint8 arrays with one 512-byte fragment per row, in the order of
data/fragments/{split}_fragments.csv (row i is frag_index i: the 512 bytes at byte offset `offset` of file
`collection_id`). The pair files in data/pairs/ refer to rows of these arrays. Every array is checked against the
SHA-256 value in data/fragments/fragment_arrays.sha256.json, so the arrays are identical to those used in the study.

The arrays are not shipped because they contain bytes of the NapierOne files. Run tools/retrieve_napierone.py first.

usage: python tools/build_fragment_arrays.py
Requires numpy.
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAG_DIR = os.path.join(ROOT, "data", "fragments")
BLOCK = 512


def main():
    with open(os.path.join(ROOT, "data", "manifests", "source_manifest.csv"), encoding="utf-8", newline="") as f:
        manifest = {r["collection_id"]: r for r in csv.DictReader(f)}
    expected = json.load(open(os.path.join(FRAG_DIR, "fragment_arrays.sha256.json"), encoding="utf-8"))
    for split in ("train", "val", "test"):
        with open(os.path.join(FRAG_DIR, f"{split}_fragments.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        X = np.empty((len(rows), BLOCK), dtype=np.uint8)
        current, data = None, b""
        for i, r in enumerate(rows):
            if int(r["frag_index"]) != i:
                sys.exit(f"{split}_fragments.csv: unexpected row order at row {i}")
            cid = r["collection_id"]
            if cid != current:
                m = manifest[cid]
                path = os.path.join(ROOT, "data", *m["stored_path"].replace("\\", "/").split("/"))
                if not os.path.isfile(path):
                    sys.exit(f"{cid} is missing; run  python tools/retrieve_napierone.py  first")
                data = open(path, "rb").read()
                if hashlib.sha256(data).hexdigest() != m["sha256"]:
                    sys.exit(f"{cid} differs from the manifest; run  python tools/verify_collection.py")
                current = cid
            off = int(r["offset"])
            X[i] = np.frombuffer(data[off:off + BLOCK], dtype=np.uint8)
        exp = expected[split]
        digest = hashlib.sha256(X.tobytes()).hexdigest()
        if list(X.shape) != exp["shape"] or digest != exp["sha256_of_array_bytes"]:
            sys.exit(f"{split}: the rebuilt array differs from the array used in the study")
        out = os.path.join(FRAG_DIR, f"{split}_frags.npy")
        np.save(out, X)
        print(f"{split}: {X.shape[0]:,} fragments x {BLOCK} bytes written to {os.path.relpath(out, ROOT)} "
              f"(SHA-256 verified)")
    print("done")


if __name__ == "__main__":
    main()
