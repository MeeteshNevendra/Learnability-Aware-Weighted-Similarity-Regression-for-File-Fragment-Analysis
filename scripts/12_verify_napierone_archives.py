"""Compare each downloaded NapierOne archive with the SHA-256 published next to it (<name>_zip_hashes.txt).
Output: data/manifests/napierone_archive_verification.csv"""
import csv
import hashlib
import os
import re

from paths import MANIFESTS, RAW

D = os.path.join(RAW, "napierone", "archives")


def main():
    rows = []
    for fn in sorted(os.listdir(D)):
        if not fn.endswith(".zip"):
            continue
        hfile = os.path.join(D, fn[:-4] + "_zip_hashes.txt")
        pub = None
        if os.path.exists(hfile):
            m = re.search(r"SHA256\s+\d+\s+([0-9a-fA-F]{64})", open(hfile, encoding="latin-1").read())
            pub = m.group(1).lower() if m else None
        h = hashlib.sha256()
        with open(os.path.join(D, fn), "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        rows.append({"archive": fn, "computed_sha256": h.hexdigest(), "published_sha256": pub,
                     "match": pub == h.hexdigest()})
        print(rows[-1], flush=True)
    with open(os.path.join(MANIFESTS, "napierone_archive_verification.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
