#!/usr/bin/env python3
"""Copy the data/ folder of the FileFragment500 data set ZIP into this repository.

usage: python tools/install_dataset.py path/to/FileFragment500_v1.0.zip

Every file is checked against the SHA256SUMS.txt inside the ZIP before it is written to data/. Files that are already
present and correct are skipped, so the command can be repeated. The configuration files in the ZIP (configs/) are
compared with this repository's copies. Afterwards, run tools/retrieve_napierone.py to add the 250 NapierOne files.
Python 3.8 or later; standard library only.
"""
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with zipfile.ZipFile(sys.argv[1]) as z:
        names = set(z.namelist())
        sums_name = next((n for n in names if n.count("/") == 1 and n.endswith("/SHA256SUMS.txt")), None)
        if sums_name is None:
            sys.exit("SHA256SUMS.txt not found: is this the FileFragment500 data set ZIP?")
        top = sums_name.split("/")[0] + "/"
        sums = {}
        for line in z.read(sums_name).decode("utf-8").splitlines():
            if line.strip():
                digest, rel = line.split("  ", 1)
                sums[rel] = digest
        data_files = sorted(rel for rel in sums if rel.startswith("data/"))
        copied = 0
        for rel in data_files:
            parts = rel.split("/")
            if any(p in ("", ".", "..") for p in parts) or (top + rel) not in names:
                sys.exit(f"unexpected entry in the ZIP: {rel}")
            dest = os.path.join(ROOT, *parts)
            if os.path.isfile(dest) and sha256_file(dest) == sums[rel]:
                continue
            data = z.read(top + rel)
            if hashlib.sha256(data).hexdigest() != sums[rel]:
                sys.exit(f"{rel}: checksum does not match SHA256SUMS.txt; the ZIP may be damaged")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            copied += 1
        for rel in sorted(r for r in sums if r.startswith("configs/")):
            mine = os.path.join(ROOT, *rel.split("/"))
            if os.path.isfile(mine) and sha256_file(mine) != sums[rel]:
                print(f"note: {rel} differs between the data set and this repository")
    print(f"{len(data_files)} data files checked, {copied} copied into data/")
    print("next: python tools/retrieve_napierone.py")


if __name__ == "__main__":
    main()
