#!/usr/bin/env python3
"""Linux and macOS only: let the scripts open the source files by their recorded stored paths.

data/manifests/source_manifest.csv records each file's stored path with Windows separators, for example
accepted_files\\png\\PNG-001.png. Scripts 14, 15 and 20 and the reconstruction step of 41_evaluate_local.py open
os.path.join("data", stored_path), which on Windows is the file data/accepted_files/png/PNG-001.png. On Linux and macOS
a backslash is an ordinary character in a file name, so this tool creates, in data/, one symbolic link per source file
whose name is the stored path and which points to the file. The scripts then run unchanged.

usage: python tools/posix_stored_paths.py            create the links (after all 500 files are in place)
       python tools/posix_stored_paths.py --remove   remove the links again
Not needed on Windows. Python 3.8 or later; standard library only.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser(description="Create (or remove) links named like the recorded stored paths.")
    ap.add_argument("--remove", action="store_true", help="remove the links")
    args = ap.parse_args()
    if os.name == "nt":
        sys.exit("Not needed on Windows: the scripts open the recorded stored paths directly.")
    with open(os.path.join(DATA, "manifests", "source_manifest.csv"), encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if "\\" in r["stored_path"]]
    done = 0
    for r in rows:
        link = os.path.join(DATA, r["stored_path"])        # one file name that contains backslashes
        target = r["stored_path"].replace("\\", "/")        # relative to data/
        if args.remove:
            if os.path.islink(link):
                os.remove(link)
                done += 1
            continue
        if not os.path.isfile(os.path.join(DATA, target)):
            sys.exit(f"data/{target} is missing: install the data set and run tools/retrieve_napierone.py first")
        if not os.path.lexists(link):
            os.symlink(target, link)
            done += 1
    print(f"{done} links {'removed' if args.remove else 'created'} in data/ ({len(rows)} source files)")


if __name__ == "__main__":
    main()
