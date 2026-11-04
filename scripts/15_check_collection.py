"""Consistency check of the accepted collection: every manifest row has its file with the recorded SHA-256 and the
accepted_files directory holds no other files. Output: data/manifests/collection_check.json"""
import hashlib
import json
import os

import pandas as pd

from paths import ACCEPTED, DATA, MANIFESTS


def main():
    m = pd.read_csv(os.path.join(MANIFESTS, "source_manifest_unsplit.csv"))
    listed = {p.replace("\\", "/") for p in m.stored_path}
    on_disk = {f"accepted_files/{g}/{f}" for g in os.listdir(ACCEPTED) for f in os.listdir(os.path.join(ACCEPTED, g))}
    bad = [r.stored_path for r in m.itertuples()
           if hashlib.sha256(open(os.path.join(DATA, r.stored_path), "rb").read()).hexdigest() != r.sha256]
    out = {"manifest_rows": int(len(m)), "files_on_disk": len(on_disk), "stale_files": sorted(on_disk - listed),
           "missing_files": sorted(listed - on_disk), "sha256_mismatches": bad,
           "unique_sha256": int(m.sha256.nunique()),
           "by_group_and_source": {f"{g}|{s}": int(n) for (g, s), n in m.groupby(["extension_group", "source_name"]).size().items()},
           "size_by_group": {g: {"min": int(x.min()), "median": float(x.median()), "max": int(x.max())}
                             for g, x in m.groupby("extension_group").size_bytes}}
    ex = pd.read_csv(os.path.join(MANIFESTS, "exclusion_log.csv"))
    out["decisions"] = {f"{g}|{d}": int(n) for (g, d), n in ex.groupby(["group", "decision"]).size().items()}
    out["licences"] = m.license.fillna("NONE").value_counts().to_dict()
    json.dump(out, open(os.path.join(MANIFESTS, "collection_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
