"""Related-version families, the fixed file-level split, the file-property summary and the collection freeze.

Family rules (union-find over accepted files; a family is kept in one partition):
  F1 Wikimedia Commons: identical normalised filename stem (lower case, extension removed, separators unified,
     trailing copy/size markers removed: "(n)", thumb/small/medium/large, crop(ped), edit(ed), NNNpx, WxH)
  F2 decodable images (PNG/JPG/JPEG/HEIC, across groups): 64-bit dHash Hamming distance <= 4
  F3 documents (PDF/DOCX/PPTX/CSV, across groups): Jaccard similarity of 5-word shingles >= 0.5
     (only documents with >= 25 shingles)
  F4 video files (MP4/WEBM): decoded frames at 25/50/75 % of the duration with mean dHash Hamming distance <= 6
     (replaces an identical-duration rule; see data/manifests/collection_protocol_corrections.log)
  F5 Zenodo 17640511 (multi-device photos): scene correspondence is not encoded in the file names, so photos
     from different devices with dHash Hamming distance <= 10 are grouped (possible same-scene captures)
Split (seed string "split-v1"): per-group counts train/val/test = 35/7/8 (PNG, JPG, HEIC, PDF, MP4) and 35/8/7
(DOCX, PPTX, JPEG, WEBM, CSV). Families with more than 3 members go to training. Remaining families are visited in
order of SHA-256("split-v1|" + family id) and placed in test, else validation, else training, whichever still has
room for every member of the family. Exact quotas are asserted.
Outputs: data/manifests/families.csv, source_manifest.csv, split_manifest.csv, results/file_properties_*.csv,
configs/collection_freeze.json
"""
import datetime as dt
import hashlib
import itertools
import json
import os
import re

import numpy as np
import pandas as pd

from paths import ACCEPTED, CONFIGS, DATA, MANIFESTS, RESULTS

QUOTA = {"png": (35, 7, 8), "jpg": (35, 7, 8), "heic": (35, 7, 8), "pdf": (35, 7, 8), "mp4": (35, 7, 8),
         "docx": (35, 8, 7), "pptx": (35, 8, 7), "jpeg": (35, 8, 7), "webm": (35, 8, 7), "csv": (35, 8, 7)}
PARTS = ("train", "val", "test")


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}
        self.why = {}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b, rule):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
        self.why.setdefault(tuple(sorted((a, b))), []).append(rule)


def stem(name):
    s = name.rsplit(".", 1)[0].lower()
    s = re.sub(r"[_\s\-]+", " ", s).strip()
    for pat in (r"\s*\(\d+\)$", r"\s*(thumb|thumbnail|small|medium|large|crop|cropped|edit|edited)$",
                r"\s*\d{2,5}px$", r"\s*\d{3,5}x\d{3,5}$"):
        s = re.sub(pat, "", s).strip()
    return s


def scene_key(name):
    """Scene number for multi-device photo names such as '<device>/<...>_07.HEIC' (None if no trailing number)."""
    m = re.search(r"(\d+)\D*$", os.path.basename(name).rsplit(".", 1)[0])
    return int(m.group(1)) if m else None


def main():
    man = pd.read_csv(os.path.join(MANIFESTS, "source_manifest_unsplit.csv"), dtype={"archive_member": str})
    feats = json.load(open(os.path.join(MANIFESTS, "family_features.json")))
    ids = man.collection_id.tolist()
    uf = UF(ids)
    g = dict(zip(man.collection_id, man.recorded_extension))

    # F1
    com = man[man.source_name == "Wikimedia Commons"]
    for _, grp in com.groupby(com.original_filename.map(stem)):
        for a, b in itertools.combinations(grp.collection_id, 2):
            uf.union(a, b, "F1 filename stem")
    # F2
    imgs = [(c, feats[c].get("dhash")) for c in ids if feats.get(c, {}).get("dhash") is not None]
    for (a, ha), (b, hb) in itertools.combinations(imgs, 2):
        if bin(ha ^ hb).count("1") <= 4:
            uf.union(a, b, "F2 dHash<=4")
    # F3
    docs = [(c, set(feats[c]["shingles"])) for c in ids if len(feats.get(c, {}).get("shingles", [])) >= 25]
    for (a, sa), (b, sb) in itertools.combinations(docs, 2):
        if len(sa & sb) / len(sa | sb) >= 0.5:
            uf.union(a, b, "F3 shingle Jaccard>=0.5")
    # F4 (content): video files (MP4 and WEBM) whose frames at 25/50/75 % of the duration have a mean dHash Hamming
    # distance <= 6. (The first design, identical container duration, was replaced before splitting: 34 of 50 MP4
    # files share the 10.024 s length of a standard 10-second clip, which is not evidence of related content.)
    import formats as FM
    vids = []
    for _, r in man[man.recorded_extension.isin(["mp4", "webm"])].iterrows():
        h = FM.video_frame_hashes(open(os.path.join(DATA, r.stored_path), "rb").read())
        if h is not None:
            vids.append((r.collection_id, h))
    for (a, ha), (b, hb) in itertools.combinations(vids, 2):
        if np.mean([bin(x ^ y).count("1") for x, y in zip(ha, hb)]) <= 6:
            uf.union(a, b, "F4 video frames mean dHash<=6")
    json.dump({c: h for c, h in vids}, open(os.path.join(MANIFESTS, "video_frame_hashes.json"), "w"))
    n_video_hashed = len(vids)
    # F5: the multi-device record names files by per-camera counters (IMG_5413..., IMG_8595...), so scene
    # correspondence is not recorded; photos from different devices with dHash distance <= 10 are grouped instead.
    sci = man[man.source_dataset_release.str.contains("17640511", regex=False)]
    note = "F5 not applicable"
    if len(sci):
        dev = dict(zip(sci.collection_id, sci.archive_member.map(lambda m: os.path.dirname(str(m)))))
        hs = [(c, feats[c].get("dhash")) for c in sci.collection_id if feats[c].get("dhash") is not None]
        n5 = 0
        dists = []
        for (a, ha), (b, hb) in itertools.combinations(hs, 2):
            if dev[a] != dev[b]:
                d = bin(ha ^ hb).count("1")
                dists.append(d)
                if d <= 10:
                    uf.union(a, b, "F5 cross-device dHash<=10 (same scene, record 17640511)")
                    n5 += 1
        note = (f"record 17640511 device folders {sorted(set(dev.values()))}; scene correspondence not encoded in "
                f"file names; {n5} cross-device pairs with dHash<=10 grouped; cross-device dHash distance "
                f"min={min(dists) if dists else None} median={float(np.median(dists)) if dists else None}")

    fam = {c: uf.find(c) for c in ids}
    man["family_id"] = man.collection_id.map(lambda c: "FAM-" + fam[c])
    fam_rows = []
    for f, grp in man.groupby("family_id"):
        members = grp.collection_id.tolist()
        rules = sorted({r for (a, b), rs in uf.why.items() if a in members and b in members for r in rs})
        fam_rows.append({"family_id": f, "size": len(members), "members": " ".join(members),
                         "groups": " ".join(sorted(set(grp.recorded_extension))), "rules": "; ".join(rules)})
    fams = pd.DataFrame(fam_rows)
    fams.to_csv(os.path.join(MANIFESTS, "families.csv"), index=False)

    # split
    remaining = {gname: dict(zip(PARTS, q)) for gname, q in QUOTA.items()}
    part = {}
    order = sorted(fams.itertuples(), key=lambda r: hashlib.sha256(f"split-v1|{r.family_id}".encode()).hexdigest())
    big = [r for r in order if r.size > 3]
    small = [r for r in order if r.size <= 3]
    for r in big + small:
        members = r.members.split()
        need = pd.Series([g[m] for m in members]).value_counts().to_dict()
        prefs = ("train",) if r.size > 3 else ("test", "val", "train")
        for p in prefs:
            if all(remaining[gg][p] >= k for gg, k in need.items()):
                for gg, k in need.items():
                    remaining[gg][p] -= k
                for m in members:
                    part[m] = p
                break
        else:
            raise RuntimeError(f"family {r.family_id} cannot be placed: {need} remaining {remaining}")
    left = {gg: v for gg, v in remaining.items() if any(v.values())}
    if left:
        raise RuntimeError(f"quotas not met exactly: {left}")
    man["partition"] = man.collection_id.map(part)
    man.to_csv(os.path.join(MANIFESTS, "source_manifest.csv"), index=False)
    man[["collection_id", "extension_group", "recorded_extension", "family_id", "partition", "sha256",
         "size_bytes"]].to_csv(os.path.join(MANIFESTS, "split_manifest.csv"), index=False)

    # file properties
    man["complete_512_blocks"] = man.size_bytes // 512
    man["trailing_bytes"] = man.size_bytes % 512
    props = man.groupby(["extension_group", "partition"]).agg(
        files=("collection_id", "count"), size_min=("size_bytes", "min"), size_median=("size_bytes", "median"),
        size_mean=("size_bytes", "mean"), size_max=("size_bytes", "max"),
        blocks_median=("complete_512_blocks", "median"), files_with_ge_80_blocks=("complete_512_blocks", lambda s: int((s >= 80).sum())))
    props.to_csv(os.path.join(RESULTS, "file_properties_by_group_partition.csv"))
    overall = man.groupby("extension_group").agg(
        files=("collection_id", "count"), sources=("source_name", lambda s: "; ".join(sorted(set(s)))),
        verified_formats=("verified_format", lambda s: "; ".join(f"{k} ({v})" for k, v in s.value_counts().items())),
        licences=("license", lambda s: "; ".join(f"{k} ({v})" for k, v in s.fillna("none").value_counts().items())),
        size_min=("size_bytes", "min"), size_median=("size_bytes", "median"), size_max=("size_bytes", "max"),
        total_bytes=("size_bytes", "sum"), families=("family_id", "nunique"))
    overall.to_csv(os.path.join(RESULTS, "file_properties_by_group.csv"))

    # freeze
    def fsha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    files_digest = hashlib.sha256("".join(sorted(man.sha256)).encode()).hexdigest()
    freeze = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "statement": "Collection and partition frozen before any model training or test evaluation.",
              "n_files": int(len(man)), "partition_counts": man.partition.value_counts().to_dict(),
              "families": int(man.family_id.nunique()), "multi_member_families": int((fams["size"] > 1).sum()),
              "family_rules_note": note, "video_files_with_frame_hashes": n_video_hashed,
              "accepted_files_sha256_digest": files_digest,
              "manifests": {n: fsha(os.path.join(MANIFESTS, n)) for n in
                            ("source_manifest.csv", "split_manifest.csv", "exclusion_log.csv", "families.csv",
                             "candidates_commons.csv", "candidates_zenodo_heic.csv", "archive_retrieval_log.csv")}}
    json.dump(freeze, open(os.path.join(CONFIGS, "collection_freeze.json"), "w"), indent=1)
    print(json.dumps(freeze, indent=1))
    print(fams[fams["size"] > 1].to_string())


if __name__ == "__main__":
    main()
