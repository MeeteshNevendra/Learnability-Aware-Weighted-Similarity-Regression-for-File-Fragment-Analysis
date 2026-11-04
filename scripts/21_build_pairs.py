"""Pair sampling, replicating the archived generate_pairs() (make_continuous_similarity_dataset.py) exactly.

  * one random.Random(42) generator, used sequentially for train (300,000), val (80,000) and test (80,000) pairs
  * file order inside a split: collection ids sorted, then random.Random(42).shuffle (stands in for the archived
    shuffled file list); fragments keep file order, then position order
  * mode of draw i = ["same_file", "same_type_diff_file", "diff_type", "random"][i % 4]  (equal representation)
      same_file           a file with >= 2 fragments chosen uniformly; random.sample of 2 of its fragments
      same_type_diff_file extension list shuffled; first extension with >= 2 files; random.sample of 2 files;
                          one fragment chosen uniformly from each
      diff_type           random.sample of 2 extensions; one fragment uniformly from each extension's fragments
      random              random.sample of 2 fragments from the whole split
  * random.sample never returns the same fragment twice, so there are no self-pairs; draws are independent, so the
    same (unordered) pair can be drawn more than once (sampling with replacement across draws); pairs keep the
    sampled order (a, b). Extension groups are recorded extensions (JPG and JPEG are separate groups).
  * 'random' draws can be same-file, same-extension or different-extension pairs: both the sampling-mode label and the
    actual provenance relation are stored.
Components are computed from fragment features and provenance (scripts/components.py).
Outputs: data/pairs/{split}_pairs.csv (+ .npz index arrays), pair_summary.json
"""
import json
import os
import random

import numpy as np
import pandas as pd

import components as C
from paths import FRAGMENTS, PAIRS

N_PAIRS = {"train": 300_000, "val": 80_000, "test": 80_000}
MODES = ["same_file", "same_type_diff_file", "diff_type", "random"]


def sample_split(meta, n_pairs, rng):
    files = sorted(meta.collection_id.unique())
    random.Random(42).shuffle(files)
    frag_by_file = {f: meta.index[meta.collection_id == f].tolist() for f in files}
    frags = [i for f in files for i in frag_by_file[f]]
    ext_of_file = dict(zip(meta.collection_id, meta.ext))
    by_file, by_ext, by_ext_file = {}, {}, {}
    for f in files:
        e = ext_of_file[f]
        by_file[f] = frag_by_file[f]
        by_ext.setdefault(e, []).extend(frag_by_file[f])
        by_ext_file.setdefault(e, {})[f] = frag_by_file[f]
    multi = [k for k, v in by_file.items() if len(v) >= 2]
    a_idx, b_idx, modes = [], [], []
    for i in range(n_pairs):
        mode = MODES[i % 4]
        pair = None
        if mode == "same_file":
            if multi:
                k = rng.choice(multi)
                pair = rng.sample(by_file[k], 2)
        elif mode == "same_type_diff_file":
            ext_keys = list(by_ext_file.keys())
            rng.shuffle(ext_keys)
            for e in ext_keys:
                fk = [k for k, v in by_ext_file[e].items() if len(v) >= 1]
                if len(fk) < 2:
                    continue
                f1, f2 = rng.sample(fk, 2)
                pair = [rng.choice(by_ext_file[e][f1]), rng.choice(by_ext_file[e][f2])]
                break
        elif mode == "diff_type":
            ek = [k for k, v in by_ext.items() if len(v) >= 1]
            if len(ek) >= 2:
                e1, e2 = rng.sample(ek, 2)
                pair = [rng.choice(by_ext[e1]), rng.choice(by_ext[e2])]
        else:
            pair = rng.sample(frags, 2)
        if pair is None:
            continue
        a_idx.append(pair[0])
        b_idx.append(pair[1])
        modes.append(mode)
    return np.array(a_idx), np.array(b_idx), modes


def main():
    rng = random.Random(42)
    summary = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(os.path.join(FRAGMENTS, f"{split}_fragments.csv"))
        feat = dict(np.load(os.path.join(FRAGMENTS, f"{split}_features.npz")))
        a, b, modes = sample_split(meta, N_PAIRS[split], rng)
        fid = meta.collection_id.to_numpy()
        ext = meta.ext.to_numpy()
        off = meta.offset.to_numpy().astype(np.float64)
        size = meta.file_size.to_numpy().astype(np.float64)
        comps = C.pair_components(feat, a, b, fid, off, size, ext)
        same_file = fid[a] == fid[b]
        same_ext = ext[a] == ext[b]
        merged = np.where(np.isin(ext, ["jpg", "jpeg"]), "jpg|jpeg", ext)
        df = pd.DataFrame({"pair_id": np.arange(len(a)), "a_frag": a, "b_frag": b, "pair_mode": modes,
                           "a_file": fid[a], "b_file": fid[b], "a_ext": ext[a], "b_ext": ext[b],
                           "a_offset": off[a].astype(np.int64), "b_offset": off[b].astype(np.int64),
                           "same_file": same_file, "same_ext": same_ext,
                           "same_format_jpg_jpeg_merged": merged[a] == merged[b],
                           "adjacent": same_file & (np.abs(off[a] - off[b]) == 512),
                           "relation": np.where(same_file, "same_file", np.where(same_ext, "same_ext_diff_file", "diff_ext")),
                           **{k: v.astype(np.float64) for k, v in comps.items()}})
        df.to_csv(os.path.join(PAIRS, f"{split}_pairs.csv"), index=False)
        np.savez(os.path.join(PAIRS, f"{split}_pairs.npz"), a=a.astype(np.int32), b=b.astype(np.int32),
                 **{k: v.astype(np.float32) for k, v in comps.items()})
        key = np.minimum(a, b).astype(np.int64) * 10_000_000 + np.maximum(a, b)
        used = np.union1d(a, b)
        summary[split] = {"pairs": int(len(df)), "mode_counts": df.pair_mode.value_counts().to_dict(),
                          "relation_by_mode": {m: g.relation.value_counts().to_dict() for m, g in df.groupby("pair_mode")},
                          "self_pairs": int((a == b).sum()), "repeated_unordered_pairs": int(len(key) - len(np.unique(key))),
                          "fragments_in_split": int(len(meta)), "distinct_fragments_in_pairs": int(len(used)),
                          "adjacent_pairs": int(df.adjacent.sum()),
                          "component_means": {k: float(df[k].mean()) for k in C.COMPONENTS}}
        print(split, json.dumps(summary[split]), flush=True)
    json.dump(summary, open(os.path.join(PAIRS, "pair_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
