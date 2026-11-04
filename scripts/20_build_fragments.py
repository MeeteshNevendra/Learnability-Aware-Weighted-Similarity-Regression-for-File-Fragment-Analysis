"""Fragment extraction for the replacement collection (documented protocol, archived position rule).

Per accepted file (partition from the frozen split manifest):
  * candidate blocks: complete, non-overlapping 512-byte blocks at offsets 0, 512, 1024, ... ; the final
    size % 512 trailing bytes (if any) are NOT used. (Deviation from the archived code, which appended one extra block
    at offset size-512 that overlapped the previous block and was not 512-aligned; see audit.)
  * if a file has more than 80 complete blocks, 80 are retained at positions
    numpy.linspace(0, n_blocks - 1, 80, dtype=int)  (archived rule; first and last complete block always kept);
    otherwise every complete block is retained.
  * byte offsets, block indices and file sizes are stored with every fragment.
Features per fragment (for target components and byte baselines): 256-bin byte counts, Shannon entropy (bits),
hashed bigram counts (archived hash table), zlib level-6 compression ratio (classic zlib 1.3.1).
Outputs (data/fragments/): {split}_frags.npy (uint8 [N,512]), {split}_fragments.csv, {split}_features.npz,
fragment_summary.json
"""
import json
import os

import numpy as np
import pandas as pd

import components as C
from paths import DATA, FRAGMENTS, MANIFESTS

BLOCK, MAX_FRAGS = 512, 80


def retained_starts(size):
    n = size // BLOCK
    if n == 0:
        return np.array([], dtype=np.int64), 0
    idx = np.arange(n) if n <= MAX_FRAGS else np.linspace(0, n - 1, MAX_FRAGS, dtype=int)
    return idx.astype(np.int64) * BLOCK, n


def main():
    man = pd.read_csv(os.path.join(MANIFESTS, "source_manifest.csv"))
    summary = {"block": BLOCK, "max_fragments_per_file": MAX_FRAGS,
               "position_rule": "numpy.linspace(0, n_complete_blocks-1, 80, dtype=int) when n_complete_blocks > 80",
               "trailing_bytes": "discarded (only complete aligned blocks are candidates)",
               "bigram_table_sha256": C.table_sha256(), "zlib": C.zlib_info(), "splits": {}}
    for split in ("train", "val", "test"):
        rows, frags = [], []
        files = man[man.partition == split].sort_values("collection_id")
        for _, r in files.iterrows():
            data = open(os.path.join(DATA, r.stored_path), "rb").read()
            assert len(data) == r.size_bytes
            starts, n_blocks = retained_starts(len(data))
            for k, s in enumerate(starts):
                frags.append(np.frombuffer(data[s:s + BLOCK], dtype=np.uint8))
                rows.append({"collection_id": r.collection_id, "ext": r.recorded_extension,
                             "family_id": r.family_id, "offset": int(s), "block_index": int(s // BLOCK),
                             "retained_rank": k, "file_size": int(len(data)), "n_complete_blocks": int(n_blocks),
                             "trailing_bytes": int(len(data) % BLOCK)})
        X = np.stack(frags)
        meta = pd.DataFrame(rows)
        meta.insert(0, "frag_index", np.arange(len(meta)))
        feat = C.fragment_features(X)
        np.save(os.path.join(FRAGMENTS, f"{split}_frags.npy"), X)
        meta.to_csv(os.path.join(FRAGMENTS, f"{split}_fragments.csv"), index=False)
        np.savez_compressed(os.path.join(FRAGMENTS, f"{split}_features.npz"), **feat)
        # adjacency within the retained sample: same file, offsets differing by exactly 512
        key = set(zip(meta.collection_id, meta.offset))
        has_adj = [((c, o + BLOCK) in key) or ((c, o - BLOCK) in key) for c, o in zip(meta.collection_id, meta.offset)]
        summary["splits"][split] = {
            "files": int(files.shape[0]), "fragments": int(len(meta)),
            "fragments_by_group": meta.ext.value_counts().to_dict(),
            "files_with_more_than_80_blocks": int((files.size_bytes // BLOCK > MAX_FRAGS).sum()),
            "files_with_trailing_bytes": int((files.size_bytes % BLOCK > 0).sum()),
            "fragments_with_an_adjacent_retained_block": int(np.sum(has_adj)),
            "adjacent_retained_block_pairs": int(sum((c, o + BLOCK) in key for c, o in zip(meta.collection_id, meta.offset)))}
        print(split, summary["splits"][split], flush=True)
    json.dump(summary, open(os.path.join(FRAGMENTS, "fragment_summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
