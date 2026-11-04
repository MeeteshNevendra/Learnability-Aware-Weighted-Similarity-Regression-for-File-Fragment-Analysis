"""Rebuild the external Govdocs1 test set (archived revision/external/build_external.py, same logic and seed).

  * archives 000.zip, 001.zip, 002.zip from https://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/
    (fetched by fetch_archives.py) are compared with the copies used in the archived run
  * candidates: members with an extension and >= 32 KiB; extension groups with >= 15 candidates, the 10 most frequent
    (stable sort by count); 15 files per group with random.Random(2026).sample over the sorted candidate list
  * per file (same generator, same call order as the archived script): aligned start (512-aligned), unaligned start
    (not a multiple of 512), 4096-aligned start, AES-256 key (32 bytes) and CTR nonce (16 bytes) from rng.randbytes,
    zlib level-6 compression of the whole file (interpreter zlib), aligned start in the compressed stream if the
    compressed file holds >= 24 blocks
  * conditions: aligned / unaligned / encrypted (AES-256-CTR of the whole file, then the aligned run) / compressed /
    b4096 (six consecutive 4096-byte blocks)
  * keys, nonces, offsets and compressed sizes are persisted; the selection is compared with the archived
    manifest.json; SHA-256 of every selected file is checked against the 500-file replacement collection
Outputs: data/govdocs1/{condition}.npz, {condition}_meta.csv, manifest.json, build_report.json
"""
import hashlib
import json
import os
import random
import zipfile
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from paths import ARCHIVE_ROOT, GOVDOCS, MANIFESTS, RAW

RAWG = os.path.join(RAW, "govdocs1")
THREADS = ["000", "001", "002"]
SEED, RUN, BIG, BIG_RUN, MIN_SIZE, FILES_PER_TYPE, MAX_TYPES = 2026, 24, 4096, 6, 32 * 1024, 15, 10


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def list_candidates():
    cands = defaultdict(list)
    for t in THREADS:
        with zipfile.ZipFile(os.path.join(RAWG, f"{t}.zip")) as z:
            for info in z.infolist():
                name = info.filename
                if info.is_dir() or "." not in os.path.basename(name):
                    continue
                ext = os.path.basename(name).rsplit(".", 1)[1].lower()
                if info.file_size >= MIN_SIZE:
                    cands[ext].append((t, name, info.file_size))
    return cands


def blocks(data, start, size, count):
    return [np.frombuffer(data[start + k * size:start + (k + 1) * size], dtype=np.uint8) for k in range(count)]


def main():
    report = {"archives": {}}
    for t in THREADS:
        mine = sha256_file(os.path.join(RAWG, f"{t}.zip"))
        old_p = os.path.join(ARCHIVE_ROOT, "revision", "external", "raw", f"{t}.zip")
        old = sha256_file(old_p) if os.path.exists(old_p) else None
        report["archives"][t] = {"sha256_downloaded": mine, "sha256_archived_copy": old, "identical": mine == old}
    rng = random.Random(SEED)
    cands = list_candidates()
    types = sorted([e for e, v in cands.items() if len(v) >= FILES_PER_TYPE], key=lambda e: -len(cands[e]))[:MAX_TYPES]
    report["candidate_counts_top_types"] = {e: len(cands[e]) for e in types}
    chosen = []
    for e in sorted(types):
        files = sorted(cands[e])
        chosen += [(e, *f) for f in rng.sample(files, FILES_PER_TYPE)]
    out = {c: ([], []) for c in ("aligned", "unaligned", "encrypted", "compressed", "b4096")}
    manifest = {"corpus": "Govdocs1 (Garfinkel et al. 2009), https://digitalcorpora.org/corpora/file-corpora/files/",
                "archive_urls": [f"https://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/{t}.zip" for t in THREADS],
                "threads": THREADS, "seed": SEED, "run_blocks_512": RUN, "run_blocks_4096": BIG_RUN,
                "min_file_size": MIN_SIZE, "files_per_type": FILES_PER_TYPE, "types": sorted(types),
                "encryption": "AES-256-CTR over the whole file (cryptography library), 32-byte key and 16-byte initial "
                              "counter block from random.Random(2026).randbytes, before fragmentation",
                "compression": f"zlib.compress(file, 6) with {zlib.ZLIB_RUNTIME_VERSION}, before fragmentation",
                "files": []}
    file_sha = []
    for ext, thread, name, size in chosen:
        with zipfile.ZipFile(os.path.join(RAWG, f"{thread}.zip")) as z:
            data = z.read(name)
        fid = os.path.basename(name)
        span = RUN * 512
        a_start = 512 * rng.randrange(0, (len(data) - span) // 512 + 1)
        u_start = rng.randrange(1, len(data) - span)
        while u_start % 512 == 0:
            u_start = rng.randrange(1, len(data) - span)
        b_start = BIG * rng.randrange(0, (len(data) - BIG * BIG_RUN) // BIG + 1)
        key, nonce = rng.randbytes(32), rng.randbytes(16)
        enc = Cipher(algorithms.AES(key), modes.CTR(nonce)).encryptor().update(data)
        comp = zlib.compress(data, 6)
        c_start = 512 * rng.randrange(0, (len(comp) - span) // 512 + 1) if len(comp) >= span else None
        h = hashlib.sha256(data).hexdigest()
        file_sha.append(h)
        rec = {"file": fid, "thread": thread, "member": name, "ext": ext, "size": size, "sha256": h,
               "aligned_start": a_start, "unaligned_start": u_start, "b4096_start": b_start,
               "compressed_size": len(comp), "compressed_start": c_start,
               "aes256_key_hex": key.hex(), "ctr_nonce_hex": nonce.hex()}
        manifest["files"].append(rec)
        for cond, src, start, bsize, n in (("aligned", data, a_start, 512, RUN), ("unaligned", data, u_start, 512, RUN),
                                           ("encrypted", enc, a_start, 512, RUN), ("compressed", comp, c_start, 512, RUN),
                                           ("b4096", data, b_start, BIG, BIG_RUN)):
            if start is None:
                continue
            for k, frag in enumerate(blocks(src, start, bsize, n)):
                out[cond][0].append(frag)
                out[cond][1].append({"file_id": fid, "ext": ext, "offset": start + k * bsize, "block": k})
    for cond, (frags, meta) in out.items():
        np.savez_compressed(os.path.join(GOVDOCS, f"{cond}.npz"), frags=np.stack(frags))
        pd.DataFrame(meta).to_csv(os.path.join(GOVDOCS, f"{cond}_meta.csv"), index=False)
        report.setdefault("conditions", {})[cond] = {"fragments": len(frags), "files": len({m["file_id"] for m in meta})}
    json.dump(manifest, open(os.path.join(GOVDOCS, "manifest.json"), "w"), indent=1)

    # comparison with the archived external set (only where the archived copy is available; reporting only)
    old_manifest = os.path.join(ARCHIVE_ROOT, "revision", "external", "data", "manifest.json")
    if not os.path.exists(old_manifest):
        report["archived_manifest_comparison"] = "archived copy not available on this machine; comparison skipped"
    else:
        old = json.load(open(old_manifest))
        keys = ("file", "thread", "ext", "size", "aligned_start", "unaligned_start", "b4096_start", "compressed_size",
                "compressed_start")
        same = [all(a[k] == b[k] for k in keys) for a, b in zip(manifest["files"], old["files"])]
        report["archived_manifest_comparison"] = {
            "types_equal": manifest["types"] == old["types"], "files_compared": len(same),
            "records_identical_on_" + "_".join(keys): int(sum(same)),
            "first_mismatch": next(({"new": {k: a[k] for k in keys}, "archived": {k: b[k] for k in keys}}
                                    for a, b, s in zip(manifest["files"], old["files"], same) if not s), None)}
        for cond in out:
            a = np.load(os.path.join(GOVDOCS, f"{cond}.npz"))["frags"]
            b = np.load(os.path.join(ARCHIVE_ROOT, "revision", "external", "data", f"{cond}.npz"))["frags"]
            report["archived_manifest_comparison"][f"{cond}_fragments_byte_identical"] = bool(a.shape == b.shape and np.array_equal(a, b))
    # independence from the replacement collection
    coll = pd.read_csv(os.path.join(MANIFESTS, "source_manifest.csv"))
    overlap = sorted(set(file_sha) & set(coll.sha256))
    report["duplicates_with_replacement_collection"] = len(overlap)
    report["duplicate_sha256"] = overlap
    report["n_files"] = len(manifest["files"])
    report["files_long_enough_after_compression"] = sum(f["compressed_start"] is not None for f in manifest["files"])
    json.dump(report, open(os.path.join(GOVDOCS, "build_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
