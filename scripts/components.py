"""Similarity components, vectorised but numerically matching the archived implementation
(Regression_new/new/make_continuous_similarity_dataset.py and revision/retrieval_eval.py).

Archived n-gram hashing: bin = hash((b0, b1)) % 512, where (b0, b1) is a tuple of Python ints. Python hashes small
ints to themselves and combines tuple items with a fixed xxHash-style routine (CPython >= 3.8), so this value does
NOT depend on PYTHONHASHSEED (only str/bytes hashing is randomised). To remove any dependence on the interpreter we
materialise the 65,536-entry lookup table once, save it (configs/bigram_hash_table.npy) and always use the table.
scripts/00_audit_archive.py checks the table against component values stored in the archived pair metadata.
"""
import hashlib
import os
import zlib

import numpy as np

from paths import CONFIGS

BLOCK = 512
NGRAM_HASH_DIM = 512
COMPONENTS = ["s_hist", "s_ent", "s_ngram", "s_comp", "s_pos", "s_type"]
BYTE_COMPONENTS = ["s_hist", "s_ent", "s_ngram", "s_comp"]
TABLE_PATH = os.path.join(CONFIGS, "bigram_hash_table.npy")


def bigram_table():
    if os.path.exists(TABLE_PATH):
        return np.load(TABLE_PATH)
    t = np.array([hash((a, b)) % NGRAM_HASH_DIM for a in range(256) for b in range(256)], dtype=np.int16)
    np.save(TABLE_PATH, t)
    return t


def table_sha256():
    return hashlib.sha256(bigram_table().tobytes()).hexdigest()


def byte_hist(frags):
    """frags uint8 [N, L] -> counts float64 [N, 256]."""
    n = len(frags)
    idx = frags.astype(np.int64) + 256 * np.arange(n)[:, None]
    return np.bincount(idx.ravel(), minlength=256 * n).reshape(n, 256).astype(np.float64)


def entropy_bits(hist):
    p = hist / np.maximum(hist.sum(1, keepdims=True), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0).sum(1)


def bigram_counts(frags, table=None):
    """Counts of hashed byte bigrams (archived hashing), float64 [N, 512]."""
    table = bigram_table() if table is None else table
    f = frags.astype(np.int64)
    bins = table[f[:, :-1] * 256 + f[:, 1:]].astype(np.int64)
    n = len(frags)
    idx = bins + NGRAM_HASH_DIM * np.arange(n)[:, None]
    return np.bincount(idx.ravel(), minlength=NGRAM_HASH_DIM * n).reshape(n, NGRAM_HASH_DIM).astype(np.float64)


CLASSIC_ZLIB_PYTHON = r"C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"   # CPython 3.12.10, zlib 1.3.1


def comp_ratio(frags, level=6):
    """Compressed length / fragment length with CLASSIC zlib (matches the archived s_comp values exactly).
    The running interpreter's zlib is used only if it is classic zlib; otherwise the helper interpreter is called."""
    frags = np.ascontiguousarray(frags, dtype=np.uint8)
    n, L = frags.shape
    if "ng" not in zlib.ZLIB_RUNTIME_VERSION:
        return np.array([len(zlib.compress(f.tobytes(), level)) / L for f in frags], dtype=np.float64)
    import subprocess
    import tempfile
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zlib_classic_lengths.py")
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        frags.tofile(src)
        subprocess.run([CLASSIC_ZLIB_PYTHON, "-I", helper, src, str(L), str(level), dst], check=True,
                       capture_output=True)
        lens = np.fromfile(dst, dtype="<u4")
    if len(lens) != n:
        raise RuntimeError("classic zlib helper returned wrong number of lengths")
    return lens.astype(np.float64) / L


def zlib_info():
    import subprocess
    out = subprocess.run([CLASSIC_ZLIB_PYTHON, "-I", "-c", "import zlib,sys;print(sys.version.split()[0], zlib.ZLIB_RUNTIME_VERSION)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return {"component_zlib_interpreter": CLASSIC_ZLIB_PYTHON, "component_zlib": out,
            "pipeline_interpreter_zlib": zlib.ZLIB_RUNTIME_VERSION}


def fragment_features(frags):
    h = byte_hist(frags)
    return {"hist": h, "ent": entropy_bits(h), "ngram": bigram_counts(frags), "comp": comp_ratio(frags)}


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def pair_components(feat, i, j, file_id, offset, file_size, ext, chunk=50_000):
    """Six components for index arrays i, j into a fragment table (provenance arrays aligned with feat).
    s_pos = clip(1 - |o_i/F - o_j/F|, 0, 1) for blocks of the same file (archived form; equals 1 - |o_i - o_j|/F),
    otherwise 0. s_type = 1 when recorded extensions are equal."""
    i, j = np.asarray(i), np.asarray(j)
    hu, gu = _unit(feat["hist"]), _unit(feat["ngram"])
    s_hist = np.concatenate([(hu[i[k:k + chunk]] * hu[j[k:k + chunk]]).sum(1) for k in range(0, len(i), chunk)]
                            or [np.zeros(0)])
    s_ngram = np.concatenate([(gu[i[k:k + chunk]] * gu[j[k:k + chunk]]).sum(1) for k in range(0, len(i), chunk)]
                             or [np.zeros(0)])
    same = file_id[i] == file_id[j]
    pos = np.where(same, np.clip(1.0 - np.abs(offset[i] / file_size[i] - offset[j] / file_size[j]), 0.0, 1.0), 0.0)
    return {
        "s_hist": s_hist,
        "s_ent": np.clip(1.0 - np.abs(feat["ent"][i] - feat["ent"][j]) / 8.0, 0.0, 1.0),
        "s_ngram": s_ngram,
        "s_comp": np.clip(1.0 - np.abs(feat["comp"][i] - feat["comp"][j]), 0.0, 1.0),
        "s_pos": pos,
        "s_type": (ext[i] == ext[j]).astype(np.float64),
    }


def composite(comps, weights):
    """Clipped weighted sum. `weights` must already be normalised (non-negative, summing to one)."""
    y = np.zeros(len(next(iter(comps.values()))), dtype=np.float64)
    for k, w in weights.items():
        y += w * np.asarray(comps[k], dtype=np.float64)
    return np.clip(y, 0.0, 1.0)


# ---------------- similarity matrices for provenance evaluation (direct byte-statistic baselines)
def cosine_matrix(V):
    U = _unit(V.astype(np.float64))
    return U @ U.T


def baseline_matrices(feat):
    s = {"hist_cosine": cosine_matrix(feat["hist"]),
         "entropy_sim": np.clip(1.0 - np.abs(feat["ent"][:, None] - feat["ent"][None, :]) / 8.0, 0, 1),
         "ngram_cosine": cosine_matrix(feat["ngram"]),
         "compress_sim": np.clip(1.0 - np.abs(feat["comp"][:, None] - feat["comp"][None, :]), 0, 1)}
    s["uniform_fusion_4"] = (s["hist_cosine"] + s["entropy_sim"] + s["ngram_cosine"] + s["compress_sim"]) / 4.0
    return s
