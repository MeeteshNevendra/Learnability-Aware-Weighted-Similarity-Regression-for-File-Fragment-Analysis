"""Checks of the implementation against the worked numbers stated in the manuscript.

Run with `python -I tests/run_tests.py` (or with pytest).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from lawsr import config  # noqa: E402
from lawsr.reference import positional_cue, provenance_reference, type_cue  # noqa: E402
from lawsr.retrieval import random_order_reference, successor_ranks, top_k_same_source  # noqa: E402
from lawsr.similarity import FragmentStore, pair_similarity, uniform_fusion, weighted_fusion  # noqa: E402
from lawsr.weighting import learnability_index, renormalized, softmax_weights  # noqa: E402


def test_eq23_worked_examples():
    # Section 3.4: positions 0.10 and 0.20 -> 0.95; 0.10 and 0.90 -> 0.60
    assert np.isclose(provenance_reference(1, 0.10, 0.20), 0.95)
    assert np.isclose(provenance_reference(1, 0.10, 0.90), 0.60)
    # Figure 2: 65,536-byte file, L = 512: neighbours 512 bytes away -> 0.9961; 32,768 bytes away -> 0.7500
    S = 65536
    assert round(float(provenance_reference(1, 32768 / S, (32768 + 512) / S)), 4) == 0.9961
    assert round(float(provenance_reference(1, 32768 / S, (32768 - 512) / S)), 4) == 0.9961
    assert round(float(provenance_reference(1, 0 / S, 32768 / S)), 4) == 0.7500
    # separation of one fifth of the source -> 0.90; different files -> 0
    assert np.isclose(provenance_reference(1, 0.3, 0.5), 0.90)
    assert provenance_reference(0, 0.1, 0.1) == 0.0
    # y = 0.5 z + 0.5 s_pos, and symmetry in the pair order
    assert np.isclose(provenance_reference(1, 0.2, 0.7), 0.5 + 0.5 * positional_cue(1, 0.2, 0.7))
    assert np.isclose(provenance_reference(1, 0.2, 0.7), provenance_reference(1, 0.7, 0.2))
    assert type_cue('png', 'png') == 1.0 and type_cue('png', 'jpg') == 0.0


def test_eq17_eq18_reproduce_table5():
    r2 = [0.069, 0.010, 0.214, 0.001]
    rho = [0.233, 0.099, 0.383, 0.020]
    lk = learnability_index(r2, rho)
    assert np.allclose(lk, [0.1510, 0.0545, 0.2985, 0.0105], atol=1e-9)
    w = softmax_weights(lk, 0.20)
    assert np.allclose(w, [0.2379, 0.1468, 0.4974, 0.1178], atol=6e-5)
    assert np.isclose(w.sum(), 1.0)
    # smaller temperatures concentrate weight on the largest index
    assert softmax_weights(lk, 0.05)[2] > softmax_weights(lk, 1.0)[2]
    # negative parts are clipped at zero
    assert learnability_index(-0.5, 0.2) == 0.1


def test_ablation_scales_and_controls():
    w = np.array([0.2379, 0.1468, 0.4974, 0.1178])
    for k in range(4):
        keep = [j for j in range(4) if j != k]
        r = renormalized(w, keep)
        assert np.isclose(r.sum(), 1.0) and np.allclose(r, w[keep] / w[keep].sum())
    assert config.HEURISTIC_SCALES == (0.333, 0.133, 0.400, 0.133)
    assert max(range(4), key=lambda i: config.HEURISTIC_SCALES[i]) == 2      # largest on byte 2-grams
    assert config.EQUAL_SCALES == (0.25,) * 4
    assert config.TEMPERATURES == (0.05, 0.10, 0.20, 0.50, 1.00)


def test_network_size():
    from lawsr.model import LAWSRNet, n_parameters
    assert n_parameters(LAWSRNet(4)) == 2433            # Section 3.3
    assert n_parameters(LAWSRNet(1)) == 2241
    assert n_parameters(LAWSRNet(5)) == 2497


def test_random_order_reference():
    # Table 8, AES-256-CTR row: Top-10 0.198% and MRR 0.0018 for 5,046 candidates
    r = random_order_reference(5046)
    assert round(r['Top-10 (%)'], 3) == 0.198 and round(r['MRR'], 4) == 0.0018
    assert np.isclose(r['Top-1 (%)'], 100 / 5046)


def test_fast_features_equal_reference_transcription():
    rng = np.random.default_rng(1)
    frags = rng.integers(0, 256, (40, 512), dtype=np.uint8)
    frags[1] = 0                                          # constant fragment: entropy 0, one 2-gram
    frags[2, :256] = 65                                   # partly compressible
    frags[3] = np.tile(np.arange(16, dtype=np.uint8), 32)  # periodic content
    store = FragmentStore(frags, workers=1)
    a = rng.integers(0, 40, 120)
    b = rng.integers(0, 40, 120)
    fast = store.pair_features(a, b)
    ref = np.array([pair_similarity(frags[i], frags[j]) for i, j in zip(a, b)])
    assert np.abs(fast - ref).max() < 1e-12
    Q = store.query_features([0, 1, 2, 3])
    for qi, q in enumerate([0, 1, 2, 3]):
        ref_q = np.array([pair_similarity(frags[q], frags[j]) for j in range(40)])
        assert np.abs(Q[qi] - ref_q).max() < 1e-12
    S = rng.random((5, 4))
    assert np.allclose(uniform_fusion(S), S.mean(axis=1))
    assert np.allclose(weighted_fusion(S, [0.1, 0.2, 0.3, 0.4]), S @ np.array([0.1, 0.2, 0.3, 0.4]))


def test_order_matters_only_for_compression():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 256, 512, dtype=np.uint8)
    y = np.concatenate([np.zeros(256, dtype=np.uint8), rng.integers(0, 256, 256, dtype=np.uint8)])
    s_xy, s_yx = pair_similarity(x, y), pair_similarity(y, x)
    assert np.allclose(s_xy[:3], s_yx[:3])                # histogram, entropy, 2-gram are symmetric


def test_rank_tie_breaking():
    scores = np.array([[0.0, 0.5, 0.9, 0.5, 0.5]])        # query in column 0, successor in column 3
    tie = np.array([0, 4, 3, 1, 2])
    # candidates: col1 (0.5, tie 4), col2 (0.9), col3 (0.5, tie 1 = successor), col4 (0.5, tie 2)
    assert successor_ranks(scores, [0], [3], tie)[0] == 2  # only the 0.9 candidate ranks ahead
    tie2 = np.array([0, 1, 3, 4, 2])
    assert successor_ranks(scores, [0], [3], tie2)[0] == 4  # col1 and col4 win the tie
    p = top_k_same_source(scores, [0], ['A'], np.array(['A', 'A', 'B', 'A', 'B']), tie, k=2)
    assert np.isclose(p[0], 0.5)                          # top-2: col2 (B), col3 (A)
