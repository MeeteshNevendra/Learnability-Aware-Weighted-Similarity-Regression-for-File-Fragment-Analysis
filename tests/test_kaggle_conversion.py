"""Checks of dataset/prepare_kaggle.py on a small synthetic release in the Kaggle format, and of the
pair sampler and the two feature implementations used by the dataset scripts."""
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lawsr.data import load_pairs, load_primary_retrieval, load_type_corruption  # noqa: E402
from lawsr.reference import provenance_reference  # noqa: E402
from lawsr.sampling import sample_pairs  # noqa: E402
from lawsr.similarity import FragmentStore, PairFeaturizer, pair_similarity  # noqa: E402


def _converter():
    spec = importlib.util.spec_from_file_location('prepare_kaggle', os.path.join(ROOT, 'dataset', 'prepare_kaggle.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_release(kdir, rng):
    """Nine files (three per extension), split 5/2/2, fragments as in the release: blocks at k*512 plus an
    end-aligned block at S - 512; pairs of every relationship with a correct positional cue."""
    exts = ['png', 'jpg', 'pdf']
    files = {}
    for k in range(9):
        fid = f'F{k + 1:04d}'
        S = int(rng.integers(3000, 9000))
        if S % 512 == 0:
            S += 7
        data = rng.integers(0, 256 if k % 2 else 64, S, dtype=np.uint8)
        starts = [j * 512 for j in range(S // 512)] + [S - 512]
        files[fid] = {'ext': exts[k % 3], 'S': S, 'data': data, 'starts': starts,
                      'split': 'train' if k < 5 else ('val' if k < 7 else 'test')}
    for split in ('train', 'val', 'test'):
        fids = [f for f in files if files[f]['split'] == split]
        frs = [(f, i) for f in fids for i in range(len(files[f]['starts']))]
        pairs = [((f, 0), (f, len(files[f]['starts']) - 1)) for f in fids]          # one same-file pair per file
        for a in frs:                                                               # every fragment occurs in a pair
            b = frs[rng.integers(len(frs))]
            if b != a:
                pairs.append((a, b))
        rows, X1, X2 = [], [], []
        for k, ((fa, ia), (fb, ib)) in enumerate(pairs):
            A, B = files[fa], files[fb]
            oa, ob = A['starts'][ia], B['starts'][ib]
            z = fa == fb
            mode = 'same_file' if z else ('same_type_diff_file' if A['ext'] == B['ext'] else 'diff_type')
            if k % 7 == 6:
                mode = 'random'
            rows.append({'pair_id': k, 'a_file_id': fa, 'a_file_ext': A['ext'], 'a_fragment_id': ia, 'a_start_offset': oa,
                         'b_file_id': fb, 'b_file_ext': B['ext'], 'b_fragment_id': ib, 'b_start_offset': ob,
                         'pair_mode': mode, 's_hist': 0.5, 's_ent': 0.9, 's_ngram': 0.7, 's_comp': 1.0,
                         's_pos': (1 - abs(oa / A['S'] - ob / B['S'])) if z else 0.0, 's_type': float(A['ext'] == B['ext']),
                         'target_similarity': 0.4})
            X1.append(A['data'][oa:oa + 512])
            X2.append(B['data'][ob:ob + 512])
        pd.DataFrame(rows).to_csv(os.path.join(kdir, f'{split}_meta.csv'), index=False)
        np.savez(os.path.join(kdir, f'{split}.npz'), X1=np.stack(X1), X2=np.stack(X2),
                 y=np.full(len(rows), 0.4, dtype=np.float32))
    return files


def test_conversion_of_a_synthetic_release():
    conv = _converter()
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        kdir, out = os.path.join(tmp, 'kaggle'), os.path.join(tmp, 'dataset')
        os.makedirs(kdir)
        files = _synthetic_release(kdir, rng)
        info = conv.convert(kdir, out, features='recompute', target='eq23', workers=1)
        assert info['files_without_source_length'] == 0
        split = pd.read_csv(os.path.join(out, '02_splits', 'file_split.csv')).set_index('file_id')
        assert all(int(split.loc[f, 'source_length']) == files[f]['S'] for f in files)
        for s in ('train', 'val', 'test'):
            D = load_pairs(out, s)
            csv = pd.read_csv(os.path.join(out, '04_pairs', f'{s}_pairs.csv'))
            S = csv['file_a'].map(lambda f: files[f]['S']).values, csv['file_b'].map(lambda f: files[f]['S']).values
            y = np.where(D['z'] == 1, provenance_reference(1, csv['offset_a'] / S[0], csv['offset_b'] / S[1]), 0.0)
            assert np.allclose(D['y'], y, atol=1e-6)
            with np.load(os.path.join(out, '04_pairs', f'{s}.npz')) as raw:
                ref = np.array([pair_similarity(a, b) for a, b in zip(raw['X1'], raw['X2'])])
            assert np.abs(ref - D['S']).max() < 1e-6                     # recomputed from the bytes
            assert set(D['mode']) <= {'same_file', 'same_extension_different_file', 'different_extension', 'random'}
        R = load_primary_retrieval(out)
        pool = pd.read_csv(os.path.join(out, '05_retrieval', 'test_candidate_pool.csv'))
        offs, src = pool['offset'].values, pool['file_id'].values
        assert (offs[R['successor_cols']] == offs[R['query_cols']] + 512).all()
        assert (src[R['successor_cols']] == src[R['query_cols']]).all()
        expected = sum(len(files[f]['starts']) - 2 for f in files if files[f]['split'] == 'test')
        assert len(R['query_cols']) == expected                          # the end-aligned block is never a successor
        assert len(load_type_corruption(out)) == 2


def test_corruption_counts_follow_the_rounding_rule():
    conv = _converter()
    for n, expected in [(75, {5: 4, 10: 8, 20: 15}), (67, {5: 3, 10: 7, 20: 13})]:
        files = pd.DataFrame({'extension': ['png', 'jpg', 'pdf'] * 30, 'split': ['test'] * n + ['train'] * (90 - n)},
                             index=[f'F{k:04d}' for k in range(90)])
        tc, n_sel = conv.corruption_table(files)
        assert n_sel == expected and len(tc) == n
        sel = tc[tc['corrupted_at_20pct'] == 1]
        assert (sel['replacement_extension'] != sel['recorded_extension']).all()
        assert set(tc.loc[tc['corrupted_at_5pct'] == 1, 'file_id']) <= set(tc.loc[tc['corrupted_at_10pct'] == 1, 'file_id'])


def test_sampler_modes():
    rng = np.random.default_rng(0)
    file_of = np.repeat(np.arange(12), 10)
    ext_of = file_of % 3
    a, b, m, modes = sample_pairs(rng, file_of, ext_of, 50)
    assert np.bincount(m).tolist() == [50] * 4
    assert (file_of[a[m == 0]] == file_of[b[m == 0]]).all() and (a != b).all()
    assert (file_of[a[m == 1]] != file_of[b[m == 1]]).all() and (ext_of[a[m == 1]] == ext_of[b[m == 1]]).all()
    assert (ext_of[a[m == 2]] != ext_of[b[m == 2]]).all()
    assert len({tuple(sorted(p)) for p in zip(a, b)}) == len(a)


def test_featurizers_agree():
    rng = np.random.default_rng(3)
    frags = rng.integers(0, 256, (50, 512), dtype=np.uint8)
    frags[::5, 100:400] = 7
    a, b = rng.integers(0, 50, 200), rng.integers(0, 50, 200)
    ref = np.array([pair_similarity(frags[i], frags[j]) for i, j in zip(a, b)])
    assert np.abs(PairFeaturizer(frags).features(a, b) - ref).max() < 1e-12
    assert np.abs(FragmentStore(frags, workers=1).pair_features(a, b) - ref).max() < 1e-12
