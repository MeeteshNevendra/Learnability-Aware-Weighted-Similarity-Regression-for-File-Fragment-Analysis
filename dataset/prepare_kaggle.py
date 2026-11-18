"""Convert the Kaggle release into the dataset layout read by experiments/.

Release: https://www.kaggle.com/datasets/meetesh/continuous-fragment-similarity-dataset
(train.npz, val.npz, test.npz with X1, X2, y; train_meta.csv, val_meta.csv, test_meta.csv).

Usage:
    python -I dataset/prepare_kaggle.py --kaggle-dir <extracted release> --out <dataset_root>
           [--features recompute|released] [--target eq23|released] [--png-sources <dir>] [--workers N]

Steps (manuscript Sections 3.4-3.6):
  1. Read the three partitions and check them: pair counts and sampling modes, file-disjoint partitions,
     identical bytes for every occurrence of a fragment, one offset per fragment.
  2. Source length S of every file from its same-file pairs: the released positional cue is
     s_pos = 1 - |p_a - p_b| with p = offset / S, so S = |o_a - o_b| / (1 - s_pos).
  3. Target (default --target eq23): the provenance reference of Eq. (23),
     y = z (0.5 + 0.5 (1 - |p_a - p_b|)), with z = 1 for fragments of the same source file.
     --target released keeps the release's target_similarity column instead.
  4. Similarities (default --features recompute): the four components of Eqs. (7)-(11) recomputed
     from the fragment bytes, pair order (a, b) as released.  --features released keeps the
     release's s_hist, s_ent, s_ngram and s_comp columns.
  5. Exact-successor retrieval records for the test partition (Section 3.5): the pool is every
     retained test fragment; a query is eligible when the fragment of its file at offset + 512 is
     retained; ties are broken by a seed-42 permutation of the fragment identifiers.
  6. Type-label corruption (Section 3.6): nested selections of 5%, 10% and 20% of the test files
     (counts rounded to the nearest integer, halves up) in a fixed seed-42 order; each selected file
     gets a different recorded extension, drawn with the same generator.
  7. Optional controlled PNG repair (Section 3.6): needs the intact test PNG files, named
     <file_id>.png, in --png-sources.  Without them the PNG experiment is skipped.
Writes dataset_info.json and reports/conversion_report.txt in the output folder.
"""
import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lawsr.config import COMPONENTS, FRAGMENT_LENGTH, TIE_ORDER_SEED, TYPE_CORRUPTION_PERCENT  # noqa: E402
from lawsr.png_validator import png_chunks, validate_png  # noqa: E402
from lawsr.reference import positional_cue, provenance_reference, type_cue  # noqa: E402
from lawsr.sampling import MODES  # noqa: E402
from lawsr.similarity import FragmentStore  # noqa: E402

SPLITS = ('train', 'val', 'test')
MODE_MAP = {'same_file': 'same_file', 'same_type_diff_file': 'same_extension_different_file',
            'diff_type': 'different_extension', 'random': 'random'}
SEED_CORRUPTION, SEED_GAP = 42, 42


def _s(col):
    return np.asarray(col.astype(str).tolist())


class Report:
    def __init__(self):
        self.lines, self.t0 = [], time.time()

    def __call__(self, *msg):
        s = ' '.join(str(m) for m in msg)
        print(f'[{time.time() - self.t0:6.1f}s] {s}', flush=True)
        self.lines.append(s)


def read_release(kdir, log):
    parts = {}
    for s in SPLITS:
        meta = pd.read_csv(os.path.join(kdir, f'{s}_meta.csv'))
        z = np.load(os.path.join(kdir, f'{s}.npz'), allow_pickle=False)
        X1, X2, y = z['X1'], z['X2'], z['y']
        if not (len(meta) == len(X1) == len(X2) == len(y)) or X1.shape[1] != FRAGMENT_LENGTH:
            raise SystemExit(f'{s}: inconsistent array and metadata sizes')
        unknown = set(_s(meta['pair_mode'])) - set(MODE_MAP)
        if unknown:
            raise SystemExit(f'{s}: unknown sampling modes {unknown}')
        parts[s] = {'meta': meta, 'X1': X1, 'X2': X2, 'y_released': y.astype(np.float64)}
        log(f'{s}: {len(meta):,} pairs; modes {meta["pair_mode"].value_counts().to_dict()}')
    return parts


def register_fragments(parts, log):
    """One row per distinct retained fragment (file id, retained index); bytes checked across occurrences."""
    keys, rows, recs, blobs = {}, [], [], []
    mismatches = 0
    for s in SPLITS:
        m = parts[s]['meta']
        for side, X in (('a', parts[s]['X1']), ('b', parts[s]['X2'])):
            fid = _s(m[f'{side}_file_id'])
            idx = m[f'{side}_fragment_id'].values.astype(np.int64)
            off = m[f'{side}_start_offset'].values.astype(np.int64)
            ext = np.char.lower(_s(m[f'{side}_file_ext']))
            key = np.char.add(np.char.add(fid, '_'), np.char.zfill(idx.astype(str), 3))
            uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
            row_of = np.empty(len(uniq), dtype=np.int64)
            for u, (k, f) in enumerate(zip(uniq, first)):
                if k not in keys:
                    keys[k] = len(recs)
                    recs.append((k, fid[f], ext[f], s, int(idx[f]), int(off[f])))
                    blobs.append(np.asarray(X[f]))
                r = keys[k]
                if recs[r][5] != off[f] or recs[r][3] != s or recs[r][2] != ext[f]:
                    raise SystemExit(f'fragment {k} has inconsistent offset, partition or extension')
                row_of[u] = r
            rows_side = row_of[inv]
            store = np.stack(blobs)
            for c in range(0, len(X), 50_000):
                mismatches += int((~(np.asarray(X[c:c + 50_000]) == store[rows_side[c:c + 50_000]]).all(axis=1)).sum())
            parts[s][f'rows_{side}'] = rows_side
    frag = pd.DataFrame(recs, columns=['fragment_id', 'file_id', 'extension', 'split', 'fragment_index', 'offset'])
    frag.insert(1, 'row', np.arange(len(frag)))
    log(f'distinct retained fragments: {len(frag):,}; occurrences whose bytes differ from the first: {mismatches}')
    if mismatches:
        raise SystemExit('the release stores different bytes for the same fragment')
    files = frag.groupby('file_id').agg(extension=('extension', 'first'), split=('split', 'first'),
                                        n_retained_fragments=('fragment_id', 'size'),
                                        max_index=('fragment_index', 'max'))
    if (frag.groupby('file_id')['split'].nunique() > 1).any():
        raise SystemExit('a source file occurs in more than one partition')
    # the retained pool is reconstructed from the fragments that occur in pairs; retained indices run
    # from 0, so a gap means that a retained fragment never occurs in any pair
    gaps = files['max_index'] + 1 - files['n_retained_fragments']
    log(f'files whose retained fragment indices have gaps: {int((gaps > 0).sum())} '
        f'({int(gaps.sum())} fragments not present in any pair)')
    return frag, np.stack(blobs), files.drop(columns='max_index')


def source_lengths(parts, files, log):
    est = []
    for s in SPLITS:
        m = parts[s]['meta']
        same = (m['a_file_id'] == m['b_file_id']) & (m['a_start_offset'] != m['b_start_offset'])
        sf = m[same]
        S = (sf['a_start_offset'] - sf['b_start_offset']).abs().values / (1.0 - sf['s_pos'].values)
        est.append(pd.DataFrame({'file_id': _s(sf['a_file_id']), 'S': S}))
    est = pd.concat(est)
    g = est.groupby('file_id')['S']
    S_int = np.rint(g.median()).astype(np.int64)
    dev = float((est['S'] - est['file_id'].map(S_int)).abs().max()) if len(est) else 0.0
    files = files.join(S_int.rename('source_length'), how='left')
    missing = int(files['source_length'].isna().sum())
    log(f'source lengths recovered for {len(files) - missing} of {len(files)} files '
        f'(max deviation of a single estimate {dev:.3g} bytes)')
    if dev > 0.5:
        raise SystemExit('the positional cue implies inconsistent source lengths')
    return files, missing


def corruption_table(files, rng_seed=SEED_CORRUPTION):
    test_files = sorted(files.index[files['split'] == 'test'])
    exts = sorted(files['extension'].unique())
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(len(test_files))
    n_sel = {p: int(np.floor(len(test_files) * p / 100 + 0.5)) for p in TYPE_CORRUPTION_PERCENT}
    rows = []
    for rank, i in enumerate(order, 1):
        f = test_files[i]
        repl = ''
        if rank <= max(n_sel.values()):
            choices = [e for e in exts if e != files.loc[f, 'extension']]
            repl = choices[int(rng.integers(len(choices)))]
        rows.append({'file_id': f, 'recorded_extension': files.loc[f, 'extension'], 'corruption_order': rank,
                     'replacement_extension': repl, **{f'corrupted_at_{p}pct': int(rank <= n) for p, n in n_sel.items()}})
    return pd.DataFrame(rows), n_sel


def build_png_repair(out, frag, X, files, png_dir, log):
    test_png = sorted(files.index[(files['split'] == 'test') & (files['extension'] == 'png')])
    pool = frag[(frag['split'] == 'test') & (frag['extension'] == 'png')].reset_index(drop=True)
    pool_bytes = X[pool['row'].values]
    os.makedirs(os.path.join(out, '07_png_repair', 'damaged'), exist_ok=True)
    rng = np.random.default_rng(SEED_GAP)
    gaps, cands, skipped = [], [], []
    for fid in test_png:
        path = next((os.path.join(png_dir, f'{fid}{e}') for e in ('.png', '.PNG') if os.path.exists(os.path.join(png_dir, f'{fid}{e}'))), None)
        if path is None:
            skipped.append(fid)
            continue
        with open(path, 'rb') as fh:
            data = fh.read()
        own = frag[frag['file_id'] == fid]
        if not all(bytes(X[r]) == data[o:o + FRAGMENT_LENGTH] for r, o in zip(own['row'], own['offset'])):
            raise SystemExit(f'{path} does not match the released fragments of {fid}')
        validate_png(data, full_decode=True)
        chunks, _ = png_chunks(data)
        eligible = []
        for ci, c in enumerate(chunks):
            if c['type'] != 'IDAT':
                continue
            o = -(-c['data_start'] // FRAGMENT_LENGTH) * FRAGMENT_LENGTH
            while o + FRAGMENT_LENGTH <= c['data_end']:
                if o >= FRAGMENT_LENGTH:
                    eligible.append((o, ci, c))
                o += FRAGMENT_LENGTH
        if not eligible:
            skipped.append(fid)
            continue
        o, ci, c = eligible[int(rng.integers(len(eligible)))]
        withheld, query = data[o:o + FRAGMENT_LENGTH], data[o - FRAGMENT_LENGTH:o]
        gap_id = f'gap{len(gaps) + 1}_{fid}'
        with open(os.path.join(out, '07_png_repair', 'damaged', f'{gap_id}.png'), 'wb') as fh:
            fh.write(data[:o] + b'\x00' * FRAGMENT_LENGTH + data[o + FRAGMENT_LENGTH:])
        w = np.frombuffer(withheld, dtype=np.uint8)
        is_query = (pool['file_id'].values == fid) & (pool['offset'].values == o - FRAGMENT_LENGTH)
        identical = (pool_bytes == w).all(axis=1)
        keep = ~is_query & ~identical
        hit = pool[(pool['file_id'] == fid) & (pool['offset'] == o)]
        ids = [hit['fragment_id'].iloc[0] if len(hit) else f'WITHHELD_{fid}'] + pool.loc[keep, 'fragment_id'].tolist()
        B = np.vstack([w[None, :], pool_bytes[keep]])
        is_true = np.zeros(len(ids), dtype=np.int8)
        is_true[0] = 1
        perm = np.random.default_rng(TIE_ORDER_SEED).permutation(np.array(sorted(ids)))
        tie = {k: r for r, k in enumerate(perm)}
        np.savez(os.path.join(out, '07_png_repair', f'{gap_id}.npz'), query=np.frombuffer(query, dtype=np.uint8),
                 candidates=B, candidate_ids=np.array(ids), is_true=is_true, tie_rank=np.array([tie[k] for k in ids]))
        cands += [{'gap_id': gap_id, 'candidate_id': k, 'is_true_block': int(t), 'tie_rank': tie[k]} for k, t in zip(ids, is_true)]
        gaps.append({'gap_id': gap_id, 'file_id': fid, 'source_length': len(data), 'gap_offset': o,
                     'gap_length': FRAGMENT_LENGTH, 'query_offset': o - FRAGMENT_LENGTH, 'idat_chunk_index': ci,
                     'n_candidates': len(ids), 'original_sha256': hashlib.sha256(data).hexdigest(),
                     'damaged_file': f'07_png_repair/damaged/{gap_id}.png', 'candidate_arrays': f'07_png_repair/{gap_id}.npz'})
    if gaps:
        pd.DataFrame(gaps).to_csv(os.path.join(out, '07_png_repair', 'gaps.csv'), index=False)
        pd.DataFrame(cands).to_csv(os.path.join(out, '07_png_repair', 'candidates.csv'), index=False)
    log(f'PNG repair: {len(gaps)} gap(s) from {len(test_png)} test PNG files; without intact file or IDAT gap: {len(skipped)}')
    return len(gaps)


def convert(kaggle_dir, out, features='recompute', target='eq23', png_sources=None, workers=None):
    log = Report()
    for d in ('02_splits', '03_fragments', '04_pairs', '05_retrieval', '06_type_cue', 'reports'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    parts = read_release(kaggle_dir, log)
    frag, X, files = register_fragments(parts, log)
    files, missing = source_lengths(parts, files, log)
    frag['source_length'] = frag['file_id'].map(files['source_length'])
    frag['rel_position'] = frag['offset'] / frag['source_length']
    frag['length'] = FRAGMENT_LENGTH
    log('files per partition and extension:\n' + pd.crosstab(files['extension'], files['split']).to_string())

    fs = FragmentStore(X, workers=workers) if features == 'recompute' else None
    info = {'source': 'Kaggle release meetesh/continuous-fragment-similarity-dataset', 'features': features,
            'target': target, 'pairs': {}, 'files': files['split'].value_counts().to_dict(),
            'distinct_fragments': int(len(frag)), 'files_without_source_length': missing}
    for s in SPLITS:
        P, m = parts[s], parts[s]['meta']
        A, B = frag.iloc[P['rows_a']].reset_index(drop=True), frag.iloc[P['rows_b']].reset_index(drop=True)
        z = (A['file_id'].values == B['file_id'].values).astype(np.int8)
        pa, pb = A['rel_position'].values, B['rel_position'].values
        y_eq23 = np.where(z == 1, provenance_reference(1, pa, pb), 0.0)
        check = float(np.abs(y_eq23 - (0.5 * z + 0.5 * m['s_pos'].values)).max())
        log(f'{s}: Eq. (23) from offsets vs 0.5 z + 0.5 s_pos (released): max |diff| {check:.2e}')
        y = y_eq23 if target == 'eq23' else P['y_released']
        released = m[list(COMPONENTS)].values.astype(np.float64)
        if fs is not None:
            S = fs.pair_features(P['rows_a'], P['rows_b'])
            log(f'{s}: recomputed vs released similarities, max |diff| per component: '
                + ', '.join(f'{c} {d:.3g}' for c, d in zip(COMPONENTS, np.abs(S - released).max(axis=0))))
        else:
            S = released
        modes = np.array([MODE_MAP[x] for x in _s(m['pair_mode'])])
        mode_idx = np.array([MODES.index(x) for x in modes], dtype=np.int8)
        np.savez(os.path.join(out, '04_pairs', f'{s}.npz'), X1=P['X1'], X2=P['X2'], y=y.astype(np.float32), z=z,
                 mode=mode_idx, S=S.astype(np.float32), feature_names=np.array(COMPONENTS), mode_names=np.array(MODES))
        csv = pd.DataFrame({
            'pair_id': [f'{s}-{k:06d}' for k in range(len(m))], 'mode': modes,
            'fragment_a': A['fragment_id'], 'fragment_b': B['fragment_id'], 'file_a': A['file_id'], 'file_b': B['file_id'],
            'ext_a': A['extension'], 'ext_b': B['extension'], 'offset_a': A['offset'], 'offset_b': B['offset'],
            'source_length_a': A['source_length'], 'source_length_b': B['source_length'], 'p_a': pa, 'p_b': pb,
            'same_file': z, 'target_y': y, 'target_eq23': y_eq23, 'target_released': P['y_released'],
            **{c: S[:, k] for k, c in enumerate(COMPONENTS)},
            's_pos': np.where(z == 1, positional_cue(1, pa, pb), 0.0), 's_type': type_cue(A['extension'], B['extension']),
            **{f'released_{c}': m[c].values for c in list(COMPONENTS) + ['s_pos', 's_type']}})
        csv.to_csv(os.path.join(out, '04_pairs', f'{s}_pairs.csv'), index=False, float_format='%.10g')
        info['pairs'][s] = {'n': int(len(m)), 'same_file': int(z.sum()), 'per_mode': pd.Series(modes).value_counts().to_dict()}

    frag[['fragment_id', 'row', 'file_id', 'extension', 'split', 'fragment_index', 'offset', 'length', 'source_length',
          'rel_position']].to_csv(os.path.join(out, '03_fragments', 'fragments.csv'), index=False)
    np.save(os.path.join(out, '03_fragments', 'fragment_bytes.npy'), X)
    files.reset_index().to_csv(os.path.join(out, '02_splits', 'file_split.csv'), index=False)
    for s in SPLITS:
        with open(os.path.join(out, '02_splits', f'{s}_files.txt'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(sorted(files.index[files['split'] == s])) + '\n')

    t = frag[frag['split'] == 'test']
    at = {(f, o): k for f, o, k in zip(t['file_id'], t['offset'], t['fragment_id'])}
    q = t[[(f, o + FRAGMENT_LENGTH) in at for f, o in zip(t['file_id'], t['offset'])]]
    pd.DataFrame({'query_id': [f'q{k:05d}' for k in range(len(q))], 'query_fragment': q['fragment_id'].values,
                  'file_id': q['file_id'].values, 'extension': q['extension'].values, 'query_offset': q['offset'].values,
                  'successor_fragment': [at[(f, o + FRAGMENT_LENGTH)] for f, o in zip(q['file_id'], q['offset'])],
                  'successor_offset': q['offset'].values + FRAGMENT_LENGTH}).to_csv(
        os.path.join(out, '05_retrieval', 'test_successor_queries.csv'), index=False)
    t[['fragment_id', 'row', 'file_id', 'extension', 'offset']].to_csv(
        os.path.join(out, '05_retrieval', 'test_candidate_pool.csv'), index=False)
    perm = np.random.default_rng(TIE_ORDER_SEED).permutation(np.array(sorted(frag['fragment_id'])))
    pd.DataFrame({'fragment_id': perm, 'tie_rank': np.arange(len(perm))}).to_csv(
        os.path.join(out, '05_retrieval', 'tie_order.csv'), index=False)
    log(f'retrieval: {len(t):,} retained test fragments, {len(q):,} eligible successor queries from {q["file_id"].nunique()} files')
    info['retrieval'] = {'pool': int(len(t)), 'queries': int(len(q))}

    tc, n_sel = corruption_table(files)
    tc.to_csv(os.path.join(out, '06_type_cue', 'test_label_corruption.csv'), index=False)
    log(f'type-label corruption over {len(tc)} test files: files relabelled {n_sel}')
    info['type_corruption_files'] = n_sel

    info['png_repair_gaps'] = build_png_repair(out, frag, X, files, png_sources, log) if png_sources else 0
    if not png_sources:
        log('PNG repair: skipped (no --png-sources); experiments/step7 will be skipped')
    with open(os.path.join(out, 'dataset_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh, indent=1, default=int)
    with open(os.path.join(out, 'reports', 'conversion_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(log.lines) + '\n')
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kaggle-dir', required=True, help='folder with train.npz, train_meta.csv, val.npz, ...')
    p.add_argument('--out', required=True, help='dataset root to create (passed to experiments as --data-root)')
    p.add_argument('--features', choices=['recompute', 'released'], default='recompute')
    p.add_argument('--target', choices=['eq23', 'released'], default='eq23')
    p.add_argument('--png-sources', default=None, help='folder with the intact test PNG files (<file_id>.png)')
    p.add_argument('--workers', type=int, default=None)
    a = p.parse_args()
    convert(os.path.abspath(a.kaggle_dir), os.path.abspath(a.out), a.features, a.target, a.png_sources, a.workers)


if __name__ == '__main__':
    main()
