"""Loaders for the dataset package (see dataset/README.md for its layout and construction)."""
import os

import numpy as np
import pandas as pd

from .config import COMPONENTS
from .reference import type_cue

SPLITS = ('train', 'val', 'test')


def _str(col):
    """A pandas string column as a plain NumPy unicode array (pandas >= 3 returns Arrow-backed arrays)."""
    return np.asarray(col.astype(str).tolist())


def _check_features(names):
    names = [str(x) for x in names]
    if tuple(names) != COMPONENTS:
        raise ValueError(f'unexpected feature order {names}; expected {COMPONENTS}')


def load_pairs(root, split):
    """Pairs of one partition: S (n x 4, Eq. 14 order), y (Eq. 23), z, mode, s_type (Eq. 12) and the
    file identifiers and extensions of both fragments."""
    with np.load(os.path.join(root, '04_pairs', f'{split}.npz')) as z:
        d = {k: z[k] for k in ('feature_names', 'mode_names', 'mode', 'y', 'z', 'S')}
    _check_features(d['feature_names'])
    meta = pd.read_csv(os.path.join(root, '04_pairs', f'{split}_pairs.csv'),
                       usecols=['pair_id', 'mode', 'file_a', 'file_b', 'ext_a', 'ext_b', 'target_y', 'same_file'])
    modes = np.asarray(d['mode_names'])[d['mode']]
    assert (modes == _str(meta['mode'])).all() and np.allclose(d['y'], meta['target_y'].values, atol=1e-6)
    ext_a, ext_b = _str(meta['ext_a']), _str(meta['ext_b'])
    return {'S': d['S'].astype(np.float32), 'y': d['y'].astype(np.float32), 'z': d['z'].astype(np.float32),
            'mode': modes, 's_type': type_cue(ext_a, ext_b).astype(np.float32),
            'file_a': _str(meta['file_a']), 'file_b': _str(meta['file_b']), 'ext_a': ext_a, 'ext_b': ext_b}


def load_primary_retrieval(root):
    """Retained test fragments (candidate pool), eligible queries and the shared tie order."""
    pool = pd.read_csv(os.path.join(root, '05_retrieval', 'test_candidate_pool.csv'))
    q = pd.read_csv(os.path.join(root, '05_retrieval', 'test_successor_queries.csv'))
    tie = pd.read_csv(os.path.join(root, '05_retrieval', 'tie_order.csv')).set_index('fragment_id')['tie_rank']
    X = np.load(os.path.join(root, '03_fragments', 'fragment_bytes.npy'), mmap_mode='r')
    col = {f: i for i, f in enumerate(pool['fragment_id'])}
    return {'frags': np.asarray(X[pool['row'].values]), 'pool_source': _str(pool['file_id']),
            'tie_rank': tie.loc[pool['fragment_id']].values.astype(np.int64),
            'query_cols': np.array([col[f] for f in q['query_fragment']]),
            'successor_cols': np.array([col[f] for f in q['successor_fragment']]),
            'query_source': _str(q['file_id']), 'query_extension': _str(q['extension'])}


def load_type_corruption(root):
    return pd.read_csv(os.path.join(root, '06_type_cue', 'test_label_corruption.csv'))


def govdocs1_conditions(root):
    return pd.read_csv(os.path.join(root, '08_govdocs1_external', 'conditions.csv'))


def load_govdocs1_pairs(root, condition):
    with np.load(os.path.join(root, '08_govdocs1_external', condition, 'pairs.npz')) as z:
        d = {k: z[k] for k in ('feature_names', 'mode_names', 'mode', 'y', 'z', 'S')}
    _check_features(d['feature_names'])
    return {'S': d['S'].astype(np.float32), 'y': d['y'].astype(np.float32), 'z': d['z'].astype(np.float32),
            'mode': np.asarray(d['mode_names'])[d['mode']]}


def load_govdocs1_retrieval(root, condition):
    base = os.path.join(root, '08_govdocs1_external', condition)
    pool = pd.read_csv(os.path.join(base, 'retrieval_pool.csv'), dtype={'member_id': str})
    q = pd.read_csv(os.path.join(base, 'retrieval_queries.csv'), dtype={'member_id': str})
    tie = pd.read_csv(os.path.join(base, 'tie_order.csv')).set_index('fragment_id')['tie_rank']
    X = np.load(os.path.join(base, 'fragment_bytes.npy'), mmap_mode='r')
    col = {f: i for i, f in enumerate(pool['fragment_id'])}
    # tie ranks are re-indexed to the retrieval pool, preserving the condition's seeded order
    order = tie.loc[pool['fragment_id']].values
    return {'frags': np.asarray(X[pool['row'].values]), 'pool_source': _str(pool['member_id']),
            'tie_rank': np.argsort(np.argsort(order, kind='stable'), kind='stable').astype(np.int64),
            'query_cols': np.array([col[f] for f in q['query_fragment']]),
            'successor_cols': np.array([col[f] for f in q['successor_fragment']]),
            'query_source': _str(q['member_id'])}


def load_png_repair(root):
    gaps = pd.read_csv(os.path.join(root, '07_png_repair', 'gaps.csv'))
    arrays = {}
    for g, p in zip(gaps['gap_id'], gaps['candidate_arrays']):
        with np.load(os.path.join(root, p)) as z:
            arrays[g] = {k: z[k] for k in z.files}
    return gaps, arrays
