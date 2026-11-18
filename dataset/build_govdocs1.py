"""Build the Govdocs1 external evaluation and stress sets (manuscript Section 3.6, Table 8).

Usage:
    python -I dataset/build_govdocs1.py <dataset_root> <govdocs1_archive001_extracted_dir>

Govdocs1 archive 001: https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/001.zip
(extract its members, without folders, into one directory).

Protocol
  * Govdocs1 archive 001; a member file is eligible when it holds at least 1,024 bytes.
  * Seven conditions: aligned blocks of 512, 256, 1,024 and 4,096 bytes; unaligned 512-byte blocks
    with a seeded start offset of 1-511 bytes per file; gzip payload (level 6, mtime 0) and
    AES-256-CTR payload (independent key and counter block per file), both with aligned 512-byte blocks.
  * Blocks are complete blocks of the (transformed) stream, at a stride equal to the block length.
  * Pair pool: a contiguous run of at most 80 blocks per file (the first blocks of the stream), from
    files with at least two complete blocks; 20,000 pairs, 5,000 per sampling mode.
  * Retrieval: the eligible member identifiers are sorted and permuted with NumPy PCG64 seed 42; the
    first 76 files in that order with at least 67 complete blocks are selected; the first 31 keep their
    first 67 blocks and the other 45 their first 66 blocks; every retained block except the last of its
    run is a query and its exact successor is the next block.  With 76 files this gives 5,047 blocks,
    4,971 queries and 5,046 candidates per query.
  * Provenance labels (Eq. 23) are recomputed in each stream from source identity and stream offsets.
"""
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from lawsr.config import MIN_SOURCE_BYTES as MIN_FILE_BYTES  # noqa: E402
from lawsr.reference import positional_cue  # noqa: E402
from lawsr.reference import provenance_reference as provenance_target  # noqa: E402
from lawsr.sampling import sample_pairs  # noqa: E402
from lawsr.similarity import PairFeaturizer  # noqa: E402

ROOT, GOV = sys.argv[1], sys.argv[2]
OUT = os.path.join(ROOT, '08_govdocs1_external')
os.makedirs(OUT, exist_ok=True)
CONDITIONS = [  # name, block length, transform, alignment
    ('aligned_512', 512, 'none', 'aligned'),
    ('aligned_256', 256, 'none', 'aligned'),
    ('aligned_1024', 1024, 'none', 'aligned'),
    ('aligned_4096', 4096, 'none', 'aligned'),
    ('unaligned_512', 512, 'none', 'unaligned'),
    ('gzip_512', 512, 'gzip', 'aligned'),
    ('aes256ctr_512', 512, 'aes-256-ctr', 'aligned'),
]
SEEDS = {'retrieval_permutation_pcg64': 42, 'unaligned_offsets': 511, 'aes_keys': 256, 'tie_order': 42,
         'fragment_ids': 2025, 'pairs': {c[0]: 200 + k for k, c in enumerate(CONDITIONS)}}
POOL_MAX, PAIRS_PER_MODE = 80, 5_000
RET_FILES, RET_MIN, RET_FIRST, RET_LONG, RET_SHORT = 76, 67, 31, 67, 66
MODES = ['same_file', 'same_extension_different_file', 'different_extension', 'random']
lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


t0 = time.time()
# ----------------------------------------------------------------------------- archive manifest
names = sorted(os.listdir(GOV))
recs = []
for nm in names:
    p = os.path.join(GOV, nm)
    with open(p, 'rb') as fh:
        b = fh.read()
    stem, ext = os.path.splitext(nm)
    recs.append({'member_id': stem, 'filename': nm, 'extension': ext.lower().lstrip('.'), 'size_bytes': len(b),
                 'sha256': hashlib.sha256(b).hexdigest(), 'eligible': int(len(b) >= MIN_FILE_BYTES)})
arch = pd.DataFrame(recs)
arch.to_csv(os.path.join(OUT, 'archive001_manifest.csv'), index=False)
elig = arch[arch['eligible'] == 1].sort_values('member_id').reset_index(drop=True)
say(f'archive 001: {len(arch)} members, {len(elig)} eligible (>= {MIN_FILE_BYTES} bytes)')
say('eligible extensions:', elig['extension'].value_counts().to_dict())
DATA = {}
for r in elig.itertuples():
    with open(os.path.join(GOV, r.filename), 'rb') as fh:
        DATA[r.member_id] = fh.read()

# per-file transform parameters (fixed for all conditions)
ru = np.random.default_rng(SEEDS['unaligned_offsets'])
UNALIGNED = {m: int(ru.integers(1, 512)) for m in elig['member_id']}       # 1..511 inclusive
ra = np.random.default_rng(SEEDS['aes_keys'])
AES = {}
for m in elig['member_id']:
    key = ra.bytes(32)
    ctr0 = ra.bytes(8) + b'\x00' * 8                                          # nonce || 64-bit counter = 0
    AES[m] = (key, ctr0)
PERM = np.random.Generator(np.random.PCG64(SEEDS['retrieval_permutation_pcg64'])).permutation(
    np.array(sorted(elig['member_id'])))


def stream(m, transform):
    b = DATA[m]
    if transform == 'none':
        return b
    if transform == 'gzip':
        return gzip.compress(b, compresslevel=6, mtime=0)
    key, ctr0 = AES[m]
    enc = Cipher(algorithms.AES(key), modes.CTR(ctr0)).encryptor()
    return enc.update(b) + enc.finalize()


ext_of = dict(zip(elig['member_id'], elig['extension']))
cond_rows = []
rid = np.random.default_rng(SEEDS['fragment_ids'])
used_ids = set()


def new_ids(n):
    out = []
    while len(out) < n:
        h = f'GX{int(rid.integers(0, 1 << 40)):010X}'
        if h not in used_ids:
            used_ids.add(h)
            out.append(h)
    return out


for cname, L, transform, align in CONDITIONS:
    cdir = os.path.join(OUT, cname)
    os.makedirs(cdir, exist_ok=True)
    smf, frows, blobs = [], [], []
    nblocks = {}
    for m in elig['member_id']:
        s = stream(m, transform)
        start = UNALIGNED[m] if align == 'unaligned' else 0
        n = max(0, (len(s) - start) // L)
        nblocks[m] = n
        rec = {'member_id': m, 'extension': ext_of[m], 'stream_length': len(s), 'start_offset': start,
               'n_complete_blocks': n, 'n_pool_blocks': min(n, POOL_MAX) if n >= 2 else 0}
        if transform == 'aes-256-ctr':
            rec['aes256_key_hex'] = AES[m][0].hex()
            rec['ctr_initial_block_hex'] = AES[m][1].hex()
        if transform == 'gzip':
            rec['gzip_level'] = 6
        smf.append(rec)
        if n >= 2:
            for k in range(min(n, POOL_MAX)):
                off = start + k * L
                frows.append((m, ext_of[m], k, off, L, len(s), off / len(s)))
                blobs.append(np.frombuffer(s[off:off + L], dtype=np.uint8))
    sm = pd.DataFrame(smf)
    frag = pd.DataFrame(frows, columns=['member_id', 'extension', 'block_index', 'offset', 'length', 'stream_length',
                                        'rel_position'])
    X = np.stack(blobs)
    frag.insert(0, 'fragment_id', new_ids(len(frag)))
    frag.insert(1, 'row', np.arange(len(frag)))

    # retrieval selection (same rule in every condition, applied to that condition's stream)
    selected = [m for m in PERM if nblocks[m] >= RET_MIN][:RET_FILES]
    keep = {m: (RET_LONG if i < RET_FIRST else RET_SHORT) for i, m in enumerate(selected)}
    sm['retrieval_order'] = sm['member_id'].map({m: i + 1 for i, m in enumerate(selected)}).astype('Int64')
    sm['retrieval_blocks'] = sm['member_id'].map(keep).fillna(0).astype(int)
    frag['in_retrieval'] = [int(k < keep.get(m, 0)) for m, k in zip(frag['member_id'], frag['block_index'])]

    # pairs
    fcode = pd.factorize(frag['member_id'])[0]
    ecode = pd.factorize(frag['extension'])[0]
    a, b, mi, modes_ = sample_pairs(np.random.default_rng(SEEDS['pairs'][cname]), fcode, ecode, PAIRS_PER_MODE, MODES)
    feat = PairFeaturizer(X, cache_bigrams=(L <= 1024))
    F = feat.features(a, b)
    A, B = frag.iloc[a].reset_index(drop=True), frag.iloc[b].reset_index(drop=True)
    z = (A['member_id'].values == B['member_id'].values).astype(np.int8)
    y = provenance_target(z, A['rel_position'].values, B['rel_position'].values)
    P = pd.DataFrame({
        'pair_id': [f'{cname}-{k:05d}' for k in range(len(a))], 'mode': [modes_[k] for k in mi],
        'fragment_a': A['fragment_id'], 'fragment_b': B['fragment_id'], 'member_a': A['member_id'],
        'member_b': B['member_id'], 'ext_a': A['extension'], 'ext_b': B['extension'], 'offset_a': A['offset'],
        'offset_b': B['offset'], 'stream_length_a': A['stream_length'], 'stream_length_b': B['stream_length'],
        'p_a': A['rel_position'], 'p_b': B['rel_position'], 'same_file': z, 'target_y': y,
        's_hist': F[:, 0], 's_ent': F[:, 1], 's_ngram': F[:, 2], 's_comp': F[:, 3],
        's_pos': positional_cue(z, A['rel_position'].values, B['rel_position'].values),
        's_type': (A['extension'].values == B['extension'].values).astype(np.int8)})

    # retrieval records
    R = frag[frag['in_retrieval'] == 1]
    keyset = {(m, k): fid for m, k, fid in zip(R['member_id'], R['block_index'], R['fragment_id'])}
    Q = R[[(m, k + 1) in keyset for m, k in zip(R['member_id'], R['block_index'])]].copy()
    Q['successor_fragment'] = [keyset[(m, k + 1)] for m, k in zip(Q['member_id'], Q['block_index'])]
    Q = Q.rename(columns={'fragment_id': 'query_fragment', 'offset': 'query_offset'})
    Q.insert(0, 'query_id', [f'{cname}-q{k:05d}' for k in range(len(Q))])
    tie = np.random.default_rng(SEEDS['tie_order']).permutation(np.array(sorted(frag['fragment_id'])))

    # write
    sm.to_csv(os.path.join(cdir, 'stream_manifest.csv'), index=False)
    frag.to_csv(os.path.join(cdir, 'fragments.csv'), index=False)
    np.save(os.path.join(cdir, 'fragment_bytes.npy'), X)
    P.to_csv(os.path.join(cdir, 'pairs.csv'), index=False, float_format='%.10g')
    np.savez(os.path.join(cdir, 'pairs.npz'), X1=X[a], X2=X[b], y=y.astype(np.float32), z=z, mode=mi.astype(np.int8),
             S=F.astype(np.float32), feature_names=np.array(['s_hist', 's_ent', 's_ngram', 's_comp']),
             mode_names=np.array(modes_))
    Q[['query_id', 'query_fragment', 'member_id', 'extension', 'block_index', 'query_offset', 'successor_fragment']].to_csv(
        os.path.join(cdir, 'retrieval_queries.csv'), index=False)
    R[['fragment_id', 'row', 'member_id', 'extension', 'block_index', 'offset']].to_csv(
        os.path.join(cdir, 'retrieval_pool.csv'), index=False)
    pd.DataFrame({'fragment_id': tie, 'tie_rank': np.arange(len(tie))}).to_csv(os.path.join(cdir, 'tie_order.csv'), index=False)

    row = {'condition': cname, 'block_length': L, 'transform': transform, 'alignment': align,
           'files_in_pair_pool': int((sm['n_pool_blocks'] > 0).sum()), 'pool_fragments': int(len(frag)),
           'pairs': int(len(P)), 'pairs_per_mode': PAIRS_PER_MODE, 'same_file_pairs_total': int(z.sum()),
           'retrieval_files': len(selected), 'retrieval_blocks': int(len(R)), 'queries': int(len(Q)),
           'candidates_per_query': int(len(R) - 1),
           'random_top10_percent': round(1000 / (len(R) - 1), 4) if len(R) > 1 else None}
    cond_rows.append(row)
    say(f'[{cname}] files in pool={row["files_in_pair_pool"]} pool fragments={row["pool_fragments"]} '
        f'pairs={row["pairs"]} | retrieval files={row["retrieval_files"]} blocks={row["retrieval_blocks"]} '
        f'queries={row["queries"]} candidates/query={row["candidates_per_query"]} ({time.time() - t0:.0f}s)')
    g = P.groupby('mode', sort=False)
    say('   mean target by mode:', g['target_y'].mean().round(4).to_dict(),
        '| mean s_ngram by mode:', g['s_ngram'].mean().round(3).to_dict())

C = pd.DataFrame(cond_rows)
C.to_csv(os.path.join(OUT, 'conditions.csv'), index=False)
with open(os.path.join(OUT, 'build_parameters.json'), 'w', encoding='utf-8') as fh:
    json.dump({'seeds': SEEDS, 'pool_max_blocks': POOL_MAX, 'pairs_per_mode': PAIRS_PER_MODE,
               'retrieval': {'files': RET_FILES, 'min_blocks': RET_MIN, 'first_files': RET_FIRST,
                             'blocks_first': RET_LONG, 'blocks_rest': RET_SHORT},
               'conditions': [dict(zip(['name', 'block_length', 'transform', 'alignment'], c)) for c in CONDITIONS]},
              fh, indent=1)
os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
with open(os.path.join(ROOT, 'reports', 'govdocs1_build_report.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n\n' + C.to_string(index=False) + '\n')
say(f'done in {time.time() - t0:.0f}s')
