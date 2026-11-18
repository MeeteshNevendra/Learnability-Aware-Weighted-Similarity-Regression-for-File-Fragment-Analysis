"""Step 5 - exact-successor retrieval on the test partition (Sections 3.5 and 4.4; Table 7; Table 11 Top-10).

The candidate pool is every retained test fragment; for each eligible query every other fragment is a
candidate, scored with the query as the first argument (Algorithm 1, steps 21-25).  Ties are broken by
the shared seed-42 permutation of fragment identifiers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, fixed_scorers, lawsr_name, parse_args, save_json, score_learned, selection  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.data import load_primary_retrieval  # noqa: E402
from lawsr.retrieval import random_order_reference, successor_ranks, summarize_ranks, top_k_same_source  # noqa: E402
from lawsr.similarity import FragmentStore  # noqa: E402


def rank_all(args, R, store, methods, chunk, log):
    ranks = {k: [] for k in methods}
    p10 = {k: [] for k in methods}
    Q = len(R['query_cols'])
    with store._pool() as ex:
        for s in range(0, Q, chunk):
            qc = R['query_cols'][s:s + chunk]
            F = store.query_features(qc, ex)
            for key, fn in methods.items():
                sc = fn(F)
                ranks[key].append(successor_ranks(sc, qc, R['successor_cols'][s:s + chunk], R['tie_rank']))
                p10[key].append(top_k_same_source(sc, qc, R['query_source'][s:s + chunk], R['pool_source'], R['tie_rank'],
                                                  config.P_AT))
            log(f'queries {min(s + chunk, Q)}/{Q}')
    return {k: np.concatenate(v) for k, v in ranks.items()}, {k: np.concatenate(v) for k, v in p10.items()}


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--chunk', type=int, default=256))
    log = Log(args, 'step5_retrieval')
    sel, w = selection(args)
    R = load_primary_retrieval(args.data_root)
    store = FragmentStore(R['frags'], workers=args.workers)
    P, Q = len(store), len(R['query_cols'])
    log(f'pool {P} retained test fragments, {Q} eligible queries, {P - 1} candidates per query')
    methods = {('fixed', k, None): (lambda F, fn=fn: fn(F)) for k, fn in fixed_scorers(w).items()}
    models = {'Matched binary classifier': 'binary', 'Unweighted regressor': 'unweighted', 'Equal scaling': 'equal',
              'Heuristic scaling': 'heuristic', 'LAWSR': lawsr_name(sel)}
    models.update({f'Without {config.COMPONENT_NAMES[c].lower()}': f'without_{c}' for c in config.COMPONENTS})
    for label, m in models.items():
        for seed in args.seeds:
            methods[('learned', label, seed)] = (lambda F, m=m, seed=seed: score_learned(args, m, seed, F))
    ranks, p10 = rank_all(args, R, store, methods, args.chunk, log)
    same = np.array([(R['pool_source'] == src).sum() - 1 for src in R['query_source']])
    out = {'pool': int(P), 'queries': int(Q), 'candidates_per_query': int(P - 1),
           'random_ordering': random_order_reference(P - 1, same), 'fixed': {}, 'learned': {},
           'queries_by_extension': {str(k): int(v) for k, v in zip(*np.unique(R['query_extension'], return_counts=True))}}
    for (kind, label, seed), r in ranks.items():
        summ = summarize_ranks(r, p10[(kind, label, seed)])
        if kind == 'fixed':
            out['fixed'][label] = summ
        else:
            out['learned'].setdefault(label, {'model': models[label], 'per_seed': {}})['per_seed'][seed] = summ
    for label, v in out['learned'].items():
        log(f"{label}: Top-10 {np.mean([x['Top-10 (%)'] for x in v['per_seed'].values()]):.1f}%")
    save_json(out, 'metrics', 'retrieval_primary.json', args=args)


if __name__ == '__main__':
    main()
