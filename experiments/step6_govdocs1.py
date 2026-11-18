"""Step 6 - Govdocs1 transfer and stress conditions with the frozen primary fits (Sections 3.6 and 4.5; Table 8).

For every condition: same-file AUC on the condition's 20,000 pairs for LAWSR (five seeds), uniform fusion
and the selected single-component score (the component with the highest validation learnability), and
exact-successor retrieval on the condition's 76-file pool (4,971 queries, 5,046 candidates) for LAWSR
and uniform fusion.  Nothing is refitted on the external files.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, lawsr_name, parse_args, save_json, score_learned, selection  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.data import govdocs1_conditions, load_govdocs1_pairs, load_govdocs1_retrieval  # noqa: E402
from lawsr.metrics import discrimination_metrics  # noqa: E402
from lawsr.retrieval import random_order_reference, successor_ranks, summarize_ranks  # noqa: E402
from lawsr.similarity import FragmentStore, uniform_fusion  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: (p.add_argument('--chunk', type=int, default=512),
                                          p.add_argument('--conditions', nargs='*', default=None)))
    log = Log(args, 'step6_govdocs1')
    if not os.path.exists(os.path.join(args.data_root, '08_govdocs1_external', 'conditions.csv')):
        log('Govdocs1 sets not found in the dataset root; build them with dataset/build_govdocs1.py. Step skipped.')
        save_json({'skipped': True, 'conditions': {}}, 'metrics', 'govdocs1.json', args=args)
        return
    sel, w = selection(args)
    lw = lawsr_name(sel)
    single = int(np.argmax([sel['learnability'][c] for c in config.COMPONENTS]))
    conds = govdocs1_conditions(args.data_root)
    out = {'single_score_component': config.COMPONENTS[single], 'conditions': {}}
    for c in conds.itertuples():
        if args.conditions and c.condition not in args.conditions:
            continue
        P = load_govdocs1_pairs(args.data_root, c.condition)
        S = P['S'].astype(np.float64)
        res = {'block_length': int(c.block_length), 'transform': c.transform, 'alignment': c.alignment,
               'pairs': int(len(P['z'])), 'same_file_pairs': int(P['z'].sum()),
               'uniform_auc': discrimination_metrics(P['z'], uniform_fusion(S))['AUC'],
               'single_score_auc': discrimination_metrics(P['z'], S[:, single])['AUC'],
               'lawsr_auc_per_seed': {s: discrimination_metrics(P['z'], score_learned(args, lw, s, P['S']))['AUC']
                                      for s in args.seeds}}
        R = load_govdocs1_retrieval(args.data_root, c.condition)
        store = FragmentStore(R['frags'], workers=args.workers)
        Qn = len(R['query_cols'])
        ranks = {'uniform': [], **{s: [] for s in args.seeds}}
        with store._pool() as ex:
            for s0 in range(0, Qn, args.chunk):
                qc = R['query_cols'][s0:s0 + args.chunk]
                tc = R['successor_cols'][s0:s0 + args.chunk]
                F = store.query_features(qc, ex)
                ranks['uniform'].append(successor_ranks(uniform_fusion(F), qc, tc, R['tie_rank']))
                for s in args.seeds:
                    ranks[s].append(successor_ranks(score_learned(args, lw, s, F), qc, tc, R['tie_rank']))
        ranks = {k: np.concatenate(v) for k, v in ranks.items()}
        res.update({'retrieval_pool': int(len(store)), 'queries': int(Qn), 'candidates_per_query': int(len(store) - 1),
                    'random_ordering': random_order_reference(len(store) - 1),
                    'uniform_retrieval': summarize_ranks(ranks['uniform']),
                    'lawsr_retrieval_per_seed': {s: summarize_ranks(ranks[s]) for s in args.seeds}})
        out['conditions'][c.condition] = res
        log(f"{c.condition}: LAWSR AUC {np.mean(list(res['lawsr_auc_per_seed'].values())):.3f} | uniform AUC "
            f"{res['uniform_auc']:.3f} | LAWSR Top-10 "
            f"{np.mean([v['Top-10 (%)'] for v in res['lawsr_retrieval_per_seed'].values()]):.1f}% | uniform Top-10 "
            f"{res['uniform_retrieval']['Top-10 (%)']:.1f}%")
        save_json(out, 'metrics', 'govdocs1.json', args=args)


if __name__ == '__main__':
    main()
