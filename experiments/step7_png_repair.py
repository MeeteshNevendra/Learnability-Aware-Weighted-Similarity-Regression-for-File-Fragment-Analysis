"""Step 7 - controlled PNG gap repair (Sections 3.6 and 4.7; Table 10; Figure 4).

For each of the seven test PNGs the candidates (the withheld block and the distractors) are scored
against the query (the block before the gap), ordered by decreasing score with the gap's fixed tie order,
and passed to the validator, which tests at most the first ten.  The learned methods use one fixed
checkpoint (seed 0 by default).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, lawsr_name, parse_args, save_json, score_learned, selection  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.data import load_png_repair  # noqa: E402
from lawsr.png_validator import repair_trial  # noqa: E402
from lawsr.similarity import FragmentStore, uniform_fusion  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--checkpoint-seed', type=int, default=config.PNG_CHECKPOINT_SEED))
    log = Log(args, 'step7_png_repair')
    if not os.path.exists(os.path.join(args.data_root, '07_png_repair', 'gaps.csv')):
        log('No PNG repair task in the dataset root (prepare it with dataset/prepare_kaggle.py --png-sources). Step skipped.')
        save_json({'skipped': True}, 'metrics', 'png_repair.json', args=args)
        return
    sel, _ = selection(args)
    s = args.checkpoint_seed
    methods = {'Uniform fusion': uniform_fusion, 'Direct byte 2-gram Jaccard': lambda F: F[..., 2],
               'Unweighted regressor': lambda F: score_learned(args, 'unweighted', s, F),
               'LAWSR': lambda F: score_learned(args, lawsr_name(sel), s, F)}
    gaps, arrays = load_png_repair(args.data_root)
    out = {'checkpoint_seed': s, 'max_checks': config.PNG_MAX_CHECKS, 'trials': {m: [] for m in methods}}
    for g in gaps.itertuples():
        A = arrays[g.gap_id]
        cands = A['candidates']
        store = FragmentStore(np.vstack([A['query'][None, :], cands]), workers=1)
        F = store.query_features([0])[0, 1:, :]                       # query first; drop the query itself
        with open(os.path.join(args.data_root, g.damaged_file), 'rb') as fh:
            damaged = fh.read()
        true_idx = int(np.argmax(A['is_true']))
        for name, fn in methods.items():
            sc = np.asarray(fn(F), dtype=np.float64)
            order = np.lexsort((A['tie_rank'], -sc))                   # decreasing score, then tie order
            pos = int(np.where(order == true_idx)[0][0])
            res = repair_trial(damaged, g.gap_offset, [cands[i].tobytes() for i in order], g.original_sha256,
                               true_index=pos)
            res.update({'gap_id': g.gap_id, 'rank_of_true_block': pos + 1, 'candidates': int(len(cands))})
            out['trials'][name].append(res)
            log(f"{g.gap_id} {name}: true block at rank {pos + 1} -> {res['outcome']}")
    out['summary'] = {m: {'Files': len(v), 'Top-10 hits': sum(bool(t['top10_hit']) for t in v),
                          'Exact': sum(t['outcome'] == 'exact' for t in v), 'Wrong': sum(t['outcome'] == 'wrong' for t in v),
                          'Abstained': sum(t['outcome'] == 'abstained' for t in v)} for m, v in out['trials'].items()}
    for m, v in out['summary'].items():
        v['Exact recovery (%)'] = 100.0 * v['Exact'] / v['Files']
        log(m, v)
    save_json(out, 'metrics', 'png_repair.json', args=args)


if __name__ == '__main__':
    main()
