"""Step 4 - test-pair evaluation (Sections 3.5, 4.1, 4.3, 4.6, 4.8; Tables 4, 6, 9, 11).

All methods are scored on the same 80,000 test pairs.  Fixed scores are compared with the reference
directly.  For graded-error metrics the binary classifier's probability is multiplied by the mean
positive reference of the training pairs (a monotonic rescaling).  Discrimination uses the same-file
label z.  Within-file Spearman uses the same-file sampling mode.  The type-assisted model is also
scored with corrupted extension labels for 4, 8 and 15 test files (nested selections).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, fixed_scorers, lawsr_name, load_json, parse_args, path, save_json, score_learned, selection  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.data import load_pairs, load_type_corruption  # noqa: E402
from lawsr.metrics import BANDS, discrimination_metrics, grouped_errors, reference_band, regression_metrics, within_file_spearman  # noqa: E402
from lawsr.reference import type_cue  # noqa: E402

MODES = ['same_file', 'same_extension_different_file', 'different_extension', 'random']


def all_metrics(y, z, mode, graded, ranking):
    m = regression_metrics(y, graded)
    m.update(discrimination_metrics(z, ranking))
    m['Within-file Spearman'] = within_file_spearman(y, ranking, mode)
    return m


def main():
    args = parse_args(__doc__)
    log = Log(args, 'step4_pair_metrics')
    sel, w = selection(args)
    prep = load_json('metrics', 'prepare.json', args=args)
    T = np.load(path(args, 'cache', 'pairs_test.npz'), allow_pickle=True)
    S, y, z, mode = T['S'], T['y'].astype(np.float64), T['z'], T['mode']
    out = {'fixed': {}, 'learned': {}, 'mean_positive_reference_train': prep['mean_positive_reference_train']}

    for name, fn in fixed_scorers(w).items():
        sc = fn(S.astype(np.float64))
        out['fixed'][name] = all_metrics(y, z, mode, sc, sc)
        log(f"{name}: R2 {out['fixed'][name]['R2']:.3f}  AUC {out['fixed'][name]['AUC']:.3f}")

    learned = {'Histogram regressor': 'single_s_hist', 'Entropy regressor': 'single_s_ent',
               '2-gram regressor': 'single_s_ngram', 'Compression regressor': 'single_s_comp',
               'Matched binary classifier': 'binary', 'Unweighted regressor': 'unweighted',
               'Equal scaling': 'equal', 'Heuristic scaling': 'heuristic', 'LAWSR': lawsr_name(sel),
               'Content plus correct type cue': 'type_assisted'}
    learned.update({f'Without {config.COMPONENT_NAMES[c].lower()}': f'without_{c}' for c in config.COMPONENTS})
    for label, model in learned.items():
        per_seed = {}
        for s in args.seeds:
            p = np.load(path(args, 'models', f'{model}__seed{s}__pred_test.npy')).astype(np.float64)
            graded = p * prep['mean_positive_reference_train'] if model == 'binary' else p
            per_seed[s] = all_metrics(y, z, mode, graded, p)
        out['learned'][label] = {'model': model, 'per_seed': per_seed}
        log(f"{label}: R2 {np.mean([v['R2'] for v in per_seed.values()]):.3f}  "
            f"AUC {np.mean([v['AUC'] for v in per_seed.values()]):.3f}")

    # type cue with corrupted labels (Section 3.6)
    D = load_pairs(args.data_root, 'test')
    tc = load_type_corruption(args.data_root)
    out['type_corruption'] = {}
    for pct in config.TYPE_CORRUPTION_PERCENT:
        hit = tc[tc[f'corrupted_at_{pct}pct'] == 1]
        repl = dict(zip(hit['file_id'], hit['replacement_extension']))
        ea = np.array([repl.get(f, e) for f, e in zip(D['file_a'], D['ext_a'])])
        eb = np.array([repl.get(f, e) for f, e in zip(D['file_b'], D['ext_b'])])
        st = type_cue(ea, eb).astype(np.float32)
        per_seed = {}
        for s in args.seeds:
            p = score_learned(args, 'type_assisted', s, S, st).astype(np.float64)
            per_seed[s] = all_metrics(y, z, mode, p, p)
        out['type_corruption'][pct] = {'files': int(len(hit)), 'percent_of_test_files': 100.0 * len(hit) / tc.shape[0],
                                       'pairs_with_changed_type_cue': int((st != D['s_type']).sum()), 'per_seed': per_seed}
        log(f"type cue, {pct}% nominal ({len(hit)} files): R2 {np.mean([v['R2'] for v in per_seed.values()]):.3f}")

    # error structure of the final LAWSR model (Table 9)
    bands = reference_band(y)
    by_mode, by_band = {}, {}
    for s in args.seeds:
        p = np.load(path(args, 'models', f'{lawsr_name(sel)}__seed{s}__pred_test.npy')).astype(np.float64)
        rows = grouped_errors(y, p, mode, MODES)
        rows.append(grouped_errors(y, p, np.full(len(y), 'Aggregate'), ['Aggregate'])[0])
        by_mode[s] = rows
        sf = mode == 'same_file'
        rows = grouped_errors(y[sf], p[sf], bands[sf], [b[2] for b in BANDS])
        rows.append(grouped_errors(y[sf], p[sf], np.full(sf.sum(), 'All pairs in the same-file mode'),
                                   ['All pairs in the same-file mode'])[0])
        by_band[s] = rows
    out['error_by_mode'], out['error_by_band'] = by_mode, by_band
    save_json(out, 'metrics', 'pair_metrics.json', args=args)


if __name__ == '__main__':
    main()
