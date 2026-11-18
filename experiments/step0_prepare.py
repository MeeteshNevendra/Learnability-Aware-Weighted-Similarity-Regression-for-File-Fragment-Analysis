"""Step 0 - cache the pair arrays and check the stored similarities against the package implementation.

Writes results/cache/pairs_{train,val,test}.npz and results/metrics/prepare.json.  When the dataset was
prepared with `--features released` (dataset_info.json), the released similarities are used as they
are and the byte-level check only reports how far they are from Eqs. (7)-(11).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, parse_args, path, save_json  # noqa: E402
from lawsr.data import SPLITS, load_pairs  # noqa: E402
from lawsr.similarity import pair_similarity  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--check-pairs', type=int, default=300,
                                                        help='pairs per partition recomputed from bytes'))
    log = Log(args, 'step0_prepare')
    ds_info_path = os.path.join(args.data_root, 'dataset_info.json')
    ds_info = json.load(open(ds_info_path, encoding='utf-8')) if os.path.exists(ds_info_path) else {}
    strict = ds_info.get('features', 'recompute') == 'recompute'
    log(f"dataset: {ds_info.get('source', args.data_root)}; similarities: {ds_info.get('features', 'recompute')}; "
        f"target: {ds_info.get('target', 'eq23')}")
    info = {'dataset_info': ds_info}
    rng = np.random.default_rng(0)
    for s in SPLITS:
        D = load_pairs(args.data_root, s)
        np.savez(path(args, 'cache', f'pairs_{s}.npz'), S=D['S'], y=D['y'], z=D['z'], s_type=D['s_type'], mode=D['mode'])
        raw = np.load(os.path.join(args.data_root, '04_pairs', f'{s}.npz'))
        idx = rng.choice(len(D['y']), min(args.check_pairs, len(D['y'])), replace=False)
        ref = np.array([pair_similarity(raw['X1'][i], raw['X2'][i]) for i in idx])
        diff = float(np.abs(ref - D['S'][idx]).max())
        info[s] = {'pairs': int(len(D['y'])), 'same_file_pairs': int(D['z'].sum()),
                   'feature_check_pairs': int(len(idx)), 'feature_check_max_abs_diff': diff}
        log(f'{s}: {len(D["y"]):,} pairs; recomputed features of {len(idx)} pairs, max |diff| = {diff:.2e}')
        if strict and diff > 1e-5:
            raise SystemExit('stored similarities differ from the package implementation')
    tr = np.load(path(args, 'cache', 'pairs_train.npz'))
    info['mean_positive_reference_train'] = float(tr['y'][tr['z'] == 1].mean())
    log('mean positive reference of the training pairs (binary-classifier rescaling):', round(info['mean_positive_reference_train'], 6))
    save_json(info, 'metrics', 'prepare.json', args=args)


if __name__ == '__main__':
    main()
