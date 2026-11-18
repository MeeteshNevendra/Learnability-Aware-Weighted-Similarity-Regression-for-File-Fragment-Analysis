"""Step 1 - component learnability (Section 3.3, Algorithm 1 steps 1-4, Table 5).

For each component a one-input regressor with the same 64-32 hidden layers is fitted on the training
pairs (five seeds) and evaluated on the validation pairs.  The learnability index (Eq. 17) is computed
from the validation R^2 and Spearman coefficient; the softmax weights (Eq. 18) are listed for every
temperature of the grid.  The single-component fits are also the "single-component regressor" baselines
(the 2-gram regressor of Table 4).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, parse_args, path, save_json  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.metrics import regression_metrics  # noqa: E402
from lawsr.training import run_job  # noqa: E402
from lawsr.weighting import learnability_index, softmax_weights  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--policy', choices=['mean', 'seed0'], default=config.LEARNABILITY_POLICY,
                                                        help='mean: R2 and Spearman averaged over seeds; seed0: first seed only'))
    log = Log(args, 'step1_learnability')
    jobs = [{'name': f'single_{c}', 'seed': s, 'objective': 'mse', 'keep': [k], 'scales': [1.0], 'type_cue': False,
             'cache_dir': path(args, 'cache'), 'out_dir': path(args, 'models')}
            for k, c in enumerate(config.COMPONENTS) for s in args.seeds]
    with ProcessPoolExecutor(max_workers=min(args.train_workers, len(jobs))) as ex:
        for r in ex.map(run_job, jobs):
            log(f"{r['name']} seed {r['seed']}: best val MSE {r['best_val_mse']:.5f} at epoch {r['best_epoch']} "
                f"({r['epochs_run']} epochs, {r['seconds']}s)")
    y_val = np.load(path(args, 'cache', 'pairs_val.npz'))['y']
    out = {'policy': args.policy, 'seeds': args.seeds, 'components': {}}
    r2s, rhos = [], []
    for c in config.COMPONENTS:
        per_seed = {}
        for s in args.seeds:
            m = regression_metrics(y_val, np.load(path(args, 'models', f'single_{c}__seed{s}__pred_val.npy')))
            per_seed[s] = {'R2': m['R2'], 'Spearman': m['Spearman']}
        if args.policy == 'mean':
            r2 = float(np.mean([v['R2'] for v in per_seed.values()]))
            rho = float(np.mean([v['Spearman'] for v in per_seed.values()]))
        else:
            r2, rho = per_seed[args.seeds[0]]['R2'], per_seed[args.seeds[0]]['Spearman']
        out['components'][c] = {'per_seed': per_seed, 'R2': r2, 'Spearman': rho}
        r2s.append(r2)
        rhos.append(rho)
    lk = learnability_index(r2s, rhos)
    out['learnability'] = dict(zip(config.COMPONENTS, map(float, lk)))
    out['weights_by_tau'] = {f'{t:.2f}': list(map(float, softmax_weights(lk, t))) for t in config.TEMPERATURES}
    for c in config.COMPONENTS:
        log(f"{c}: validation R2 {out['components'][c]['R2']:.4f}  Spearman {out['components'][c]['Spearman']:.4f}  "
            f"learnability {out['learnability'][c]:.4f}")
    save_json(out, 'metrics', 'learnability.json', args=args)


if __name__ == '__main__':
    main()
