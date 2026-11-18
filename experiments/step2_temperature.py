"""Step 2 - temperature selection (Section 3.3, Algorithm 1 steps 5-20).

For each temperature of the grid {0.05, 0.10, 0.20, 0.50, 1.00} the four-input regressor is fitted on
the training pairs with the softmax weights w(tau) as fixed input scales, and its best checkpoint is
retained by validation MSE.  The temperature with the lowest validation MSE (mean over the five seeds,
implementation choice) is selected; its weights w* and checkpoints are frozen for all later scoring.
The test files are not used.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import Log, load_json, parse_args, path, save_json  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.training import run_job  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--policy', choices=['mean', 'seed0'], default=config.TEMPERATURE_POLICY))
    log = Log(args, 'step2_temperature')
    L = load_json('metrics', 'learnability.json', args=args)
    jobs = [{'name': f'lawsr_tau{t:.2f}', 'seed': s, 'objective': 'mse', 'keep': [0, 1, 2, 3],
             'scales': L['weights_by_tau'][f'{t:.2f}'], 'type_cue': False,
             'cache_dir': path(args, 'cache'), 'out_dir': path(args, 'models')}
            for t in config.TEMPERATURES for s in args.seeds]
    val = {}
    with ProcessPoolExecutor(max_workers=min(args.train_workers, len(jobs))) as ex:
        for r in ex.map(run_job, jobs):
            val.setdefault(r['name'], {})[r['seed']] = r['best_val_mse']
            log(f"{r['name']} seed {r['seed']}: best val MSE {r['best_val_mse']:.5f} at epoch {r['best_epoch']} ({r['seconds']}s)")
    grid = []
    for t in config.TEMPERATURES:
        v = val[f'lawsr_tau{t:.2f}']
        grid.append({'tau': t, 'weights': L['weights_by_tau'][f'{t:.2f}'], 'val_mse_per_seed': v,
                     'val_mse_mean': float(np.mean(list(v.values())))})
    if args.policy == 'mean':
        best = min(grid, key=lambda g: g['val_mse_mean'])
    else:
        best = min(grid, key=lambda g: g['val_mse_per_seed'][args.seeds[0]])
    sel = {'policy': args.policy, 'grid': grid, 'tau_selected': best['tau'], 'weights_selected': best['weights'],
           'learnability': L['learnability'], 'components': list(config.COMPONENTS)}
    for g in grid:
        log(f"tau {g['tau']:.2f}: weights {np.round(g['weights'], 4).tolist()}  mean val MSE {g['val_mse_mean']:.6f}")
    log(f"selected tau = {best['tau']:.2f}; w* = {np.round(best['weights'], 4).tolist()}")
    save_json(sel, 'metrics', 'selection.json', args=args)


if __name__ == '__main__':
    main()
