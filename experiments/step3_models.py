"""Step 3 - remaining learned models (Sections 3.5, 3.6 and 4.8), five seeds each.

    unweighted     same network, unscaled components (scales 1)               Tables 4, 6, 7
    equal          every component scaled by 0.25                              Table 6
    heuristic      fixed vector (0.333, 0.133, 0.400, 0.133)                   Table 6
    binary         matched binary classifier: unscaled, binary cross-entropy   Tables 4, 7
    without_<c>    content ablation: the other three components, scales w*
                   renormalized to sum to one, refitted                        Table 11, panel A
    type_assisted  w* * s plus extension agreement (Eq. 12) as a fifth input   Table 11, panel B
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Log, parse_args, path, selection  # noqa: E402
from lawsr import config  # noqa: E402
from lawsr.training import run_job  # noqa: E402
from lawsr.weighting import renormalized  # noqa: E402


def main():
    args = parse_args(__doc__, lambda p: p.add_argument('--type-cue-scale', type=float, default=config.TYPE_CUE_SCALE))
    log = Log(args, 'step3_models')
    _, w = selection(args)
    base = {'objective': 'mse', 'type_cue': False, 'cache_dir': path(args, 'cache'), 'out_dir': path(args, 'models')}
    specs = [('unweighted', [0, 1, 2, 3], list(config.UNSCALED), {}),
             ('equal', [0, 1, 2, 3], list(config.EQUAL_SCALES), {}),
             ('heuristic', [0, 1, 2, 3], list(config.HEURISTIC_SCALES), {}),
             ('binary', [0, 1, 2, 3], list(config.UNSCALED), {'objective': 'bce'})]
    for k, c in enumerate(config.COMPONENTS):
        keep = [j for j in range(4) if j != k]
        specs.append((f'without_{c}', keep, list(renormalized(w, keep)), {}))
    specs.append(('type_assisted', [0, 1, 2, 3], list(w) + [args.type_cue_scale], {'type_cue': True}))
    jobs = [{**base, 'name': n, 'keep': keep, 'scales': sc, 'seed': s, **extra}
            for n, keep, sc, extra in specs for s in args.seeds]
    with ProcessPoolExecutor(max_workers=min(args.train_workers, len(jobs))) as ex:
        for r in ex.map(run_job, jobs):
            log(f"{r['name']} seed {r['seed']}: best val MSE {r['best_val_mse']:.5f} at epoch {r['best_epoch']} "
                f"({r['epochs_run']} epochs, {r['seconds']}s)")


if __name__ == '__main__':
    main()
