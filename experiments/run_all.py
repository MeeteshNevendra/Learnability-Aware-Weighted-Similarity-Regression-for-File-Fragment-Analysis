"""Run every experiment step in order.

    python -I experiments/run_all.py --data-root /path/to/dataset [--out results] [--workers N]

Steps (see README.md for the mapping to the manuscript):
    0 prepare        cache pair arrays; check stored similarities against the package implementation
    1 learnability   single-component regressors -> Table 5 (Eq. 17)
    2 temperature    LAWSR temperature grid and selection (Eq. 18, Algorithm 1)
    3 models         unweighted / equal / heuristic / binary / ablations / type-assisted
    4 pair_metrics   Tables 4, 6, 9, 11 (pair metrics)
    5 retrieval      Table 7 and Table 11 Top-10
    6 govdocs1       Table 8
    7 png_repair     Table 10
    8 report         RESULTS.md, tables/*.csv, Figure 3
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['step0_prepare', 'step1_learnability', 'step2_temperature', 'step3_models', 'step4_pair_metrics',
         'step5_retrieval', 'step6_govdocs1', 'step7_png_repair', 'step8_report']


def main():
    argv = sys.argv[1:]
    start = 0
    if '--from-step' in argv:
        i = argv.index('--from-step')
        start = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    for k, step in enumerate(STEPS):
        if k < start:
            continue
        t = time.time()
        print(f'=== {step} ===', flush=True)
        r = subprocess.run([sys.executable, '-I', os.path.join(HERE, f'{step}.py')] + argv)
        if r.returncode != 0:
            sys.exit(f'{step} failed with exit code {r.returncode}')
        print(f'=== {step} finished in {time.time() - t:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
