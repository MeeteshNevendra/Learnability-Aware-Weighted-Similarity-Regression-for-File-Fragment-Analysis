"""Shared helpers for the experiment scripts."""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from lawsr import config  # noqa: E402
from lawsr.similarity import uniform_fusion, weighted_fusion  # noqa: E402
# torch is imported lazily (in `learned`): worker processes started for the compression component
# re-import this module, and loading the CUDA-enabled torch in dozens of processes can exhaust the
# Windows commit limit (WinError 1455).


def parse_args(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-root', required=True, help='dataset package (the folder that holds 04_pairs/, 05_retrieval/, ...)')
    p.add_argument('--out', default=os.path.join(REPO, 'results'), help='output folder (default: results/)')
    p.add_argument('--workers', type=int, default=None,
                   help='worker processes for the compression component (default: CPU count - 2, at most 32)')
    p.add_argument('--train-workers', type=int, default=None,
                   help='parallel network fits (default: min(12, CPU count - 2)); each loads PyTorch')
    p.add_argument('--seeds', type=int, nargs='+', default=list(config.SEEDS))
    if extra:
        extra(p)
    a = p.parse_args()
    a.data_root = os.path.abspath(a.data_root)
    a.out = os.path.abspath(a.out)
    for sub in ('cache', 'models', 'metrics', 'tables', 'figures', 'logs'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    if a.workers is None:
        a.workers = max(1, min(32, (os.cpu_count() or 2) - 2))
    if a.train_workers is None:
        a.train_workers = max(1, min(12, (os.cpu_count() or 2) - 2))
    return a


def path(args, *parts):
    return os.path.join(args.out, *parts)


def save_json(obj, *parts, args):
    p = path(args, *parts)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, default=float)
    return p


def load_json(*parts, args):
    with open(path(args, *parts), encoding='utf-8') as fh:
        return json.load(fh)


class Log:
    def __init__(self, args, name):
        self.fh = open(path(args, 'logs', f'{name}.log'), 'a', encoding='utf-8')
        self.t0 = time.time()

    def __call__(self, *msg):
        s = f'[{time.time() - self.t0:7.1f}s] ' + ' '.join(str(m) for m in msg)
        print(s, flush=True)
        self.fh.write(s + '\n')
        self.fh.flush()


# ----------------------------------------------------------------------------- methods
def selection(args):
    """Learnability, the selected temperature and the frozen LAWSR scales w*."""
    sel = load_json('metrics', 'selection.json', args=args)
    return sel, np.asarray(sel['weights_selected'], dtype=np.float64)


def lawsr_name(sel):
    return f"lawsr_tau{sel['tau_selected']:.2f}"


def fixed_scorers(w_star):
    """Fixed-score baselines (Section 3.5): the four similarities, Eq. (15) and Eq. (16)."""
    return {'Histogram cosine': lambda S: S[..., 0], 'Entropy similarity': lambda S: S[..., 1],
            'Byte 2-gram Jaccard': lambda S: S[..., 2], 'Compression similarity': lambda S: S[..., 3],
            'Uniform fusion': uniform_fusion, 'Weighted linear fusion': lambda S: weighted_fusion(S, w_star)}


_SCORERS = {}


def learned(args, name, seed):
    key = (name, seed)
    if key not in _SCORERS:
        import torch
        from lawsr.model import load_scorer
        p = path(args, 'models', f'{name}__seed{seed}.pt')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        _SCORERS[key] = (load_scorer(p), ck['keep'], ck['type_cue'], ck)
    return _SCORERS[key]


def score_learned(args, name, seed, S, s_type=None):
    scorer, keep, uses_type, _ = learned(args, name, seed)
    X = np.asarray(S)[..., keep]
    if uses_type:
        X = np.concatenate([X, np.asarray(s_type)[..., None]], axis=-1)
    return scorer(X.astype(np.float32))


def mean_sd(values):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return float('nan'), float('nan')
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
