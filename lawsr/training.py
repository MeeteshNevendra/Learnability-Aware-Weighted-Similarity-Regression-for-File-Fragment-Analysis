"""Network training (manuscript Eq. 22, Algorithm 1, Section 3.7).

For one fit: initialize the network and Adam with the given seed; for each epoch shuffle the
training pairs, take mini-batches of 1,024 pairs and update with Adam (learning rate 0.001);
after every epoch compute the validation MSE; keep the checkpoint with the lowest validation
MSE; stop after 10 consecutive epochs without improvement or after 200 epochs.

objective 'mse'  regression on the provenance reference y (Eq. 22)
objective 'bce'  matched binary classifier: binary cross-entropy on the common-origin label z.
                 All learned variants share the checkpoint criterion (Section 3.5), so its
                 checkpoint is also chosen by validation MSE, here between the predicted
                 probability and its own training label z (implementation choice).
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE
from .model import LAWSRNet, ScaledScorer


def fit_network(S_train, t_train, S_val, t_val, scales, seed, objective='mse', lr=LEARNING_RATE,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    w = torch.as_tensor(np.asarray(scales, dtype=np.float64), dtype=torch.float32)
    Xtr = torch.as_tensor(np.ascontiguousarray(S_train, dtype=np.float32)) * w
    ytr = torch.as_tensor(np.ascontiguousarray(t_train, dtype=np.float32))
    Xva = torch.as_tensor(np.ascontiguousarray(S_val, dtype=np.float32)) * w
    yva = torch.as_tensor(np.ascontiguousarray(t_val, dtype=np.float32))
    net = LAWSRNet(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = F.mse_loss if objective == 'mse' else F.binary_cross_entropy
    best, best_state, best_epoch, bad, history = math.inf, None, 0, 0, []
    n = len(Xtr)
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            loss = loss_fn(net(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(net(Xva), yva).item()
        history.append((epoch, total / n, val_mse))
        if val_mse < best:
            best, best_epoch, bad = val_mse, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    return {'state_dict': best_state, 'best_val_mse': best, 'best_epoch': best_epoch, 'epochs_run': epoch,
            'history': history}


# ----------------------------------------------------------------------------- parallel fitting jobs
_CACHE = {}


def _pairs(cache_dir):
    if cache_dir not in _CACHE:
        _CACHE[cache_dir] = {s: dict(np.load(os.path.join(cache_dir, f'pairs_{s}.npz'))) for s in ('train', 'val', 'test')}
    return _CACHE[cache_dir]


def job_inputs(D, split, job):
    S = D[split]['S'][:, job['keep']]
    if job.get('type_cue'):
        S = np.column_stack([S, D[split]['s_type']])
    return S.astype(np.float32)


def run_job(job):
    """job keys: name, seed, objective, keep (component indices), type_cue (bool), scales, cache_dir, out_dir."""
    torch.set_num_threads(1)
    t0 = time.time()
    D = _pairs(job['cache_dir'])
    target = 'y' if job['objective'] == 'mse' else 'z'
    res = fit_network(job_inputs(D, 'train', job), D['train'][target], job_inputs(D, 'val', job), D['val'][target],
                      job['scales'], job['seed'], job['objective'])
    n_inputs = len(job['keep']) + (1 if job.get('type_cue') else 0)
    os.makedirs(job['out_dir'], exist_ok=True)
    stem = os.path.join(job['out_dir'], f"{job['name']}__seed{job['seed']}")
    torch.save({'state_dict': res['state_dict'], 'scales': list(map(float, job['scales'])), 'n_inputs': n_inputs,
                'keep': list(job['keep']), 'type_cue': bool(job.get('type_cue')), 'objective': job['objective'],
                'seed': job['seed'], 'name': job['name'], 'best_val_mse': res['best_val_mse'],
                'best_epoch': res['best_epoch'], 'epochs_run': res['epochs_run']}, stem + '.pt')
    scorer = ScaledScorer(res['state_dict'], job['scales'], n_inputs)
    for split in ('val', 'test'):
        np.save(stem + f'__pred_{split}.npy', scorer(job_inputs(D, split, job)))
    with open(stem + '__history.json', 'w', encoding='utf-8') as fh:
        json.dump(res['history'], fh)
    return {'name': job['name'], 'seed': job['seed'], 'best_val_mse': res['best_val_mse'], 'best_epoch': res['best_epoch'],
            'epochs_run': res['epochs_run'], 'seconds': round(time.time() - t0, 1)}
