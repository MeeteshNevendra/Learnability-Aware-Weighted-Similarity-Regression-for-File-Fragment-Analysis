"""Pair-level evaluation (manuscript Section 3.5, Eq. 24)."""
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

BANDS = [(0.5, 0.6, '0.50 to <0.60'), (0.6, 0.7, '0.60 to <0.70'), (0.7, 0.8, '0.70 to <0.80'),
         (0.8, 0.9, '0.80 to <0.90'), (0.9, 1.0000001, '0.90 to 1.00')]


def _corr(fn, a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if np.std(a) == 0 or np.std(b) == 0:
            return float('nan')
        return float(fn(a, b)[0])


def regression_metrics(y, pred):
    """Eq. (24) and correlations; e = prediction - reference."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    e = pred - y
    sse = float((e ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {'R2': 1.0 - sse / sst if sst > 0 else float('nan'), 'MSE': float((e ** 2).mean()),
            'RMSE': float(np.sqrt((e ** 2).mean())), 'MAE': float(np.abs(e).mean()),
            'Pearson': _corr(stats.pearsonr, pred, y), 'Spearman': _corr(stats.spearmanr, pred, y)}


def discrimination_metrics(z, score):
    """Same-file discrimination with z as the class label: AUC and average precision."""
    z = np.asarray(z).astype(int)
    score = np.asarray(score, dtype=np.float64)
    return {'AUC': float(roc_auc_score(z, score)), 'AP': float(average_precision_score(z, score))}


def within_file_spearman(y, score, mode):
    """Spearman correlation on the same-file sampling mode (constant origin label, graded reference)."""
    m = np.asarray(mode) == 'same_file'
    return _corr(stats.spearmanr, np.asarray(score, dtype=np.float64)[m], np.asarray(y, dtype=np.float64)[m])


def reference_band(y):
    y = np.asarray(y, dtype=np.float64)
    out = np.full(len(y), '', dtype=object)
    for lo, hi, name in BANDS:
        out[(y >= lo) & (y < hi)] = name
    return out


def grouped_errors(y, pred, groups, order):
    """Pairs, mean reference, mean prediction, MAE, MSE and mean signed residual for each group."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    groups = np.asarray(groups)
    rows = []
    for g in order:
        m = groups == g
        e = pred[m] - y[m]
        rows.append({'group': g, 'pairs': int(m.sum()), 'mean_target': float(y[m].mean()),
                     'mean_prediction': float(pred[m].mean()), 'MAE': float(np.abs(e).mean()),
                     'MSE': float((e ** 2).mean()), 'mean_residual': float(e.mean())})
    return rows
