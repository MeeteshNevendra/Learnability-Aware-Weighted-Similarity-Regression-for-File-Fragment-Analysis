"""Learnability index and softmax weights (manuscript Section 3.3)."""
import numpy as np


def learnability_index(r2, rho):
    """Eq. (17): l_k = (max(0, R_k^2) + max(0, rho_k)) / 2."""
    return (np.maximum(0.0, np.asarray(r2, dtype=np.float64)) + np.maximum(0.0, np.asarray(rho, dtype=np.float64))) / 2.0


def softmax_weights(learnability, tau):
    """Eq. (18): w_k = exp(l_k / tau) / sum_m exp(l_m / tau)."""
    z = np.asarray(learnability, dtype=np.float64) / float(tau)
    z = np.exp(z - z.max())
    return z / z.sum()


def renormalized(weights, keep):
    """Input scales of a reduced model (Section 4.8): the remaining scales renormalized to sum to one."""
    w = np.asarray(weights, dtype=np.float64)[list(keep)]
    return w / w.sum()
