"""Pair sampling in four equally represented modes (manuscript Section 3.4).

Used to build the Govdocs1 external pair sets (5,000 pairs per mode).  The primary pairs come from the
released dataset and are not resampled.
"""
import numpy as np

MODES = ('same_file', 'same_extension_different_file', 'different_extension', 'random')


def sample_pairs(rng, file_of, ext_of, n_per_mode, modes=MODES):
    """Sample ordered fragment pairs within one partition.

    file_of, ext_of: integer arrays giving the source file and recorded extension of each retained
    fragment.  The anchor is drawn uniformly from the retained pool and the partner uniformly from the
    fragments that satisfy the mode's condition:
        same_file                       two distinct fragments of one source file
        same_extension_different_file   distinct files with the same recorded extension
        different_extension             distinct files with different recorded extensions
        random                          any two distinct fragments (may come from one file)
    Unordered pairs are unique across all modes; the output interleaves the modes (pair k has mode
    k mod 4).  Returns (a, b, mode_index, modes).
    """
    n = len(file_of)
    by_file, by_ext = {}, {}
    for i in range(n):
        by_file.setdefault(int(file_of[i]), []).append(i)
        by_ext.setdefault(int(ext_of[i]), []).append(i)
    by_file = {k: np.array(v) for k, v in by_file.items()}
    by_ext = {k: np.array(v) for k, v in by_ext.items()}
    seen = set()
    out = {m: [] for m in range(len(modes))}

    def take(m, i, j):
        key = (i, j) if i < j else (j, i)
        if i == j or key in seen:
            return False
        seen.add(key)
        out[m].append((i, j))
        return True

    for m, mode in enumerate(modes):
        tries = 0
        while len(out[m]) < n_per_mode:
            tries += 1
            if tries > 200 * n_per_mode:
                raise RuntimeError(f'could not sample {n_per_mode} unique pairs for {mode}')
            i = int(rng.integers(n))
            if mode == 'same_file':
                cand = by_file[int(file_of[i])]
                if cand.size < 2:
                    continue
                j = int(cand[rng.integers(cand.size)])
            elif mode == 'same_extension_different_file':
                cand = by_ext[int(ext_of[i])]
                j = int(cand[rng.integers(cand.size)])
                if file_of[j] == file_of[i]:
                    continue
            elif mode == 'different_extension':
                j = int(rng.integers(n))
                if ext_of[j] == ext_of[i]:
                    continue
            elif mode == 'random':
                j = int(rng.integers(n))
            else:
                raise ValueError(mode)
            take(m, i, j)
    a, b, mi = [], [], []
    for k in range(n_per_mode):
        for m in range(len(modes)):
            i, j = out[m][k]
            a.append(i)
            b.append(j)
            mi.append(m)
    return np.array(a), np.array(b), np.array(mi), list(modes)
