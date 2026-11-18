"""Provenance reference and context cues (manuscript Sections 3.2 and 3.4).

These quantities come from the extraction records (source-file identity, starting offset, source
length, recorded extension), never from fragment bytes.  The main model does not receive them.
"""
import numpy as np


def relative_position(offset, source_length):
    """p_i = o_i / S_i."""
    return np.asarray(offset, dtype=np.float64) / np.asarray(source_length, dtype=np.float64)


def provenance_reference(same_file, p_i, p_j):
    """Eq. (23): y_ij = z_ij (0.5 + 0.5 (1 - |p_i - p_j|)) = 0.5 z_ij + 0.5 s_pos."""
    z = np.asarray(same_file, dtype=np.float64)
    return z * (0.5 + 0.5 * (1.0 - np.abs(np.asarray(p_i, dtype=np.float64) - np.asarray(p_j, dtype=np.float64))))


def positional_cue(same_file, p_i, p_j):
    """Eq. (13): s_pos = z_ij (1 - |p_i - p_j|)."""
    z = np.asarray(same_file, dtype=np.float64)
    return z * (1.0 - np.abs(np.asarray(p_i, dtype=np.float64) - np.asarray(p_j, dtype=np.float64)))


def type_cue(ext_i, ext_j):
    """Eq. (12): s_type = 1{t_i = t_j}."""
    return (np.asarray(ext_i) == np.asarray(ext_j)).astype(np.float64)
