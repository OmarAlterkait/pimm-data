"""
PDG code → semantic label mapping (fallback when no labl file is available).

Pure numpy: given a per-point PDG array, return an integer label array.
"""

import numpy as np


MOTIF_MAP = {
    22: 0, 11: 0, -11: 0,          # shower (photon/electron)
    13: 1, -13: 1,                  # track (muon)
    211: 1, -211: 1,                # track (pion)
    2212: 1,                         # track (proton)
    321: 1, -321: 1,                # track (kaon)
}

PID_MAP = {
    22: 0,                           # photon
    11: 1, -11: 1,                  # electron
    13: 2, -13: 2,                  # muon
    211: 3, -211: 3,                # pion
    2212: 4,                         # proton
}

_SCHEMES = {
    'motif_5cls': (MOTIF_MAP, 4),   # shower(0), track(1), michel(2), delta(3), led(4)
    'pid_6cls':   (PID_MAP, 5),     # photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
}


def pdg_to_semantic(pdg, scheme='motif_5cls', custom_map=None):
    """Map a per-point PDG array to integer class indices.

    Parameters
    ----------
    pdg : np.ndarray
        1D integer array of PDG codes (any shape; flattened if needed).
    scheme : str
        One of 'motif_5cls', 'pid_6cls', 'custom', 'none'.
    custom_map : dict or None
        Required when scheme='custom': ``{pdg_code: class_index}``.

    Returns
    -------
    labels : np.ndarray or None
        Integer class indices, same length as pdg. Returns None if
        scheme='none'.

    Notes
    -----
    For the motif_5cls scheme, classes michel(2), delta(3), led(4) are
    not assignable from PDG alone — they collapse to the default class
    (led=4). For pid_6cls, unmapped codes map to other(5).
    """
    if scheme == 'none':
        return None

    if scheme == 'custom':
        if custom_map is None:
            raise ValueError("scheme='custom' requires custom_map")
        mapping = custom_map
        default = max(custom_map.values()) + 1
    else:
        if scheme not in _SCHEMES:
            raise ValueError(f"Unknown label scheme: {scheme!r}. "
                             f"Valid: {list(_SCHEMES.keys()) + ['custom', 'none']}")
        mapping, default = _SCHEMES[scheme]

    pdg_arr = np.asarray(pdg).ravel()
    labels = np.full(pdg_arr.shape[0], default, dtype=np.int32)
    for code, cls in mapping.items():
        labels[pdg_arr == code] = cls
    return labels
