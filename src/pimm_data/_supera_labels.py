"""Per-point labels from a ``supera-labl`` event (see ``docs/LABELS.md``).

A point (step deposit or hit) reaches its labels through one key: the
``particle`` row of the fragment owning its JAXTPC group (``-1`` when no
fragment owns it — the group left no hit the readout recorded). Every label is
a gather from there:

=======================  ====================================================
key                      from
=======================  ====================================================
``segment_motif``        the fragment's ``fragment_sem_type``, as
                         shower 0, track 1, michel 2, delta 3, le 4
``instance_fragment``    the fragment row
``instance_particle``    its ``fragment_instance`` row
``segment_pid``          the instance's ``pdg`` as photon 0, electron 1,
                         muon 2, pion 3, proton 4; 5 (le) when the instance is
                         a low-energy scatter; -1 for any other species
``instance_interaction`` the instance's ``interaction`` row
``is_primary``           the instance is its own ``parent``
``momentum``             ``|p|`` of the instance (MeV/c): at creation
                         (``start``, PILArNet's meaning) or at its first
                         visible step (``first``)
``vertex``               its interaction's ``x, y, z`` (mm)
=======================  ====================================================

Ids are per-event row positions (compact by construction). Void is -1 for
every key, floats included, as PILArNet does; NaN stays in the tables, where an
instance with no visible points has no ``first_*``.
"""

import numpy as np

# pysupera SemanticType (1 shower, 2 track, 3 delta, 4 michel, 5 le_scatter,
# 6 unknown) -> motif class (shower 0, track 1, michel 2, delta 3, le 4).
_MOTIF_OF_SEM = np.array([-1, 0, 1, 3, 2, 4, -1], dtype=np.int32)
SEM_LE = 5
MOTIF_NAMES = ('shower', 'track', 'michel', 'delta', 'le')
PID_NAMES = ('photon', 'electron', 'muon', 'pion', 'proton', 'le')
_PID_OF_PDG = {22: 0, 11: 1, 13: 2, 211: 3, 2212: 4}


def motif_of_sem(sem):
    sem = np.asarray(sem, dtype=np.int64)
    ok = (sem >= 0) & (sem < len(_MOTIF_OF_SEM))
    return np.where(ok, _MOTIF_OF_SEM[np.clip(sem, 0, len(_MOTIF_OF_SEM) - 1)],
                    -1).astype(np.int32)


def pid_of(pdg, inst_sem):
    """PID class per particle row (sign-blind PDG; LE instances are 5)."""
    apdg = np.abs(np.asarray(pdg, dtype=np.int64))
    out = np.full(apdg.shape, -1, dtype=np.int32)
    for code, cls in _PID_OF_PDG.items():
        out[apdg == code] = cls
    out[np.asarray(inst_sem) == SEM_LE] = 5
    return out


def _take(col, rows, fill):
    """``col[rows]`` with ``fill`` where ``rows == -1``."""
    rows = np.asarray(rows, dtype=np.int64)
    col = np.asarray(col)
    shape = (len(rows),) + col.shape[1:]
    out = np.full(shape, fill, dtype=np.result_type(col.dtype, type(fill)))
    ok = rows >= 0
    if ok.any():
        out[ok] = col[rows[ok]]
    return out


def fragment_rows(group_fragment, gid):
    """Particle row of each point's group; -1 outside the table."""
    gid = np.asarray(gid, dtype=np.int64)
    ok = (gid >= 0) & (gid < len(group_fragment))
    out = np.full(gid.shape, -1, dtype=np.int32)
    out[ok] = group_fragment[gid[ok]]
    return out


def point_labels(frag, labl, momentum='start'):
    """Named per-point keys from fragment rows ``frag`` (N,)."""
    if momentum not in ('start', 'first'):
        raise ValueError(f"momentum={momentum!r}: use 'start' or 'first'")
    part = labl['particle']
    inter = labl.get('interaction', {})
    frag = np.asarray(frag, dtype=np.int64)
    inst = _take(part['fragment_instance'], frag, -1)

    out = {
        'segment_motif': motif_of_sem(_take(part['fragment_sem_type'], frag, -1)),
        'instance_fragment': frag.astype(np.int32),
        'instance_particle': inst.astype(np.int32),
        'segment_pid': _take(pid_of(part['pdg'], part['instance_sem_type']),
                             inst, -1).astype(np.int32),
        'instance_interaction': _take(part['interaction'], inst,
                                      -1).astype(np.int32),
    }
    rows = np.arange(len(part['parent']))
    out['is_primary'] = _take((part['parent'] == rows).astype(np.int32),
                              inst, -1).astype(np.int32)
    mom = part[f'{momentum}_mom'][:, :3]
    pmag = np.sqrt((mom.astype(np.float64) ** 2).sum(axis=1))
    out['momentum'] = np.nan_to_num(_take(pmag, inst, -1.0),
                                    nan=-1.0).astype(np.float32)
    if all(c in inter for c in ('x', 'y', 'z')):
        vtx = np.stack([inter['x'], inter['y'], inter['z']], axis=1)
        out['vertex'] = _take(vtx, out['instance_interaction'],
                              -1.0).astype(np.float32)
    # (N,1) like every named key; vertex is (N,3)
    return {k: (v if v.ndim == 2 else v[:, None]) for k, v in out.items()}


def pdg_of_points(frag, labl):
    """Raw PDG of each point's instance (label_key='pdg' back-compat)."""
    part = labl['particle']
    inst = _take(part['fragment_instance'], frag, -1)
    return _take(part['pdg'], inst, -1).astype(np.int32)
