"""Follow references between parts to per-row labels (``docs/LABELS.md``).

An event is a set of parts — point clouds (``step``, ``hits``) and tables
(``group``, ``particle``, ``interaction``) — linked by ``('ref', part)`` columns
declared in each part's ``_roles``. A label is a walk along those links:

    step.group -> group.fragment -> particle.fragment_instance -> particle.pdg

:func:`follow` performs one such walk. It reads the target of every hop from
the roles, so it knows nothing about JAXTPC or pysupera; a detector only has to
name its parts and declare its references. Walks work per event (numpy) and
stay valid after a point subsample, because the reference columns are sliced
with their points and the tables are never cropped.

The class vocabularies the evaluators expect are a property of the tables, not
of any point cloud, so :func:`add_vocabulary` derives them once per event as
extra table columns (``particle.motif``, ``particle.pid``, …). :data:`PRESETS`
then names the walks that produce the PILArNet keys every pimm evaluator reads.
"""

import numpy as np

# pysupera SemanticType (1 shower, 2 track, 3 delta, 4 michel, 5 le_scatter,
# 6 unknown) -> motif class (shower 0, track 1, michel 2, delta 3, le 4).
_MOTIF_OF_SEM = np.array([-1, 0, 1, 3, 2, 4, -1], dtype=np.int32)
SEM_LE = 5
MOTIF_NAMES = ('shower', 'track', 'michel', 'delta', 'le')
PID_NAMES = ('photon', 'electron', 'muon', 'pion', 'proton', 'le')
_PID_OF_PDG = {22: 0, 11: 1, 13: 2, 211: 3, 2212: 4}


def _ref_target(part, col):
    r = (part.get('_roles') or {}).get(col)
    if isinstance(r, (tuple, list)) and len(r) == 2 and r[0] == 'ref':
        return r[1]
    return None


def _take(col, rows, fill):
    """``col[rows]``; ``fill`` where ``rows`` is -1 (or out of range)."""
    rows = np.asarray(rows, dtype=np.int64)
    col = np.asarray(col)
    out = np.full((len(rows),) + col.shape[1:], fill,
                  dtype=np.result_type(col.dtype, type(fill)))
    ok = (rows >= 0) & (rows < len(col))
    if ok.any():
        out[ok] = col[rows[ok]]
    return out


def follow(data, on, path, fill=-1):
    """Walk ``path`` from part ``on``; one value per row of ``on``.

    ``path[0]`` is a column of ``on``; each later name is a column of the part
    the previous column references. Every hop but the last must be a
    ``('ref', part)`` column. A row whose chain hits -1 gets ``fill``.
    """
    part = data[on]
    vals = np.asarray(part[path[0]])
    target = _ref_target(part, path[0])
    for name in path[1:]:
        if target is None:
            raise KeyError(f"{on}: {path!r} continues past a column that is "
                           f"not a ('ref', part) column")
        tbl = data[target]
        vals = _take(tbl[name], vals, fill)
        target = _ref_target(tbl, name)
    return vals


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


def add_vocabulary(data):
    """Derived table columns, once per event (idempotent):

    ``particle.motif``    fragment class (shower 0 … le 4) of a fragment row
    ``particle.pid``      photon 0, electron 1, muon 2, pion 3, proton 4, le 5
    ``particle.is_primary`` the row is its own ``parent``
    ``particle.momentum_start`` / ``momentum_first``  ``|p|`` (MeV/c) at
                          creation (its Geant4 track's) / at the first visible
                          step (NaN if none)
    ``interaction.vertex`` ``(I, 3)`` x, y, z (mm)
    """
    part = data.get('particle')
    if part is not None and 'motif' not in part:
        rows = np.arange(len(part['parent']))
        part['motif'] = motif_of_sem(part['fragment_sem_type'])
        part['pid'] = pid_of(part['pdg'], part['instance_sem_type'])
        part['is_primary'] = (np.asarray(part['parent']) == rows).astype(np.int32)
        # creation momentum is the Geant4 track's; first visible step is
        # pysupera's own (NaN for an instance with no visible points)
        trk = data.get('track')
        if trk is not None and 'track' in part:
            start = _take(np.asarray(trk['start_mom'], np.float64),
                          part['track'], np.nan)
        else:
            start = np.full((len(rows), 3), np.nan)
        for which, mom in (('start', start),
                           ('first', np.asarray(part['first_mom'],
                                                np.float64)[:, :3])):
            part[f'momentum_{which}'] = np.sqrt((mom[:, :3] ** 2).sum(axis=1)) \
                .astype(np.float32)
    inter = data.get('interaction')
    if inter is not None and 'vertex' not in inter \
            and all(c in inter for c in ('x', 'y', 'z')):
        inter['vertex'] = np.stack(
            [inter['x'], inter['y'], inter['z']], axis=1).astype(np.float32)
    return data


_INST = ('group', 'fragment', 'fragment_instance')

#: out key -> (path, fill). ``{momentum}`` is ``start`` or ``first``.
PRESETS = {
    'pilarnet': {
        'instance_fragment': (('group', 'fragment'), -1),
        'segment_motif': (('group', 'fragment', 'motif'), -1),
        'instance_particle': (_INST, -1),
        'segment_pid': (_INST + ('pid',), -1),
        'instance_interaction': (_INST + ('interaction',), -1),
        'is_primary': (_INST + ('is_primary',), -1),
        'momentum': (_INST + ('momentum_{momentum}',), -1.0),
        'vertex': (_INST + ('interaction', 'vertex'), -1.0),
    },
}

# label_key -> the preset key the bare ``segment`` copies; None = raw PDG.
SEGMENT_OF_LABEL_KEY = {'pdg': None, 'motif': 'segment_motif',
                        'pid': 'segment_pid', 'particle': 'instance_particle',
                        'interaction': 'instance_interaction'}


def resolve(data, on, preset='pilarnet', momentum='start', keys=None):
    """Write the preset's keys onto part ``on`` as ``(N, 1)`` (vertex
    ``(N, 3)``). NaN in a table (no visible step) becomes the fill."""
    if momentum not in ('start', 'first'):
        raise ValueError(f"momentum={momentum!r}: use 'start' or 'first'")
    add_vocabulary(data)
    sub = data[on]
    for out, (path, fill) in PRESETS[preset].items():
        if keys is not None and out not in keys:
            continue
        path = tuple(p.format(momentum=momentum) for p in path)
        if path[-1] not in data[_last_part(data, on, path)]:
            continue          # the table lacks the column (e.g. no vertex)
        v = follow(data, on, path, fill=fill)
        if v.dtype.kind == 'f':
            v = np.nan_to_num(v, nan=fill).astype(np.float32)
        else:
            v = v.astype(np.int32)
        sub[out] = v if v.ndim == 2 else v[:, None]
    return data


def _last_part(data, on, path):
    target = on
    for name in path[:-1]:
        target = _ref_target(data[target], name)
    return target
