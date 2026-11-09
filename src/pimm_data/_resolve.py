"""Follow references between parts to per-row labels (``docs/LABELS.md``).

An event is a set of parts — point clouds (``step``, ``hits``) and tables
(``group``, ``particle``, ``interaction``) — linked by ``('ref', part)`` columns
declared in each part's ``_roles``. A label is a walk along those links:

    step.group -> group.fragment -> particle.fragment_instance -> particle.pdg

Labels follow pysupera's own convention (its ``torchdata``): a point's fragment
is pysupera's owner of its group (per hit, pysupera's ``hit_labels``), its
instance that fragment's instance, and its class the instance's semantic type,
or ``le`` for a point of a low-energy-scatter fragment (motif and PID alike).

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

    ``particle.motif``    point class of a fragment row (shower 0 … le 4): le
                          for an LE fragment, else its instance's semantic type
    ``particle.is_le``    the fragment row is a low-energy scatter
    ``particle.pid``      photon 0, electron 1, muon 2, pion 3, proton 4, le 5
                          (of an instance row; a point of an LE fragment is le)
    ``particle.is_primary`` the row is its own ``parent``
    ``particle.parent_pdg`` PDG of the ``parent`` row (pysupera's
                          ``parent_pdg``; equal on every row of 1M)
    ``particle.momentum_start`` / ``momentum_first``  ``|p|`` (MeV/c) at
                          creation / at the first visible step (NaN if none)
    ``interaction.vertex`` ``(I, 3)`` x, y, z (mm)
    """
    part = data.get('particle')
    if part is not None and 'motif' not in part:
        rows = np.arange(len(part['parent']))
        le = np.asarray(part['fragment_sem_type']) == SEM_LE
        part['is_le'] = le.astype(np.int32)
        # the instance's semantic type, read on the INSTANCE row: pysupera
        # sets instance_sem_type only on instance rows (-1 elsewhere), and a
        # shower fragment merged into a shower instance is not one
        fi = np.asarray(part['fragment_instance'])
        isem = np.asarray(part['instance_sem_type'])
        isem_of_frag = np.where(fi >= 0, isem[np.maximum(fi, 0)], -1) \
            if len(fi) else isem
        part['motif'] = np.where(le, MOTIF_NAMES.index('le'),
                                 motif_of_sem(isem_of_frag)).astype(np.int32)
        part['pid'] = pid_of(part['pdg'], part['instance_sem_type'])
        par = np.asarray(part['parent'])
        part['is_primary'] = (par == rows).astype(np.int32)
        part['parent_pdg'] = np.where(
            par >= 0, np.asarray(part['pdg'])[np.maximum(par, 0)], 0)
        for which in ('start', 'first'):
            mom = np.asarray(part[f'{which}_mom'], dtype=np.float64)[:, :3]
            part[f'momentum_{which}'] = np.sqrt((mom ** 2).sum(axis=1)) \
                .astype(np.float32)
    inter = data.get('interaction')
    if inter is not None and 'vertex' not in inter \
            and all(c in inter for c in ('x', 'y', 'z')):
        inter['vertex'] = np.stack(
            [inter['x'], inter['y'], inter['z']], axis=1).astype(np.float32)
    return data


_INST = ('_instance',)
_FRAG = ('_fragment',)

#: out key -> (path, fill). ``{momentum}`` is ``start`` or ``first``. The
#: names are the PILArNet keys pimm's heads and evaluators read; classes are
#: ``segment_*``, ids ``instance_*`` (rows of ``particle``, or of
#: ``interaction`` for ``instance_interaction``).
PRESETS = {
    'pilarnet': {
        'segment_motif': (_FRAG + ('motif',), -1),
        'segment_pid': (_INST + ('pid',), -1),
        'segment_pdg': (_INST + ('pdg',), -1),     # raw, for RemapSegment
        'instance_fragment': (_FRAG, -1),
        'instance_particle': (_INST, -1),
        'instance_interaction': (_INST + ('interaction',), -1),
        'is_primary': (_INST + ('is_primary',), -1),
        'momentum': (_INST + ('momentum_{momentum}',), -1.0),
        'vertex': (_INST + ('interaction', 'vertex'), -1.0),
    },
}

#: Panoptic levels, nested (a fragment is in one particle, a particle in one
#: interaction): ``level -> (instance key, class key, stuff class)``. A level
#: is a class per point and an id per object of a ``thing`` class; a point of
#: the stuff class is no object at that level (-1). Each level takes its stuff
#: from its OWN class, so "stuff <=> no instance" holds by construction: a
#: point of a low-energy-scatter fragment is le in motif and PID alike
#: (pysupera's convention), so with ``le_is_stuff`` it has no fragment and no
#: particle id; without it, it keeps its host's, as pysupera's ``include_le``
#: does. Interactions have no class; every point of one, low-energy or not,
#: belongs to it.
#:
#: ``'pysupera'`` is pysupera's own panoptic target (its ``torchdata``): the
#: instance's semantic type (le on the instance's LE fragments) with the
#: instance. pysupera is the reference, so this is the pairing to use; motif is
#: constant on every instance (0 mixed on 1M, 50 events per readout). Its stuff
#: mask is the particle level's: a point's PID is le exactly when its motif is
#: (no non-LE fragment sits in an LE instance), checked on real data by
#: ``tests/test_pysupera_golden.py``.
LEVELS = {
    'pysupera': ('instance_particle', 'segment_motif', MOTIF_NAMES.index('le')),
    'fragment': ('instance_fragment', 'segment_motif', MOTIF_NAMES.index('le')),
    'particle': ('instance_particle', 'segment_pid', PID_NAMES.index('le')),
    'interaction': ('instance_interaction', None, None),
}
# one stuff mask per instance key; 'pysupera' shares the particle level's
_STUFF = {inst: (cls, stuff) for name, (inst, cls, stuff) in LEVELS.items()
          if cls is not None and name != 'pysupera'}


def expand(labels, preset='pilarnet'):
    """Label keys for ``labels``: preset keys, or a level name for that
    level's (class, instance) pair — ``'pysupera'`` (pysupera's panoptic
    target) is ``('segment_motif', 'instance_particle')``, ``'particle'`` is
    ``('segment_pid', 'instance_particle')``. ``'all'`` is every preset key."""
    if labels == 'all':
        return tuple(PRESETS[preset])
    out = []
    for name in ((labels,) if isinstance(labels, str) else labels):
        if name in LEVELS:
            inst, cls, _ = LEVELS[name]
            out += [k for k in (cls, inst) if k is not None]
        elif name in PRESETS[preset]:
            out.append(name)
        else:
            raise ValueError(f"labels={labels!r}: {name!r} is neither a level "
                             f"{list(LEVELS)} nor a key {list(PRESETS[preset])}.")
    if not out:
        raise ValueError(f"labels={labels!r}: empty.")
    return tuple(dict.fromkeys(out))


def resolve(data, on, keys, preset='pilarnet', momentum='start',
            le_is_stuff=True):
    """Write the preset's ``keys`` onto part ``on`` as ``(N, 1)`` (vertex
    ``(N, 3)``). NaN in a table (no visible step) becomes the fill. With
    ``le_is_stuff``, an instance key of a :data:`LEVELS` level is -1 on the
    points of that level's stuff class. A no-op on an event without label
    tables."""
    if momentum not in ('start', 'first'):
        raise ValueError(f"momentum={momentum!r}: use 'start' or 'first'")
    if on not in data or 'particle' not in data or 'group' not in data[on]:
        return data
    add_vocabulary(data)
    sub = data[on]
    own = '_fragment' not in sub
    if own:
        point_refs(data, on)
    if '_fragment' not in sub:
        return data
    for out in keys:
        path, fill = PRESETS[preset][out]
        path = tuple(p.format(momentum=momentum) for p in path)
        if path[-1] not in data[_last_part(data, on, path)]:
            continue          # the table lacks the column (e.g. no vertex)
        v = _value(data, on, out, preset, momentum)
        if le_is_stuff and out in _STUFF:
            cls, stuff = _STUFF[out]
            v[_value(data, on, cls, preset, momentum).reshape(v.shape)
              == stuff] = -1
        sub[out] = v if v.ndim == 2 else v[:, None]
        if out.startswith('instance_'):
            # a categorical id: collate compacts it per event and makes it
            # unique across the batch, so merged or batched events never
            # share one; per event it is a table row until then
            sub.setdefault('_roles', {})[out] = ('label', out)
    if own:
        drop_point_refs(sub)
    return data


def _value(data, on, out, preset, momentum):
    path, fill = PRESETS[preset][out]
    path = tuple(p.format(momentum=momentum) for p in path)
    v = follow(data, on, path, fill=fill)
    if v.dtype.kind == 'f':
        return np.nan_to_num(v, nan=fill).astype(np.float32)
    v = v.astype(np.int32)
    if out == 'segment_pid':        # a point of an LE fragment is le
        v[follow(data, on, _FRAG + ('is_le',)).reshape(v.shape) == 1] = \
            PID_NAMES.index('le')
    return v


def point_refs(data, on, hit_fix=None):
    """Each point's fragment and instance (``_fragment``, ``_instance``,
    refs into ``particle``): its group's pysupera owner and that fragment's
    instance. ``hit_fix`` (hits only): ``[(start, n, plane corrections)]``
    from labl, which make them pysupera's per-hit labels: no label off the
    sensor, and each listed hit's own fragment and instance."""
    if on not in data or 'particle' not in data or 'group' not in data[on] \
            or 'fragment' not in data.get('group', {}):
        return
    sub = data[on]
    frag = follow(data, on, ('group', 'fragment')).astype(np.int32)
    inst = _take(data['particle']['fragment_instance'], frag, -1) \
        .astype(np.int32)
    for start, n, c in hit_fix or ():
        if 'on_sensor' in c:
            off = start + np.flatnonzero(~c['on_sensor'])
            frag[off] = -1
            inst[off] = -1
        if 'override_index' in c:
            i = start + np.asarray(c['override_index'], np.int64)
            frag[i] = c['override_fragment']
            inst[i] = c['override_instance']
    sub['_fragment'], sub['_instance'] = frag, inst
    sub.setdefault('_roles', {}).update(_fragment=('ref', 'particle'),
                                        _instance=('ref', 'particle'))


def drop_point_refs(sub):
    for k in ('_fragment', '_instance'):
        sub.pop(k, None)
        (sub.get('_roles') or {}).pop(k, None)


def _last_part(data, on, path):
    target = on
    for name in path[:-1]:
        target = _ref_target(data[target], name)
    return target
