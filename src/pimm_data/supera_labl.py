"""Write a JAXTPC ``labl`` file from one pysupera (format 4.x) output.

The label of a deposit or a hit is pysupera's: the fragment pysupera says owns
its JAXTPC *group* (``groups/fragment_id``), and per hit pysupera's
``hit_labels``. pysupera is the reference; nothing here relabels. ``labl``
stores, per event:

    group -> fragment -> instance -> interaction,   group -> track

* per volume, ``group_fragment``: ``groups/fragment_id`` as particle rows (-1:
  pysupera left the group unowned), and JAXTPC's ``group_to_track``: the one
  link from a group to its Geant4 track (step has no per-deposit track id); and,
  only when the step file does not carry it (files produced before the
  shared-step layout), a copy of ``deposit_to_group``, so step+labl needs no
  hits;
* per volume and hits plane, only where ``hit_labels`` are not the group's:
  ``on_sensor`` (pixel; packed bits) and ``override_index`` /
  ``override_fragment`` / ``override_instance`` (hits of groups pysupera split);
* ``track/id``: the visible Geant4 tracks (owning a group), sorted;
* the ``particle`` table (fragment -> instance -> interaction, and the rest of
  each row; pysupera's float64 mass and kinematics stored as float32) and the
  ``interaction`` table;
* given the step and sensor files, ``deposit_in_window`` per volume: whether
  the deposit's charge arrives inside the readout window. An unowned group is
  either out of the window or under threshold, and only this tells them apart.
  It is the one computed array, and goes away once JAXTPC stops saving
  out-of-window deposits.

Everything is copied; the one transformation is that pysupera ids (``id``,
``frag_id``, ``parent_id``, …, which are *not* row positions) become row
positions within the event. What is left out, and why, is in
``docs/LABELS.md``: voxels and point ranges, ``hit_labels`` and
``groups/is_le`` (both derivable from the group map), ``parent_pdg``
(derivable from ``parent``) and pysupera's own bookkeeping.

The file mirrors the JAXTPC file it labels: the same ``entry_NNNNNN`` groups in
the same order, each carrying ``event_id`` / ``job_id`` / ``job_event``, and the
same ``/dprod/events`` table. Name it ``{dataset}_labl_{task}.h5`` so it pairs
with ``{dataset}_{step,hits,sensor}_{task}.h5`` by the trailing task token.

Import-light on purpose (numpy, h5py, hdf5plugin only), so a production stage
can run it without the rest of pimm_data::

    python -m pimm_data.supera_labl SUPERA.h5 HITS.h5 OUT.h5 \
        [--step STEP.h5 --sensor SENSOR.h5]

or, over a whole doraemon campaign (every pysupera file, in parallel)::

    python -m pimm_data.supera_labl campaign DATA_ROOT OUT_ROOT \
        [--readouts wire pixel] [--workers N] [--shard I/N]
"""

import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

LABL_FORMAT = 'supera-labl/1'
SUPPORTED_SUPERA = ('4.',)

# (labl column, pysupera column). Id-valued: converted to row positions.
ID_COLUMNS = (
    ('fragment', 'frag_id'),
    ('instance', 'inst_id'),
    ('fragment_instance', 'frag_inst_id'),
    ('fragment_parent', 'frag_parent_id'),
    ('instance_parent', 'inst_parent_id'),
    ('parent', 'parent_id'),
    ('ancestor', 'ancestor_id'),
)
# Copied unchanged.
VALUE_COLUMNS = (
    ('interaction', 'interaction_id'),     # already a row index into inter/
    ('pdg', 'pdg'),
    ('sem_type', 'sem_type'),
    ('fragment_sem_type', 'frag_sem_type'),
    ('instance_sem_type', 'inst_sem_type'),
    ('interaction_type', 'interaction_type'),
    ('geant4_trackid', 'geant4_trackid'),
    ('geant4_parent_trackid', 'geant4_parent_trackid'),
    ('geant4_parent_pdg', 'geant4_parent_pdg'),
    ('fragment_merge_count', 'frag_merge_count'),  # pysupera particles in it
    ('instance_merge_count', 'inst_merge_count'),  # (split pieces and buried
                                                   # deltas count; -1: not that
                                                   # level). Not derivable.
    ('mass', 'mass'),
    ('start_pos', 'start_pos'), ('start_mom', 'start_mom'),
    ('first_pos', 'first_pos'), ('first_mom', 'first_mom'),
    ('last_pos', 'last_pos'), ('last_mom', 'last_mom'),
    ('end_pos', 'end_pos'), ('end_mom', 'end_mom'),
)
INTERACTION_COLUMNS = ('x', 'y', 'z', 'time', 'vertex_id',
                       'energy_sum', 'ke_sum')
# Copied when present: an empty column is not worth a dataset, but the
# generator source belongs here as soon as the generator records it.
OPTIONAL_INTERACTION_COLUMNS = ('generator', 'reaction')

SEM_TYPES = {1: 'shower', 2: 'track', 3: 'delta', 4: 'michel',
             5: 'le_scatter', 6: 'unknown'}

_SMALL = 4096   # bytes: below this compression costs more than it saves


def _create(g, name, data, comp):
    data = np.ascontiguousarray(data)
    kw = comp if data.nbytes > _SMALL else {}
    g.create_dataset(name, data=data, **kw)


def _to_rows(ids, row_of_id, what):
    """pysupera ids -> row positions (-1 stays -1). Raises on a dangling id,
    which would mean the input broke pysupera's own guarantee."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.full(ids.shape, -1, dtype=np.int32)
    valid = ids >= 0
    if valid.any():
        sorter, sorted_ids = row_of_id
        pos = np.searchsorted(sorted_ids, ids[valid])
        pos = np.minimum(pos, len(sorted_ids) - 1)
        if not np.array_equal(sorted_ids[pos], ids[valid]):
            raise ValueError(f'{what}: id with no particle row')
        out[valid] = sorter[pos]
    return out


def _decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def _window(sensor_path):
    """Readout window from the JAXTPC sensor config: per volume the anode x
    (the x bound away from the shared central cathode), plus drift velocity
    (mm/us), tick (us), ticks, and the pre-window (us)."""
    with h5py.File(sensor_path, 'r') as f:
        c = f['config']
        ranges = np.asarray(c['volume_ranges'][:], np.float64)
        a = c.attrs
        centre = 0.5 * (ranges[:, 0, :].min() + ranges[:, 0, :].max())
        anode = [r[0][np.argmax(np.abs(r[0] - centre))] for r in ranges]
        return dict(anode=anode, v=float(a['velocity_cm_us']) * 10.0,
                    dt=float(a['time_step_us']), n=int(a['num_time_steps']),
                    pre=float(a.get('pre_window_us', 0.0)))


def _in_window(sv, anode, w):
    """Per deposit: its arrival tick ``(t0 + drift / v + pre) / dt`` lies in
    ``[0, n)``. Checked on 1M task 0: hit times sit a constant offset from
    this drift time (wire about -4 us, pixel about -0.25 us), and groups with
    hits but no in-window deposit are a few per 10k (boundary straddlers)."""
    q = sv['positions'][:]
    if q.dtype.kind in 'ui':
        x = sv.attrs['pos_origin_x'] + q[:, 0].astype(np.float64) \
            * sv.attrs['pos_step_mm']
    else:
        x = q[:, 0].astype(np.float64)
    t = sv['t0_us'][:].astype(np.float64) + np.abs(x - anode) / w['v']
    tick = np.floor((t + w['pre']) / w['dt'])
    return (tick >= 0) & (tick < w['n'])


def _hit_label_planes(su):
    """pysupera ``hit_labels`` plane groups by (volume, plane name)."""
    out = {}
    for vk, vg in (su['hit_labels'].items() if 'hit_labels' in su else ()):
        for pk, pg in vg.items():
            out[(int(pg.attrs['volume']), _decode([pg.attrs['source']])[0])] = pg
    return out


def _hit_overrides(ov, hv, HL, v, e, gv, finst, row_of_id, key, comp):
    """Per hits plane of volume ``v``: where pysupera's per-hit labels differ
    from the group's (fragment ``gv[group]``, instance ``finst[fragment]``).

    ``plane_<name>/on_sensor`` (pixel; packed bits, attr ``n``): pysupera labels
    only hits on a sensor pixel; the others get no label. ``plane_<name>/
    override_index / override_fragment / override_instance``: the on-sensor
    hits (index within the plane, in the hits file's CSR order) whose pysupera
    fragment or instance differs from their group's (groups whose hits pysupera
    split). With both, per-hit labels equal pysupera's ``hit_labels``."""
    for pk, pg in hv.items():
        if not isinstance(pg, h5py.Group) or 'group_ids' not in pg:
            continue
        hl = HL.get((v, pk))
        if hl is None:
            raise ValueError(f'{key}/volume_{v}/{pk}: no pysupera hit_labels')
        sizes = pg['group_sizes'][:].astype(np.int64)
        gid = np.repeat(pg['group_ids'][:], sizes)
        a, b = int(hl['offsets'][e]), int(hl['offsets'][e + 1])
        if b - a != len(gid):
            raise ValueError(f'{key}/volume_{v}/{pk}: {b - a} pysupera hit '
                             f'labels, {len(gid)} hits')
        hf = _to_rows(hl['fragment_id'][a:b], row_of_id, f'{key} hit fragment')
        hi = _to_rows(hl['instance_id'][a:b], row_of_id, f'{key} hit instance')
        fd = np.where(gid >= 0, gv[np.maximum(gid, 0)], -1) if len(gv) \
            else np.full(len(gid), -1, np.int32)
        idd = np.where(fd >= 0, finst[np.maximum(fd, 0)], -1)
        on = hl['on_sensor'][a:b].astype(bool) if 'on_sensor' in hl \
            else np.ones(len(gid), bool)
        fd = np.where(on, fd, -1)
        idd = np.where(on, idd, -1)
        bad = np.flatnonzero((hf != fd) | (hi != idd))
        if 'on_sensor' not in hl and not len(bad):
            continue
        og = ov.create_group(f'plane_{pk}')
        if 'on_sensor' in hl:
            d = og.create_dataset('on_sensor', data=np.packbits(on), **comp)
            d.attrs['n'] = len(on)
        if len(bad):
            _create(og, 'override_index', bad.astype(np.int32), comp)
            _create(og, 'override_fragment', hf[bad], comp)
            _create(og, 'override_instance', hi[bad], comp)


class _Block:
    """A pysupera CSR table read ``size`` events at a time: one h5py read
    per column per block, not per event (per-call overhead dominated)."""

    def __init__(self, grp, cols, off, size):
        self.grp, self.cols, self.off, self.size = grp, cols, off, size
        self.lo = self.hi = -1

    def __call__(self, e):
        if not self.lo <= e < self.hi:
            self.lo = e
            self.hi = min(e + self.size, len(self.off) - 1)
            a, b = int(self.off[self.lo]), int(self.off[self.hi])
            self.base = a
            self.data = {c: self.grp[c][a:b] for c in self.cols}
        a = int(self.off[e]) - self.base
        b = int(self.off[e + 1]) - self.base
        return {c: v[a:b] for c, v in self.data.items()}


def write_labl(supera_path, hits_path, out_path, compression='lz4',
               max_events=None, step_path=None, sensor_path=None,
               block=100, track_path=None):
    """Extract ``supera_path`` into ``out_path``, aligned to ``hits_path``.

    ``hits_path`` is the JAXTPC hits file pysupera read (one file per task
    with merged inputs); it is the alignment check: every entry's
    ``event_id`` and group count must agree. The JAXTPC groups
    (``deposit_to_group``, ``group_to_track``) come from ``step_path`` when
    that step carries them (shared-step layout; labl then holds no
    ``deposit_to_group``, step does), else from the hits file.
    ``group_to_track`` is label truth, carried by no point file in the
    shared-step layout: it comes from ``track_path`` (any JAXTPC file of the
    task with ``entry/volume/group_to_track``, e.g. the produced hits), else
    from step or hits.
    ``step_path`` and ``sensor_path`` (the same task's JAXTPC step and sensor
    files) add ``deposit_in_window``; give both or neither. ``max_events``
    truncates (for testing); the result is then partial. pysupera is read
    ``block`` events at a time."""
    if (step_path is None) != (sensor_path is None):
        raise ValueError('give step_path and sensor_path together')
    win = _window(sensor_path) if sensor_path is not None else None
    st = h5py.File(step_path, 'r') if step_path is not None else None
    tr = h5py.File(track_path, 'r') if track_path is not None else None
    if compression == 'gzip':
        comp = {'compression': 'gzip', 'compression_opts': 4}
    else:
        import hdf5plugin       # only the lz4 filter needs the plugin
        comp = hdf5plugin.Blosc(cname='lz4', clevel=5,
                                shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(supera_path, 'r') as su, h5py.File(hits_path, 'r') as hi, \
            h5py.File(out_path + '.tmp', 'w') as out:
        fmt = su['format_version'][()]
        fmt = fmt.decode() if isinstance(fmt, bytes) else str(fmt)
        if not fmt.startswith(SUPPORTED_SUPERA):
            raise ValueError(f'{supera_path}: pysupera format {fmt}; '
                             f'need {SUPPORTED_SUPERA}')
        n_events = int(su['n_events'][()])
        if max_events is not None:
            n_events = min(n_events, int(max_events))
        ev = su['events']
        entries = ev['jaxtpc_entry'][:]
        event_ids = ev['event_id'][:]
        id_cols = {c: ev[c][:] for c in ('job_id', 'job_event') if c in ev}
        p_off = ev['offsets'][:]
        g_off = su['groups/offsets'][:]
        v_off = su['groups/volume_offsets'][:]
        v_idx = su['groups/volume_offsets_index'][:]
        i_off = su['inter/offsets'][:]
        I = su['inter']
        icols = list(INTERACTION_COLUMNS) + [
            c for c in OPTIONAL_INTERACTION_COLUMNS
            if c in I and any(_decode(I[c][:min(len(I[c]), 4096)]))]
        P = _Block(su['particles'], ['id'] + sorted(
            {src for _, src in ID_COLUMNS + VALUE_COLUMNS}), p_off, block)
        G = _Block(su['groups'], ['fragment_id'], g_off, block)
        HL = _hit_label_planes(su)     # (volume, plane name) -> pysupera group
        I = _Block(I, icols, i_off, block)

        cfg = out.create_group('config')
        cfg.attrs['labl_format'] = LABL_FORMAT
        cfg.attrs['supera_format_version'] = fmt
        cfg.attrs['readout_type'] = str(su.attrs.get(
            'readout_type', hi['config'].attrs.get('readout_type', '')))
        cfg.attrs['n_events'] = n_events
        cfg.attrs['sem_type_names'] = np.array(
            [f'{k}:{v}' for k, v in SEM_TYPES.items()],
            dtype=h5py.string_dtype())
        for k in ('dataset_name', 'n_volumes', 'production_version'):
            if k in hi['config'].attrs:
                cfg.attrs[k] = hi['config'].attrs[k]
        cfg.create_dataset('source_files', data=np.array(
            [os.path.abspath(supera_path), os.path.abspath(hits_path)],
            dtype=h5py.string_dtype()))
        if 'issues' in su:
            for k, v in su['issues'].attrs.items():
                cfg.attrs[f'supera_issues_{k}'] = v
        # how pysupera labelled: its run attrs (kinematics_steps defines first/
        # last; pixel: hit_energy, pixel_geometry) and its effective settings
        # (config.yaml plus the readout overlay; they differ by readout)
        for k, v in su.attrs.items():
            cfg.attrs[f'supera_{k}'] = v
        _copy_provenance(su, cfg)
        issues = _issues_by_event(su)

        dp = out.create_group('dprod/events')
        dp.attrs['order'] = 'sorted'
        dp.create_dataset('event_id',
                          data=event_ids[:n_events].astype(np.int64))
        for c, v in id_cols.items():
            dp.create_dataset(c, data=v[:n_events].astype(np.int64))

        for e in range(n_events):
            key = f'entry_{int(entries[e]):06d}'
            if key not in hi:
                raise ValueError(f'{key} (event_id {event_ids[e]}) is not in '
                                 f'{hits_path}')
            hev = hi[key]
            if int(hev.attrs['event_id']) != int(event_ids[e]):
                raise ValueError(
                    f'{key}: event_id {hev.attrs["event_id"]} in hits, '
                    f'{event_ids[e]} in pysupera')
            oev = out.create_group(key)
            oev.attrs['event_id'] = int(event_ids[e])
            if e in issues:     # e.g. OUTSIDE_READOUT_WINDOW: written empty
                codes, msgs = zip(*issues[e])
                oev.attrs['supera_issue_codes'] = np.array(codes, np.int32)
                oev.attrs['supera_issue_messages'] = np.array(
                    msgs, dtype=h5py.string_dtype())
            for c, v in id_cols.items():
                oev.attrs[c] = int(v[e])

            pe = P(e)
            ids = pe['id'].astype(np.int64)
            sorter = np.argsort(ids, kind='stable').astype(np.int32)
            row_of_id = (sorter, ids[sorter])

            n_vol = sum(1 for k in hev if k.startswith('volume_'))
            finst = _to_rows(pe['frag_inst_id'], row_of_id,
                             f'{key} frag_inst_id')
            # Where the groups come from: a step that carries them (the
            # shared-step layout: deposit_to_group, group_to_track, n_groups
            # are JAXTPC step output), else the hits file (as produced so far).
            grp, from_step, g2t_of = {}, {}, {}
            for v in range(n_vol):
                sv = st[f'{key}/volume_{v}'] if (
                    st is not None and f'{key}/volume_{v}' in st) else None
                from_step[v] = sv is not None and 'n_groups' in sv.attrs
                grp[v] = sv if from_step[v] else hev[f'volume_{v}']
                srcs = ([tr[f'{key}/volume_{v}']] if tr is not None else []) \
                    + [grp[v], hev[f'volume_{v}']]
                g2t_src = next((x for x in srcs if 'group_to_track' in x),
                               None)
                if g2t_src is None:
                    raise ValueError(
                        f'{key}/volume_{v}: no group_to_track (shared-step '
                        'files carry none: pass track_path)')
                if tr is not None and int(tr[key].attrs['event_id']) != \
                        int(event_ids[e]):
                    raise ValueError(f'{key}: event_id differs in track_path')
                g2t_of[v] = g2t_src['group_to_track'][:]
                if 'n_groups' in grp[v].attrs and \
                        int(grp[v].attrs['n_groups']) != len(g2t_of[v]):
                    raise ValueError(
                        f'{key}/volume_{v}: n_groups '
                        f'{int(grp[v].attrs["n_groups"])}, group_to_track has '
                        f'{len(g2t_of[v])}')
                hn = hev[f'volume_{v}'].attrs.get('n_groups')
                if from_step[v] and hn is not None and \
                        int(hn) != int(sv.attrs['n_groups']):
                    raise ValueError(f'{key}/volume_{v}: n_groups {int(hn)} '
                                     f'in hits, {int(sv.attrs["n_groups"])} '
                                     'in step')
            gfrag = _to_rows(G(e)['fragment_id'], row_of_id,
                             f'{key} groups/fragment_id')
            voffs = v_off[int(v_idx[e]):int(v_idx[e + 1])]
            # A volume with no deposits: JAXTPC keeps it (padding group only,
            # no deposit_to_group); pysupera may drop it (wire) or keep it
            # (pixel). Map pysupera's volume blocks onto the hits volumes.
            vols = list(range(n_vol))
            if len(voffs) != n_vol:
                vols = [v for v in vols if 'deposit_to_group' in grp[v]]
                if len(voffs) != len(vols):
                    raise ValueError(f'{key}: {len(voffs)} pysupera volumes, '
                                     f'{len(vols)} non-empty in hits')
            bounds = dict(zip(vols, zip(
                voffs, list(voffs[1:]) + [len(gfrag)])))
            own_t = []
            for v in range(n_vol):
                hv = grp[v]
                n_groups = len(g2t_of[v])               # == attrs n_groups
                if v in bounds:
                    gv = gfrag[int(bounds[v][0]):int(bounds[v][1])]
                else:                                    # dropped: empty
                    gv = np.full(n_groups, -1, np.int32)
                if len(gv) != n_groups:
                    raise ValueError(
                        f'{key}/volume_{v}: {len(gv)} groups in pysupera, '
                        f'{n_groups} in JAXTPC')
                g2t = g2t_of[v]
                own_t.append(g2t[gv >= 0])
                ov = oev.create_group(f'volume_{v}')
                # pysupera's owner of each group, as a particle row (-1:
                # unowned). Taken as is: pysupera is the reference.
                _create(ov, 'group_fragment', gv.astype(np.int32), comp)
                _hit_overrides(ov, hev[f'volume_{v}'], HL, v, e, gv, finst,
                               row_of_id, key, comp)
                d2g = (hv['deposit_to_group'][:].astype(np.int32)
                       if 'deposit_to_group' in hv
                       else np.zeros(0, np.int32))
                if not from_step[v]:    # step has it: the loader reads it there
                    _create(ov, 'deposit_to_group', d2g, comp)
                _create(ov, 'group_to_track',
                        g2t.astype(np.int32), comp)
                if st is not None:
                    sv = st[f'{key}/volume_{v}']
                    if int(st[key].attrs['event_id']) != int(event_ids[e]):
                        raise ValueError(f'{key}: event_id differs in step')
                    n_dep = len(sv['t0_us']) if 't0_us' in sv else 0
                    if n_dep != len(d2g):
                        raise ValueError(f'{key}/volume_{v}: {n_dep} step '
                                         f'deposits, {len(d2g)} in '
                                         f'deposit_to_group')
                    _create(ov, 'deposit_in_window',
                            _in_window(sv, win['anode'][v], win) if n_dep
                            else np.zeros(0, bool), comp)

            # the visible Geant4 tracks (those with an owned group), sorted:
            # the row space group -> track points into
            tid = np.unique(np.concatenate(own_t).astype(np.int64)) \
                if own_t else np.zeros(0, np.int64)
            ot = oev.create_group('track')
            _create(ot, 'id', tid.astype(np.int32), comp)

            op = oev.create_group('particle')
            for name, src in ID_COLUMNS:
                _create(op, name, _to_rows(pe[src], row_of_id,
                                           f'{key} {src}'), comp)
            for name, src in VALUE_COLUMNS:
                col = pe[src]
                if col.dtype == np.float64:     # mass, kinematics: float32
                    col = col.astype(np.float32)  # is ample, half the bytes
                _create(op, name, col, comp)

            oi = oev.create_group('interaction')
            ie = I(e)
            for c in icols:
                col = ie[c]
                if col.dtype.kind in 'OS':
                    col = np.array(_decode(col), dtype=h5py.string_dtype())
                _create(oi, c, col, comp)
    for h in (st, tr):
        if h is not None:
            h.close()
    os.replace(out_path + '.tmp', out_path)
    return out_path


def _issues_by_event(su):
    """pysupera's per-event issues: {event row: [(code, message), ...]}."""
    if 'issues' not in su or 'event' not in su['issues']:
        return {}
    I = su['issues']
    out = {}
    for ev, code, msg in zip(I['event'][:], I['code'][:],
                             _decode(I['message'][:]) if 'message' in I
                             else [''] * len(I['event'])):
        out.setdefault(int(ev), []).append((int(code), str(msg)))
    return out


def _copy_provenance(su, cfg):
    """The pysupera commit, config hashes and the effective settings, as
    flat attrs (whatever of the dprod provenance tree the input has). The
    settings are the task's ``config/pysupera_*`` groups (config.yaml, and the
    reader and readout overlays that override it)."""
    if 'provenance' not in su:
        return

    def visit(name, obj):
        top = name.split('/')[0]
        for k, v in obj.attrs.items():
            if k == '_json':
                continue
            if ('pysupera' in name or 'supera' in top) and (
                    'commit' in k or 'dirty' in k or 'sha256' in k):
                cfg.attrs[f'provenance/{name}/{k}'] = v
            elif 'supera' in top and '/config/pysupera_' in f'/{name}':
                try:
                    cfg.attrs[f'supera_settings/{name.split("/config/", 1)[1]}/{k}'] = v
                except (TypeError, ValueError):     # empty or odd attrs
                    cfg.attrs[f'supera_settings/{name.split("/config/", 1)[1]}/{k}'] = str(v)
    su['provenance'].visititems(visit)


def campaign_tasks(data_root, out_root, readouts=('wire', 'pixel'),
                   supera_root=None):
    """One job per pysupera file of a doraemon campaign::

        {data_root}/supera_{ro}/{bucket}/supera_{ro}_{task}/supera_*.h5
        {data_root}/jaxtpc_{ro}/{bucket}/jaxtpc_{ro}_{task}/{hits,step,sensor}/

    The labl file goes where JAXTPC would put a fourth modality, so the output
    tree is the JAXTPC tree with only ``labl/`` in it::

        {out_root}/jaxtpc_{ro}/{bucket}/jaxtpc_{ro}_{task}/labl/{name}_labl_{...}.h5

    (``{name}_labl_`` replaces ``{name}_hits_`` in the hits file name.)
    With ``link_inputs`` (:func:`write_campaign`), the task directory also
    gets ``hits``/``step``/``sensor`` symlinks to the JAXTPC ones, so the
    output tree is a complete campaign tree of its own.

    ``supera_root`` (default ``data_root``) is where the pysupera tree is, for a
    JAXTPC tree that has none of its own (e.g. a shared-step rewrite)."""
    jobs = []
    for ro in readouts:
        pat = os.path.join(supera_root or data_root, f'supera_{ro}', '*', f'supera_{ro}_*',
                           f'supera_{ro}_*.h5')
        for su in sorted(glob.glob(pat)):
            bucket = os.path.basename(os.path.dirname(os.path.dirname(su)))
            task = os.path.basename(os.path.dirname(su))[len(f'supera_{ro}_'):]
            jt = f'jaxtpc_{ro}_{task}'
            src = os.path.join(data_root, f'jaxtpc_{ro}', bucket, jt)
            mods = {m: glob.glob(os.path.join(src, m, '*.h5'))
                    for m in ('hits', 'step', 'sensor')}
            hits = mods['hits'][0] if len(mods['hits']) == 1 else None
            name = (os.path.basename(hits).replace('_hits_', '_labl_')
                    if hits else f'sim_{ro}_labl_{jt}.h5')
            jobs.append(dict(
                supera=su, hits=hits, src=src,
                step=mods['step'][0] if len(mods['step']) == 1 else None,
                sensor=mods['sensor'][0] if len(mods['sensor']) == 1 else None,
                out=os.path.join(out_root, f'jaxtpc_{ro}', bucket, jt, 'labl',
                                 name)))
    return jobs


def _link_inputs(job):
    """``{task}/{hits,step,sensor}`` -> the JAXTPC directories (idempotent)."""
    if not os.path.isdir(job['src']):
        return                  # no JAXTPC task: the job reports it
    task = os.path.dirname(os.path.dirname(job['out']))
    os.makedirs(task, exist_ok=True)
    for m in ('hits', 'step', 'sensor'):
        src, dst = os.path.join(job['src'], m), os.path.join(task, m)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.symlink(src, dst)


def _run_job(job):
    t = time.time()
    try:
        missing = [m for m in ('hits', 'step', 'sensor') if job[m] is None]
        if missing:
            raise FileNotFoundError(f'no single JAXTPC {missing} file')
        os.makedirs(os.path.dirname(job['out']), exist_ok=True)
        write_labl(job['supera'], job['hits'], job['out'],
                   step_path=job['step'], sensor_path=job['sensor'])
        return job['out'], time.time() - t, None
    except Exception as exc:                      # report, keep going
        if os.path.exists(job['out'] + '.tmp'):
            os.remove(job['out'] + '.tmp')
        return job['out'], time.time() - t, f'{type(exc).__name__}: {exc}'


def write_campaign(data_root, out_root, readouts=('wire', 'pixel'),
                   workers=1, shard=(0, 1), overwrite=False, link_inputs=False,
                   log=print, supera_root=None):
    """Every pysupera file of a campaign -> its labl file (see
    :func:`campaign_tasks`), ``workers`` files at a time. Files already
    written are skipped unless ``overwrite`` (output appears only when
    complete, so a rerun resumes). ``shard=(i, n)`` takes every n-th job
    from i, for a Slurm array. ``link_inputs`` symlinks each task's JAXTPC
    modalities beside its ``labl/`` (see :func:`campaign_tasks`). Returns the
    failures as ``{out: error}``."""
    jobs = campaign_tasks(data_root, out_root, readouts, supera_root)
    i, n = shard
    jobs = jobs[i::n]
    if link_inputs:
        for j in jobs:
            _link_inputs(j)
    todo = [j for j in jobs if overwrite or not os.path.exists(j['out'])]
    log(f'{len(jobs)} files in shard {i}/{n}; {len(todo)} to write, '
        f'{workers} workers')
    failed, t0 = {}, time.time()
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(workers, maxtasksperchild=8)
        results = pool.imap_unordered(_run_job, todo)
    else:
        pool, results = None, map(_run_job, todo)
    for k, (out, dt, err) in enumerate(results, 1):
        if err:
            failed[out] = err
        log(f'[{k}/{len(todo)}] {dt:6.1f}s {"FAILED " + err if err else "ok"} '
            f'{out}')
    if pool is not None:
        pool.close()
        pool.join()
    log(f'done in {time.time() - t0:.0f}s: {len(todo) - len(failed)} written, '
        f'{len(failed)} failed')
    return failed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['campaign']:
        ap = argparse.ArgumentParser(
            prog='python -m pimm_data.supera_labl campaign',
            description=write_campaign.__doc__.split('\n\n')[0])
        ap.add_argument('data_root')
        ap.add_argument('out_root')
        ap.add_argument('--readouts', nargs='+', default=['wire', 'pixel'])
        ap.add_argument('--workers', type=int, default=1)
        ap.add_argument('--shard', default='0/1', help='I/N (Slurm array)')
        ap.add_argument('--overwrite', action='store_true')
        ap.add_argument('--supera-root', default=None,
                        help='pysupera tree, if not under data_root')
        ap.add_argument('--link-inputs', action='store_true',
                        help='symlink hits/step/sensor beside each labl/')
        a = ap.parse_args(argv[1:])
        i, n = (int(x) for x in a.shard.split('/'))
        failed = write_campaign(a.data_root, a.out_root, a.readouts,
                                workers=a.workers, shard=(i, n),
                                overwrite=a.overwrite,
                                link_inputs=a.link_inputs,
                                supera_root=a.supera_root,
                                log=lambda m: print(m, flush=True))
        return 1 if failed else 0
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('supera')
    ap.add_argument('hits')
    ap.add_argument('out')
    ap.add_argument('--compression', choices=('lz4', 'gzip'), default='lz4')
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--step', default=None,
                    help='JAXTPC step file (with --sensor: deposit_in_window)')
    ap.add_argument('--sensor', default=None, help='JAXTPC sensor file')
    a = ap.parse_args(argv)
    write_labl(a.supera, a.hits, a.out, compression=a.compression,
               max_events=a.max_events, step_path=a.step,
               sensor_path=a.sensor)


if __name__ == '__main__':
    sys.exit(main())
