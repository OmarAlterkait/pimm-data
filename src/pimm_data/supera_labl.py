"""Write a JAXTPC ``labl`` file from one pysupera (format 4.x) output.

The label of a deposit or a hit is the label of its JAXTPC *group*: pysupera
records, per group, the fragment that owns it (``groups/fragment_id``), and
every other label is a property of that fragment, its instance or its
interaction. The sensor is the summed readout and carries no labels. So a
``labl`` file holds two things per event:

* per volume, ``group_fragment`` — the particle row owning each group — plus
  copies of the JAXTPC ``deposit_to_group`` and ``group_to_track`` (as
  ``group_track``), so step+labl needs no hits file: the Geant4 track of a
  group is also the one identity shared by the wire and pixel partitions;
* the ``particle`` and ``interaction`` tables those rows index;
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
(derivable from ``parent``), merge counts and pysupera's own bookkeeping.

The file mirrors the JAXTPC file it labels: the same ``entry_NNNNNN`` groups in
the same order, each carrying ``event_id`` / ``job_id`` / ``job_event``, and the
same ``/dprod/events`` table. Name it ``{dataset}_labl_{task}.h5`` so it pairs
with ``{dataset}_{step,hits,sensor}_{task}.h5`` by the trailing task token.

Import-light on purpose (numpy, h5py, hdf5plugin only), so a production stage
can run it without the rest of pimm_data::

    python -m pimm_data.supera_labl SUPERA.h5 HITS.h5 OUT.h5 \
        [--step STEP.h5 --sensor SENSOR.h5]
"""

import argparse
import os
import sys

import h5py
import hdf5plugin
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


def write_labl(supera_path, hits_path, out_path, compression='lz4',
               max_events=None, step_path=None, sensor_path=None):
    """Extract ``supera_path`` into ``out_path``, aligned to ``hits_path``.

    ``hits_path`` is the JAXTPC hits file pysupera read (one file per task
    with merged inputs). It supplies ``deposit_to_group`` and is the
    alignment check: every entry's ``event_id`` and group count must agree.
    ``step_path`` and ``sensor_path`` (the same task's JAXTPC step and sensor
    files) add ``deposit_in_window``; give both or neither. ``max_events``
    truncates (for testing); the result is then partial."""
    if (step_path is None) != (sensor_path is None):
        raise ValueError('give step_path and sensor_path together')
    win = _window(sensor_path) if sensor_path is not None else None
    st = h5py.File(step_path, 'r') if step_path is not None else None
    comp = ({'compression': 'gzip', 'compression_opts': 4}
            if compression == 'gzip' else
            hdf5plugin.Blosc(cname='lz4', clevel=5,
                             shuffle=hdf5plugin.Blosc.SHUFFLE))
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
        P, G, I = su['particles'], su['groups'], su['inter']
        icols = list(INTERACTION_COLUMNS) + [
            c for c in OPTIONAL_INTERACTION_COLUMNS
            if c in I and any(_decode(I[c][:min(len(I[c]), 4096)]))]

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
        _copy_provenance(su, cfg)

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
            for c, v in id_cols.items():
                oev.attrs[c] = int(v[e])

            ps = slice(int(p_off[e]), int(p_off[e + 1]))
            ids = P['id'][ps].astype(np.int64)
            sorter = np.argsort(ids, kind='stable').astype(np.int32)
            row_of_id = (sorter, ids[sorter])

            n_vol = sum(1 for k in hev if k.startswith('volume_'))
            gfrag = _to_rows(G['fragment_id'][int(g_off[e]):int(g_off[e + 1])],
                             row_of_id, f'{key} groups/fragment_id')
            voffs = v_off[int(v_idx[e]):int(v_idx[e + 1])]
            if len(voffs) != n_vol:
                raise ValueError(f'{key}: {len(voffs)} pysupera volumes, '
                                 f'{n_vol} in hits')
            bounds = list(voffs) + [len(gfrag)]
            for v in range(n_vol):
                hv = hev[f'volume_{v}']
                gv = gfrag[int(bounds[v]):int(bounds[v + 1])]
                n_groups = len(hv['group_to_track'])     # == attrs n_groups
                if len(gv) != n_groups:
                    raise ValueError(
                        f'{key}/volume_{v}: {len(gv)} groups in pysupera, '
                        f'{n_groups} in hits')
                ov = oev.create_group(f'volume_{v}')
                _create(ov, 'group_fragment', gv, comp)
                _create(ov, 'deposit_to_group',
                        hv['deposit_to_group'][:].astype(np.int32), comp)
                _create(ov, 'group_track',
                        hv['group_to_track'][:].astype(np.int32), comp)
                if st is not None:
                    sv = st[f'{key}/volume_{v}']
                    if int(st[key].attrs['event_id']) != int(event_ids[e]):
                        raise ValueError(f'{key}: event_id differs in step')
                    if len(sv['t0_us']) != len(hv['deposit_to_group']):
                        raise ValueError(f'{key}/volume_{v}: step and hits '
                                         f'deposit counts differ')
                    _create(ov, 'deposit_in_window',
                            _in_window(sv, win['anode'][v], win), comp)

            op = oev.create_group('particle')
            for name, src in ID_COLUMNS:
                _create(op, name, _to_rows(P[src][ps], row_of_id,
                                           f'{key} {src}'), comp)
            for name, src in VALUE_COLUMNS:
                _create(op, name, P[src][ps], comp)

            oi = oev.create_group('interaction')
            isl = slice(int(i_off[e]), int(i_off[e + 1]))
            for c in icols:
                col = I[c][isl]
                if col.dtype.kind in 'OS':
                    col = np.array(_decode(col), dtype=h5py.string_dtype())
                _create(oi, c, col, comp)
    if st is not None:
        st.close()
    os.replace(out_path + '.tmp', out_path)
    return out_path


def _copy_provenance(su, cfg):
    """The pysupera commit and config hashes, as flat attrs (whatever of the
    dprod provenance tree the input has)."""
    if 'provenance' not in su:
        return

    def visit(name, obj):
        for k, v in obj.attrs.items():
            if ('pysupera' in name or 'supera' in name.split('/')[0]) and (
                    'commit' in k or 'dirty' in k or 'sha256' in k):
                cfg.attrs[f'provenance/{name}/{k}'] = v
    su['provenance'].visititems(visit)


def main(argv=None):
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
