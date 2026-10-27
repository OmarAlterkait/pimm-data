"""Write a JAXTPC ``labl`` file from one pysupera (format 4.x) output.

Point files carry nothing above the JAXTPC *group* (step: deposit -> group;
hits: entry -> group). The group is the correspondence key step and hits share,
and is not a label. Everything above it — every piece of track or particle
truth — lives here, per event:

* per volume: ``group_fragment``, the ``particle`` row owning each group
  (pysupera's ``groups/fragment_id``, -1 when nothing visible owns it), and
  ``group_track``, its Geant4 ``track`` row; plus a copy of the JAXTPC
  ``deposit_to_group``, which belongs in step but is written to hits today;
* ``track``: every Geant4 track of the event (edep-sim ``particle/geant4``),
  whether or not the readout saw it;
* ``interaction``: every vertex of the event (edep-sim ``vertex/geant4``).
  pysupera keeps only the vertices a stored particle references, and drops
  exactly the ones the readout never saw;
* ``particle``: pysupera's fragments, instances and their links, with
  ``track`` and ``interaction`` references into the two tables above.

Everything is copied. The one transformation is to row positions: pysupera ids
(``id``, ``frag_id``, …) are not rows, and its interaction rows are
renumbered survivors, mapped back through ``vertex_id``. Edep-sim track ids and
vertex ids already are rows (checked). Left out, and why: ``docs/LABELS.md``.

The file mirrors the JAXTPC file it labels: the same ``entry_NNNNNN`` groups in
the same order, each carrying ``event_id`` / ``job_id`` / ``job_event``, and the
same ``/dprod/events`` table. Name it ``{dataset}_labl_{task}.h5`` so it pairs
with ``{dataset}_{step,hits,sensor}_{task}.h5`` by the trailing task token.

Import-light on purpose (numpy, h5py, hdf5plugin only), so a production stage
can run it without the rest of pimm_data::

    python -m pimm_data.supera_labl SUPERA.h5 HITS.h5 OUT.h5
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
# Copied unchanged. (``track``: edep-sim track ids are its rows.) Start
# kinematics, mass and Geant4 parentage are the track's, so they are read
# through ``track``; end kinematics stay, as pysupera corrects a converting
# photon's end momentum; first/last (visible steps of the instance) are
# pysupera's own.
VALUE_COLUMNS = (
    ('track', 'geant4_trackid'),
    ('pdg', 'pdg'),
    ('sem_type', 'sem_type'),
    ('fragment_sem_type', 'frag_sem_type'),
    ('instance_sem_type', 'inst_sem_type'),
    ('interaction_type', 'interaction_type'),
    ('first_pos', 'first_pos'), ('first_mom', 'first_mom'),
    ('last_pos', 'last_pos'), ('last_mom', 'last_mom'),
    ('end_pos', 'end_pos'), ('end_mom', 'end_mom'),
)
# edep-sim vertex/geant4 -> interaction. ``t`` is renamed ``time`` (pysupera's
# name). Strings are copied when any is non-empty: the generator source belongs
# here as soon as the generator records it.
INTERACTION_COLUMNS = (('x', 'x'), ('y', 'y'), ('z', 'z'), ('time', 't'),
                       ('energy_sum', 'energy_sum'), ('ke_sum', 'ke_sum'),
                       ('num_particles', 'num_particles'))
OPTIONAL_INTERACTION_COLUMNS = ('generator', 'reaction')
# edep-sim particle/geant4 -> track (units mm, ns, MeV; momentum 3-vectors and
# kinetic energy, as edep-sim records them).
TRACK_SCALARS = (('pdg', 'pdg'), ('mass', 'mass'),
                 ('start_ke', 'ke'), ('end_ke', 'end_ke'),
                 ('proc_start', 'proc_start'),
                 ('subproc_start', 'subproc_start'),
                 ('proc_end', 'proc_end'), ('subproc_end', 'subproc_end'))
TRACK_VECTORS = (('start_pos', ('x', 'y', 'z', 't')),
                 ('start_mom', ('px', 'py', 'pz')),
                 ('end_pos', ('end_x', 'end_y', 'end_z', 'end_t')),
                 ('end_mom', ('end_px', 'end_py', 'end_pz')))

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


def write_labl(supera_path, hits_path, out_path, compression='lz4',
               max_events=None, edepsim_paths=None):
    """Extract ``supera_path`` into ``out_path``, aligned to ``hits_path``.

    ``hits_path`` is the JAXTPC hits file pysupera read (one file per task
    with merged inputs). It supplies ``deposit_to_group`` / ``group_to_track``
    and is the alignment check: every entry's ``event_id`` and group count
    must agree. The edep-sim files (tracks and vertices) are the ones pysupera
    recorded in ``inputs/edepsim``, unless ``edepsim_paths`` replaces them
    (same order). ``max_events`` truncates (for testing)."""
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
        ed_paths = (list(edepsim_paths) if edepsim_paths is not None
                    else _decode(su['inputs/edepsim'][:]))
        ed_in = ev['input'][:]
        ed_event = ev['input_event'][:]
        ed_files = {}

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
            [os.path.abspath(supera_path), os.path.abspath(hits_path)]
            + [os.path.abspath(x) for x in ed_paths],
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

            op = oev.create_group('particle')
            for name, src in ID_COLUMNS:
                _create(op, name, _to_rows(P[src][ps], row_of_id,
                                           f'{key} {src}'), comp)
            # edep-sim: every track and every vertex of this event
            i = int(ed_in[e])
            if i not in ed_files:
                ed_files[i] = h5py.File(ed_paths[i], 'r')
            ed = ed_files[i]
            ie = int(ed_event[e])
            tracks = ed['particle/geant4'][ie]
            verts = ed['vertex/geant4'][ie]
            prims = ed['primary/geant4'][ie]
            if not np.array_equal(tracks['track_id'], np.arange(len(tracks))):
                raise ValueError(f'{key}: edep-sim track ids are not rows')
            if not np.array_equal(verts['interaction_id'],
                                  np.arange(len(verts))):
                raise ValueError(f'{key}: edep-sim vertex ids are not rows')

            for name, src in VALUE_COLUMNS:
                col = P[src][ps]
                if name == 'track' and len(col) and (
                        col.min() < 0 or col.max() >= len(tracks)):
                    raise ValueError(f'{key}: geant4_trackid outside edep-sim')
                _create(op, name, col, comp)
            # pysupera's interaction rows are renumbered survivors: map them
            # back to vertex rows through its vertex_id.
            isl = slice(int(i_off[e]), int(i_off[e + 1]))
            vid = I['vertex_id'][isl].astype(np.int64)
            pint = P['interaction_id'][ps].astype(np.int64)
            _create(op, 'interaction', np.where(
                pint >= 0, vid[np.maximum(pint, 0)] if len(vid) else -1,
                -1).astype(np.int32), comp)

            ot = oev.create_group('track')
            for name, src in TRACK_SCALARS:
                _create(ot, name, tracks[src], comp)
            for name, srcs in TRACK_VECTORS:
                _create(ot, name, np.stack([tracks[c] for c in srcs], 1),
                        comp)
            _create(ot, 'parent', tracks['parent_track_id'].astype(np.int32),
                    comp)
            _create(ot, 'ancestor',
                    tracks['ancestor_track_id'].astype(np.int32), comp)
            # every ancestor is a primary; the primary table names its vertex
            vertex_of = np.full(len(tracks), -1, np.int32)
            vertex_of[prims['track_id']] = prims['interaction_id']
            anc = tracks['ancestor_track_id']
            _create(ot, 'interaction', np.where(
                anc >= 0, vertex_of[np.maximum(anc, 0)], -1).astype(np.int32),
                comp)

            oi = oev.create_group('interaction')
            for name, src in INTERACTION_COLUMNS:
                _create(oi, name, verts[src], comp)
            for c in OPTIONAL_INTERACTION_COLUMNS:
                if c in verts.dtype.names and any(_decode(verts[c])):
                    _create(oi, c, np.array(_decode(verts[c]),
                                            dtype=h5py.string_dtype()), comp)
        for f in ed_files.values():
            f.close()
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
    ap.add_argument('--edepsim', nargs='+', default=None,
                    help="edep-sim files, replacing pysupera's inputs/edepsim")
    a = ap.parse_args(argv)
    write_labl(a.supera, a.hits, a.out, compression=a.compression,
               max_events=a.max_events, edepsim_paths=a.edepsim)


if __name__ == '__main__':
    sys.exit(main())
