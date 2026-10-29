"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:

* ``step/`` — 3D truth deposits
* ``sensor/`` — raw sparse wire / pixel readout
* ``hits/`` — per-instance sensor decomposition
* ``labl/`` — pysupera's labels (``docs/LABELS.md``), read when ``labels=``
  is set

Modalities: ``'step'``, ``'sensor'``, ``'hits'``.

Returns a **nested** dict: each loaded modality owns a sub-dict with clean
unprefixed keys::

    {
      'step':   {'coord': (N,3), 'energy': (N,1), 'volume_id': ..., ...},
      'sensor': {'coord': (M,D), 'energy': (M,1), 'plane_id': ...,
                 'readout_type': 'wire'|'pixel',
                 'raw': {plane_label: {'wire'|'py'+'pz', 'time', 'value'}}},
      'hits':   {'coord': (E,D), 'energy': (E,1), 'plane_id': (E,1),
                 'group': (E,), 'readout_type': 'wire'|'pixel'},
      'group':  {'volume': (G,), ...},  # hits loaded, or labels
      # labels= only (docs/LABELS.md):
      'track': {...}, 'particle': {...}, 'interaction': {...},
      'name': str, 'split': str,
    }

Missing modalities have no top-level key. There is no bare ``coord`` / no
precedence / no prefixed aliases — transforms pick a modality explicitly (see
``Apply(on=...)`` and ``Collect(part=...)``).

Registered in :data:`pimm_data.DATASETS` for config-driven construction
via ``dict(type="JAXTPCDataset", ...)``.
"""

import logging

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from . import _resolve
from .readers.jaxtpc_step import JAXTPCStepReader
from .readers.jaxtpc_sensor import JAXTPCSensorReader
from .readers.jaxtpc_labl import JAXTPCLablReader
from .readers.jaxtpc_hits import JAXTPCHitsReader

log = logging.getLogger(__name__)

# Wire-readout plane types, in canonical order. A plane label is
# ``volume_{v}_{T}`` (T in _PLANE_TYPES) or ``volume_{v}_Pixel``.
_PLANE_TYPES = ('U', 'V', 'Y')


def canonical_plane_id(label):
    """Stable, file-independent global id for a plane label.

    ``volume_{v}_{U|V|Y}`` -> ``v * 3 + {U:0,V:1,Y:2}``; ``volume_{v}_Pixel`` -> ``v``.
    Unlike the per-event positional ``plane_id`` (a dense index into the planes
    present *in that event*), this never shifts when a plane is empty or under
    ``volume=`` filtering, so a downstream densify can key per-plane grids/geometry
    by it consistently across a batch.
    """
    parts = label.split('_')
    if len(parts) != 3 or parts[0] != 'volume':
        raise ValueError(
            f"canonical_plane_id: expected a 'volume_{{v}}_{{U|V|Y|Pixel}}' label, "
            f"got {label!r}")
    try:
        v = int(parts[1])
    except ValueError:
        raise ValueError(f"canonical_plane_id: non-integer volume in {label!r}")
    t = parts[2]
    if t == 'Pixel':
        # NOTE: pixel ids share the integer space with wire ids by volume; the
        # dense path is wire-only (plane_geometry() returns {} for pixel), so a
        # mixed wire+pixel registry is unsupported.
        return v
    if t not in _PLANE_TYPES:
        raise ValueError(
            f"canonical_plane_id: unknown plane type {t!r} in {label!r} "
            f"(expected one of {_PLANE_TYPES} or 'Pixel')")
    return v * len(_PLANE_TYPES) + _PLANE_TYPES.index(t)


@DATASETS.register_module()
class JAXTPCDataset(ShardEventDataset):
    """LArTPC multimodal dataset with nested per-modality output.

    The point clouds carry one truth reference, ``group`` (the JAXTPC group:
    the key step and hits share). Labels are lookups from it into pysupera's
    tables, read from ``labl`` when ``labels=`` is set (``docs/LABELS.md``).

    Parameters
    ----------
    data_root : str
        Root directory with ``step/``, ``sensor/``, ``hits/``, ``labl/``
        subdirectories.
    split : str
        Split name for file discovery (one run directory).
    runs : list[str] | str, optional
        Multi-run selection in one dataset (B2): a list of run names, or a
        glob pattern (``'run_*'``) expanded once (sorted) against the first
        modality root with matches. Mutually exclusive with ``split=``.
        Event names become ``'<run>/<file>_evtNNN'`` — unique across runs —
        so content-addressed noise seeds, holdout hashing and cache identity
        never collide (single-run ``split=`` names are unchanged). Nested
        patterns work: doraemon's ``'j*/*'`` selects every task dir.
    files : dict[str, str | list[str]], optional
        Per-modality shard globs (``{'step': '.../step/*.h5', ...}``),
        replacing ``data_root``/``split``/``runs`` discovery for every loaded
        modality. Shards pair across modalities by their trailing filename
        token (the doraemon task tag ``j000000-000004``), so each modality may
        live under its own root — e.g. ``labl`` produced separately.
    modalities : tuple[str]
        Any non-empty subset of ``'step'``, ``'sensor'``, ``'hits'``.
    dataset_name : str
        File prefix (e.g., ``'sim'`` for ``sim_step_0000.h5``).
    volume : int or None
        Load only this volume index. ``None`` = all volumes.
    labels : str or None
        Read ``labl`` and label ``step``/``hits`` points: ``segment`` is
        ``'motif'`` (shower, track, michel, delta, le), ``'pid'``, ``'pdg'``
        (the instance's raw PDG), ``'particle'`` or ``'interaction'``;
        ``instance`` is the particle (instance) row. The tables (``group``,
        ``track``, ``particle``, ``interaction``) come with them. ``None``:
        no labl, no labels.
    label_keys : tuple[str] or 'all'
        Further per-point keys from ``_resolve.PRESETS['pilarnet']``
        (``segment_pid``, ``instance_interaction``, ``momentum``, ``vertex``,
        …; ``'all'`` for every one). Each is a lookup over every point, so
        ask only for what a model or metric reads.
    momentum : str
        The per-point ``momentum`` is ``|p|`` of the instance at creation
        (``'start'``, PILArNet's meaning) or at its first visible step
        (``'first'``).
    min_deposits : int
        Minimum 3D deposits per event (step reader filter).
    include_physics : bool
        Whether step reader loads dx, theta, phi, charge, photons, etc.
    transform : list or None
        Transform pipeline.
    loop, max_len, ignore_index : standard dataset parameters.
    """


    VALID_MODALITIES = ('step', 'sensor', 'hits')

    def __init__(
        self,
        data_root,
        split='train',
        runs=None,
        files=None,
        modalities=('step',),
        dataset_name='sim',
        volume=None,
        labels=None,
        label_keys=(),
        momentum='start',
        min_deposits=0,
        include_physics=True,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
        cache=False,
        num_time_steps=None,
        n_wires_per_plane=None,
        pedestal_per_plane=None,
        wire_lengths_per_plane=None,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        # Per-plane wire lengths (METERS) for incoherent-noise ENC, keyed by
        # plane TYPE ('U'/'V'/'Y') or full label, value an (n_wires,) array or a
        # (lo, hi) pair expanded via linspace. Detector geometry — not in the
        # file — so it is config, surfaced through plane_geometry().
        self._wire_lengths_per_plane = dict(wire_lengths_per_plane or {})

        # A3: min_deposits filters on step deposit counts, so it is a no-op
        # (and silently so) without the step modality. Fail loud instead.
        if min_deposits > 0 and 'step' not in self._modalities:
            raise ValueError(
                f"min_deposits={min_deposits} filters on step deposit counts "
                f"but modalities={self._modalities} does not include 'step'. "
                "Add 'step' to modalities or set min_deposits=0.")

        if labels is not None:
            if labels not in _resolve.SEGMENTS:
                raise ValueError(
                    f"labels={labels!r}: use one of "
                    f"{sorted(_resolve.SEGMENTS)} or None.")
            if not {'step', 'hits'} & set(self._modalities):
                raise ValueError(
                    "labels= labels step or hits points; modalities="
                    f"{self._modalities} has neither.")
        preset = _resolve.PRESETS['pilarnet']
        label_keys = tuple(preset) if label_keys == 'all' else tuple(label_keys)
        unknown = set(label_keys) - set(preset)
        if unknown:
            raise ValueError(f"label_keys {sorted(unknown)}: not in "
                             f"{sorted(preset)}.")
        if momentum not in ('start', 'first'):
            raise ValueError(f"momentum={momentum!r}: 'start' or 'first'.")

        self._dataset_name = dataset_name
        self._volume = volume
        self._labels = labels
        self._label_keys = label_keys
        self._momentum = momentum
        self._min_deposits = min_deposits
        self._include_physics = include_physics
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        # B2 runs=: multi-run selection in ONE dataset. A list/tuple names
        # runs verbatim; a string is a glob expanded once (sorted) against
        # the first modality's root. Readers receive the explicit list, and
        # get_data_name() qualifies names as '<run>/<file>_evtNNN' so noise
        # seeds / holdout hashes / cache identity stay unique across runs
        # (single-run split= names are unchanged — byte-identical legacy).
        if runs is not None:
            if split != 'train':
                raise ValueError(
                    f"Pass either runs= (multi-run) or split= (single run), "
                    f"not both (got runs={runs!r}, split={split!r}).")
            split = self._expand_runs(runs, self._modalities)
        self._source_split = split
        if files is not None:
            missing = [m for m in self._modalities if m not in files] + (
                ['labl'] if labels is not None and 'labl' not in files
                else [])
            if missing:
                raise ValueError(
                    f"files= must name every loaded modality; missing "
                    f"{missing} (got {sorted(files)}).")
        files = dict(files or {})

        self.step_reader = None
        self.sensor_reader = None
        self.labl_reader = None
        self.hits_reader = None

        if 'step' in self._modalities:
            self.step_reader = JAXTPCStepReader(
                data_root=self._modality_root('step'), split=split,
                files=files.get('step'),
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        # sensor/hits readout auto-detection. Build the readers unfiltered
        # first so they can detect readout_type, then apply the volume
        # plane filter using the correct plane labels for this readout.
        if 'sensor' in self._modalities:
            self.sensor_reader = JAXTPCSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                files=files.get('sensor'),
                dataset_name=dataset_name, planes='all',
                n_wires_per_plane=n_wires_per_plane,
                num_time_steps=num_time_steps,
                pedestal_per_plane=pedestal_per_plane)

        if labels is not None:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                files=files.get('labl'), dataset_name=dataset_name)

        if 'hits' in self._modalities:
            self.hits_reader = JAXTPCHitsReader(
                data_root=self._modality_root('hits'), split=split,
                files=files.get('hits'),
                dataset_name=dataset_name, planes='all')

        # Resolve readout_type once from whichever reader can tell us.
        self._readout_type = 'wire'
        for r in (self.sensor_reader, self.hits_reader):
            if r is not None:
                self._readout_type = r.readout_type
                break

        # Now that readout_type is known, apply per-volume plane filter.
        if volume is not None:
            if self._readout_type == 'pixel':
                planes = [f'volume_{volume}_Pixel']
            else:
                planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                          f'volume_{volume}_Y']
            if self.sensor_reader is not None:
                self.sensor_reader.planes = planes
            if self.hits_reader is not None:
                self.hits_reader.planes = planes

        self._canonical_reader = (self.step_reader or self.sensor_reader
                                  or self.hits_reader or self.labl_reader)
        # Phase A / D42: build ONE joint cross-modality event index and inject
        # it into every reader, so a single global idx maps to the SAME
        # physics event in all modalities. (Replaces the old
        # `_n_events = min(len(r) ...)`, which left each reader mapping idx
        # through its own present-key index → silent desync under
        # min_deposits>0 or a gap present in some-but-not-all modalities.)
        self._build_joint_index(
            source_label=f"JAXTPCDataset({self._source_data_root!r})",
            filter_label=(f"min_deposits={self._min_deposits}"
                          if self._min_deposits > 0 else ''))

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )

        # Fail fast on empty data_list — otherwise get_data() crashes later
        # with an opaque ZeroDivisionError on `idx % len(self.data_list)`.
        if len(self.data_list) == 0:
            raise ValueError(
                f"JAXTPCDataset(data_root={data_root!r}) yielded 0 events "
                f"after filters (min_deposits={min_deposits}, "
                f"max_len={max_len}). Lower min_deposits or verify the "
                f"dataset has events meeting it.")

    def get_data(self, idx):
        """Load one event as a nested dict (schema: see module docstring)."""
        # data_list VALUES are event ids. Identical to `idx % len(...)` while
        # the list is range(n), and the only correct form once it is not.
        real_idx = self.data_list[idx % len(self.data_list)]

        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }

        labl = None
        if self.labl_reader is not None:
            labl = self.labl_reader.read_event(real_idx)
            if self._volume is not None:
                labl = {k: v for k, v in labl.items()
                        if not k.startswith('v') or k == f'v{self._volume}'}
        hits_raw = hits_gid = None
        if self.hits_reader is not None:
            hits_raw = self.hits_reader.read_event(real_idx)
            data['hits'], hits_gid = self._build_hits_cloud(hits_raw)

        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor_cloud(
                self.sensor_reader.read_event(real_idx))
            # Surface the event id for transforms that need a stable per-event
            # seed (e.g. AddNoise) while scoped to the sensor sub-dict.
            data['sensor']['name'] = data['name']

        if self.step_reader is not None:
            data['step'] = self.step_reader.read_event(real_idx)

        self._build_relations(data, labl, hits_raw, hits_gid)
        if labl is not None:
            for on in ('step', 'hits'):
                if on in data:
                    _resolve.resolve_points(
                        data, on, segment=self._labels,
                        momentum=self._momentum, keys=self._label_keys)
        return data

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------

    def _build_sensor_cloud(self, sensor_raw):
        """Merge per-plane sensor raw into a point cloud + raw passthrough."""
        coord_keys = self._coord_keys()
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            sensor_raw, prefix='sensor', value_key='value',
            coord_keys=coord_keys)
        sub = {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'planes': planes, 'raw': raw,
            'readout_type': self._readout_type,
        }
        # Per-plane fixed grid extent for a downstream Densify, from the reader's
        # surfaced geometry. Wire readout only; absent when geometry is unknown
        # (Densify then errors unless require_shape=False). Pedestal (raw-ADC
        # offset) is surfaced alongside for a downstream Digitize.
        if self._readout_type != 'pixel' and self.sensor_reader is not None:
            nts = getattr(self.sensor_reader, 'num_time_steps', None)
            nwbp = getattr(self.sensor_reader, 'n_wires_per_plane', {})
            shape = {p: (int(nwbp[p]), int(nts)) for p in planes
                     if p in nwbp and nts is not None}
            if shape:
                sub['shape'] = shape
            pbp = getattr(self.sensor_reader, 'pedestal_per_plane', {})
            pedestal = {p: int(pbp[p]) for p in planes if p in pbp}
            if pedestal:
                sub['pedestal'] = pedestal
            # Flat integer scatter inputs for the dense path (a downstream
            # Collect(into='sensor') pulls these; densify scatters them).
            # coord/energy are float32 but wire/time are small exact integers.
            if planes:
                sub['wire'] = coord[:, 0].astype(np.int32)
                sub['time'] = coord[:, 1].astype(np.int32)
                sub['value'] = energy[:, 0].astype(np.float32)
                # canonical (stable) plane id per point, parallel to the
                # positional plane_id (which is left unchanged for back-compat)
                gid_of_pos = np.array([canonical_plane_id(p) for p in planes],
                                      dtype=np.int32)
                sub['plane_gid'] = gid_of_pos[plane_id[:, 0]].astype(np.int32)
        return sub

    def _wire_lengths_for(self, label, n_wires):
        """Per-wire lengths (m) for a plane: (n_wires,) array, or (lo,hi)->linspace."""
        ptype = label.split('_')[-1]
        spec = self._wire_lengths_per_plane.get(
            label, self._wire_lengths_per_plane.get(ptype))
        if spec is None:
            return None
        arr = np.asarray(spec, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == n_wires:
            return arr
        if arr.size == 2:
            return np.linspace(float(arr.flat[0]), float(arr.flat[1]),
                               n_wires, dtype=np.float32)
        raise ValueError(
            f"wire_lengths for {label!r}: expected ({n_wires},) array or "
            f"(lo, hi) pair, got shape {arr.shape}")

    def plane_geometry(self):
        """Canonical-plane-id -> geometry registry for the dense path.

        Returns ``{gid: {'n_wires', 'n_ticks', 'pedestal', ['wire_lengths']}}``
        built from the sensor reader's file-surfaced geometry plus the
        ``wire_lengths_per_plane`` config. Geometry is sourced HERE (the registry),
        not carried per-sample through collate; the post-collate densify/noise
        stages take this dict. ``wire_lengths`` is present only when configured
        (the incoherent-noise stage requires it; densify/coherent/digitize do not).
        Note the reader populates ``n_wires_per_plane`` lazily — read at least one
        event (or build after indexing) before relying on this.
        """
        if self.sensor_reader is None or self._readout_type == 'pixel':
            return {}
        # the reader fills n_wires_per_plane lazily on read; populate it if this
        # is called before the loader has iterated (e.g. trainer before_train).
        if not getattr(self.sensor_reader, 'n_wires_per_plane', {}):
            try:
                self.sensor_reader.read_event(0)
            except Exception:
                pass
        nts = getattr(self.sensor_reader, 'num_time_steps', None)
        nwbp = getattr(self.sensor_reader, 'n_wires_per_plane', {})
        pbp = getattr(self.sensor_reader, 'pedestal_per_plane', {})
        geom = {}
        for label, nw in nwbp.items():
            entry = {'n_wires': int(nw)}
            if nts is not None:
                entry['n_ticks'] = int(nts)
            if label in pbp:
                entry['pedestal'] = int(pbp[label])
            wl = self._wire_lengths_for(label, int(nw))
            if wl is not None:
                entry['wire_lengths'] = wl
            geom[canonical_plane_id(label)] = entry
        return geom

    def _build_hits_cloud(self, hits_raw):
        """Merge the planes' hit entries into one cloud. Returns it and the
        entries' volume-local group ids (``_build_relations`` turns them into
        the ``group`` reference)."""
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            hits_raw, prefix='hits', value_key='charge',
            coord_keys=self._coord_keys(), extra_keys=('group_id',))
        gid = [raw[p]['group_id'] for p in planes]
        sub = {'coord': coord, 'energy': energy, 'plane_id': plane_id,
               'planes': planes, 'readout_type': self._readout_type}
        return sub, gid

    # ------------------------------------------------------------------
    # Relations: the group hub and the label tables (docs/LABELS.md)
    # ------------------------------------------------------------------

    def _build_relations(self, data, supera, hits_raw, hits_gid):
        """Parts and references (docs/LABELS.md).

        Step and hits carry nothing above the group: one ``group`` reference
        per deposit / hit entry. The group is the correspondence key the two
        share, so it exists without labl too (the matching regime: hits gives
        both ``deposit_to_group`` and the entries' group ids). Everything from
        the group up — its Geant4 track, the track's owning fragment, the
        particle and interaction tables — comes from labl alone.

        ``group`` — one row per JAXTPC group, event-wide (each volume's local
        ids are offset by the group counts of the volumes before it, as
        pysupera does, since a particle can cross volumes): ``volume``,
        ``track`` (ref -> ``track``, labl's visible tracks; -1 if the readout
        saw none of it) and ``fragment`` (ref -> ``particle``: the track's
        owner, gathered here so every walk starts ``group.fragment``). ``step.group`` / ``hits.group`` reference it;
        ``step.qs`` is the deposit's share of its group's step charge (derived
        here; it is what JAXTPC stores as ``qs_fractions``). ``particle`` and
        ``interaction`` are the labl tables with their row references
        declared. Hits only cross-check group counts."""
        if supera is None and hits_raw is None:
            return
        per_vol = {}       # v -> dict(n, track, fragment, d2g)
        if supera is not None:
            for k, vd in supera.items():
                if k.startswith('v'):
                    per_vol[int(k[1:])] = dict(
                        n=len(vd['group_track']),
                        track=vd['group_track'],
                        ovr=(vd.get('override_group'),
                             vd.get('override_fragment')),
                        d2g=vd.get('deposit_to_group'),
                        inw=vd.get('deposit_in_window'))
        if hits_raw is not None:
            for k, g2t in hits_raw.items():
                if not k.startswith('group_to_track_v'):
                    continue
                v = int(k.rsplit('_v', 1)[1])
                if supera is not None and v not in per_vol:
                    continue
                d = per_vol.setdefault(v, dict(n=len(g2t)))
                if d['n'] != len(g2t):
                    raise ValueError(
                        f"volume {v}: {d['n']} groups in labl, {len(g2t)} in "
                        f"hits — labl and hits are not the same event/task")
                # hits only cross-checks: track identity is labl's alone
                if d.get('d2g') is None:
                    d['d2g'] = hits_raw.get(f'deposit_to_group_v{v}')
        if self._volume is not None:
            per_vol = {v: d for v, d in per_vol.items() if v == self._volume}
        if not per_vol:
            return
        vols = sorted(per_vol)
        base, b = {}, 0
        for v in vols:
            base[v] = b
            b += per_vol[v]['n']
        cat = lambda key, dt: np.concatenate(
            [np.asarray(per_vol[v][key]).astype(dt) if per_vol[v].get(key)
             is not None else np.full(per_vol[v]['n'], -1, dt) for v in vols])
        group = {'volume': np.concatenate(
                     [np.full(per_vol[v]['n'], v, np.int32) for v in vols]),
                 '_roles': {}}
        if supera is not None:
            # everything above the group comes from labl only: group -> track
            # (Geant4 id -> visible-track row) -> owning fragment
            trk = supera['track']
            tid = np.asarray(trk['id'], np.int64)
            g2t = cat('track', np.int64)
            row = np.full(len(g2t), -1, np.int32)
            if len(tid):
                pos = np.minimum(np.searchsorted(tid, g2t), len(tid) - 1)
                row = np.where(tid[pos] == g2t, pos, -1).astype(np.int32)
            group['track'] = row
            group['fragment'] = _resolve._take(trk['fragment'], row, -1) \
                .astype(np.int32)
            for v in vols:      # groups of a track pysupera split
                g, f = per_vol[v].get('ovr', (None, None))
                if g is not None and len(g):
                    group['fragment'][np.asarray(g) + base[v]] = f
            group['_roles'].update(track=('ref', 'track'),
                                   fragment=('ref', 'particle'))
            data['track'] = dict(trk, _roles={'fragment': ('ref', 'particle')})
        data['group'] = group

        if 'step' in data and 'volume_id' in data['step']:
            st = data['step']
            vid = st['volume_id'].ravel()
            ref = np.full(len(vid), -1, np.int32)
            inw = np.zeros(len(vid), np.int32)
            has_inw = any(per_vol[v].get('inw') is not None for v in vols)
            for v in vols:
                d2g = per_vol[v].get('d2g')
                mask = vid == v
                if d2g is None or not mask.any():
                    continue
                if len(d2g) != int(mask.sum()):
                    raise ValueError(
                        f"volume {v}: {len(d2g)} deposit_to_group entries, "
                        f"{int(mask.sum())} step deposits")
                ref[mask] = np.where(d2g >= 0, d2g + base[v], -1)
                if per_vol[v].get('inw') is not None:
                    inw[mask] = per_vol[v]['inw']
            st['group'] = ref
            if has_inw:
                # labl's: does the deposit's charge reach the readout window
                # (an unowned group is out of it, or under threshold)
                st['in_window'] = inw[:, None]
            if 'charge' in st:
                # A deposit's share of its group's recombined charge — JAXTPC's
                # qs_fractions, which is exactly this (to float16): a step
                # quantity, the weight that splits a hit's charge over the
                # deposits of its group. Derived, so no file needs to carry it.
                q = st['charge'][:, 0].astype(np.float64)
                tot = np.bincount(np.maximum(ref, 0), weights=np.where(
                    ref >= 0, q, 0.0), minlength=len(group['volume']))
                st['qs'] = np.where(ref >= 0, q / np.maximum(
                    tot[np.maximum(ref, 0)], 1e-30), np.nan) \
                    .astype(np.float32)[:, None]
            st.setdefault('_roles', {})['group'] = ('ref', 'group')

        if 'hits' in data:
            h = data['hits']
            refs = []
            for plane, gid in zip(h['planes'], hits_gid):
                v = int(plane.split('_')[1])
                refs.append(np.where(gid >= 0, gid + base[v], -1)
                            if v in base else np.full(len(gid), -1))
            h['group'] = (np.concatenate(refs).astype(np.int32) if refs
                          else np.zeros(0, np.int32))
            h.setdefault('_roles', {})['group'] = ('ref', 'group')

        if supera is not None:
            part = dict(supera['particle'])
            part['_roles'] = {c: ('ref', 'particle') for c in (
                'fragment', 'instance', 'fragment_instance', 'fragment_parent',
                'instance_parent', 'parent', 'ancestor') if c in part}
            part['_roles']['interaction'] = ('ref', 'interaction')
            data['particle'] = part
            data['interaction'] = dict(supera.get('interaction', {}),
                                       _roles={})

    def _coord_keys(self):
        """Per-plane column names that build the sensor/hits coord vector."""
        if self._readout_type == 'pixel':
            return ('py', 'pz', 'time')
        return ('wire', 'time')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_plane_dotted(raw_dict, prefix, value_key, coord_keys,
                            extra_keys=()):
        """Merge `{prefix}.{plane}.{col}` flat keys into a single point cloud.

        ``coord_keys`` is the ordered tuple of per-plane column names that
        form the coord vector (e.g. ``('wire', 'time')`` for wire readout,
        ``('py', 'pz', 'time')`` for pixel). Coord shape is ``(M, len(coord_keys))``.

        Returns (planes, coord, energy, plane_id, raw_nested).
        raw_nested is ``{plane_label: {*coord_keys, value_key, *extra_keys}}``.
        """
        coord_dim = len(coord_keys)
        # Discover planes by scanning for any coord-key suffix (they all
        # share the same plane set, so the first key is enough).
        first_key = coord_keys[0]
        planes = sorted(set(
            k.split('.')[1] for k in raw_dict
            if k.startswith(prefix + '.') and k.endswith('.' + first_key)
        ))
        # Gather per-plane columns + sizes (and build raw_nested) in one pass,
        # then fill preallocated output arrays per-plane. This avoids the
        # np.stack-per-plane + concatenate-across-planes copies (the single
        # biggest reader op on large events); the slice assignment casts
        # int -> float32 in place.
        raw_nested = {}
        plane_cols, sizes = [], []
        for plane in planes:
            cols_arrays = [raw_dict[f'{prefix}.{plane}.{ck}']
                           for ck in coord_keys]
            value = raw_dict[f'{prefix}.{plane}.{value_key}']
            plane_cols.append((cols_arrays, value))
            sizes.append(len(cols_arrays[0]))
            cols = {ck: arr for ck, arr in zip(coord_keys, cols_arrays)}
            cols[value_key] = value
            for ek in extra_keys:
                cols[ek] = raw_dict[f'{prefix}.{plane}.{ek}']
            raw_nested[plane] = cols

        N = sum(sizes)
        coord = np.empty((N, coord_dim), dtype=np.float32)
        energy = np.empty((N, 1), dtype=np.float32)
        plane_id = np.empty((N, 1), dtype=np.int32)
        off = 0
        for i, ((cols_arrays, value), n) in enumerate(zip(plane_cols, sizes)):
            sl = slice(off, off + n)
            for j, arr in enumerate(cols_arrays):
                coord[sl, j] = arr
            energy[sl, 0] = value
            plane_id[sl, 0] = i
            off += n

        return planes, coord, energy, plane_id, raw_nested
