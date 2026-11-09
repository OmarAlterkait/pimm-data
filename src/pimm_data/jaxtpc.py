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
                 'group': (E,)          # volume-numbered hits, or plane-local + labl
                 | 'plane_group': (E,), # plane-local hits without labl (no ref)
                 'readout_type': 'wire'|'pixel'},
      'group':  {'volume': (G,), ...},  # step or hits groups, or labels
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
    labels : tuple[str], 'all' or None
        Read ``labl`` and write these per-point labels on ``step``/``hits``.
        A panoptic level name (``'pysupera'`` — pysupera's own target, the
        reference —, ``'fragment'``, ``'particle'``, ``'interaction'``, see
        :data:`pimm_data._resolve.LEVELS`) stands for
        that level's class and instance keys. Otherwise keys, each
        ``(N, 1)``, named as pimm's PILArNet heads read them. The per-point
        labels (``'all'``): classes ``segment_motif`` (shower, track,
        michel, delta, le), ``segment_pid`` (photon, electron, muon, pion,
        proton, le); ids ``instance_fragment``, ``instance_particle`` (rows
        of ``particle``), ``instance_interaction`` (rows of
        ``interaction``). Object properties broadcast to points, only when
        named (their own values are ``particle`` / ``interaction`` table
        columns): ``segment_pdg``, ``is_primary``, ``momentum``, ``vertex``
        (``(N, 3)``). Which one is a training
        target is the pipeline's choice (``Collect`` keys). Each is a lookup
        over every point, so ask only for what is used. The tables
        (``group``, ``track``, ``particle``, ``interaction``) come with any
        of them. ``None``: no labl.
    le_is_stuff : bool
        Low-energy scatters are stuff, as in PILArNet, at each level by that
        level's own class: -1 in ``instance_fragment`` where the motif is le,
        in ``instance_particle`` where the pid is le; never in
        ``instance_interaction``.
    momentum : str
        The per-point ``momentum`` is ``|p|`` of the instance at creation
        (``'start'``, PILArNet's meaning) or at its first visible step
        (``'first'``).
    min_deposits : int
        Minimum 3D deposits per event (step reader filter).
    allow_partial : bool
        Missing or partial data raises by default: a shard absent from a
        loaded modality, a shard that does not open, an event present in
        some modalities and not others. True warns and reads what every
        modality has (a corpus mid-build). Event filters are never gaps.
    include_physics : bool
        Whether step reader loads dx, theta, phi, charge, photons, etc.
    transform : list or None
        Transform pipeline.
    loop, max_len, ignore_index : standard dataset parameters.
    """


    VALID_MODALITIES = ('step', 'sensor', 'hits')
    EXTRA_READERS = ('labl',)

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
        le_is_stuff=True,
        momentum='start',
        min_deposits=0,
        include_physics=True,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        allow_partial=False,
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
            labels = _resolve.expand(labels)
            if not {'step', 'hits'} & set(self._modalities):
                raise ValueError(
                    "labels= labels step or hits points; modalities="
                    f"{self._modalities} has neither.")
        elif momentum != 'start':
            raise ValueError("momentum= shapes the momentum label; it needs "
                             "labels=.")
        if momentum not in ('start', 'first'):
            raise ValueError(f"momentum={momentum!r}: 'start' or 'first'.")

        self._dataset_name = dataset_name
        self._volume = volume
        self._labels = labels
        self._le_is_stuff = le_is_stuff
        self._momentum = momentum
        self._min_deposits = min_deposits
        self._include_physics = include_physics
        self._max_len = max_len
        self._allow_partial = allow_partial
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
        fix = data.pop('_hit_fix', None)
        if labl is not None:
            for on in ('step', 'hits'):
                if on in data:
                    _resolve.point_refs(data, on, fix if on == 'hits' else None)
                    _resolve.resolve(data, on, self._labels,
                                     momentum=self._momentum,
                                     le_is_stuff=self._le_is_stuff)
                    _resolve.drop_point_refs(data[on])
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
        share, so it exists without labl too. Each deposit's group is read
        from step (``deposit_to_group``), else from labl's or hits' copy (files
        before step carried it); each hit entry's from hits' own group ids. Everything from
        the group up — its Geant4 track, the track's owning fragment, the
        particle and interaction tables — comes from labl alone.

        ``group`` — one row per JAXTPC group, event-wide (each volume's local
        ids are offset by the group counts of the volumes before it, as
        pysupera does, since a particle can cross volumes): ``volume``,
        ``track`` (ref -> ``track``, labl's visible tracks; -1 if the readout
        saw none of it) and ``fragment`` (ref -> ``particle``: the track's
        owner, gathered here so every walk starts ``group.fragment``). ``step.group`` / ``hits.group`` reference it;
        ``step.qs`` is the deposit's share of its group's step charge: step's
        ``qs_fractions`` when it carries them, else derived from step charge
        (the same to float16). ``particle`` and
        ``interaction`` are the labl tables with their row references
        declared. Hits only cross-check group counts."""
        # deposit -> group from step itself (where it belongs), popped so it
        # never reaches the step part as a column of its own
        step_d2g, step_ng, step_qs = {}, {}, None
        st0 = data.get('step')
        if st0 is not None:
            step_qs = st0.pop('qs_local', None)
        if st0 is not None and '_n_groups' in st0:
            step_ng = st0.pop('_n_groups')
            gl = st0.pop('group_local')
            if len(gl):
                vid0 = st0['volume_id'].ravel()
                step_d2g = {v: gl[vid0 == v] for v in np.unique(vid0).tolist()}
        if supera is None and hits_raw is None and not step_ng:
            return
        per_vol = {}       # v -> dict(n, track, ovr, d2g, inw)

        def _volume(v, n, what):
            """The volume's entry; group counts must agree across files."""
            d = per_vol.get(v)
            if d is None:
                if supera is not None:
                    # labl decides which volumes exist; a volume with groups
                    # that labl lacks would load silently unlabelled
                    if n and (self._volume is None or v == self._volume):
                        raise ValueError(
                            f"volume {v}: {n} groups in {what} but labl has "
                            "no such volume — the files are not the same event/task")
                    return None
                d = per_vol[v] = dict(n=n)
            elif d['n'] != n:
                raise ValueError(
                    f"volume {v}: {d['n']} groups in labl/hits, {n} in "
                    f"{what} — the files are not the same event/task")
            return d

        if supera is not None:
            n_part = len(supera.get('particle', {}).get('parent', ()))
            for k, vd in supera.items():
                if k.startswith('v'):
                    g2t_, gf_ = vd['group_to_track'], vd['group_fragment']
                    if len(gf_) != len(g2t_):
                        raise ValueError(f"labl {k}: group_fragment has {len(gf_)} "
                                         f"groups, group_to_track {len(g2t_)}")
                    if len(gf_) and (np.min(gf_) < -1 or np.max(gf_) >= n_part):
                        raise ValueError(f"labl {k}: group_fragment outside the "
                                         f"{n_part} particle rows")
                    per_vol[int(k[1:])] = dict(
                        n=len(vd['group_to_track']),
                        track=vd['group_to_track'],
                        gfrag=vd['group_fragment'],
                        planes={c[len('plane_'):]: p for c, p in vd.items()
                                if c.startswith('plane_')},
                        d2g=vd.get('deposit_to_group'),
                        inw=vd.get('deposit_in_window'))
        if hits_raw is not None:
            counts = {int(k.rsplit('_v', 1)[1]): int(n)
                      for k, n in hits_raw.items() if k.startswith('n_groups_v')}
            for k, g2t in hits_raw.items():     # older files: no attr
                if k.startswith('group_to_track_v'):
                    v = int(k.rsplit('_v', 1)[1])
                    if counts.setdefault(v, len(g2t)) != len(g2t):
                        raise ValueError(
                            f"volume {v}: hits n_groups {counts[v]}, "
                            f"group_to_track has {len(g2t)}")
            for v, n in counts.items():
                d = _volume(v, n, 'hits')
                if d is None:
                    continue
                hd = hits_raw.get(f'deposit_to_group_v{v}')
                if d.get('d2g') is None:
                    d['d2g'] = hd
                elif hd is not None and not np.array_equal(d['d2g'], hd):
                    raise ValueError(f"volume {v}: deposit_to_group in labl "
                                     f"differs from hits")
        for v, n in step_ng.items():
            d = _volume(v, n, 'step')
            if d is None:
                continue
            if v in step_d2g:
                old = d.get('d2g')
                if old is not None and not np.array_equal(old, step_d2g[v]):
                    raise ValueError(f"volume {v}: deposit_to_group in step "
                                     f"differs from labl/hits")
                d['d2g'] = step_d2g[v]
        # Every copy of the same JAXTPC array must agree: group_to_track in
        # hits (older files) and labl's; qs_fractions in step and
        # hits (to float16 ulps: the readout runs round apart).
        for v, d in per_vol.items():
            copies = [(w, np.asarray(a)) for w, a in (
                ('labl', d.get('track')),
                ('hits', None if hits_raw is None
                 else hits_raw.get(f'group_to_track_v{v}'))) if a is not None]
            for w, a in copies[1:]:
                if not np.array_equal(a, copies[0][1]):
                    raise ValueError(f"volume {v}: group_to_track in {w} "
                                     f"differs from {copies[0][0]}")
            hq = None if hits_raw is None else \
                hits_raw.get(f'qs_fractions_v{v}')
            if hq is not None and step_qs is not None:
                sq = step_qs.ravel()[st0['volume_id'].ravel() == v]
                if len(sq) != len(hq) or not np.allclose(
                        sq, hq, atol=2e-3, equal_nan=True):
                    raise ValueError(f"volume {v}: qs_fractions in step "
                                     f"differs from hits")
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
            # (Geant4 id -> visible-track row), and group -> fragment, which is
            # pysupera's own owner of the group (-1: pysupera left it unowned)
            trk = supera['track']
            tid = np.asarray(trk['id'], np.int64)
            if len(tid) > 1 and not (np.diff(tid) > 0).all():
                raise ValueError("labl track/id is not sorted and unique")
            g2t = cat('track', np.int64)
            row = np.full(len(g2t), -1, np.int32)
            if len(tid):
                pos = np.minimum(np.searchsorted(tid, g2t), len(tid) - 1)
                row = np.where(tid[pos] == g2t, pos, -1).astype(np.int32)
            group['track'] = row
            group['fragment'] = cat('gfrag', np.int32)
            group['_roles'].update(track=('ref', 'track'),
                                   fragment=('ref', 'particle'))
            data['track'] = dict(trk, _roles={})
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
                if not mask.any():
                    continue
                if d2g is None:
                    if per_vol[v]['n']:
                        raise ValueError(
                            f"volume {v}: {per_vol[v]['n']} groups but no "
                            "file gives the step deposits' deposit_to_group "
                            "(a step without groups needs the hits or labl "
                            "that carry it)")
                    continue
                if len(d2g) != int(mask.sum()):
                    raise ValueError(
                        f"volume {v}: {len(d2g)} deposit_to_group entries, "
                        f"{int(mask.sum())} step deposits")
                if len(d2g) and (np.min(d2g) < -1 or np.max(d2g) >= per_vol[v]['n']):
                    raise ValueError(
                        f"volume {v}: deposit_to_group outside its "
                        f"{per_vol[v]['n']} groups")
                ref[mask] = np.where(d2g >= 0, d2g + base[v], -1)
                if per_vol[v].get('inw') is not None:
                    inw[mask] = per_vol[v]['inw']
            st['group'] = ref
            if has_inw:
                # labl's: does the deposit's charge reach the readout window
                # (an unowned group is out of it, or under threshold)
                st['in_window'] = inw[:, None]
            if step_qs is not None and len(step_qs) == len(ref):
                # JAXTPC's own qs_fractions, carried by step
                st['qs'] = np.where(ref >= 0, step_qs.ravel(), np.nan) \
                    .astype(np.float32)[:, None]
            elif not len(ref):      # no deposits: same keys as any event
                st['qs'] = np.zeros((0, 1), np.float32)
            elif 'charge' in st:
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
            if hits_raw.get('_group_id_space', 'volume') == 'plane':
                # Each plane is numbered on its own: plane id k means nothing on
                # another plane or in step. Only labl's per-plane map
                # (plane_<P>/group) ties a plane id to its volume group; without
                # it, hits keep the plane-local id, which groups the entries of
                # one block within its plane and references nothing.
                if supera is None:
                    h['plane_group'] = (np.concatenate(hits_gid).astype(np.int32)
                                        if hits_gid else np.zeros(0, np.int32))
                    hits_gid = None
                else:
                    mapped = []
                    for plane, gid in zip(h['planes'], hits_gid):
                        v, pk = int(plane.split('_')[1]), plane.split('_', 2)[2]
                        if v not in base:
                            mapped.append(gid)
                            continue
                        m = per_vol[v].get('planes', {}).get(pk, {}).get('group')
                        if m is None:
                            raise ValueError(
                                f"{plane}: hits group ids are plane-local but "
                                "labl has no plane_<P>/group map for this plane")
                        m = np.asarray(m, np.int64)
                        ng = hits_raw.get(f'n_groups_plane:{plane}')
                        if ng is not None and len(m) != ng:
                            raise ValueError(
                                f"{plane}: labl's plane map has {len(m)} groups, "
                                f"the hits plane {ng}")
                        if len(m) and (m.min() < 0 or m.max() >= per_vol[v]['n']):
                            raise ValueError(
                                f"{plane}: labl's plane map points outside volume "
                                f"{v}'s {per_vol[v]['n']} groups")
                        if len(gid) and (gid.max() >= len(m) or gid.min() < 0):
                            raise ValueError(
                                f"{plane}: plane group id out of range of labl's "
                                f"map ({len(m)} groups)")
                        mapped.append(m[gid])
                    hits_gid = mapped
            elif supera is not None and any(
                    'group' in c for d in per_vol.values()
                    for c in d.get('planes', {}).values()):
                raise ValueError(
                    "labl has plane_<P>/group maps (plane-local hits) but the "
                    "hits file's group_id_space is 'volume' — not the same layout")
            if hits_gid is not None:
                refs = []
                for plane, gid in zip(h['planes'], hits_gid):
                    v = int(plane.split('_')[1])
                    refs.append(np.where(gid >= 0, gid + base[v], -1)
                                if v in base else np.full(len(gid), -1))
                h['group'] = (np.concatenate(refs).astype(np.int32) if refs
                              else np.zeros(0, np.int32))
                h.setdefault('_roles', {})['group'] = ('ref', 'group')
            if supera is not None:
                # pysupera's per-hit labels where they are not the group's:
                # off-sensor hits get none, split groups' hits their own
                fix, o = [], 0
                for plane, gid in zip(h['planes'], hits_gid):
                    v, pk = int(plane.split('_')[1]), plane.split('_', 2)[2]
                    c = per_vol.get(v, {}).get('planes', {}).get(pk)
                    if c:
                        if 'on_sensor' in c and len(c['on_sensor']) != len(gid):
                            raise ValueError(f"{plane}: labl on_sensor has "
                                             f"{len(c['on_sensor'])} hits, "
                                             f"hits {len(gid)}")
                        if 'override_index' in c:
                            oi = np.asarray(c['override_index'])
                            if not (len(oi) == len(c.get('override_fragment', ()))
                                    == len(c.get('override_instance', ()))) or (
                                    len(oi) and (oi.min() < 0 or oi.max() >= len(gid))):
                                raise ValueError(
                                    f"{plane}: labl overrides inconsistent with the "
                                    f"plane's {len(gid)} hits")
                        fix.append((o, len(gid), c))
                    o += len(gid)
                data['_hit_fix'] = fix

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
