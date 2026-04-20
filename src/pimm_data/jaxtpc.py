"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:

* ``seg/`` — 3D truth deposits
* ``sensor/`` — raw sparse wire / pixel readout
* ``inst/`` — per-instance sensor decomposition
* ``labl/`` — track_id → label lookup tables

Modality strings follow ``docs/DATASET_DESIGN.md``: ``'seg'``, ``'sensor'``,
``'inst'``, ``'labl'``.

Returns a **nested** dict: each loaded modality owns a sub-dict with clean
unprefixed keys::

    {
      'seg':    {'coord': (N,3), 'energy': (N,1), 'volume_id': ..., ...},
      'sensor': {'coord': (M,2), 'energy': (M,1), 'plane_id': ...,
                 'raw': {plane_label: {'wire', 'time', 'value'}}},
      'inst':   {'coord': (E,2), 'energy': (E,1), 'instance': ..., ...,
                 'raw': {plane_label: {'wire', 'time', 'group_id', 'charge'}}},
      'labl':   {'v0': {'track_ids': (T,), 'track_pdg': (T,),
                        'segment_to_track': (N_v,), ...}, 'v1': {...}},
      'bridges':{'group_to_track_v0': (G,), 'segment_to_group_v0': (N_v,),
                 'qs_fractions_v0': ..., ...},
      'name': str, 'split': str,
    }

Missing modalities have no top-level key. There is no bare ``coord`` / no
precedence / no prefixed aliases — transforms pick a stream explicitly (see
``ApplyToStream`` and ``Collect(stream=...)``).

Registered in :data:`pimm_data.DATASETS` for config-driven construction
via ``dict(type="JAXTPCDataset", ...)``.
"""

import os
import logging
from copy import deepcopy

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from .readers.jaxtpc_seg import JAXTPCSegReader
from .readers.jaxtpc_sensor import JAXTPCSensorReader
from .readers.jaxtpc_labl import JAXTPCLablReader
from .readers.jaxtpc_inst import JAXTPCInstReader

log = logging.getLogger(__name__)


@DATASETS.register_module()
class JAXTPCDataset(DefaultDataset):
    """LArTPC multimodal dataset with nested per-stream output.

    Parameters
    ----------
    data_root : str
        Root directory with ``seg/``, ``sensor/``, ``inst/``, ``labl/``
        subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Any subset of ``'seg'``, ``'sensor'``, ``'inst'``, ``'labl'``.
        ``('labl',)`` and ``('sensor', 'labl')`` are invalid (see
        ``docs/DATASET_DESIGN.md#invalid-combinations``).
    dataset_name : str
        File prefix (e.g., ``'sim'`` for ``sim_seg_0000.h5``).
    volume : int or None
        Load only this volume index. ``None`` = all volumes.
    label_key : str
        Which labl column to decorate the point clouds with. Must match a
        column in the labl files (``'pdg'``, ``'cluster'``, ``'interaction'``,
        ``'ancestor'``). Raw values from ``track_{label_key}`` are broadcast
        to each deposit / pixel entry; use a downstream ``RemapSegment`` to
        map raw values to task-specific class indices.
    min_deposits : int
        Minimum 3D deposits per event (seg reader filter).
    include_physics : bool
        Whether seg reader loads dx, theta, phi, charge, photons, etc.
    label_keys : list or None
        Which label datasets to load from labl files (None → all).
    transform : list or None
        Transform pipeline.
    test_mode, test_cfg, loop, max_len, ignore_index, cache : standard
        :class:`DefaultDataset` parameters.
    """


    def __init__(
        self,
        data_root,
        split='train',
        modalities=('seg',),
        dataset_name='sim',
        volume=None,
        label_key='pdg',
        min_deposits=0,
        include_physics=True,
        label_keys=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        cache=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)

        self._dataset_name = dataset_name
        self._volume = volume
        self._label_key = label_key
        self._min_deposits = min_deposits
        self._include_physics = include_physics
        self._label_keys = label_keys
        self._max_len = max_len
        self._source_data_root = data_root
        self._source_split = split

        self.seg_reader = None
        self.sensor_reader = None
        self.labl_reader = None
        self.inst_reader = None

        planes = 'all'
        if volume is not None:
            planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                      f'volume_{volume}_Y']

        if 'seg' in self._modalities:
            self.seg_reader = JAXTPCSegReader(
                data_root=self._modality_root('seg'), split=split,
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        if 'sensor' in self._modalities:
            self.sensor_reader = JAXTPCSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name, planes=planes)

        if 'labl' in self._modalities:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name, label_keys=label_keys)

        if 'inst' in self._modalities:
            self.inst_reader = JAXTPCInstReader(
                data_root=self._modality_root('inst'), split=split,
                dataset_name=dataset_name, planes=planes)

        active_readers = [r for r in (self.seg_reader, self.sensor_reader,
                                       self.labl_reader, self.inst_reader)
                          if r is not None]
        self._canonical_reader = (self.seg_reader or self.sensor_reader
                                  or self.inst_reader or self.labl_reader)
        self._n_events = min(len(r) for r in active_readers)

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

    @staticmethod
    def _validate_modalities(modalities):
        mods = set(modalities)
        if not mods:
            raise ValueError("modalities is empty; must load at least one")
        unknown = mods - {'seg', 'sensor', 'inst', 'labl'}
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; valid: "
                "'seg', 'sensor', 'inst', 'labl'")
        if mods == {'labl'}:
            raise ValueError(
                "Invalid modality combination ('labl',): labl is a "
                "dimension table and requires an instance-bearing modality "
                "('seg' or 'inst') to join against. "
                "See docs/DATASET_DESIGN.md#invalid-combinations.")
        if mods == {'sensor', 'labl'}:
            raise ValueError(
                "Invalid modality combination ('sensor', 'labl'): sensor has "
                "no instance separation, so labl cannot be attached. Add "
                "'inst' or 'seg' to the modalities tuple. "
                "See docs/DATASET_DESIGN.md#invalid-combinations.")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self._source_data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        return self._source_data_root

    def get_data_list(self):
        n = getattr(self, '_n_events', 0)
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            n = min(n, max_len)
        return list(range(n))

    def get_data(self, idx):
        """Load one event as a nested dict (schema: see module docstring)."""
        real_idx = idx % len(self.data_list)

        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }

        labl_by_volume = {}
        if self.labl_reader is not None:
            labl_by_volume = self._build_labl(self.labl_reader.read_event(real_idx))
            if self._volume is not None:
                # Drop labl volumes the user isn't loading (keeps the
                # dataset's view consistent with ``volume=`` on other readers).
                keep = f'v{self._volume}'
                labl_by_volume = {k: v for k, v in labl_by_volume.items()
                                  if k == keep}
            if labl_by_volume:
                data['labl'] = labl_by_volume

        if self.inst_reader is not None:
            inst_raw = self.inst_reader.read_event(real_idx)
            data['inst'] = self._build_inst_cloud(inst_raw, labl_by_volume)
            bridges = self._build_bridges(inst_raw)
            if bridges:
                data['bridges'] = bridges

        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor_cloud(
                self.sensor_reader.read_event(real_idx))

        if self.seg_reader is not None:
            data['seg'] = self._build_seg_cloud(
                self.seg_reader.read_event(real_idx), labl_by_volume)

        return data

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------

    def _build_seg_cloud(self, seg_raw, labl_by_volume):
        """3D deposit sub-dict; decorates with segment/instance if labl present."""
        sub = {}
        for k, v in seg_raw.items():
            sub[k] = v  # coord, energy, volume_id, physics — readers emit bare

        if labl_by_volume and 'volume_id' in sub:
            segment, instance = self._decorate_seg_from_labl(
                sub['volume_id'], labl_by_volume)
            sub['segment'] = segment
            sub['instance'] = instance

        return sub

    def _build_sensor_cloud(self, sensor_raw):
        """Merge per-plane sensor raw into a 2D point cloud + raw passthrough."""
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            sensor_raw, prefix='sensor', value_key='value')
        return {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'planes': planes, 'raw': raw,
        }

    def _build_inst_cloud(self, inst_raw, labl_by_volume):
        """Merge per-plane inst raw into a 2D point cloud + raw passthrough.

        Attaches ``segment`` when labl available (via group_to_track chain).
        ``instance`` is always attached (== group_id).
        """
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            inst_raw, prefix='inst', value_key='charge',
            extra_keys=('group_id',))
        # instance = per-entry group_id
        instance = np.concatenate(
            [raw[p]['group_id'] for p in planes], axis=0
        ).astype(np.int32) if planes else np.zeros(0, dtype=np.int32)

        sub = {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'instance': instance, 'planes': planes, 'raw': raw,
        }
        if labl_by_volume:
            sub['segment'] = self._decorate_inst_from_labl(
                planes, raw, inst_raw, labl_by_volume)
        return sub

    def _build_labl(self, labl_flat):
        """Convert flat labl_v{N}_col keys into nested {v{N}: {col: arr}}."""
        by_volume = {}
        for k, v in labl_flat.items():
            # Key format: labl_v{idx}_{col}; col may contain underscores
            assert k.startswith('labl_v'), k
            rest = k[len('labl_v'):]
            # Split on first underscore after idx
            idx_end = 0
            while idx_end < len(rest) and rest[idx_end].isdigit():
                idx_end += 1
            vid = 'v' + rest[:idx_end]
            col = rest[idx_end + 1:]  # skip the separator underscore
            by_volume.setdefault(vid, {})[col] = v
        return by_volume

    def _build_bridges(self, inst_raw):
        """Extract per-volume bridge arrays (g2t, segment_to_group, qs_fractions)."""
        bridges = {}
        for k, v in inst_raw.items():
            if (k.startswith('group_to_track_v')
                    or k.startswith('segment_to_group_v')
                    or k.startswith('qs_fractions_v')):
                bridges[k] = v
        return bridges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_plane_dotted(raw_dict, prefix, value_key, extra_keys=()):
        """Merge `{prefix}.{plane}.{col}` flat keys into a single point cloud.

        Returns (planes, coord, energy, plane_id, raw_nested).
        raw_nested is ``{plane_label: {'wire', 'time', value_key, *extra_keys}}``.
        """
        planes = sorted(set(
            k.split('.')[1] for k in raw_dict
            if k.startswith(prefix + '.') and k.endswith('.wire')
        ))
        all_coord, all_val, all_plane_id = [], [], []
        raw_nested = {}
        for i, plane in enumerate(planes):
            wire = raw_dict[f'{prefix}.{plane}.wire']
            time = raw_dict[f'{prefix}.{plane}.time']
            value = raw_dict[f'{prefix}.{plane}.{value_key}']
            n = len(wire)
            all_coord.append(np.stack([wire, time], axis=1).astype(np.float32))
            all_val.append(value[:, None].astype(np.float32))
            all_plane_id.append(np.full((n, 1), i, dtype=np.int32))
            cols = {'wire': wire, 'time': time, value_key: value}
            for ek in extra_keys:
                cols[ek] = raw_dict[f'{prefix}.{plane}.{ek}']
            raw_nested[plane] = cols

        if planes:
            coord = np.concatenate(all_coord, axis=0)
            energy = np.concatenate(all_val, axis=0)
            plane_id = np.concatenate(all_plane_id, axis=0)
        else:
            coord = np.zeros((0, 2), dtype=np.float32)
            energy = np.zeros((0, 1), dtype=np.float32)
            plane_id = np.zeros((0, 1), dtype=np.int32)

        return planes, coord, energy, plane_id, raw_nested

    def _decorate_seg_from_labl(self, volume_id, labl_by_volume):
        """Broadcast per-track labl data onto each seg deposit.

        Uses ``labl[vN]['segment_to_track']`` (row-aligned to the volume's
        seg deposits) as the per-deposit FK, then looks up
        ``labl[vN]['track_{label_key}']`` via binary search on ``track_ids``.
        """
        vid_flat = volume_id.ravel()
        n_total = vid_flat.shape[0]
        instance = np.full(n_total, -1, dtype=np.int32)
        segment = np.full(n_total, -1, dtype=np.int32)
        meta_col = f'track_{self._label_key}'

        for vkey, vdata in labl_by_volume.items():
            vol_num = int(vkey[1:])
            mask = vid_flat == vol_num
            if not mask.any():
                continue
            if 'segment_to_track' not in vdata:
                continue
            per_dep_tid = vdata['segment_to_track'].astype(np.int32)
            n_vol = int(mask.sum())
            if per_dep_tid.shape[0] != n_vol:
                log.warning("labl.%s.segment_to_track len %d != seg vol %d len %d",
                            vkey, per_dep_tid.shape[0], vol_num, n_vol)
                continue
            instance[mask] = per_dep_tid

            if 'track_ids' in vdata and meta_col in vdata:
                tids = vdata['track_ids']
                vals = vdata[meta_col]
                order = np.argsort(tids)
                s_tids = tids[order]
                s_vals = vals[order]
                pos = np.searchsorted(s_tids, per_dep_tid)
                pos = np.clip(pos, 0, len(s_tids) - 1)
                matched = s_tids[pos] == per_dep_tid
                segment[mask] = np.where(matched, s_vals[pos], -1)

        return segment, instance

    def _decorate_inst_from_labl(self, planes, raw_nested, inst_flat,
                                 labl_by_volume):
        """Per-inst-entry segment label via group_to_track → track lookup."""
        meta_col = f'track_{self._label_key}'
        all_labels = []
        for plane in planes:
            cols = raw_nested[plane]
            gid = cols['group_id']
            # plane label is 'volume_{v}_{U|V|Y}' — extract volume index
            vol_idx_str = plane.split('_')[1]
            vkey = f'v{vol_idx_str}'

            n = gid.shape[0]
            labels = np.full(n, -1, dtype=np.int32)

            g2t_key = f'group_to_track_v{vol_idx_str}'
            g2t = inst_flat.get(g2t_key)
            if g2t is None or vkey not in labl_by_volume:
                all_labels.append(labels)
                continue
            vdata = labl_by_volume[vkey]
            if 'track_ids' not in vdata or meta_col not in vdata:
                all_labels.append(labels)
                continue

            valid = (gid >= 0) & (gid < len(g2t))
            tids = np.where(valid, g2t[gid], -1)
            labl_tids = vdata['track_ids']
            labl_vals = vdata[meta_col]
            order = np.argsort(labl_tids)
            s_tids = labl_tids[order]
            s_vals = labl_vals[order]
            pos = np.searchsorted(s_tids, tids)
            pos = np.clip(pos, 0, len(s_tids) - 1)
            matched = s_tids[pos] == tids
            labels[matched] = s_vals[pos[matched]]
            all_labels.append(labels)

        if not all_labels:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(all_labels, axis=0)

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1])
                       if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_test_data(self, idx):
        """Test-time data prep.

        Expects ``segment`` to be produced at the top level by a terminal
        :class:`Collect` transform (e.g. ``Collect(stream='seg', ...)``).
        """
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        if "segment" in data_dict:
            result_dict["segment"] = data_dict.pop("segment")
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = [aug(deepcopy(data_dict)) for aug in self.aug_transform]
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part
        fragment_list = [self.post_transform(f) for f in fragment_list]
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __del__(self):
        for attr in ('seg_reader', 'sensor_reader', 'labl_reader',
                     'inst_reader'):
            reader = getattr(self, attr, None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
