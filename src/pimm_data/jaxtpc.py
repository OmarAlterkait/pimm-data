"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:
seg (3D deposits), resp (2D wire signals), corr (3D->2D correspondence),
labl (track_id->label lookup tables).

Inherits from :class:`pimm_data.DefaultDataset` so transforms, test_mode
fragment lists, loop, and max_len work out of the box and the class is
registered in the :data:`pimm_data.DATASETS` registry for config-driven
construction (``dict(type="JAXTPCDataset", ...)``).
"""

import os
import logging
from copy import deepcopy

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from .readers.jaxtpc_seg import JAXTPCSegReader
from .readers.jaxtpc_resp import JAXTPCRespReader
from .readers.jaxtpc_labl import JAXTPCLablReader
from .readers.jaxtpc_corr import JAXTPCCorrReader

log = logging.getLogger(__name__)


@DATASETS.register_module()
class JAXTPCDataset(DefaultDataset):
    """LArTPC multimodal dataset.

    Parameters
    ----------
    data_root : str
        Root directory with seg/, resp/, corr/, labl/ subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Which to load: 'seg', 'resp', 'labl', 'corr'.
    dataset_name : str
        File prefix (e.g., 'sim' for 'sim_seg_0000.h5').
    volume : int or None
        Load only this volume index. None = all volumes.
    label_key : str
        Which label to use as 'segment': 'particle', 'cluster', 'interaction'.
    min_deposits : int
        Minimum 3D deposits per event (seg reader filter).
    include_physics : bool
        Whether seg reader loads dx, theta, phi, charge, photons, etc.
    label_keys : list or None
        Which label datasets to load from labl files.
    transform : list or None
        Transform pipeline (dict-based configs and/or raw callables).
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
        label_key='particle',
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
        self.resp_reader = None
        self.labl_reader = None
        self.corr_reader = None

        planes = 'all'
        if volume is not None:
            planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                      f'volume_{volume}_Y']

        if 'seg' in self._modalities:
            self.seg_reader = JAXTPCSegReader(
                data_root=self._modality_root('seg'), split=split,
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        if 'resp' in self._modalities:
            self.resp_reader = JAXTPCRespReader(
                data_root=self._modality_root('resp'), split=split,
                dataset_name=dataset_name, planes=planes)

        if 'labl' in self._modalities:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name, label_keys=label_keys)

        if 'corr' in self._modalities:
            self.corr_reader = JAXTPCCorrReader(
                data_root=self._modality_root('corr'), split=split,
                dataset_name=dataset_name, planes=planes)

        active_readers = [r for r in (self.seg_reader, self.resp_reader,
                                       self.labl_reader, self.corr_reader)
                          if r is not None]
        if not active_readers:
            raise ValueError(f"Need at least one modality, got {self._modalities}")
        self._canonical_reader = (self.seg_reader or self.resp_reader
                                  or self.corr_reader or self.labl_reader)
        self._n_events = min(len(r) for r in active_readers)

        if (self.resp_reader and self.labl_reader
                and not self.corr_reader and not self.seg_reader):
            log.warning(
                "modalities=('resp','labl') without 'corr': labl provides "
                "track_id->label tables but resp pixels can't be mapped to "
                "track_ids without corr. No 'segment' will be produced.")

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

    def _modality_root(self, modality):
        """Resolve root directory for a modality."""
        mod_dir = os.path.join(self._source_data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        return self._source_data_root

    def get_data_list(self):
        """Range of event indices, optionally capped by max_len."""
        n = getattr(self, '_n_events', 0)
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            n = min(n, max_len)
        return list(range(n))

    def get_data(self, idx):
        """Load one event. Who owns coord depends on modalities:

        - seg present: coord = 3D deposits. Resp/corr as namespaced keys.
        - seg absent, corr+labl present: coord = 2D corr entries with labels.
        - seg absent, resp present (no corr): coord = 2D resp merged.
        """
        real_idx = idx % len(self.data_list)
        data_dict = {}

        if self.seg_reader is not None:
            data_dict.update(self.seg_reader.read_event(real_idx))

        labl_data = {}
        if self.labl_reader is not None:
            labl_data = self.labl_reader.read_event(real_idx)

        if self.seg_reader is not None and labl_data:
            self._apply_labl_to_3d(data_dict, labl_data)

        resp_data = {}
        if self.resp_reader is not None:
            resp_data = self.resp_reader.read_event(real_idx)

        corr_data = {}
        if self.corr_reader is not None:
            corr_data = self.corr_reader.read_event(real_idx)

        has_seg = self.seg_reader is not None
        has_resp = bool(resp_data)
        has_corr = bool(corr_data)

        if has_resp:
            self._merge_resp_planes(data_dict, resp_data, prefix='resp_')
            data_dict.update(resp_data)

        if has_corr and labl_data:
            self._build_corr_pointcloud(data_dict, corr_data, labl_data, prefix='corr_')
        elif has_corr:
            data_dict.update(corr_data)

        if has_seg:
            pass
        elif has_corr and labl_data:
            data_dict['coord'] = data_dict['corr_coord']
            data_dict['energy'] = data_dict['corr_energy']
            data_dict['segment'] = data_dict['corr_segment']
            data_dict['instance'] = data_dict['corr_instance']
            data_dict['plane_id'] = data_dict['corr_plane_id']
        elif has_resp:
            data_dict['coord'] = data_dict['resp_coord']
            data_dict['energy'] = data_dict['resp_energy']
            data_dict['plane_id'] = data_dict['resp_plane_id']

        if labl_data:
            for k, v in labl_data.items():
                if k not in data_dict:
                    data_dict[k] = v

        data_dict['name'] = self.get_data_name(real_idx)
        data_dict['split'] = self.split if isinstance(self.split, str) else 'custom'
        return data_dict

    def _apply_labl_to_3d(self, data_dict, labl_data):
        """Map 3D deposits' track_ids to labels via labl lookup. Vectorized."""
        track_ids = data_dict.get('track_ids')
        volume_ids = data_dict.get('volume_id')
        if track_ids is None:
            return

        n = len(track_ids)
        labels = np.full(n, -1, dtype=np.int32)

        vol_indices = sorted(set(
            k.split('_')[1] for k in labl_data
            if k.startswith('labl_v') and k.endswith('_track_ids')
        ))

        for vi in vol_indices:
            tids_key = f'labl_{vi}_track_ids'
            label_key = f'labl_{vi}_{self._label_key}'
            if tids_key not in labl_data or label_key not in labl_data:
                continue

            vol_tids = labl_data[tids_key]
            vol_labels = labl_data[label_key]
            vol_num = int(vi[1:])

            if volume_ids is not None:
                vol_mask = volume_ids.ravel() == vol_num
            else:
                vol_mask = np.ones(n, dtype=bool)

            sort_idx = np.argsort(vol_tids)
            sorted_tids = vol_tids[sort_idx]
            sorted_labels = vol_labels[sort_idx]

            deposit_tids = track_ids[vol_mask]
            insert_pos = np.searchsorted(sorted_tids, deposit_tids)
            insert_pos = np.clip(insert_pos, 0, len(sorted_tids) - 1)
            matched = sorted_tids[insert_pos] == deposit_tids
            labels[vol_mask] = np.where(matched, sorted_labels[insert_pos], -1)

        data_dict['segment'] = labels

    def _merge_resp_planes(self, data_dict, resp_data, prefix=''):
        """Merge all planes into {prefix}coord (M,2), {prefix}energy (M,1), {prefix}plane_id (M,1)."""
        planes = sorted(set(
            k.split('.')[1] for k in resp_data if k.endswith('.wire')
        ))

        all_coord, all_energy, all_plane_id = [], [], []
        for i, plane in enumerate(planes):
            wire = resp_data[f'plane.{plane}.wire']
            time = resp_data[f'plane.{plane}.time']
            value = resp_data[f'plane.{plane}.value']
            n = len(wire)
            all_coord.append(np.stack([wire, time], axis=1).astype(np.float32))
            all_energy.append(value[:, None].astype(np.float32))
            all_plane_id.append(np.full((n, 1), i, dtype=np.int32))

        data_dict[f'{prefix}coord'] = np.concatenate(all_coord, axis=0)
        data_dict[f'{prefix}energy'] = np.concatenate(all_energy, axis=0)
        data_dict[f'{prefix}plane_id'] = np.concatenate(all_plane_id, axis=0)

    def _build_corr_pointcloud(self, data_dict, corr_data, labl_data, prefix=''):
        """Build 2D labeled point cloud from corr + labl.

        Each corr entry is a point: coord=(wire,time), feature=charge,
        instance=group_id, segment from g2t+labl chain.
        Overlapping instances at the same pixel are separate points.
        """
        planes = sorted(set(
            k.split('.')[1] for k in corr_data if k.endswith('.wire')
        ))

        all_coord, all_charge, all_gid, all_segment, all_plane_id = [], [], [], [], []

        for pi, plane in enumerate(planes):
            wire_key = f'corr.{plane}.wire'
            if wire_key not in corr_data:
                continue

            wire = corr_data[f'corr.{plane}.wire']
            time = corr_data[f'corr.{plane}.time']
            gid = corr_data[f'corr.{plane}.group_id']
            charge = corr_data[f'corr.{plane}.charge']
            n = len(wire)

            all_coord.append(np.stack([wire, time], axis=1).astype(np.float32))
            all_charge.append(charge[:, None].astype(np.float32))
            all_gid.append(gid.astype(np.int32))
            all_plane_id.append(np.full((n, 1), pi, dtype=np.int32))

            vol_idx = plane.split('_')[1]
            g2t = corr_data.get(f'g2t_v{vol_idx}')

            labels = np.full(n, -1, dtype=np.int32)
            if g2t is not None:
                valid_gid = (gid >= 0) & (gid < len(g2t))
                track_ids = np.where(valid_gid, g2t[gid], -1)

                tids_key = f'labl_v{vol_idx}_track_ids'
                lbl_key = f'labl_v{vol_idx}_{self._label_key}'
                if tids_key in labl_data and lbl_key in labl_data:
                    labl_tids = labl_data[tids_key]
                    labl_vals = labl_data[lbl_key]
                    sort_idx = np.argsort(labl_tids)
                    sorted_tids = labl_tids[sort_idx]
                    sorted_vals = labl_vals[sort_idx]
                    insert_pos = np.searchsorted(sorted_tids, track_ids)
                    insert_pos = np.clip(insert_pos, 0, len(sorted_tids) - 1)
                    matched = sorted_tids[insert_pos] == track_ids
                    labels[matched] = sorted_vals[insert_pos[matched]]

            all_segment.append(labels)

        if not all_coord:
            return

        data_dict[f'{prefix}coord'] = np.concatenate(all_coord, axis=0)
        data_dict[f'{prefix}energy'] = np.concatenate(all_charge, axis=0)
        data_dict[f'{prefix}instance'] = np.concatenate(all_gid, axis=0)
        data_dict[f'{prefix}segment'] = np.concatenate(all_segment, axis=0)
        data_dict[f'{prefix}plane_id'] = np.concatenate(all_plane_id, axis=0)

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1])
                       if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_test_data(self, idx):
        """Test-time data prep. More lenient than DefaultDataset: ``segment``
        is only copied into result_dict when present (unlabeled modes
        e.g. resp-only don't produce a segment key).
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
        for attr in ('seg_reader', 'resp_reader', 'labl_reader', 'corr_reader'):
            reader = getattr(self, attr, None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
