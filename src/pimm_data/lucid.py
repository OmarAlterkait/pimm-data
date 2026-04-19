"""
LUCiDDataset — dataset for Water Cherenkov detector simulation output.

Loads PMT sensor data and/or 3D track segments. Inherits from
:class:`pimm_data.DefaultDataset` so transforms, test_mode, loop and
max_len work out of the box.
"""

import os
import logging
from copy import deepcopy

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from .readers.lucid_seg import LUCiDSegReader
from .readers.lucid_sensor import LUCiDSensorReader

log = logging.getLogger(__name__)


@DATASETS.register_module()
class LUCiDDataset(DefaultDataset):
    """Water Cherenkov detector dataset.

    Parameters
    ----------
    data_root : str
        Root directory with seg/ and/or sensor/ subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Which to load: 'seg', 'sensor'.
    dataset_name : str
        File prefix (e.g., 'wc' for 'wc_seg_0000.h5').
    output_mode : str
        How to format sensor data for the model:
        - 'response': PMT point cloud with total PE/T features
        - 'labels': sparse per-particle entries with instance/semantic labels
        - 'separate': keep raw reader keys (pmt_coord, pmt_pe, pp_* keys)
    include_labels : bool
        Whether sensor reader loads per-particle decomposition.
    pe_threshold : float
        Minimum PE for sparsifying PE_per_particle.
    min_segments : int
        Minimum segments per event (seg reader filter).
    pmt_positions : ndarray or None
        Optional PMT positions override for LUCiDSensorReader.
    pmt_positions_file : str or None
        Optional .npy path with PMT positions.
    transform, test_mode, test_cfg, loop, max_len, ignore_index, cache :
        Standard :class:`DefaultDataset` parameters.
    """

    def __init__(
        self,
        data_root,
        split='',
        modalities=('sensor',),
        dataset_name='wc',
        output_mode='response',
        include_labels=True,
        pe_threshold=0.0,
        min_segments=0,
        pmt_positions=None,
        pmt_positions_file=None,
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
        self._output_mode = output_mode
        self._max_len = max_len
        self._source_data_root = data_root
        self._source_split = split

        self.seg_reader = None
        self.sensor_reader = None

        if 'seg' in self._modalities:
            seg_root = self._modality_root('seg')
            self.seg_reader = LUCiDSegReader(
                data_root=seg_root, split=split,
                dataset_name=dataset_name, min_segments=min_segments)

        if 'sensor' in self._modalities:
            sensor_root = self._modality_root('sensor')
            self.sensor_reader = LUCiDSensorReader(
                data_root=sensor_root, split=split,
                dataset_name=dataset_name,
                include_labels=include_labels,
                pe_threshold=pe_threshold,
                pmt_positions=pmt_positions,
                pmt_positions_file=pmt_positions_file)

        active_readers = [r for r in (self.seg_reader, self.sensor_reader)
                          if r is not None]
        if not active_readers:
            raise ValueError(f"Need 'seg' or 'sensor' in modalities, got {self._modalities}")
        self._canonical_reader = active_readers[0]
        self._n_events = min(len(r) for r in active_readers)

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

    def _modality_root(self, modality):
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
        real_idx = idx % len(self.data_list)
        data_dict = {}

        if self.seg_reader is not None:
            seg_data = self.seg_reader.read_event(real_idx)
            if self.sensor_reader is not None and self._output_mode == 'separate':
                for k, v in seg_data.items():
                    data_dict[f'seg3d.{k}'] = v
            else:
                data_dict.update(seg_data)

        if self.sensor_reader is not None:
            sensor_data = self.sensor_reader.read_event(real_idx)

            if self._output_mode == 'response':
                n = len(sensor_data['pmt_pe'])
                if 'pmt_coord' in sensor_data:
                    data_dict['coord'] = sensor_data['pmt_coord']
                else:
                    data_dict['coord'] = np.arange(n, dtype=np.float32)[:, None]
                data_dict['energy'] = sensor_data['pmt_pe'][:, None]
                data_dict['time'] = sensor_data['pmt_t'][:, None]

            elif self._output_mode == 'labels':
                if 'pp_sensor_idx' in sensor_data:
                    sidx = sensor_data['pp_sensor_idx']
                    if 'pmt_coord' in sensor_data:
                        data_dict['coord'] = sensor_data['pmt_coord'][sidx]
                    else:
                        data_dict['coord'] = sidx.astype(np.float32)[:, None]
                    data_dict['energy'] = sensor_data['pp_pe'][:, None]
                    data_dict['segment'] = sensor_data['pp_category']
                    data_dict['instance'] = sensor_data['pp_particle_idx']
                    if 'pp_t' in sensor_data:
                        data_dict['time'] = sensor_data['pp_t'][:, None]

            elif self._output_mode == 'separate':
                data_dict.update(sensor_data)

        data_dict['name'] = self.get_data_name(real_idx)
        data_dict['split'] = self.split if isinstance(self.split, str) else 'custom'
        return data_dict

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_test_data(self, idx):
        """More lenient than DefaultDataset: ``segment`` is only copied into
        result_dict when present (response mode has no labels)."""
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        if "segment" in data_dict:
            result_dict["segment"] = data_dict.pop("segment")

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
        for reader in (getattr(self, 'seg_reader', None),
                       getattr(self, 'sensor_reader', None)):
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
