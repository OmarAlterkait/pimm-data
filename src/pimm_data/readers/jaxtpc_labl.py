"""
JAXTPCLablReader — reads per-volume track_id → label lookup tables.

The labl file stores a mapping from track_id to labels (particle, cluster,
interaction) per volume. This is used for:
  - 3D tasks: deposit's track_id → look up label directly
  - 2D tasks: pixel → group_id → group_to_track → track_id → look up label

Output: a dict of numpy arrays per volume, keyed ``labl_v{N}_{col}`` where
``col`` is each dataset present in the labl file, e.g.:
    labl_v0_track_ids:        (T,) int32  — unique track IDs (primary key)
    labl_v0_track_pdg:        (T,) int32  — PDG code per track
    labl_v0_track_cluster:    (T,) int32  — cluster ID per track
    labl_v0_track_interaction:(T,) int32  — interaction ID per track
    labl_v0_track_ancestor:   (T,) int32  — ancestor track per track
    labl_v0_deposit_to_track: (N_v,) int32 — per-deposit FK into track_ids
(Exact columns depend on the labl writer / the reader's ``label_keys``.)

A ``supera-labl`` file (``/config.attrs['labl_format']``, written by
:mod:`pimm_data.supera_labl`) is read whole instead, as nested tables::

    {'particle': {col: (P,…)}, 'interaction': {col: (I,…)},
     'track': {'id': (T,), 'fragment': (T,)},
     'v{N}': {'group_track': (G_v,), 'deposit_to_group': (N_v,)}}

``reader.kind`` says which: ``'supera'`` or ``'track'``.
"""

import numpy as np
import h5py

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase


class JAXTPCLablReader(ShardReaderBase):
    """Reads per-volume track_id → label lookup tables.

    Parameters
    ----------
    data_root : str
        Directory containing labl shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_labl_0000.h5').
    label_keys : list of str or None
        Which label datasets to load (default: all available).
    """

    _MODALITY = 'labl'

    def __init__(self, data_root, split='train', dataset_name='sim',
                 label_keys=None, files=None):
        self.data_root = data_root
        self.files = files
        self.split = split
        self.dataset_name = dataset_name
        self.label_keys = label_keys
        self._init_shards()
        fmt = str(read_shard_meta(self.h5_files[0])['config_attrs'].get(
            'labl_format', ''))
        self.kind = 'supera' if fmt.startswith('supera-labl') else 'track'

    def read_event(self, idx):
        """Read one event's per-volume label lookup tables.

        Returns dict with keys like:
            labl_v0_track_ids:   (T,) int32
            labl_v0_particle:    (T,) int32
            labl_v0_cluster:     (T,) int32
            labl_v0_interaction: (T,) int32
        """
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        if self.kind == 'supera':
            return self._read_supera(evt)

        data_dict = {}
        for vol_key in evt:
            vol = evt[vol_key]
            if not isinstance(vol, h5py.Group):
                continue
            if 'track_ids' not in vol:
                continue

            vol_idx = vol_key.replace('volume_', '')
            prefix = f'labl_v{vol_idx}'

            data_dict[f'{prefix}_track_ids'] = vol['track_ids'][:].astype(np.int32)

            for lk in vol:
                if lk == 'track_ids':
                    continue
                if self.label_keys is not None and lk not in self.label_keys:
                    continue
                data_dict[f'{prefix}_{lk}'] = vol[lk][:].astype(np.int32)

        return data_dict

    @staticmethod
    def _read_supera(evt):
        """Whole event, nested; nothing is cast or filtered (the tables are
        small, and their dtypes — float64 kinematics — are deliberate)."""
        out = {}
        for k, g in evt.items():
            if not isinstance(g, h5py.Group):
                continue
            name = f'v{k[len("volume_"):]}' if k.startswith('volume_') else k
            out[name] = {c: d[()] for c, d in g.items()}
        return out
