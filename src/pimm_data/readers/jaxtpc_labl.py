"""
JAXTPCLablReader — reads pysupera's labels (``docs/LABELS.md``).

A ``labl`` file is written by :mod:`pimm_data.supera_labl` from pysupera's
output and pairs with a JAXTPC task's hits file entry for entry. Each event
is read whole, as nested tables::

    {'particle': {col: (P,…)}, 'interaction': {col: (I,…)},
     'track': {'id': (T,)},
     'v{N}': {'group_to_track': (G_v,), 'group_fragment': (G_v,),
              ['deposit_in_window': (N_v,)], ['deposit_to_group': (N_v,)],
              ['plane_<name>': {['on_sensor': (E,) bool],
                                ['override_index', 'override_fragment',
                                 'override_instance': (K,)]}]}}
"""

import h5py
import numpy as np

from .._shard_meta import read_shard_meta
from ..supera_labl import LABL_FORMAT
from ._base import ShardReaderBase


class JAXTPCLablReader(ShardReaderBase):
    """Reads ``labl`` files.

    Parameters
    ----------
    data_root : str
        Directory containing labl shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_labl_0000.h5').
    """

    _MODALITY = 'labl'

    def __init__(self, data_root, split='train', dataset_name='sim',
                 files=None):
        self.data_root = data_root
        self.files = files
        self.split = split
        self.dataset_name = dataset_name
        self._init_shards()
        for path in self.h5_files:
            fmt = str(read_shard_meta(path)['config_attrs'].get(
                'labl_format', ''))
            if fmt != LABL_FORMAT:
                raise ValueError(
                    f"{path}: labl_format={fmt!r}; this reader reads "
                    f"{LABL_FORMAT!r} (pimm_data.supera_labl).")

    def read_event(self, idx):
        """The whole event, nested; nothing is cast or filtered (the tables
        are small, and their dtypes are the writer's)."""
        f, event_key = self._locate_event(idx)
        out = {}
        for k, g in f[event_key].items():
            if not isinstance(g, h5py.Group):
                continue
            name = f'v{k[len("volume_"):]}' if k.startswith('volume_') else k
            out[name] = {c: (_plane(d) if isinstance(d, h5py.Group) else d[()])
                         for c, d in g.items()}
        return out


def _plane(g):
    """A ``plane_<name>`` group: per-hit label corrections (on_sensor is
    stored as packed bits with its length in attr ``n``)."""
    out = {}
    for c, d in g.items():
        if c == 'on_sensor':
            out[c] = np.unpackbits(d[()], count=int(d.attrs['n'])).astype(bool)
        else:
            out[c] = d[()]
    return out
