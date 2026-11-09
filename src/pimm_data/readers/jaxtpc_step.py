"""
JAXTPCStepReader — reads 3D truth energy deposits from JAXTPC step files.

Produces raw geometry and physics as numpy arrays. Labels are not
applied here — that is the dataset's responsibility (see jaxtpc.py).

Output dict:
    coord (N,3), energy (N,1), volume_id (N,1),
    and optionally: dx, theta, phi, t0_us, charge, photons
"""

import numpy as np

from .._shard_meta import read_deposit_counts
from ._base import ShardReaderBase


class JAXTPCStepReader(ShardReaderBase):
    """Reads 3D truth deposits from JAXTPC step HDF5 files.

    Concatenates volumes into a single point cloud with a volume_id feature.
    No label computation — just raw data.

    Parameters
    ----------
    data_root : str
        Directory containing step shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_step_0000.h5').
    min_deposits : int
        Minimum deposits per event to include in index.
    include_physics : bool
        Whether to load dx, theta, phi, charge, photons, etc.
    volume : int or None
        Load only this volume index. None = all volumes.
    """

    _MODALITY = 'step'

    def __init__(self, data_root, split='train', dataset_name='sim',
                 min_deposits=0, include_physics=True, volume=None,
                 files=None):
        self.data_root = data_root
        self.files = files
        self.split = split
        self.dataset_name = dataset_name
        self.min_deposits = min_deposits
        self.include_physics = include_physics
        self.volume = volume
        self._init_shards()

    def _index_for_shard(self, h5_path):
        """Present events, filtered by ``min_deposits`` when set.

        Uses the cached per-event deposit-count scan (F16) — the dominant
        index-build cost when min_deposits>0; memoized so train/val/test (and
        tiered) datasets over the same shards share one pass. With the filter
        off, falls back to the base's plain present-events (gap-tolerant — F6).
        """
        if self.min_deposits > 0:
            counts = read_deposit_counts(h5_path)
            return np.array(
                [num for num in sorted(counts)
                 if self._count_from_cache(counts[num]) >= self.min_deposits],
                dtype=np.int64)
        return super()._index_for_shard(h5_path)

    def _count_from_cache(self, c):
        """Deposit total for the ``min_deposits`` filter from a cached
        :func:`read_deposit_counts` entry.

        Volume-aware (A3): when ``self.volume`` is set, count only that
        volume's deposits — i.e. exactly what ``read_event`` will return — so
        an event whose deposits all live in *another* volume is excluded
        rather than kept and then read back empty. With no volume filter the
        count is the sum over all present volumes.
        """
        if c['n_volumes'] > 1:
            if self.volume is not None:
                pv = c['per_vol']
                return pv[self.volume] if self.volume < len(pv) else 0
            return sum(c['per_vol'])
        return c['positions']

    def read_event(self, idx):
        """Read one event, return flat dict of numpy arrays.

        No label computation — just raw geometry, physics, and IDs.
        """
        f, event_key = self._locate_event(idx)
        n_volumes = int(f['config'].attrs.get('n_volumes', 1))
        evt = f[event_key]

        vol_arrays = []
        n_groups = {}       # volume -> group count, when step carries groups

        if n_volumes > 1:
            for v in range(n_volumes):
                if self.volume is not None and v != self.volume:
                    continue
                vk = f'volume_{v}'
                if vk not in evt:
                    continue
                vg = evt[vk]
                if 'n_groups' in vg.attrs:
                    n_groups[v] = int(vg.attrs['n_groups'])
                n = int(vg.attrs.get('n_actual', 0))
                if n == 0:
                    continue
                vol_arrays.append(self._read_volume(vg, n, v))
        else:
            if 'positions' in evt:
                n = evt['positions'].shape[0]
                vol_arrays.append(self._read_volume_flat(evt, n, 0))

        # Groups are all-or-none per file: every volume with deposits carries
        # them, or none does (a mixture cannot be concatenated in step order).
        has = [('group_local' in d) for d in vol_arrays]
        if any(has) and not all(has):
            raise ValueError(f'{event_key}: step carries deposit_to_group in '
                             'some volumes only')
        if not all('qs_local' in d for d in vol_arrays):
            for d in vol_arrays:        # partial qs: the dataset derives it
                d.pop('qs_local', None)
        out = (self._concat_volumes(vol_arrays) if vol_arrays
               else self._empty_dict())
        if 'group_local' in out and not n_groups:
            raise ValueError(f'{event_key}: step has deposit_to_group but no '
                             'n_groups attrs')
        if n_groups:
            # The dataset turns these into ``step.group`` and pops them.
            out['_n_groups'] = n_groups
            out.setdefault('group_local', np.zeros(0, np.int32))
        return out

    def _read_volume(self, vg, n, vol_idx):
        """Read physics arrays from a volume group.

        Step carries deposit-level physics and, when the file has it, each
        deposit's JAXTPC group (``deposit_to_group``, volume-local; the
        dataset makes it the event-wide ``step.group``) and its share of the
        group's charge (``qs_fractions``). Everything above the group lives in
        labl.
        """
        step = float(vg.attrs['pos_step_mm'])
        origin = np.array([vg.attrs['pos_origin_x'],
                           vg.attrs['pos_origin_y'],
                           vg.attrs['pos_origin_z']], dtype=np.float32)

        pos = vg['positions'][:].astype(np.float32)
        pos *= step       # in-place: avoid the *step and +origin temporaries
        pos += origin
        d = {
            'coord': pos,
            'energy': vg['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
        }

        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)
            for key in ('charge', 'photons'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)
        if 'deposit_to_group' in vg:
            d['group_local'] = vg['deposit_to_group'][:].astype(np.int32)
        elif 'n_groups' in vg.attrs:
            if int(vg.attrs['n_groups']):
                raise ValueError(f'{vg.name}: {int(vg.attrs["n_groups"])} '
                                 'groups but no deposit_to_group')
            d['group_local'] = np.full(n, -1, np.int32)   # no groups at all
        if 'qs_fractions' in vg:
            d['qs_local'] = vg['qs_fractions'][:].astype(np.float32)
        elif 'group_local' in d and (d['group_local'] < 0).all():
            d['qs_local'] = np.full(n, np.nan, np.float32)
        for k in ('group_local', 'qs_local'):
            if k in d and len(d[k]) != n:
                raise ValueError(f'{vg.name}: {len(d[k])} {k[:-6]} entries, '
                                 f'{n} deposits')

        return d

    def _read_volume_flat(self, evt, n, vol_idx):
        """Read from legacy flat event format (no volume subgroups)."""
        step = float(evt.attrs['pos_step_mm'])
        origin = np.array([evt.attrs['pos_origin_x'],
                           evt.attrs['pos_origin_y'],
                           evt.attrs['pos_origin_z']], dtype=np.float32)

        pos = evt['positions'][:].astype(np.float32)
        pos *= step
        pos += origin
        d = {
            'coord': pos,
            'energy': evt['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
        }

        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)
            for key in ('charge', 'photons'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)

        return d

    def _concat_volumes(self, vol_arrays):
        """Concatenate per-volume dicts into a single flat dict."""
        keys = vol_arrays[0].keys()
        data_dict = {}
        for k in keys:
            arrays = [v[k] for v in vol_arrays if k in v]
            # Single-volume (common with volume= filter): skip the
            # concatenate, which would otherwise copy every array.
            combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
            if k == 'coord':
                data_dict[k] = combined
            elif k in ('energy', 'dx', 'theta', 'phi', 't0_us',
                       'charge', 'photons'):
                data_dict[k] = combined[:, None]
            elif k == 'volume_id':
                data_dict[k] = combined[:, None]
            else:
                data_dict[k] = combined

        return data_dict

    def _empty_dict(self):
        """Minimal valid dict for empty events."""
        return {
            'coord': np.zeros((0, 3), dtype=np.float32),
            'energy': np.zeros((0, 1), dtype=np.float32),
            'volume_id': np.zeros((0, 1), dtype=np.int32),
        }
