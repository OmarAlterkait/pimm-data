"""
LUCiDSensorReader — reads sparse PMT hits from LUCiD ``sensor/`` HDF5
files (``format_version: 3``).

Layout: per-event groups. Each ``event_XXX`` holds flat per-hit arrays
``PE``, ``T``, ``sensor_idx``. The full PMT position table lives once per
file at ``config/sensor_positions``. Per-particle decomposition and
label columns are *not* in this file — they live in ``hits/`` and
``labl/``.

Output dict:

    sensor_idx  (H,)          int32    — PMT index per hit
    pmt_pe      (H,)          float32
    pmt_t       (H,)          float32
    pmt_coord   (n_sensors,3) float32  — full PMT geometry

The reader also exposes ``detector_geometry`` — a dict with ``half_height``,
``radius`` (meters) and ``axis`` ((3,) float32) — read from the config
attrs when present (WAND fv5) and otherwise derived from the PMT position
table, so consumers never hardcode detector dimensions.
"""

import h5py
import numpy as np

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase


class LUCiDSensorReader(ShardReaderBase):
    """Reads sparse per-event PMT hits from LUCiD ``sensor/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing sensor shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_sensor_*.h5``.
    pmt_positions : ndarray or None
        Optional override for ``config/sensor_positions``. Kept for
        geometry substitution experiments; normally leave ``None``.
    pmt_positions_file : str or None
        Optional path to a ``.npy`` file overriding PMT positions.
    """

    _MODALITY = 'sensor'

    def __init__(self, data_root, split='', dataset_name='wc',
                 pmt_positions=None, pmt_positions_file=None, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name

        if pmt_positions is not None:
            self._pmt_positions = np.asarray(pmt_positions, dtype=np.float32)
        elif pmt_positions_file is not None:
            self._pmt_positions = np.load(pmt_positions_file).astype(np.float32)
        else:
            self._pmt_positions = None

        self._n_sensors = None       # set as a side-effect of the index build
        self._init_shards()
        self.detector_geometry = self._read_detector_geometry()

    def _read_detector_geometry(self):
        """Detector geometry, without hardcoding: prefer explicit
        ``detector_half_height`` / ``detector_radius`` attrs and the
        ``detector_axis`` dataset (step files carry them in WAND fv5),
        else derive from the PMT positions themselves — the sensors sit on
        the detector surface, so ``half_height = max |pos . axis|`` and
        ``radius = max |pos - (pos . axis) axis|``. ``None`` values only
        when neither source is available."""
        geom = dict(half_height=None, radius=None, axis=None)
        positions = self._pmt_positions
        for h5_path in self.h5_files:
            # First shard that actually opens — a dangling shard (F17) is
            # skipped here just like everywhere else. Geometry is per-file
            # metadata shared across shards.
            try:
                f = h5py.File(h5_path, 'r', libver='latest', swmr=True)
            except OSError:
                continue
            with f:
                cfg = f.get('config')
                if cfg is not None:
                    if 'detector_half_height' in cfg.attrs:
                        geom['half_height'] = float(
                            cfg.attrs['detector_half_height'])
                    if 'detector_radius' in cfg.attrs:
                        geom['radius'] = float(cfg.attrs['detector_radius'])
                    if 'detector_axis' in cfg:
                        geom['axis'] = cfg['detector_axis'][:].astype(
                            np.float32)
                    if positions is None and 'sensor_positions' in cfg:
                        positions = cfg['sensor_positions'][:].astype(
                            np.float32)
            break
        if geom['axis'] is None:
            geom['axis'] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if positions is not None and positions.ndim == 2 \
                and positions.shape[1] == 3:
            along = positions @ geom['axis']
            perp = positions - np.outer(along, geom['axis'])
            if geom['half_height'] is None:
                geom['half_height'] = float(np.abs(along).max())
            if geom['radius'] is None:
                geom['radius'] = float(
                    np.sqrt((perp ** 2).sum(axis=1)).max())
        return geom

    def _index_for_shard(self, h5_path):
        """Present events; also caches n_sensors from the first shard's config
        (one read via the A1 cache — no extra open)."""
        meta = read_shard_meta(h5_path)
        if self._n_sensors is None:
            self._n_sensors = int(meta['config_attrs'].get('n_sensors', 0))
        return meta['present_events']

    def h5py_worker_init(self):
        super().h5py_worker_init()
        if self._pmt_positions is None:
            # First shard that actually opened (file 0 may be a skipped
            # empty/dangling shard — F17). PMT geometry is shared across shards.
            f0 = next((h for h in self._h5data if h is not None), None)
            if f0 is not None and 'config' in f0 \
                    and 'sensor_positions' in f0['config']:
                self._pmt_positions = f0['config']['sensor_positions'][:].astype(
                    np.float32)

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data = {
            'sensor_idx': evt['sensor_idx'][:].astype(np.int32),
            'pmt_pe':     evt['PE'][:].astype(np.float32),
            'pmt_t':      evt['T'][:].astype(np.float32),
        }
        if self._pmt_positions is not None:
            data['pmt_coord'] = self._pmt_positions
        return data
