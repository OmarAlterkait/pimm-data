"""Fixtures stamp a stable source_event_idx; read_shard_meta surfaces it.

Foundation for the holdout/identity layer (D26/D27): the per-file
``config/source_event_idx`` vector is the O(1)/file identity path.
"""
import os

from pimm_data.testing import make_lucid_sample
from pimm_data._shard_meta import read_shard_meta, clear_cache


def test_lucid_fixture_stamps_source_event_idx(tmp_path):
    clear_cache()
    root = make_lucid_sample(str(tmp_path), n_events=2)
    meta = read_shard_meta(os.path.join(root, 'sensor', 'wc_sensor_0000.h5'))
    assert meta['source_event_idx'].tolist() == [0, 1]
