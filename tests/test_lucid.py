"""Tests for LUCiDDataset core loading logic (no transforms)."""

import numpy as np
import pytest

from pimm_data import LUCiDDataset


def make_ds(lucid_data_root, **kwargs):
    defaults = dict(data_root=lucid_data_root, split='', dataset_name='wc')
    defaults.update(kwargs)
    return LUCiDDataset(**defaults)


def test_sensor_response(lucid_data_root):
    """Sensor response — one entry per sensor."""
    ds = make_ds(lucid_data_root,
                 modalities=('sensor',), output_mode='response',
                 include_labels=False)
    d = ds.get_data(0)
    assert 'coord' in d
    assert 'energy' in d
    assert 'time' in d
    assert 'segment' not in d
    assert d['coord'].shape[0] > 1000


def test_sensor_labels(lucid_data_root):
    """Sensor with labels — sparse per-particle entries."""
    ds = make_ds(lucid_data_root,
                 modalities=('sensor',), output_mode='labels',
                 include_labels=True)
    d = ds.get_data(0)
    assert 'coord' in d
    assert 'segment' in d
    assert 'instance' in d
    assert d['coord'].shape[0] > 0
    assert len(np.unique(d['instance'])) > 1
    assert len(np.unique(d['segment'])) >= 1


def test_sensor_separate(lucid_data_root):
    """Sensor separate — raw reader keys."""
    ds = make_ds(lucid_data_root,
                 modalities=('sensor',), output_mode='separate')
    d = ds.get_data(0)
    assert 'pmt_pe' in d
    assert 'pmt_t' in d
    assert 'pp_sensor_idx' in d
    assert 'pp_category' in d
    assert 'coord' not in d


def test_seg_only(lucid_data_root):
    """3D track segments."""
    ds = make_ds(lucid_data_root, modalities=('seg',))
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 3
    assert d['energy'].shape[1] == 1
    assert 'track_ids' in d
    assert 'pdg' in d


def test_mixed_separate(lucid_data_root):
    """Seg + sensor separate."""
    ds = make_ds(lucid_data_root,
                 modalities=('seg', 'sensor'), output_mode='separate')
    d = ds.get_data(0)
    seg_keys = [k for k in d if k.startswith('seg3d.')]
    assert len(seg_keys) > 0
    assert 'pmt_pe' in d


def test_len_and_getitem(lucid_data_root):
    ds = make_ds(lucid_data_root, modalities=('sensor',))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)
    assert 'coord' in sample
    assert isinstance(sample['coord'], np.ndarray)


def test_dataloader_workers(lucid_data_root):
    """Dataset is fork-safe via lazy h5py_worker_init()."""
    import torch
    ds = make_ds(lucid_data_root,
                 modalities=('sensor',), output_mode='response',
                 include_labels=False)
    if len(ds) < 2:
        pytest.skip("Need at least 2 events")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda batch: batch)
    seen = 0
    for batch in loader:
        assert isinstance(batch[0], dict)
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1
