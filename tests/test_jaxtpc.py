"""Tests for JAXTPCDataset core loading logic (no transforms)."""

import numpy as np
import pytest

from pimm_data import JAXTPCDataset


MAX_LEN = 4


def make_ds(jaxtpc_data_root, **kwargs):
    defaults = dict(data_root=jaxtpc_data_root, split='', dataset_name='sim')
    defaults.update(kwargs)
    return JAXTPCDataset(**defaults)


def test_seg_only(jaxtpc_data_root):
    """seg only — 3D point cloud, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('seg',))
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 3, f"coord 3D: {d['coord'].shape}"
    assert d['energy'].shape[1] == 1, f"energy: {d['energy'].shape}"
    assert 'segment' not in d, "no segment without labl"


def test_seg_labl(jaxtpc_data_root):
    """seg + labl — 3D with labels from lookup."""
    ds = make_ds(jaxtpc_data_root, modalities=('seg', 'labl'),
                 label_key='particle')
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 3
    assert 'segment' in d
    assert d['segment'].shape[0] == d['coord'].shape[0]


def test_resp_only(jaxtpc_data_root):
    """resp only — all planes merged into 2D point cloud, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('resp',))
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 2
    assert 'plane_id' in d
    assert 'segment' not in d
    assert len(np.unique(d['plane_id'])) > 1


def test_resp_corr_labl(jaxtpc_data_root):
    """resp + corr + labl — 2D labeled point cloud from corr chain."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('resp', 'corr', 'labl'),
                 label_key='particle')
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 2
    assert 'segment' in d
    assert 'instance' in d
    assert 'plane_id' in d
    resp_keys = [k for k in d if k.startswith('plane.')]
    assert len(resp_keys) > 0
    _, counts = np.unique(d['coord'], axis=0, return_counts=True)
    assert np.sum(counts > 1) > 0, "expected overlapping pixels"


def test_seg_resp_corr_labl(jaxtpc_data_root):
    """All modalities — seg owns coord, resp/corr as separate point clouds."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('seg', 'resp', 'corr', 'labl'),
                 label_key='particle')
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 3
    assert 'segment' in d
    assert 'resp_coord' in d and d['resp_coord'].shape[1] == 2
    assert 'corr_coord' in d
    assert 'corr_segment' in d
    assert 'corr_instance' in d
    plane_keys = [k for k in d if k.startswith('plane.')]
    assert len(plane_keys) > 0


def test_resp_corr_no_labl(jaxtpc_data_root):
    """resp + corr (no labl) — resp merged, corr namespaced."""
    ds = make_ds(jaxtpc_data_root, modalities=('resp', 'corr'))
    d = ds.get_data(0)
    assert d['coord'].shape[1] == 2
    assert 'segment' not in d
    corr_keys = [k for k in d if k.startswith('corr.')]
    assert len(corr_keys) > 0


def test_volume_filter(jaxtpc_data_root):
    """volume=0 — only volume 0 data (fewer points than all volumes)."""
    ds_all = make_ds(jaxtpc_data_root, modalities=('resp',))
    ds_v0 = make_ds(jaxtpc_data_root, modalities=('resp',), volume=0)
    d_all = ds_all.get_data(0)
    d_v0 = ds_v0.get_data(0)
    assert d_v0['coord'].shape[0] < d_all['coord'].shape[0]


@pytest.mark.parametrize('label_key', ['particle', 'cluster', 'interaction'])
def test_different_label_keys(jaxtpc_data_root, label_key):
    ds = make_ds(jaxtpc_data_root,
                 modalities=('seg', 'labl'), label_key=label_key)
    d = ds.get_data(0)
    assert len(np.unique(d['segment'])) > 1


def test_len_and_getitem(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('seg',))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)
    assert 'coord' in sample
    assert isinstance(sample['coord'], np.ndarray)


def test_name_and_split(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('seg',))
    d = ds.get_data(0)
    assert 'name' in d
    assert 'split' in d


def test_dataloader_workers(jaxtpc_data_root):
    """Dataset is fork-safe via lazy h5py_worker_init()."""
    import torch
    ds = make_ds(jaxtpc_data_root, modalities=('seg', 'labl'),
                 label_key='particle', min_deposits=0)
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
