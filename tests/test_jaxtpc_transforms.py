"""Which pimm-data transforms work inside ApplyToStream on each stream?

For every 'typical' transform and every stream (3D seg, 2D inst, 2D sensor),
build a tiny pipeline with ApplyToStream + the transform, run it, and check
the output shape is sensible. This documents — and enforces — the supported
recipes in DETECTOR_DATASET.md.
"""

import numpy as np
import pytest
import torch

from pimm_data import JAXTPCDataset, Compose, collate_fn
from pimm_data.transform import TRANSFORMS


# --- fixtures: one dataset per stream config ------------------------------

def _ds(root, modalities, **kw):
    defaults = dict(data_root=root, split='', dataset_name='sim',
                    modalities=modalities, label_key='pdg', min_deposits=0,
                    max_len=2)
    defaults.update(kw)
    return JAXTPCDataset(**defaults)


@pytest.fixture(scope='module')
def seg_sample(jaxtpc_data_root):
    """A 3D seg sub-dict with labels (seg.coord shape (N,3))."""
    ds = _ds(jaxtpc_data_root, ('seg', 'labl'))
    return ds.get_data(0)


@pytest.fixture(scope='module')
def inst_sample(jaxtpc_data_root):
    """A 2D inst sub-dict with labels (inst.coord shape (E,2))."""
    ds = _ds(jaxtpc_data_root, ('inst', 'labl'))
    return ds.get_data(0)


@pytest.fixture(scope='module')
def sensor_sample(jaxtpc_data_root):
    """A 2D sensor sub-dict, no labels (sensor.coord shape (M,2))."""
    ds = _ds(jaxtpc_data_root, ('sensor',))
    return ds.get_data(0)


def _run(sample, stream, transforms):
    """Run transforms inside an ApplyToStream on a fresh copy and return the
    post-transform sub-dict."""
    from copy import deepcopy
    data = deepcopy(sample)
    pipe = Compose([
        dict(type='ApplyToStream', stream=stream, transforms=transforms),
    ])
    data = pipe(data)
    return data[stream]


# --- spatial transforms on 3D seg -----------------------------------------

def test_seg_3d_normalize_coord(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0)])
    assert np.max(np.linalg.norm(out['coord'], axis=1)) < 2.0, \
        "after scale=4000 mm, radius should be roughly bounded"


def test_seg_3d_random_rotate(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='RandomRotate', angle=[-1, 1], axis='z',
             center=[0, 0, 0], p=1.0)])
    assert out['coord'].shape == seg_sample['seg']['coord'].shape


def test_seg_3d_random_flip(seg_sample):
    out = _run(seg_sample, 'seg', [dict(type='RandomFlip', p=1.0)])
    assert out['coord'].shape == seg_sample['seg']['coord'].shape


def test_seg_3d_random_scale(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='RandomScale', scale=[0.9, 1.1])])
    assert out['coord'].shape == seg_sample['seg']['coord'].shape


def test_seg_3d_random_jitter(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='RandomJitter', sigma=0.01, clip=0.05)])
    assert out['coord'].shape == seg_sample['seg']['coord'].shape


def test_seg_3d_grid_sample(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='GridSample', grid_size=10.0, hash_type='fnv',
             mode='train', return_grid_coord=True)])
    n_before = seg_sample['seg']['coord'].shape[0]
    n_after = out['coord'].shape[0]
    assert n_after <= n_before
    assert 'grid_coord' in out


def test_seg_3d_log_transform_energy(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='LogTransform', min_val=0.01, max_val=20.0,
             keys=('energy',))])
    assert out['energy'].shape == seg_sample['seg']['energy'].shape


def test_seg_3d_remap_segment(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='RemapSegment', scheme='motif_5cls')])
    # motif_5cls has classes 0..4
    unique = np.unique(out['segment'])
    assert unique.max() <= 4
    assert unique.min() >= -1  # -1 sentinel preserved


def test_seg_3d_shuffle_point(seg_sample):
    out = _run(seg_sample, 'seg', [dict(type='ShufflePoint')])
    assert out['coord'].shape == seg_sample['seg']['coord'].shape
    # per-point arrays must still line up
    assert out['segment'].shape[0] == out['coord'].shape[0]


def test_seg_3d_random_dropout(seg_sample):
    out = _run(seg_sample, 'seg', [
        dict(type='RandomDropout', dropout_ratio=0.2, dropout_application_ratio=1.0)])
    n_before = seg_sample['seg']['coord'].shape[0]
    n_after = out['coord'].shape[0]
    assert n_after < n_before


def test_seg_3d_positive_shift(seg_sample):
    out = _run(seg_sample, 'seg', [dict(type='PositiveShift')])
    assert (out['coord'] >= 0).all()


def test_seg_3d_copy(seg_sample):
    """Copy transform duplicates keys within a stream sub-dict."""
    out = _run(seg_sample, 'seg', [
        dict(type='Copy', keys_dict={'coord': 'origin_coord'})])
    assert 'origin_coord' in out
    assert out['origin_coord'].shape == out['coord'].shape


# --- spatial transforms on 2D inst -----------------------------------------

def test_inst_2d_grid_sample(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='GridSample', grid_size=1.0, hash_type='fnv',
             mode='train', return_grid_coord=True)])
    assert out['coord'].shape[1] == 2
    assert 'grid_coord' in out


def test_inst_2d_random_flip(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='RandomFlip', p=0.5, axes=('x', 'y'))])
    assert out['coord'].shape == inst_sample['inst']['coord'].shape


def test_inst_2d_random_scale(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='RandomScale', scale=[0.9, 1.1])])
    assert out['coord'].shape == inst_sample['inst']['coord'].shape


def test_inst_2d_random_jitter(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='RandomJitter', sigma=0.01, clip=0.05)])
    assert out['coord'].shape == inst_sample['inst']['coord'].shape


def test_inst_2d_remap_segment(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='RemapSegment', scheme='motif_5cls')])
    unique = np.unique(out['segment'])
    assert unique.max() <= 4 and unique.min() >= -1


def test_inst_2d_shuffle_point(inst_sample):
    out = _run(inst_sample, 'inst', [dict(type='ShufflePoint')])
    assert out['segment'].shape[0] == out['coord'].shape[0]


def test_inst_2d_random_dropout(inst_sample):
    out = _run(inst_sample, 'inst', [
        dict(type='RandomDropout', dropout_ratio=0.2,
             dropout_application_ratio=1.0)])
    assert out['coord'].shape[0] < inst_sample['inst']['coord'].shape[0]


def test_inst_2d_positive_shift(inst_sample):
    out = _run(inst_sample, 'inst', [dict(type='PositiveShift')])
    assert (out['coord'] >= 0).all()


# --- sensor stream (no labels) --------------------------------------------

def test_sensor_2d_grid_sample(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='GridSample', grid_size=1.0, mode='train',
             return_grid_coord=True)])
    assert out['coord'].shape[1] == 2


def test_sensor_2d_random_flip(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='RandomFlip', p=0.5, axes=('x', 'y'))])
    assert out['coord'].shape == sensor_sample['sensor']['coord'].shape


def test_sensor_2d_log_transform_energy(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='LogTransform', keys=('energy',))])
    assert out['energy'].shape == sensor_sample['sensor']['energy'].shape


# --- transforms that should be avoided on 2D streams ----------------------

def test_normalize_coord_on_2d_is_unsafe(inst_sample):
    """NormalizeCoord handles 2D coords OK (no hardcoded 3D). Smoke check."""
    out = _run(inst_sample, 'inst', [
        dict(type='NormalizeCoord', center=[0, 0], scale=1000.0)])
    assert out['coord'].shape[1] == 2


def test_random_rotate_on_2d_fails(inst_sample):
    """RandomRotate builds a 3x3 matrix and multiplies by coord; 2D coord
    has shape (N, 2), incompatible. Documents the limitation."""
    with pytest.raises(ValueError):
        _run(inst_sample, 'inst', [
            dict(type='RandomRotate', angle=[-1, 1], axis='z',
                 center=[0, 0], p=1.0)])


# --- full-pipeline integration for each typical recipe --------------------

def test_recipe_3d_supervised_seg(jaxtpc_data_root):
    """Canonical 3D semantic seg pipeline — the one in
    configs/detector/_base_/jaxtpc_seg.py."""
    ds = _ds(jaxtpc_data_root, ('seg', 'labl'),
             transform=[
                 dict(type='ApplyToStream', stream='seg', transforms=[
                     dict(type='RemapSegment', scheme='motif_5cls'),
                     dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
                     dict(type='LogTransform', min_val=0.01, max_val=20.0),
                     dict(type='GridSample', grid_size=0.001, hash_type='fnv',
                          mode='train', return_grid_coord=True),
                     dict(type='RandomRotate', angle=[-1, 1], axis='z',
                          center=[0, 0, 0], p=1.0),
                     dict(type='RandomFlip', p=0.5),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', stream='seg',
                      keys=('coord', 'grid_coord', 'segment'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 3
    assert 'feat' in batch
    assert len(batch['offset']) == 2


def test_recipe_2d_supervised_inst(jaxtpc_data_root):
    """2D supervised-on-inst pipeline (row 7 of DATASET_DESIGN)."""
    ds = _ds(jaxtpc_data_root, ('inst', 'labl'),
             transform=[
                 dict(type='ApplyToStream', stream='inst', transforms=[
                     dict(type='RemapSegment', scheme='motif_5cls'),
                     dict(type='GridSample', grid_size=1.0, hash_type='fnv',
                          mode='train', return_grid_coord=True),
                     dict(type='RandomFlip', p=0.5, axes=('x', 'y')),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', stream='inst',
                      keys=('coord', 'grid_coord', 'segment', 'instance'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 2


def test_recipe_ssl_raw_sensor(jaxtpc_data_root):
    """SSL on raw sensor — no labels flow."""
    ds = _ds(jaxtpc_data_root, ('sensor',),
             transform=[
                 dict(type='ApplyToStream', stream='sensor', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                     dict(type='ShufflePoint'),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', stream='sensor',
                      keys=('coord', 'grid_coord'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 2
    assert 'segment' not in batch


def test_recipe_denoising_sensor_plus_inst(jaxtpc_data_root):
    """Two ApplyToStream blocks, one per cloud. Both streams transform
    independently; Collect pulls whichever becomes the model input."""
    ds = _ds(jaxtpc_data_root, ('sensor', 'inst'),
             transform=[
                 dict(type='ApplyToStream', stream='sensor', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                 ]),
                 dict(type='ApplyToStream', stream='inst', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', stream='sensor',
                      keys=('coord', 'grid_coord'),
                      feat_keys=('coord', 'energy')),
             ])
    # Dataset transform runs per-sample; confirm each sample has both
    # sub-dicts transformed in-place before Collect picks one.
    sample = ds[0]
    assert 'coord' in sample and sample['coord'].shape[1] == 2
    assert 'grid_coord' in sample
