"""Phase A regression tests: cross-modality event alignment (joint index).

The dataset passes one global ``idx`` to every modality reader. Each reader
maps ``idx`` through its *own* present-event index, so when the present-event
sets diverge — ``min_deposits>0`` masks step only, or a production gap is
present in some modalities but not others — the same ``idx`` resolved to
*different physics events* in different modalities (silent; corrupts every
cross-modality join). See ``docs/shard_event_filtering_handoff.md`` §4 and
decision D42.

``JAXTPCDataset._build_joint_index`` fixes this by intersecting the present
event numbers across every loaded modality and injecting one shared index into
all readers. These tests lock that: the same physics event is returned across
every modality for every idx (A5), plus A3 (min_deposits without step raises;
volume-aware counting) and A4 (``strict_lengths``).

All of these FAIL on the pre-Phase-A code.
"""
import os

import h5py
import numpy as np
import pytest

from pimm_data.testing import make_jaxtpc_sample
from pimm_data.jaxtpc import JAXTPCDataset
from pimm_data.readers.jaxtpc_step import JAXTPCStepReader
from pimm_data.readers.jaxtpc_sensor import JAXTPCSensorReader

from _joint_index_helpers import readers as _readers, assert_aligned as _assert_aligned

_ALL = ('step', 'sensor', 'hits')


# ---------------------------------------------------------------------------
# Baseline: no gaps, no filter — joint index is a no-op, everything aligned.
# ---------------------------------------------------------------------------

def test_no_gap_all_modalities_aligned(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    ds = JAXTPCDataset(data_root=root, split='', modalities=_ALL, labels='segment_motif')
    assert len(ds) == 3
    _assert_aligned(ds)
    # get_data must succeed and stay FK-consistent for every idx.
    for idx in range(len(ds)):
        ds.get_data(idx)


# ---------------------------------------------------------------------------
# §4 #2 — gap present in one modality but not others.
# ---------------------------------------------------------------------------

def test_gap_in_one_modality_realigns(tmp_path):
    """entry_000001 missing from sensor only → joint excludes it everywhere."""
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'sensor', 'sim_sensor_0000.h5'), 'r+') as f:
        del f['entry_000001']                       # present in step/hits/labl

    ds = JAXTPCDataset(data_root=root, split='', modalities=_ALL,
                       labels='segment_motif', allow_partial=True)
    # Joint = intersection = {0, 2}; entry_000001 dropped from EVERY modality.
    assert len(ds) == 2
    for r in _readers(ds):
        assert r.indices[0].tolist() == [0, 2]
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)                          # no KeyError, aligned


def test_raw_readers_desync_without_joint_index(tmp_path):
    """Documents the bug: standalone readers (no joint index) DO desync.

    This is what the dataset did pre-Phase-A. step keeps all 3 events, sensor
    skips the gap → the SAME idx resolves to different events.
    """
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'sensor', 'sim_sensor_0000.h5'), 'r+') as f:
        del f['entry_000001']

    step = JAXTPCStepReader(data_root=os.path.join(root, 'step'), split='',
                            dataset_name='sim')
    sensor = JAXTPCSensorReader(data_root=os.path.join(root, 'sensor'),
                                split='', dataset_name='sim')
    assert step.indices[0].tolist() == [0, 1, 2]
    assert sensor.indices[0].tolist() == [0, 2]    # gap → different mapping
    step.h5py_worker_init(); sensor.h5py_worker_init()
    # At idx 1 the two readers point at DIFFERENT physics events — the desync
    # the joint index removes.
    assert step._locate_event(1)[1] != sensor._locate_event(1)[1]


# ---------------------------------------------------------------------------
# §4 #1 — min_deposits masks step only.
# ---------------------------------------------------------------------------

def test_min_deposits_desync_realigns(tmp_path):
    """min_deposits drops a low-deposit step event; joint drops it everywhere."""
    root = make_jaxtpc_sample(str(tmp_path), n_events=3, n_volumes=2)
    # Zero entry_000001's deposits in step so min_deposits filters it from step
    # only (sensor/hits/labl still hold entry_000001).
    with h5py.File(os.path.join(root, 'step', 'sim_step_0000.h5'), 'r+') as f:
        evt = f['entry_000001']
        for vk in [k for k in evt if k.startswith('volume_')]:
            evt[vk].attrs['n_actual'] = 0

    ds = JAXTPCDataset(data_root=root, split='', modalities=_ALL, labels='segment_motif',
                       min_deposits=1)
    assert len(ds) == 2
    for r in _readers(ds):
        assert r.indices[0].tolist() == [0, 2]
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


# ---------------------------------------------------------------------------
# A3 — min_deposits without step raises; volume-aware counting.
# ---------------------------------------------------------------------------

def test_min_deposits_without_step_raises(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=2)
    with pytest.raises(ValueError, match="min_deposits"):
        JAXTPCDataset(data_root=root, split='', modalities=('hits',), labels='segment_motif',
                      min_deposits=5)


def test_min_deposits_is_volume_aware(tmp_path):
    """volume=0 + min_deposits counts only volume 0's deposits (A3 #3).

    An event whose deposits all live in volume 1 must be excluded when
    filtering volume 0 — not kept and then read back empty.
    """
    root = make_jaxtpc_sample(str(tmp_path), n_events=2, n_volumes=2)
    with h5py.File(os.path.join(root, 'step', 'sim_step_0000.h5'), 'r+') as f:
        f['entry_000000']['volume_0'].attrs['n_actual'] = 0   # vol 0 empty here

    step_dir = os.path.join(root, 'step')
    # Volume-aware: counting only volume 0 excludes entry_000000.
    r_v0 = JAXTPCStepReader(data_root=step_dir, split='', dataset_name='sim',
                            min_deposits=1, volume=0)
    assert r_v0.indices[0].tolist() == [1]
    # All-volume count keeps it (volume 1 still has deposits).
    r_all = JAXTPCStepReader(data_root=step_dir, split='', dataset_name='sim',
                             min_deposits=1, volume=None)
    assert r_all.indices[0].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Missing or partial data raises unless allow_partial; a filter is no gap.
# ---------------------------------------------------------------------------

def test_partial_data_raises_unless_allowed(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'sensor', 'sim_sensor_0000.h5'), 'r+') as f:
        del f['entry_000001']
    with pytest.raises(ValueError, match='allow_partial'):
        JAXTPCDataset(data_root=root, split='', modalities=_ALL,
                      labels='segment_motif')
    ds = JAXTPCDataset(data_root=root, split='', modalities=_ALL,
                       labels='segment_motif', allow_partial=True)
    assert len(ds) == 2


def test_unreadable_shard_raises_unless_allowed(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=3, n_files=2)
    victim = os.path.join(root, 'hits', 'sim_hits_0001.h5')
    os.remove(victim)
    os.symlink(os.path.join(root, 'hits', 'vanished.h5'), victim)
    with pytest.raises(ValueError, match='do not open'):
        JAXTPCDataset(data_root=root, split='', modalities=('hits',))
    assert len(JAXTPCDataset(data_root=root, split='', modalities=('hits',),
                             allow_partial=True)) == 3


def test_mismatched_event_ids_always_raise(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'hits', 'sim_hits_0000.h5'), 'r+') as f:
        f['entry_000002'].attrs['event_id'] = 99
        f['dprod/events/event_id'][2] = 99
    for partial in (False, True):
        with pytest.raises(ValueError, match='event_id'):
            JAXTPCDataset(data_root=root, split='',
                          modalities=('step', 'hits'), allow_partial=partial)


def test_readers_in_their_own_id_space_are_checked(tmp_path, monkeypatch):
    """A reader that numbers events itself (helix's coeff/cells: identity
    hashes) is compared on its own ids, and modalities that hold none of
    the same events raise, partial or not."""
    from pimm_data.readers.jaxtpc_hits import JAXTPCHitsReader
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    base = JAXTPCHitsReader._index_for_shard
    monkeypatch.setattr(JAXTPCHitsReader, '_index_for_shard',
                        lambda self, p: base(self, p) + 1000)
    with pytest.raises(ValueError, match='allow_partial'):
        JAXTPCDataset(data_root=root, split='', modalities=('step', 'hits'))
    with pytest.raises(ValueError, match='no event is present'):
        JAXTPCDataset(data_root=root, split='', modalities=('step', 'hits'),
                      allow_partial=True)


def test_subclass_readers_are_aligned(tmp_path):
    """A dataset with its own modalities (helix's coeff datasets) gets its
    readers into the joint index: they come from VALID_MODALITIES, not a
    fixed list of JAXTPC names."""
    from pimm_data._dataset_base import ShardEventDataset

    root = make_jaxtpc_sample(str(tmp_path), n_events=3)

    class Coeff(ShardEventDataset):
        VALID_MODALITIES = ('coeff',)

        def __init__(self):
            self._allow_partial = False
            self.coeff_reader = JAXTPCStepReader(
                data_root=os.path.join(root, 'step'), split='',
                dataset_name='sim')
            self._canonical_reader = self.coeff_reader
            self._build_joint_index(source_label='Coeff')
            super().__init__(split='', data_root=root)

        def get_data(self, idx):
            return self.coeff_reader.read_event(self.data_list[idx])

    ds = Coeff()
    assert [n for n, _ in ds._readers_named()] == ['coeff']
    assert len(ds) == 3
    # the base reader's shard-meta cache is under its own name
    r = ds.coeff_reader
    r._meta_by_path = {'subclass': 'state'}
    r.h5py_worker_init()
    r._locate_event(0)
    assert r._meta_by_path == {'subclass': 'state'}


# ---------------------------------------------------------------------------
# Provenance: what wrote each modality; one production per dataset.
# ---------------------------------------------------------------------------

def test_provenance_is_exposed_and_one_production(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=2)
    ds = JAXTPCDataset(data_root=root, split='', modalities=_ALL,
                       labels='segment_motif')
    assert set(ds.provenance) == {'step', 'sensor', 'hits', 'labl'}
    assert ds.provenance['labl']['labl_format'] == 'supera-labl/1'
    assert 'git_commit' in ds.provenance['sensor']
    with h5py.File(os.path.join(root, 'hits', 'sim_hits_0000.h5'), 'r+') as f:
        f['config'].attrs['production_version'] = 'another_campaign'
    with pytest.raises(ValueError, match='more than one production'):
        JAXTPCDataset(data_root=root, split='', modalities=_ALL,
                      labels='segment_motif')


def test_unknown_labl_format_is_refused(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=2)
    with h5py.File(os.path.join(root, 'labl', 'sim_labl_0000.h5'), 'r+') as f:
        f['config'].attrs['labl_format'] = 'supera-labl/2'
    with pytest.raises(ValueError, match='labl_format'):
        JAXTPCDataset(data_root=root, split='', modalities=('step',),
                      labels='segment_motif')
