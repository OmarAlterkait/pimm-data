"""JAXTPC format 2 (doraemon 1M onward): ``entry_NNNNNN`` groups, the
campaign-unique ``event_id`` in ``/dprod/events``, and the task-dir layout
``{bucket}/{task}/{modality}/{name}_{modality}_{task}.h5``.

The fixture's flat layout and the same shards moved into task directories
must read back identically."""

import os
import shutil

import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data._shard_meta import clear_cache
from pimm_data.testing import make_jaxtpc_sample

from _format2_helpers import MODS, N_EVENTS, ID_BASE, _to_tasks


@pytest.fixture(scope='module')
def roots(tmp_path_factory):
    clear_cache()
    base = tmp_path_factory.mktemp('fmt2')
    src = make_jaxtpc_sample(str(base / 'f1'), n_events=N_EVENTS, n_files=2)
    tasks = ['jaxtpc_wire_j000000-000004', 'jaxtpc_wire_j000005-000009']
    return src, _to_tasks(src, str(base / 'f2'), tasks), tasks


def _assert_same(a, b):
    for part in ('step', 'hits'):
        for k in ('coord', 'energy', 'group', 'segment', 'instance',
                  'instance_fragment'):
            np.testing.assert_array_equal(a[part][k], b[part][k])


def test_runs_nested_pattern_reads_same_events(roots):
    src, dst, _ = roots
    f1 = JAXTPCDataset(data_root=src, split='', labels='pdg', label_keys=('instance_fragment',),
                       modalities=('step', 'hits'))
    f2 = JAXTPCDataset(data_root=dst, runs='j*/*', labels='pdg', label_keys=('instance_fragment',),
                       modalities=('step', 'hits'))
    assert len(f1) == len(f2) == 2 * N_EVENTS
    for i in range(len(f1)):
        _assert_same(f1.get_data(i), f2.get_data(i))


def test_files_per_modality(roots):
    src, dst, _ = roots
    files = {m: os.path.join(dst, 'j*', '*', m, '*.h5') for m in MODS}
    f1 = JAXTPCDataset(data_root=src, split='', labels='pdg', label_keys=('instance_fragment',),
                       modalities=('step', 'hits'))
    f2 = JAXTPCDataset(data_root=dst, files=files, labels='pdg', label_keys=('instance_fragment',),
                       modalities=('step', 'hits'))
    assert len(f2) == len(f1)
    for i in range(len(f1)):
        _assert_same(f1.get_data(i), f2.get_data(i))


def test_files_must_cover_every_modality(roots):
    _, dst, _ = roots
    with pytest.raises(ValueError, match='missing'):
        JAXTPCDataset(data_root=dst, labels='pdg', label_keys=('instance_fragment',),
                      files={'step': os.path.join(dst, '*', '*', 'step', '*')},
                      modalities=('step', 'hits'))


def test_identity_is_event_id(roots):
    _, dst, _ = roots
    ds = JAXTPCDataset(data_root=dst, runs='j*/*', modalities=('step',))
    assert [ds.event_identity(i)[1] for i in range(len(ds))] == \
        list(range(ID_BASE, ID_BASE + 2 * N_EVENTS))


def test_entry_gap_in_one_modality_is_dropped(roots, tmp_path):
    clear_cache()
    _, dst, tasks = roots
    gap = str(tmp_path / 'gap')
    shutil.copytree(dst, gap)
    hits = os.path.join(gap, 'j000000', tasks[0], 'hits', f'sim_hits_{tasks[0]}.h5')
    with h5py.File(hits, 'a') as f:
        del f['entry_000001']
    ds = JAXTPCDataset(data_root=gap, runs='j*/*', modalities=('step', 'hits'))
    assert len(ds) == 2 * N_EVENTS - 1
    # entry 1 of the first task is skipped in BOTH modalities, and the
    # identity of what remains still comes from the files' event ids
    ids = [ds.event_identity(i)[1] for i in range(len(ds))]
    assert ID_BASE + 1 not in ids and len(set(ids)) == len(ids)
