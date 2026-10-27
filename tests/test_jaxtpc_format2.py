"""JAXTPC format 2 (doraemon 1M onward): ``entry_NNNNNN`` groups, the
campaign-unique ``event_id`` in ``/dprod/events``, and the task-dir layout
``{bucket}/{task}/{modality}/{name}_{modality}_{task}.h5``.

The fixture is the format-1 synthetic sample rewritten into that shape, so
every event must read back identically through both formats."""

import os
import shutil

import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data._shard_meta import clear_cache
from pimm_data.testing import make_jaxtpc_sample

MODS = ('step', 'sensor', 'hits', 'labl')
N_EVENTS = 3
ID_BASE = 1000      # event ids deliberately unlike positions


def _to_format2(src, dst, tasks):
    """Rewrite format-1 shard ``i`` as doraemon task ``tasks[i]``."""
    for i, task in enumerate(tasks):
        for mod in MODS:
            d = os.path.join(dst, 'j000000', task, mod)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f'sim_{mod}_{task}.h5')
            shutil.copy(os.path.join(src, mod, f'sim_{mod}_{i:04d}.h5'), out)
            with h5py.File(out, 'a') as f:
                ids = []
                for k in sorted(k for k in f if k.startswith('event_')):
                    n = int(k.split('_')[1])
                    new = f'entry_{n:06d}'
                    f.move(k, new)
                    eid = ID_BASE + i * N_EVENTS + n
                    f[new].attrs['event_id'] = eid
                    ids.append(eid)
                ev = f.create_group('dprod/events')
                ev.attrs['order'] = 'sorted'
                ev.create_dataset('event_id', data=np.array(ids, np.int64))
                if 'config' in f:
                    f['config'].attrs['format_version'] = 2
    return dst


@pytest.fixture(scope='module')
def roots(tmp_path_factory):
    clear_cache()
    base = tmp_path_factory.mktemp('fmt2')
    src = make_jaxtpc_sample(str(base / 'f1'), n_events=N_EVENTS, n_files=2)
    tasks = ['jaxtpc_wire_j000000-000004', 'jaxtpc_wire_j000005-000009']
    return src, _to_format2(src, str(base / 'f2'), tasks), tasks


def _assert_same(a, b):
    for part in ('step', 'hits'):
        for k in ('coord', 'energy', 'segment', 'instance'):
            np.testing.assert_array_equal(a[part][k], b[part][k])


def test_runs_nested_pattern_reads_same_events(roots):
    src, dst, _ = roots
    f1 = JAXTPCDataset(data_root=src, split='', labels='pdg',
                       modalities=('step', 'hits'))
    f2 = JAXTPCDataset(data_root=dst, runs='j*/*', labels='pdg',
                       modalities=('step', 'hits'))
    assert len(f1) == len(f2) == 2 * N_EVENTS
    for i in range(len(f1)):
        _assert_same(f1.get_data(i), f2.get_data(i))


def test_files_per_modality(roots):
    src, dst, _ = roots
    files = {m: os.path.join(dst, 'j*', '*', m, '*.h5') for m in MODS}
    f1 = JAXTPCDataset(data_root=src, split='', labels='pdg',
                       modalities=('step', 'hits'))
    f2 = JAXTPCDataset(data_root=dst, files=files, labels='pdg',
                       modalities=('step', 'hits'))
    assert len(f2) == len(f1)
    for i in range(len(f1)):
        _assert_same(f1.get_data(i), f2.get_data(i))


def test_files_must_cover_every_modality(roots):
    _, dst, _ = roots
    with pytest.raises(ValueError, match='missing'):
        JAXTPCDataset(data_root=dst, labels='pdg',
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
