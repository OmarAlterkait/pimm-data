"""Joint index aligns shards by (run, tag) identity, not glob position.

The hazard this pins (audit F-H7): with positional alignment, a middle
shard missing in ONE modality (partial transfer, failed per-modality
production job, hand-deleted corrupt file) shifts every later shard's
pairing — and because event numbers restart per shard, the per-shard
intersection stays large, silently pairing *different physics events*
across modalities. Tag alignment drops the unmatched shard from every
modality instead (warn; raise under strict_lengths).
"""

import glob
import os

import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data.testing import make_jaxtpc_sample

N_EVENTS, N_FILES = 2, 3


@pytest.fixture()
def gap_root(tmp_path):
    """3-shard fixture with the MIDDLE sensor shard deleted."""
    make_jaxtpc_sample(str(tmp_path), n_events=N_EVENTS, n_files=N_FILES,
                       seed=7)
    victim = os.path.join(str(tmp_path), 'sensor', 'sim_sensor_0001.h5')
    os.remove(victim)
    return str(tmp_path)


def _tags(reader):
    return [os.path.basename(f).rsplit('_', 1)[-1] for f in reader.h5_files]


def test_middle_shard_gap_aligns_by_tag_not_position(gap_root, caplog):
    with caplog.at_level('WARNING'):
        ds = JAXTPCDataset(gap_root, modalities=('step', 'sensor'))
    # Both readers end on the SAME shard identity sequence: 0000, 0002.
    assert _tags(ds.step_reader) == ['0000.h5', '0002.h5']
    assert _tags(ds.sensor_reader) == ['0000.h5', '0002.h5']
    assert len(ds) == 2 * N_EVENTS
    # Loud: the unmatched shard is named in the warning.
    assert any('unmatched' in r.message and '0001' in r.message
               for r in caplog.records)
    # Every idx resolves to the same (file tag, event) in both modalities —
    # the invariant positional alignment silently broke.
    for idx in range(len(ds)):
        sf, se = ds.step_reader.locate(idx)
        nf, ne = ds.sensor_reader.locate(idx)
        assert (os.path.basename(ds.step_reader.h5_files[sf]).rsplit('_', 1)[-1],
                se) == \
               (os.path.basename(ds.sensor_reader.h5_files[nf]).rsplit('_', 1)[-1],
                ne)
        ds.get_data(idx)  # loads both modalities without error


def test_middle_shard_gap_strict_raises(gap_root):
    with pytest.raises(ValueError, match='unmatched'):
        JAXTPCDataset(gap_root, modalities=('step', 'sensor'),
                      strict_lengths=True)


def test_no_gap_is_identity(tmp_path):
    """The common case (complete shards) is untouched by tag alignment."""
    make_jaxtpc_sample(str(tmp_path), n_events=N_EVENTS, n_files=N_FILES,
                       seed=8)
    ds = JAXTPCDataset(str(tmp_path), modalities=('step', 'sensor', 'hits'))
    assert len(ds) == N_FILES * N_EVENTS
    for r in (ds.step_reader, ds.sensor_reader, ds.hits_reader):
        assert _tags(r) == ['0000.h5', '0001.h5', '0002.h5']


def test_multirun_gap_drops_only_that_run(tmp_path):
    """runs= composes with tag alignment: a gap in run_b drops (run_b, tag)
    only — run_a keeps all shards."""
    import shutil
    runs = ('run_a', 'run_b')
    for i, run in enumerate(runs):
        stage = tmp_path / f'stage{i}'
        make_jaxtpc_sample(str(stage), n_events=N_EVENTS, n_files=N_FILES,
                           seed=20 + i)
        for mod in ('step', 'sensor', 'hits', 'labl'):
            os.makedirs(tmp_path / mod, exist_ok=True)
            shutil.move(str(stage / mod), str(tmp_path / mod / run))
    os.remove(str(tmp_path / 'sensor' / 'run_b' / 'sim_sensor_0001.h5'))

    ds = JAXTPCDataset(str(tmp_path), runs=list(runs),
                       modalities=('step', 'sensor'))
    assert len(ds) == (N_FILES + N_FILES - 1) * N_EVENTS
    # run_a intact, run_b missing 0001 — in BOTH readers.
    for r in (ds.step_reader, ds.sensor_reader):
        keyed = [(r.run_of(i), os.path.basename(f).rsplit('_', 1)[-1])
                 for i, f in enumerate(r.h5_files)]
        assert keyed == [('run_a', '0000.h5'), ('run_a', '0001.h5'),
                         ('run_a', '0002.h5'), ('run_b', '0000.h5'),
                         ('run_b', '0002.h5')]
    # Names stay run-qualified and consistent after re-selection.
    names = [ds.get_data_name(i) for i in range(len(ds))]
    assert len(set(names)) == len(names)
    assert names[-1].startswith('run_b/')
