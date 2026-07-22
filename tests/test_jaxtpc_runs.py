"""B2 ``runs=``: multi-run selection in ONE JAXTPCDataset with run-qualified
event names.

The name is load-bearing identity: content-addressed noise seeds
(``blake2b(name|…)``), holdout hashing and cache keys all derive from it.
Pre-B2, multi-run access (the unescaped-glob ``split='run_*'`` trick or
manual concatenation) collided names across runs because every run ships the
same shard basenames (``sim_step_0000.h5`` …). These tests pin:

* multi-run discovery + run-qualified, collision-free names,
* glob expansion == explicit list,
* single-run ``split=`` names byte-identical to the pre-``runs=`` era,
* loud failures (runs+split both set; a run missing for a modality),
* cross-modality alignment through the joint index in multi-run mode.
"""

import os
import shutil

import pytest

from pimm_data import JAXTPCDataset
from pimm_data.testing import make_jaxtpc_sample

RUNS = ('run_0000000001', 'run_0000000002')
N_EVENTS, N_FILES = 2, 2


@pytest.fixture(scope='module')
def multirun_root(tmp_path_factory):
    """Type-first layout: ``root/{mod}/{run}/sim_{mod}_NNNN.h5``."""
    root = tmp_path_factory.mktemp('jaxtpc_runs')
    for i, run in enumerate(RUNS):
        stage = tmp_path_factory.mktemp(f'stage_{i}')
        make_jaxtpc_sample(str(stage), n_events=N_EVENTS, n_files=N_FILES,
                           seed=10 + i)
        for mod in ('step', 'sensor', 'hits', 'labl'):
            os.makedirs(root / mod, exist_ok=True)
            shutil.move(str(stage / mod), str(root / mod / run))
    return str(root)


def test_runs_list_names_qualified_and_unique(multirun_root):
    ds = JAXTPCDataset(multirun_root, runs=list(RUNS), modalities=('step',))
    assert len(ds) == len(RUNS) * N_FILES * N_EVENTS
    names = [ds.get_data_name(i) for i in range(len(ds))]
    assert len(set(names)) == len(names), "run-qualified names must be unique"
    assert all(n.split('/', 1)[0] in RUNS for n in names)
    # The regression B2 fixes: the unqualified tails DO collide across runs.
    tails = [n.split('/', 1)[1] for n in names]
    assert len(set(tails)) == len(names) // len(RUNS)


def test_runs_glob_matches_explicit_sorted_list(multirun_root):
    ds_glob = JAXTPCDataset(multirun_root, runs='run_*', modalities=('step',))
    ds_list = JAXTPCDataset(multirun_root, runs=sorted(RUNS),
                            modalities=('step',))
    assert [ds_glob.get_data_name(i) for i in range(len(ds_glob))] == \
           [ds_list.get_data_name(i) for i in range(len(ds_list))]


def test_single_split_names_unchanged(multirun_root):
    """Legacy single-run names stay byte-identical (no run prefix) — the
    existing-cache / seed-compatibility contract of the D48 decision."""
    ds = JAXTPCDataset(multirun_root, split=RUNS[0], modalities=('step',))
    assert ds.get_data_name(0) == 'sim_step_0000.h5_evt000'
    assert '/' not in ds.get_data_name(0)


def test_runs_and_split_mutually_exclusive(multirun_root):
    with pytest.raises(ValueError, match='runs=.*split='):
        JAXTPCDataset(multirun_root, split=RUNS[0], runs=list(RUNS),
                      modalities=('step',))


def test_missing_run_raises(multirun_root):
    with pytest.raises(FileNotFoundError, match='run_nope'):
        JAXTPCDataset(multirun_root, runs=[RUNS[0], 'run_nope'],
                      modalities=('step',))


def test_glob_no_match_raises(multirun_root):
    with pytest.raises(FileNotFoundError, match='no run directories'):
        JAXTPCDataset(multirun_root, runs='nope_*', modalities=('step',))


def test_multirun_cross_modality_alignment(multirun_root):
    """The joint index spans runs: every idx resolves in all modalities and
    the canonical name's run matches the position in the concatenation."""
    ds = JAXTPCDataset(multirun_root, runs=list(RUNS),
                       modalities=('step', 'sensor', 'hits'))
    per_run = N_FILES * N_EVENTS
    assert len(ds) == len(RUNS) * per_run
    for idx in (0, per_run - 1, per_run, len(ds) - 1):
        expected_run = RUNS[idx // per_run]
        assert ds.get_data_name(idx).startswith(expected_run + '/')
        data = ds.get_data(idx)
        assert data['name'] == ds.get_data_name(idx)
        for mod in ('step', 'sensor', 'hits'):
            assert mod in data


def test_multirun_events_differ_across_runs(multirun_root):
    """Same (file, event) position in different runs is different physics —
    the fixture seeds differ, so the step clouds must differ."""
    import numpy as np
    ds = JAXTPCDataset(multirun_root, runs=list(RUNS), modalities=('step',))
    per_run = N_FILES * N_EVENTS
    a = ds.get_data(0)['step']['coord']
    b = ds.get_data(per_run)['step']['coord']
    assert a.shape != b.shape or not np.allclose(a, b)
