"""The group correspondence and the per-plane layout, on the synthetic sample.

Label values against pysupera are checked in ``test_supera_labl.py``."""

import h5py
import numpy as np

from pimm_data import JAXTPCDataset


def make_ds(root, modalities, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, **kw)


def _volume_of(d, on):
    if on == 'step':
        return d['step']['volume_id'].ravel()
    planes = np.array([int(p.split('_')[1]) for p in d[on]['planes']])
    return planes[d[on]['plane_id'].ravel()]


def test_group_refs_without_labels(jaxtpc_data_root):
    """step + hits alone: every deposit and hit entry references a group row
    of its own volume (or -1), and qs splits each group's charge."""
    d = make_ds(jaxtpc_data_root, ('step', 'hits')).get_data(0)
    n_groups = len(d['group']['volume'])
    for on in ('step', 'hits'):
        ref = d[on]['group']
        assert ((ref >= -1) & (ref < n_groups)).all()
        own = ref >= 0
        assert (d['group']['volume'][ref[own]] == _volume_of(d, on)[own]).all()
    ref, qs = d['step']['group'], d['step']['qs'][:, 0]
    tot = np.bincount(ref[ref >= 0], weights=qs[ref >= 0], minlength=n_groups)
    assert np.allclose(tot[np.unique(ref[ref >= 0])], 1.0, atol=1e-4)


def test_hits_group_is_the_files_group_id(jaxtpc_data_root):
    """hits.group = the entry's group_id, offset by the groups of the volumes
    before it; step.group = deposit_to_group, offset the same way."""
    ds = make_ds(jaxtpc_data_root, ('step', 'hits'))
    d = ds.get_data(0)
    path = ds.hits_reader.h5_files[0]
    with h5py.File(path, 'r') as f:
        ev = f['entry_000000']
        vols = sorted(k for k in ev if k.startswith('volume_'))
        n = [len(ev[v]['group_to_track']) for v in vols]
        base = np.concatenate([[0], np.cumsum(n)[:-1]])
        d2g = np.concatenate([ev[v]['deposit_to_group'][:] + b
                              for v, b in zip(vols, base)])
    np.testing.assert_array_equal(d['step']['group'], d2g)
    assert len(d['group']['volume']) == sum(n)
    hv = _volume_of(d, 'hits')
    assert (d['group']['volume'][d['hits']['group']] == hv).all()


def test_labl_gives_steps_their_group_without_hits(jaxtpc_data_root):
    """labl carries deposit_to_group, so step + labels has the group too."""
    a = make_ds(jaxtpc_data_root, ('step', 'hits')).get_data(0)
    b = make_ds(jaxtpc_data_root, ('step',), labels='motif').get_data(0)
    np.testing.assert_array_equal(a['step']['group'], b['step']['group'])


def test_plane_id_is_dense_index(jaxtpc_data_root):
    d = make_ds(jaxtpc_data_root, ('sensor', 'hits')).get_data(0)
    for on in ('sensor', 'hits'):
        pid = d[on]['plane_id'].ravel()
        assert 0 <= pid.min() and pid.max() < len(d[on]['planes'])


def test_volume_filter_shrinks_everything(jaxtpc_data_root):
    mods = ('step', 'sensor', 'hits')
    a = make_ds(jaxtpc_data_root, mods, labels='motif').get_data(0)
    b = make_ds(jaxtpc_data_root, mods, labels='motif', volume=0).get_data(0)
    for on in mods:
        assert len(b[on]['coord']) < len(a[on]['coord']), on
    assert set(np.unique(a['group']['volume'])) == {0, 1}
    assert set(np.unique(b['group']['volume'])) == {0}
    assert (b['step']['volume_id'] == 0).all()
