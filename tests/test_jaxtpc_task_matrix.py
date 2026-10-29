"""What JAXTPCDataset emits for every modality set, with and without labels.

Point clouds are the modalities asked for. ``group`` (the JAXTPC group, the
key step and hits share) exists whenever something defines it: hits, or
labl. The label tables and per-point ``segment``/``instance`` come with
``labels=`` only, and only on step/hits. No bare top-level point keys.
"""

from itertools import combinations

import pytest

from pimm_data import JAXTPCDataset

MODS = ('step', 'sensor', 'hits')
COMBOS = [c for n in (1, 2, 3) for c in combinations(MODS, n)]
TABLES = {'track', 'particle', 'interaction'}


def make_ds(root, modalities, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, **kw)


def _cases():
    for mods in COMBOS:
        yield mods, None
        if {'step', 'hits'} & set(mods):
            yield mods, 'motif'


@pytest.mark.parametrize('mods,labels', list(_cases()), ids=str)
def test_matrix(jaxtpc_data_root, mods, labels):
    d = make_ds(jaxtpc_data_root, mods, labels=labels).get_data(0)
    grouped = 'hits' in mods or labels is not None
    want = set(mods) | ({'group'} if grouped else set()) \
        | (TABLES if labels else set())
    assert set(d) - {'name', 'split'} == want
    for part, dim in (('step', 3), ('sensor', 2), ('hits', 2)):
        if part not in d:
            continue
        sub = d[part]
        n = len(sub['coord'])
        assert sub['coord'].shape == (n, dim) and sub['energy'].shape == (n, 1)
        labelled = labels is not None and part != 'sensor'
        assert ('segment' in sub) == labelled == ('instance' in sub)
        if labelled:
            assert sub['segment'].shape == sub['instance'].shape == (n,)
        if part != 'sensor':
            assert ('group' in sub) == grouped
            if grouped:
                assert sub['group'].shape == (n,)
                assert sub['_roles']['group'] == ('ref', 'group')
    if grouped and not labels:
        assert set(d['group']) == {'volume', '_roles'}  # nothing above it


@pytest.mark.parametrize('kw,match', [
    (dict(modalities=('sensor',), labels='motif'), 'neither'),
    (dict(modalities=('step', 'labl')), 'Unknown'),
    (dict(modalities=()), 'empty'),
    (dict(modalities=('step',), labels='cluster'), 'labels='),
    (dict(modalities=('step',), labels='particle'), 'labels='),
    (dict(modalities=('step',), labels='motif', instance='track'), 'instance='),
    (dict(modalities=('step',), label_keys='all'), 'need labels='),
    (dict(modalities=('step',), instance='fragment'), 'need labels='),
    (dict(modalities=('step',), labels='motif', label_keys=('x',)), 'label_keys'),
    (dict(modalities=('step',), labels='motif', momentum='end'), 'momentum'),
])
def test_rejected(jaxtpc_data_root, kw, match):
    with pytest.raises(ValueError, match=match):
        JAXTPCDataset(data_root=jaxtpc_data_root, split='', **kw)


@pytest.mark.parametrize('instance,key', [
    ('particle', 'instance_particle'), ('fragment', 'instance_fragment'),
    ('interaction', 'instance_interaction')])
def test_instance_levels(jaxtpc_data_root, instance, key):
    s = make_ds(jaxtpc_data_root, ('step', 'hits'), labels='motif',
                instance=instance, label_keys='all').get_data(0)
    for on in ('step', 'hits'):
        assert (s[on]['instance'] == s[on][key][:, 0]).all()
        assert (s[on]['segment'] == s[on]['segment_motif'][:, 0]).all()


def test_label_keys_add_only_what_is_asked(jaxtpc_data_root):
    few = make_ds(jaxtpc_data_root, ('step',), labels='pid',
                  label_keys=('momentum',)).get_data(0)['step']
    assert {k for k in few if k.startswith(('segment', 'instance'))} \
        == {'segment', 'instance'} and 'momentum' in few
    every = make_ds(jaxtpc_data_root, ('step',), labels='pid',
                    label_keys='all').get_data(0)['step']
    assert {'segment_motif', 'segment_pid', 'instance_particle',
            'instance_interaction', 'is_primary', 'momentum',
            'vertex'} <= set(every)
