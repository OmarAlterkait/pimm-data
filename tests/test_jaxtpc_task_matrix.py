"""What JAXTPCDataset emits for every modality set, with and without labels.

Point clouds are the modalities asked for. ``group`` (the JAXTPC group, the
key step and hits share) exists whenever something defines it: hits, or
labl. The label tables and the named per-point labels come with ``labels=``
only, and only on step/hits. No bare top-level point keys.
"""

from itertools import combinations

import numpy as np

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
            yield mods, ('segment_motif', 'instance_fragment')


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
        named = {k for k in sub if k.startswith(('segment', 'instance'))}
        assert named == (set(labels) if labelled else set())
        for k in named:
            assert sub[k].shape == (n, 1)
        if part != 'sensor':
            assert ('group' in sub) == grouped
            if grouped:
                assert sub['group'].shape == (n,)
                assert sub['_roles']['group'] == ('ref', 'group')
    if grouped and not labels:
        assert set(d['group']) == {'volume', '_roles'}  # nothing above it


@pytest.mark.parametrize('kw,match', [
    (dict(modalities=('sensor',), labels='segment_motif'), 'neither'),
    (dict(modalities=('step', 'labl')), 'Unknown'),
    (dict(modalities=()), 'empty'),
    (dict(modalities=('step',), labels='motif'), 'labels='),
    (dict(modalities=('step',), labels=()), 'labels='),
    (dict(modalities=('step',), momentum='first'), 'needs labels='),
    (dict(modalities=('step',), labels='momentum', momentum='end'), 'momentum'),
])
def test_rejected(jaxtpc_data_root, kw, match):
    with pytest.raises(ValueError, match=match):
        JAXTPCDataset(data_root=jaxtpc_data_root, split='', **kw)


def test_labels_are_exactly_what_is_asked(jaxtpc_data_root):
    one = make_ds(jaxtpc_data_root, ('step', 'hits'),
                  labels='segment_pid').get_data(0)
    every = make_ds(jaxtpc_data_root, ('step', 'hits'),
                    labels='all').get_data(0)
    for on in ('step', 'hits'):
        assert {k for k in one[on] if k.startswith(('segment', 'instance'))} \
            == {'segment_pid'}
        assert {'segment_motif', 'segment_pid', 'segment_pdg',
                'instance_fragment', 'instance_particle',
                'instance_interaction', 'is_primary', 'momentum',
                'vertex'} <= set(every[on])
        # a label does not depend on which others were asked for
        assert (one[on]['segment_pid'] == every[on]['segment_pid']).all()


def test_pid_is_constant_on_particles_motif_on_fragments(jaxtpc_data_root):
    """The panoptic pairs: a class is a property of one instance level."""
    s = make_ds(jaxtpc_data_root, ('step',), labels='all',
                le_is_stuff=False).get_data(0)['step']
    for cls, inst in (('segment_motif', 'instance_fragment'),
                      ('segment_pid', 'instance_particle'),
                      ('segment_pid', 'instance_fragment')):
        c, i = s[cls][:, 0], s[inst][:, 0]
        for r in np.unique(i[i >= 0]):
            assert len(np.unique(c[i == r])) == 1, (cls, inst, r)


def test_mix_collate_keeps_named_instances_apart(jaxtpc_data_root):
    """point_collate_fn's mix merges event pairs; every instance_* level of
    the second event moves past the first's ids, -1 stays -1."""
    import random
    import torch
    from pimm_data.collate import point_collate_fn
    lab = ('instance_fragment', 'instance_particle')
    ds = make_ds(jaxtpc_data_root, ('step',), labels=lab, transform=[
        dict(type='Collect', part='step', keys=('coord',) + lab,
             feat_keys=('coord',))])
    a, b = ds[0], ds[1]
    random.seed(0)
    m = point_collate_fn([a, b], mix_prob=1.0)
    n = len(a['coord'])
    for k in lab:
        first, second = m[k][:n], m[k][n:]
        assert torch.equal(first, a[k]) and torch.equal(second < 0, b[k] < 0)
        assert not set(first[first >= 0].tolist()) & \
            set(second[second >= 0].tolist())


def test_hard_example_crop_reads_a_named_label(jaxtpc_data_root):
    from pimm_data.transform import TRANSFORMS
    s = make_ds(jaxtpc_data_root, ('step',),
                labels='segment_motif').get_data(0)['step']
    crop = TRANSFORMS.build(dict(type='HardExampleCrop', point_max=20,
                                 hard_labels=(3,), p=1.0,
                                 segment_key='segment_motif'))
    out = crop(dict(s))
    assert len(out['coord']) == 20 and (out['segment_motif'] == 3).any()


def test_label_role_compacts_instance_rows(jaxtpc_data_root):
    """Instance ids are particle rows (gaps, per event); a ('label', grp) role
    in Collect compacts them per event and makes them unique in the batch."""
    import torch
    from pimm_data import collate_fn
    ds = make_ds(jaxtpc_data_root, ('step',), labels='instance_particle',
                 transform=[dict(type='Collect', parts={'step': dict(
                     keys=('coord', 'instance_particle'),
                     roles={'instance_particle': ('label', 'p')},
                     feat_keys=('coord',))})])
    b = collate_fn([ds[0], ds[1]])
    ids = b['step_instance_particle'].reshape(-1)
    n0 = int(b['step_offset'][0])
    k0 = len(torch.unique(ids[:n0][ids[:n0] >= 0]))
    assert set(ids[:n0][ids[:n0] >= 0].tolist()) == set(range(k0))
    rest = ids[n0:][ids[n0:] >= 0]
    assert rest.min() == k0           # the second event follows the first
