"""EventLoader: batches out of the box, the pipeline's tail included.

One transform list per dataset; the head (... Collect) runs per event in the
workers, the tail on each collated batch. These pin that every loader applies
the tail, that a transform on the wrong side of Collect is refused when the
dataset is built, that Mix3D is an ordinary tail entry, and that seeded
workers reproduce."""

import pytest
import torch

from pimm_data import EventLoader, JAXTPCDataset, collate_fn

_WL = {'U': (0.42, 4.63), 'V': (0.42, 4.63), 'Y': (2.33, 2.33)}
_SENSOR = dict(type='Collect', part='sensor',
               keys=('wire', 'time', 'value', 'plane_gid'))
_STEP = dict(type='Collect', part='step', keys=('coord',),
             feat_keys=('coord', 'energy'))


def _ds(root, transform, modalities=('sensor',), **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, wire_lengths_per_plane=_WL,
                         transform=transform, **kw)


class _Stamp:
    """Records what the tail saw."""
    scope = 'batch'

    def __init__(self):
        self.seen = []

    def __call__(self, batch):
        self.seen.append((batch['_epoch'], batch['_rank']))
        return batch


def test_loader_applies_the_tail(jaxtpc_data_root):
    """The tail (ToDevice -> Densify) runs on every batch the loader yields,
    equal to running it by hand on the collated batch."""
    plain = _ds(jaxtpc_data_root, [_SENSOR])
    plain.get_data(0)
    tf = [_SENSOR, dict(type='ToDevice', device='cpu'),
          dict(type='Densify', modality=None, geom=plain.plane_geometry())]
    ds = _ds(jaxtpc_data_root, tf)
    assert [type(t).__name__ for t in ds.transform.transforms] == ['Collect']
    batches = list(EventLoader(ds, batch_size=2))
    assert batches and all('dense' in b for b in batches)
    by_hand = ds.batch_transform(collate_fn([ds[0], ds[1]]))
    for gid, grid in by_hand['dense'].items():
        assert torch.equal(batches[0]['dense'][gid], grid)


def test_loader_without_a_tail_is_the_dataloader(jaxtpc_data_root):
    ds = _ds(jaxtpc_data_root, [_SENSOR])
    assert ds.batch_transform is None
    (a,), b = list(EventLoader(ds, batch_size=len(ds))), collate_fn(
        [ds[i] for i in range(len(ds))])
    assert torch.equal(a['wire'], b['wire']) and '_epoch' not in a


def test_loader_stamps_epoch_and_rank(jaxtpc_data_root):
    stamp = _Stamp()
    loader = EventLoader(_ds(jaxtpc_data_root, [_SENSOR, stamp]),
                         batch_size=2, rank=3)
    list(loader)
    loader.set_epoch(4)
    list(loader)
    assert {s for s in stamp.seen} == {(0, 3), (4, 3)}


@pytest.mark.parametrize('tf, match', [
    # a per-event transform after Collect would run on the whole batch
    ([_STEP, dict(type='GridSample', grid_size=0.5, mode='train')],
     "scope='sample'"),
    ([_STEP, lambda b: b], "scope='sample'"),
    # a batch transform before Collect: CUDA in a worker, or one event mixed
    ([dict(type='ToDevice', device='cpu'), _STEP], "scope='batch'"),
    ([dict(type='Mix3D', prob=0.5), _STEP], "scope='batch'"),
])
def test_wrong_side_of_collect_is_refused(jaxtpc_data_root, tf, match):
    with pytest.raises(ValueError, match=match):
        _ds(jaxtpc_data_root, tf, modalities=('step',))


def test_mix3d_is_a_tail_entry(jaxtpc_data_root):
    """Mix3D after Collect halves the events of each batch; the same
    pipeline without it (validation) does not."""
    train = _ds(jaxtpc_data_root, [_STEP, dict(type='Mix3D', prob=1.0)],
                modalities=('step',))
    val = _ds(jaxtpc_data_root, [_STEP], modalities=('step',))
    assert len(val) == 2
    (t,), (v,) = (list(EventLoader(d, batch_size=2)) for d in (train, val))
    assert len(v['offset']) == 2 and len(t['offset']) == 1
    assert t['offset'].tolist() == v['offset'][1::2].tolist()
    assert torch.equal(t['coord'], v['coord'])
    with pytest.raises(ValueError, match='prob'):
        _ds(jaxtpc_data_root, [_STEP, dict(type='Mix3D', prob=2)],
            modalities=('step',))


def test_seeded_workers_reproduce(jaxtpc_data_root):
    """Same seed, same random augmentation per event, across loaders with
    workers; another seed differs."""
    tf = [dict(type='Apply', on='step', transforms=[
              dict(type='RandomRotate', angle=[-1, 1], axis='z', p=1.0)]),
          _STEP]
    ds = _ds(jaxtpc_data_root, tf, modalities=('step',))

    def coords(seed):
        return [b['coord'] for b in EventLoader(
            ds, batch_size=1, num_workers=2, seed=seed)]

    a, b, c = coords(7), coords(7), coords(8)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))
