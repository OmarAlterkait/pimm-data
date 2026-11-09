"""The final v2 layout (``testing.to_final_v2``): per-readout step with only in-window
deposits, rows shuffled; groups without in-window deposits or hits dropped and the rest
renumbered; plane-local hits; labl with plane maps and no ``deposit_in_window``.

Every deposit kept and every hit get the labels the shared-step layout gives them; the
keys are what v2 promises; and every inconsistency between the files raises instead of
loading wrong labels."""

import os
import shutil

import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data._shard_meta import clear_cache
from pimm_data.supera_labl import write_labl
from pimm_data.testing import to_final_v2, to_shared_step

from test_supera_labl import TASK, _make_sensor_config, built  # noqa: F401

LAB = ('segment_motif', 'segment_pid', 'instance_fragment', 'instance_particle',
       'instance_interaction')


@pytest.fixture(scope='module')
def v2(built, tmp_path_factory):  # noqa: F811
    root, supera, hits, _, _ = built
    d = tmp_path_factory.mktemp('v2')
    step = os.path.join(root, 'j000000', TASK, 'step', f'sim_step_{TASK}.h5')
    # a short window, so some deposits fall outside it and some groups drop
    sensor = _make_sensor_config(step, str(d / 'sensor.h5'), 400)
    (d / 'a').mkdir(); (d / 'b').mkdir()
    s1, h1 = to_shared_step(step, hits, str(d / 'a' / f'sim_step_{TASK}.h5'),
                            str(d / 'a' / f'sim_hits_{TASK}.h5'))
    l1 = write_labl(supera, h1, str(d / 'a' / f'sim_labl_{TASK}.h5'),
                    step_path=s1, sensor_path=sensor, track_path=hits)
    _drop_offwindow_hits(h1, l1, s1)
    s2, h2, l2 = (str(d / 'b' / f'sim_{m}_{TASK}.h5') for m in ('step', 'hits', 'labl'))
    perm = to_final_v2(s1, h1, l1, s2, h2, l2, seed=5)
    return root, dict(step=s1, hits=h1, labl=l1), dict(step=s2, hits=h2, labl=l2), perm, d


def _drop_offwindow_hits(hits, labl, step):
    """A group whose deposits all arrive outside the readout window makes no hits:
    remove its blocks (the synthetic sample gives every group hits), so v2 drops it.
    labl's per-hit corrections (overrides, on_sensor) follow the remaining hits."""
    with h5py.File(hits, 'a') as H, h5py.File(labl, 'a') as L, h5py.File(step) as S:
        for e in (k for k in L if k.startswith('entry_')):
            for v in (k for k in L[e] if k.startswith('volume_')):
                inw = L[e][v]['deposit_in_window'][()].astype(bool)
                d2g = S[e][v]['deposit_to_group'][()]
                live = np.zeros(int(H[e][v].attrs['n_groups']), bool)
                live[d2g[inw & (d2g >= 0)]] = True
                for p in [k for k in H[e][v] if isinstance(H[e][v][k], h5py.Group)]:
                    o = H[e][v][p]
                    gid, sz = o['group_ids'][()], o['group_sizes'][()].astype(np.int64)
                    keep = live[gid]
                    if keep.all():
                        continue
                    per_entry = np.repeat(keep, sz)
                    for k in list(o):
                        x = o[k][()]
                        x = x[keep] if len(x) == len(gid) else x[per_entry]
                        del o[k]
                        o.create_dataset(k, data=x)
                    c = L[e][v].get(f'plane_{p}')
                    if c is None:
                        continue
                    new_of = np.cumsum(per_entry) - 1          # old entry -> new entry
                    if 'override_index' in c:
                        oi = c['override_index'][()]
                        ok = per_entry[oi]
                        vals = {k: c[k][()][ok] for k in ('override_fragment', 'override_instance')}
                        ni = new_of[oi[ok]].astype(oi.dtype)
                        for k, x in [('override_index', ni)] + list(vals.items()):
                            del c[k]
                            c.create_dataset(k, data=x)
                    if 'on_sensor' in c:
                        n = int(c['on_sensor'].attrs['n'])
                        bits = np.unpackbits(c['on_sensor'][()], count=n)[per_entry]
                        del c['on_sensor']
                        c.create_dataset('on_sensor', data=np.packbits(bits))
                        c['on_sensor'].attrs['n'] = len(bits)


def _ds(root, files, mods, labels='all', **kw):
    clear_cache()
    return JAXTPCDataset(data_root=root, files={m: files[m] for m in mods + (('labl',) if labels else ())},
                         modalities=mods, labels=labels, **kw)


def test_fixture_drops_deposits_and_groups(v2):
    _, _, _, perm, _ = v2
    assert any((pi < 0).any() for pi, _ in perm.values()), 'no group dropped'
    with h5py.File(v2[1]['labl']) as l1:
        n_out = sum(int((~l1[e][v]['deposit_in_window'][()].astype(bool)).sum())
                    for (e, v) in perm)
    assert n_out > 0, 'no deposit outside the window'


def test_final_v2_loads_the_same(v2):
    root, a, b, perm, _ = v2
    for mods in (('step',), ('hits',), ('step', 'hits')):
        A, B = _ds(root, a, mods), _ds(root, b, mods)
        assert len(A) == len(B)
        for i in range(len(A)):
            x, y = A.get_data(i), B.get_data(i)
            if 'step' in mods:
                vid = x['step']['volume_id'].ravel()
                ent = sorted({k[0] for k in perm})[i]
                rows = np.concatenate([np.flatnonzero(vid == v)[perm[(ent, f'volume_{v}')][1]]
                                       for v in np.unique(vid)])
                for k in LAB + ('coord', 'energy'):
                    np.testing.assert_array_equal(x['step'][k][rows], y['step'][k], err_msg=f'{mods} {i} {k}')
                assert 'in_window' not in y['step']
            if 'hits' in mods:
                cols = lambda h: np.column_stack([np.asarray(h[k], np.float64).reshape(len(h['coord']), -1)
                                                  for k in ('plane_id', 'coord', 'energy') + LAB])
                rx, ry = cols(x['hits']), cols(y['hits'])
                np.testing.assert_array_equal(rx[np.lexsort(rx.T[::-1])], ry[np.lexsort(ry.T[::-1])],
                                              err_msg=f'{mods} {i} hits')


def test_final_v2_keys_and_groups(v2):
    root, _, b, perm, _ = v2
    y = _ds(root, b, ('step', 'hits')).get_data(0)
    assert {k for k in y['step'] if k.startswith(('segment', 'instance'))} == set(LAB)
    assert 'in_window' not in y['step']
    ent = sorted({k[0] for k in perm})[0]
    n_kept = sum(int((perm[(e, v)][0] >= 0).sum()) for (e, v) in perm if e == ent)
    assert len(y['group']['volume']) == n_kept
    ref = np.zeros(n_kept, bool)
    ref[y['step']['group'][y['step']['group'] >= 0]] = True
    ref[y['hits']['group']] = True
    assert ref.all(), 'a kept group has neither deposits nor hits'
    z = _ds(root, b, ('step', 'hits'), labels=None).get_data(0)
    assert 'group' not in z['hits'] and len(z['hits']['plane_group']) == len(z['hits']['coord'])


def test_plane_group_survives_sampling(v2):
    root, _, b, _, _ = v2
    ds = _ds(root, b, ('hits',), labels=None, transform=[dict(type='Apply', on='hits', transforms=[
        dict(type='RandomDropout', dropout_ratio=0.5, dropout_application_ratio=1.0)])])
    h = ds[0]['hits']
    assert len(h['plane_group']) == len(h['coord'])


def _mutate(src, dst, fn):
    shutil.copyfile(src, dst)
    with h5py.File(dst, 'a') as f:
        e = sorted(k for k in f if k.startswith('entry_'))[0]
        fn(f, f[e])


def _set(g, name, data):
    del g[name]
    g.create_dataset(name, data=data)


def _first_plane(ev):
    v = sorted(k for k in ev if k.startswith('volume_'))[0]
    p = sorted(k for k in ev[v] if k.startswith('plane_'))[0]
    return ev[v], ev[v][p]


MUTATIONS = {
    'labl lacks a volume': ('labl', lambda f, ev: ev.__delitem__(sorted(k for k in ev if k.startswith('volume_'))[0])),
    'hits lost group_id_space': ('hits', lambda f, ev: f.attrs.__delitem__('group_id_space')),
    'plane map out of range': ('labl', lambda f, ev: _set(_first_plane(ev)[1], 'group',
                                                          _first_plane(ev)[1]['group'][()] + 10 ** 6)),
    'plane map has -1': ('labl', lambda f, ev: _set(_first_plane(ev)[1], 'group',
                                                    np.where(np.arange(len(_first_plane(ev)[1]['group'])) == 0, -1,
                                                             _first_plane(ev)[1]['group'][()]).astype(np.int32))),
    'plane map too long': ('labl', lambda f, ev: _set(_first_plane(ev)[1], 'group',
                                                      np.append(_first_plane(ev)[1]['group'][()], 0).astype(np.int32))),
    'group_fragment length': ('labl', lambda f, ev: _set(_first_plane(ev)[0], 'group_fragment',
                                                         _first_plane(ev)[0]['group_fragment'][()][:-1])),
    'group_fragment out of range': ('labl', lambda f, ev: _set(_first_plane(ev)[0], 'group_fragment',
                                                               np.full_like(_first_plane(ev)[0]['group_fragment'][()], 10 ** 6))),
    'track/id unsorted': ('labl', lambda f, ev: _set(ev['track'], 'id', ev['track']['id'][()][::-1].copy())),
    'step deposit_to_group out of range': ('step', lambda f, ev: _set(
        ev[sorted(k for k in ev if k.startswith('volume_'))[0]], 'deposit_to_group',
        np.full_like(ev[sorted(k for k in ev if k.startswith('volume_'))[0]]['deposit_to_group'][()], 10 ** 6))),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_inconsistent_v2_raises(v2, name):
    root, _, b, _, d = v2
    which, fn = MUTATIONS[name]
    (d / 'bad').mkdir(exist_ok=True)
    files = dict(b)
    dst = str(d / 'bad' / os.path.basename(b[which]))
    _mutate(b[which], dst, fn)
    files[which] = dst
    # the other files go into the same directory so the shard pairing holds
    for m, p in b.items():
        if m != which:
            q = str(d / 'bad' / os.path.basename(p))
            shutil.copyfile(p, q)
            files[m] = q
    mods = ('step', 'hits')
    with pytest.raises((ValueError, KeyError)) as err:
        ds = _ds(root, files, mods)
        for i in range(len(ds)):
            ds.get_data(i)
    print(f'{name}: {err.value}')
    shutil.rmtree(d / 'bad')
