"""The shared-step layout: one step per task carrying the JAXTPC groups
(``deposit_to_group``, ``qs_fractions``, ``n_groups``), hits with entries and
group ids only, labl without ``deposit_to_group`` (written from that step).
``group_to_track`` is label truth and lives in labl only (``group_to_track``); the
writer takes it from another file (here the produced hits).

Every modality combination must load what the produced layout loads, strictly
(dtypes included; qs to float16 rounding), for wire and pixel, every ``volume=``,
two files, with deposits outside every group and a volume with groups but no
deposits. The only intended difference: step without hits or labl gains the
groups and qs, and they must equal what step+hits gives. Inconsistent files
must raise."""

import glob
import os
import shutil

import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data._shard_meta import clear_cache
from pimm_data.supera_labl import write_labl
from pimm_data.testing import make_jaxtpc_sample, to_shared_step

COMBOS = [('step',), ('hits',), ('sensor',), ('step', 'hits'), ('step', 'sensor'),
          ('hits', 'sensor'), ('step', 'hits', 'sensor')]
UNGROUPED = 5       # deposits of entry 0 / volume 0 put outside every group


def _sensor_cfg(step_path, out_path):
    """The sensor config the in-window flag needs (as in test_supera_labl)."""
    with h5py.File(step_path) as s, h5py.File(out_path, 'w') as f:
        c = f.create_group('config')
        c['volume_ranges'] = s['config/volume_ranges'][:]
        c.attrs.update(velocity_cm_us=0.16, time_step_us=0.5,
                       num_time_steps=1350, pre_window_us=0.0)


def _edit_produced(root):
    """Real-data cases the plain fixture lacks, applied to the produced files
    before conversion, so both layouts see them."""
    step = sorted(glob.glob(f'{root}/step/*.h5'))
    hits = sorted(glob.glob(f'{root}/hits/*.h5'))
    with h5py.File(hits[0], 'a') as h, h5py.File(step[0]) as s:
        hv = h['entry_000000/volume_0']        # deposits outside every group
        x = hv['deposit_to_group'][:]
        x[:UNGROUPED] = -1
        hv['deposit_to_group'][...] = x
        # JAXTPC's qs over the groups that are left (0 where no group)
        q = s['entry_000000/volume_0/charge'][:].astype(np.float64)
        tot = np.bincount(np.maximum(x, 0), weights=np.where(x >= 0, q, 0))
        hv['qs_fractions'][...] = np.where(
            x >= 0, q / tot[np.maximum(x, 0)], 0).astype(
                hv['qs_fractions'].dtype)
    # entry 1 / volume 1 of file 1: groups (and hits) but no deposits, as in
    # real empty volumes (step group empty, n_actual 0; hits has no d2g / qs)
    with h5py.File(step[1], 'a') as s, h5py.File(hits[1], 'a') as h:
        sv = s['entry_000001/volume_1']
        for k in list(sv):
            del sv[k]
        sv.attrs['n_actual'] = 0
        hv = h['entry_000001/volume_1']
        for k in ('deposit_to_group', 'qs_fractions'):
            del hv[k]


@pytest.fixture(scope='module', params=['wire', 'pixel'])
def layouts(request, tmp_path_factory):
    """(old files, new files) for one readout; each a {modality: [paths]}."""
    clear_cache()
    ro = request.param
    base = tmp_path_factory.mktemp(f'shared_{ro}')
    root = make_jaxtpc_sample(str(base / 'old'), n_events=3, n_files=2,
                              readout_type=ro)
    _edit_produced(root)
    new = base / 'new'
    for m in ('step', 'hits', 'labl', 'sensor'):
        (new / m).mkdir(parents=True)
    old_files = {m: [] for m in ('step', 'hits', 'labl', 'sensor')}
    new_files = {m: [] for m in old_files}
    for i in range(2):
        f = lambda m, r=root: os.path.join(r, m, f'sim_{m}_{i:04d}.h5')
        g = lambda m: str(new / m / f'sim_{m}_{i:04d}.h5')
        cfg = str(base / f'sensor_cfg_{i}.h5')
        _sensor_cfg(f('step'), cfg)
        # produced layout: labl with deposit_in_window, as in production
        write_labl(f('supera'), f('hits'), f('labl'), compression='gzip',
                   step_path=f('step'), sensor_path=cfg)
        to_shared_step(f('step'), f('hits'), g('step'), g('hits'))
        write_labl(f('supera'), g('hits'), g('labl'), compression='gzip',
                   step_path=g('step'), sensor_path=cfg, track_path=f('hits'))
        shutil.copy(f('sensor'), g('sensor'))
        for m in old_files:
            old_files[m].append(f(m))
            new_files[m].append(g(m))
    return ro, root, old_files, new_files


def _load(root, files, mods, labels, **kw):
    m = mods + (('labl',) if labels else ())
    ds = JAXTPCDataset(data_root=root, modalities=mods, labels=labels,
                       files={x: files[x] for x in m}, **kw)
    return [ds.get_data(i) for i in range(len(ds))]


def _qs_close(a, b):
    """Stored float16 vs derived float32: one float16 ulp, same NaNs."""
    na, nb = np.isnan(a), np.isnan(b)
    np.testing.assert_array_equal(na, nb)
    np.testing.assert_array_less(np.abs(a[~na] - b[~nb]),
                                 2.0 ** -11 * np.abs(b[~nb]) + 1e-7)


def _same(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), (path, sorted(set(a) ^ set(b)))
        for k in a:
            _same(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, \
            (path, a.dtype, b.dtype, a.shape, b.shape)
        if path.endswith('step/qs'):
            _qs_close(a, b)
        else:
            np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, (path, a, b)


@pytest.mark.parametrize('volume', [None, 0, 1])
@pytest.mark.parametrize('labels', [None, 'all'])
@pytest.mark.parametrize('mods', COMBOS, ids='+'.join)
def test_every_combination_loads_the_same(layouts, mods, labels, volume):
    ro, root, old, new = layouts
    if labels and mods == ('sensor',):
        pytest.skip('sensor+labl is rejected by design')
    a = _load(root, old, mods, labels, volume=volume)
    b = _load(root, new, mods, labels, volume=volume)
    assert len(a) == len(b) == 6
    gains = 'step' in mods and 'hits' not in mods and not labels
    for i, (x, y) in enumerate(zip(a, b)):
        if gains:
            # the one intended difference; checked against step+hits below
            assert 'group' not in x and 'group' in y
            y = {k: v for k, v in y.items() if k != 'group'}
            y['step'] = {k: v for k, v in y['step'].items()
                         if k not in ('group', 'qs')}
            y['step']['_roles'] = {k: v for k, v in y['step']['_roles'].items()
                                   if k != 'group'}
            if not y['step']['_roles']:
                del y['step']['_roles']
        _same(x, y, f'{ro} {mods} {labels} vol={volume} ev{i}')


@pytest.mark.parametrize('volume', [None, 0, 1])
def test_step_alone_gets_the_groups_step_hits_gives(layouts, volume):
    """New step alone: step.group, step.qs and the group table equal what
    step+hits gives in the produced layout, and ungrouped deposits stay -1."""
    ro, root, old, new = layouts
    alone = _load(root, new, ('step',), None, volume=volume)
    ref = _load(root, old, ('step', 'hits'), None, volume=volume)
    n_neg = 0
    for y, x in zip(alone, ref):
        np.testing.assert_array_equal(y['step']['group'], x['step']['group'])
        _qs_close(y['step']['qs'], x['step']['qs'])
        _same(y['group'], x['group'])
        n_neg += int((y['step']['group'] < 0).sum())
    assert n_neg == (UNGROUPED if volume in (None, 0) else 0)


def _copy_new(layouts, tmp_path):
    ro, root, old, new = layouts
    out = {}
    for m, paths in new.items():
        (tmp_path / m).mkdir()
        out[m] = [shutil.copy(p, tmp_path / m) for p in paths]
    return root, old, out


def test_stored_qs_is_read_not_rederived(layouts, tmp_path):
    """Step's qs_fractions are what step.qs returns (NaN where no group)."""
    root, _, new = _copy_new(layouts, tmp_path)
    with h5py.File(new['step'][0], 'a') as s:
        for v in ('volume_0', 'volume_1'):
            d = s[f'entry_000000/{v}/qs_fractions']
            d[...] = (d[:].astype(np.float32) * 0.5).astype(d.dtype)
        want = np.concatenate([s[f'entry_000000/{v}/qs_fractions'][:]
                               for v in ('volume_0', 'volume_1')])
    s = _load(root, new, ('step',), None)[0]['step']
    g = s['group'][:, 0] if s['group'].ndim > 1 else s['group']
    exp = np.where(g >= 0, want.astype(np.float32), np.nan)
    np.testing.assert_array_equal(s['qs'][:, 0], exp)
    assert np.isnan(s['qs'][:UNGROUPED, 0]).all()


def test_partial_qs_falls_back_to_derived(layouts, tmp_path):
    root, old, new = _copy_new(layouts, tmp_path)
    with h5py.File(new['step'][0], 'a') as s:
        del s['entry_000000/volume_1/qs_fractions']
    y = _load(root, new, ('step',), None)[0]['step']['qs']
    x = _load(root, old, ('step', 'hits'), None)[0]['step']['qs']
    _qs_close(y, x)


@pytest.mark.parametrize('case', ['d2g', 'g2t', 'n_groups'])
def test_inconsistent_files_raise(layouts, tmp_path, case):
    root, old, new = _copy_new(layouts, tmp_path)
    files = dict(new)
    with h5py.File(new['step'][0], 'a') as s, \
            h5py.File(new['labl'][0], 'a') as lb:
        v = s['entry_000000/volume_1']
        if case == 'd2g':       # step vs old labl's copy
            d = v['deposit_to_group']
            d[...] = d[:][::-1]
            files['labl'] = old['labl']
        elif case == 'g2t':     # labl group_to_track vs old hits' group_to_track
            d = lb['entry_000000/volume_1/group_to_track']
            d[...] = d[:] + 1
            files['hits'] = old['hits']
        else:                   # step vs labl/hits group counts
            v.attrs['n_groups'] = int(v.attrs['n_groups']) + 1
    with pytest.raises(ValueError):
        _load(root, files, ('step', 'hits'), 'all')


def test_old_step_with_new_hits_raises(layouts):
    """No file gives the deposits' groups: refuse, never label them -1."""
    ro, root, old, new = layouts
    with pytest.raises(ValueError, match='deposit_to_group'):
        _load(root, dict(new, step=old['step']), ('step', 'hits'), 'all')


def test_new_labl_has_no_deposit_to_group(layouts):
    ro, root, old, new = layouts
    with h5py.File(new['labl'][0]) as lb, h5py.File(new['hits'][0]) as h:
        for e in (k for k in lb if k.startswith('entry_')):
            for v in (k for k in lb[e] if k.startswith('volume_')):
                assert 'deposit_to_group' not in lb[e][v]
                assert 'group_to_track' in lb[e][v]
                assert not {'deposit_to_group', 'qs_fractions',
                            'group_to_track'} & set(h[e][v])
    with h5py.File(new['step'][0]) as s:
        for e in (k for k in s if k.startswith('entry_')):
            for v in (k for k in s[e] if k.startswith('volume_')):
                assert 'group_to_track' not in s[e][v]


def test_writer_needs_a_track_source(layouts, tmp_path):
    ro, root, old, new = layouts
    sup = os.path.join(root, 'supera', 'sim_supera_0000.h5')
    cfg = str(tmp_path / 'cfg.h5')
    _sensor_cfg(new['step'][0], cfg)
    with pytest.raises(ValueError, match='track_path'):
        write_labl(sup, new['hits'][0], str(tmp_path / 'l.h5'),
                   compression='gzip', step_path=new['step'][0],
                   sensor_path=cfg)


def _copy_all(layouts, tmp_path):
    """Writable copies of both layouts' files."""
    ro, root, old, new = layouts
    out = {}
    for name, files in (('old', old), ('new', new)):
        out[name] = {}
        for m, paths in files.items():
            d = tmp_path / name / m
            d.mkdir(parents=True)
            out[name][m] = [shutil.copy(p, d) for p in paths]
    return root, out['old'], out['new']


@pytest.mark.parametrize('case, match', [
    ('n_groups_no_d2g', 'no deposit_to_group'),
    ('d2g_no_n_groups', 'no n_groups'),
    ('d2g_some_volumes', 'some volumes only'),
])
def test_malformed_step_raises(layouts, tmp_path, case, match):
    root, _, new = _copy_all(layouts, tmp_path)
    with h5py.File(new['step'][0], 'a') as s:
        e = s['entry_000000']
        if case == 'n_groups_no_d2g':
            del e['volume_1/deposit_to_group']
        elif case == 'd2g_no_n_groups':
            for v in ('volume_0', 'volume_1'):
                del e[v].attrs['n_groups']
        else:               # volume 0 not grouped at all, volume 1 grouped
            del e['volume_0/deposit_to_group']
            del e['volume_0'].attrs['n_groups']
    with pytest.raises(ValueError, match=match):
        _load(root, new, ('step',), None)


def test_old_labl_and_hits_d2g_must_agree(layouts, tmp_path):
    root, old, _ = _copy_all(layouts, tmp_path)
    with h5py.File(old['labl'][0], 'a') as lb:
        d = lb['entry_000000/volume_1/deposit_to_group']
        d[...] = d[:][::-1]
    with pytest.raises(ValueError, match='labl differs from hits'):
        _load(root, old, ('step', 'hits'), 'all')


@pytest.mark.parametrize('rel, raises', [(1e-2, True), (2.0 ** -11, False)])
def test_step_and_old_hits_qs_must_agree(layouts, tmp_path, rel, raises):
    """New step + old hits: qs agrees to float16 rounding, not beyond."""
    root, old, new = _copy_all(layouts, tmp_path)
    with h5py.File(old['hits'][0], 'a') as h:
        d = h['entry_000000/volume_1/qs_fractions']
        q = d[:].astype(np.float64)
        d[...] = (q + (1e-2 if rel == 1e-2 else q * rel)).astype(d.dtype)
    files = dict(new, hits=old['hits'])
    if raises:
        with pytest.raises(ValueError, match='qs_fractions in step'):
            _load(root, files, ('step', 'hits'), None)
    else:
        _load(root, files, ('step', 'hits'), None)


def test_hits_n_groups_must_match_group_to_track(layouts, tmp_path):
    root, old, _ = _copy_all(layouts, tmp_path)
    with h5py.File(old['hits'][0], 'a') as h:
        hv = h['entry_000000/volume_1']
        hv.attrs['n_groups'] = len(hv['group_to_track']) + 1
    with pytest.raises(ValueError, match='group_to_track has'):
        _load(root, old, ('hits',), None)


@pytest.mark.parametrize('case, match', [
    ('hits_vs_step', 'in hits'),
    ('step_vs_g2t', 'group_to_track has'),
    ('other_task_tracks', 'track_path'),
])
def test_writer_consistency_raises(layouts, tmp_path, case, match):
    root, old, new = _copy_all(layouts, tmp_path)
    sup = os.path.join(root, 'supera', 'sim_supera_0000.h5')
    cfg = str(tmp_path / 'cfg.h5')
    _sensor_cfg(new['step'][0], cfg)
    track = old['hits'][0]
    if case == 'hits_vs_step':
        with h5py.File(new['hits'][0], 'a') as h:
            a = h['entry_000000/volume_1'].attrs
            a['n_groups'] = int(a['n_groups']) + 1
    elif case == 'step_vs_g2t':
        for p in (new['step'][0], new['hits'][0]):
            with h5py.File(p, 'a') as f:
                a = f['entry_000000/volume_1'].attrs
                a['n_groups'] = int(a['n_groups']) + 1
    else:
        track = old['hits'][1]          # another file's events
    with pytest.raises(ValueError, match=match):
        write_labl(sup, new['hits'][0], str(tmp_path / 'l.h5'),
                   compression='gzip', step_path=new['step'][0],
                   sensor_path=cfg, track_path=track)


def test_group_links_in_every_combination(layouts):
    """hit -> group from hits alone; qs splits a group's charge over its deposits
    (sums to 1, NaN exactly off-group, conserves hit charge pushed down); every hit
    carries its group's deposit labels; refs identical across combinations."""
    ro, root, old, new = layouts
    comb = {n: _load(root, new, m, lab) for n, (m, lab) in dict(
        H=(('hits',), None), HL=(('hits',), 'all'), S=(('step',), None),
        SL=(('step',), 'all'), SH=(('step', 'hits'), None),
        SHL=(('step', 'hits'), 'all'),
        ALL=(('step', 'hits', 'sensor'), 'all')).items()}
    for i, x in enumerate(comb['SHL']):
        h, s, g = x['hits'], x['step'], x['group']
        G = len(g['volume'])
        for n in ('H', 'HL', 'SH', 'ALL'):
            np.testing.assert_array_equal(comb[n][i]['hits']['group'], h['group'])
        for n in ('S', 'SL', 'SH', 'ALL'):
            np.testing.assert_array_equal(comb[n][i]['step']['group'], s['group'])
        hg, sg = h['group'], s['group']
        assert ((hg >= 0) & (hg < G)).all()
        pv = np.array([int(p.split('_')[1]) for p in h['planes']])[
            h['plane_id'].ravel()]
        np.testing.assert_array_equal(g['volume'][hg], pv)
        qs, m = s['qs'][:, 0], sg >= 0
        np.testing.assert_array_equal(np.isnan(qs), ~m)
        qsum = np.bincount(sg[m], weights=qs[m].astype(np.float64), minlength=G)
        nd = np.bincount(sg[m], minlength=G)
        np.testing.assert_allclose(qsum[nd > 0], 1, atol=2 ** -10 * nd.max())
        hq = np.bincount(hg, weights=h['energy'][:, 0].astype(np.float64),
                         minlength=G)
        back = np.bincount(sg[m], weights=qs[m] * hq[sg[m]], minlength=G)
        ok = (nd > 0) & (hq > 0)
        np.testing.assert_allclose(back[ok], hq[ok], rtol=2e-3)
        for c in (k for k in h if k.startswith(('segment_', 'instance_'))):
            lab = np.full(G, -10 ** 9, np.int64)
            lab[sg[m]] = s[c][m, 0]
            np.testing.assert_array_equal(lab[sg[m]], s[c][m, 0])  # one per group
            # hits pysupera labels as their group (not off-sensor, not in a
            # group it split) carry the group's deposit labels
            hh = (nd[hg] > 0) & (h['instance_fragment'][:, 0] == np.where(
                hg >= 0, g['fragment'][hg], -1))
            np.testing.assert_array_equal(h[c][hh, 0], lab[hg[hh]])
            for n in ('HL', 'ALL'):
                np.testing.assert_array_equal(comb[n][i]['hits'][c], h[c])
