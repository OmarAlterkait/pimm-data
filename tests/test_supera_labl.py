"""pysupera 4.x -> ``labl`` (``pimm_data.supera_labl``) -> JAXTPCDataset labels.

A minimal pysupera file is drawn against the format-2 synthetic JAXTPC sample.
Its ids are deliberately not row positions (pysupera's are not either), so any
place that indexes with an id instead of a row fails here. Expected labels are
computed straight from the pysupera arrays, independently of the code under
test."""

import os

import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data._shard_meta import clear_cache
from pimm_data.supera_labl import write_labl
from pimm_data.testing import SUPERA_FRAGMENTS, make_jaxtpc_sample

from _format2_helpers import ID_BASE, N_EVENTS, _to_tasks

TASK = 'jaxtpc_wire_j000000-000004'
FRAGMENTS = SUPERA_FRAGMENTS
EXPECT_MOTIF = {2: 1, 1: 0, 5: 4, 3: 3, 4: 2}


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    clear_cache()
    base = tmp_path_factory.mktemp('supera')
    src = make_jaxtpc_sample(str(base / 'f1'), n_events=N_EVENTS, n_files=1)
    root = _to_tasks(src, str(base / 'f2'), [TASK])
    task = os.path.join(root, 'j000000', TASK)
    hits = os.path.join(task, 'hits', f'sim_hits_{TASK}.h5')
    supera = os.path.join(task, 'supera', f'sim_supera_{TASK}.h5')
    labl = os.path.join(task, 'labl', f'sim_labl_{TASK}.h5')
    files = {m: os.path.join(root, 'j*', '*', m, '*.h5')
             for m in ('step', 'hits', 'labl')}
    return root, supera, hits, labl, files


def test_rows_point_where_the_ids_pointed(built):
    _, supera, _, labl, _ = built
    with h5py.File(supera) as su, h5py.File(labl) as lb:
        off = su['events/offsets'][:]
        g_off = su['groups/offsets'][:]
        for e in range(N_EVENTS):
            ids = su['particles/id'][off[e]:off[e + 1]]
            ev = lb[f'entry_{e:06d}']
            assert ev.attrs['event_id'] == ID_BASE + e
            back = lambda rows: np.where(rows >= 0, ids[np.maximum(rows, 0)], -1)
            for name, src in (('fragment_instance', 'frag_inst_id'),
                              ('parent', 'parent_id'), ('instance', 'inst_id')):
                np.testing.assert_array_equal(
                    back(ev['particle'][name][:]),
                    su[f'particles/{src}'][off[e]:off[e + 1]])
            # each group's fragment is pysupera's owner, verbatim
            g2t = np.concatenate([ev[f'volume_{v}/group_to_track'][:]
                                  for v in range(2)])
            gf = su['groups/fragment_id'][g_off[e]:g_off[e + 1]]
            got = np.concatenate([back(ev[f'volume_{v}/group_fragment'][:])
                                  for v in range(2)])
            np.testing.assert_array_equal(got, gf)
            tid = ev['track/id'][:]
            assert np.all(np.diff(tid) > 0)
            assert set(tid) == set(g2t[gf >= 0])
            assert 'fragment' not in ev['track']


def test_merge_counts_kinematics_and_parent_pdg(built):
    """Merge counts are copied; float64 kinematics are stored float32;
    ``parent_pdg`` is derived at load and equals the parent row's PDG."""
    from pimm_data._resolve import add_vocabulary
    _, supera, _, labl, _ = built
    with h5py.File(supera) as su, h5py.File(labl) as lb:
        off = su['events/offsets'][:]
        for e in range(N_EVENTS):
            p = lb[f'entry_{e:06d}/particle']
            sl = slice(off[e], off[e + 1])
            for name, src in (('fragment_merge_count', 'frag_merge_count'),
                              ('instance_merge_count', 'inst_merge_count')):
                np.testing.assert_array_equal(p[name][:],
                                              su[f'particles/{src}'][sl])
            assert p['start_mom'].dtype == np.float32
            assert p['mass'].dtype == np.float32
            part = {k: p[k][:] for k in p}
            add_vocabulary({'particle': part})
            np.testing.assert_array_equal(
                part['parent_pdg'], part['pdg'][part['parent']])


def test_dataset_labels_match_supera(built):
    root, supera, hits, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='all',
                       modalities=('step', 'hits'), le_is_stuff=False)
    assert len(ds) == N_EVENTS
    with h5py.File(supera) as su, \
            h5py.File(hits, 'r', libver='latest', swmr=True) as hi:
        off = su['events/offsets'][:]
        g_off = su['groups/offsets'][:]
        vo = su['groups/volume_offsets'][:]
        vi = su['groups/volume_offsets_index'][:]
        for e in range(N_EVENTS):
            x = ds.get_data(e)
            ids = su['particles/id'][off[e]:off[e + 1]]
            row_of = {int(i): r for r, i in enumerate(ids)}
            fsem = su['particles/frag_sem_type'][off[e]:off[e + 1]]
            finst = su['particles/frag_inst_id'][off[e]:off[e + 1]]
            pdg = su['particles/pdg'][off[e]:off[e + 1]]
            gfrag = su['groups/fragment_id'][g_off[e]:g_off[e + 1]]
            voff = vo[vi[e]:vi[e + 1]]
            isem = su['particles/inst_sem_type'][off[e]:off[e + 1]]
            # expected per deposit (pysupera's convention): the group's owner,
            # -1 if pysupera left it unowned; class le for an LE fragment,
            # else the instance's semantic type
            exp_f, exp_m, exp_i, exp_p = [], [], [], []
            for v in range(2):
                d2g = hi[f'entry_{e:06d}/volume_{v}/deposit_to_group'][:]
                g2t_v = hi[f'entry_{e:06d}/volume_{v}/group_to_track'][:]
                gf_v = gfrag[voff[v]:voff[v] + len(g2t_v)]
                fr = np.array([row_of.get(int(gf_v[g]), -1) for g in d2g])
                inst = np.array([row_of[int(finst[r])] if r >= 0 else -1
                                 for r in fr])
                exp_f.append(fr)
                exp_m.append([-1 if r < 0 else 4 if fsem[r] == 5
                              else EXPECT_MOTIF[isem[r]] for r in fr])
                exp_i.append(inst)
                exp_p.append([pdg[i] if i >= 0 else -1 for i in inst])
            s = x['step']
            np.testing.assert_array_equal(s['instance_fragment'][:, 0],
                                          np.concatenate(exp_f))
            np.testing.assert_array_equal(s['segment_motif'][:, 0],
                                          np.concatenate(exp_m))
            np.testing.assert_array_equal(s['instance_particle'][:, 0],
                                          np.concatenate(exp_i))
            # instances: muon (row 0) -> 2, photon (1) -> 0, le (2) -> 5;
            # a point of an LE fragment is le whatever its instance
            pid = s['segment_pid'][:, 0]
            inst = s['instance_particle'][:, 0]
            le = s['segment_motif'][:, 0] == 4
            for r, cls in ((0, 2), (1, 0), (2, 5)):
                assert np.all(pid[(inst == r) & ~le] == cls)
            assert np.all(pid[le] == 5)
            assert np.all(pid[inst < 0] == -1)
            # primaries are the two instances that are their own parent
            prim = s['is_primary'][:, 0]
            assert np.all(prim[inst == 0] == 1) and np.all(prim[inst == 2] == 0)
            # hits reach the same fragment through their group reference,
            # and the group table's track is labl's copy of group_to_track
            h = x['hits']
            assert h['instance_fragment'].shape[0] == h['coord'].shape[0]
            assert h['_roles']['group'] == ('ref', 'group')
            g = x['group']
            # hits carry pysupera's per-hit labels (hit_labels), exactly
            hf, hi_ = [], []
            for p in h['planes']:
                v, pk = int(p.split('_')[1]), p.split('_', 2)[2]
                hl = su[f'hit_labels/volume{v}/plane{pk}']
                a, b = hl['offsets'][e], hl['offsets'][e + 1]
                hf.append([row_of.get(int(i), -1) for i in hl['fragment_id'][a:b]])
                hi_.append([row_of.get(int(i), -1) for i in hl['instance_id'][a:b]])
            np.testing.assert_array_equal(h['instance_fragment'][:, 0],
                                          np.concatenate(hf))
            np.testing.assert_array_equal(h['instance_particle'][:, 0],
                                          np.concatenate(hi_))
            assert (np.concatenate(hf) != np.where(
                h['group'] >= 0, g['fragment'][h['group']], -1)).any()
            g2t = np.concatenate([hi[f'entry_{e:06d}/volume_{v}/group_to_track'][:]
                                  for v in range(2)])
            seen = g['track'] >= 0
            np.testing.assert_array_equal(
                x['track']['id'][g['track'][seen]], g2t[seen])
            assert not set(g2t[~seen]) & set(x['track']['id'])
            # step and hits carry nothing above the group
            for part in (s, h):
                assert not {'track', 'fragment', 'deposit_to_track'} & set(part)


def test_misaligned_event_id_raises(built, tmp_path):
    _, supera, hits, _, _ = built
    bad = str(tmp_path / 'bad.h5')
    with h5py.File(supera) as src, h5py.File(bad, 'w') as dst:
        for k in src:
            src.copy(k, dst)
        ids = dst['events/event_id'][:]
        ids[1] += 1
        dst['events/event_id'][...] = ids
    with pytest.raises(ValueError, match='event_id'):
        write_labl(bad, hits, str(tmp_path / 'out.h5'))


def test_le_is_stuff(built):
    """Each level's stuff is its own class: -1 in instance_fragment where the
    motif is le, in instance_particle where the pid is le; classes and
    instance_interaction are untouched."""
    root, _, _, _, files = built
    kw = dict(data_root=root, files=files, labels='all',
              modalities=('step', 'hits'))
    things, every = JAXTPCDataset(**kw), JAXTPCDataset(le_is_stuff=False, **kw)
    n_scatter = 0
    for idx in range(len(things)):
        for on in ('step', 'hits'):
            t, e = things.get_data(idx)[on], every.get_data(idx)[on]
            for k, cls in (('instance_fragment', 'segment_motif'),
                           ('instance_particle', 'segment_pid')):
                stuff = e[cls][:, 0] == (4 if cls == 'segment_motif' else 5)
                assert stuff.any() and (~stuff & (e[k][:, 0] >= 0)).any()
                assert (t[k][stuff] == -1).all()
                np.testing.assert_array_equal(t[k][~stuff], e[k][~stuff])
            for k in ('segment_motif', 'segment_pid', 'instance_interaction'):
                np.testing.assert_array_equal(t[k], e[k])
            # a low-energy scatter (pysupera's convention): le in motif and
            # PID; as stuff it has no fragment and no particle, without
            # le_is_stuff it keeps its host's (pysupera's include_le)
            m, p = e['segment_motif'][:, 0], e['segment_pid'][:, 0]
            scatter = m == 4
            assert (p[scatter] == 5).all()
            n_scatter += int((scatter & (e['instance_particle'][:, 0] >= 0)).sum())
            assert (t['instance_fragment'][scatter] == -1).all()
            assert (t['instance_particle'][scatter] == -1).all()
    assert n_scatter


def test_panoptic_levels_are_consistent(built):
    """At every level: one class per instance, stuff <=> no instance, and
    levels nest (fragment in one particle, particle in one interaction)."""
    from pimm_data._resolve import LEVELS
    root, _, _, _, files = built
    for stuff_on in (True, False):
        d = JAXTPCDataset(data_root=root, files=files, le_is_stuff=stuff_on,
                          labels=tuple(LEVELS), modalities=('step', 'hits'))
        for idx in range(len(d)):
            for on in ('step', 'hits'):
                s = {k: v[:, 0] for k, v in d.get_data(idx)[on].items()
                     if k.startswith(('segment_', 'instance_'))}
                for inst, cls, stuff in LEVELS.values():
                    ids = s[inst]
                    if cls is not None:
                        for u in np.unique(ids[ids >= 0]):   # LE points of a
                            c = s[cls][(ids == u) & (s[cls] != stuff)]  # host
                            assert len(np.unique(c)) <= 1    # are le (stuff)
                        if stuff_on:
                            np.testing.assert_array_equal(
                                s[cls] == stuff, (ids < 0) & (s[cls] >= 0))
                for inner, outer in (('instance_fragment', 'instance_particle'),
                                     ('instance_particle', 'instance_interaction')):
                    ids = s[inner]
                    for u in np.unique(ids[ids >= 0]):
                        assert len(np.unique(s[outer][ids == u])) == 1


def test_level_names_expand():
    from pimm_data._resolve import expand
    assert expand('particle') == ('segment_pid', 'instance_particle')
    assert expand(('fragment', 'segment_motif', 'interaction')) == (
        'segment_motif', 'instance_fragment', 'instance_interaction')
    with pytest.raises(ValueError, match='neither a level'):
        expand(('bogus',))


def test_segment_pdg_is_the_instances_pdg(built):
    root, _, _, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='all',
                       modalities=('step',), le_is_stuff=False)
    s = ds.get_data(0)['step']
    inst = s['instance_particle'][:, 0]
    assert set(np.unique(s['segment_pdg'][inst == 0, 0])) == {13}
    assert set(np.unique(s['segment_pdg'][inst < 0, 0])) <= {-1}


def test_keys_stay_aligned_through_sampling_and_collate(built):
    """Every per-point label survives GridSample/Dropout row-aligned, and
    collates — the keys pimm's PILArNet evaluators read, unchanged."""
    from pimm_data import collate_fn
    root, _, _, _, files = built
    keys = ('segment_motif', 'segment_pid', 'instance_particle',
            'instance_interaction', 'is_primary', 'momentum', 'vertex')
    ds = JAXTPCDataset(
        data_root=root, files=files, labels='all', modalities=('step',),
        transform=[
            dict(type='Apply', on='step', transforms=[
                dict(type='GridSample', grid_size=200.0, hash_type='fnv',
                     mode='train', return_grid_coord=True),
                dict(type='RandomDropout', dropout_ratio=0.3,
                     dropout_application_ratio=1.0),
            ]),
            dict(type='ToTensor'),
            dict(type='Collect', part='step',
                 keys=('coord', 'grid_coord') + keys,
                 feat_keys=('coord', 'energy')),
        ])
    raw = ds.get_data(0)['step']
    b = collate_fn([ds[0], ds[1]])
    n = b['coord'].shape[0]
    for k in keys:
        assert b[k].shape[0] == n, k
    assert b['vertex'].shape[1] == 3
    # sampled rows keep their (instance -> pid) pairing from the raw event
    pairs = set(zip(raw['instance_particle'][:, 0].tolist(),
                    raw['segment_pid'][:, 0].tolist()))
    e0 = int(b['offset'][0])
    got = set(zip(b['instance_particle'][:e0, 0].tolist(),
                  b['segment_pid'][:e0, 0].tolist()))
    assert got <= pairs


def test_refs_shift_through_collate_and_split(built):
    """('ref', part) columns batch like cross-store edges: shifted by the
    target part's running row count, and split_event undoes it."""
    from pimm_data import collate_fn, split_event
    from pimm_data.transform import Compose
    root, _, _, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='all',
                       modalities=('step',), le_is_stuff=False)
    pipe = Compose([dict(type='Collect', parts={
        'step': dict(keys=('coord', 'group')),
        'group': dict(keys=('fragment', 'volume'),
                      offset_keys_dict=dict(offset='fragment')),
        'particle': dict(keys=('pdg', 'fragment_instance'),
                         offset_keys_dict=dict(offset='pdg')),
    })])
    ev = [ds.get_data(i) for i in range(2)]
    b = collate_fn([pipe(e) for e in ev])
    assert b['_roles']['step_group'] == ('ref', 'group')
    n_g0 = len(ev[0]['group']['fragment'])
    n_p0 = len(ev[0]['particle']['pdg'])
    s1 = int(b['step_offset'][0])
    raw1 = ev[1]['step']['group']
    got1 = b['step_group'][s1:].numpy()
    np.testing.assert_array_equal(got1[raw1 >= 0], raw1[raw1 >= 0] + n_g0)
    assert np.all(got1[raw1 < 0] == -1)
    # a batched walk agrees with the per-event one
    g = b['group_fragment'][b['step_group'].clamp(min=0)]
    frag = np.where(b['step_group'].numpy() >= 0, g.numpy(), -1)
    exp = np.concatenate([ev[0]['step']['instance_fragment'][:, 0],
                          np.where(ev[1]['step']['instance_fragment'][:, 0] >= 0,
                                   ev[1]['step']['instance_fragment'][:, 0] + n_p0,
                                   -1)])
    np.testing.assert_array_equal(frag, exp)
    one = split_event(b, 1)
    np.testing.assert_array_equal(one['step_group'].numpy(), raw1)


def test_qs_is_derived_from_step_charge(built):
    """step.qs is each deposit's share of its group's step charge; it sums to
    one per group and needs nothing from the hits file."""
    root, _, _, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='all',
                       modalities=('step',))
    s = ds.get_data(0)['step']
    g, qs, q = s['group'], s['qs'][:, 0], s['charge'][:, 0]
    ok = g >= 0
    sums = np.bincount(g[ok], weights=qs[ok])
    assert np.allclose(sums[np.unique(g[ok])], 1.0, atol=1e-5)
    tot = np.bincount(g[ok], weights=q[ok])
    np.testing.assert_allclose(qs[ok], q[ok] / tot[g[ok]], rtol=1e-5)


def test_groups_without_labl_for_matching(built):
    """No labl: step and hits still share the group key (the matching
    regime), and nothing above the group appears."""
    root, _, _, _, files = built
    files = {m: f for m, f in files.items() if m != 'labl'}
    ds = JAXTPCDataset(data_root=root, files=files,
                       modalities=('step', 'hits'))
    x = ds.get_data(0)
    assert set(x['group']) == {'volume', '_roles'}
    for part in ('step', 'hits'):
        assert x[part]['_roles']['group'] == ('ref', 'group')
        assert not [k for k in x[part] if k.startswith('segment')]
    assert not {'particle', 'interaction'} & set(x)


def _make_sensor_config(step_path, out_path, n_ticks):
    """Just the sensor config the window needs (velocity 1.6 mm/us, 0.5 us
    ticks, no pre-window), volume_ranges copied from step."""
    with h5py.File(step_path) as s, h5py.File(out_path, 'w') as f:
        c = f.create_group('config')
        c['volume_ranges'] = s['config/volume_ranges'][:]
        c.attrs.update(velocity_cm_us=0.16, time_step_us=0.5,
                       num_time_steps=n_ticks, pre_window_us=0.0)
    return out_path


def test_deposit_in_window(built, tmp_path):
    """Arrival tick (t0 + drift to the outer anode / v) / dt in [0, n); the
    dataset exposes it per step deposit, in volume order."""
    root, supera, hits, _, files = built
    step = os.path.join(root, 'j000000', TASK, 'step', f'sim_step_{TASK}.h5')
    sensor = _make_sensor_config(step, str(tmp_path / 'sensor.h5'), 1350)
    with pytest.raises(ValueError, match='together'):
        write_labl(supera, hits, str(tmp_path / 'x.h5'), step_path=step)
    ldir = tmp_path / 'labl'
    ldir.mkdir()
    labl = write_labl(supera, hits, str(ldir / f'sim_labl_{TASK}.h5'),
                      step_path=step, sensor_path=sensor)
    with h5py.File(step) as s, h5py.File(labl) as lb:
        ranges = s['config/volume_ranges'][:]
        key = sorted(k for k in lb if k.startswith('entry_'))[0]
        got_all, frac = [], []
        for v in range(len(ranges)):
            sv = s[f'{key}/volume_{v}']
            x = sv.attrs['pos_origin_x'] + sv['positions'][:, 0] \
                * sv.attrs['pos_step_mm']
            anode = ranges[v, 0, np.argmax(np.abs(ranges[v, 0]))]
            tick = np.floor((sv['t0_us'][:].astype(np.float64)
                             + np.abs(x - anode) / 1.6) / 0.5)
            got = lb[f'{key}/volume_{v}/deposit_in_window'][:]
            np.testing.assert_array_equal(got, (tick >= 0) & (tick < 1350))
            got_all.append(got)
        inw = np.concatenate(got_all)
        assert 0 < inw.mean() < 1           # the window splits the sample
    files = dict(files, labl=labl)
    x = JAXTPCDataset(data_root=root, files=files, labels='all',
                      modalities=('step',)).get_data(0)
    assert np.all(np.diff(x['step']['volume_id'].ravel()) >= 0)
    np.testing.assert_array_equal(x['step']['in_window'][:, 0], inw)


def test_campaign_writes_the_jaxtpc_tree_and_resumes(built, tmp_path):
    """campaign: every pysupera file of a doraemon tree -> labl/ beside where
    JAXTPC's modalities live; reruns skip finished files; a task missing a
    JAXTPC modality is reported, not fatal."""
    from pimm_data.supera_labl import write_campaign
    root, supera, hits, labl, _ = built
    camp = tmp_path / 'camp'
    src = os.path.join(root, 'j000000', TASK)
    dst = camp / 'jaxtpc_wire' / 'j000000' / TASK
    for m in ('step', 'hits'):
        (dst / m).mkdir(parents=True)
        f = os.listdir(os.path.join(src, m))[0]
        os.symlink(os.path.join(src, m, f), dst / m / f)
    (dst / 'sensor').mkdir()
    _make_sensor_config(str(dst / 'step' / f'sim_step_{TASK}.h5'),
                        str(dst / 'sensor' / f'sim_sensor_{TASK}.h5'), 1350)
    task = TASK[len('jaxtpc_wire_'):]
    for t in (task, 'j000005-000009'):          # the second has no JAXTPC
        d = camp / 'supera_wire' / 'j000000' / f'supera_wire_{t}'
        d.mkdir(parents=True)
        os.symlink(supera, d / f'supera_wire_{t}.h5')
    out = tmp_path / 'out'
    failed = write_campaign(str(camp), str(out), ('wire',), workers=2,
                            log=lambda m: None)
    got = out / 'jaxtpc_wire' / 'j000000' / TASK / 'labl' / f'sim_labl_{TASK}.h5'
    assert got.exists() and len(failed) == 1
    assert 'FileNotFoundError' in next(iter(failed.values()))
    with h5py.File(got) as a, h5py.File(labl) as b:
        k = 'entry_000000'
        np.testing.assert_array_equal(a[f'{k}/track/id'][:],
                                      b[f'{k}/track/id'][:])
        assert 'deposit_in_window' in a[f'{k}/volume_0']
    mtime = got.stat().st_mtime
    write_campaign(str(camp), str(out), ('wire',), log=lambda m: None)
    assert got.stat().st_mtime == mtime          # skipped, not rewritten
    # linked: the output tree is a campaign tree of its own
    write_campaign(str(camp), str(out), ('wire',), link_inputs=True,
                   log=lambda m: None)
    task_dir = out / 'jaxtpc_wire' / 'j000000' / TASK
    assert (task_dir / 'hits').is_symlink()
    assert got.stat().st_mtime == mtime
    ds = JAXTPCDataset(data_root=str(out / 'jaxtpc_wire'), runs='j*/*',
                       labels='all', modalities=('step', 'hits'))
    assert len(ds) == N_EVENTS
    assert 'segment_motif' in ds.get_data(0)['hits']


def test_shared_step_layout_loads_the_same(built, tmp_path):
    """The group arrays moved from hits to step (``to_shared_step``), and labl
    written from that step: every modality combination loads array-for-array
    what the produced layout loads, and the new labl has no deposit_to_group."""
    from pimm_data.testing import to_shared_step
    root, supera, hits, _, files = built
    step = os.path.join(root, 'j000000', TASK, 'step', f'sim_step_{TASK}.h5')
    sensor = _make_sensor_config(step, str(tmp_path / 'sensor.h5'), 1350)
    old, new = tmp_path / 'old', tmp_path / 'new'
    for d in (old, new):
        d.mkdir()
    old_labl = write_labl(supera, hits, str(old / f'sim_labl_{TASK}.h5'),
                          step_path=step, sensor_path=sensor)
    n_step, n_hits = to_shared_step(step, hits, str(new / f'sim_step_{TASK}.h5'),
                                    str(new / f'sim_hits_{TASK}.h5'))
    new_labl = write_labl(supera, n_hits, str(new / f'sim_labl_{TASK}.h5'),
                          step_path=n_step, sensor_path=sensor,
                          track_path=hits)
    with h5py.File(new_labl) as lb, h5py.File(n_hits) as hh:
        for e in (k for k in lb if k.startswith('entry_')):
            for v in (k for k in lb[e] if k.startswith('volume_')):
                assert 'deposit_to_group' not in lb[e][v]
                assert 'group_to_track' in lb[e][v]
                assert not set(hh[e][v]) & {'deposit_to_group', 'qs_fractions',
                                            'group_to_track'}
    lay = {'old': dict(files, labl=old_labl),
           'new': dict(step=n_step, hits=n_hits, labl=new_labl)}

    def same(a, b, path=''):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for k in a:
                if k not in ('name', 'split'):
                    same(a[k], b[k], f'{path}/{k}')
        elif isinstance(a, np.ndarray) and path.endswith('/qs'):
            # derived from charge (old) vs JAXTPC's stored float16 (new)
            np.testing.assert_allclose(a, b, atol=5e-4, err_msg=path)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=path)
        else:
            assert a == b, path

    for mods in (('step',), ('hits',), ('step', 'hits')):
        for labels in (None, 'all'):
            if labels is None and mods == ('step',):
                continue    # old layout: step alone has no groups
            got = {}
            for k, f in lay.items():
                f = {m: f[m] for m in mods + (('labl',) if labels else ())}
                ds = JAXTPCDataset(data_root=root, files=f, labels=labels,
                                   modalities=mods)
                got[k] = [ds.get_data(i) for i in range(len(ds))]
            for a, b in zip(got['old'], got['new']):
                same(a, b, f'{mods} {labels}')
    # what only the new layout gives: groups and qs from step alone
    s = JAXTPCDataset(data_root=root, files={'step': n_step},
                      modalities=('step',)).get_data(0)['step']
    assert (s['group'] >= 0).all() and 'qs' in s


def test_issues_and_settings_reach_labl(built, tmp_path):
    """pysupera's per-event issues become entry attrs; its run attrs and the
    effective settings (config plus readout overlay) land in /config."""
    import shutil
    root, supera, hits, _, _ = built
    su = shutil.copy(supera, tmp_path / 'su.h5')
    with h5py.File(su, 'a') as f:
        for k in [k for k in ('issues', 'provenance') if k in f]:
            del f[k]
        I = f.create_group('issues')
        I.attrs['n_outside_readout_window'] = 1
        I['event'] = np.array([1])
        I['code'] = np.array([4], np.int32)
        I['geant4_trackid'] = np.array([-1])
        I['message'] = np.array([b'none of its deposits arrived inside the window'],
                                dtype=h5py.string_dtype())
        f.attrs['kinematics_steps'] = 'visible'
        p = f.create_group('provenance/supera_pixel/j0/config')
        p.create_group('pysupera_config/particle').attrs['min_pc_size'] = 5
        p.create_group('pysupera_readout/particle').attrs['min_pc_size'] = 63
    out = write_labl(str(su), hits, str(tmp_path / 'l.h5'), compression='gzip')
    with h5py.File(out) as lb:
        c = lb['config'].attrs
        assert c['supera_kinematics_steps'] == 'visible'
        assert c['supera_issues_n_outside_readout_window'] == 1
        assert c['supera_settings/pysupera_config/particle/min_pc_size'] == 5
        assert c['supera_settings/pysupera_readout/particle/min_pc_size'] == 63
        e = sorted(k for k in lb if k.startswith('entry_'))
        assert list(lb[e[1]].attrs['supera_issue_codes']) == [4]
        assert 'window' in lb[e[1]].attrs['supera_issue_messages'][0]
        assert 'supera_issue_codes' not in lb[e[0]].attrs


def test_plane_local_layout_loads_the_same(built, tmp_path):
    """Hits numbered per plane (``to_plane_local``: volume groups renumbered,
    blocks shuffled, plane ids 0..G-1, labl plane maps): with labl every
    deposit and hit gets the labels the shared-step layout gives it (hits as a
    multiset, since their order changed); without labl, hits carry only the
    plane-local id and no reference into ``group``."""
    from pimm_data.testing import to_plane_local, to_shared_step
    root, supera, hits, _, files = built
    step = os.path.join(root, 'j000000', TASK, 'step', f'sim_step_{TASK}.h5')
    sensor = _make_sensor_config(step, str(tmp_path / 'sensor.h5'), 1350)
    a, b = tmp_path / 'a', tmp_path / 'b'
    for d in (a, b):
        d.mkdir()
    s1, h1 = to_shared_step(step, hits, str(a / f'sim_step_{TASK}.h5'),
                            str(a / f'sim_hits_{TASK}.h5'))
    l1 = write_labl(supera, h1, str(a / f'sim_labl_{TASK}.h5'),
                    step_path=s1, sensor_path=sensor, track_path=hits)
    s2, h2, l2 = to_plane_local(s1, h1, l1, str(b / f'sim_step_{TASK}.h5'),
                                str(b / f'sim_hits_{TASK}.h5'),
                                str(b / f'sim_labl_{TASK}.h5'), seed=3)
    keys = None
    for mods in (('step',), ('hits',), ('step', 'hits')):
        dsa = JAXTPCDataset(data_root=root, labels='all', modalities=mods,
                            files={m: f for m, f in dict(step=s1, hits=h1, labl=l1).items()
                                   if m in mods + ('labl',)})
        dsb = JAXTPCDataset(data_root=root, labels='all', modalities=mods,
                            files={m: f for m, f in dict(step=s2, hits=h2, labl=l2).items()
                                   if m in mods + ('labl',)})
        assert len(dsa) == len(dsb)
        for i in range(len(dsa)):
            x, y = dsa.get_data(i), dsb.get_data(i)
            for part in mods:
                px, py = x[part], y[part]
                keys = sorted(k for k in px if k.startswith(
                    ('segment', 'instance', 'is_primary', 'momentum', 'vertex')))
                assert keys and keys == sorted(k for k in py if k in keys or k.startswith(
                    ('segment', 'instance', 'is_primary', 'momentum', 'vertex'))), part
                cols = [px['coord'], px['energy']] + [px[k] for k in keys]
                if part == 'hits':
                    cols.insert(0, px['plane_id'])
                rows_x = np.column_stack([np.asarray(c, np.float64).reshape(len(c), -1)
                                          for c in cols])
                cols = [py['coord'], py['energy']] + [py[k] for k in keys]
                if part == 'hits':
                    cols.insert(0, py['plane_id'])
                rows_y = np.column_stack([np.asarray(c, np.float64).reshape(len(c), -1)
                                          for c in cols])
                if part == 'step':      # rows unchanged
                    np.testing.assert_array_equal(rows_x, rows_y, err_msg=f'{mods} {i}')
                else:
                    sx = rows_x[np.lexsort(rows_x.T[::-1])]
                    sy = rows_y[np.lexsort(rows_y.T[::-1])]
                    np.testing.assert_array_equal(sx, sy, err_msg=f'{mods} {i}')
                    assert not (py['group'] == px['group']).all()
    # without labl: no cross-plane / step key for hits, only the plane-local id
    y = JAXTPCDataset(data_root=root, files=dict(step=s2, hits=h2),
                      modalities=('step', 'hits')).get_data(0)
    assert 'group' not in y['hits'] and 'group' not in y['hits'].get('_roles', {})
    assert len(y['hits']['plane_group']) == len(y['hits']['coord'])
    assert y['step']['_roles']['group'] == ('ref', 'group')
    # labl without a plane map on plane-local hits is refused
    import shutil
    (tmp_path / 'nomap').mkdir()
    l3 = str(tmp_path / 'nomap' / f'sim_labl_{TASK}.h5')
    shutil.copyfile(l2, l3)
    with h5py.File(l3, 'a') as lb:
        e = next(k for k in lb if k.startswith('entry_'))
        v = next(k for k in lb[e] if k.startswith('volume_'))
        p = next(k for k in lb[e][v] if k.startswith('plane_'))
        del lb[e][v][p]['group']
    clear_cache()
    with pytest.raises(ValueError, match='plane-local'):
        ds = JAXTPCDataset(data_root=root, labels='all', modalities=('hits',),
                           files=dict(hits=h2, labl=l3))
        [ds.get_data(i) for i in range(len(ds))]
