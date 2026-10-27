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
from pimm_data.testing import make_jaxtpc_sample

from _format2_helpers import ID_BASE, N_EVENTS, _to_format2

TASK = 'jaxtpc_wire_j000000-000004'
# (pdg, sem_type) of fragment rows; rows 0..2 head instances
FRAGMENTS = [(13, 2), (22, 1), (2212, 5), (11, 3), (11, 4), (321, 2)]
EXPECT_MOTIF = {2: 1, 1: 0, 5: 4, 3: 3, 4: 2}


def _make_supera(hits_path, out_path, seed=0):
    """One pysupera-4.1-shaped file for every entry of ``hits_path``.

    Per event: the six FRAGMENTS (rows 0-5) plus one merged delta-ray row (6).
    Rows 0, 1, 2 head instances; fragments 3, 4 belong to instance 0 and 5 to
    instance 1. Interactions: rows 0 and 1; instance 2 is in interaction 1.
    Groups are owned at random (some unowned)."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in (
        'id', 'frag_id', 'inst_id', 'frag_inst_id', 'frag_parent_id',
        'inst_parent_id', 'parent_id', 'ancestor_id', 'interaction_id',
        'pdg', 'sem_type', 'frag_sem_type', 'inst_sem_type',
        'interaction_type', 'geant4_trackid', 'geant4_parent_trackid',
        'geant4_parent_pdg', 'mass')}
    kin = {k: [] for k in ('start_pos', 'start_mom', 'first_pos', 'first_mom',
                           'last_pos', 'last_mom', 'end_pos', 'end_mom')}
    p_off, g_off, v_off, v_idx, i_off = [0], [0], [], [0], [0]
    frag_ids, inter = [], {k: [] for k in ('x', 'y', 'z', 'time', 'vertex_id',
                                           'energy_sum', 'ke_sum')}
    with h5py.File(hits_path, 'r') as hi:
        entries = sorted(k for k in hi if k.startswith('entry_'))
        event_ids = [int(hi[k].attrs['event_id']) for k in entries]
        for k in entries:
            ids = rng.permutation(1000)[:7] + 50          # never row positions
            inst_of_frag = [0, 1, 2, 0, 0, 1]
            for r in range(7):
                fr = r < 6
                pdg, sem = FRAGMENTS[r] if fr else (11, 3)
                inst = inst_of_frag[r] if fr else 0
                cols['id'].append(ids[r])
                cols['frag_id'].append(ids[r] if fr else -1)
                cols['inst_id'].append(ids[r] if r < 3 else -1)
                cols['frag_inst_id'].append(ids[inst] if fr else -1)
                cols['frag_parent_id'].append(ids[r])
                cols['inst_parent_id'].append(ids[r] if r < 3 else -1)
                cols['parent_id'].append(ids[r] if r in (0, 1) else ids[0])
                cols['ancestor_id'].append(ids[0] if r != 1 else ids[1])
                cols['interaction_id'].append(1 if inst == 2 else 0)
                cols['pdg'].append(pdg)
                cols['sem_type'].append(sem)
                cols['frag_sem_type'].append(sem if fr else -1)
                cols['inst_sem_type'].append(FRAGMENTS[inst][1])
                cols['interaction_type'].append(r)
                cols['geant4_trackid'].append(r + 1)
                cols['geant4_parent_trackid'].append(-1 if r < 2 else 1)
                cols['geant4_parent_pdg'].append(-1 if r < 2 else 13)
                cols['mass'].append(105.66)
            for c in kin:
                kin[c].append(rng.normal(size=(7, 4)) * 100)
            p_off.append(p_off[-1] + 7)
            for c, v in zip(('x', 'y', 'z', 'time', 'vertex_id', 'energy_sum',
                             'ke_sum'), rng.normal(size=(7, 2)) * 100):
                inter[c].extend(v)
            i_off.append(i_off[-1] + 2)
            # owner is a function of the group's track (as in pysupera);
            # some groups of an owned track stay unowned (out of window)
            vks = sorted(x for x in hi[k] if x.startswith('volume_'))
            tracks = np.unique(np.concatenate(
                [hi[k][vk]['group_to_track'][:] for vk in vks]))
            owner = dict(zip(tracks.tolist(),
                             rng.integers(-1, 6, size=len(tracks)).tolist()))
            base = 0
            for vk in vks:
                g2t = hi[k][vk]['group_to_track'][:]
                n_g = len(g2t)
                own = np.array([owner[int(t)] for t in g2t])
                own[rng.random(n_g) < 0.1] = -1
                own[0] = -1
                frag_ids.extend(np.where(own >= 0, ids[np.maximum(own, 0)], -1))
                v_off.append(base)
                base += n_g
            g_off.append(g_off[-1] + base)
            v_idx.append(len(v_off))
    with h5py.File(out_path, 'w') as f:
        f['format_version'] = '4.1.0'
        f['n_events'] = len(entries)
        f.attrs['readout_type'] = 'wire'
        ev = f.create_group('events')
        ev['offsets'] = np.array(p_off, np.int64)
        ev['jaxtpc_entry'] = np.array([int(k[6:]) for k in entries], np.int64)
        ev['event_id'] = np.array(event_ids, np.int64)
        ev['job_id'] = np.zeros(len(entries), np.int64)
        ev['job_event'] = np.arange(len(entries), dtype=np.int64)
        g = f.create_group('groups')
        g['fragment_id'] = np.array(frag_ids, np.int32)
        g['is_le'] = np.zeros(len(frag_ids), np.int8)
        g['offsets'] = np.array(g_off, np.int64)
        g['volume_offsets'] = np.array(v_off, np.int64)
        g['volume_offsets_index'] = np.array(v_idx, np.int64)
        p = f.create_group('particles')
        for c, v in cols.items():
            p[c] = np.array(v, np.float64 if c == 'mass' else np.int32)
        for c, v in kin.items():
            p[c] = np.concatenate(v)
        i = f.create_group('inter')
        for c, v in inter.items():
            i[c] = np.array(v, np.float32)
        i['offsets'] = np.array(i_off, np.int64)
        i['reaction'] = np.array([b''] * len(inter['x']), dtype=object)
    return out_path


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    clear_cache()
    base = tmp_path_factory.mktemp('supera')
    src = make_jaxtpc_sample(str(base / 'f1'), n_events=N_EVENTS, n_files=1)
    root = _to_format2(src, str(base / 'f2'), [TASK])
    hits = os.path.join(root, 'j000000', TASK, 'hits', f'sim_hits_{TASK}.h5')
    supera = _make_supera(hits, str(base / 'supera.h5'))
    labl_dir = base / 'labl'
    labl_dir.mkdir()
    labl = write_labl(supera, hits, str(labl_dir / f'sim_labl_{TASK}.h5'))
    files = {m: os.path.join(root, 'j*', '*', m, '*.h5')
             for m in ('step', 'hits')}
    files['labl'] = labl
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
            # every owned group's fragment is its track's
            g2t = np.concatenate([ev[f'volume_{v}/group_track'][:]
                                  for v in range(2)])
            gf = su['groups/fragment_id'][g_off[e]:g_off[e + 1]]
            tid, tf = ev['track/id'][:], back(ev['track/fragment'][:])
            assert np.all(np.diff(tid) > 0)
            np.testing.assert_array_equal(
                tf[np.searchsorted(tid, g2t[gf >= 0])], gf[gf >= 0])
            assert set(tid) == set(g2t[gf >= 0])


def test_dataset_labels_match_supera(built):
    root, supera, hits, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='motif',
                       modalities=('step', 'hits'))
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
            # a group's fragment is its track's owner (from any owned group)
            t_own = {}
            for v in range(2):
                g2t_v = hi[f'entry_{e:06d}/volume_{v}/group_to_track'][:]
                gf_v = gfrag[voff[v]:voff[v] + len(g2t_v)]
                t_own.update(zip(g2t_v[gf_v >= 0].tolist(),
                                 gf_v[gf_v >= 0].tolist()))
            # expected per deposit, volume by volume (step concatenates them)
            exp_f, exp_m, exp_i, exp_p = [], [], [], []
            for v in range(2):
                d2g = hi[f'entry_{e:06d}/volume_{v}/deposit_to_group'][:]
                g2t_v = hi[f'entry_{e:06d}/volume_{v}/group_to_track'][:]
                fid = np.array([t_own.get(int(t), -1) for t in g2t_v[d2g]])
                fr = np.array([row_of.get(int(i), -1) for i in fid])
                inst = np.array([row_of[int(finst[r])] if r >= 0 else -1
                                 for r in fr])
                exp_f.append(fr)
                exp_m.append([EXPECT_MOTIF[fsem[r]] if r >= 0 else -1
                              for r in fr])
                exp_i.append(inst)
                exp_p.append([pdg[i] if i >= 0 else -1 for i in inst])
            s = x['step']
            np.testing.assert_array_equal(s['instance_fragment'][:, 0],
                                          np.concatenate(exp_f))
            np.testing.assert_array_equal(s['segment_motif'][:, 0],
                                          np.concatenate(exp_m))
            np.testing.assert_array_equal(s['instance_particle'][:, 0],
                                          np.concatenate(exp_i))
            np.testing.assert_array_equal(s['segment'], s['segment_motif'][:, 0])
            np.testing.assert_array_equal(s['instance'],
                                          s['instance_particle'][:, 0])
            # instances: muon (row 0) -> 2, photon (1) -> 0, le (2) -> 5
            pid = s['segment_pid'][:, 0]
            inst = s['instance_particle'][:, 0]
            for r, cls in ((0, 2), (1, 0), (2, 5)):
                assert np.all(pid[inst == r] == cls)
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
            np.testing.assert_array_equal(
                h['instance_fragment'][:, 0], np.where(
                    h['group'] >= 0, g['fragment'][h['group']], -1))
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


def test_label_key_pdg_keeps_remap_configs_working(built):
    root, _, _, _, files = built
    ds = JAXTPCDataset(data_root=root, files=files, labels='pdg',
                       modalities=('step',))
    s = ds.get_data(0)['step']
    inst = s['instance_particle'][:, 0]
    assert set(np.unique(s['segment'][inst == 0])) == {13}
    assert set(np.unique(s['segment'][inst < 0])) <= {-1}


def test_keys_stay_aligned_through_sampling_and_collate(built):
    """Every per-point label survives GridSample/Dropout row-aligned, and
    collates — the keys pimm's PILArNet evaluators read, unchanged."""
    from pimm_data import collate_fn
    root, _, _, _, files = built
    keys = ('segment_motif', 'segment_pid', 'instance_particle',
            'instance_interaction', 'is_primary', 'momentum', 'vertex')
    ds = JAXTPCDataset(
        data_root=root, files=files, labels='motif', modalities=('step',),
        transform=[
            dict(type='Apply', on='step', transforms=[
                dict(type='GridSample', grid_size=200.0, hash_type='fnv',
                     mode='train', return_grid_coord=True),
                dict(type='RandomDropout', dropout_ratio=0.3,
                     dropout_application_ratio=1.0),
            ]),
            dict(type='ToTensor'),
            dict(type='Collect', part='step',
                 keys=('coord', 'grid_coord', 'segment', 'instance') + keys,
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
    ds = JAXTPCDataset(data_root=root, files=files, labels='motif',
                       modalities=('step',))
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
    ds = JAXTPCDataset(data_root=root, files=files, labels='motif',
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
        assert 'segment' not in x[part]
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
    x = JAXTPCDataset(data_root=root, files=files, labels='motif',
                      modalities=('step',)).get_data(0)
    assert np.all(np.diff(x['step']['volume_id'].ravel()) >= 0)
    np.testing.assert_array_equal(x['step']['in_window'][:, 0], inw)
