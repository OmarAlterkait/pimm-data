"""Semantic invariants — not just 'key exists' but 'FK chain produces the
right values'. Complements the structural shape checks in
test_jaxtpc_task_matrix.py.
"""

import numpy as np

from pimm_data import JAXTPCDataset


def make_ds(root, modalities, **kw):
    defaults = dict(data_root=root, split='', dataset_name='sim',
                    modalities=modalities, label_key='pdg', min_deposits=0)
    defaults.update(kw)
    return JAXTPCDataset(**defaults)


# --- step + labl: deposit_to_track FK is row-aligned to step per volume ---

def test_labl_deposit_to_track_lengths_match_step_volumes(jaxtpc_data_root):
    """Σ over volumes of labl.vN.deposit_to_track == step.coord row count."""
    ds = make_ds(jaxtpc_data_root, ('step', 'labl'))
    d = ds.get_data(0)
    vid = d['step']['volume_id'].ravel()
    for vkey, vdata in d['labl'].items():
        vnum = int(vkey[1:])
        step_rows_in_vol = int((vid == vnum).sum())
        stt_rows = int(vdata['deposit_to_track'].shape[0])
        assert stt_rows == step_rows_in_vol, \
            f"vol {vnum}: labl.{vkey}.deposit_to_track len {stt_rows} " \
            f"!= step vol rows {step_rows_in_vol}"


def test_step_segment_is_raw_pdg(jaxtpc_data_root):
    """step.segment (label_key='pdg') contains raw PDG codes, not class indices.

    A class-remapped segment would only have a handful of small integers; raw
    PDG has a mix including typical MeV-stable-particle codes.
    """
    ds = make_ds(jaxtpc_data_root, ('step', 'labl'), label_key='pdg')
    d = ds.get_data(0)
    seg = d['step']['segment']
    unique = np.unique(seg[seg != -1])
    # PDG codes include values like 22 (photon), 11 (e-), 13 (mu), 2112 (n),
    # 2212 (proton), etc. — always absolute values >= 11 for interesting hits
    assert unique.max() > 20, (
        f"segment max {unique.max()} — does not look like raw PDG; expected "
        f"values like 22, 13, 211, 2212")


def test_step_instance_matches_labl_fk(jaxtpc_data_root):
    """step.instance is the per-deposit track_id copied from labl.

    Check per-volume: step.instance[vol_mask] == labl.vN.deposit_to_track.
    """
    ds = make_ds(jaxtpc_data_root, ('step', 'labl'))
    d = ds.get_data(0)
    vid = d['step']['volume_id'].ravel()
    inst = d['step']['instance']
    for vkey, vdata in d['labl'].items():
        vnum = int(vkey[1:])
        mask = vid == vnum
        if not mask.any():
            continue
        expected = vdata['deposit_to_track']
        got = inst[mask]
        assert np.array_equal(got, expected), \
            f"step.instance for vol {vnum} does not match " \
            f"labl.{vkey}.deposit_to_track"


# --- inst + labl: group_to_track → track_ids → track_{label_key} ---

def test_hits_segment_via_g2t_chain(jaxtpc_data_root):
    """Spot-check hits.segment[i] matches the full chain manually.

    For each plane, pick a few entries, read group_id, look up g2t[group_id]
    for that volume, then look up track_pdg via track_ids. Must match
    hits.segment[i].
    """
    ds = make_ds(jaxtpc_data_root, ('hits', 'labl'))
    d = ds.get_data(0)
    hits_raw = ds.hits_reader.read_event(ds.data_list[0])
    seg = d['hits']['segment']
    hits_instance = d['hits']['instance']
    plane_id = d['hits']['plane_id'].ravel()

    # Build a per-plane offset table so we can map flat index → plane
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(seg), size=min(50, len(seg)), replace=False)

    planes = d['hits']['planes']
    for fi in sample_idx:
        p = planes[int(plane_id[fi])]
        vol_num = p.split('_')[1]
        vkey = f'v{vol_num}'
        g2t = hits_raw[f'group_to_track_v{vol_num}']
        gid = int(hits_instance[fi])
        if gid < 0 or gid >= len(g2t):
            continue
        tid = int(g2t[gid])
        labl_v = d['labl'][vkey]
        tids = labl_v['track_ids']
        pos = np.searchsorted(np.sort(tids), tid)
        sorted_tids = np.sort(tids)
        if pos >= len(sorted_tids) or sorted_tids[pos] != tid:
            assert seg[fi] == -1, \
                f"no track match but segment={seg[fi]} (want -1)"
            continue
        # Look up pdg via same order
        order = np.argsort(tids)
        pdg_sorted = labl_v['track_pdg'][order]
        expected = int(pdg_sorted[pos])
        assert int(seg[fi]) == expected, \
            f"hits.segment[{fi}]={seg[fi]}, chain yields {expected}"


# --- the group correspondence (step <-> hits), no labels needed ---

def test_group_refs_without_labels(jaxtpc_data_root):
    """step + hits alone: every deposit and hit entry references a group row
    (or -1), and qs splits each group's charge over its deposits."""
    ds = make_ds(jaxtpc_data_root, ('step', 'hits'))
    d = ds.get_data(0)
    n_groups = len(d['group']['volume'])
    for on in ('step', 'hits'):
        ref = d[on]['group']
        assert ref.shape == (len(d[on]['coord']),)
        assert ((ref >= -1) & (ref < n_groups)).all()
        assert d[on]['_roles']['group'] == ('ref', 'group')
        # a group's points lie in that group's volume
        own = ref >= 0
        vol = (d[on]['volume_id'].ravel() if on == 'step' else
               np.array([int(p.split('_')[1]) for p in d[on]['planes']])
               [d[on]['plane_id'].ravel()])
        assert (d['group']['volume'][ref[own]] == vol[own]).all()
    if 'qs' in d['step']:
        ref = d['step']['group']
        tot = np.bincount(ref[ref >= 0], weights=d['step']['qs'][ref >= 0, 0],
                          minlength=n_groups)
        assert np.allclose(tot[np.unique(ref[ref >= 0])], 1.0, atol=1e-4)


# --- sensor is always label-free ---

def test_sensor_carries_no_labels(jaxtpc_data_root):
    """Sensor sub-dict must never carry segment/instance — design invariant."""
    for mods in [('sensor',), ('sensor', 'hits'), ('sensor', 'step'),
                 ('sensor', 'hits', 'labl'), ('sensor', 'step', 'labl'),
                 ('step', 'sensor', 'hits', 'labl')]:
        ds = make_ds(jaxtpc_data_root, mods)
        d = ds.get_data(0)
        assert 'segment' not in d['sensor'], f"sensor got segment in {mods}"
        assert 'instance' not in d['sensor'], f"sensor got instance in {mods}"


# --- group presence iff hits ---

def test_group_lifecycle(jaxtpc_data_root):
    """The group part appears exactly when hits is loaded (hits carries the
    group ids and deposit_to_group); labl alone does not add it."""
    for mods in [('step',), ('sensor',), ('step', 'labl'), ('step', 'sensor')]:
        d = make_ds(jaxtpc_data_root, mods).get_data(0)
        assert 'group' not in d and 'bridges' not in d, mods
    for mods in [('hits',), ('sensor', 'hits'), ('hits', 'labl'),
                 ('step', 'hits'), ('step', 'hits', 'labl'),
                 ('step', 'sensor', 'hits', 'labl')]:
        d = make_ds(jaxtpc_data_root, mods).get_data(0)
        assert 'group' in d and 'bridges' not in d, mods


# --- label absence ---

def test_no_labels_without_labl(jaxtpc_data_root):
    """Without labl, step.segment and hits.segment must be absent."""
    for mods in [('step',), ('hits',), ('step', 'sensor'), ('step', 'hits'),
                 ('sensor', 'hits'), ('step', 'sensor', 'hits')]:
        ds = make_ds(jaxtpc_data_root, mods)
        d = ds.get_data(0)
        if 'step' in d:
            assert 'segment' not in d['step'], f"step got segment in {mods}"
            assert 'instance' not in d['step'], f"step got instance in {mods}"
        if 'hits' in d:
            assert 'segment' not in d['hits'], f"hits got segment in {mods}"
            # hits.instance is always present (= group_id), NOT from labl


# --- plane_id consistency ---

def test_plane_id_is_dense_index(jaxtpc_data_root):
    """plane_id values are a dense 0..len(planes)-1 index into `planes` list."""
    ds = make_ds(jaxtpc_data_root, ('sensor', 'hits'))
    d = ds.get_data(0)
    for stream in ('sensor', 'hits'):
        plane_id = d[stream]['plane_id'].ravel()
        planes = d[stream]['planes']
        assert plane_id.max() < len(planes)
        assert plane_id.min() >= 0


# --- volume filter sanity ---

def test_volume_filter_shrinks_all_clouds(jaxtpc_data_root):
    ds_all = make_ds(jaxtpc_data_root, ('step', 'sensor', 'hits', 'labl'))
    ds_v0 = make_ds(jaxtpc_data_root, ('step', 'sensor', 'hits', 'labl'),
                    volume=0)
    a = ds_all.get_data(0)
    b = ds_v0.get_data(0)
    # volume=0 means only volume 0 labeled/physics content
    for stream in ('step', 'sensor', 'hits'):
        assert b[stream]['coord'].shape[0] < a[stream]['coord'].shape[0], \
            f"{stream}: filter did not shrink cloud"
    # labl: only v0 retained
    assert set(b['labl'].keys()) == {'v0'}
