"""pysupera is the reference: every label JAXTPCDataset builds equals pysupera's own.

Real data only. Point it at one pysupera file and the JAXTPC run (step/hits/labl) it was
made from, in the produced hit order (the shared-step layout, e.g. subset_100k):

    PYSUPERA_GOLDEN_SUPERA=.../supera_wire_j000000-000004.h5 \\
    PYSUPERA_GOLDEN_JAXTPC=.../subset_100k/jaxtpc_wire \\
    PYSUPERA_GOLDEN_RUN=j000000/jaxtpc_wire_j000000-000004 \\
    [PYSUPERA_GOLDEN_EVENTS=0,500,999] pytest tests/test_pysupera_golden.py

The expected labels are computed from the pysupera file alone, the way pysupera's own
``torchdata`` builds its panoptic target: per instance row, its points
``[inst_pc_start, inst_pc_end)`` get the instance, its interaction (``inter/pc_*``), and
its ``inst_sem_type``, except the LE run ``[inst_pc_le_start, inst_pc_end)``, which is
kLEScatter. A fragment's class / instance / interaction is that of the instance range
holding its own points (deposits reach it through ``groups/fragment_id``). Per hit,
pysupera's ``hit_labels`` are the reference: its instance, its fragment, ``is_le``, and
the instance row's ``inst_sem_type`` / ``interaction_id``."""

import os

import h5py
import numpy as np
import pytest

SUPERA = os.environ.get('PYSUPERA_GOLDEN_SUPERA')
ROOT = os.environ.get('PYSUPERA_GOLDEN_JAXTPC')
RUN = os.environ.get('PYSUPERA_GOLDEN_RUN')
EVENTS = [int(x) for x in os.environ.get('PYSUPERA_GOLDEN_EVENTS', '0,1,2').split(',')]
pytestmark = pytest.mark.skipif(
    not (SUPERA and ROOT and RUN),
    reason='set PYSUPERA_GOLDEN_SUPERA, PYSUPERA_GOLDEN_JAXTPC, PYSUPERA_GOLDEN_RUN')

LE_SEM = 5
MOTIF_OF_SEM = np.array([-1, 0, 1, 3, 2, 4, -1])   # pysupera SemanticType -> motif
LE_MOTIF, LE_PID = 4, 5


def test_pysupera_level_expands_to_its_target():
    from pimm_data._resolve import expand
    assert expand(('pysupera',)) == ('segment_motif', 'instance_particle')


@pytest.fixture(scope='module')
def golden():
    import hdf5plugin  # noqa: F401
    from pimm_data import JAXTPCDataset
    su = h5py.File(SUPERA, 'r')
    ro = su.attrs['readout_type']
    ro = ro.decode() if isinstance(ro, bytes) else str(ro)
    kw = dict(data_root=ROOT, runs=RUN, dataset_name=f'sim_{ro}',
              modalities=('step', 'hits'), labels='all')
    return su, JAXTPCDataset(le_is_stuff=False, **kw), JAXTPCDataset(le_is_stuff=True, **kw)


def _expected(su, e):
    """pysupera's own answer for event e: per particle row, its torchdata class,
    instance and interaction (via the point ranges), and the row lookup."""
    off, ioff, goff = (su['events/offsets'], su['inter/offsets'], su['groups/offsets'])
    a, b = int(off[e]), int(off[e + 1])
    P = {c: su['particles'][c][a:b] for c in su['particles']}
    ids = P['id'].astype(np.int64)
    srt = np.argsort(ids)
    sid = ids[srt]

    def rows(v):
        v = np.asarray(v, np.int64)
        out = np.full(v.shape, -1, np.int64)
        m = v >= 0
        out[m] = srt[np.searchsorted(sid, v[m])]
        assert np.array_equal(sid[np.searchsorted(sid, v[m])], v[m]), 'dangling id'
        return out

    n = b - a
    r = np.arange(n)
    frag, inst = rows(P['frag_id']), rows(P['inst_id'])
    isem = P['inst_sem_type'].astype(np.int64)
    ir = np.flatnonzero((inst == r) & (P['inst_pc_end'] > P['inst_pc_start']))
    ir = ir[np.argsort(P['inst_pc_start'][ir])]
    ist, ile = P['inst_pc_start'][ir], P['inst_pc_le_start'][ir]
    cls, ins = np.full(n, -1), np.full(n, -1)
    fr = np.flatnonzero((frag == r) & (P['frag_pc_end'] > P['frag_pc_start']))
    k = np.searchsorted(ist, P['frag_pc_start'][fr], side='right') - 1
    assert (k >= 0).all() and (P['frag_pc_end'][fr] <= P['inst_pc_end'][ir[k]]).all(), \
        'a fragment outside every instance range'
    ins[fr] = ir[k]
    le = (ile[k] >= 0) & (P['frag_pc_start'][fr] >= ile[k])
    cls[fr] = np.where(le, LE_MOTIF, MOTIF_OF_SEM[np.clip(isem[ir[k]], 0, 6)])
    ipc = su['inter/pc_start'][int(ioff[e]):int(ioff[e + 1])]
    intr = np.full(n, -1)
    intr[ir] = np.searchsorted(ipc, ist, side='right') - 1
    gfrag = rows(su['groups/fragment_id'][int(goff[e]):int(goff[e + 1])])
    return dict(rows=rows, cls=cls, ins=ins, intr=intr, gfrag=gfrag, isem=isem,
                inter_id=P['interaction_id'].astype(np.int64))


def _hit_labels(su, planes, e, rows):
    hl = {}
    for vg in su['hit_labels'].values():
        for g in vg.values():
            s = g.attrs['source']
            hl[(int(g.attrs['volume']), s.decode() if isinstance(s, bytes) else str(s))] = g
    f, i, le = [], [], []
    for p in planes:
        g = hl[(int(p.split('_')[1]), p.split('_', 2)[2])]
        a, b = int(g['offsets'][e]), int(g['offsets'][e + 1])
        f.append(rows(g['fragment_id'][a:b]))
        i.append(rows(g['instance_id'][a:b]))
        le.append(g['is_le'][a:b].astype(bool))
    return np.concatenate(f), np.concatenate(i), np.concatenate(le)


@pytest.mark.parametrize('e', EVENTS)
def test_labels_are_pysupera_s(golden, e):
    su, keep, stuff = golden
    x, y = keep.get_data(e), stuff.get_data(e)
    g = _expected(su, e)
    # the group owner is pysupera's
    np.testing.assert_array_equal(x['group']['fragment'], g['gfrag'])
    # per hit: pysupera's hit_labels, off-sensor and split groups included
    hf, hi, hle = _hit_labels(su, x['hits']['planes'], e, g['rows'])
    np.testing.assert_array_equal(x['hits']['instance_fragment'][:, 0], hf)
    np.testing.assert_array_equal(x['hits']['instance_particle'][:, 0], hi)
    for part in ('step', 'hits'):
        s, t = x[part], y[part]
        f = s['instance_fragment'][:, 0]
        o = f >= 0
        fo = np.maximum(f, 0)
        if part == 'step':
            # torchdata's class, instance and interaction (include_le=True)
            cls, ins = np.where(o, g['cls'][fo], -1), np.where(o, g['ins'][fo], -1)
            intr = np.where(o, g['intr'][np.maximum(ins, 0)], -1)
        else:
            # hit_labels: the hit's instance, its type, le from is_le
            ins = hi
            io = np.maximum(hi, 0)
            cls = np.where(hi >= 0, np.where(hle, LE_MOTIF, MOTIF_OF_SEM[np.clip(g['isem'][io], 0, 6)]), -1)
            intr = np.where(hi >= 0, g['inter_id'][io], -1)
        np.testing.assert_array_equal(s['segment_motif'][:, 0], cls, err_msg=f'{part} class')
        np.testing.assert_array_equal(s['instance_particle'][:, 0], ins, err_msg=f'{part} instance')
        np.testing.assert_array_equal(s['instance_interaction'][:, 0], intr,
                                      err_msg=f'{part} interaction')
        # LE as stuff: no instance exactly on le points, and PID le == motif le
        le = s['segment_motif'][:, 0] == LE_MOTIF
        np.testing.assert_array_equal(t['instance_particle'][:, 0] < 0, ~o | le, err_msg=f'{part} stuff')
        np.testing.assert_array_equal((s['segment_pid'][:, 0] == LE_PID)[o], le[o],
                                      err_msg=f'{part} pid le != motif le')
        np.testing.assert_array_equal(t['instance_interaction'][:, 0], s['instance_interaction'][:, 0])
        # unowned points carry no label
        for k in ('segment_motif', 'segment_pid', 'instance_particle', 'instance_interaction'):
            assert (s[k][~o, 0] == -1).all(), f'{part} unowned {k}'
