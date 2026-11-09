"""Voxelize: every point counts (energy summed per cell), each cell takes the labels of
its winning point (class priority track > shower > delta > michel > le > unlabelled,
then earliest time), every per-point key follows the winner, coord is the cell centre."""
import numpy as np

from pimm_data.transform import Voxelize


def test_voxelize_merges_cells_by_priority_then_time():
    # two cells of 3 mm: points 0-2 in cell (0,0,0), 3-4 in cell (1,0,0), 5 in (0,1,0)
    coord = np.array([[0.5, 0.5, 0.5], [1.0, 2.0, 0.1], [2.9, 0.0, 2.9],
                      [3.1, 0.2, 0.2], [5.9, 1.0, 1.0], [0.0, 3.5, 0.0]], np.float32)
    motif = np.array([[0], [1], [1], [4], [2], [-1]])            # shower, track, track | le, michel | none
    t0 = np.array([[1.0], [5.0], [2.0], [0.5], [9.0], [0.0]], np.float32)
    d = dict(coord=coord, energy=np.arange(1, 7, dtype=np.float32)[:, None], t0_us=t0,
             segment_motif=motif, instance_particle=np.array([[7], [8], [9], [3], [4], [-1]]),
             group=np.array([10, 11, 12, 13, 14, 15], np.int32), _roles={'group': ('ref', 'group')})
    v = Voxelize(grid_size=3.0, origin=0.0)(d)
    order = np.lexsort(v['grid_coord'].T[::-1])
    gc = v['grid_coord'][order]
    np.testing.assert_array_equal(gc, [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(v['energy'][order, 0], [1 + 2 + 3, 6, 4 + 5])
    # cell (0,0,0): tracks beat the shower, the earlier track (point 2) wins
    # cell (1,0,0): michel beats le; cell (0,1,0): the unlabelled point alone
    np.testing.assert_array_equal(v['segment_motif'][order, 0], [1, -1, 2])
    np.testing.assert_array_equal(v['instance_particle'][order, 0], [9, -1, 4])
    np.testing.assert_array_equal(v['group'][order], [12, 15, 14])
    np.testing.assert_allclose(v['coord'][order], (gc + 0.5) * 3.0)
    for k in ('coord', 'energy', 't0_us', 'segment_motif', 'instance_particle', 'group', 'grid_coord'):
        assert len(v[k]) == 3, k


def test_voxelize_without_labels_keeps_every_point_energy():
    rng = np.random.default_rng(0)
    coord = rng.uniform(0, 30, (500, 3)).astype(np.float32)
    e = rng.uniform(0, 1, (500, 1)).astype(np.float32)
    v = Voxelize(grid_size=3.0, origin=0.0)(dict(coord=coord, energy=e.copy()))
    np.testing.assert_allclose(v['energy'].sum(), e.sum(), rtol=1e-5)
    assert len(np.unique(v['grid_coord'], axis=0)) == len(v['coord'])
