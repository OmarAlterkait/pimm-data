"""step + labl: 3D panoptic segmentation of the deposits, pysupera's target.

    python examples/doraemon_final/step_labl.py [wire|pixel]

Per deposit: segment_motif (shower, track, michel, delta, le) and instance_particle
(pysupera's instance; -1 on LE points with le_is_stuff, and on unowned deposits).
Change labels= for another choice ('all', 'fragment', ('segment_pid',), ...).
"""
import sys

from pimm_data.loader import EventLoader
from site_doraemon import dataset
import _show

ro = sys.argv[1] if len(sys.argv) > 1 else 'wire'
transform = [
    dict(type='Apply', on='step', transforms=[
        # all deposits -> 3 mm voxels (energy summed, labels of the winning deposit)
        dict(type='Voxelize', grid_size=3.0, origin=-2160.0),
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=2160.0 * 3 ** 0.5),
        dict(type='LogTransform', min_val=0.01, max_val=20.0),
    ]),
    dict(type='Collect', parts={'step': dict(
        keys=('coord', 'grid_coord', 'segment_motif', 'instance_particle'),
        feat_keys=('coord', 'energy'))}),
]
ds = dataset(ro, modalities=('step',), labels=('pysupera',), split='train', transform=transform)
print(f'{len(ds)} events ({ro}); one event before transforms:')
_show.event(ds.get_data(0))
print('a batch of 2 after Voxelize + Collect:')
_show.batch(next(iter(EventLoader(ds, batch_size=2, shuffle=True, num_workers=2, seed=0))))
