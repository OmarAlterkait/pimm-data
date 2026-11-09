"""hits + labl: per-plane 2D panoptic segmentation of the hit entries.

    python examples/doraemon_final/hits_labl.py [wire|pixel]

hits coord is detector-index space: (wire, time) on wire planes, (py, pz, time) on pixel;
plane_id says which plane. Labels reach hits only through labl's per-plane maps
(hits.group), so the same instance id appears on every plane it crosses. Pixel hits that
are not on a sensor pixel (about 13%) carry -1, as pysupera labels none of them.
"""
import sys

from pimm_data.loader import EventLoader
from site_doraemon import dataset
import _show

ro = sys.argv[1] if len(sys.argv) > 1 else 'wire'
transform = [
    dict(type='Apply', on='hits', transforms=[
        dict(type='LogTransform', min_val=0.01, max_val=20.0, keys=('energy',)),
    ]),
    dict(type='Collect', parts={'hits': dict(
        keys=('coord', 'plane_id', 'segment_motif', 'instance_particle'),
        feat_keys=('coord', 'energy'))}),
]
ds = dataset(ro, modalities=('hits',), labels=('pysupera',), split='train', transform=transform)
print(f'{len(ds)} events ({ro}); one event before transforms:')
_show.event(ds.get_data(0))
print('a batch of 2 after Collect:')
_show.batch(next(iter(EventLoader(ds, batch_size=2, shuffle=True, num_workers=2, seed=0))))
