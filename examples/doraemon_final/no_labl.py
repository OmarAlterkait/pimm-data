"""No labl: self-supervised loading. Nothing ties planes, or hits and step, together.

    python examples/doraemon_final/no_labl.py [wire|pixel]

step keeps its deposit groups (step.group), hits only plane_group (which entries form one
block on a plane; ids are numbered per plane and reference nothing), sensor is the readout.
"""
import sys

from site_doraemon import dataset
import _show

ro = sys.argv[1] if len(sys.argv) > 1 else 'wire'
ds = dataset(ro, modalities=('step', 'hits', 'sensor'), labels=None, split='train')
print(f'{len(ds)} events ({ro}); one event:')
_show.event(ds.get_data(0))
