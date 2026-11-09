"""step + hits + labl: the full truth, with deposit <-> hit <-> plane correspondence.

    python examples/doraemon_final/step_hits_labl.py [wire|pixel]

Both point parts reference the same `group` rows (step.group from step's deposit_to_group,
hits.group through labl's plane maps), group.fragment references `particle`, particle
rows reference `interaction`. Collate shifts every reference by its target's running row
count, so a batched walk (e.g. group_fragment[step_group]) works as in one event.
step.qs is each deposit's share of its group's charge (splits a hit's charge over the
deposits of its group). Object targets come from the tables, not from the points.
"""
import sys

from pimm_data.loader import EventLoader
from site_doraemon import dataset
import _show

ro = sys.argv[1] if len(sys.argv) > 1 else 'wire'
transform = [dict(type='Collect', parts={
    'step': dict(keys=('coord', 'energy', 'group', 'qs', 'segment_motif', 'instance_particle')),
    'hits': dict(keys=('coord', 'energy', 'plane_id', 'group', 'segment_motif', 'instance_particle')),
    'group': dict(keys=('volume', 'fragment'), offset_keys_dict=dict(offset='volume')),
    'particle': dict(keys=('pdg', 'is_primary', 'momentum_start', 'interaction', 'fragment_instance'),
                     offset_keys_dict=dict(offset='pdg')),
    'interaction': dict(keys=('vertex',), offset_keys_dict=dict(offset='vertex')),
})]
ds = dataset(ro, modalities=('step', 'hits'), labels='all', split='train', transform=transform)
print(f'{len(ds)} events ({ro}); one event before Collect:')
_show.event(ds.get_data(0))
b = next(iter(EventLoader(ds, batch_size=2, shuffle=True, num_workers=2, seed=0)))
print('a batch of 2:')
_show.batch(b)
# the correspondence: a hit's group has the deposits that made it, in the same batch,
# except for the few groups straddling the readout-window edge whose deposits all
# arrive outside it (kept because they have hits; about 1e-5 of hits)
has = b['step_group'].new_zeros(len(b['group_volume']), dtype=bool)
has[b['step_group'][b['step_group'] >= 0]] = True
n = int((~has[b['hits_group']]).sum())
print(f'hits whose group has no in-window deposit: {n} of {len(b["hits_group"])}')
