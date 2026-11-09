"""The doraemon final-layout sample at S3DF: one place for the hard-coded location.

    from site_doraemon import dataset, TASKS
    ds = dataset('wire', modalities=('step',), labels=('pysupera',), split='train')

ROOT holds jaxtpc_{wire,pixel}/j000000/jaxtpc_{ro}_<task>/{step,hits,sensor,labl} for the
first 10 tasks (10,000 events) of doraemon_2026_1M_lartpc. Split by task: the last task is
val, the rest train. Override the location with DORAEMON_FINAL_ROOT.
"""
import os

from pimm_data import JAXTPCDataset

ROOT = os.environ.get('DORAEMON_FINAL_ROOT',
                      '/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/final_10k')
BUCKET = 'j000000'
READOUTS = ('wire', 'pixel')


def tasks(readout='wire'):
    """Complete tasks only: every modality file present (a task being written has .tmp)."""
    d = os.path.join(ROOT, f'jaxtpc_{readout}', BUCKET)

    def complete(t):
        task = t[len(f'jaxtpc_{readout}_'):]
        return all(os.path.exists(os.path.join(d, t, m, f'sim_{readout}_{m}_jaxtpc_{readout}_{task}.h5'))
                   for m in ('step', 'hits', 'sensor', 'labl'))
    return sorted(t for t in os.listdir(d) if t.startswith(f'jaxtpc_{readout}_') and complete(t))


TASKS = {ro: tasks(ro) for ro in READOUTS if os.path.isdir(os.path.join(ROOT, f'jaxtpc_{ro}'))}


def runs(readout='wire', split='all'):
    t = TASKS[readout]
    t = {'train': t[:-1], 'val': t[-1:], 'all': t}[split]
    return [f'{BUCKET}/{x}' for x in t]


def dataset(readout='wire', modalities=('step',), labels='all', split='all', **kw):
    """JAXTPCDataset over this sample. labels: 'all' (the five per-point labels), a level
    ('pysupera', 'fragment', 'particle', 'interaction'), keys, or None (no labl)."""
    return JAXTPCDataset(data_root=os.path.join(ROOT, f'jaxtpc_{readout}'),
                         runs=runs(readout, split), dataset_name=f'sim_{readout}',
                         modalities=tuple(modalities), labels=labels, **kw)
