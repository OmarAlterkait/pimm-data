"""The doraemon final-layout sample at S3DF: one place for the hard-coded location.

    from site_doraemon import dataset, TASKS
    ds = dataset('wire', modalities=('step',), labels=('pysupera',), split='train')

ROOT holds jaxtpc_{wire,pixel}/j000000/jaxtpc_{ro}_<task>/{step,hits,sensor,labl} for 100
tasks (100,000 events) of doraemon_2026_1M_lartpc: the first 100 of bucket j000000 with no
empty volume. Split by task: the last N_VAL tasks are val, the rest train. Override the
location with DORAEMON_FINAL_ROOT. (The directory keeps its first name, final_10k.)
"""
import os

from pimm_data import JAXTPCDataset

ROOT = os.environ.get('DORAEMON_FINAL_ROOT',
                      '/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/final_10k')
BUCKET = 'j000000'
READOUTS = ('wire', 'pixel')
N_VAL = 5                                  # tasks (1,000 events each) held out for val


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
    t = {'train': t[:-N_VAL], 'val': t[-N_VAL:], 'all': t}[split]
    return [f'{BUCKET}/{x}' for x in t]


def dataset(readout='wire', modalities=('step',), labels='all', split='all', **kw):
    """JAXTPCDataset over this sample. labels: 'all' (the five per-point labels), a level
    ('pysupera', 'fragment', 'particle', 'interaction'), keys, or None (no labl)."""
    return JAXTPCDataset(data_root=os.path.join(ROOT, f'jaxtpc_{readout}'),
                         runs=runs(readout, split), dataset_name=f'sim_{readout}',
                         modalities=tuple(modalities), labels=labels, **kw)
