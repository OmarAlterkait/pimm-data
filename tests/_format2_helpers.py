"""Rewrite the format-1 synthetic JAXTPC sample as doraemon format 2."""

import os
import shutil

import h5py
import numpy as np

MODS = ('step', 'sensor', 'hits', 'labl')
N_EVENTS = 3
ID_BASE = 1000      # event ids deliberately unlike positions


def _to_format2(src, dst, tasks):
    """Rewrite format-1 shard ``i`` as doraemon task ``tasks[i]``."""
    for i, task in enumerate(tasks):
        for mod in MODS:
            d = os.path.join(dst, 'j000000', task, mod)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, f'sim_{mod}_{task}.h5')
            shutil.copy(os.path.join(src, mod, f'sim_{mod}_{i:04d}.h5'), out)
            with h5py.File(out, 'a') as f:
                ids = []
                for k in sorted(k for k in f if k.startswith('event_')):
                    n = int(k.split('_')[1])
                    new = f'entry_{n:06d}'
                    f.move(k, new)
                    eid = ID_BASE + i * N_EVENTS + n
                    f[new].attrs['event_id'] = eid
                    ids.append(eid)
                ev = f.create_group('dprod/events')
                ev.attrs['order'] = 'sorted'
                ev.create_dataset('event_id', data=np.array(ids, np.int64))
                if 'config' in f:
                    f['config'].attrs['format_version'] = 2
    return dst
