"""Lay the synthetic JAXTPC sample out as doraemon task directories."""

import os
import shutil

from pimm_data.testing import EVENT_ID_BASE as ID_BASE  # noqa: F401

MODS = ('step', 'sensor', 'hits', 'labl', 'supera')
N_EVENTS = 3


def _to_tasks(src, dst, tasks):
    """Move shard ``i`` of every modality to task ``tasks[i]``:
    ``{dst}/j000000/{task}/{mod}/sim_{mod}_{task}.h5``."""
    for i, task in enumerate(tasks):
        for mod in MODS:
            d = os.path.join(dst, 'j000000', task, mod)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(src, mod, f'sim_{mod}_{i:04d}.h5'),
                        os.path.join(d, f'sim_{mod}_{task}.h5'))
    return dst
