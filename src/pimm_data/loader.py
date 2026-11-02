"""Batches out of the box: a DataLoader plus the pipeline's tail.

A dataset's ``transform`` list is cut after its terminal ``Collect``
(:func:`pimm_data.transform.split_pipeline`): the head runs per event in the
workers, the tail (``dataset.batch_transform``: ``Mix3D``, ``ToDevice``,
``Densify``, …) on each collated batch. :class:`EventLoader` owns both halves,
so every consumer — training, validation, test, a probe, a notebook — gets the
same batches from the same config; a loop over a bare ``DataLoader`` would
silently skip the tail.
"""

import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .collate import collate_fn as _collate_fn


def seed_worker(worker_id, num_workers, rank, seed):
    """Seed a DataLoader worker: ``seed + num_workers * rank + worker_id``,
    so workers and ranks draw different augmentations, reproducibly."""
    s = seed + num_workers * rank + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class EventLoader:
    """A ``DataLoader`` over ``dataset`` that applies
    ``dataset.batch_transform`` to every collated batch, in the main process,
    stamping ``_epoch`` / ``_rank`` (the per-event noise seeds read them).

    ``distributed``: shard with a ``DistributedSampler`` (rank and world size
    from the process group; ``rank`` defaults to it). ``seed``: seed the
    workers (:func:`seed_worker`); None leaves them unseeded. Call
    :meth:`set_epoch` before each epoch. The other arguments are the
    ``DataLoader``'s.
    """

    def __init__(self, dataset, batch_size=1, *, shuffle=False, num_workers=0,
                 collate_fn=_collate_fn, distributed=False, rank=None,
                 seed=None, drop_last=False, pin_memory=True,
                 persistent_workers=None):
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            rank = sampler.rank if rank is None else rank
        else:
            sampler = None
        self.rank = 0 if rank is None else int(rank)
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        self.loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle and sampler is None, sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            persistent_workers=persistent_workers,
            worker_init_fn=None if seed is None else partial(
                seed_worker, num_workers=num_workers, rank=self.rank,
                seed=seed))
        self.batch_transform = getattr(dataset, 'batch_transform', None)
        self.epoch = 0

    dataset = property(lambda self: self.loader.dataset)
    sampler = property(lambda self: self.loader.sampler)
    batch_size = property(lambda self: self.loader.batch_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(self.epoch)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        tail = self.batch_transform
        for batch in self.loader:
            if tail is not None:
                if not isinstance(batch, dict):
                    raise TypeError(
                        f"EventLoader: the tail {tail.transforms} needs a "
                        f"dict batch, got {type(batch).__name__}.")
                batch['_epoch'], batch['_rank'] = self.epoch, self.rank
                batch = tail(batch)
            yield batch
