"""
Transforms on the collated batch: the tail of a pipeline.

A pipeline is one ``transform`` list. Everything after its terminal ``Collect``
is the tail, ``dataset.batch_transform``: it runs on each collated batch in the
main process, where :class:`pimm_data.loader.EventLoader` applies it (Lightning:
:class:`BatchTransformMixin`). Entries there are ``scope='batch'`` (``Mix3D``,
``ToDevice``) or ``scope='any'`` (``Densify`` and helix's noise, which dispatch
on a per-event sub-dict or a collated batch); ``Compose`` rejects anything else.
``ToDevice`` is the device step: what follows it runs on the device, so dense
grids are born there and only sparse data crosses PCIe.

Seeds are content-addressed per event (``blake2b(name|base_seed|epoch|rank)``), so
the same event gets the same noise across batch position / worker / resume;
``EventLoader`` stamps ``batch['_epoch']`` / ``_rank`` for it. The **incoherent**
component uses a torch ``Generator`` whose CPU/CUDA streams differ, so incoherent
realizations are device-specific (coherent uses numpy, device-independent).
Geometry is a registry baked into the transform, not carried per-sample.
"""

from __future__ import annotations

import hashlib
import random
import warnings

import torch

from . import dense_ops
from .transform import TRANSFORMS


# ── device move (idempotent) ──────────────────────────────────────────────
def move_to_device(batch, device, non_blocking=True):
    """Recursively move tensors to ``device``; non-tensors untouched.

    Idempotent — a tensor already on ``device`` is returned by ``.to`` without a
    copy. Use ``DataLoader(pin_memory=True)`` for ``non_blocking`` to be truly async.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    return batch


# ── content-addressed seeding ─────────────────────────────────────────────
def content_seed(event_name, base_seed=0, epoch=0, rank=0):
    """Stable per-event seed: same event -> same noise across batch/worker/resume."""
    payload = f"{event_name}|{int(base_seed)}|{int(epoch)}|{int(rank)}".encode()
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, 'little') & ((1 << 63) - 1)


def _seeds_for(batch, base_seed):
    """Per-event content-addressed seeds from ``batch['name']`` (self-contained — no
    runner injects them). Epoch/rank are read from optional ``batch['_epoch']`` /
    ``batch['_rank']`` the trainer may stamp (default 0 = stable within a run)."""
    names = batch.get('name')
    if isinstance(names, str):
        names = [names]
    epoch = int(batch.get('_epoch', 0)) if not torch.is_tensor(batch.get('_epoch')) else int(batch['_epoch'])
    rank = int(batch.get('_rank', 0)) if not torch.is_tensor(batch.get('_rank')) else int(batch['_rank'])
    if not names:
        warnings.warn(
            "AddNoise: batch has no 'name' -> seeding falls back to batch position, "
            "NOT reproducible across shuffles/epochs/runs. Ensure Collect passes "
            "'name' through.", RuntimeWarning, stacklevel=2)
        # length B from the dense grid is unknown here; caller passes count via name
        return None
    return [content_seed(nm, base_seed, epoch, rank) for nm in names]


# ── batch transforms (after Collect; registered, scope='batch') ─────────────
@TRANSFORMS.register_module()
class Mix3D:
    """Mix3D: with probability ``prob``, events ``2i`` and ``2i+1`` of the
    batch become one (:func:`pimm_data.collate.mix_pairs` — offsets only, so
    it holds for any collected keys; ids stay apart by their roles). Nothing
    is densified or summed: a mixed sparse readout carries both events' rows.
    A training augmentation — put it in the train pipeline only, after
    ``Collect`` and before ``ToDevice``."""

    scope = 'batch'

    def __init__(self, prob=0.8):
        if not 0 <= prob <= 1:
            raise ValueError(f"Mix3D: prob={prob!r} is not in [0, 1]")
        self.prob = prob

    def __call__(self, batch):
        from .collate import mix_pairs
        if '_roles' not in batch:
            raise ValueError("Mix3D needs a Collect-ed batch (with _roles).")
        if self.prob and random.random() < self.prob:
            batch = mix_pairs(batch)
        return batch


@TRANSFORMS.register_module()
class ToDevice:
    """Move the whole batch to a device — the device move expressed as a
    transform: put ``dict(type='ToDevice', device='cuda')`` after ``Collect``
    and everything after it runs on the device. Idempotent (a tensor already on
    ``device`` is returned without a copy). Never per event: CUDA in a forked
    DataLoader worker is an error."""

    scope = 'batch'

    def __init__(self, device, *, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, batch):
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)


class BatchTransformMixin:
    """Mixin for a LightningModule, which builds its own DataLoaders (instead of
    :class:`pimm_data.loader.EventLoader`): run the post-collate transforms in
    ``on_after_batch_transfer``.

    Set ``self.batch_stages`` to a ``Compose`` (e.g. the dataset's
    ``batch_transform``). Lightning calls this with the batch on-device;
    we stamp ``_epoch``/``_rank`` so the self-seeding noise is per-epoch/rank, then
    run the transforms. (For a custom loop, just call your ``Compose`` on the batch.)
    """

    batch_stages = None

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if not self.batch_stages:
            return batch
        trainer = getattr(self, 'trainer', None)
        batch['_epoch'] = getattr(trainer, 'current_epoch', 0) or 0
        batch['_rank'] = getattr(trainer, 'global_rank', 0) or 0
        return self.batch_stages(batch)
