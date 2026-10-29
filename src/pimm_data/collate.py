"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from . import _roles


def collate_with_roles(samples):
    """Role-driven reduce for the flat-prefixed batch contract (REDESIGN.md §5).

    Each sample is a flat dict of underscore-prefixed keys (``step_coord``,
    ``step_offset``) optionally carrying a ``_roles`` map (key -> role spec, §3).
    Dispatch per key by role: point/raw -> concat by offset; offset -> cumsum;
    edge -> concat+shift; label -> compact+distinct-renumber; instance -> concat
    (its ``*_inst_offset`` cumsums); event / unprefixed -> stack/list. ``_roles``
    is carried through (post-collate consumers need it). ``offset`` stays ``(B,)``
    no-leading-0.
    """
    roles = samples[0].get('_roles', {})
    keys = [k for k in samples[0] if k != '_roles']
    parts = _roles.parts_from_keys(keys, roles)
    col = lambda key: [s[key] for s in samples]
    # the bare form's part is '' and its offset is plain 'offset'
    okey = lambda part: f'{part}_offset' if part else 'offset'
    offs = lambda part: [s[okey(part)] for s in samples]
    owner = lambda key: _roles.part_of(key, parts) or ('' if '' in parts else None)
    out = {}
    for key in keys:
        spec = roles.get(key)
        if key == 'offset' or key.endswith('_offset'):
            out[key] = _roles.cat_offset(col(key)); continue
        kind = _roles.role_kind(spec) if spec is not None else None
        if kind == _roles.EVENT:
            out[key] = _roles.stack_event(col(key))
        elif kind == 'edge':
            tgt = spec[1]
            if tgt == 'self':
                out[key] = _roles.cat_edge_self(col(key), offs(owner(key)))
            else:
                src, dst = tgt
                out[key] = _roles._shift_concat(col(key), offs(src), offs(dst))
        elif kind == 'label':
            out[key] = _roles.cat_label_col(col(key))
        elif kind == 'ref':
            if okey(spec[1]) not in samples[0]:
                raise KeyError(
                    f"collate: {key!r} references part {spec[1]!r}, which was "
                    f"not collected; collect both with Collect(parts=...).")
            out[key] = _roles.cat_ref(col(key), offs(spec[1]))
        elif kind in (_roles.POINT, _roles.RAW, 'instance'):
            out[key] = _roles.cat_point(col(key))
        elif spec is None:
            # default: belongs to a part -> point (concat); else whole-event -> stack/list
            if _roles.part_of(key, parts) is not None:
                out[key] = _roles.cat_point(col(key))
            else:
                out[key] = _roles.stack_event(col(key))
        else:
            raise ValueError(f"collate: unhandled role {spec!r} for {key!r}")
    # always carry _roles when the roles path ran (even if empty) so post-collate
    # consumers (split_event, the tail-runner) can rely on its presence.
    out['_roles'] = roles
    return out


def mix_pairs(batch):
    """Mix3D on a role-collated batch: events ``2i`` and ``2i+1`` become one.

    Merging events is offset bookkeeping only: every ``*_offset`` (each part's
    and each second row-space's) keeps the end of every pair. It needs nothing
    from key names, so it holds for any keys a pipeline collects — what makes
    it correct is the roles. Ids that must stay apart across events are
    ``('label', …)`` (compacted, batch-unique) or ``('ref', part)`` (shifted to
    the batch's rows) after collate, so a merged pair never shares one. A
    ``point``-role column is per-event by contract: an id carried that way
    would collide, so declare its role in ``Collect``. An event-level tensor
    (one row per event) has no meaning for a merged pair and is refused.
    """
    roles = batch.get('_roles', {})
    events = [k for k, r in roles.items() if k in batch
              and r is not None and _roles.role_kind(r) == _roles.EVENT
              and isinstance(batch[k], torch.Tensor)]
    if events:
        raise ValueError(f"mix: event-level keys {events} cannot follow two "
                         f"events merged into one; drop them or mix_prob=0.")
    out = dict(batch)
    for k, v in batch.items():
        if (k == 'offset' or k.endswith('_offset')) \
                and isinstance(v, torch.Tensor) and len(v) > 1:
            out[k] = torch.cat([v[1:-1:2], v[-1:]])
    return out


def collate_fn(batch, mix_prob=0):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'

    ``mix_prob``: with this probability, a role-collated batch (from
    ``Collect``) is mixed — see :func:`mix_pairs`.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    # Two LIVE paths, not one live and one legacy. A sample carrying `_roles`
    # comes from `Collect` (which always stamps it) and is collated by role;
    # everything else — raw-sensor SSL recipes and any pipeline that does not end
    # in `Collect` — falls through to the generic collate below. That branch was
    # commented as "legacy"; making it raise showed 13 tests reach it, so the
    # label was wrong in the direction that invites deletion.
    if isinstance(batch[0], Mapping) and '_roles' in batch[0]:
        out = collate_with_roles(batch)
        if mix_prob and random.random() < mix_prob:
            out = mix_pairs(out)
        return out
    if mix_prob:
        raise ValueError("collate_fn: mix_prob needs a Collect-ed batch (with "
                         "_roles); point_collate_fn mixes the bare layout.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            )
            for key in batch[0]
            if not key.startswith("_")  # skip non-tensor metadata
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))


def inseg_collate_fn(batch, mix_prob=0):
    """
    Collate function for instance segmentation that handles lists of data dictionaries.
    Each item in the batch is a list of data dictionaries (one per query).
    """
    assert isinstance(
        batch[0], list
    )  # currently, only support input_dict, rather than input_list
    assert isinstance(
        batch[0][0], Mapping
    )  # currently, only support input_dict, rather than input_list
    # Flatten the batch - each item is already a list of dictionaries

    flattened_batch = []
    for item_list in batch:
        flattened_batch.extend(item_list)
    
    # Use the original collate function on the flattened batch
    return collate_fn(flattened_batch, mix_prob=mix_prob)
