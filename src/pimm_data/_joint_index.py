"""Joint cross-modality event index (Phase A / D42).

Each modality reader indexes the ``event_*`` groups present in its own
shards and maps a global ``idx`` through *that* reader's index. A multimodal
dataset that passes one ``idx`` to every reader (with ``_n_events =
min(len(r))``) is only correct if every reader holds the SAME ordered list of
physics events. It does not, whenever the present-event sets diverge:

* an step event filter (``min_deposits`` / ``min_segments``) masks the step
  reader's index but not the others;
* a production gap (a skipped ``event_NNN``) is present in some modalities but
  not others.

Either way a global ``idx`` then resolves to *different physics events* in
different modalities — silently — corrupting every cross-modality join
(``deposit_to_track`` / ``group_to_track`` / ``bridges``).

:func:`build_joint_index` fixes this for any multimodal dataset: it intersects
the present event numbers across every loaded modality (per shard) and
overwrites each reader's ``indices`` / ``cumulative_lengths`` with the shared
joint index, so all readers resolve a global ``idx`` to the same
``(file, event_num)``.

Shards align by **identity**, ``(run, shard tag)`` — the tag is the trailing
``_NNNN`` filename token, the one identifier invariant across modalities
(B1 scoping note). Positional (sorted-glob order) alignment mispairs
*different physics events* silently whenever a middle shard is missing in
one modality (partial transfer, failed per-modality production job, a
hand-deleted corrupt file): every later shard shifts by one, and because
event numbers restart per shard the per-shard intersection stays large.
Tag alignment instead drops the unmatched shard from every modality (warn,
or raise under ``strict_lengths``). Sorted-glob position remains only as a
fallback when tags are not unique within a modality.
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)


def _shard_key(reader, i):
    """Cross-modality shard identity: ``(run, tag)`` where the tag is the
    trailing ``_``-separated filename token (``sim_step_0054.h5`` →
    ``'0054'``) — per-modality filenames differ, the tag doesn't."""
    base = os.path.basename(reader.h5_files[i])
    stem = base[:-3] if base.endswith('.h5') else base
    tag = stem.rsplit('_', 1)[-1]
    run = reader.run_of(i) if hasattr(reader, 'run_of') else ''
    return (run, tag)


def build_joint_index(named_readers, *, strict_lengths=False,
                      source_label='', filter_label=''):
    """Intersect present events across modalities; inject one shared index.

    Parameters
    ----------
    named_readers : list[tuple[str, reader]]
        ``(modality_name, reader)`` for every loaded modality. Each reader
        must expose ``h5_files``, ``indices`` (list of per-shard ``np.int64``
        event-number arrays) and ``cumulative_lengths``.
    strict_lengths : bool
        If True, raise on any cross-modality shard-count or event mismatch
        instead of warning and aligning on the intersection.
    source_label : str
        Identifier for log/error messages (e.g. the dataset ``data_root``).
    filter_label : str
        Human description of any active event filter (e.g.
        ``"min_deposits=5"``) for the mismatch message.

    Returns
    -------
    int
        Total number of jointly-present events (the dataset's ``_n_events``).

    Side effects
    ------------
    Overwrites ``reader.indices`` / ``reader.cumulative_lengths`` and
    re-selects ``reader.h5_files`` (and ``_file_runs``) to the identity-
    aligned common shard sequence on every reader.
    """
    readers = [r for _, r in named_readers]
    if not readers:
        return 0

    # Shards align by identity (run, tag), not glob position — a missing
    # middle shard in one modality must drop that shard everywhere, never
    # shift the pairing of every later shard.
    keys_per = {n: [_shard_key(r, i) for i in range(len(r.h5_files))]
                for n, r in named_readers}
    tags_unique = all(len(set(ks)) == len(ks) for ks in keys_per.values())

    if tags_unique:
        common = set.intersection(*(set(ks) for ks in keys_per.values()))
        first_keys = keys_per[named_readers[0][0]]
        order = [k for k in first_keys if k in common]
        dropped = {n: [k for k in ks if k not in common]
                   for n, ks in keys_per.items()}
        if any(dropped.values()):
            msg = (f"{source_label}: shard mismatch across modalities — "
                   f"aligning on {len(order)} common shard(s) by (run, tag); "
                   f"per-modality unmatched shards: "
                   f"{ {n: d for n, d in dropped.items() if d} }.")
            if strict_lengths:
                raise ValueError(msg)
            log.warning(msg)
        sel = {n: [ks.index(k) for k in order] for n, ks in keys_per.items()}
    else:
        # Fallback: tags not unique within a modality (unexpected writer
        # naming) — legacy positional alignment on the common prefix.
        shard_counts = {n: len(r.h5_files) for n, r in named_readers}
        n_files = min(shard_counts.values())
        msg = (f"{source_label}: shard tags are not unique per modality "
               f"— falling back to sorted-glob positional alignment"
               + (f"; shard-count mismatch {shard_counts}, aligning on the "
                  f"first {n_files} shard(s)."
                  if len(set(shard_counts.values())) > 1 else "."))
        if strict_lengths and len(set(shard_counts.values())) > 1:
            raise ValueError(msg)
        log.warning(msg)
        sel = {n: list(range(n_files)) for n, _ in named_readers}

    # Re-select each reader's shard list to the aligned sequence (identity
    # when nothing is missing — the common case).
    for n, r in named_readers:
        s = sel[n]
        if s != list(range(len(r.h5_files))):
            r.h5_files = [r.h5_files[i] for i in s]
            r.indices = [r.indices[i] for i in s]
            runs = getattr(r, '_file_runs', None)
            if runs is not None:
                r._file_runs = [runs[i] for i in s]
    n_files = len(next(iter(sel.values())))

    raw_totals = {n: int(sum(len(r.indices[s]) for s in range(n_files)))
                  for n, r in named_readers}

    joint = []
    for s in range(n_files):
        common = {int(e) for e in readers[0].indices[s]}
        for r in readers[1:]:
            common &= {int(e) for e in r.indices[s]}
        joint.append(np.array(sorted(common), dtype=np.int64))

    cum = (np.cumsum([len(a) for a in joint]).astype(np.int64)
           if joint else np.zeros(0, dtype=np.int64))
    total = int(cum[-1]) if len(cum) else 0

    # A4: surface any event dropped to keep modalities aligned. Expected under
    # an step event filter; otherwise it flags a real cross-modality gap.
    if any(t != total for t in raw_totals.values()):
        extra = f" (or filtered by {filter_label})" if filter_label else ""
        msg = (f"{source_label}: joint cross-modality index = {total} events; "
               f"per-modality present counts {raw_totals}. Events not present "
               f"in every loaded modality{extra} are excluded to keep all "
               f"modalities aligned.")
        if strict_lengths:
            raise ValueError(msg)
        log.warning(msg)

    for r in readers:
        r.indices = joint
        r.cumulative_lengths = cum
    return total
