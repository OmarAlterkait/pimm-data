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
(``group_to_track`` / ``deposit_to_group`` / the ``labl`` tables).

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

Missing or partial data is an error by default: a shard missing from a
modality, a shard that does not open, an event present in some modalities
and not others, or shard tags that do not identify shards. ``allow_partial``
(reading a corpus mid-build) turns these into a warning and aligns on what
every modality has. An event filter (``min_deposits`` / ``min_segments``)
is not a gap: presence is compared before filtering, filters intersect
after. Two modalities naming different ``event_id`` for the same event is
corruption and always raises.
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


def _complain(msg, allow_partial):
    if not allow_partial:
        raise ValueError(msg + " Pass allow_partial=True to read what every "
                         "modality has.")
    log.warning(msg)


def build_joint_index(named_readers, *, allow_partial=False,
                      source_label='', filter_label=''):
    """Intersect present events across modalities; inject one shared index.

    Parameters
    ----------
    named_readers : list[tuple[str, reader]]
        ``(modality_name, reader)`` for every loaded modality. Each reader
        must expose ``h5_files``, ``indices`` (list of per-shard ``np.int64``
        event-number arrays, after any filter) and ``cumulative_lengths``;
        ``present`` (unfiltered) and ``unreadable`` when it has them.
    allow_partial : bool
        Warn instead of raising on missing or partial data (see the module
        docstring) and align on the intersection.
    source_label : str
        Identifier for log/error messages (e.g. the dataset ``data_root``).
    filter_label : str
        Human description of any active event filter (e.g.
        ``"min_deposits=5"``) for log messages.

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

    bad = {n: [r.h5_files[i] for i in getattr(r, 'unreadable', ())]
           for n, r in named_readers}
    if any(bad.values()):
        _complain(f"{source_label}: shards that do not open: "
                  f"{ {n: b for n, b in bad.items() if b} }.", allow_partial)

    # Shards align by identity (run, tag), not glob position — a missing
    # middle shard in one modality must drop that shard everywhere, never
    # shift the pairing of every later shard.
    keys_per = {n: [_shard_key(r, i) for i in range(len(r.h5_files))]
                for n, r in named_readers}
    dup = {n: sorted({k for k in ks if ks.count(k) > 1})
           for n, ks in keys_per.items()}
    if any(dup.values()):
        raise ValueError(
            f"{source_label}: shard tags do not identify shards (the trailing "
            f"filename token repeats within a modality): "
            f"{ {n: d for n, d in dup.items() if d} }; shards cannot be "
            f"paired across modalities.")
    common = set.intersection(*(set(ks) for ks in keys_per.values()))
    order = [k for k in keys_per[named_readers[0][0]] if k in common]
    dropped = {n: [k for k in ks if k not in common]
               for n, ks in keys_per.items()}
    if any(dropped.values()):
        _complain(f"{source_label}: shard mismatch across modalities — "
                  f"{len(order)} common shard(s) by (run, tag); per-modality "
                  f"unmatched shards: "
                  f"{ {n: d for n, d in dropped.items() if d} }.",
                  allow_partial)
    sel = {n: [ks.index(k) for k in order] for n, ks in keys_per.items()}

    # Re-select each reader's shard list to the aligned sequence (identity
    # when nothing is missing — the common case).
    for n, r in named_readers:
        s = sel[n]
        if s != list(range(len(r.h5_files))):
            r.h5_files = [r.h5_files[i] for i in s]
            r.indices = [r.indices[i] for i in s]
            if getattr(r, 'present', None) is not None:
                r.present = [r.present[i] for i in s]
            runs = getattr(r, '_file_runs', None)
            if runs is not None:
                r._file_runs = [runs[i] for i in s]
    n_files = len(order)

    # Presence, before any filter, must agree shard by shard.
    def present(r, s):
        p = getattr(r, 'present', None)
        return {int(e) for e in (p[s] if p is not None else r.indices[s])}
    gaps = []
    for s in range(n_files):
        sets = {n: present(r, s) for n, r in named_readers}
        union = set.union(*sets.values())
        missing = {n: sorted(union - v)[:5] for n, v in sets.items()
                   if union - v}
        if missing:
            gaps.append((os.path.basename(readers[0].h5_files[s]), missing))
    if gaps:
        _complain(f"{source_label}: events present in some modalities and "
                  f"not others ({len(gaps)} shard(s); first: {gaps[:3]}).",
                  allow_partial)

    joint = []
    for s in range(n_files):
        common = {int(e) for e in readers[0].indices[s]}
        for r in readers[1:]:
            common &= {int(e) for e in r.indices[s]}
        joint.append(np.array(sorted(common), dtype=np.int64))
    _check_event_ids(named_readers, joint, source_label)

    cum = (np.cumsum([len(a) for a in joint]).astype(np.int64)
           if joint else np.zeros(0, dtype=np.int64))
    total = int(cum[-1]) if len(cum) else 0
    if filter_label:
        log.info("%s: %d events jointly present after %s.", source_label,
                 total, filter_label)

    for r in readers:
        r.indices = joint
        r.cumulative_lengths = cum
    return total


def _check_event_ids(named_readers, joint, source_label):
    """Every modality that records an ``event_id`` (format 2: ``/dprod`` or
    entry attrs) must name the same one for each joint event."""
    from ._shard_meta import read_shard_meta
    for s, events in enumerate(joint):
        if not len(events):
            continue
        ids = {}
        for n, r in named_readers:
            try:
                v = read_shard_meta(r.h5_files[s]).get('event_ids')
            except OSError:
                continue
            if v is not None and len(v) > events[-1]:
                ids[n] = v[events]
        names = list(ids)
        for n in names[1:]:
            a, b = ids[names[0]], ids[n]
            diff = np.flatnonzero((a != b) & (a >= 0) & (b >= 0))
            if len(diff):
                e = int(events[diff[0]])
                raise ValueError(
                    f"{source_label}: shard {s} event {e}: {names[0]} has "
                    f"event_id {int(ids[names[0]][diff[0]])}, {n} has "
                    f"{int(ids[n][diff[0]])} ({len(diff)} such event(s)). "
                    f"The modalities do not hold the same events.")
