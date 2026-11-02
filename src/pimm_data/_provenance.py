"""What produced the data a dataset reads (``dataset.provenance``).

Every shard's ``config`` attributes say which program wrote it: JAXTPC's
``step``/``sensor``/``hits`` carry ``git_repo``/``git_commit``/``git_dirty``
and ``production_version``; ``labl`` carries ``labl_format``,
``supera_format_version`` and pysupera's commit and input hashes under
``provenance/...``. This collects them per modality over every shard (from
the cached shard metadata, no extra opens) and refuses a dataset whose
modalities or shards come from different productions: their events would not
be the same events.
"""

import logging

from ._shard_meta import read_shard_meta

log = logging.getLogger(__name__)

#: attrs reported as they are; ``provenance/.../<suffix>`` keys by suffix
_KEYS = ('production_version', 'format_version', 'git_repo', 'git_commit',
         'git_dirty', 'labl_format', 'supera_format_version')
_NESTED = {'software/pysupera/git_commit': 'pysupera_commit',
           'software/pysupera/git_dirty': 'pysupera_dirty',
           'files/pysupera_config.sha256': 'pysupera_config_sha256',
           'campaign_config_sha256': 'campaign_config_sha256'}


def _summary(attrs):
    out = {k: attrs[k] for k in _KEYS if k in attrs}
    for k, v in attrs.items():
        if k.startswith('provenance/'):
            for suffix, name in _NESTED.items():
                if k.endswith('/' + suffix):
                    out[name] = v
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in out.items()}


def collect_provenance(named_readers, source_label=''):
    """``{modality: {field: value}}`` over every shard; a field whose shards
    disagree is a sorted list of its values. Raises if the data comes from
    more than one ``production_version``; warns on a dirty source tree."""
    prov = {}
    for name, reader in named_readers:
        fields = {}
        for path in reader.h5_files:
            try:
                attrs = read_shard_meta(path)['config_attrs']
            except OSError:
                continue
            for k, v in _summary(attrs).items():
                fields.setdefault(k, set()).add(v)
        prov[name] = {k: (next(iter(v)) if len(v) == 1 else sorted(v, key=str))
                      for k, v in fields.items()}
    productions = {p for m in prov.values()
                   for p in _as_list(m.get('production_version'))}
    if len(productions) > 1:
        raise ValueError(
            f"{source_label}: the data comes from more than one production "
            f"{ {n: m.get('production_version') for n, m in prov.items()} }; "
            f"its modalities do not hold the same events.")
    dirty = [n for n, m in prov.items()
             if True in _as_list(m.get('git_dirty'))
             or True in _as_list(m.get('pysupera_dirty'))]
    if dirty:
        log.warning("%s: %s written from a dirty source tree; the commit does "
                    "not reproduce it.", source_label, dirty)
    return prov


def _as_list(v):
    return [] if v is None else (v if isinstance(v, list) else [v])
