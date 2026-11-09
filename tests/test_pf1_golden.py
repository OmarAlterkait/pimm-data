"""PF1 golden snapshots — lock single-modality byte-identity + label decoration.

These fingerprints are captured from the CURRENT code on the synthetic (seed=0)
fixture. They make the RESTRUCTURE plan's two breaking phases verifiable:

* Phase 0 (``stream``→``modality`` rename): pure renaming — the produced tensors
  must be **byte-identical**.
* Phase 1 (``labl`` out of ``modalities=`` → ``labels=``, direct FK joins): the API
  changes but the decorated ``segment``/``instance`` values must be **unchanged**.

After those phases, migrate the dataset-build calls below to the new API
(``modalities=``/``labels=``, ``Collect(part=)``) — the frozen fingerprints
MUST still match. A diff here means the refactor changed data, not just names.

The hashes are specific to the synthetic seed=0 fixture, so the full comparison
is gated on ``jaxtpc_is_synthetic``; on a real-data override only the structural
shape/dtype is checked.
"""

import hashlib

import numpy as np
import pytest
import torch

from pimm_data import JAXTPCDataset, collate_fn

# --- frozen fingerprints (captured from current behavior, synthetic seed=0) ---
# Labels re-captured when they became pysupera's named labels (low-energy
# scatters are stuff: no particle instance), and again when they followed
# pysupera's convention exactly (its group owners and hit_labels; class from
# the instance, le for an LE fragment).
# value = (shape, dtype, sha1(bytes)[:16]) for arrays; repr(...) for metadata.
C1_BATCH = {
    'coord': ((240, 3), 'float32', 'ec942392e142f1f7'),
    'feat':  ((240, 4), 'float32', '8194f574fbe936da'),
    'name':  "['sim_step_0000.h5_evt000', 'sim_step_0000.h5_evt001']",
    'offset': ((2,), 'int64', '2bf2fc17c20c1d7d'),
    'split': "['', '']",
}
C2_BATCH = {
    'coord': ((240, 3), 'float32', 'ec942392e142f1f7'),
    'feat':  ((240, 4), 'float32', '8194f574fbe936da'),
    'name':  "['sim_step_0000.h5_evt000', 'sim_step_0000.h5_evt001']",
    'offset': ((2,), 'int64', '2bf2fc17c20c1d7d'),
    'segment_pdg': ((240, 1), 'int32', '7422220c1bc87329'),
    'split': "['', '']",
}
STEP_SEG = ((120, 1), 'int32', '3479f3d912448eac')
STEP_INST = ((120, 1), 'int32', '8bbd4f196b6a4c62')
HITS_SEG = ((240, 1), 'int32', '0b7455135d79ae4a')
HITS_INST = ((240, 1), 'int32', '3b032b9590abe675')


def _fp(x):
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        a = x
    else:
        return repr(x)
    a = np.ascontiguousarray(a)
    return (tuple(a.shape), str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest()[:16])


def _fps(d):
    return {k: _fp(v) for k, v in sorted(d.items()) if not isinstance(v, dict)}


def _ds(root, modalities, transform=None, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, min_deposits=0,
                         max_len=2, transform=transform, **kw)


def _structural(fp_tuple):
    """(shape, dtype) only — for the real-data path where hashes won't match."""
    return fp_tuple[:2] if isinstance(fp_tuple, tuple) else fp_tuple


def _assert(actual, expected, synthetic):
    if synthetic:
        assert actual == expected
    else:
        # real-data override: hashes/shapes differ, only lock dtype layout
        a = {k: (v[1] if isinstance(v, tuple) else v) for k, v in actual.items()}
        e = {k: (v[1] if isinstance(v, tuple) else v) for k, v in expected.items()}
        assert a == e


def test_pf1_c1_step_ssl_batch(jaxtpc_data_root, jaxtpc_is_synthetic):
    ds = _ds(jaxtpc_data_root, ('step',),
             transform=[dict(type='Collect', part='step',
                             keys=('coord',), feat_keys=('coord', 'energy'))])
    batch = collate_fn([ds[0], ds[1]])
    _assert(_fps(batch), C1_BATCH, jaxtpc_is_synthetic)


def test_pf1_c2_step_seg_batch(jaxtpc_data_root, jaxtpc_is_synthetic):
    ds = _ds(jaxtpc_data_root, ('step',), labels='segment_pdg',
             transform=[dict(type='Collect', part='step',
                             keys=('coord', 'segment_pdg'),
                             feat_keys=('coord', 'energy'))])
    batch = collate_fn([ds[0], ds[1]])
    _assert(_fps(batch), C2_BATCH, jaxtpc_is_synthetic)


def test_pf1_label_decoration_parity(jaxtpc_data_root, jaxtpc_is_synthetic):
    lab = ('segment_pdg', 'instance_particle')
    step = _ds(jaxtpc_data_root, ('step',), labels=lab).get_data(0)['step']
    hits = _ds(jaxtpc_data_root, ('hits',), labels=lab).get_data(0)['hits']
    if jaxtpc_is_synthetic:
        assert _fp(step['segment_pdg']) == STEP_SEG
        assert _fp(step['instance_particle']) == STEP_INST
        assert _fp(hits['segment_pdg']) == HITS_SEG
        assert _fp(hits['instance_particle']) == HITS_INST
    else:
        assert _structural(_fp(step['segment_pdg'])) == _structural(STEP_SEG)
        assert _structural(_fp(hits['instance_particle'])) == \
            _structural(HITS_INST)
