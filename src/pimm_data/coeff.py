"""CoeffTPCDataset — the wavelet-coefficient corpus (the FM's input).

A single-family sibling of :class:`~pimm_data.optical.OpticalDataset`. It composes
:class:`~pimm_data.readers.coeff_tpc.CoeffTPCReader` over flat-columnar coeff
shards and emits a **nested** dict with a ``coeff`` part of per-coefficient rows::

    {
      'coeff': {
        'band':      (M,)   int32    — band index [cA, cD_L, …, cD_1]
        'plane_gid': (M,)   int32    — canonical plane id (v*3 + {U,V,Y})
        'wire':      (M,)   int32    — signal index within the plane
        'tau':       (M,)   int32    — within-band coeff index
        'value':     (M, 1) float32  — RAW coeff (normalized at tokenize)
      },
      'name': str, 'split': str,
    }

Targets are separate joined modalities (``modalities=('coeff','coeff_clean')``):
the clean target is its own ``coeff_clean`` shard family, aligned by identity. The
model tokenizer (in pimm) turns rows → tokens and normalizes with the shard's
``norm_sigma`` (exposed on the reader); this dataset only yields rows.

Canonical Collect (coeff rows = one part, offset counts coeffs/event)::

    CoeffTPCDataset(data_root=..., dataset_name='...', transform=[
        dict(type='Collect', parts={'coeff': dict(
            keys=('band', 'plane_gid', 'wire', 'tau', 'value'), feat_keys=('value',),
            offset_keys_dict=dict(offset='band'))})])

Registered in :data:`pimm_data.DATASETS`.
"""

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from .readers.coeff_tpc import CoeffTPCReader


@DATASETS.register_module()
class CoeffTPCDataset(ShardEventDataset):
    """Flat-columnar wavelet-coefficient corpus (noisy ``coeff`` + optional
    joined ``coeff_clean`` target)."""

    VALID_MODALITIES = ('coeff', 'coeff_clean')

    def __init__(
        self,
        data_root,
        split='',
        modalities=('coeff',),
        dataset_name='coeff_tpc',
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        self._dataset_name = dataset_name
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.coeff_reader = None
        self.coeff_clean_reader = None
        if 'coeff' in self._modalities:
            self.coeff_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff'), split=split,
                dataset_name=dataset_name, modality='coeff')
        if 'coeff_clean' in self._modalities:
            self.coeff_clean_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff_clean'), split=split,
                dataset_name=dataset_name, modality='coeff_clean')

        self._canonical_reader = self.coeff_reader or self.coeff_clean_reader
        self._build_joint_index(source_label=f"CoeffTPCDataset({data_root!r})")

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )

    def get_data(self, idx):
        real_idx = idx % len(self.data_list)
        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }
        if self.coeff_reader is not None:
            data['coeff'] = self._build_coeff(self.coeff_reader.read_event(real_idx))
        if self.coeff_clean_reader is not None:
            data['coeff_clean'] = self._build_coeff(self.coeff_clean_reader.read_event(real_idx))
        return data

    @staticmethod
    def _build_coeff(raw):
        """Neutral coeff rows: coords as columns + value as an (M,1) feature."""
        return {
            'band': raw['band'],
            'plane_gid': raw['plane_gid'],
            'wire': raw['wire'],
            'tau': raw['tau'],
            'value': raw['value'][:, None].astype(np.float32),
        }

    @property
    def norm_sigma(self):
        """The shard's cross-event normalization σ table (for the tokenizer)."""
        return self._canonical_reader.norm_sigma if self._canonical_reader else None

    @property
    def band_lengths(self):
        return self._canonical_reader.band_lengths if self._canonical_reader else None
