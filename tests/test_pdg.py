"""Tests for pimm_data.utils.pdg."""

import numpy as np
import pytest

from pimm_data.utils import pdg_to_semantic


def test_motif_5cls():
    pdg = np.array([22, 11, 13, 2212, 211, 9999], dtype=np.int32)
    labels = pdg_to_semantic(pdg, scheme='motif_5cls')
    # 22/11 → shower(0), 13/211/2212 → track(1), 9999 → default led(4)
    assert labels.tolist() == [0, 0, 1, 1, 1, 4]


def test_pid_6cls():
    pdg = np.array([22, 11, 13, 211, 2212, 321], dtype=np.int32)
    labels = pdg_to_semantic(pdg, scheme='pid_6cls')
    # 22→0, 11→1, 13→2, 211→3, 2212→4, 321→other(5)
    assert labels.tolist() == [0, 1, 2, 3, 4, 5]


def test_none_returns_none():
    assert pdg_to_semantic(np.array([22, 13]), scheme='none') is None


def test_custom():
    pdg = np.array([100, 200, 300], dtype=np.int32)
    labels = pdg_to_semantic(pdg, scheme='custom',
                              custom_map={100: 0, 200: 1})
    # 300 → default = max(0,1)+1 = 2
    assert labels.tolist() == [0, 1, 2]


def test_custom_requires_map():
    with pytest.raises(ValueError):
        pdg_to_semantic(np.array([22]), scheme='custom')


def test_unknown_scheme():
    with pytest.raises(ValueError):
        pdg_to_semantic(np.array([22]), scheme='bogus')


def test_flattens_input():
    pdg = np.array([[22, 11], [13, 2212]], dtype=np.int32)
    labels = pdg_to_semantic(pdg, scheme='motif_5cls')
    assert labels.shape == (4,)
    assert labels.tolist() == [0, 0, 1, 1]
