"""Shared pytest fixtures.

Tests run against real JAXTPC / LUCiD production datasets when available.
Set JAXTPC_DATA_ROOT and LUCID_DATA_ROOT to override default paths.
Tests that need data skip gracefully when the dataset is absent.
"""

import os
import pytest

_JAXTPC_DEFAULT = '/home/oalterka/desktop_linux/JAXTPC/dataset_20'
_LUCID_DEFAULT = '/home/oalterka/desktop_linux/diffWC/LUCiD/viewer/sample_v3'

# v3 layout requires all four modality subdirs; tests skip if any are missing
# rather than crashing partway through on a pre-v3 (seg/sensor only) dataset.
_REQUIRED_SUBDIRS = ('seg', 'sensor', 'inst', 'labl')


def _resolve(env_var, default):
    path = os.environ.get(env_var, default)
    if not os.path.isdir(path):
        return None, f"root not found: {path}"
    missing = [d for d in _REQUIRED_SUBDIRS
               if not os.path.isdir(os.path.join(path, d))]
    if missing:
        return None, (f"{path} is not a v3 layout "
                      f"(missing: {', '.join(missing)})")
    return path, None


@pytest.fixture(scope='session')
def jaxtpc_data_root():
    path, reason = _resolve('JAXTPC_DATA_ROOT', _JAXTPC_DEFAULT)
    if path is None:
        pytest.skip(f"JAXTPC dataset not available ({reason}); "
                    "set JAXTPC_DATA_ROOT to a v3 dataset to enable")
    return path


@pytest.fixture(scope='session')
def lucid_data_root():
    path, reason = _resolve('LUCID_DATA_ROOT', _LUCID_DEFAULT)
    if path is None:
        pytest.skip(f"LUCiD dataset not available ({reason}); "
                    "set LUCID_DATA_ROOT to a v3 dataset to enable")
    return path
