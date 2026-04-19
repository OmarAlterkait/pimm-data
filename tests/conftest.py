"""Shared pytest fixtures.

Tests run against real JAXTPC / LUCiD production datasets when available.
Set JAXTPC_DATA_ROOT and LUCID_DATA_ROOT to override default paths.
Tests that need data skip gracefully when the dataset is absent.
"""

import os
import pytest

_JAXTPC_DEFAULT = '/home/oalterka/desktop_linux/JAXTPC/dataset_1'
_LUCID_DEFAULT = '/home/oalterka/desktop_linux/JAXTPC/dataset_wc'


def _resolve(env_var, default):
    path = os.environ.get(env_var, default)
    if not os.path.isdir(path):
        return None
    return path


@pytest.fixture(scope='session')
def jaxtpc_data_root():
    path = _resolve('JAXTPC_DATA_ROOT', _JAXTPC_DEFAULT)
    if path is None:
        pytest.skip("JAXTPC dataset not available; "
                    "set JAXTPC_DATA_ROOT to enable")
    return path


@pytest.fixture(scope='session')
def lucid_data_root():
    path = _resolve('LUCID_DATA_ROOT', _LUCID_DEFAULT)
    if path is None:
        pytest.skip("LUCiD dataset not available; "
                    "set LUCID_DATA_ROOT to enable")
    return path
