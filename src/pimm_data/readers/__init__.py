"""Readers for detector HDF5 files.

Each reader is self-contained: given a data_root and split, it discovers
shard files, builds an event index, and exposes ``read_event(idx)``
returning a flat ``dict[str, np.ndarray]``.
"""

from .jaxtpc_seg import JAXTPCSegReader
from .jaxtpc_sensor import JAXTPCSensorReader
from .jaxtpc_labl import JAXTPCLablReader
from .jaxtpc_inst import JAXTPCInstReader
from .lucid_seg import LUCiDSegReader
from .lucid_sensor import LUCiDSensorReader
from .lucid_inst import LUCiDInstReader
from .lucid_labl import LUCiDLablReader

__all__ = [
    "JAXTPCSegReader",
    "JAXTPCSensorReader",
    "JAXTPCLablReader",
    "JAXTPCInstReader",
    "LUCiDSegReader",
    "LUCiDSensorReader",
    "LUCiDInstReader",
    "LUCiDLablReader",
]
