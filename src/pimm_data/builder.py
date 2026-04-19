"""Dataset registry and build function."""

from ._registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build a dataset instance from a config dict ``{type: ClassName, ...}``."""
    return DATASETS.build(cfg)
