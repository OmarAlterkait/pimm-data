# pimm-data

Multimodal detector dataset loaders for particle-imaging ML workflows.

Reads simulation output produced by:

- **JAXTPC** — Liquid Argon TPC simulation (`seg` / `sensor` / `inst` / `labl` HDF5)
- **LUCiD** — Water Cherenkov / PhotonSim simulation (`seg` / `sensor` HDF5)

Datasets inherit from `torch.utils.data.Dataset` and return flat
`dict[str, np.ndarray]` samples. No transforms, no test-time augmentation,
no framework glue — that belongs in the downstream training framework
(e.g. pimm).

## Install (local editable)

```bash
pip install -e /path/to/pimm-data
```

## Quick start

```python
from pimm_data import JAXTPCDataset, LUCiDDataset

ds = JAXTPCDataset(
    data_root="/path/to/dataset",
    modalities=("seg", "labl"),
    label_key="particle",
)
sample = ds[0]       # dict[str, np.ndarray]
print(sample.keys()) # coord, energy, segment, ...
```

## Layout

```
src/pimm_data/
    jaxtpc.py          JAXTPCDataset
    lucid.py           LUCiDDataset
    readers/           Per-modality HDF5 readers
    utils/pdg.py       pdg_to_semantic(pdg, scheme)
```

## Tests

```bash
pytest
```

Synthetic HDF5 fixtures are generated lazily into `tests/fixtures/`
(gitignored).
