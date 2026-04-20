# pimm-data

Multimodal detector dataset loaders for particle-imaging ML workflows.

Reads simulation output produced by:

- **JAXTPC** — Liquid Argon TPC simulation (`seg` / `sensor` / `inst` / `labl` HDF5)
- **LUCiD** — Water Cherenkov / PhotonSim simulation (`seg` / `sensor` / `inst` / `labl` HDF5, `format_version >= 3`)

Datasets inherit from `torch.utils.data.Dataset` and return nested
`dict[str, dict[str, np.ndarray]]` samples (one sub-dict per modality).
No transforms, no test-time augmentation, no framework glue — that
belongs in the downstream training framework (e.g. pimm).

## Install (local editable)

```bash
pip install -e /path/to/pimm-data
```

## Quick start

```python
from pimm_data import JAXTPCDataset, LUCiDDataset

ds = JAXTPCDataset(
    data_root="/path/to/jaxtpc_dataset",
    modalities=("seg", "labl"),
    label_key="particle",
)
sample = ds[0]            # nested dict
print(sample['seg'].keys())  # coord, energy, segment, instance, ...

wc = LUCiDDataset(
    data_root="/path/to/wc_dataset",
    modalities=("sensor", "inst", "labl"),
)
sample = wc[0]
print(sample['inst'].keys())  # coord, energy, instance, segment, ...
```

LUCiD labels (`instance`, `segment`) are particle-level by default
(`instance = particle_idx`, `segment = per_particle.category`). For
ancestor-level grouping, the labl reader precomputes
`ancestor_particle_idx` on both `per_particle` and `per_track` so a
downstream transform can remap in one line:

```python
# inst → ancestor-level instance
data['inst']['instance'] = (
    data['labl']['particle']['ancestor_particle_idx']
    [data['inst']['particle_idx']]
)
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
