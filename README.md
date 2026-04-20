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
sample = ds[0]               # nested dict
print(sample['seg'].keys())  # coord, energy, segment, instance, ...

wc = LUCiDDataset(
    data_root="/path/to/wc_dataset",
    modalities=("sensor", "inst", "labl"),
)
sample = wc[0]
print(sample['inst'].keys())  # coord, energy, instance, segment, ...
```

## Data layout

Both datasets expect `data_root` to be a directory with one
subdirectory per modality, each holding sharded HDF5 files named
`{dataset_name}_{modality}_{shard}.h5`.

```
data_root/
    seg/     {dataset_name}_seg_0000.h5     {dataset_name}_seg_0001.h5    ...
    sensor/  {dataset_name}_sensor_0000.h5  {dataset_name}_sensor_0001.h5 ...
    inst/    {dataset_name}_inst_0000.h5    {dataset_name}_inst_0001.h5   ...
    labl/    {dataset_name}_labl_0000.h5    {dataset_name}_labl_0001.h5   ...
```

Readers are index-synchronized by event ordinal across modalities —
`seg/` event 0 and `sensor/` event 0 refer to the same physics event.
Modality subdirectories you don't request can be absent. If `data_root`
itself contains the `.h5` files (no modality subdirs), the readers fall
back to searching `data_root` directly.

`dataset_name` is the file prefix: `'sim'` is JAXTPC's default, `'wc'`
is LUCiD's.

## Modality combinations

Each modality produces a sub-dict at a top-level key of the same name.

| Modality | Contains | Point-cloud dim |
|---|---|---|
| `seg`    | 3D truth deposits (one row per Geant4 step) | 3D |
| `sensor` | Raw sparse detector response (pixels / PMT hits) | 2D (JAXTPC), 3D (LUCiD) |
| `inst`   | Per-instance decomposition of `sensor` (same pixels/PMTs split by source particle) | same as `sensor` |
| `labl`   | Dimension tables: `per_event`, `per_particle`, `per_track` (JAXTPC: `v{N}` per volume) | — |

`labl` has no point cloud of its own. It's joined onto an
instance-bearing modality (`seg` or `inst`) to attach `segment` and
`instance` columns. Two combinations are therefore rejected at
construction time:

- `('labl',)` — labl alone has nothing to join against.
- `('sensor', 'labl')` — `sensor` is aggregated across particles, so
  per-particle labels can't be attached row-wise. Add `inst` (or `seg`)
  to the tuple.

## Output schema

```python
sample = {
    'seg':    {'coord': (N,3), 'energy': (N,1), 'time': ...,
               'track_idx': (N,), 'instance': (N,), 'segment': (N,),
               ...},                            # only if 'seg' in modalities
    'sensor': {'coord': (H,d), 'energy': (H,1), 'time': ...,
               'sensor_idx': (H,)},             # only if 'sensor' in modalities
    'inst':   {'coord': (E,d), 'energy': (E,1), 'time': ...,
               'sensor_idx': (E,), 'particle_idx': (E,),
               'instance': (E,), 'segment': (E,)},  # only if 'inst' in modalities
    'labl':   {'event':    {'t0': (), 'overall_containment': (), ...},
               'particle': {'category': (P,), 'ancestor_particle_idx': (P,), ...},
               'track':    {'track_id': (T,), 'pdg': (T,),
                            'ancestor_particle_idx': (T,), ...}},  # LUCiD
                                                # JAXTPC labl is keyed by volume:
                                                # {'v0': {...}, 'v1': {...}, ...}
    'name':  str,
    'split': str,
}
```

`segment` and `instance` are attached to `inst` and `seg` only when
`labl` is also in `modalities`. By default, `instance = particle_idx`
(LUCiD) or `group_id` (JAXTPC), and `segment` is the per-particle /
per-track label from `labl`.

### LUCiD: ancestor-level grouping

The LUCiD labl reader precomputes `ancestor_particle_idx` on both
`per_particle` and `per_track`, so switching from particle-level to
ancestor-level instance segmentation is one lookup:

```python
# inst → ancestor-level instance
data['inst']['instance'] = (
    data['labl']['particle']['ancestor_particle_idx']
    [data['inst']['particle_idx']]
)
# seg → ancestor-level instance (joins through per_track)
data['seg']['instance'] = (
    data['labl']['track']['ancestor_particle_idx']
    [data['seg']['track_idx']]
)
```

Both remaps share the same `particle_idx` index space, so `inst` and
`seg` stay joinable after the swap.

## Using with transforms

Transforms from pimm (and `pimm_data.transform`) assume flat keys
(`coord`, `segment`, …). Two adapters bridge the nested schema:

- **`ApplyToStream(stream=..., transforms=[...])`** — dispatches a
  sub-pipeline into one sub-dict. Inner transforms see a plain flat
  dict and are unmodified.
- **`Collect(stream=..., keys=..., feat_keys=...)`** — pulls keys from
  `data_dict[stream]` instead of the top level, and passes
  `name` / `split` through automatically.

```python
from pimm_data.transform import Compose

transform = Compose([
    dict(type='ApplyToStream', stream='seg', transforms=[
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='GridSample', grid_size=0.001, mode='train',
             return_grid_coord=True),
    ]),
    dict(type='ToTensor'),
    dict(type='Collect', stream='seg',
         keys=('coord', 'grid_coord', 'segment'),
         feat_keys=('coord', 'energy')),
])
```

Pass `required=True` to `ApplyToStream` to raise when the named stream
is absent; otherwise the transform is a no-op — handy for configs that
stay valid across different `modalities` choices.

## Layout

```
src/pimm_data/
    jaxtpc.py          JAXTPCDataset
    lucid.py           LUCiDDataset
    readers/           Per-modality HDF5 readers
    transform.py       Compose, TRANSFORMS, Collect, ...
    detector_transforms.py  PDGToSemantic, RemapSegment, ApplyToStream
    utils/pdg.py       pdg_to_semantic(pdg, scheme)
```

## Tests

```bash
pytest
```

Tests load real simulation output; point them at local datasets with
two environment variables:

```bash
export JAXTPC_DATA_ROOT=/path/to/jaxtpc_dataset
export LUCID_DATA_ROOT=/path/to/wc_dataset
pytest
```

Tests skip gracefully when the corresponding root isn't set or doesn't
exist, so partial setups still run the rest of the suite.
