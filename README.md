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
    label_key="pdg",          # or "cluster" / "interaction" / "ancestor"
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
| `sensor` | Raw sparse detector response (pixels / PMT hits) | 2D (JAXTPC wire), 3D (JAXTPC pixel, LUCiD) |
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

A sample is a **nested dict**. Top level has one key per active
modality, plus `name` and `split`. Each modality's sub-dict is **flat**
— every listed key is a direct sibling, nothing is nested further.

> Naming collision, up front: `seg` the *modality* (a point cloud of 3D
> particle-track segments) is different from `segment` the *column*
> (pimm's convention for a per-point semantic class label). `segment`
> is a `(N,)` array that sits alongside `coord`, `energy`, and the rest
> — it's not a sub-dict.

LUCiD and JAXTPC have genuinely different columns per modality, so each
is documented separately.

### LUCiD sub-dicts

```python
data['seg'] = {
    'coord':        (N, 3),    # midpoint of Geant4 step
    'energy':       (N, 1),    # edep
    'time':         (N, 1),
    'track_idx':    (N,),      # FK → labl.track
    # include_physics=True (default):
    'direction':    (N, 3),
    'beta_start':   (N, 1),
    'n_cherenkov':  (N, 1),
    # present only when 'labl' is also in modalities:
    'particle_idx': (N,),      # = labl.track.particle_idx[track_idx]
    'instance':     (N,),      # = particle_idx
    'segment':      (N,),      # = labl.particle.category[particle_idx]
}

data['sensor'] = {
    'coord':       (H, 3),     # PMT xyz, indexed via sensor_positions[sensor_idx]
    'energy':      (H, 1),     # post-smearing PE
    'time':        (H, 1),
    'sensor_idx':  (H,),
}

data['inst'] = {
    'coord':        (E, 3),    # same PMT geometry as sensor
    'energy':       (E, 1),    # per-particle PE (pre-smearing)
    'time':         (E, 1),
    'sensor_idx':   (E,),
    'particle_idx': (E,),      # FK → labl.particle
    'instance':     (E,),      # = particle_idx
    # present only when 'labl' is also in modalities:
    'segment':      (E,),      # = labl.particle.category[particle_idx]
}

data['labl'] = {
    'event':    {'t0': (), 'overall_containment': ()},
    'particle': {'category':              (P,),
                 'containment':           (P,),
                 'ancestor_particle_idx': (P,),
                 'genealogy_data':        (G,),
                 'genealogy_offsets':     (P+1,),
                 'ext_genealogy_data':    (...,),
                 'ext_genealogy_offsets': (P+1,)},
    'track':    {'track_id':              (T,),
                 'pdg':                   (T,),
                 'parent_id':             (T,),
                 'particle_idx':          (T,),  # FK → labl.particle
                 'ancestor':              (T,),  # root ancestor track_id
                 'ancestor_particle_idx': (T,),
                 'interaction':           (T,),
                 'initial_energy':        (T,),
                 'n_cherenkov':           (T,)},
}
```

### JAXTPC sub-dicts

JAXTPC is volume-partitioned with two readout types:

- **wire** (U/V/Y planes, 2D `wire × time` per plane)
- **pixel** (single `Pixel` plane per volume, 3D `py × pz × time`)

`sensor` and `inst` are point clouds merged across planes, with per-plane
raw arrays kept in a nested `raw` dict for transforms that need them.
The readout type is auto-detected from the HDF5 `/config.readout_type`
attribute and surfaced as `data['sensor']['readout_type']` /
`data['inst']['readout_type']` (both always equal). `coord` shape,
`planes` labels, and `raw` per-plane column names all follow the readout
type — see below.

```python
data['seg'] = {
    'coord':     (N, 3),
    'energy':    (N, 1),
    'volume_id': (N, 1),       # which TPC volume each deposit came from
    # include_physics=True (default):
    'dx': (N, 1), 'theta': (N, 1), 'phi': (N, 1),
    'charge': (N, 1), 'photons': (N, 1), 't0_us': (N, 1),
    # present only when 'labl' is also in modalities:
    'instance': (N,),          # = segment_to_track (raw Geant4 track_id)
    'segment':  (N,),          # = track_{label_key} (e.g. track_pdg)
}

data['sensor'] = {
    'coord':         (M, D),        # D=2 for wire (wire,time), D=3 for pixel (py,pz,time)
    'energy':        (M, 1),
    'plane_id':      (M, 1),
    'planes':        [str, ...],    # plane labels in plane_id order
                                    #   wire:  'volume_{v}_{U|V|Y}'
                                    #   pixel: 'volume_{v}_Pixel'
    'readout_type':  'wire' | 'pixel',
    'raw': {plane_label: {...}},    # wire:  {'wire', 'time', 'value'}
                                    # pixel: {'py', 'pz', 'time', 'value'}
}

data['inst'] = {
    'coord':         (E, D),        # D=2 for wire, D=3 for pixel
    'energy':        (E, 1),
    'plane_id':      (E, 1),
    'planes':        [str, ...],
    'readout_type':  'wire' | 'pixel',
    'raw': {plane_label: {...}},    # wire:  {'wire', 'time', 'group_id', 'charge'}
                                    # pixel: {'py', 'pz', 'time', 'group_id', 'charge'}
    'instance': (E,),               # = group_id
    # present only when 'labl' is also in modalities:
    'segment':  (E,),               # = track_{label_key}, joined via group_to_track
}

data['labl'] = {              # keyed by volume
    'v0': {'track_ids':        (T0,),  # unique track_ids in this volume
           'track_pdg':        (T0,),
           'track_cluster':    (T0,),
           'track_interaction':(T0,),
           'track_ancestor':   (T0,),
           'segment_to_track': (N_v0,)},  # seg deposit → track_id (row-aligned)
    'v1': {...},
}

data['bridges'] = {           # only when 'inst' is loaded
    'group_to_track_v0':   (G0,),   # inst group_id → track_id
    'segment_to_group_v0': (N_v0,), # seg deposit → group_id
    'qs_fractions_v0':     ...,
    # ...and _v1, _v2, etc.
}
```

**Note on `instance` semantics.** LUCiD puts `instance = particle_idx`
(one instance per physics particle, coarsening over Geant4 tracks that
belong to the same particle). JAXTPC puts `instance = track_id` (raw
Geant4 track IDs) for seg, and `instance = group_id` (inst's native
grouping key) for inst. If a task needs one convention across both
datasets, remap in a transform.

**sensor vs inst.** `sensor` is the detector-level response (post
smearing / noise, when enabled). `inst` is the particle-level
decomposition (pre-smearing truth). You cannot assume they align
row-for-row: when smearing is on, `sensor` can carry hits that have
no counterpart in `inst` — those hits have no particle of origin and
should be treated as background (e.g. `ignore_index`) by downstream
training.

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
