# pimm-data

Multimodal detector dataset loaders for particle-imaging ML workflows.

Reads simulation output produced by:

- **JAXTPC** — Liquid Argon TPC simulation (`step` / `sensor` / `hits` / `labl` HDF5)
- **LUCiD** — Water Cherenkov / PhotonSim simulation (`step` / `sensor` / `hits` / `labl` HDF5, `format_version >= 3`)
- **Optical** — PMT light (goop) output as per-chunk waveforms (`OpticalDataset`; label & east/west schemas)

Datasets inherit from `torch.utils.data.Dataset`. `get_data(idx)` returns a
nested `dict[str, dict[str, np.ndarray]]` sample (one sub-dict per modality);
`ds[idx]` runs the transform pipeline, whose terminal `Collect` emits a **flat,
underscore-prefixed** dict (`step_coord`, `sensor_wire`, …) + per-part `offset`
+ a small `_roles` map. `collate_fn` reduces those samples into one batch
(concat by `offset`, role-driven). See **Transform pipeline** below.

## Quick start

### Install

```bash
pip install -e /path/to/pimm-data
```

### Load a JAXTPC dataset

```python
from pimm_data import JAXTPCDataset

ds = JAXTPCDataset(data_root="/path/to/jaxtpc_data", split="", modalities=("step",),
                   labels=("segment_motif", "instance_fragment"))
sample = ds.get_data(0)          # numpy arrays, no transforms
print(sample["step"].keys())     # coord, energy, group, segment_motif, instance_fragment, ...
```

### Load a LUCiD dataset

```python
from pimm_data import LUCiDDataset

ds = LUCiDDataset(data_root="/path/to/wc_data", modalities=("sensor",))
sample = ds.get_data(0)
print(sample["sensor"].keys())   # coord, energy, time, sensor_idx
```

### Load an Optical (PMT light) dataset

```python
from pimm_data import OpticalDataset

# label schema (per-interaction chunks); 'east_west' for light_output.h5
ds = OpticalDataset(data_root="/path/to/optical", dataset_name="...", schema="label")
sample = ds.get_data(0)
print(sample["sensor"].keys())   # pmt_id, t0_ns, length, pe, instance, adc, _roles
```

Each `sensor` row is a waveform **chunk**; the raw samples are a packed second
row-space (`adc`), tagged `('instance','sensor_wave_offset')` so collate concats
them and `split_event` slices by sample count. See `docs/CAMPAIGN.md`.

### Training-ready example

```python
from torch.utils.data import DataLoader
from pimm_data import JAXTPCDataset, collate_fn

ds = JAXTPCDataset(
    data_root="/path/to/jaxtpc_data",
    split="",
    modalities=("step",),
    labels=("segment_motif",),          # read labl; per-point motif class
    transform=[
        dict(type="Apply", on="step", transforms=[
            dict(type="GridSample", grid_size=0.5, mode="train",
                 return_grid_coord=True),
        ]),
        dict(type="Collect", parts={
            "step": dict(keys=("coord", "grid_coord", "segment_motif"),
                         feat_keys=("coord", "energy"))}),
    ],
)
loader = DataLoader(ds, batch_size=4, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)
batch = next(iter(loader))
# batch -> step_coord, step_grid_coord, step_segment_motif, step_feat, step_offset, ...
```

> **`transform` takes a plain list of dicts**, not a `Compose` object.
> `Compose` wrapping is handled internally.

> **`get_data(idx)`** returns raw numpy arrays with no transforms.
> **`ds[idx]`** applies the full transform pipeline and returns tensors.

> **`split` parameter:** JAXTPC defaults to `split='train'`, looking in
> `data_root/step/train/`. LUCiD defaults to `split=''`. If your files
> sit directly in `data_root/step/` with no split subdirectory, pass
> `split=''`.

> **`labels=`** reads `labl`. JAXTPC: name the per-point labels wanted —
> classes `segment_motif`, `segment_pid`, `segment_pdg`; ids `instance_fragment`,
> `instance_particle`, `instance_interaction`; `is_primary`, `momentum`, `vertex`
> (or `'all'`) — the PILArNet key names; `Collect` picks the training target.
> LUCiD: `labels=True` attaches `segment`/`instance`. See docs/LABELS.md.

---

## Naming conventions

### Modality names (top-level dict keys, directory names, file tags)

| Modality | Meaning | Why this name |
|----------|---------|---------------|
| `step`   | 3D energy deposits (one row per Geant4 simulation step) | Standard HEP abbreviation. |
| `sensor` | Aggregated detector readout (pixels, wires, or PMTs) | What the detector measures. |
| `hits`   | Per-particle charge attribution at sensor elements | Natural detector term ("PMT hits" in WC, "track hits" in TPC). |
| `labl`   | Label tables (JAXTPC: pysupera's; read via `labels=`) | Short for "labels." |

These appear as directory names (`step/`), HDF5 file tags
(`sim_step_0000.h5`), and top-level keys (`data['step']`).

### ML column names (per-point arrays inside a modality sub-dict)

| Column     | Meaning | Lives inside |
|------------|---------|-------------|
| `segment`  | Semantic segmentation class ID per point (LUCiD) | `data['step']['segment']`, `data['hits']['segment']` |
| `instance` | Instance segmentation ID per point (LUCiD) | `data['step']['instance']`, `data['hits']['instance']` |
| `segment_*` / `instance_*` | Named per-point classes / ids (JAXTPC, PILArNet names) | `data['step']['segment_motif']`, `data['hits']['instance_particle']`, … |

These are **not** modalities — they are per-point label columns attached
when labels are requested.

### Foreign-key arrays

| FK array | Meaning | Scope |
|----------|---------|-------|
| `deposit_to_group` | Step deposit → hits group ID | Per volume, in the hits file; loaded as `step.group` |
| `group_to_track`   | Hits group → track ID | Per volume, in the hits file (supera `labl`: `group_track`) |

All are suffixed `_v{N}` for multi-volume detectors (e.g. `deposit_to_group_v0`).

### The `group` part

`group` appears whenever `hits` is loaded or `labels=` is set (JAXTPC only):
one row per JAXTPC group, event-wide (`volume`; with labels also `track`,
`fragment`).
`step.group` and `hits.group` reference it, and `step.qs` is each deposit's
share of its group's charge. It is the step <-> hits correspondence, with or
without labels.

---

## Data layout

Each dataset expects sharded HDF5 files organized by modality:

```
data_root / [modality_subdir/] / [split/] / {dataset_name}_{modality}_{shard:04d}.h5
```

**JAXTPC** (`dataset_name='sim'`):

```
data_root/
  step/     sim_step_0000.h5    sim_step_0001.h5    ...
  sensor/   sim_sensor_0000.h5  ...
  hits/     sim_hits_0000.h5    ...
  labl/     sim_labl_0000.h5    ...
```

**LUCiD** (`dataset_name='wc'`):

```
data_root/
  step/     wc_step_0000.h5     ...
  sensor/   wc_sensor_0000.h5   ...
  hits/     wc_hits_0000.h5     ...
  labl/     wc_labl_0000.h5     ...
```

**Path resolution rules:**

1. **Modality subdirectory** — `data_root/modality/` is checked first.
   If it doesn't exist, the reader falls back to `data_root` directly.

2. **Split subdirectory** — Readers glob `reader_root/split/{name}_{mod}_*.h5`
   first, falling back to `reader_root/{name}_{mod}_*.h5` if empty.
   `split=''` (LUCiD default) skips the split subdirectory entirely.

3. **Only requested modalities must be present** — Directories for
   modalities not in your `modalities` tuple are never accessed.

Readers are index-synchronized by a **joint cross-modality index**
(`_joint_index.build_joint_index`): one global `idx` resolves to the same
physics `(file, event_num)` in every loaded modality. The index is the
*intersection* of each modality's present events per shard, so an event missing
from one modality (a production gap, a corrupt/dangling shard, or filtered out
by `min_deposits`/`min_segments`) is dropped from **all** loaded modalities to
keep them aligned — never silently misaligned. Shards align by sorted-glob
position; set `strict_lengths=True` to raise on any cross-modality shard/event
mismatch instead of aligning on the intersection.

---

## Modality combinations

Each modality produces a sub-dict at a top-level key of the same name.

| Modality | Contains | Point-cloud dim |
|---|---|---|
| `step`   | 3D truth deposits (one row per Geant4 step) | 3D |
| `sensor` | Raw sparse detector response | 2D (wire), 3D (pixel, PMT) |
| `hits`   | Per-particle decomposition of `sensor` | same as `sensor` |

JAXTPC takes `step`, `sensor`, `hits` in any combination; `labels=` adds the
label tables and per-point labels on `step`/`hits` (it needs one of them).
LUCiD also accepts `labl` as a modality; `('labl',)` and `('sensor', 'labl')`
are rejected there (nothing to join against).

---

## Output schema

### Top-level structure

```python
data = ds.get_data(idx)
# {
#     'name': str, 'split': str,
#     'step':    {...},    # when 'step' in modalities
#     'sensor':  {...},    # when 'sensor' in modalities
#     'hits':    {...},    # when 'hits' in modalities
#     'labl':    {...},    # LUCiD, when 'labl' in modalities
#     'group':   {...},    # JAXTPC, when 'hits' is loaded or labels= is set
#     'track', 'particle', 'interaction': {...},   # JAXTPC, labels= only
# }
```

### JAXTPC sub-dicts

JAXTPC supports two readout types (auto-detected from HDF5):
**wire** (U/V/Y planes, coord is `(M, 2)`) and **pixel** (coord is `(M, 3)`).

```python
data['step'] = {
    'coord':     (N, 3) float32,
    'energy':    (N, 1) float32,
    'volume_id': (N, 1) int32,
    # include_physics=True (default):
    'dx': (N, 1), 'theta': (N, 1), 'phi': (N, 1),
    'charge': (N, 1), 'photons': (N, 1), 't0_us': (N, 1),
    # hits loaded or labels= set:
    'group':    (N,) int32,     # ref -> group (-1: no group)
    'qs':       (N, 1) float32, # share of its group's charge
    # labels= only, the names asked for, each (N, 1):
    'segment_motif': ..., 'instance_particle': ..., ...
}

data['sensor'] = {
    'coord':        (M, D) float32,    # D=2 wire, D=3 pixel
    'energy':       (M, 1) float32,
    'plane_id':     (M, 1) int32,
    'planes':       [str, ...],        # plane labels in plane_id order
    'readout_type': 'wire' | 'pixel',
    'raw': {plane_label: {...}},       # per-plane arrays before merge
}

data['hits'] = {
    'coord':        (E, D) float32,
    'energy':       (E, 1) float32,
    'plane_id':     (E, 1) int32,
    'group':        (E,) int32,        # ref -> group
    'planes':       [str, ...],
    'readout_type': 'wire' | 'pixel',
    # labels= only, as for step
}

data['group'] = {             # hits loaded or labels= set
    'volume': (G,) int32,        # + 'track', 'fragment' with labels=
}
# labels= only: data['track'], data['particle'], data['interaction']
# (pysupera's tables, docs/LABELS.md)
```

### LUCiD sub-dicts

```python
data['step'] = {
    'coord':       (N, 3) float32,    # midpoint of Geant4 step
    'energy':      (N, 1) float32,
    'time':        (N, 1) float32,
    'track_idx':   (N,) int32,        # FK → labl.track
    'contained':   (N,) bool,         # step inside detector volume
    # include_physics=True (default):
    'direction':   (N, 3), 'beta_start': (N, 1), 'n_cherenkov': (N, 1),
    # present when 'labl' in modalities:
    'particle_idx': (N,), 'instance': (N,), 'segment': (N,),
}

data['sensor'] = {
    'coord':      (H, 3) float32,    # PMT positions
    'energy':     (H, 1) float32,    # post-smearing PE
    'time':       (H, 1) float32,
    'sensor_idx': (H,) int32,
}

data['hits'] = {
    'coord':        (E, 3) float32,
    'energy':       (E, 1) float32,
    'time':         (E, 1) float32,
    'sensor_idx':   (E,) int32,
    'particle_idx': (E,) int32,
    'instance':     (E,) int32,       # = particle_idx
    # present when 'labl' in modalities:
    'segment':      (E,) int32,       # = labl.particle.category
}

data['labl'] = {
    'event':    {'t0': () float32, 'contained': () bool},
    'particle': {'category': (P,) int32, 'contained': (P,) bool,
                 'ancestor_particle_idx': (P,) int32, ...},
    'track':    {'track_id': (T,) int32, 'pdg': (T,) int32,
                 'parent_id': (T,) int32, 'particle_idx': (T,) int32,
                 'ancestor': (T,) int32,
                 'ancestor_particle_idx': (T,) int32, ...},
}
```

LUCiD's HDF5 also carries a `per_interaction/` group (per-neutrino-vertex
metadata: source type, t0, vertex, neutrino PDG/energy, primaries CSR).
The reader does not yet surface it; expose it when pile-up workflows
need per-vertex grouping.

### Instance and segment semantics

**Instance** has different physical meaning per dataset:

| Dataset | Modality | `instance` = |
|---------|----------|-------------|
| JAXTPC  | both     | named: `instance_fragment` / `instance_particle` (rows of `data['particle']`), `instance_interaction` (rows of `data['interaction']`) |
| LUCiD   | both     | `particle_idx` (FK into labl.particle) |

JAXTPC instance ids are table rows per event (`get_data`): shared by step and
hits, with gaps; stuff and unowned points are -1. The dataset tags them
`('label', key)`, so `Collect(parts=...)` + collate compacts them per event and
makes them unique across the batch (-1 stays -1). Both `Collect` forms carry
roles (the single-part form is one unprefixed part), so batching is the same
for either.

**Mixing (Mix3D).** `dict(type='Mix3D', prob=p)` after `Collect` (train pipeline
only, before `ToDevice`) merges events `2i` and `2i+1`
of a `Collect`-ed batch (either form) by their offsets alone, so it works for any keys a
pipeline collects. It is correct because of roles: ids that must stay apart are
`('label', …)` or `('ref', part)` and are batch-unique after collate. A pipeline
that collects its own ids declares them in `Collect(..., roles=...)`; an id left
`point`-role would collide. Event-level tensors are refused (a merged pair has no
single value), and a batch without roles raises. Nothing is summed or densified:
a mixed sparse readout carries both events' rows.
If a task needs uniform instance convention, remap in a transform.

**Segment**:

| Dataset | Source | Values |
|---------|--------|------------|
| JAXTPC  | named | `segment_motif`, `segment_pid`, raw `segment_pdg` |
| LUCiD   | `labl.particle.category` | Detector-specific category integers |

Use `RemapSegment` to map raw values (e.g. `segment_pdg`, with `key=`) to dense class indices:

```python
dict(type='RemapSegment', scheme='motif_5cls')
```

---

## Transform pipeline

The dataset returns a nested dict → transforms process it → `Collect` emits a
flat, underscore-prefixed dict → `collate_fn` reduces samples into a batch.

```
get_data()  →  Apply(on='step', transforms=[...])  →  Collect(parts={'step': ...})  →  collate_fn
nested dict     augments step sub-dict (numpy)         flat step_* (tensors) + _roles      one batch
```

### Apply

Targets the part-agnostic transform library at one modality. `Apply(on='step')`
exposes `data['step']`'s keys as **bare** canonical names (`coord`, `segment`,
`energy`), runs the inner transforms, and writes them back into the sub-dict. A
multi-part `Apply(on=('step','sensor'))` is **implicitly shared** — same RNG draw
per part (co-registered rotate/flip); independent augmentation = separate `Apply`
blocks. If a part is missing it's a no-op (`required=True` to raise).
(`Apply(on='step', …)` is the single-part form; the old
`ApplyToModality`/`ApplyToStream` aliases have been removed.)

```python
dict(type='Apply', on='step', transforms=[
    dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
    dict(type='GridSample', grid_size=0.001, mode='train', return_grid_coord=True),
])
```

### Collect

Terminal transform. `parts={part: dict(keys=…, feat_keys=…, …)}` pulls each
part's keys, builds `feat` (concatenated `feat_keys`), tensorizes numpy → torch,
and flattens to **`part_key`** (`step_coord`, `step_segment_motif`, `step_feat`) plus
a derived `<part>_offset` and a `_roles` map for non-default-role keys. `name` /
`split` pass through. `offset_keys_dict=` declares extra offsets (e.g. a second
row-space); `roles=` tags keys (`raw` / `edge` / `instance` / `label` / `event`).

```python
dict(type='Collect', parts={
    'step': dict(keys=('coord', 'grid_coord', 'segment_motif'),
                 feat_keys=('coord', 'energy'))})
# Batch (after collate_fn): step_coord, step_grid_coord, step_segment_motif,
#                           step_feat, step_offset, name, split, _roles
```

`ToTensor` is **not needed** — `Collect` tensorizes (enabling fd-IPC under
`num_workers>0`). `Collect` must be **last**.

### Practical configs

**SSL on 3D step (multi-crop, no labels):**

```python
transform=[
    dict(type='Apply', on='step', transforms=[
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='Copy', keys_dict={'coord': 'origin_coord'}),
    ]),
    dict(type='MultiCrop', on='step', view_keys=('coord', 'origin_coord', 'energy'),
         global_view_num=2, local_view_num=6),
    dict(type='Collect', parts={
        'global': dict(keys=('coord', 'origin_coord', 'energy', 'offset'),
                       offset_keys_dict={}, feat_keys=('coord', 'energy')),
        'local':  dict(keys=('coord', 'energy', 'offset'),
                       offset_keys_dict={}, feat_keys=('coord', 'energy'))}),
]
```

**Semantic segmentation (step, `labels=('segment_motif',)`):**

```python
transform=[
    dict(type='Apply', on='step', transforms=[
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
        dict(type='GridSample', grid_size=0.001, mode='train', return_grid_coord=True),
    ]),
    dict(type='Collect', parts={
        'step': dict(keys=('coord', 'grid_coord', 'segment_motif'),
                     feat_keys=('coord', 'energy'))}),
]
```

**Particle panoptic on hits (`labels=('segment_pid', 'instance_particle')`):**

```python
transform=[
    dict(type='Apply', on='hits', transforms=[
        dict(type='GridSample', grid_size=1.0, mode='train', return_grid_coord=True),
    ]),
    dict(type='Collect', parts={
        'hits': dict(keys=('coord', 'grid_coord', 'segment_pid', 'instance_particle'),
                     feat_keys=('coord', 'energy'))}),
]
```

The panoptic pairs: `segment_motif` + `instance_fragment`, `segment_pid` +
`instance_particle`; `instance_interaction` is class-agnostic. A model that
reads the bare `segment`/`instance` gets them with
`dict(type='Copy', keys_dict={'segment_pid': 'segment', 'instance_particle': 'instance'})`.

Ready-to-run data-loader recipes for many challenges live in `configs/`
(see `docs/CAMPAIGN.md`).

---

## DataLoader and performance

### Standard setup

```python
from torch.utils.data import DataLoader
from pimm_data import JAXTPCDataset, collate_fn

ds = JAXTPCDataset(data_root="...", modalities=("step",),
                   labels=("segment_motif",), transform=[...])
loader = DataLoader(ds, batch_size=4, num_workers=4,
                    collate_fn=collate_fn, pin_memory=True)
```

### Why Collect must be the last transform

`Collect` converts numpy arrays to torch tensors. PyTorch's
`ForkingPickler` transfers tensors via file-descriptor sharing (~400
bytes) instead of pickling the full array. Without tensor output,
`num_workers > 0` is **slower** than serial for large events.

### Choosing num_workers

| GPU step time | Recommended workers |
|---|---|
| > 200 ms | 2 |
| 100–200 ms | 4 |
| < 100 ms | 8+ |

Per-worker memory is ~240 MB. **Set `prefetch_factor=1`** — the
PyTorch default of 2 costs ~25% throughput and 3× the p95 latency
jitter on LUCiD-style workloads (workers can keep up without buffering
ahead, so extra buffered batches just add memory pressure and queue
overhead). **Never use `num_workers=1`** — it pays full IPC cost with
no concurrency and is uniformly slower than `num_workers=0`.

Empirical numbers — LUCiD WAND `config_000013` (101k events, A100 box):

| workers | batch | prefetch | samples/s | ms/batch | notes |
|---|---|---|---|---|---|
| 0  | 32 | —  | 113  | 282 | single-process baseline |
| 4  | 32 | 1  | 333  |  96 | |
| 8  | 32 | 1  | 491  |  65 | |
| 16 | 32 | 1  | 821  |  39 | |
| **20** | **32** | **1** | **967** | **33** | **peak operating point** |
| 24 | 32 | 1  | 855  |  37 | past the knee; oversubscription |

The single-process per-event cost is 91% HDF5 I/O; pimm-data wrappers
+ transforms add ~9%. Past ~20 workers the cap shifts from the
filesystem to per-batch tensor IPC across the worker→main queue.
See `scripts/` for the diagnostic tools that produced these numbers.

### Profiling and benchmarking

Scripts under `scripts/` cover loader performance from complementary
angles. All three profilers take `--dataset {lucid,jaxtpc}` (default
`lucid`) plus `--data-root`/`--split`, so they run against either the
LUCiD (`wc_*`) or JAXTPC (`sim_*`) v3 datasets via one code path
(shared specs in `scripts/_profile_common.py`).

* **`benchmark_loader.py`** — sweep `(num_workers, batch_size, prefetch)`
  on real data and emit latency / throughput plots (shaded IQR bands).
  Per-batch histograms via `--hist-workers/--hist-batches` (`--hist-logx`
  for the heavy tail). `--target-events`/`--max-seconds-per-cell` give
  equal statistical power per cell; `--replot`/`--hist-replot` redraw from
  saved CSVs.
* **`profile_loader.py`** — per-stage breakdown of `__getitem__`
  (raw h5py I/O vs reader wrappers vs dataset assembly vs `Collect`),
  plus a cProfile hotspot dump.
* **`profile_scaling.py`** — isolates whether the loader ceiling is
  filesystem bandwidth or PyTorch DataLoader IPC by comparing raw
  `multiprocessing.Pool`, `ThreadPoolExecutor`, and same-shard
  contention scaling.
* **`transcode_codec.py`** — re-encode production HDF5 between codecs
  (gzip → blosc/zstd/lz4) to compare file size and read speed.
* **`examples/bench_dense.py`** — times the post-collate dense tail
  (`Densify`→`AddNoise`→`Digitize`) CPU vs GPU (`ToDevice`) on real
  doraemon wire data (`/sdf/data/neutrino/doraemon/wire_test_00_00_02`), and verifies the output bit-exact against a manual
  numpy scatter and a device-independent numpy oracle (`coherent_numpy=True`).
  Measured ~29× GPU speedup; only the sparse COO crosses PCIe.

**Codec finding:** JAXTPC loading is bound by gzip **decompression**
(CPU/GIL-serial — scales with worker *processes*, not threads), not
filesystem I/O. Re-encoding the output to `blosc-zstd` (now the JAXTPC
production default) reads ~2.3× faster *and* yields smaller files than
gzip; `blosc-lz4hc` reads ~4× faster at gzip's size. Reading any
blosc/zstd/lz4 file requires `hdf5plugin` (a declared dependency;
registered automatically on `import pimm_data`).

### Batch output

`collate_fn` concatenates variable-length point clouds per part and tracks
boundaries with a cumulative **`<part>_offset`** tensor (`(B,)`, no leading 0).
Keys are flat and underscore-prefixed; it is role-driven (a carried `_roles`
map): point/raw concat by offset, edges shift, labels renumber, instance/second
row-spaces concat by their own offset, event keys stack:

```python
batch['step_coord'].shape    # (total_points, 3)
batch['step_feat'].shape     # (total_points, 4)
batch['step_segment'].shape  # (total_points,)
batch['step_offset']         # tensor([N0, N0+N1, ...])  (B,), no leading 0
# e.g. optical also carries a second row-space:
# batch['sensor_offset'] = chunks/event, batch['sensor_wave_offset'] = samples/event
```

`inseg_collate_fn` flattens instance-segmentation query batches.

### Performance tips

1. **Use `volume=0`** to load a single TPC volume (2–4x faster).
2. **Load only needed modalities** — `hits` is heaviest (~90 MB/event).
3. **End transforms with `Collect`** — enables zero-copy IPC.
4. **`GridSample` early** — downsample before expensive transforms.
5. **`cache=True`** with `SharedArray` installed eliminates HDF5
   decompression after the first epoch.

---

## API reference

### JAXTPCDataset

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_root` | str | *(required)* | Root directory with modality subdirectories |
| `split` | str | `'train'` | Split subdirectory. Pass `''` for flat layouts. |
| `modalities` | tuple | `('step',)` | Subset of `'step'`, `'sensor'`, `'hits'` |
| `dataset_name` | str | `'sim'` | File prefix (matches `sim_step_*.h5`) |
| `volume` | int/None | `None` | Single-volume filter. `None` = all volumes. |
| `labels` | tuple/str/None | `None` | Read `labl`; the per-point labels to write (PILArNet names, or `'all'`) |
| `le_is_stuff` | bool | `True` | Low-energy scatters keep their class, get no fragment/particle instance |
| `momentum` | str | `'start'` | `momentum` key: `|p|` at creation or first visible step (`'first'`) |
| `min_deposits` | int | `0` | Drop events with fewer deposits |
| `include_physics` | bool | `True` | Load dx, theta, phi, charge, photons, t0_us |
| `transform` | list/None | `None` | List of transform dicts (NOT a Compose object) |
| `max_len` | int | `-1` | Cap on event count (`-1` = no cap) |
| `cache` | bool | `False` | Shared-memory caching (requires `SharedArray`) |

### LUCiDDataset

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_root` | str | *(required)* | Root directory with modality subdirectories |
| `split` | str | `''` | Split subdirectory (default: no split) |
| `modalities` | tuple | `('sensor',)` | Subset of `'step'`, `'sensor'`, `'hits'`, `'labl'` |
| `dataset_name` | str | `'wc'` | File prefix (matches `wc_sensor_*.h5`) |
| `min_segments` | int | `0` | Drop events with fewer step segments |
| `include_physics` | bool | `True` | Load direction, beta_start, n_cherenkov |
| `pe_threshold` | float | `0.0` | Drop hits entries with PE ≤ threshold |
| `transform` | list/None | `None` | List of transform dicts |

### MultiModalEventDataset

A detector-agnostic **event-selection wrapper**. It composes one single-source
dataset (`LUCiDDataset` / `JAXTPCDataset` / any nested-output dataset) per
`source`, and adds source mixture and a deterministic train/val/test holdout —
without re-implementing readers or the joint index. Used by the WAND SSL
configs. Cross-modality alignment within a source is the wrapped dataset's
joint index; this layer only owns *which events exist and which split they're in*.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_dataset` | dict | *(required)* | Config of the per-source dataset (e.g. `dict(type='LUCiDDataset', modalities=('sensor',), dataset_name='wc')`) |
| `sources` | list[dict] | *(required)* | Per-source `dict(name=…, label=…, config_id=…)` (subdir + integer event label) |
| `data_root` | str/None | `None` | Root containing the source subdirs |
| `split` | str | `'train'` | `'train'` \| `'val'` \| `'test'` \| `'all'` |
| `holdout` | dict/None | `None` | `dict(seed=…, n_per_config=…)` — deterministic, hash-keyed partition |
| `min_points` | int | `0` | Drop events with fewer points (post-aggregation) |
| `max_events_per_source` | int | `-1` | Cap per source (`-1` = no cap) |
| `mixture` | list/None | `None` | Per-source sampling weights |
| `transform` | list/None | `None` | List of transform dicts |

The deterministic split is keyed on a stable `(config_id, source_event_idx)`
blake2b hash, so it is reproducible across machines and invariant to shard
add/remove/reorder.

### OpticalDataset

PMT light (goop) as per-chunk waveforms. `schema='label'` reads the doraemon
`event_NNN/label_K/{adc,offsets,t0_ns,pmt_id,pe_counts}` layout (group =
interaction); `schema='east_west'` reads `light_output.h5`
(`event_NNN/{east,west}/…`, group = side, files globbed `*.h5`). Single `sensor`
modality. Per-chunk: `pmt_id`, `t0_ns`, `length`, `pe`, `instance` (group); the
packed `adc` (ΣL,) is a second row-space. Key params: `data_root`,
`dataset_name`, `schema`, `decode_digitization=True`, `max_len`. TPC truth
(`tpc_*`) is left on disk.

### PILArNetH5Dataset

Loads PILArNet-M directly from packed HDF5 (no per-event preprocessing). Key
parameters: `data_root`, `split='train'`, `revision='v2'` (`'v1'`/`'v2'`/`'v3'`),
`energy_threshold=0.0`, `min_points=1024`, `max_len=-1`,
`remove_low_energy_scatters=False`, and event-overlay options
(`overlay_n_events`, `overlay_prob`, `overlay_allow_repeats`). 5 semantic
classes (shower/track/michel/delta/led).

### Config-driven construction

```python
from pimm_data import build_dataset
ds = build_dataset(dict(type='JAXTPCDataset', data_root='...', modalities=('step',)))
```

---

## Layout

```
src/pimm_data/
    jaxtpc.py          JAXTPCDataset
    lucid.py           LUCiDDataset
    optical.py         OpticalDataset (PMT light, per-chunk waveforms; label/east_west)
    multimodal.py      MultiModalEventDataset (source-mixture + holdout wrapper)
    pilarnet.py        PILArNetH5Dataset (PILArNet-M h5)
    readers/           Per-modality HDF5 readers (incl. optical_sensor)
    _dataset_base.py   ShardEventDataset base + modality validation
    _joint_index.py    build_joint_index — cross-modality event alignment
    _roles.py          per-part roles + role-driven batching primitives
    transform.py       Compose, TRANSFORMS, Collect, Apply, MultiCrop, ...
    detector_transforms.py  Apply, MultiCrop, AggregateSensorHits,
                            SetupGraph/BuildNexus/Align, PDGToSemantic, RemapSegment
    batch_ops.py       to_batched_coords, split_event (boundary helpers)
    collate.py         collate_fn, inseg_collate_fn, mix_pairs
    loader.py          EventLoader (DataLoader + the tail after Collect)
    builder.py         DATASETS, build_dataset
    utils/pdg.py       pdg_to_semantic(pdg, scheme)
configs/               campaign data-loader recipes (see docs/CAMPAIGN.md)
```

## Tests

```bash
pytest                              # full suite, synthetic fixtures (~0.5 s)
pytest tests/test_lucid.py          # one file
pytest -k labl                      # filter by name
```

### Real datasets

Tests default to synthetic v3 fixtures generated by
`pimm_data.testing`. To run the same tests against real data,
point the matching `*_DATA_ROOT` env var at the dataset root:

```bash
LUCID_DATA_ROOT=/sdf/data/neutrino/cjesus/DORAEMON/OLD/SK_like/config_000001 pytest
JAXTPC_DATA_ROOT=/path/to/jaxtpc                                              pytest
JAXTPC_PIXEL_DATA_ROOT=/path/to/jaxtpc_pixel pytest tests/test_jaxtpc_pixel.py
```

The conftest validates that the override directory has the v3 layout
(`step/`, `sensor/`, `hits/`, `labl/` subdirs); incomplete overrides
skip with a descriptive reason.

Tests that only make sense on real production data carry
`@pytest.mark.real_data_only` — they auto-skip when no `*_DATA_ROOT` is set.

### LUCiD WAND dataset

The canonical real LUCiD-format dataset at SLAC is the WAND SK-like sample —
18 configs in native v3 layout under:

```
/sdf/data/neutrino/cjesus/DORAEMON/OLD/SK_like/
  config_000001/  single mu-
  config_000002/  single pi+
  config_000003/  single e-
  ...
  config_000013/  GENIE numu
  config_000018/  pile-up 2 × bomb
```

Each `config_NNNNNN/` is a v3 root with `sensor/`, `hits/`, `labl/`, and the 3D
segments under **`edep/`** (`wc_edep_*.h5`). The LUCiD `sensor`/`hits` recipes
point `LUCID_DATA_ROOT` straight at one config. The LUCiD **`step`** modality
globs `step/wc_step_*`, which WAND does not ship — run
`scripts/make_wand_step_mirror.sh` to build a symlink view exposing `edep` as
`step/wc_step_*` (`/sdf/data/neutrino/omara/wand_sk_like_step/config_NNNNNN`),
then point `step`/recon recipes there. (Per-config shard counts are regenerated
over time, so they are not pinned here.)

A parallel sweep across all 18 configs is available:

```bash
bash tests/run_wand_sweep.sh        # default 6-way parallel, ~75 s wall
bash tests/run_wand_sweep.sh 12     # 12-way parallel
```

Typical runtimes (Sherlock-class shared FS, warm cache):

| Scope                                    | Wall time |
|------------------------------------------|-----------|
| Synthetic suite (`pytest`)               | ~0.5 s    |
| Single small config (e.g. `config_000006`) | ~7 s    |
| Single large config (e.g. `config_000003`)             | ~40 s |
| Full 18-config sweep (sequential)        | ~10 min   |
| Full 18-config sweep (parallel -j6)      | ~75 s     |

The largest single-config cost (`test_step_min_segments_filter`) walks
every event's `n_segments` attribute to build a filter index — that
scales with total events, not with file count, and dominates large
configs.
