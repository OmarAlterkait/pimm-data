# JAXTPC ↔ pimm-data on-disk HDF5 schema

The contract between the **writer** (JAXTPC `production/save.py` + `make_labl.py`,
dispatched by `run_batch.py`) and the **reader** (pimm-data `readers/*`). It is
documented here so external consumers can write their own readers against a
stable format. Treated as stable: if the writer changes, update this doc. There
is no `schema_version` attribute (deliberate — see ADR §5).

> **Codec.** All sim datasets are compressed with the run's `--codec` (default
> `blosc-zstd`). **Reading any blosc/zstd/lz4 output requires `import hdf5plugin`**
> (pimm-data registers it at import). A missing backend raises (no silent gzip
> fallback). `make_labl` writes labl datasets with hardcoded `gzip` (no plugin
> needed).

## Cross-cutting conventions

- **Files / sharding.** One batch shard per file: `{dataset}_{modality}_{NNNN}.h5`,
  `modality ∈ {sensor, step, hits, labl}`, `NNNN` = 4-digit shard index. Reader
  glob tries `{root}/{split}/{name}_{modality}_*.h5` then flat `{root}/{name}_{modality}_*.h5`.
- **Events.** Each event is a top-level group `event_{local_idx:03d}` (`:03d` is a
  minimum width; events ≥1000 widen). The numeric suffix is the **file-local**
  0-based index. There is **no index dataset/offset table** — present events are
  found by scanning group names (gap-tolerant). Global index → `(file, event_num)`
  via cumulative-length `searchsorted`.
- **Identity.** Every event carries `evt.attrs['source_event_idx']` (global id) and
  optional `evt.attrs['event_id']`. *Latent gap (informational): pimm-data's identity
  resolution prefers a `config/source_event_idx` dataset that the production writer
  never emits; it falls back to `config.attrs['global_event_offset'] + event_num`
  (which the writer does emit). The per-event `source_event_idx` attr is written but
  not consumed by the reader.*
- **Volumes.** Deposits are split into N volumes by x-position; each event has
  `volume_{v}` subgroups (empty volumes skipped). `n_volumes` in `config.attrs` and
  per-event `evt.attrs['n_volumes']`.
- **Coordinates.** Sim runs in volume-*local* frame; step positions are transformed
  **local→global at write time** (`x_global = x_anode_cm*10 − drift_dir*x_local`;
  y/z += yz_center*10). Sensor/hits indices (wire/pixel/time) are detector index
  space — no transform.
- **Group ids are 1-based** in hits/ files (`group_to_track[0]` unused).
- **Units (by design):** wire hits = ENC (electrons); pixel hits = ADC.

---

## 1. SENSOR — `{dataset}_sensor_{NNNN}.h5`
Sparse, thresholded, delta-encoded raw readout.

**`/config` attrs:** `dataset_name, file_index, source_file, n_events,
global_event_offset, num_time_steps, time_step_us, pre_window_us, post_window_us,
electrons_per_adc, velocity_cm_us, lifetime_us, recombination_model,
include_{intrinsic_noise,coherent_noise,electronics,digitize}, threshold_adc,
n_volumes, readout_type` (`'wire'`/`'pixel'`); `n_bits` only if digitized; optional
provenance (`production_version, run_id, batch_timestamp, git_*`).
**`/config` datasets:** `num_wires` int32 `(n_volumes, max_planes)`; `volume_ranges`
float32 `(n_volumes, 3, 2)` mm; `pedestals` int32 `(n_volumes, max_planes)` (digitized only).

**Per event** `event_{i:03d}`: attrs `source_event_idx`, `event_id?`, `n_volumes`,
`n_vol{v}` (per-volume deposit count). Subgroups `volume_{v}/{plane_label}` (wire =
`U/V/Y`, pixel = configured label; empty planes omitted).

- **Wire plane** (entries sorted `lexsort((time, wire))`): `delta_wire` int16
  (`diff(wire, prepend=wire[0])`), `delta_time` int16, `values` float32 (or uint16 if
  digitized: `round(value+pedestal).clip(0,65535)`; decode `signal = value − pedestal`).
  Attrs `wire_start`, `time_start`, `n_pixels`, `pedestal?`. Decode wire k:
  `wire = wire_start + cumsum(delta_wire)[k]` (same for time).
- **Pixel plane** (`lexsort((time, pz, py))`): `delta_py`, `delta_pz`, `delta_time`
  int16; `values` float32 (pixel never digitized). Attrs `py_start, pz_start,
  time_start, n_pixels`.

**Reader output:** wire → `sensor.{vol}_{plane}.{wire,time,value}`; pixel →
`sensor.{vol}_{plane}.{py,pz,time,value}`.

---

## 2. STEP — `{dataset}_step_{NNNN}.h5`
Pure-physics 3D truth deposits (no track/group/instance — those live in hits/labl).

**`/config`:** attrs `dataset_name, file_index, source_file, n_events,
global_event_offset, group_size, gap_threshold_mm, n_volumes, readout_type` +
provenance; datasets `num_wires`, `volume_ranges` (as sensor).

**Per event / `volume_{v}`** (attr `n_actual` = N; empty volumes set only `n_actual`).
Datasets, length N, **row order = canonical deposit index** that hits/labl join against:

| Dataset | dtype | meaning / decode |
|---|---|---|
| `positions` | uint16 `(N,3)` | voxelized; `pos_mm = positions*pos_step_mm + origin` (attrs `pos_origin_{x,y,z}`, `pos_step_mm`). **Global** frame. |
| `de` | float16 `(N,)` | energy deposit (MeV) |
| `dx` | float16 | step length (cm) |
| `theta`,`phi` | float16 | track angles |
| `t0_us` | float16 | deposit time (µs) |
| `charge` | float32 | recombined ionization electrons |
| `photons` | float32 | scintillation photons |

**Reader output:** concatenated cloud — `coord (N,3)`, `energy (N,1)`, `volume_id`,
optional `dx/theta/phi/t0_us/charge/photons`. Supports `volume=` filter; legacy flat
layout accepted.

---

## 3. HITS — `{dataset}_hits_{NNNN}.h5`
Per-particle charge attribution at sensor elements (group↔element correspondence) +
group machinery.

**`/config`:** attrs `dataset_name, file_index, source_file, n_events,
global_event_offset, group_size, gap_threshold_mm, num_time_steps, pre/post_window_us,
n_volumes, readout_type` + provenance; datasets `num_wires`, `volume_ranges`.
`num_time_steps` is required to invert the wire CSR packed key.

**Per event / `volume_{v}`** (attr `n_actual` = N; event attr `threshold`):
- `deposit_to_group` int32 `(N,)` — **row-aligned with step**, per-deposit group id (1-based).
- `qs_fractions` float16 `(N,)` — each deposit's fraction of its group's recombined charge.
- `group_to_track` int32 `(G,)` (if available) — group id → Geant4 track_id (1-based; attr `n_groups`).

**Per plane** `volume_{v}/{label}` (attrs `n_groups_plane`, `n_entries`) — CSR correspondence:

*Wire* (`packed_key = wire*num_time_steps + time`; per-group peak-charge center):
`group_ids` int32 `(G,)`; `group_sizes` uint8 `(G,)` (**CSR indptr = cumsum−sizes**);
`center_wires`/`center_times` int16 `(G,)`; `peak_charges` float32 `(G,)`;
`delta_wires`/`delta_times` int8 `(E,)`; `charges_u16` uint16 `(E,)`.
Decode group g (size sz, start s): `wire = center_wires[g] + delta_wires[s:s+sz]`;
`time = center_times[g] + delta_times[...]`; `charge = peak_charges[g]*charges_u16[...]/65535` (≥0).

*Pixel* (`spatial_key = py*num_pz + pz`, `num_pz = pixel_shape[1]`; max-|charge| center, **signed**):
`group_ids`, `group_sizes`, `center_py`/`center_pz`/`center_times` int16, `peak_charges`
float32 (signed), `delta_py`/`delta_pz`/`delta_times` int8, `charges_i16` int16
(`round(charge/abs(peak)*32767)`). Decode: `charge = abs(peak_charges[g])*charges_i16/32767`
(**abs** on peak — the i16 already carries the sign). Readers also accept legacy
`charges_u16` (scale 65535).

**Reader output:** per plane `hits.{vol}_{label}.{wire|py,pz,time,group_id,charge}`;
per volume `group_to_track_v{N}`, `deposit_to_group_v{N}`, `qs_fractions_v{N}`.

---

## 4. LABL — `{dataset}_labl_{NNNN}.h5`
Per-deposit→track FK + per-track dimension table (written separately by `make_labl.py`;
a temporary stand-in). All datasets gzip + int32.

**`/config` attrs:** `dataset_name, file_index, n_events, n_volumes, label_names`
(`['track_pdg','track_cluster','track_interaction','track_ancestor']`), `source`,
`generator`, + hits config passthrough (`source_file, global_event_offset, group_size,
gap_threshold_mm, git_*`). No `num_wires`/`volume_ranges`/`readout_type`.

**Per event / `volume_{v}`** (qualifies iff it has `track_ids`):

| Dataset | shape | meaning |
|---|---|---|
| `deposit_to_track` | `(N,)` | per-deposit FK → track_id; **row-aligned with step**; `−1` if group out of range |
| `track_ids` | `(T,)` | unique track ids (PK) |
| `track_pdg` | `(T,)` | raw PDG (`−1` if missing) |
| `track_interaction` | `(T,)` | raw interaction id |
| `track_cluster` | `(T,)` | **dummy = track_id** (placeholder) |
| `track_ancestor` | `(T,)` | raw ancestor/root track id |

`deposit_to_track = group_to_track[deposit_to_group]`, so labl derives from hits.

**Reader output:** `labl_v{N}_track_ids` + `labl_v{N}_{col}` for each other dataset.

---

## 5. COEFF — `{dataset}_coeff_{NNNN}.h5` / `{dataset}_coeff_clean_{NNNN}.h5`

Wavelet-coefficient corpus produced by helix DSP (schema of record:
`helix/COEFF_CORPUS_DESIGN.md` §3; reference codec `helix/core/coeff_io.py`).
Unlike the modalities above this is **flat columnar, not per-event groups**: one
shard-wide array per column plus an `event_offset` index, so an event is one
contiguous slice. Read by `CoeffTPCReader` / `CoeffTPCDataset` with standalone
h5py — pimm-data never imports helix at read time.

**`/config`** attrs: `n_events, dataset_name, file_index, global_event_offset,
readout_type, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, sigma_norm,
basis_digest, removal_json, threshold_json, has_coords, noise_json`.
Datasets: `band_lengths (n_bands,) int32`, `gids (G,) int32`,
`n_wires (G,) int32`, `norm_sigma (G, n_bands) float32`.

| Dataset | shape | meaning |
|---|---|---|
| `coord/band` | `(M,)` uint8 | band index into `[cA, cD_L, …, cD_1]` |
| `coord/plane_gid` | `(M,)` **int32** | canonical plane id `v*3 + {U,V,Y}` (int32: gid can exceed 255) |
| `coord/wire` | `(M,)` int32 | signal index within the plane |
| `coord/tau` | `(M,)` int32 | within-band coefficient index |
| `coord/event_offset` | `(n_events+1,)` int64 | event boundaries; slice `[off[i]:off[i+1]]` |
| `coord/coord_digest` | `(n_events,)` uint64 | blake2b of the event's 4 coord columns |
| `coord/sigma_threshold` | `(n_events, G, n_bands)` f32 | per-event threshold σ |
| `value` | `(M,)` float32 | **RAW** coefficient (normalized at tokenize, not on disk) |
| `ident/{run,source_file,event}` | `(n_events,)` | per-event identity — the join key |

**Two invariants that are load-bearing:**

- `norm_sigma` and `sigma_threshold` rows are indexed by **POSITION in `gids`**,
  never by gid value. With a dead plane the two differ and `norm_sigma[gid]`
  silently selects another plane's σ. Resolve through `helix.tokenize.gid_rows`.
- `norm_sigma` must be **identical across every shard of a corpus** (build with
  `--norm-sigma`). `CoeffTPCReader` verifies this and refuses shards that
  disagree on it, on `gids`/`n_wires`/`band_lengths`, or on `sigma_norm`.

**`coeff_clean` is values-only** (`has_coords=False`): the clean target is
co-supported with the noisy input — same `(band, plane_gid, wire, tau)` by
construction — so it stores no coordinate columns and inherits them from the
paired `coeff` shard. `coord_digest` is written to both and compared on join, so
a mispairing raises rather than misaligning every target row. It therefore cannot
be read alone: `modalities=('coeff_clean',)` is rejected.

**Reader output** (`CoeffTPCDataset.get_data`): a `coeff` part with
`band, plane_gid, wire, tau, value (M,1)` plus `_meta`
(`gids, n_wires, band_lengths, norm_sigma, sigma_norm`) so a worker-side
tokenizer is self-sufficient. Tokenization itself lives in `helix.tokenize`.

---

## In-memory output-dict contract (companion)
Datasets emit a **nested** dict (not bare `coord`); consumers pick a stream via
`ApplyToStream`/`Collect(stream=...)`. Top-level: `name`, `split`, plus loaded
modalities `step`/`hits`/`sensor`/`labl` (+ `bridges` when hits loaded). Per-modality
sub-dict keys are documented in the dataset module docstrings
(`jaxtpc.py`, `lucid.py`).
