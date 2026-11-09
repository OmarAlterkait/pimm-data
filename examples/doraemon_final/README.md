# Loading the doraemon final layout

The layout the models read, as built for `doraemon_2026_1M_lartpc`. The 10k-event sample,
10 tasks with no empty volumes, is at
`/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/final_10k`:

```
<data_root>/jaxtpc_{wire,pixel}/<bucket>/jaxtpc_{ro}_<task>/
    step/    in-window deposits of this readout, rows in position order;
             deposit_to_group (volume group ids, numbered by position), qs_fractions
    hits/    per plane; group_ids numbered per plane (0..G-1); root attr group_id_space='plane'
    sensor/  JAXTPC's readout, unchanged
    labl/    pysupera's labels: group_to_track, group_fragment; plane_<P>/group (plane id ->
             volume group); pixel on_sensor / overrides; tables track, particle, interaction
```

Correspondence is kept out of the readout files:
- Planes share no ids, and hits and step share no ids.
- Rows and blocks are in position order, not Geant4-track order.
- Only labl links them.

So:
- `hits.group` exists only when labl is loaded. Without labl, hits carry `plane_group`
  (which entries form one block on a plane) and reference nothing.
- `step.group` comes from step's own `deposit_to_group`.

## Run it (S3DF)

`site_doraemon.py` hard-codes the sample's location
(`/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/final_10k`; override with
`DORAEMON_FINAL_ROOT`). Its `dataset(readout, modalities, labels, split)` returns a
`JAXTPCDataset`, split by task (last task val).

```bash
bash examples/doraemon_final/run_examples.sh wire      # or pixel: every example below
```

| example | modalities | labels | shows |
|---|---|---|---|
| `step_labl.py` | step | `('pysupera',)` | 3D panoptic on deposits: every deposit voxelized at 3 mm (`Voxelize`), then Collect |
| `hits_labl.py` | hits | `('pysupera',)` | per-plane 2D panoptic on hit entries |
| `step_hits_labl.py` | step + hits | `'all'` | correspondence: both reference the same `group` rows; `particle` / `interaction` tables as object targets |
| `no_labl.py` | step + hits + sensor | none | self-supervised: `step.group`, hits only `plane_group` |

Each one prints one event's parts (keys, shapes, references) and a collated batch.
Single example: `PYTHONPATH=<repo>/src:examples/doraemon_final python examples/doraemon_final/hits_labl.py pixel`.

Training recipe (data half): `configs/jaxtpc/final_panoptic.py`. It is 3D panoptic on step
with pysupera's target. Set `JAXTPC_DATA_ROOT=<root>/jaxtpc_wire`, and for pixel also
`JAXTPC_DATASET_NAME=sim_pixel`.

## Labels

pysupera is the reference: every label is pysupera's own answer (`tests/test_pysupera_golden.py`
checks it on real data).

| `labels=` | keys |
|---|---|
| `'all'` | the five per-point labels: `segment_motif`, `segment_pid`, `instance_fragment`, `instance_particle`, `instance_interaction` |
| `'pysupera'` | `segment_motif` + `instance_particle`: pysupera's panoptic target |
| `'fragment'` | `segment_motif` + `instance_fragment` |
| `'particle'` | `segment_pid` + `instance_particle` |
| `'interaction'` | `instance_interaction` |
| any keys | e.g. `('segment_motif',)`, `('fragment', 'interaction')` |

- **Classes.** `segment_motif`: shower, track, michel, delta, le. `segment_pid`: photon,
  electron, muon, pion, proton, le.
- **Instance ids.** They are rows of `particle` (`interaction` for `instance_interaction`).
  The same object has the same id in step, in hits, on every plane, and in every volume.
- **LE (low-energy) points.** They stay as input with class le. With `le_is_stuff` (the
  default) they get no fragment / particle id; `le_is_stuff=False` keeps their host's
  (pysupera's `include_le=True`).
- **Unlabelled points (-1).** Deposits of groups pysupera does not own (under threshold);
  pixel hits that are not on a sensor pixel (about 13%, to be removed later).
- **Per-object properties.** PDG, primary, momentum, start/end and first/last points,
  ancestry and vertex are columns of the `particle` / `interaction` tables. Reach them
  through a point's instance id. `segment_pdg`, `is_primary`, `momentum` and `vertex` can
  also be broadcast to points by naming them, for losses that read them per point.

## What each combination gives

| modalities | without labl | with labl |
|---|---|---|
| step | deposits, `step.group` (deposit groups) | + labels; `group` -> track, fragment |
| hits | entries, `plane_group` only | + labels; `hits.group` through the plane maps |
| step + hits | both, no link between them | + labels; step and hits reference the same `group` rows (deposit <-> hit <-> plane correspondence, `step.qs` to split charge) |
| sensor | readout | (sensor carries no labels) |

## Checks
- `tests/test_supera_labl.py::test_plane_local_layout_loads_the_same`: plane-local hits give
  the labels the volume-numbered layout gives (synthetic).
- `tests/test_pysupera_golden.py`: every label equals pysupera's own (real data; set
  `PYSUPERA_GOLDEN_SUPERA`, `PYSUPERA_GOLDEN_JAXTPC`, `PYSUPERA_GOLDEN_RUN`).
- The building and verification scripts for the layout live with the data
  (`final_10k/_build/`: `build_final_v2.py`, `verify_final_v2.py`, per-task results in `checks/`). Every array maps back exactly to
  the produced files, and pimm-data's labels are compared deposit by deposit and hit by hit.
