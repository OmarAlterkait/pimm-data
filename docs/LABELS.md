# The `labl` file: what to extract from pysupera

Status: implemented on branch `doraemon-labels` (2026-10-27). Writer:
`pimm_data/supera_labl.py`. Reader: `JAXTPCLablReader` (`kind == 'supera'`).
Labels: `pimm_data/_supera_labels.py`. Measured on
`doraemon_2026_1M_lartpc` (JAXTPC f0d2ed3, pysupera dab92d8 / format 4.1.0), task 0,
wire and pixel.

## What a label attaches to

- **The sensor carries no labels.** It is the summed, thresholded readout; it does
  not record which particle produced which charge, so there is no key in it to join a
  label to. Labels reach the readout only through `hits`, which is the sensor split
  by group. `('sensor', 'labl')` stays rejected.
- **The JAXTPC group is the unit of truth.** Both released point clouds reach labels
  through it:
  - step deposit → `deposit_to_group` → group
  - hit entry → CSR `group_ids` → group

  pysupera's answer, per group, is the fragment that owns it (`groups/fragment_id`).
  Everything else is a property of that fragment, its instance or its interaction.
- **So `labl` is two things:** a group → fragment map, and the tables it points into.
  It holds no per-hit or per-deposit label arrays, and computes no aggregates.

Grouping is identical between the wire and pixel runs: step, `deposit_to_group` and
`group_to_track` are byte-identical. Ownership differs by readout, because pysupera
partitions wire on deposits and pixel on hits. So there is one `labl` per readout.

## What lives where

The rule: **a point file carries nothing above the group.** Step carries
deposit → group, and hits carries hit entry → group. Everything from the group
up lives in `labl`: group → track, group → fragment, the particle and interaction
tables, and the references between them.

On disk today `deposit_to_group` and `group_to_track` sit in the JAXTPC hits file.
`labl` copies both (`deposit_to_group`, `group_track`), so step+labl and
hits+labl are each complete.

- `deposit_to_group` belongs in step by the rule. When JAXTPC writes it there,
  `labl` drops its copy.
- `group_to_track` belongs in `labl`; hits keeps it only for older data.

Measured (10 events per readout, 1M task 0): every visible Geant4 track lies in
exactly one fragment and one instance (0 exceptions in about 80k tracks). So
group → track → instance is a chain of functions too. 26% (wire) and 39% (pixel)
of visible tracks have no particle row, because they are merged shower members.

## How pimm-data loads it

`JAXTPCDataset` with a supera `labl` emits the event as parts linked by
`('ref', part)` columns (`_roles.py`). Collate shifts each reference by its
target part's running row count, and `split_event` undoes the shift.

```
step.group, hits.group            -> group          (only this, per point)
group: volume, track, fragment    fragment -> particle
particle: labl columns            fragment/instance/fragment_instance/parent/
                                  ancestor/... -> particle, interaction -> interaction
interaction: labl columns
```

Labels are walks along these references (`_resolve.follow`). The PILArNet keys
are one preset (`_resolve.PRESETS['pilarnet']`). Class vocabularies are derived
table columns (`particle.motif`, `pid`, `is_primary`, `momentum_*`,
`interaction.vertex`), computed when loading, never stored. Point subsamples
slice the references with their points and never touch the tables.

## Verified facts that set the key list

| fact | consequence |
|---|---|
| Wire `hit_labels` (fragment, instance, is_le) equal the group-derived labels on every hit | `hit_labels` is redundant |
| Pixel `hit_labels` differ only by `on_sensor = false` (13–15% of hits) and about 10 hits/event in groups that straddle a fragment boundary, which `groups/` resolves by majority | `on_sensor` is a hits-vs-sensor fact, not a label; the group label is the definition |
| `groups/is_le` == (owner's `frag_sem_type` == kLEScatter) on every owned group; 0 on unowned | `is_le` is redundant (true since 4.0: LE is per fragment) |
| Every owner is a fragment row; every fragment row's `frag_inst_id` resolves to an instance row; every instance row is also a fragment row | group → fragment → instance needs no point ranges |
| `parent_pdg` == `pdg[parent_id]` on every row | `parent_pdg` is redundant |
| All id references (`parent_id`, `ancestor_id`, `frag_parent_id`, `inst_parent_id`, `frag_inst_id`, `groups/fragment_id`) resolve to rows of the same event | they can be stored as row indices |
| Instance rows with `geant4_parent_trackid == -1` are exactly those with `parent_id == id` | "primary" has one definition |
| Rows that are neither fragment nor instance are almost all `kDelta` delta rays merged into their track (plus a few ancestry links) | they stay: they are the delta-ray labels and the hierarchy |

## Contents

Identity and layout follow JAXTPC format 2: one `labl` file per JAXTPC task file,
same `entry_NNNNNN` entries in the same order, the same `event_id` / `job_id` /
`job_event` attributes and `/dprod/events` table, and a `/config` carrying
`format_version`, readout type and the pysupera provenance (commit, config hashes).

```
entry_NNNNNN/
  volume_{v}/
    group_fragment    (G_v,) int32   particle row owning group g (1-based groups;
                                     slot 0 and unowned groups = -1)
                                     <- groups/fragment_id, split by volume_offsets
    deposit_to_group  (N_v,) int32   copy from hits; the step -> group bridge
    group_track       (G_v,) int32   copy of hits group_to_track (Geant4 track)
  particle/           (P rows)
  interaction/        (I rows)
```

`deposit_to_group` is the one non-pysupera array. Step+labl must work without hits,
and the step file does not carry it. The alternative is for JAXTPC to write it into
step, since grouping is a step property and readout-independent; that change is not
ours to make.

### `particle/` — kept columns

All id-valued columns are converted to row indices within the event (`-1` = none).
That is a reindex, not a computation; pysupera ids are never needed downstream.

| column | from pysupera | why |
|---|---|---|
| `fragment` | `frag_id` | own row if this row heads a fragment, else -1 |
| `instance` | `inst_id` | own row if this row heads an instance, else -1 |
| `fragment_instance` | `frag_inst_id` | instance labels: group → fragment → instance |
| `fragment_parent`, `instance_parent` | `frag_parent_id`, `inst_parent_id` | fragment and instance hierarchy |
| `parent`, `ancestor` | `parent_id`, `ancestor_id` | particle hierarchy; primary ⇔ `parent == row` |
| `interaction` | `interaction_id` | already a row index into `interaction/` |
| `pdg` | `pdg` | PID |
| `sem_type` | `sem_type` | own class; marks merged delta rays (`kDelta` on a non-fragment row) |
| `fragment_sem_type` | `frag_sem_type` | the semantic class of every point (incl. LE) |
| `instance_sem_type` | `inst_sem_type` | instance class |
| `interaction_type` | `interaction_type` | creation process |
| `geant4_trackid`, `geant4_parent_trackid`, `geant4_parent_pdg` | same | provenance to edep-sim; true parent PDG is not derivable when the parent has no row |
| `mass` | `mass` | KE = E − mass |
| `start_pos/mom`, `end_pos/mom` | same | Geant4 creation and destruction, `[x,y,z,t]` mm/ns and `[px,py,pz,E]` MeV |
| `first_pos/mom`, `last_pos/mom` | same | instance rows: first/last visible step; `last` NaN for showers; NaN for instances with no visible points |

### `interaction/` — kept columns

`x`, `y`, `z`, `time` (vertex), `vertex_id` (link to the edep-sim vertex list),
`energy_sum`, `ke_sum`.

### Dropped, and why

| pysupera | reason |
|---|---|
| `points/*`, every `*pc_start/le_start/end` | voxels; no voxels |
| `hit_labels/*` | derivable from the group map (above); `on_sensor` is not a label |
| `groups/is_le` | derivable from `fragment_sem_type` |
| `parent_pdg` | derivable from `parent` |
| `id`, `inter/interaction_id` | replaced by row position |
| `frag_merge_count`, `inst_merge_count` | pysupera bookkeeping, not truth |
| `inter/part_*`, `inter/pc_*` | point/particle ranges; `particle.interaction` covers it |
| `inter/reaction` | empty in 4.1 |
| `events/input`, `input_event`, `jaxtpc_entry`, `inputs/` | pysupera's own provenance; `labl` is entry-aligned with JAXTPC by construction |
| `issues/` | carried as counts in `/config` attrs only |

## What pimm-data does with it

Everything a model or metric uses is a gather; nothing new is stored:

- `segment_motif` = `fragment_sem_type[group_fragment[g]]`, mapped to the shared
  motif vocabulary (shower, track, michel, delta, le)
- `instance_particle` = `fragment_instance[...]`
- `segment_pid` = `pdg[instance]`, mapped to the PID classes
- `instance_interaction` = `interaction[instance]`
- `momentum` / `vertex` / `is_primary` = `particle` and `interaction` columns
- object-level targets come straight from the tables

These are exactly the keys pimm's evaluators and heads already read for PILArNet.
Deposits and hits in unowned groups are -1 everywhere. That is about 10% of wire and
17% of pixel deposits in 1M, mostly charge outside the readout window.

## Implemented, and checked on real data

On 1M task 0 (20 events per readout), the per-hit labels `JAXTPCDataset` builds from
`labl` + hits were compared with pysupera's own `hit_labels`:

- **wire:** 0 mismatches over 6.46M hits;
- **pixel:** 16 fragment / 2 instance / 4 LE mismatches over 11.9M on-sensor hits,
  all in the majority-resolved straddling groups.

Writing takes about 2 minutes per 1000-event task, and the output is about
0.8 MB/event (wire). Most of that is the float64 kinematics (about 560 KB),
kept because pysupera chose float64 to keep slow-nucleus KE. Storing them as
float32 would halve the file.

Hits carry the JAXTPC group id as `instance_group`, and `instance` becomes the
particle row.

## Still needed from producers

1. **Generator source per interaction** (MPV vertex vs MPR single particle). It is
   not recorded anywhere:
   - edep-sim's `vertex/geant4` has `generator`, `reaction` and `generator_vertex_id`
     fields, but DLPGenerator leaves them empty or unset;
   - pysupera copies `reaction` through `vertex_id`, so `inter/reaction` is empty too.

   It can only be inferred today (checked on 200 events). Vertices are written in
   generator order, Generator 4 (MPR) always has exactly one particle, and MPV
   vertices are at t < 1.6 µs while MPR vertices are spread over 0–1.31 ms. A
   1-particle vertex at t < 1.6 µs is still ambiguous: MPV also makes 1-particle
   vertices (236 of 1790), and an MPR lands that early about once per 100 events.

   The fix is upstream: DLPGenerator fills `generator` (e.g. `Generator1..4`), and
   pysupera then carries it next to `reaction`. With that, `interaction/` gains
   `generator`.
2. **A doraemon `labl` stage.** It does not exist in the 1M campaign.
3. **Unchanged from before:** pixel window 0/0, `t0_us` float16, no `in_window` flag.

## Open

- Whether low-energy-scatter instances (about 86% of instance rows) count as instances
  or as stuff in panoptic metrics; PILArNet treats them as stuff.
- Which momentum is the default `momentum` gather: `|start_mom|` (creation) or
  `|first_mom|` (first visible step). Both are stored.
