# The `labl` file: what to extract from pysupera

Status: implemented on branch `doraemon-labels` (2026-10-27). Writer:
`pimm_data/supera_labl.py`. Reader: `JAXTPCLablReader`. Labels:
`pimm_data/_resolve.py`, through `JAXTPCDataset(labels=...)`. Measured on
`doraemon_2026_1M_lartpc` (JAXTPC f0d2ed3, pysupera dab92d8 / format 4.1.0), task 0,
wire and pixel.

## What a label attaches to

- **The sensor carries no labels.** It is the summed, thresholded readout; it does
  not record which particle produced which charge, so there is no key in it to join a
  label to. Labels reach the readout only through `hits`, which is the sensor split
  by group. `labels=` with only `sensor` is rejected.
- **The JAXTPC group is the unit of truth.** Both released point clouds reach labels
  through it:
  - step deposit → `deposit_to_group` → group
  - hit entry → CSR `group_ids` → group

  pysupera's answer, per group, is the fragment that owns it (`groups/fragment_id`).
  Everything else is a property of that fragment, its instance or its interaction.
- **So `labl` is the hierarchy, stored once as maps:**
  group → track → fragment → instance → interaction, plus the tables they point into.
  It holds no per-hit or per-deposit label arrays, and computes no aggregates.

Grouping is identical between the wire and pixel runs: step, `deposit_to_group` and
`group_to_track` are byte-identical. Ownership differs by readout, because pysupera
partitions wire on deposits and pixel on hits. So there is one `labl` per readout.

## Three regimes, each self-sufficient

| given | can train | through |
|---|---|---|
| step / hits / sensor + `labl` | supervised | `step.group` / `hits.group` → `labl` |
| step / hits / sensor, no `labl` | self-supervised | the point files alone |
| step + hits, no `labl` | matching | `step.group` == `hits.group`, weighted by `step.qs` |

The group is the correspondence key that step and hits share. It is not a
label, so pimm-data exposes it (`step.group`, `hits.group` → a `group` part)
whenever hits is loaded or `labels=` is set. Every truth above the group
(track id, fragment, instance, PDG, kinematics, interaction) is in `labl` and
nowhere else.

## What lives where

The rule: **a point file carries nothing above the group.** Step carries
deposit → group, and hits carries hit entry → group. Everything from the group
up lives in `labl`: group → track, track → fragment, the particle and interaction
tables, and the references between them.

On disk today `deposit_to_group` and `group_to_track` sit in the JAXTPC hits file.
`labl` copies both (`deposit_to_group`, `group_track`), so step+labl and
hits+labl are each complete.

- `deposit_to_group` belongs in step by the rule. When JAXTPC writes it there,
  `labl` drops its copy.
- `group_to_track` belongs in `labl`; hits keeps it only for older data.

`qs_fractions` is not a hits quantity. It is a deposit's share of its group's
recombined step charge (JAXTPC `compute_qs_fractions`), and it equals
`charge / group-sum(charge)` from the step file to float16 precision. It is the
weight that splits a hit's charge over its group's deposits. It is derivable,
so pimm-data computes it as `step.qs`, and no file needs to carry it.

**Track information is implicit in today's point files.** It shows in their
ordering (1M task 0, verified):
- JAXTPC numbers groups in track-id order, so each track's groups are one
  contiguous run of ids;
- step rows follow edep-sim's track-sorted step table, so each track's deposits
  are one contiguous run and `deposit_to_group` is non-decreasing.

The partition into tracks (not the ids or PDGs) can therefore be read from
step or hits without `labl`. Explicitly, hits also carries `group_to_track`.
Why it matters: without `labl`, the group correspondence should be the only
truth a model can use. Ordering gives away more:
- runs of consecutive group ids are whole tracks;
- step rows trace each track's trajectory in order.

A self-supervised or matching model, or a baseline that uses point order, could
then recover track membership by index arithmetic. Its "label-free" result
would not be. Keeping track information in `labl` alone means a release-time step:
- drop `group_to_track` from hits;
- renumber groups and permute step rows by one random per-event permutation,
  applied consistently to step, hits' group ids and `labl`.

This has to come after pysupera, which relies on the step row order.

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

`JAXTPCDataset(labels=...)` reads `labl` and writes two keys on each step / hits
point: `segment`, the class (`'motif'`, `'pid'` or `'pdg'`), and `instance`, the id
at the level `instance=` names (`'particle'`, the default; `'fragment'`; `'interaction'`). Any other key from `_resolve.PRESETS` is
opt-in through `label_keys=`, because each one is a lookup over every point: on a
1M wire event, all eight cost 0.98 s of `get_data`, the two defaults 0.26 s.
Object-level targets (momentum, vertex, primary) are better read from the
`particle` / `interaction` tables through `group` than broadcast to points.

Labels are walks along these references (`_resolve.follow`). The PILArNet keys
are one preset (`_resolve.PRESETS['pilarnet']`). Class vocabularies are derived
table columns (`particle.motif`, `pid`, `is_primary`, `momentum_*`,
`interaction.vertex`), computed when loading, never stored. Point subsamples
slice the references with their points and never touch the tables.

Without `labl`, the `group` part still exists whenever hits is loaded (only
`volume`), with `step.group`, `step.qs` and `hits.group`. That is the step <-> hits
correspondence: hits give each readout entry's group and `deposit_to_group`, so the
group is the shared key, label-free. Nothing above the group appears. There is no
`bridges` part; its arrays were duplicates of these references.

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
    group_track       (G_v,) int32   Geant4 track id of group g (copy of hits
                                     group_to_track)
    deposit_to_group  (N_v,) int32   copy from hits; the step -> group bridge
    deposit_in_window (N_v,) bool    the deposit's charge arrives inside the
                                     readout window (needs --step --sensor)
  track/      (T rows)  each visible Geant4 track: id (sorted), fragment (the
                        particle row owning it) <- owned groups of groups/fragment_id
  particle/   (P rows)  pysupera particles: fragments/instances, their links,
                        Geant4 ids, kinematics, interaction ref
  interaction/(I rows)  pysupera inter/: one row per interaction it kept
```

`labl` is what pysupera outputs, curated, and nothing more. It does not reach
back into edep-sim: tracks and vertices the readout never saw are not part of
what we label, and anything pysupera adds later is picked up by extending the
column lists. Interactions are separated by row (`particle.interaction`); which
generator made each one is pysupera's to add, and `interaction/` gains it then.

### Why track → fragment, and not group → fragment or track → particle

Measured on 1M task 0 (20 events per readout):

| map | exact? |
|---|---|
| group → track | yes, by construction (JAXTPC) |
| track → fragment | all but ~1 track in 10⁶: over 200 events, 2 (wire) / 3 (pixel) tracks of ~2M are split by pysupera between two fragments (7 / 20 groups). `track.fragment` is the owner of most of the track's groups; the others are listed per volume as `override_group` / `override_fragment`, so labels stay exact |
| fragment → instance → interaction | yes, in the particle table (read the instance's `interaction`: 19 pixel LE fragment rows disagree with their instance) |
| track → particle row (`geant4_trackid`) | no: 26% / 39% of visible tracks have no row (merged into another fragment, mostly LE and shower e/γ); 6% have a row that is not their owner (delta rays merged into the parent); 9 pixel tracks have two rows |

So a group's fragment is `track.fragment[group.track]`, about 10k rows per event
instead of about 56k groups. Hit labels are unchanged by it (0 differences on both
readouts). Step labels differ only in groups pysupera left unowned while their track
is owned elsewhere: 3.8% (wire) / 5.1% (pixel) of deposits, 99% of them out of the
window (`deposit_in_window` masks those, and JAXTPC will drop them). The remaining
0.05% are in-window groups under threshold, now labelled with their track's fragment.

### Producing it

    python -m pimm_data.supera_labl campaign DATA_ROOT OUT_ROOT --workers N [--shard I/N]

runs over every pysupera file of a campaign and writes
`OUT_ROOT/jaxtpc_{ro}/{bucket}/jaxtpc_{ro}_{task}/labl/*_labl_*.h5`, the JAXTPC
tree with only `labl/` in it. Finished files are skipped (output appears only when
complete), failures are reported without stopping. Measured on milano, 16 workers,
17 real 1000-event files: about 65 s per file per core, no slowdown from running 16
at once, under 1 GB per worker, 0.82 GB per file. The whole 1M campaign (1960
files) is about 35 core-hours and 1.6 TB. Once pysupera runs this itself, the stage
goes away.

Size, 1M task 0: about 0.9 MB/event; writing takes about 2 minutes per 1000-event
task.

`deposit_to_group` is the one non-pysupera array. Step+labl must work without hits,
and the step file does not carry it. The alternative is for JAXTPC to write it into
step, since grouping is a step property and readout-independent; that change is not
ours to make.

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

- `segment_motif` = `fragment_sem_type[track.fragment[group.track]]`, mapped to the shared
  motif vocabulary (shower, track, michel, delta, le)
- `instance_particle` = `fragment_instance[...]`
- `segment_pid` = `pdg[instance]`, mapped to the PID classes
- `instance_interaction` = `interaction[instance]`
- `momentum` / `vertex` / `is_primary` = `particle` and `interaction` columns
- object-level targets come straight from the tables

These are exactly the keys pimm's evaluators and heads already read for PILArNet.
Deposits and hits of tracks no fragment owns are -1 everywhere. pysupera leaves about
10% of wire and 17% of pixel deposits unowned in 1M; `deposit_in_window` says why
(1M task 0, 20 events, share of all deposits):

| | wire | pixel |
|---|---|---|
| unowned, out of window | 8.4% | 16.4% |
| unowned, in window (under threshold) | 2.4% | 1.8% |
| owned, out of window (boundary straddlers) | 0.08% | 0.01% |

The flag is the arrival tick `(t0 + |x - anode| / v + pre) / dt` in `[0, n)`, with
the window from the sensor config. It is the one computed array in `labl`; it goes
away once JAXTPC stops saving out-of-window deposits. pimm-data exposes it as
`step.in_window`. On wire, a group is owned exactly when it has a hit.

## Implemented, and checked on real data

On 1M task 0 (20 events per readout), the per-hit labels `JAXTPCDataset` builds from
`labl` + hits were compared with pysupera's own `hit_labels`:

- **wire:** 0 mismatches over 6.46M hits;
- **pixel:** 16 fragment / 2 instance / 4 LE mismatches over 11.9M on-sensor hits,
  all in the majority-resolved straddling groups.

Hits and step carry the JAXTPC group id as `group` (a reference into the `group`
table), and `instance` becomes the particle row.

## Still needed from producers

1. **A doraemon `labl` stage.** It does not exist in the 1M campaign.
2. **JAXTPC:** drop out-of-window deposits (then `deposit_in_window` goes); write
   `deposit_to_group` into step.
3. **pysupera (later):** the generator source per interaction.

## Open

- Whether low-energy-scatter instances (about 86% of instance rows) count as instances
  or as stuff in panoptic metrics; PILArNet treats them as stuff.
- Which momentum is the default `momentum` gather: `|start_mom|` (creation) or
  `|first_mom|` (first visible step). Both are stored.
