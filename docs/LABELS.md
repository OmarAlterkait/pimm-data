# The `labl` file: what to extract from pysupera

Status: implemented on branch `doraemon-labels` (2026-10-27). Writer:
`pimm_data/supera_labl.py`. Reader: `JAXTPCLablReader`. Labels:
`pimm_data/_resolve.py`, through `JAXTPCDataset(labels=...)`. Measured on
`doraemon_2026_1M_lartpc` (JAXTPC f0d2ed3, pysupera dab92d8 / format 4.1.0), task 0,
wire and pixel.

## What a label attaches to

- **The sensor carries no labels.** It is the summed, thresholded readout; it does
  not record which particle produced which charge, so there is no key in it to join a
  label to. Labels reach the readout only through `hits`, the charge each group
  puts on each channel and tick. `labels=` with only `sensor` is rejected.
  On **pixel** the sensor is exactly the sum of hits (where the sum clears the
  sensor threshold). On **wire** it is not: hits are the collected charge per
  (wire, tick) before the field response, so they are unipolar and 2-7 ticks off
  the sensor, which is after it (bipolar on U/V). Wire hits are the per-group
  arriving charge, not a decomposition of the sensor waveform (measured on 1M
  tasks j002500 and j004995).
- **The JAXTPC group is the unit of truth.** Both released point clouds reach labels
  through it:
  - step deposit → `deposit_to_group` → group
  - hit entry → CSR `group_ids` → group

  pysupera's answer, per group, is the fragment that owns it (`groups/fragment_id`).
  Everything else is a property of that fragment, its instance or its interaction.
- **So `labl` is the hierarchy, stored once as maps:**
  group → fragment (pysupera's owner, verbatim) → instance → interaction, and
  group → track, plus the tables they point into. Per hit it stores only where
  pysupera's `hit_labels` differ from the group's (off-sensor pixel hits, groups
  pysupera split). It computes no aggregates. **pysupera is the reference: every
  label is pysupera's own answer, never one we derive differently.**

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
up lives in `labl`: group → track, group → fragment, the particle and interaction
tables, and the references between them.

**The shared-step layout** (the target; the 100k subset
`doraemon_2026_1M_lartpc/subset_100k` is written in it):

| file | per readout? | holds |
|---|---|---|
| step | no: one file, both readouts | deposits (all of them, in or out of either window), and per volume `deposit_to_group` (N), `qs_fractions` (N), attr `n_groups` |
| hits | yes | entries and their group ids, attr `n_groups` |
| sensor | yes | readout |
| labl | yes | `group_to_track`, `deposit_in_window`, track/particle/interaction tables |

- Grouping is readout-independent (step, `deposit_to_group` and
  `group_to_track` are byte-identical between the wire and pixel runs;
  `qs_fractions` to 1 float16 ulp), so it is JAXTPC step output, written once.
- `group_to_track` is label truth (which Geant4 track made the group) and no
  point file carries it: self-supervised use of step or hits needs at most the
  group id, as the key matching deposits to hit entries. It lives in `labl` as
  `group_to_track`. JAXTPC step has no per-deposit track id, so the labl writer
  takes it from another file (`write_labl(track_path=...)`; for the 1M campaign,
  the produced hits). Where it should come from in production is open:
  pysupera writing each group's track id beside `groups/fragment_id`, or
  edep-sim's per-step `track_id` (JAXTPC step rows are edep-sim's `pstep` rows
  in order, and it reproduces `group_to_track` for every group measured).
- `qs_fractions` (a deposit's share of its group's charge, `charge /
  group-sum(charge)` to float16) is kept in step as JAXTPC writes it; pimm-data
  derives it from step charge only for files without it. Groups whose charge is
  all zero (6% of groups: zero-charge photon/neutron steps) have qs 0.
- Out-of-window deposits stay in step. The pixel window is inside the wire
  window, so only deposits outside the wire window are outside both; they are
  kept for the light truth. `labl.deposit_in_window` marks them per readout.

Files produced so far (all of the 1M campaign) have the three group arrays in
hits instead, and `labl` also carries a copy of `deposit_to_group` so step+labl
works without hits. pimm-data reads both layouts; when two files carry the same
array it checks they agree.

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

## Plane-local hits (`group_id_space = 'plane'`)

A hits file whose root attr `group_id_space` is `'plane'` numbers each plane's groups
0..G_plane-1 on its own: plane id k means nothing on another plane or in step, so the
readout files carry no cross-plane or hit <-> deposit correspondence. labl then has, per
volume and hits plane, `plane_<P>/group` (plane id -> volume group id), and pimm-data maps
every hit through it, so `hits.group`, every label and the `group` part are what the
volume-numbered layout gives (`tests/test_supera_labl.py::test_plane_local_layout_loads_the_same`,
and on 1M tasks every deposit and hit, `shuffle_demo/e2e_final.py`). Without labl, hits
carry `plane_group` (the plane-local id: which entries share a block) and no `group`
reference; the matching regime (step.group == hits.group) needs labl. A plane-local hits
file with a labl that lacks a plane's map raises. Example: the 5-task demo
`doraemon_2026_1M_lartpc/shuffle_demo/final` (its README has the full layout).

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

`JAXTPCDataset(labels=...)` reads `labl` and writes the per-point labels it names,
under the PILArNet keys pimm's heads read: classes `segment_motif`, `segment_pid`,
`segment_pdg`; ids `instance_fragment`, `instance_particle`, `instance_interaction`;
`is_primary`, `momentum`, `vertex`. None of them is "the" target: `Collect` picks.
Each is a lookup over every point, so a pipeline asks only for what it uses (1M
wire: all nine cost about 1 s of `get_data`, one or two about 0.25 s).

**pysupera is the reference.** Its own panoptic target (`torchdata`) is the instance's
semantic type (le on the instance's LE fragments) with the instance: in pimm-data
`segment_motif` + `instance_particle`, the level `labels=('pysupera',)`. Motif is
constant on every instance and every fragment, and PID on every fragment and
instance (1M, 50 events per readout, after e48d395; an earlier "motif mixed on a
third of particles" was the motif bug, which read the instance's type on the
fragment's own row). `tests/test_pysupera_golden.py` compares every label with
pysupera's own output on real data. The other levels are views of the same
hierarchy: `segment_motif` + `instance_fragment`, `segment_pid` +
`instance_particle`, class-agnostic `instance_interaction`.
Low-energy scatters (94% of fragments, 83-88% of particles, 7% of the energy) are
stuff by default, as in PILArNet, and each level takes its stuff from its own
class (`_resolve.LEVELS`): -1 in `instance_fragment` where the motif is le, in
`instance_particle` where the pid is le, never in `instance_interaction`. A
Compton scatter of a shower electron is stuff as a fragment but still part of
that electron, so "stuff <=> no instance" holds at both levels. A point's PID is
le exactly when its motif is, so the `pysupera` level's stuff is the particle
level's. LE input differs from pysupera's default: pysupera drops LE points from
its input (`include_le=False`); pimm-data keeps them as input with class le, no
instance with `le_is_stuff`, their host's without (pysupera's `include_le=True`).
`labels=('particle',)` is shorthand for that level's pair; `le_is_stuff=False`
keeps every scatter an instance.
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
| Every owner is a fragment row; every fragment row that owns a group has a `frag_inst_id` that resolves to an instance row (11 fragment rows in 300 events per readout have -1: LE photons that own nothing); every instance row is also a fragment row | group → fragment → instance needs no point ranges |
| `parent_pdg` == `pdg[parent_id]` on every row | `parent_pdg` is redundant |
| All id references (`parent_id`, `ancestor_id`, `frag_parent_id`, `inst_parent_id`, `frag_inst_id`, `groups/fragment_id`) resolve to rows of the same event | they can be stored as row indices |
| Instance rows with `geant4_parent_trackid == -1` are exactly those with `parent_id == id` | "primary" has one definition |
| Rows that are neither fragment nor instance are almost all `kDelta` delta rays merged into their track (plus a few ancestry links) | they stay as the hierarchy. No label the loader emits points at them: a merged delta's points carry its host's fragment; the row is reachable only by `group.track` == `particle.geant4_trackid` |

## Contents

Identity and layout follow JAXTPC format 2: one `labl` file per JAXTPC task file,
same `entry_NNNNNN` entries in the same order, the same `event_id` / `job_id` /
`job_event` attributes and `/dprod/events` table, and a `/config` carrying
`format_version`, readout type and the pysupera provenance (commit, config hashes).

```
entry_NNNNNN/
  volume_{v}/
    group_to_track       (G_v,) int32   Geant4 track id of group g (JAXTPC
                                     group_to_track, from step or hits)
    deposit_to_group  (N_v,) int32   ONLY when step lacks it (produced layout):
                                     copy from hits, the step -> group bridge
    deposit_in_window (N_v,) bool    the deposit's charge arrives inside the
                                     readout window (needs --step --sensor)
    group_fragment    (G_v,) int32   pysupera groups/fragment_id as a particle row
                                     (-1: pysupera left the group unowned)
    plane_<name>/                    per hits plane, only where needed:
      on_sensor       packed bits    pixel: hits on a sensor pixel (attr n); the
                                     others get no label, as in hit_labels
      override_index / override_fragment / override_instance
                                     on-sensor hits whose hit_labels differ from
                                     their group's (groups pysupera split)
  track/      (T rows)  id: the visible Geant4 tracks (owning a group), sorted
  particle/   (P rows)  pysupera particles: fragments/instances, their links,
                        Geant4 ids, kinematics, interaction ref
  interaction/(I rows)  pysupera inter/: one row per interaction it kept
```

`labl` is what pysupera outputs, curated, and nothing more. It does not reach
back into edep-sim: tracks and vertices the readout never saw are not part of
what we label, and anything pysupera adds later is picked up by extending the
column lists. Interactions are separated by row (`particle.interaction`); which
generator made each one is pysupera's to add, and `interaction/` gains it then.

### Group → fragment is pysupera's, verbatim

`group_fragment` is `groups/fragment_id`, ids mapped to rows. An earlier version
compressed it to track → fragment with per-group overrides and gave unowned groups
their track's fragment; that labelled 3.0% (wire) / 4.6% (pixel) of deposits that
pysupera leaves unowned, so it was dropped: pysupera's ownership is the label.

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

The writer takes the JAXTPC groups from the step file when it carries them
(shared-step layout; it then writes no `deposit_to_group`), else from hits. Rerun
on the shared-step task 0 it reproduces the produced `labl` minus
`deposit_to_group`.

### Dropped, and why

| pysupera | reason |
|---|---|
| `points/*`, every `*pc_start/le_start/end` | voxels; no voxels |
| `hit_labels/*` | derivable from the group map (above); `on_sensor` is not a label |
| `groups/is_le` | derivable from `fragment_sem_type` |
| `parent_pdg` | derivable from `parent`; pimm-data gathers it as `particle.parent_pdg` |
| `id`, `inter/interaction_id` | replaced by row position |
| `inter/part_*`, `inter/pc_*` | point/particle ranges; `particle.interaction` covers it |
| `inter/reaction` | empty in 4.1 |
| `events/input`, `input_event`, `jaxtpc_entry`, `inputs/` | pysupera's own provenance; `labl` is entry-aligned with JAXTPC by construction |
| `issues/` | carried as counts in `/config` attrs only |

## What pimm-data does with it

Everything a model or metric uses is a gather; nothing new is stored:

Each point's fragment is its group's pysupera owner (per hit, pysupera's
`hit_labels`); its instance is that fragment's `fragment_instance`. Then, as
pysupera's own `torchdata` does:
- `segment_motif` = le for a point of an LE fragment, else the instance's semantic
  type (`instance_sem_type`), mapped to (shower, track, michel, delta, le)
- `instance_particle` = the instance
- `segment_pid` = le for a point of an LE fragment, else `pdg[instance]` mapped to
  the PID classes (an LE instance is le)
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

## pysupera semantics (audited against its source, dab92d8, and on 300 events per readout)

- **Wire and pixel were labelled with different settings.** The pixel run loads
  `conf/readout/jaxtpc_pixel.yaml` over `config.yaml`: `min_pc_size` 63 (wire 5),
  `distance_threshold` 6.2 (5.2), `delta_min_standoff` 8 mm (0, so pixel deltas
  within 8 mm of their track are merged into it), `check_group_ownership` off
  (on), and `point_source` hits (wire: deposits). So the LE and delta classes are
  not defined alike in the two readouts. `labl/config` now records the effective
  settings (`supera_settings/...`) and pysupera's run attrs (`supera_*`:
  `kinematics_steps`, `point_source`, pixel `hit_energy`, `pixel_geometry`).
- **Merge counts** (`fragment_merge_count`, `instance_merge_count`) count pysupera
  particles in the fragment/instance: split pieces of one Geant4 track and buried
  deltas count; -1 off that level. Not derivable from labl, so kept.
- **Kinematics**: `*_pos` is (x, y, z, t) in mm and ns, `*_mom` is (px, py, pz, E)
  in MeV with E the total energy. `start`/`end` are the Geant4 particle's own;
  `first`/`last` exist on instance rows only and are the earliest / latest
  *visible* step (a step whose group left a hit), NaN on instances with no visible
  step. `last_*` is NaN on shower instances.
- **Interactions are per readout.** pysupera keeps an edep-sim vertex exactly when
  one of its groups is owned, then renumbers, so `interaction` rows (and
  `instance_interaction`) are not comparable between readouts: 89% of events keep
  different sets (pixel's is almost always a subset of wire's). `vertex_id` is
  the shared key. `interaction.time` is ns. `particle.interaction_type` is the
  particle's Geant4 creation process (edep-sim `InteractionType`), not a property
  of the interaction.
- **Issues**: per-event pysupera issues are on the labl entry
  (`supera_issue_codes`, `supera_issue_messages`). In the campaign: 4 pixel events,
  all OUTSIDE_READOUT_WINDOW, written empty; none on wire.
- **Split-piece rows** (`defragment`): `geant4_parent_trackid` equals the row's
  own track id, because the piece's parent is the original particle.

Labels follow pysupera exactly (decided 2026-11-05: pysupera is the reference):
- a group pysupera leaves unowned has no fragment, whatever its track;
- pixel hits off the sensor have no label (`hit_labels` -1);
- hits of groups pysupera split carry pysupera's per-hit fragment and instance;
- class = the instance's semantic type, or le for an LE fragment (motif and PID);
- interaction = the instance's (81 pixel fragments, 78 LE, sit in an instance of
  another interaction; their own particle row keeps their true vertex);
- `le_is_stuff=False` is pysupera's `include_le` exactly (LE points keep their
  host's instance and interaction); `True` (default) also drops their fragment
  and particle ids.

## Implemented, and checked on real data

On 1M task 0 (20 events per readout), the per-hit labels `JAXTPCDataset` builds from
`labl` + hits were compared with pysupera's own `hit_labels`:

- **wire:** 0 mismatches over 6.46M hits;
- **pixel:** 0 with labl's per-hit overrides (without them: about 15 fragment /
  2 instance / 4 LE per event, all in groups pysupera splits; 50 events, 286M hits).

Hits and step carry the JAXTPC group id as `group` (a reference into the `group`
table), and `instance` becomes the particle row.

## Still needed from producers

1. **A doraemon `labl` stage.** It does not exist in the 1M campaign.
2. **JAXTPC:** the shared-step layout above: one step per task for both
   readouts, carrying `deposit_to_group`, `qs_fractions`, `group_to_track` and
   `n_groups`; hits without those three arrays. Also per-step momentum `p`
   (edep-sim has it) for the momentum targets.
3. **pysupera (later):** the generator source per interaction.

## Open

- Which momentum is the default `momentum` gather: `|start_mom|` (creation) or
  `|first_mom|` (first visible step). Both are stored.
