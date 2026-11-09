"""Data-loader recipe — doraemon final layout, 3D panoptic segmentation, pysupera's target.

Labels: labels=('pysupera',) = segment_motif (the instance's semantic type; le on its LE
fragments) + instance_particle, pysupera's own panoptic target (the reference; see
docs/LABELS.md and tests/test_pysupera_golden.py). LE points stay as input with class le
and, with le_is_stuff (default), no instance.

Data: the final layout (per-readout step with in-window deposits, plane-local hits, labl
with plane maps), e.g. JAXTPC_DATA_ROOT=/sdf/data/neutrino/omara/doraemon_2026_1M_lartpc/
final_10k/jaxtpc_wire. Split by task directory.

Batch after collate (one row per 3 mm voxel): step_coord (N,3) step_grid_coord (N,3) step_feat (N,4)
step_segment_motif (N,1) step_instance_particle (N,1) step_offset (B,).
Data-loading half only; the model half is filled when lifted into a pimm config.
"""
import os

_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/final/jaxtpc_wire")
_dataset_name = os.environ.get("JAXTPC_DATASET_NAME", "sim_wire")
_center, _scale = [0.0, 0.0, 0.0], 2160.0 * 3 ** 0.5
_tasks = sorted(os.listdir(os.path.join(_data_root, "j000000"))) if os.path.isdir(
    os.path.join(_data_root, "j000000")) else []
_train_runs = [f"j000000/{t}" for t in _tasks[:-1]] or "j000000/*"
_val_runs = [f"j000000/{t}" for t in _tasks[-1:]] or "j000000/*"

_head = [
    # every deposit counts: 3 mm voxels on pysupera's grid, energy summed, each voxel
    # labelled by its winning deposit (track > shower > delta > michel > le, then earliest)
    dict(type="Voxelize", grid_size=3.0, origin=-2160.0),
    dict(type="NormalizeCoord", center=_center, scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
]
_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
_collect = dict(type="Collect", parts={"step": dict(
    keys=("coord", "grid_coord", "segment_motif", "instance_particle"),
    feat_keys=("coord", "energy"))})
transform = [dict(type="Apply", on="step", transforms=_head + _aug), _collect]
test_transform = [dict(type="Apply", on="step", transforms=_head), _collect]
_common = dict(type="JAXTPCDataset", data_root=_data_root, dataset_name=_dataset_name,
               modalities=("step",), labels=("pysupera",), min_deposits=1024)
data = dict(
    num_classes=5, ignore_index=-1, names=["shower", "track", "michel", "delta", "le"],
    train=dict(_common, runs=_train_runs, transform=transform, max_len=-1),
    val=dict(_common, runs=_val_runs, transform=test_transform, max_len=1000),
)
