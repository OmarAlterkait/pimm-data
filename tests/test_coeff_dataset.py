"""CoeffTPCDataset / CoeffTPCReader — flat-columnar coeff corpus.

- self-contained: pimm-data write_coeff_shard → reader/Dataset/Collect round-trip.
- cross-repo golden: helix's reference codec ⇄ pimm-data reader agree on the layout
  (skipped when helix is not importable).
"""
import json
import os
import sys

import numpy as np
import pytest

from pimm_data.readers.coeff_tpc import CoeffTPCReader, write_coeff_shard
from pimm_data.coeff import CoeffTPCDataset
from pimm_data.transform import Compose

BAND_LENGTHS = [8, 8, 16]
GIDS = [0, 1]
N_WIRES = [4, 4]
BASIS_ATTRS = dict(
    wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0,
    sigma_norm=2.6, basis_digest="deadbeef",
    removal_json=json.dumps(dict(kind="gate", kgate=3.0)),
    threshold_json=json.dumps(dict(method="universal")),
)


def _rand_event(rng, ev):
    band, gid, wire, tau, val = [], [], [], [], []
    for gi, g in enumerate(GIDS):
        for b, L in enumerate(BAND_LENGTHS):
            n = rng.integers(0, 6)
            if n == 0:
                continue
            band.append(np.full(n, b, np.uint8))
            gid.append(np.full(n, g, np.uint8))
            wire.append(rng.integers(0, N_WIRES[gi], n).astype(np.int32))
            tau.append(rng.integers(0, L, n).astype(np.int32))
            val.append(rng.standard_normal(n).astype(np.float32))
    cat = lambda xs, dt: (np.concatenate(xs).astype(dt) if xs else np.empty(0, dt))
    return dict(
        band=cat(band, np.uint8), plane_gid=cat(gid, np.uint8),
        wire=cat(wire, np.int32), tau=cat(tau, np.int32), value=cat(val, np.float32),
        sigma_threshold=rng.random((len(GIDS), len(BAND_LENGTHS))).astype(np.float32),
        run="run_000", source_file="cx_sensor_0000.h5", event=ev)


def _write_shard(path, n_events=3, seed=0):
    rng = np.random.default_rng(seed)
    events = [_rand_event(rng, e) for e in range(n_events)]
    write_coeff_shard(path, events, band_lengths=BAND_LENGTHS, gids=GIDS,
                      n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="cx")
    return events


def test_reader_roundtrip(tmp_path):
    events = _write_shard(tmp_path / "cx_coeff_0000.h5")
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert len(r) == 3
    np.testing.assert_array_equal(r.band_lengths, BAND_LENGTHS)
    for i, e in enumerate(events):
        got = r.read_event(i)
        for k in ("band", "plane_gid", "wire", "tau", "value"):
            np.testing.assert_array_equal(got[k], e[k], err_msg=f"event {i} {k}")


def test_dataset_get_data(tmp_path):
    events = _write_shard(tmp_path / "cx_coeff_0000.h5")
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")
    assert len(ds) == 3
    d = ds.get_data(0)
    assert set(d["coeff"]) == {"band", "plane_gid", "wire", "tau", "value", "_meta"}
    # shard tables ride WITH the sample so a worker-side tokenizer is self-sufficient
    assert set(d["coeff"]["_meta"]) >= {"gids", "n_wires", "band_lengths", "norm_sigma"}
    np.testing.assert_array_equal(d["coeff"]["band"], events[0]["band"])
    assert d["coeff"]["value"].shape == (len(events[0]["value"]), 1)
    np.testing.assert_array_equal(ds.band_lengths, BAND_LENGTHS)


def test_dataset_collect(tmp_path):
    _write_shard(tmp_path / "cx_coeff_0000.h5")
    collect = dict(type="Collect", parts={"coeff": dict(
        keys=("band", "plane_gid", "wire", "tau", "value"), feat_keys=("value",),
        offset_keys_dict=dict(offset="band"))})
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx", transform=[collect])
    out = ds[0]
    assert "coeff_offset" in out and "coeff_value" in out
    n = int(out["coeff_offset"][-1])
    assert out["coeff_value"].shape[0] == n
    assert out["coeff_band"].shape[0] == n


# ── cross-repo golden: helix reference codec ⇄ pimm-data reader ──────────────

_HELIX = "/sdf/group/neutrino/omara/helix-consolidate"


def _import_helix():
    if _HELIX not in sys.path:
        sys.path.insert(0, _HELIX)
    try:
        import helix.core.coeff_io  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not os.path.isdir(_HELIX) or not _import_helix(),
                    reason="helix (consolidation worktree) not importable")
def test_cross_repo_golden(tmp_path):
    from helix.core.backend import set_backend
    from helix.core.provenance import BasisDescriptor, derive_band_lengths
    from helix.core.wavelet import SparseResult
    from helix.core.coeff_event import CoeffEvent
    from helix.core.coeff_io import write_coeff_shard as hx_write
    set_backend("numpy")

    wl, lvl, mode, nt = "db2", 2, "periodization", 32
    bl = derive_band_lengths(wl, lvl, mode, nt)
    basis = BasisDescriptor(wavelet=wl, level=lvl, mode=mode, n_ticks_raw=nt, pad=0,
                            band_lengths=bl, removal=dict(kind="gate"),
                            threshold=dict(method="universal"), sigma_norm=2.6)
    rng = np.random.default_rng(7)
    results = {}
    for gid, nw in ((0, 4), (2, 3)):
        coeffs = []
        for L in bl:
            c = rng.standard_normal((nw, L)).astype(np.float32)
            c[rng.random((nw, L)) > 0.4] = 0.0
            coeffs.append(c)
        results[gid] = SparseResult(coeffs=coeffs, n_kept=0, n_total=0,
                                    sigma_per_band=np.ones(len(bl), np.float32),
                                    wavelet=wl, level=lvl, mode=mode)
    ce = CoeffEvent.from_sparse_results(results, basis=basis, run="run_000",
                                        source_file="cx_sensor_0000.h5", event=0)
    # helix WRITES the shard; pimm-data READS it
    path = tmp_path / "cx_coeff_0000.h5"
    hx_write(path, [ce], dataset_name="cx")
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    got = r.read_event(0)
    for k in ("band", "plane_gid", "wire", "tau", "value"):
        np.testing.assert_array_equal(got[k], getattr(ce, k), err_msg=f"golden {k}")
    np.testing.assert_array_equal(r.band_lengths, np.asarray(bl, np.int32))


# ── regression: audit fixes ─────────────────────────────────────────────────

def _tagged(path, event_ids, value_offset=0.0, gids=GIDS, nwires=N_WIRES):
    """One coeff per event; value encodes (event id + offset) so pairing is checkable."""
    events = [dict(
        band=np.array([1], np.uint8), plane_gid=np.array([gids[0]], np.int32),
        wire=np.array([0], np.int32), tau=np.array([0], np.int32),
        value=np.array([ev + value_offset], np.float32),
        sigma_threshold=np.ones((len(gids), len(BAND_LENGTHS)), np.float32),
        run="run_000", source_file="cx_sensor_0000.h5", event=ev) for ev in event_ids]
    write_coeff_shard(path, events, band_lengths=BAND_LENGTHS, gids=gids,
                      n_wires=nwires, basis_attrs=BASIS_ATTRS, dataset_name="cx")


def test_glob_no_coeff_clean_collision(tmp_path):
    _write_shard(tmp_path / "cx_coeff_0000.h5", n_events=3)
    _write_shard(tmp_path / "cx_coeff_clean_0000.h5", n_events=3)   # same dir
    assert len(CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx", modality="coeff")) == 3
    assert len(CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx", modality="coeff_clean")) == 3
    assert len(CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")) == 3


def test_join_by_identity_not_position(tmp_path):
    _tagged(tmp_path / "cx_coeff_0000.h5", [10, 11, 12], value_offset=0.0)
    _tagged(tmp_path / "cx_coeff_clean_0000.h5", [10, 12], value_offset=0.5)   # event 11 dropped
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"))
    assert len(ds) == 2                                    # intersection {10,12}
    for i in range(2):
        d = ds.get_data(i)
        nv = float(d["coeff"]["value"][0, 0])
        cv = float(d["coeff_clean"]["value"][0, 0])
        assert cv - nv == 0.5, f"idx {i}: noisy {nv} paired with clean {cv} (mispair)"


def test_writer_validates_basis(tmp_path):
    ev = [dict(band=np.array([1], np.uint8), plane_gid=np.array([0], np.int32),
               wire=np.array([0], np.int32), tau=np.array([0], np.int32),
               value=np.array([1.0], np.float32),
               sigma_threshold=np.ones((len(GIDS), len(BAND_LENGTHS)), np.float32),
               run="r", source_file="s.h5", event=0)]
    missing = {k: v for k, v in BASIS_ATTRS.items() if k != "pad"}
    with pytest.raises(ValueError, match="missing required"):
        write_coeff_shard(tmp_path / "x_coeff_0000.h5", ev, band_lengths=BAND_LENGTHS,
                          gids=GIDS, n_wires=N_WIRES, basis_attrs=missing, dataset_name="x")
    with pytest.raises(ValueError, match="inconsistent"):
        write_coeff_shard(tmp_path / "y_coeff_0000.h5", ev, band_lengths=[8, 8, 99],
                          gids=GIDS, n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="y")


def test_plane_gid_beyond_255(tmp_path):
    _tagged(tmp_path / "cx_coeff_0000.h5", [0], gids=[300, 301])
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert int(r.read_event(0)["plane_gid"][0]) == 300      # no uint8 wrap


@pytest.mark.skipif(not os.path.isdir(_HELIX) or not _import_helix(),
                    reason="helix not importable")
def test_reverse_golden_pimm_write_helix_read(tmp_path):
    from helix.core.provenance import BasisDescriptor, derive_band_lengths
    from helix.core.coeff_io import read_coeff_event as hx_read
    bl = derive_band_lengths("db2", 2, "periodization", 32)      # == BAND_LENGTHS
    basis = BasisDescriptor(wavelet="db2", level=2, mode="periodization", n_ticks_raw=32,
                            pad=0, band_lengths=bl, removal={}, threshold={}, sigma_norm=2.6)
    battrs = dict(wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32,
                  pad=0, sigma_norm=2.6, basis_digest=basis.digest(),
                  removal_json="{}", threshold_json="{}")
    ev = [dict(band=np.array([1], np.uint8), plane_gid=np.array([0], np.int32),
               wire=np.array([0], np.int32), tau=np.array([0], np.int32),
               value=np.array([3.0], np.float32),
               sigma_threshold=np.ones((len(GIDS), len(bl)), np.float32),
               run="r", source_file="s.h5", event=0)]
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", ev, band_lengths=list(bl),
                      gids=GIDS, n_wires=N_WIRES, basis_attrs=battrs, dataset_name="cx")
    ce = hx_read(tmp_path / "cx_coeff_0000.h5", 0)              # helix reads + validates
    assert float(ce.value[0]) == 3.0 and int(ce.plane_gid[0]) == 0
