"""Coeff-corpus reader/writer — the flat-columnar wavelet-coefficient shards.

The coeff corpus is a purpose-built training cache (produced by helix DSP), read
hot in the loop, so it is **flat columnar**, not per-event groups: shard-wide
``/coord`` + ``/value`` arrays plus an ``event_offset`` index; one contiguous
slice per event; ``plane_gid`` is a column. This is pimm-data's own offset idiom
(cf. the optical second row-space) applied shard-wide.

One modality per file (``{name}_coeff_*.h5`` noisy input, ``{name}_coeff_clean_*``
the clean target), joined by identity — never mixed. The on-disk layout is the
shared contract defined in helix ``COEFF_CORPUS_DESIGN.md`` §3; this reader
implements it with **standalone h5py** (no helix import at read time — no cycle),
and is round-trip golden-tested against helix's reference codec.

Layout::

    /config  attrs: n_events, dataset_name, file_index, global_event_offset,
                    readout_type, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
                    sigma_norm, basis_digest, removal_json, threshold_json
             datasets: band_lengths (n_bands,), gids (G,), n_wires (G,),
                       [norm_sigma (G, n_bands)]
    /coord   band uint8, plane_gid uint8, wire int32, tau int32   (all (M,))
             event_offset int64 (n_events+1,); sigma_threshold f32 (n_events,G,n_bands)
    /value   float32 (M,)
    /ident   run (n_events,), source_file (n_events,), event int64 (n_events,)
"""

import glob
import json
import os

import numpy as np
import h5py

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase

_STR = h5py.string_dtype(encoding="utf-8")


class CoeffTPCReader(ShardReaderBase):
    """Flat-columnar coeff-shard reader. ``modality`` selects the file family
    (``'coeff'`` noisy / ``'coeff_clean'`` target)."""

    _MODALITY = "coeff"

    def __init__(self, data_root, split="", dataset_name="coeff_tpc", modality="coeff", **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self._MODALITY = modality               # instance attr shadows the class default
        # shard-shared metadata, cached from the first shard's /config
        self.band_lengths = None
        self.gids = None
        self.n_wires = None
        self.norm_sigma = None
        self.sigma_norm = None
        self.config_attrs = None
        self._id2pos = {}                       # h5_path -> {physical event id: write-order position}
        self._init_shards()

    def _find_files(self):
        """Glob shards with the tag anchored to a digit — ``{name}_{modality}_[0-9]*.h5``
        — so ``modality='coeff'`` does NOT also match ``{name}_coeff_clean_*.h5``
        (``coeff`` is a prefix of ``coeff_clean``)."""
        pat = f"{self.dataset_name}_{self._MODALITY}_[0-9]*.h5"
        if isinstance(self.split, (list, tuple)):
            files, file_runs = [], []
            for run in self.split:
                rf = sorted(glob.glob(os.path.join(self.data_root, run, pat)))
                if not rf:
                    raise FileNotFoundError(
                        f"runs=: no {self._MODALITY} shards for run {run!r} under {self.data_root}")
                files.extend(rf); file_runs.extend([run] * len(rf))
            self._file_runs = file_runs
            return files
        for p in (os.path.join(self.data_root, self.split, pat),
                  os.path.join(self.data_root, pat)):
            files = sorted(glob.glob(p))
            if files:
                return files
        return []

    def _index_for_shard(self, h5_path):
        """Present event **ids** for one flat shard — the ``/ident/event`` physical
        ids (in write order) so the cross-modality joint index aligns coeff↔coeff_clean
        by IDENTITY, not position. Also caches shared basis/geometry, and a
        physical-id→write-position map (event_offset is positional)."""
        meta = read_shard_meta(h5_path)
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
            cfg = f["config"]
            if self.band_lengths is None:
                self.config_attrs = dict(cfg.attrs)
                self.band_lengths = cfg["band_lengths"][:].astype(np.int32)
                self.gids = cfg["gids"][:].astype(np.int32) if "gids" in cfg else None
                self.n_wires = cfg["n_wires"][:].astype(np.int32) if "n_wires" in cfg else None
                self.norm_sigma = cfg["norm_sigma"][:].astype(np.float32) if "norm_sigma" in cfg else None
                self.sigma_norm = float(cfg.attrs.get("sigma_norm", 1.0))
            if "ident" in f and "event" in f["ident"]:
                ev = f["ident"]["event"][:].astype(np.int64)
            else:                                       # legacy shard without identity
                ev = np.arange(int(meta["n_events"]), dtype=np.int64)
        # keyed by path (not shard index) so it survives build_joint_index re-selection
        self._id2pos[str(h5_path)] = {int(e): i for i, e in enumerate(ev)}
        return np.asarray(ev, dtype=np.int64)

    def _locate_flat(self, idx):
        self._ensure_open()
        file_idx, event_id = self.locate(idx)           # event_id = physical /ident/event id
        pos = self._id2pos[str(self.h5_files[file_idx])][int(event_id)]
        return self._h5data[file_idx], pos

    def read_event(self, idx):
        """One event → flat dict of coeff rows (sliced by ``event_offset`` at the
        physical event's write position)."""
        f, ev = self._locate_flat(idx)
        coord = f["coord"]
        off = coord["event_offset"]
        a, b = int(off[ev]), int(off[ev + 1])
        return {
            "band": coord["band"][a:b].astype(np.int32),
            "plane_gid": coord["plane_gid"][a:b].astype(np.int32),
            "wire": coord["wire"][a:b].astype(np.int32),
            "tau": coord["tau"][a:b].astype(np.int32),
            "value": f["value"][a:b].astype(np.float32),
        }


def _ds(grp, name, data, comp):
    kw = dict(compression=comp, compression_opts=4) if comp == "gzip" else {}
    grp.create_dataset(name, data=data, **kw)


def write_coeff_shard(path, events, *, band_lengths, gids, n_wires, basis_attrs,
                      norm_sigma=None, dataset_name="", file_index=0,
                      global_event_offset=0, compression="gzip"):
    """Write a flat-columnar coeff shard (standalone; the layout helix reads).

    ``events`` is a list of per-event dicts with keys ``band, plane_gid, wire,
    tau, value, sigma_threshold, run, source_file, event``. All events share the
    plane set (``gids`` / ``n_wires``) and the basis. ``basis_attrs`` supplies the
    ``/config`` scalar attrs (wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
    sigma_norm, basis_digest, removal_json, threshold_json).
    """
    if not events:
        raise ValueError("write_coeff_shard: no events")

    # validate the basis contract so a pimm-written shard is not silently malformed
    # (helix's reader validates on read; fail here at write time instead).
    _REQUIRED = ("wavelet", "dwt_level", "dwt_mode", "n_ticks_raw", "pad", "sigma_norm")
    missing = [k for k in _REQUIRED if k not in basis_attrs]
    if missing:
        raise ValueError(f"basis_attrs missing required keys {missing}")
    import pywt
    padded = int(basis_attrs["n_ticks_raw"]) + int(basis_attrs["pad"])
    want = tuple(int(c.shape[-1]) for c in pywt.wavedec(
        np.zeros(padded, np.float32), str(basis_attrs["wavelet"]),
        level=int(basis_attrs["dwt_level"]), mode=str(basis_attrs["dwt_mode"])))
    if tuple(int(x) for x in band_lengths) != want:
        raise ValueError(
            f"band_lengths {tuple(int(x) for x in band_lengths)} inconsistent with basis "
            f"({basis_attrs['wavelet']} L{basis_attrs['dwt_level']} @ {padded}) → expected {want}")

    def cat(key, dt):
        parts = [np.asarray(e[key], dt) for e in events]
        return np.concatenate(parts) if parts else np.empty(0, dt)

    band = cat("band", np.uint8)
    plane_gid = cat("plane_gid", np.int32)          # int32: gid can exceed 255 (matches helix)
    wire = cat("wire", np.int32)
    tau = cat("tau", np.int32)
    value = cat("value", np.float32)
    counts = np.array([len(e["value"]) for e in events], np.int64)
    event_offset = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sigma = np.stack([np.asarray(e["sigma_threshold"], np.float32) for e in events])

    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["n_events"] = len(events)
        cfg.attrs["dataset_name"] = dataset_name
        cfg.attrs["file_index"] = file_index
        cfg.attrs["global_event_offset"] = global_event_offset
        cfg.attrs["readout_type"] = "wire"
        for k, v in basis_attrs.items():
            cfg.attrs[k] = v
        cfg.create_dataset("band_lengths", data=np.asarray(band_lengths, np.int32))
        cfg.create_dataset("gids", data=np.asarray(gids, np.int32))
        cfg.create_dataset("n_wires", data=np.asarray(n_wires, np.int32))
        if norm_sigma is not None:
            cfg.create_dataset("norm_sigma", data=np.asarray(norm_sigma, np.float32))

        coord = f.create_group("coord")
        _ds(coord, "band", band, compression)
        _ds(coord, "plane_gid", plane_gid, compression)
        _ds(coord, "wire", wire, compression)
        _ds(coord, "tau", tau, compression)
        coord.create_dataset("event_offset", data=event_offset)
        _ds(coord, "sigma_threshold", sigma, compression)

        _ds(f, "value", value, compression)

        ident = f.create_group("ident")
        ident.create_dataset("run", data=np.array([e.get("run", "") for e in events], dtype=_STR))
        ident.create_dataset("source_file", data=np.array([e.get("source_file", "") for e in events], dtype=_STR))
        ident.create_dataset("event", data=np.array([e.get("event", -1) for e in events], np.int64))
