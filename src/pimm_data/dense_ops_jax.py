"""JAX/GPU densification — the thin jax twin of :func:`pimm_data.dense_ops.densify`.

`dense_ops` is the full torch GPU dense path (densify + noise + digitize). This
module deliberately mirrors ONLY the scatter, for pipelines that run their DSP in
jax and would otherwise pay a CPU densify + a host→device copy of the dense grid.
Everything else (noise, digitize) has jax twins in :mod:`pimm_data.noise_jax`.

Semantics match `dense_ops.densify` exactly: scatter-add into a zero grid, which
on the unique COO the readers produce is identical to last-wins assignment (and
so matches the numpy reference), and is collision-free/deterministic on GPU.

jax is imported lazily, so importing pimm-data never pulls it in.
"""

import numpy as np

__all__ = ["densify_plane_jax", "densify_jax"]

# A single monotonic scatter capacity. A per-call power-of-two ladder still
# recompiles whenever an event crosses a bucket; one capacity that only ever
# grows means the scatter compiles ONCE for a given detector (twice at worst, if
# an early event is unusually small).
_CAP = 1 << 19


_BUF = {}


def _cap_for(n):
    global _CAP
    while n > _CAP:
        _CAP <<= 1
        _BUF.clear()
    return _CAP


def _padded(w, t, v, cap, n_wires):
    """Fill REUSED host buffers instead of allocating via np.concatenate.

    Padding to a static capacity is what keeps the scatter from recompiling, but
    doing it with concatenate allocates and copies three arrays per plane per
    event — measured 17.7 ms/plane against 0.4 ms for the scatter itself. Writing
    into preallocated buffers removes the allocation entirely.
    """
    b = _BUF.get(cap)
    if b is None:
        b = (np.empty(cap, np.int32), np.empty(cap, np.int32), np.empty(cap, np.float32))
        _BUF[cap] = b
    bw, bt, bv = b
    n = w.shape[0]
    bw[:n] = w; bt[:n] = t; bv[:n] = v
    bw[n:] = n_wires          # out of range -> dropped by the scatter
    bt[n:] = 0; bv[n:] = 0.0
    return bw, bt, bv


def _jnp():
    import jax.numpy as jnp
    return jnp


def densify_plane_jax(wire, time, value, n_wires, n_ticks):
    """Scatter one plane's sparse COO into a dense ``(n_wires, n_ticks)`` grid.

    ``wire``/``time`` are absolute integer grid indices, ``value`` the per-hit
    (pedestal-subtracted) ADC. Returns a float32 device array.

    The hit count varies per event, and a varying input shape makes jax recompile
    the scatter EVERY event (measured: it doubled a 36-event build). The COO is
    therefore padded up to the next power of two, with the padding pointing at an
    out-of-range cell that ``mode='drop'`` discards — so only a handful of shapes
    ever compile and the result is unchanged.
    """
    jnp = _jnp()
    w = np.asarray(wire).reshape(-1)
    t = np.asarray(time).reshape(-1)
    v = np.asarray(value).reshape(-1)
    n = w.shape[0]
    if not (t.shape[0] == n == v.shape[0]):
        raise ValueError(
            f"densify_plane_jax: wire/time/value length mismatch "
            f"({n}/{t.shape[0]}/{v.shape[0]})")
    grid = jnp.zeros((int(n_wires), int(n_ticks)), jnp.float32)
    if n == 0:
        return grid
    cap = _cap_for(n)                                   # ONE shape, monotonic
    w, t, v = _padded(w, t, v, cap, int(n_wires))
    return grid.at[jnp.asarray(w), jnp.asarray(t)].add(jnp.asarray(v), mode="drop")


def densify_jax(planes, geom=None):
    """Densify a ``{plane_id: (wire, time, value, n_wires, n_ticks)}`` mapping.

    Convenience wrapper for the per-event corpus path; ``geom`` (if given) supplies
    ``{plane_id: {'n_wires', 'n_ticks'}}`` and overrides the per-plane extents, so
    a fixed detector geometry is used rather than data-inferred (event-dependent)
    extents — the same contract as ``Densify(require_shape=True)``.
    """
    out = {}
    for pid, spec in planes.items():
        wire, time, value = spec[0], spec[1], spec[2]
        nw, nt = (spec[3], spec[4]) if len(spec) >= 5 else (None, None)
        if geom is not None and pid in geom:
            nw = geom[pid].get("n_wires", nw)
            nt = geom[pid].get("n_ticks", nt)
        if nw is None or nt is None:
            raise ValueError(f"densify_jax: no extent for plane {pid}")
        out[pid] = densify_plane_jax(wire, time, value, nw, nt)
    return out
