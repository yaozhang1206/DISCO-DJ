"""Small host-side helpers for per-process shard files written by Simulation.save_local_shards."""
from __future__ import annotations

import glob

import numpy as np


def load_shards(prefix: str, out=None) -> np.ndarray:
    """Reassemble ``{prefix}.proc*.npz`` into one numpy array (only for arrays that fit in host memory)."""
    files = sorted(glob.glob(f"{prefix}.proc*.npz"))
    if not files:
        raise FileNotFoundError(f"no shard files {prefix}.proc*.npz")
    for fn in files:
        with np.load(fn) as z:
            if out is None:
                first = z["data_0"]
                out = np.empty(tuple(z["shape"]), first.dtype)
            i = 0
            while f"data_{i}" in z:
                d, st = z[f"data_{i}"], z[f"start_{i}"]
                out[tuple(slice(s, s + n) for s, n in zip(st, d.shape))] = d
                i += 1
    return out


def gather_to_host(arr) -> np.ndarray:
    """Full array on every process (all-gather). Debug / small runs only."""
    import jax
    from jax.experimental import multihost_utils
    if jax.process_count() == 1:
        return np.asarray(arr)
    return np.asarray(multihost_utils.process_allgather(arr, tiled=True))
