"""Distributed matter power spectrum of the particles (no host gather of particles or mesh)."""
from __future__ import annotations

import numpy as np


def measure_pk(sim, psi, max_block_elems: int = 1 << 26) -> dict:
    """CIC-deposit the particles on the (res_pm)^3 PM mesh, slab rFFT, bin |delta_k|^2 in shells of width k_f.

    The CIC window is deconvolved per axis (sinc^2); no shot-noise subtraction (returned as ``shot``).
    The |k| binning runs in a fori_loop over kx blocks of <= ``max_block_elems`` elements so that no full-size
    k-grid temporaries exist. Returns numpy arrays, identical on every process.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from discodj_dist.core.distributed_pm import scatter_dx_distributed
    from discodj_dist.nbody.acc_distributed import _slab_prfft3d_x

    cfg = sim.cfg
    N, box, res = cfg.res_pm, cfg.boxsize, cfg.res
    mesh = sim.sharding_pm_disp.mesh
    ax = mesh.axis_names
    if mesh.shape[ax[1]] != 1:
        raise ValueError("power_spectrum needs the slab PM mesh (pm_fft_backend='JAX_RFFT')")
    sf = NamedSharding(mesh, P(ax[0], ax[1], None))
    nb = int(np.ceil(np.sqrt(3) * N / 2)) + 2

    @jax.jit
    def _pk(ps):
        delta = scatter_dx_distributed(ps, halo_size=cfg.halo_size, worder=2, n_part_tot=res ** 3, res_pm=N,
                                       boxsize=box, dtype_num=32, sharding=sim.sharding_pm_disp,
                                       disp_div=box / N, disp_dtype=jnp.float32)
        fk = _slab_prfft3d_x(delta.astype(jnp.float32), sharding=sf)   # global (N, N, N//2+1), P(None, x, None)

        def local(f):
            ly = f.shape[1]
            iy = (lax.axis_index(ax[0]) * ly + jnp.arange(ly)).astype(jnp.float32)
            iy = jnp.where(iy >= N // 2, iy - N, iy)
            iz = jnp.arange(N // 2 + 1, dtype=jnp.float32)
            sinc2 = lambda i: jnp.where(i == 0, 1.0, (jnp.sin(np.pi * i / N) / (np.pi * i / N)) ** 2)
            wyz = (sinc2(iy)[:, None] * sinc2(iz)[None, :]) ** 2
            mult = jnp.where((iz == 0) | (iz == N // 2), 1.0, 2.0)[None, :]      # rfft half-space weights
            rows = max(1, min(N, max_block_elems // (ly * (N // 2 + 1))))
            while N % rows:
                rows -= 1

            def body(c, acc):
                blk = lax.dynamic_slice_in_dim(f, c * rows, rows, axis=0)
                ix = (c * rows + jnp.arange(rows)).astype(jnp.float32)
                ix = jnp.where(ix >= N // 2, ix - N, ix)
                kn = jnp.sqrt(ix[:, None, None] ** 2 + iy[None, :, None] ** 2 + iz[None, None, :] ** 2)
                b = jnp.floor(kn + 0.5).astype(jnp.int32).reshape(-1)
                pw = (jnp.abs(blk) ** 2 / (sinc2(ix)[:, None, None] ** 2 * wyz[None])) * mult[None]
                m = jnp.broadcast_to(mult[None], kn.shape)
                return (acc[0].at[b].add(pw.reshape(-1)), acc[1].at[b].add(m.reshape(-1)),
                        acc[2].at[b].add((kn * m).reshape(-1)))

            z = jnp.zeros(nb, jnp.float32)
            acc = lax.fori_loop(0, N // rows, body, (z, z, z))
            return tuple(lax.psum(a, ax[0]) for a in acc)

        return shard_map(local, mesh=mesh, in_specs=(P(None, ax[0], None),), out_specs=(P(), P(), P()),
                         check_rep=False)(fk)

    sp, sm, sk = (np.asarray(jax.device_get(a)).astype(np.float64) for a in _pk(psi))
    ok = sm > 0
    kf = 2 * np.pi / box
    return dict(k=sk[ok] / sm[ok] * kf, pk=sp[ok] / sm[ok] * box ** 3 / float(N) ** 6, nmodes=sm[ok],
                shot=box ** 3 / res ** 3)
