"""High-level multi-GPU PM simulation: Gaussian / external ICs -> distributed 2LPT -> DKD PM steps -> P(k).

Typical use (see examples/run_sim.py)::

    from discodj_mgpu import init_distributed, SimConfig, Simulation
    init_distributed()                                   # before any other JAX call
    sim = Simulation(SimConfig(boxsize=2000., res=1024, res_pm=2048,
                               cosmo=dict(Omega_m=0.3175, Omega_b=0.049, h=0.6711, n_s=0.9624, sigma8=0.834)))
    delta = sim.gaussian_delta(pk_table, seed=1)          # linear delta extrapolated to z = 0
    psi, mom = sim.lpt2(delta, a=1/128)                   # consumes delta
    psi, mom, info = sim.evolve(psi, mom, 1/128, 1.0, n_steps=20)
    pk = sim.power_spectrum(psi)

All big arrays are global, sharded jax.Arrays: the particle state (psi, mom) has shape (res, res, res, 3) on the
slab PM mesh (n_dev, 1); fields used during the IC/LPT stage live on a squarer (a, b) mesh.
psi = comoving displacement [Mpc/h] from the Lagrangian grid q = (i, j, k) * boxsize / res;
mom = d psi / d D (D = linear growth normalised to 1 at a = 1).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from functools import partial

import numpy as np

from .runtime import init_distributed, log


class HaloOverflowError(RuntimeError):
    """Particles moved further than the halo exchanged between neighbouring GPUs; results would be wrong."""


@dataclass
class SimConfig:
    boxsize: float                      # Mpc/h
    res: int                            # particles per side
    res_pm: int                         # PM mesh cells per side (integer multiple of res)
    cosmo: dict = field(default_factory=lambda: dict(Omega_m=0.3175, Omega_b=0.049, h=0.6711, n_s=0.9624,
                                                     sigma8=0.834))
    # Halo (PM cells) exchanged between neighbouring slabs. Usable displacement = halo/2 - worder//2 - safety
    # (96 -> 45 cells). Low-Om / high-s8 cosmologies at z ~ 0 can need 240 (usable 117), at ~+5 % memory.
    halo_size: int = 96
    halo_safety_cells: float = 2.0
    worder: int = 2                     # mass assignment order (2 = CIC)
    grad_order: int = 4                 # PM gradient kernel order
    lap_order: int = 0                  # PM Laplacian kernel order (0 = exact -1/k^2)
    lpt_grad_order: int = 0             # gradient kernel in 2LPT (0 = exact ik)
    stepper: str = "bullfrog"           # 'bullfrog' or 'fastpm' (DKD Pi-integrators, time variable D)
    pm_fft_backend: str = "JAX_RFFT"    # slab real FFT; the memory/speed optimised path. 'JAX' = jaxdecomp pencils
    ic_fft_backend: str = "JAX"         # jaxdecomp backend for IC / 2LPT FFTs

    def __post_init__(self):
        if self.res_pm % self.res:
            raise ValueError(f"res_pm={self.res_pm} must be a multiple of res={self.res}")
        c = dict(self.cosmo)
        if "Omega_m" in c:              # discodj_dist wants Omega_c
            om = c.pop("Omega_m")
            c.setdefault("Omega_c", om - c.get("Omega_b", 0.0))
        self.cosmo = c

    @property
    def halo_usable_cells(self) -> float:
        return self.halo_size / 2.0 - (self.worder // 2) - self.halo_safety_cells


def _best_pdims(n: int, res: int):
    best = None
    for a in range(1, int(n ** 0.5) + 1):
        if n % a == 0 and res % a == 0 and res % (n // a) == 0:
            best = (a, n // a)
    if best is None:
        raise ValueError(f"no 2-D device mesh for {n} devices with both factors dividing res={res}")
    return best


class Simulation:
    """One box configuration + cosmology on a fixed set of devices; reuse it for many realisations / ICs.

    A different cosmology needs a new Simulation (growth tables and step coefficients are built from it).
    """

    def __init__(self, cfg: SimConfig, devices=None, verbose: bool = True):
        init_distributed(verbose=verbose)
        import jax
        from jax.experimental.mesh_utils import create_device_mesh
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from discodj_dist import DiscoDJ

        self.cfg = cfg
        self.verbose = verbose
        devs = list(devices) if devices is not None else jax.devices()
        n = len(devs)
        self.n_dev = n
        if cfg.pm_fft_backend == "JAX_RFFT" and (cfg.res % n or cfg.res_pm % n):
            raise ValueError(f"slab PM needs res={cfg.res} and res_pm={cfg.res_pm} divisible by n_dev={n}")
        self.lpt_pdims = (1, 1) if n == 1 else _best_pdims(n, cfg.res)
        self.pm_pdims = (n, 1) if cfg.pm_fft_backend == "JAX_RFFT" else self.lpt_pdims
        kw = {} if devices is None else dict(devices=devs)
        self.mesh = Mesh(create_device_mesh(self.lpt_pdims, **kw), axis_names=("x", "y"))
        self.sharding_field = NamedSharding(self.mesh, P("x", "y", None))
        self.sharding_disp = NamedSharding(self.mesh, P("x", "y", None, None))
        if self.pm_pdims == self.lpt_pdims:
            self.pm_mesh = self.mesh
        else:
            self.pm_mesh = Mesh(create_device_mesh(self.pm_pdims, **kw), axis_names=("pm_x", "pm_y"))
        ax = self.pm_mesh.axis_names
        self.sharding_pm_disp = NamedSharding(self.pm_mesh, P(ax[0], ax[1], None, None))
        self.cell = cfg.boxsize / cfg.res_pm
        if cfg.halo_usable_cells <= 0:
            raise ValueError(f"halo_size={cfg.halo_size} too small")

        platform = devs[0].platform
        self.dj = DiscoDJ(dim=3, res=cfg.res, device=None if jax.process_count() > 1 else platform,
                          precision="single", boxsize=cfg.boxsize, cosmo=cfg.cosmo, name="discodj_mgpu").with_timetables()
        self.cosmology = self.dj.cosmo
        self._step = self._build_step()
        if verbose:
            log(f"Simulation: box {cfg.boxsize} Mpc/h, {cfg.res}^3 particles, {cfg.res_pm}^3 PM mesh "
                f"(cell {self.cell:.4f} Mpc/h), {n} devices, LPT mesh {self.lpt_pdims}, PM mesh {self.pm_pdims}, "
                f"halo {cfg.halo_size} (usable {cfg.halo_usable_cells:.1f} cells), stepper {cfg.stepper}")

    # ------------------------------------------------------------------ cosmology helpers
    def growth(self, a: float) -> float:
        """Linear growth D+(a), normalised to 1 at a = 1."""
        tt = self.cosmology._timetables
        return float(np.interp(np.log10(a), np.log10(np.asarray(tt["a"])), np.asarray(tt["Dplus"])))

    def growth_rate_factor(self, a: float) -> float:
        """discodj Fplus(a); peculiar velocity [km/s] = mom * Fplus(a) / a * 100."""
        return float(self.cosmology.Fplus(a))

    # ------------------------------------------------------------------ initial conditions
    def gaussian_delta(self, pk, seed: int = 0):
        """Gaussian linear density field at z = 0 on the (res)^3 particle grid, sharded on the LPT mesh.

        ``pk``: (k [h/Mpc], P(k) [(Mpc/h)^3]) linear z = 0 table: a 2-column array, a (k, P) tuple or a text file.
        White noise = jax.random.normal(PRNGKey(seed)); it is partitionable, so the same seed gives the same field
        on any number of GPUs (it is NOT the N-GenIC / 2LPTic random sequence).
        """
        import jax
        import jax.numpy as jnp
        import jaxdecomp
        from jax import lax
        from discodj_dist.lpt.nlpt_distributed import build_k_vecs_dist

        if isinstance(pk, (str, os.PathLike)):
            pk = np.loadtxt(pk)
        if isinstance(pk, (tuple, list)):
            k_tab, p_tab = (np.asarray(x, np.float64) for x in pk)
        else:
            k_tab, p_tab = np.asarray(pk)[:, 0], np.asarray(pk)[:, 1]
        lk = jnp.asarray(np.log(k_tab), jnp.float32)
        lp = jnp.asarray(np.log(p_tab), jnp.float32)
        res, box, sf, backend = self.cfg.res, self.cfg.boxsize, self.sharding_field, self.cfg.ic_fft_backend

        @partial(jax.jit, out_shardings=sf)
        def _make(key):
            w = lax.with_sharding_constraint(jax.random.normal(key, (res,) * 3, jnp.float32), sf)
            wk = jaxdecomp.pfft3d(w.astype(jnp.complex64), norm="backward", backend=backend)
            kk = jnp.sqrt(sum(ki ** 2 for ki in build_k_vecs_dist(wk, boxsize=box, res=res)))
            # <|w_k|^2> = N^3 for unit white noise and <|delta_k|^2> = P(k) N^6 / V
            amp = jnp.where(kk > 0, jnp.exp(0.5 * jnp.interp(jnp.log(jnp.maximum(kk, 1e-30)), lk, lp)), 0.0)
            amp = amp * np.float32(np.sqrt(res ** 3 / box ** 3))
            return jaxdecomp.pifft3d(wk * amp, norm="backward", backend=backend).real.astype(jnp.float32)

        return _make(jax.random.PRNGKey(seed))

    def delta_from_array(self, arr, a: float | None = None):
        """Shard an external (res)^3 linear density field (array or .npy path, any float dtype).

        ``a``: scale factor the field is given at (e.g. 1/128 for a z = 127 2LPTic field); it is divided by D+(a)
        to extrapolate it to z = 0. ``a=None`` means it is already at z = 0. A .npy path is memory-mapped and each
        process reads only its own shards.
        """
        import jax
        if isinstance(arr, (str, os.PathLike)):
            arr = np.load(arr, mmap_mode="r")
        if arr.shape != (self.cfg.res,) * 3:
            raise ValueError(f"field shape {arr.shape} != {(self.cfg.res,) * 3}")
        scale = np.float32(1.0 if a is None else 1.0 / self.growth(a))
        return jax.make_array_from_callback(
            arr.shape, self.sharding_field,
            lambda idx: np.ascontiguousarray(np.asarray(arr[idx], dtype=np.float32) * scale))

    def lpt2(self, delta0, a: float):
        """2LPT displacement and momentum at scale factor ``a`` from the z = 0 linear field (consumes ``delta0``).

        Returns (psi, mom) of shape (res, res, res, 3) sharded on the PM mesh.
        """
        import jax
        import jax.numpy as jnp
        import jaxdecomp
        from jax import lax
        from discodj_dist.lpt.nlpt_distributed import build_k_vecs_dist, compute_2lpt_initial_state_distributed

        cfg, sf = self.cfg, self.sharding_field

        @partial(jax.jit, donate_argnums=(0,))
        def _fphi(delta):
            fd = lax.with_sharding_constraint(
                jaxdecomp.pfft3d(delta.astype(jnp.complex64), norm="backward", backend=cfg.ic_fft_backend), sf)
            k2 = sum(ki ** 2 for ki in build_k_vecs_dist(fd, boxsize=cfg.boxsize, res=cfg.res))
            mask = (k2 != 0).astype(jnp.float32)
            k2 = lax.with_sharding_constraint(k2.at[0, 0, 0].set(jnp.float32(1.0)), sf)
            return lax.with_sharding_constraint((-1.0 / k2) * mask * fd, sf)

        fphi = _fphi(delta0)
        psi, mom = compute_2lpt_initial_state_distributed(
            fphi, res=cfg.res, boxsize=cfg.boxsize, Dplus=self.growth(a), grad_kernel_order=cfg.lpt_grad_order,
            dtype_num=32, dtype_c_num=64, no_factors=False, field_sharding=sf, disp_sharding=self.sharding_disp,
            fft_sharding=getattr(fphi, "sharding", None), fft_backend=cfg.ic_fft_backend,
            mu2_fft_backend=cfg.ic_fft_backend)
        del fphi
        if self.pm_pdims != self.lpt_pdims:
            spd = self.sharding_pm_disp
            psi, mom = jax.jit(lambda p, m: (lax.with_sharding_constraint(p, spd), lax.with_sharding_constraint(m, spd)),
                               donate_argnums=(0, 1), out_shardings=(spd, spd))(psi, mom)
        psi.block_until_ready()
        return psi, mom

    # ------------------------------------------------------------------ N-body
    def _build_step(self):
        import jax
        import jax.numpy as jnp
        from discodj_dist.nbody.acc_distributed import kick_PM_distributed

        cfg, spd, cell = self.cfg, self.sharding_pm_disp, self.cell

        @partial(jax.jit, donate_argnums=(0, 1))
        def step(psi, mom, coeffs):
            d1, d2, alpha, beta = coeffs
            p = psi + d1 * mom                                       # drift
            maxd = jnp.max(jnp.abs(p / cell), axis=(0, 1, 2))      # |displacement| per axis in PM cells
            m = kick_PM_distributed(p, mom, alpha=alpha, beta=beta, dim=3, res_pm=cfg.res_pm, boxsize=cfg.boxsize,
                                    halo_size=cfg.halo_size, sharding=spd, grad_order=cfg.grad_order,
                                    lap_order=cfg.lap_order, dtype_num=32, worder=cfg.worder, deconvolve=False,
                                    fft_backend=cfg.pm_fft_backend)
            return p + d2 * m, m, maxd                               # drift

        return step

    def step_coefficients(self, a_ini: float, a_end: float, n_steps: int) -> np.ndarray:
        """(n_steps, 4) float32 array of (drift1, drift2, alpha, beta) for the configured DKD stepper."""
        from discodj_dist.nbody.steppers.dkd_pi_integrator import DKDPiIntegrator
        solver = DKDPiIntegrator(cosmo=self.cosmology,
                                 time_dict={"n_steps": n_steps, "a_ini": a_ini, "a_end": a_end, "time_var": "D"},
                                 integrator_name=self.cfg.stepper, use_diffrax=False, dtype_num=32)
        ia = solver.get_integrator_args()
        return np.stack([np.asarray(ia[k], np.float32) for k in ("ddrift1", "ddrift2", "alpha", "beta")], axis=1)

    def evolve(self, psi, mom, a_ini: float, a_end: float, n_steps: int, check_halo: str = "raise",
               verbose_steps: bool = False):
        """``n_steps`` PM steps from ``a_ini`` to ``a_end`` (consumes psi, mom). Returns (psi, mom, info).

        check_halo: 'raise' (HaloOverflowError), 'warn' or 'off'. The check syncs once per step (negligible).
        To write several snapshots, call evolve once per segment (a_ini -> a_1 -> a_2 ...).
        """
        import jax.numpy as jnp
        coeffs = self.step_coefficients(a_ini, a_end, n_steps)
        sharded_axes = [i for i, q in enumerate(self.pm_pdims) if q > 1]
        usable = self.cfg.halo_usable_cells
        worst, t0 = 0.0, time.perf_counter()
        for i in range(n_steps):
            psi, mom, maxd = self._step(psi, mom, jnp.asarray(coeffs[i]))
            if check_halo != "off" or verbose_steps:
                md = np.asarray(maxd)
                worst = max([worst] + [float(md[a]) for a in sharded_axes])
                if verbose_steps:
                    log(f"  step {i + 1}/{n_steps}: max |disp| per axis = {np.round(md, 2).tolist()} PM cells")
                if check_halo != "off" and worst > usable:
                    msg = (f"max displacement {worst:.1f} PM cells along a sharded axis exceeds the usable halo "
                           f"{usable:.1f} (halo_size={self.cfg.halo_size}); increase SimConfig.halo_size")
                    if check_halo == "raise":
                        raise HaloOverflowError(msg)
                    log("WARNING: " + msg)
        psi.block_until_ready()
        info = dict(seconds=time.perf_counter() - t0, max_sharded_disp_cells=worst, usable_halo_cells=usable,
                    a_ini=a_ini, a_end=a_end, n_steps=n_steps)
        if self.verbose:
            log(f"evolve a={a_ini:.5f}->{a_end:.5f}, {n_steps} steps: {info['seconds']:.1f} s, "
                f"max sharded disp {worst:.1f} / {usable:.1f} cells")
        return psi, mom, info

    # ------------------------------------------------------------------ outputs
    def positions(self, psi):
        """Comoving particle positions [Mpc/h] in [0, boxsize), same sharding as psi."""
        import jax
        import jax.numpy as jnp
        from discodj_dist.core.distributed_pm import get_local_q_grid
        cfg = self.cfg
        q = get_local_q_grid(res_pm=cfg.res, boxsize=cfg.boxsize, sharding=self.sharding_pm_disp, dtype=jnp.float32)
        return jax.jit(lambda p, q: jnp.mod(p + q, cfg.boxsize))(psi, q)

    def velocities(self, mom, a: float):
        """Peculiar velocities [km/s] (pm.py convention v = mom * Fplus(a) / a * 100)."""
        import jax
        f = np.float32(self.growth_rate_factor(a) / a * 100.0)
        return jax.jit(lambda m: m * f)(mom)

    def power_spectrum(self, psi, save: str | None = None, extra: dict | None = None) -> dict:
        """Matter P(k) of the particles: CIC on the PM mesh, CIC window deconvolved, bins of width k_f.

        Returns dict(k, pk, nmodes, shot) (numpy, identical on every process); ``pk`` is NOT shot-noise subtracted
        (shot = boxsize^3 / res^3). Needs the slab PM mesh. ``save``: .npz path written by process 0.
        """
        from .pk import measure_pk
        out = measure_pk(self, psi)
        if save and __import__("jax").process_index() == 0:
            os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
            np.savez(save, **out, box=self.cfg.boxsize, res=self.cfg.res, res_pm=self.cfg.res_pm, **(extra or {}))
        return out

    def save_local_shards(self, arr, prefix: str, dtype=None) -> str:
        """Each process writes the shards it holds to ``{prefix}.proc{p}.npz`` (keys data_i, start_i).

        No cross-node gather; reassemble with :func:`discodj_mgpu.shards.load_shards` for small runs.
        """
        import jax
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        payload = {}
        for i, s in enumerate(arr.addressable_shards):
            d = np.asarray(s.data)
            payload[f"data_{i}"] = d if dtype is None else d.astype(dtype)
            payload[f"start_{i}"] = np.array([sl.start or 0 for sl in s.index], np.int64)
        fn = f"{prefix}.proc{jax.process_index()}.npz"
        np.savez(fn, shape=np.array(arr.shape), **payload)
        return fn

    def describe(self) -> dict:
        d = asdict(self.cfg)
        d.update(n_dev=self.n_dev, lpt_pdims=self.lpt_pdims, pm_pdims=self.pm_pdims, cell=self.cell,
                 halo_usable_cells=self.cfg.halo_usable_cells)
        return d
