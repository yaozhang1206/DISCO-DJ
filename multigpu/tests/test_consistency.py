"""Self-checks of the kit. Exit code 0 = all passed.

  CPU (any machine, 4 fake devices, ~1-2 min):   python tests/test_consistency.py --cpu
  GPU (one 4-GPU node):                          $PY tests/test_consistency.py --res 256 --res-pm 512 --box 500

Checks
 1. PM kick: memory-saving code paths (force-component loop, particle chunks) == plain path (float32 roundoff)
 2. Gaussian ICs on 1 device == on all devices (partitionable RNG; FFT roundoff only)
 3. full pipeline (2LPT + PM steps to z=0) on 1 device vs all devices: max |d psi| in PM cells
 4. distributed P(k) == numpy CIC P(k) of the gathered particles
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true", help="run on 4 fake CPU devices")
ap.add_argument("--res", type=int, default=32)
ap.add_argument("--res-pm", type=int, default=64)
ap.add_argument("--box", type=float, default=200.0)
ap.add_argument("--halo", type=int, default=16)
ap.add_argument("--steps", type=int, default=5)
args = ap.parse_args()
if args.cpu:
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np                                          # noqa: E402
from discodj_mgpu import init_distributed, SimConfig, Simulation, gather_to_host  # noqa: E402

init_distributed()
import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

FAIL = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def toy_pk(k):
    """Smooth CDM-like linear P(k) at z=0 (sigma8 ~ 0.8), only for tests."""
    k0 = 0.02
    return 2.2e4 * (k / k0) / (1 + (k / k0) ** 2) ** 1.4


kt = np.logspace(-4, 2, 400)
PK = np.stack([kt, toy_pk(kt)], 1)
cfg = SimConfig(boxsize=args.box, res=args.res, res_pm=args.res_pm, halo_size=args.halo)
sim_n = Simulation(cfg)
sim_1 = Simulation(cfg, devices=jax.devices()[:1], verbose=False)
nd = sim_n.n_dev
cell = sim_n.cell

# 1. kick variants ------------------------------------------------------------------------------------------
from discodj_dist.nbody.acc_distributed import kick_PM_distributed  # noqa: E402
k1, k2 = jax.random.split(jax.random.PRNGKey(1))
shape = (args.res,) * 3 + (3,)
psi0 = jax.device_put(np.asarray(2.5 * cell * jax.random.normal(k1, shape, jnp.float32)), sim_n.sharding_pm_disp)
mom0 = jax.device_put(np.asarray(jax.random.normal(k2, shape, jnp.float32)), sim_n.sharding_pm_disp)


def kick(env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:   # env toggles are read at trace time -> fresh jit
        f = jax.jit(lambda p, m: kick_PM_distributed(p, m, alpha=0.9, beta=1.3, dim=3, res_pm=args.res_pm,
                    boxsize=args.box, halo_size=args.halo, sharding=sim_n.sharding_pm_disp, grad_order=4,
                    lap_order=0, dtype_num=32, worder=2, deconvolve=False, fft_backend="JAX_RFFT"))
        return np.asarray(gather_to_host(f(psi0, mom0)))
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


ref = kick({"DISCO_PM_FORCE_LOOP": "0", "DISCO_PM_PART_CHUNK": "0", "DISCO_PM_STENCIL_LOOP": "0"})
scale = np.abs(ref).max()
small_chunk = str(max(1, args.res * args.res * 2))          # forces several particle chunks per GPU
for name, env in [("default (force loop + stencil loop)", {}),
                  ("particle chunks", {"DISCO_PM_PART_CHUNK": small_chunk})]:
    d = np.abs(kick(env) - ref).max() / scale
    check(f"kick: {name} vs plain", d < 1e-5, f"max rel diff {d:.2e}")

# 2. ICs on 1 vs N devices ------------------------------------------------------------------------------------
dn = gather_to_host(sim_n.gaussian_delta(PK, seed=7))
d1 = gather_to_host(sim_1.gaussian_delta(PK, seed=7))
d = np.abs(dn - d1).max() / dn.std()
check(f"gaussian ICs 1 vs {nd} devices", d < 1e-4, f"max diff / std = {d:.2e} (std {dn.std():.3f})")

# 3. pipeline 1 vs N devices -------------------------------------------------------------------------------
a_ini = 1 / 128
out = {}
for tag, sim in (("n", sim_n), ("1", sim_1)):
    psi, mom = sim.lpt2(sim.gaussian_delta(PK, seed=7), a=a_ini)
    p_ic = gather_to_host(psi)
    psi, mom, info = sim.evolve(psi, mom, a_ini, 1.0, args.steps)
    out[tag] = (p_ic, gather_to_host(psi), info, psi, sim)
dic = np.abs(out["n"][0] - out["1"][0]).max() / cell
dz0 = np.abs(out["n"][1] - out["1"][1]).max() / cell
dz0_rms = np.sqrt(((out["n"][1] - out["1"][1]) ** 2).mean()) / cell
rms = np.sqrt((out["1"][1] ** 2).mean()) / cell
check(f"2LPT ICs 1 vs {nd} devices", dic < 1e-3, f"max |d psi| = {dic:.2e} PM cells")
# float32 roundoff (different FFT decompositions) grows chaotically in the densest regions: bound the rms tightly
check(f"z=0 after {args.steps} steps 1 vs {nd} devices", dz0_rms < 1e-3 and dz0 < 0.25,
      f"rms |d psi| = {dz0_rms:.2e}, max = {dz0:.2e} PM cells (rms psi {rms:.2f} cells; max disp {out['n'][2]['max_sharded_disp_cells']:.1f})")

# 4. P(k) vs numpy ------------------------------------------------------------------------------------------
psi_n, sim = out["n"][3], out["n"][4]
pk = sim.power_spectrum(psi_n)
if args.res_pm <= 256:
    N, L = args.res_pm, args.box
    pos = gather_to_host(sim.positions(psi_n)).reshape(-1, 3).astype(np.float64) / (L / N)
    i0 = np.floor(pos).astype(np.int64); f = pos - i0
    rho = np.zeros((N,) * 3)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                np.add.at(rho, ((i0[:, 0] + dx) % N, (i0[:, 1] + dy) % N, (i0[:, 2] + dz) % N), w)
    delta = rho / rho.mean() - 1
    fk = np.fft.rfftn(delta)
    ii = np.fft.fftfreq(N, 1 / N); iz = np.arange(N // 2 + 1)
    s2 = lambda i: np.where(i == 0, 1, np.sinc(i / N) ** 2)
    W = (s2(ii)[:, None, None] * s2(ii)[None, :, None] * s2(iz)[None, None, :]) ** 2
    kn = np.sqrt(ii[:, None, None] ** 2 + ii[None, :, None] ** 2 + iz[None, None, :] ** 2)
    mult = np.where((iz == 0) | (iz == N // 2), 1, 2)[None, None, :] * np.ones_like(kn)
    b = np.floor(kn + 0.5).astype(int).ravel()
    P = np.bincount(b, (np.abs(fk) ** 2 / W * mult).ravel()); M = np.bincount(b, mult.ravel())
    ok = M > 0
    pk_np = P[ok] / M[ok] * L ** 3 / N ** 6
    sel = slice(1, min(len(pk_np), len(pk["pk"])) - 1)
    d = np.abs(pk["pk"][sel] / pk_np[sel] - 1).max()
    check("P(k) distributed vs numpy", d < 1e-3, f"max rel diff {d:.2e} over {len(pk_np[sel])} bins")
else:
    print("[SKIP] P(k) vs numpy (res_pm > 256)")

print("ALL PASSED" if not FAIL else f"FAILED: {FAIL}", flush=True)
sys.exit(1 if FAIL else 0)
