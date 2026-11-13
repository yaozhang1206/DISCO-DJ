"""Run one box: ICs -> 2LPT -> PM steps to each output redshift; save P(k) (+ optional particle shards).

Examples (inside an allocation, env from env/setup_delta.sh; one process per node):
  # 1 Gpc/h, 1024^3 particles on a 1024^3 mesh, Gaussian ICs from a linear P(k) table, 20 bullfrog steps to z=0
  $PY examples/run_sim.py --box 1000 --res 1024 --res-pm 1024 --pk-table Pk_lin_z0.txt --seed 1 --out runs/c1
  # 2 snapshots (z = 1 after 10 steps, z = 0.5 after 5 more), fastpm stepper, write positions/velocities
  $PY examples/run_sim.py ... --z-out 1,0.5 --steps 10,5 --stepper fastpm --save-particles
  # external IC: 2LPTic/N-GenIC style density at z=127 (res^3 .npy), rescaled to z=0 internally
  $PY examples/run_sim.py ... --ic-npy IC_delta.npy --ic-a 0.0078125
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from discodj_mgpu import init_distributed, SimConfig, Simulation, log, gpu_memory_gb  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", type=float, required=True, help="box size [Mpc/h]")
    ap.add_argument("--res", type=int, required=True, help="particles per side")
    ap.add_argument("--res-pm", type=int, required=True, help="PM mesh per side (multiple of --res)")
    ap.add_argument("--cosmo", type=float, nargs=5, default=[0.3175, 0.049, 0.6711, 0.9624, 0.834],
                    metavar=("Om", "Ob", "h", "ns", "s8"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pk-table", help="linear z=0 P(k) text file, columns k [h/Mpc], P [(Mpc/h)^3]")
    g.add_argument("--ic-npy", help="res^3 linear density field (.npy) at scale factor --ic-a")
    ap.add_argument("--ic-a", type=float, default=None, help="scale factor of --ic-npy (omit if already at z=0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--a-ini", type=float, default=1 / 128, help="scale factor of the 2LPT ICs (default z=127)")
    ap.add_argument("--z-out", default="0", help="comma list of output redshifts (decreasing)")
    ap.add_argument("--steps", default="20", help="comma list: PM steps for each segment to the next output")
    ap.add_argument("--stepper", default="bullfrog", choices=["bullfrog", "fastpm"])
    ap.add_argument("--halo", type=int, default=96, help="halo size in PM cells (usable = halo/2 - 3)")
    ap.add_argument("--no-pk", action="store_true", help="skip P(k) at the outputs")
    ap.add_argument("--pk-ic", action="store_true", help="also measure P(k) of the 2LPT ICs")
    ap.add_argument("--save-particles", action="store_true", help="write per-process pos/vel shard files")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    init_distributed()
    om, ob, h, ns, s8 = a.cosmo
    cfg = SimConfig(boxsize=a.box, res=a.res, res_pm=a.res_pm, halo_size=a.halo, stepper=a.stepper,
                    cosmo=dict(Omega_m=om, Omega_b=ob, h=h, n_s=ns, sigma8=s8))
    sim = Simulation(cfg)
    z_out = [float(z) for z in a.z_out.split(",")]
    steps = [int(n) for n in a.steps.split(",")]
    if len(z_out) != len(steps):
        raise SystemExit("--z-out and --steps need the same number of entries")
    os.makedirs(a.out, exist_ok=True)
    tag = f"res{a.res}_pm{a.res_pm}_box{a.box:g}_seed{a.seed}"
    meta = dict(config=sim.describe(), args=vars(a), outputs=[], timing={})

    t0 = time.perf_counter()
    delta = sim.gaussian_delta(a.pk_table, seed=a.seed) if a.pk_table else sim.delta_from_array(a.ic_npy, a=a.ic_a)
    psi, mom = sim.lpt2(delta, a=a.a_ini)
    del delta
    meta["timing"]["ic+2lpt"] = time.perf_counter() - t0
    log(f"ICs + 2LPT at a={a.a_ini:.5f}: {meta['timing']['ic+2lpt']:.1f} s")
    if a.pk_ic and not a.no_pk:
        sim.power_spectrum(psi, save=f"{a.out}/pk_ic_{tag}.npz", extra=dict(a=a.a_ini, dplus=sim.growth(a.a_ini)))

    a_cur = a.a_ini
    for z, n in zip(z_out, steps):
        a_next = 1.0 / (1.0 + z)
        psi, mom, info = sim.evolve(psi, mom, a_cur, a_next, n)
        info["z"] = z
        if not a.no_pk:
            t = time.perf_counter()
            pk = sim.power_spectrum(psi, save=f"{a.out}/pk_z{z:g}_{tag}.npz", extra=dict(a=a_next, z=z))
            info["pk_seconds"] = time.perf_counter() - t
            log(f"z={z:g}: P(k) saved, P(k~0.1) = {float(__import__('numpy').interp(0.1, pk['k'], pk['pk'])):.1f}")
        if a.save_particles:
            sim.save_local_shards(sim.positions(psi), f"{a.out}/pos_z{z:g}_{tag}")
            sim.save_local_shards(sim.velocities(mom, a_next), f"{a.out}/vel_z{z:g}_{tag}")
        meta["outputs"].append(info)
        a_cur = a_next
    meta["timing"]["total"] = time.perf_counter() - t0
    meta["peak_gpu_gb"] = gpu_memory_gb()
    if __import__("jax").process_index() == 0:
        with open(f"{a.out}/meta_{tag}.json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
    log(f"done in {meta['timing']['total']:.1f} s -> {a.out}")


if __name__ == "__main__":
    main()
