# DISCO-DJ multi-GPU kit (`multigpu/`)

Multi-GPU and multi-node particle-mesh (PM) N-body simulations with DISCO-DJ's distributed code (`discodj_dist`).
The pipeline is: Gaussian or external initial conditions → distributed 2LPT → DKD PM steps (BullFrog or FastPM)
→ P(k), positions and velocities. The large arrays stay sharded across GPUs for the whole run.

The kit has two parts, both in this fork of `shivampcosmo/DISCO-DJ` (branch `multigpu-kit`):

- **`src/discodj_dist`**: `discodj_dist` from `shivampcosmo/DISCO-DJ` at commit `f264894`, with two correctness
  fixes and memory optimisations ([details below](#changes-to-discodj_dist-vs-upstream-f264894)). The
  optimisations let large boxes fit on far fewer GPUs, for example 3 Gpc/h with 1536³ particles on a 3072³ mesh on
  8 GH200s.
- **`src/discodj_mgpu`**: a small driver that hides the device mesh, sharding and `jax.distributed` setup. Use it from
  your own project instead of copying a monolithic script.

Validated on NCSA Delta GH200 nodes (4 × 96 GB GPUs per node, Slingshot), with JAX 0.6.2 and jaxdecomp 0.2.8.

```
DISCO-DJ/                    (this fork, branch multigpu-kit)
├── src/
│   ├── discodj_dist/        modified DISCO-DJ distributed package (import name unchanged)
│   ├── discodj_native/      upstream; falls back to a pure-python shim (the C++ extension is not needed here)
│   ├── discodj_mgpu/        driver: runtime.py (process/GPU setup), simulation.py (SimConfig, Simulation),
│   │                        pk.py (distributed P(k)), shards.py (per-process output files)
│   └── ...                  upstream discodj_demo, logo
└── multigpu/                this kit
    ├── README.md
    ├── examples/
    │   ├── run_sim.py       command-line driver: ICs -> 2LPT -> PM -> P(k) / particles at each output z
    │   └── plot_pk.py       plot P(k) files against linear theory
    ├── tests/test_consistency.py   self-checks (CPU fake devices or GPUs)
    ├── jobs/
    │   ├── run_sim.sbatch   generic Slurm launcher (1 process per node, 4 GPUs each)
    │   └── test_kit.sbatch  self-test on one GPU node
    └── env/
        ├── setup_delta.sh   module purge/load, NCCL network plugin, PYTHONPATH (= ../src), XLA memory settings
        ├── delta_ofi_env.sh aws-ofi-nccl + libfabric (cxi) for multi-node NCCL on Slingshot
        ├── requirements.txt pinned packages that were tested
        └── pip_freeze_pm.txt full `pip freeze` of the tested conda env
```

Every change to `discodj_dist` relative to upstream is one commit on this branch:
`git diff f264894 -- src/discodj_dist`. DISCO-DJ is GPL-3.0 (`../LICENSE`).

---

## 1. Requirements

**Hardware**

- NVIDIA GPUs with CUDA 12 support.
- Memory sizing assumes about 96 GB per GPU (GH200); see [§5](#5-sizing-how-many-gpus).
- For more than one node, a fast interconnect with an NCCL network plugin. Without one, NCCL falls back to TCP
  sockets, which is 4–5× slower on 2 nodes.

**Software** (tested versions in `env/requirements.txt`)

| Package | Version | Note |
|---|---|---|
| python | 3.10 | |
| jax / jaxlib / jax-cuda12-plugin | 0.6.2 | `pip install "jax[cuda12]==0.6.2"`; brings the CUDA, cuFFT and NCCL wheels |
| jaxdecomp | 0.2.8 | only the JAX backend is used (no cuDecomp build needed) |
| diffrax, equinox, lineax, optimistix, jaxtyping | 0.7.0, 0.13.2, 0.0.8, 0.0.11, 0.3.3 | imported by `discodj_dist` |
| numpy, scipy, einops, typing_extensions | 2.2.6, 1.15.3, 0.8.1, 4.15.0 | |
| matplotlib | any | `examples/plot_pk.py` only |

Nothing needs to be compiled. The kit is used through `PYTHONPATH=<kit>/src`, with no `pip install -e`.

### Using the existing environment on Delta

The tested env is `/u/yzhang116/.conda/envs/pm`. `env/setup_delta.sh` uses it unless you set `DISCODJ_PY` to
another python.

### Making a new environment

```bash
conda create -n discodj python=3.10 -y && conda activate discodj
pip install -r /path/to/DISCO-DJ/multigpu/env/requirements.txt
export DISCODJ_PY=$(which python)        # tell env/setup_delta.sh to use it
```

### Environment inside a job (Delta)

`source env/setup_delta.sh` does four things:

1. **Module reset.** Runs `module purge; module load cuda/12.4.0 cudnn/9.3.0.75 gsl`. Delta's login-default
   modules put HPC-SDK CUDA 12.9 and NCCL on `LD_LIBRARY_PATH`, which shadows JAX's bundled libraries and crashes
   at the first collective with `free(): double free detected`.
2. **NCCL network plugin** (`env/delta_ofi_env.sh`):
   - On more than one node it adds aws-ofi-nccl 1.18 and Cray libfabric (cxi), plus Delta's `FI_CXI_*` and
     `NCCL_*` settings, so NCCL uses Slingshot.
   - On one node it is off (`OFI_PLUGIN=none`), because GPUs on a node talk over NVLink. Override with
     `OFI_PLUGIN=1.18|1.14|none`.
3. **Kit on the path.** Sets `PYTHONPATH=<kit>/src` and `PY=<env python>`.
4. **JAX memory settings:**
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
   - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97`: JAX otherwise caps itself at 75 % of the GPU.
   - `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: the default BFC allocator fragments after 2LPT and fails near the
     memory limit, for example on the 3072³ mesh on 8 GPUs. The platform allocator showed no slowdown here.
     `init_distributed()` sets the same defaults when they are unset.

**Other clusters:** replace step 1 with whatever gives a clean environment (JAX needs no system CUDA). Replace
step 2 with your site's NCCL network plugin. Keep steps 3 and 4.

---

## 2. Quick start

```bash
git clone -b multigpu-kit git@github.com:yaozhang1206/DISCO-DJ.git   # once
KIT=/u/yzhang116/DISCO-DJ-multigpu/multigpu        # this folder; jobs/*.sbatch default to it, or export KIT

# 1) Self-test on CPU (any machine, about 1 min). Also run before/after changing discodj_dist.
cd $KIT && JAX_PLATFORMS=cpu python tests/test_consistency.py --cpu

# 2) Self-test on one GPU node (1 GPU vs 4 GPUs, P(k) vs numpy)
sbatch $KIT/jobs/test_kit.sbatch

# 3) A run: 2 Gpc/h, 1024^3 particles, 2048^3 mesh, 20 BullFrog steps z=127 -> 0, on 1 node / 4 GPUs (~1 min)
sbatch -N 1 $KIT/jobs/run_sim.sbatch --box 2000 --res 1024 --res-pm 2048 \
    --cosmo 0.3175 0.049 0.6711 0.9624 0.834 --pk-table /path/Pk_lin_z0.txt --seed 1 --pk-ic --out $PWD/runs/c2

# 4) 3 Gpc/h, 1536^3 / 3072^3 on 2 nodes / 8 GPUs (~3 min)
sbatch -N 2 -t 00:30:00 $KIT/jobs/run_sim.sbatch --box 3000 --res 1536 --res-pm 3072 ... --out $PWD/runs/c4

# 5) Plot
python $KIT/examples/plot_pk.py runs/*/pk_z0_*.npz --linear /path/Pk_lin_z0.txt --out pk.png
```

**Changing the account or partition:** `jobs/*.sbatch` default to account `bdne-dtai-gh`, partition `ghx4`. Override
on the command line (`sbatch -A <acct> -p <partition> -t <time> ...`). If the kit moves, set `KIT=` in the script or
the environment (`sbatch --export=ALL,KIT=/new/path`).

### `examples/run_sim.py` options

| Option | Meaning |
|---|---|
| `--box`, `--res`, `--res-pm` | box size [Mpc/h], particles per side, PM mesh per side (a multiple of `res`) |
| `--cosmo Om Ob h ns s8` | flat ΛCDM parameters |
| `--pk-table FILE` | Gaussian ICs from a linear z=0 P(k) table (columns: k [h/Mpc], P [(Mpc/h)³]), seeded with `--seed` |
| `--ic-npy FILE --ic-a A` | instead: external res³ linear density at scale factor A (for example a 2LPTic field at z=127, A=1/128) |
| `--a-ini` | scale factor of the 2LPT ICs (default 1/128) |
| `--z-out 1,0.5 --steps 10,5` | output redshifts and PM steps in each segment |
| `--stepper` | `bullfrog` (default) or `fastpm` |
| `--halo` | halo width in PM cells (default 96, usable displacement 45 cells) |
| `--pk-ic`, `--no-pk` | also measure P(k) of the ICs / skip P(k) |
| `--save-particles` | write positions [Mpc/h] and velocities [km/s] as per-process shard files |

**Outputs in `--out`:**

- `pk_z{z}_res..._seed..npz`: P(k) at each output (k, pk, nmodes, shot, box, res, res_pm).
- `pk_ic_*.npz`: P(k) of the ICs, with `--pk-ic`.
- `meta_*.json`: configuration, timings, maximum displacement and peak memory.
- `pos_*/vel_*.proc{p}.npz`: particle shards, with `--save-particles`.

---

## 3. Using it from your own code

```python
import sys; sys.path.insert(0, "/u/yzhang116/DISCO-DJ-multigpu/src")   # the repo's src/, or PYTHONPATH
from discodj_mgpu import init_distributed, SimConfig, Simulation, HaloOverflowError

init_distributed()                       # FIRST, before any other JAX call (starts jax.distributed on >1 node)
sim = Simulation(SimConfig(boxsize=2000.0, res=1024, res_pm=2048,
                           cosmo=dict(Omega_m=0.3175, Omega_b=0.049, h=0.6711, n_s=0.9624, sigma8=0.834),
                           halo_size=96, stepper="bullfrog"))

delta = sim.gaussian_delta("Pk_lin_z0.txt", seed=42)   # or sim.delta_from_array("ic.npy", a=1/128)
psi, mom = sim.lpt2(delta, a=1/128)                     # consumes delta (buffer donated)
for a0, a1, n in [(1/128, 1/3, 10), (1/3, 1/2, 5), (1/2, 2/3, 5)]:   # snapshots at z = 2, 1, 0.5
    psi, mom, info = sim.evolve(psi, mom, a0, a1, n)                  # raises HaloOverflowError if needed
    pk = sim.power_spectrum(psi, save=f"pk_a{a1:.3f}.npz")            # dict(k, pk, nmodes, shot)
    pos = sim.positions(psi)                  # sharded (res,res,res,3) jax.Array, Mpc/h in [0, box)
    vel = sim.velocities(mom, a1)             # km/s
    # your own analysis on the sharded arrays, or write what each process holds:
    sim.save_local_shards(pos, f"out/pos_a{a1:.3f}")
```

**Conventions**

- `psi` is the comoving displacement [Mpc/h] from the Lagrangian grid `q = (i,j,k)·box/res`.
- `mom = dψ/dD`, with D the linear growth factor normalised to D(a=1)=1.
- Velocity is `mom · Fplus(a) / a · 100` km/s, the same convention as `pm.py`.
- `gaussian_delta` and `delta_from_array` return the linear density field at **z=0**. `delta_from_array(..., a=A)`
  divides by D(A).
- The particle state `psi`, `mom` has global shape (res, res, res, 3), sharded in slabs along x across all GPUs:
  `sim.sharding_pm_disp` on mesh `sim.pm_pdims = (n_dev, 1)`. 2LPT fields use a squarer mesh (`sim.lpt_pdims`).
- `arr.addressable_shards` gives each process its own blocks. `discodj_mgpu.gather_to_host` (all-gather) is only for
  small debugging runs.

**Reusing a `Simulation`**

A `Simulation` keeps the compiled step for its box configuration. Reuse it for many seeds or ICs. A new cosmology
needs a new `Simulation`: the growth tables are built at construction, and the kick is recompiled (a few seconds).

**Halo check**

`evolve(check_halo="raise"|"warn"|"off")`. After every step, the maximum displacement along the sharded axis is
compared with the usable halo `halo_size/2 − 1 − 2` cells.

- If particles move further than that, the PM force is wrong. The default therefore raises `HaloOverflowError`;
  rerun with a larger `halo_size`.
- At z=0, about 1 Mpc/h cells reach 35–40 cells for Planck-like cosmologies, so 96 is enough.
- Low-Ωm / high-σ8 cosmologies reach about 80 cells at z=0.5. Use `halo_size=240` for those (about 2 s per box
  extra at 2048³).

**Device subsets:** `Simulation(cfg, devices=jax.devices()[:1])` runs on a subset of the devices. The tests use it to
compare 1 GPU with 4 GPUs in one process.

---

## 4. Launch model and constraints

**Launch**

- One process per node (`srun --ntasks-per-node=1 --gpus-per-node=4`); each process drives its node's 4 GPUs.
- `init_distributed()` reads `SLURM_STEP_NUM_TASKS` / `SLURM_PROCID` (or `JAX_NUM_PROCESSES` / `JAX_PROCESS_ID`) and uses the
  first node of the job as coordinator (port `JAX_COORDINATOR_PORT`, default 12355).

**Divisibility**

- The PM uses slab real FFTs (`pm_fft_backend="JAX_RFFT"`, mesh `(n_dev, 1)`), so `res` and `res_pm` must both be
  divisible by `n_dev`.
- 2LPT needs a 2-D mesh `(a, b)` with `a·b = n_dev` and both a and b dividing `res`.

**cuFFT size limit.** cuFFT (as bundled with jax 0.6.2, cufft 11.4) returns garbage when one FFT operand has
≥ 2³¹ elements. The run still "succeeds", but displacements reach about 10⁷ cells.

- The modified code splits such FFTs into chunks (`fft_in_chunks`).
- It raises an error if a 2LPT local block would still be ≥ 2³¹ elements.
- The pencil PM path (`pm_fft_backend="JAX"`) raises at that size.

**Never pipe `srun` through `head`.** The SIGPIPE kills the run.

---

## 5. Sizing: how many GPUs

These are measured end-to-end runs with this code (2LPT at z=127, 20 BullFrog steps to z=0, P(k) at the IC and at
z=0, platform allocator). Times include the JIT compile of the first step.

| Box [Mpc/h] | Particles | PM mesh | GPUs (nodes) | ICs + 2LPT | 20 PM steps | Max disp / usable halo |
|---|---|---|---|---|---|---|
| 1000 | 1024³ | 1024³ | 4 (1) | 11 s | 8.2 s (5.6 s warm) | 34.5 / 45 cells |
| 2000 | 1024³ | 2048³ | 4 (1) | 12 s | 19.1 s (18.1 s warm) | 38.6 / 45 |
| 3000 | 1536³ | 3072³ | 8 (2) | 22 s | 124 s | 38.9 / 45 |

**Minimum GPU counts found**

- **2 GPUs:** 1024³/1024³ (measured).
- **4 GPUs:** 1024³/2048³ (measured).
- **8 GPUs:** 1536³/3072³ (measured, about 74 GB per GPU; 6 GPUs do not fit).

**Rules of thumb for other sizes** (fit to the runs above; leave about 10 % margin):

- **2LPT stage:** about **130 bytes per particle** in total, spread over the GPUs.
- **PM step:** about **21 bytes per mesh cell** in total, including the particles for `res_pm = 2·res`.
- **GPU count:** n_GPU ≥ max(130·res³, 21·res_pm³) / 90 GB, rounded up to a count that satisfies the divisibility
  rules in §4.

Estimates from these rules, not run:

- 2048³ / 2048³: about 16 GPUs, limited by 2LPT.
- 3072³ / 3072³: about 48 GPUs, limited by 2LPT.

**Use the fewest nodes that fit.** The slab FFT all-to-all dominates once it crosses nodes. For example,
2000 Mpc/h with 1024³ particles on a 2048³ mesh took 34 s on 2 nodes (8 GPUs) but 18–23 s on 1 node (4 GPUs), even
with the OFI plugin. Without the plugin (TCP sockets), multi-node runs are another 4–5× slower.

---

## 6. Validation

### Physics

**Single-GPU agreement.** At 512³ particles, 1024³ mesh, 1 Gpc/h, 20 BullFrog steps from the Quijote LH663 IC, the
multi-GPU code matched the original single-GPU DISCO-DJ (`discodj` package):

- 2LPT ICs: relative rms difference 1.7·10⁻⁵.
- z=0 displacements: maximum difference 0.056 PM cells. For scale, two runs of the multi-GPU code itself differ by
  up to 0.0014 cells, because GPU `scatter_add` ordering is not deterministic.

**Memory optimisations.** Each was checked against the unoptimised path:

- Unchunked: bit-identical.
- Chunked or looped: differences at float32 round-off only (≤ 2·10⁻⁷ relative on the kick).

**P(k)** for the three configurations in the sizing table, all from Gaussian ICs with the Quijote LH663 linear P(k):

- At the ICs, P / D²P_lin = 0.995 / 1.009 / 1.004 for 0.02 < k < 0.06.
- At z=0 the three boxes agree to about 2 % for 0.03 ≲ k ≲ 1 h/Mpc.
- P/P_lin at k = 0.3 h/Mpc is 1.50 in all three; at k = 1 it is about 5.0–5.1.

### This kit (`tests/test_consistency.py`, and the runs reproduced through `discodj_mgpu`)

| Check | Result |
|---|---|
| CPU, 4 fake devices: kick variants vs plain path | 1.8·10⁻⁷ relative |
| CPU: Gaussian ICs, 1 vs 4 devices | 7·10⁻⁷ of std |
| CPU: 2LPT psi, 1 vs 4 devices | 7·10⁻⁹ cells max |
| CPU: z=0 psi, 1 vs 4 devices (5 steps) | rms 4.8·10⁻⁶, max 3.8·10⁻⁵ cells |
| CPU: distributed P(k) vs numpy CIC P(k) | 1.2·10⁻⁶ relative |
| GPU (4× GH200), 128³/256³, 500 Mpc/h, 10 steps: kick variants / ICs / 2LPT | 2.5·10⁻⁷ / 1.9·10⁻⁶ of std / 3.8·10⁻⁸ cells |
| GPU 128³/256³: z=0 psi 1 vs 4 GPUs; P(k) vs numpy | rms 3.0·10⁻⁵, max 3.4·10⁻⁴ cells; 5.5·10⁻⁶ |
| GPU 256³/512³, 500 Mpc/h, 20 steps: z=0 psi 1 vs 4 GPUs | rms 7.5·10⁻⁵, max 1.2·10⁻³ cells |

**Reproduction of the production runs.** `examples/run_sim.py` (seed 663, LH663 cosmology, Quijote P(k) table) was
re-run on 2026-11-12 and compared with the P(k) of the runs that validated the original code. Bins are compared up to
the particle Nyquist frequency.

| Case | GPUs | ICs + 2LPT | 20 steps | Total (incl. compile) | P(k) vs earlier run, IC / z=0 (max rel.) |
|---|---|---|---|---|---|
| 1 Gpc/h, 1024³ / 1024³ | 4 (1 node) | 9.3 s | 8.1 s | 19.6 s | 2.8·10⁻⁶ / 1.7·10⁻⁶ |
| 2 Gpc/h, 1024³ / 2048³ | 4 (1 node) | 11.5 s | 19.6 s | 34.2 s | 9.8·10⁻⁶ / 1.9·10⁻⁶ |
| 3 Gpc/h, 1536³ / 3072³ | 8 (2 nodes) | 19.2 s | 125 s | 150 s | 1.4·10⁻⁵ / 3.1·10⁻⁶ |

The residual differences come from non-deterministic GPU `scatter_add` ordering. "Total" includes JIT compilation and
P(k) measurement. The 3 Gpc/h case prints a harmless XLA warning, "Can't reduce memory use below … by
rematerialization"; the run fits.

---

## 7. Changes to `discodj_dist` vs upstream `f264894`

All changes are in one commit on this branch (`git diff f264894 -- src/discodj_dist`). Each optimisation has an environment switch, read when
the function is traced, so A/B tests need a fresh `jit`.

### Correctness fixes

1. **`lpt/nlpt_distributed.py: pad_fourier_full`** is now built from slices and `concatenate`.
   - The original `.at[].set` block scatter into a sharded zeros array is miscompiled by XLA on multi-GPU meshes
     (jax 0.6.2) when fused with the preceding derivative inside one `jit`.
   - The result was a 2LPT term about 0.77× too small (correlation 0.88).
   - Do not reintroduce that pattern.
2. **cuFFT ≥ 2³¹ elements** gives wrong results. `core/distributed_pm.py: fft_in_chunks` splits FFTs along a
   non-transformed axis. Size guards raise instead of silently producing garbage.

### Memory optimisations

| Change | Where | Switch (default) |
|---|---|---|
| Lean dealiased 2LPT FFTs: pad/crop each axis next to its own 1-D FFT; 2LPT peak 57 → 35 GB/GPU at 1024³ on 4 GPUs | `lpt/nlpt_distributed.py` | `DISCO_LPT_LEAN_FFT=1` |
| CIC scatter/gather loop over the 27 stencil offsets with `lax.fori_loop` (a Python loop is not enough: XLA hoists every offset); PM memory −40 % | `core/distributed_pm.py` | `DISCO_PM_STENCIL_LOOP=1` |
| Scatter/gather in blocks of particle rows, so per-particle temporaries exist for one block only | `core/distributed_pm.py` | `DISCO_PM_PART_CHUNK=67108864` (max particles per block; ≤0 off) |
| The 3 force components run in a `fori_loop` (`lax.switch` picks the k-kernel), so one padded force field exists at a time instead of 3–4; momentum is updated in place; psi is converted to grid units per chunk | `nbody/acc_distributed.py: kick_PM_distributed` (slab RFFT path only) | `DISCO_PM_FORCE_LOOP=1` |
| Static unrolled FFT chunks with optimisation barriers (`0` = `fori_loop`) | `core/distributed_pm.py` | `DISCO_FFT_CHUNK_STATIC=1`, `DISCO_FFT_CHUNK_MAX_ELEMS=2^31-1` |

**Effect:** 3 Gpc/h at 1536³ particles on a 3072³ mesh went from not fitting on 8 GPUs (CPU estimate 118 GB/GPU) to
running at 74 GB/GPU, with no slowdown of the PM step (1 Gpc/h, 1024³ mesh on 4 GPUs: 5.6 s vs 5.5 s for 20 steps).

**Tried and rejected** (no gain):

- Halo-free slab scatter/gather.
- Optimisation barriers between force components.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `free(): double free detected`, exit 134 at the first collective | HPC-SDK CUDA/NCCL on `LD_LIBRARY_PATH`; `source env/setup_delta.sh` (module purge) |
| Multi-node run 4–5× slower than expected | NCCL fell back to sockets: check `OFI_PLUGIN`. NCCL_DEBUG=INFO should print `Using network Libfabric` |
| `RESOURCE_EXHAUSTED` / OOM close to 96 GB | `XLA_PYTHON_CLIENT_ALLOCATOR=platform` and `MEM_FRACTION=0.97`; otherwise more GPUs (§5) |
| Run "works" but displacements are about 10⁷ cells | an FFT operand ≥ 2³¹ elements outside the guarded paths; use more GPUs |
| `HaloOverflowError` | increase `halo_size` (e.g. 240) |
| XLA rendezvous timeout (SIGABRT) at the first collective, roughly 1 in 200 boxes | transient; restart the box |
| `Input sharding was found to be none` | jaxdecomp pencil FFTs inside a `fori_loop`; the force loop is only used on the slab path |
| Login node: `numpy` import fails with a thread error | per-user thread limit: `OPENBLAS_NUM_THREADS=4`, or run on a compute node |
| `discodj_native ... extension is not available` | TreePM short-range forces or the N-GenIC RNG were called: the shim does not provide them. Install the compiled `discodj` (needs libgsl) |
| `XLA_FLAGS=--xla_gpu_deterministic_ops=true` fails | int16 CIC stencil indices are not supported in deterministic mode (`core/scatter_and_gather.py`); deterministic runs are also about 4–5× slower |

## 9. Limitations

- **Particle and stepper options:** 2LPT ICs only (no 1LPT/3LPT through the driver). CIC mass assignment
  (`worder=2`). DKD BullFrog / FastPM in D-time. float32.
- **Gaussian ICs** use JAX's (partitionable) RNG, not the N-GenIC/2LPTic sequence. To match an N-GenIC/2LPTic
  realisation, generate the density on the host and pass it to `delta_from_array`.
- **The memory optimisations target the slab-RFFT PM path.** The pencil (`JAX`) path still works but has neither the
  force loop nor the memory savings.
