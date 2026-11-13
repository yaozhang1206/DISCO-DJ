"""Process/GPU setup for multi-GPU, multi-node discodj_dist runs.

Launch model: ONE process per node (srun --ntasks-per-node=1), each process drives all GPUs of its node.
Call :func:`init_distributed` before any other JAX call (it sets XLA allocator env vars and starts
``jax.distributed`` + ``jaxdecomp`` when more than one process is launched).
"""
from __future__ import annotations

import os
import subprocess

_INITIALIZED = False

# Defaults that were validated on NCSA Delta GH200 (96 GB/GPU); setdefault, so the caller's env wins.
XLA_ENV_DEFAULTS = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    # JAX caps GPU memory at 75 % by default; the big configurations need ~97 % of the 96 GB.
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
    # The default BFC allocator fragments after 2LPT and OOMs near the limit (e.g. 3072^3 mesh on 8 GPUs);
    # 'platform' (cudaMalloc/cudaFree per buffer) fits; no measurable slowdown for these PM steps.
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
}


def _env_int(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def _coordinator_address() -> str:
    explicit = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if explicit:
        return explicit
    host = os.environ.get("JAX_COORDINATOR_HOST")
    if not host:
        nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        if not nodelist:
            raise RuntimeError("multi-process run but neither JAX_COORDINATOR_ADDRESS nor SLURM_NODELIST is set")
        host = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()[0]
    port = os.environ.get("JAX_COORDINATOR_PORT", "12355")
    return host if ":" in host else f"{host}:{port}"


def _local_device_ids():
    explicit = os.environ.get("JAX_LOCAL_DEVICE_IDS")
    if explicit:
        return [int(t) for t in explicit.split(",") if t.strip()]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd and cvd not in {"NoDevFiles", "none", "void"}:
        return list(range(len([t for t in cvd.split(",") if t.strip()])))
    return None


def init_distributed(verbose: bool = True) -> dict:
    """Initialise JAX (+ jax.distributed and jaxdecomp for >1 process). Idempotent.

    Process count / rank come from JAX_NUM_PROCESSES / JAX_PROCESS_ID or the Slurm step variables
(SLURM_STEP_NUM_TASKS / SLURM_PROCID); the coordinator is the first node of the step (port JAX_COORDINATOR_PORT).
    Returns a small dict describing the run (process index/count, device counts, platform).
    """
    global _INITIALIZED
    for k, v in XLA_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    # A stale NCCL_NET_PLUGIN pointer makes NCCL fail; the net plugin (if any) is found as libnccl-net.so on
    # LD_LIBRARY_PATH (see env/delta_ofi_env.sh).
    os.environ.pop("NCCL_NET_PLUGIN", None)

    import jax

    first = not _INITIALIZED
    if first:
        # SLURM_STEP_NUM_TASKS (not SLURM_NTASKS): python started directly in a multi-node batch script, without
        # srun, must stay a single process instead of waiting forever for peers.
        nproc = _env_int("JAX_NUM_PROCESSES", "SLURM_STEP_NUM_TASKS", default=1)
        rank = _env_int("JAX_PROCESS_ID", "SLURM_PROCID", default=0)
        if nproc > 1:
            jax.distributed.initialize(coordinator_address=_coordinator_address(), num_processes=nproc,
                                       process_id=rank, local_device_ids=_local_device_ids())
            import jaxdecomp
            jaxdecomp.init()
        _INITIALIZED = True

    info = dict(process=jax.process_index(), nproc=jax.process_count(), n_dev=jax.device_count(),
                n_local=jax.local_device_count(), platform=jax.devices()[0].platform)
    if verbose and first and info["process"] == 0:
        print(f"[discodj_mgpu] {info['nproc']} process(es), {info['n_dev']} {info['platform']} device(s) "
              f"({info['n_local']} per process); allocator={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}, "
              f"mem_fraction={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}", flush=True)
    return info


def is_leader() -> bool:
    import jax
    return jax.process_index() == 0


def log(msg: str, all_procs: bool = False) -> None:
    import time
    import jax
    if all_procs or jax.process_index() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] [proc {jax.process_index()}] {msg}", flush=True)


def gpu_memory_gb(key: str = "peak_bytes_in_use") -> list[float]:
    """Per local device memory statistic in GB (0 when the allocator does not report, e.g. 'platform')."""
    import jax
    return [round(((d.memory_stats() or {}).get(key, 0)) / 1e9, 2) for d in jax.local_devices()]
