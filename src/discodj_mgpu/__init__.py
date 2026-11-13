"""discodj_mgpu: thin driver around the memory-optimised discodj_dist (multi-GPU / multi-node DISCO-DJ PM).

    from discodj_mgpu import init_distributed, SimConfig, Simulation
"""
from .runtime import init_distributed, log, is_leader, gpu_memory_gb
from .simulation import SimConfig, Simulation, HaloOverflowError
from .shards import load_shards, gather_to_host

__all__ = ["init_distributed", "log", "is_leader", "gpu_memory_gb", "SimConfig", "Simulation", "HaloOverflowError",
           "load_shards", "gather_to_host"]
