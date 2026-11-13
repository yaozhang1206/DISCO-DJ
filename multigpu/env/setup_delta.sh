# Source inside a Slurm job on NCSA Delta (ghx4, GH200) before running the kit:
#   source /path/to/DISCO-DJ/multigpu/env/setup_delta.sh
# Override before sourcing: DISCODJ_PY (python of the conda env), OFI_PLUGIN (see delta_ofi_env.sh).
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # DISCO-DJ/multigpu
REPO=$(cd "$KIT/.." && pwd)                             # DISCO-DJ checkout (holds src/)
# Login-default modules put HPC-SDK CUDA 12.9 / NCCL on LD_LIBRARY_PATH, shadowing JAX's bundled libraries
# -> "free(): double free detected" at the first collective. Purge and load only these.
module purge
module load cuda/12.4.0 cudnn/9.3.0.75 gsl
export PY=${DISCODJ_PY:-/u/yzhang116/.conda/envs/pm/bin/python}
source "$KIT/env/delta_ofi_env.sh"
export PYTHONPATH=$REPO/src${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}
# JAX memory settings (discodj_mgpu.init_distributed sets the same defaults if unset)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.97}
export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
echo "[setup_delta] REPO=$REPO PY=$PY OFI_PLUGIN=$OFI_PLUGIN nodes=${SLURM_JOB_NUM_NODES:-?}"
