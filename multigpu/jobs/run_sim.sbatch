#!/bin/bash
# Generic launcher: one process per node, all 4 GPUs per process. Arguments go to examples/run_sim.py.
#   sbatch -N 1 jobs/run_sim.sbatch --box 2000 --res 1024 --res-pm 2048 --pk-table Pk.txt --seed 1 --out runs/c2
#   sbatch -N 2 jobs/run_sim.sbatch --box 3000 --res 1536 --res-pm 3072 ...
# Change account / partition / time on the command line (sbatch -A ... -p ... -t ...) or here.
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=discodj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
set -eo pipefail
KIT=${KIT:-/u/yzhang116/DISCO-DJ-multigpu/multigpu}      # sbatch copies this script, so the kit path must be explicit
source "$KIT/env/setup_delta.sh"
NN=${SLURM_JOB_NUM_NODES:-1}
# never pipe srun through `head` (SIGPIPE kills the run)
srun -N "$NN" -n "$NN" --ntasks-per-node=1 --gpus-per-node=4 "$PY" -u "$KIT/examples/run_sim.py" "$@"
