#!/bin/bash
# Kit self-test on one node: CPU fake-device checks + GPU checks (1 GPU vs 4 GPUs, P(k) vs numpy). ~5 min.
#   sbatch jobs/test_kit.sbatch
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=discodj_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
set -eo pipefail
KIT=${KIT:-/u/yzhang116/DISCO-DJ-multigpu/multigpu}
source "$KIT/env/setup_delta.sh"
echo "== CPU (4 fake devices)"; "$PY" -u "$KIT/tests/test_consistency.py" --cpu
echo "== GPU, small (P(k) checked against numpy)"; "$PY" -u "$KIT/tests/test_consistency.py" --res 128 --res-pm 256 --box 500 --halo 64 --steps 10
echo "== GPU, 256^3 / 512^3 mesh"; "$PY" -u "$KIT/tests/test_consistency.py" --res 256 --res-pm 512 --box 500 --halo 96 --steps 20
