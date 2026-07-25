#!/bin/bash
#SBATCH --job-name=tp-h2-noframe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=176G
#SBATCH --output=cluster/logs/h2_noframe_%j.out

# H2 no-frame baseline — the neutral condition the 2x2 design lacks.
#
# 196 bare statements, no instruction prefix. Without this the design identifies
# only the DIFFERENCE between the two frames, not which one moved; with it, the
# lie-frame-compresses vs truth-frame-amplifies question is decidable.
#
# Extract phase only. Analysis is CPU:
#   python scripts/h2_05_noframe.py --phase analyze

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

source cluster/lastping.sh
lp_init "Truth-Probes: H2 no-frame baseline"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

time python scripts/h2_05_noframe.py --phase extract
