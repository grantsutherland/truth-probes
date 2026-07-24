#!/bin/bash
#SBATCH --job-name=tp-h2-knows
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=176G
#SBATCH --output=cluster/logs/h2_knows_%j.out

# H2 Phase 0 — the model_knows filter, SCORE phase only.
#
# 200 short forward passes, no interventions, no caching of activations: this is
# the cheapest GPU job in the project. It still needs 176G host RAM for the same
# reason every other one does — TransformerLens upcasts the state dict to fp32
# during weight processing, so dtype=bfloat16 does not bound peak RSS (measured
# 151 GiB during extraction). gpu_devel's 32G cap cannot load 9B at all.
#
# The FILTER phase is deliberately not run here. Picking the margin threshold
# requires looking at the distribution this job produces, and rerunning it after
# a retune would mean re-queuing a GPU job to recompute numbers that never
# changed. Run it on the login node afterwards:
#
#     python scripts/h2_00_model_knows.py --phase filter --threshold <T>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

source cluster/lastping.sh
lp_init "Truth-Probes: H2 Phase 0 model_knows"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

time python scripts/h2_00_model_knows.py --phase score
