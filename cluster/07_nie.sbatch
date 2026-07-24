#!/bin/bash
#SBATCH --job-name=tp-nie
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=176G
#SBATCH --output=cluster/logs/nie_%j.out

# The causal experiment (M&T Sec 6.1) — the one the 9B run exists for.
#
# 2B could not test the antipodal prediction: larger_than had no causal effect on
# sp_en_trans there (NIE ~0.03, cos to the sp_en truth direction ~0.002), so the
# combined larger+smaller condition had no baseline to degrade from. M&T see the
# cross-dataset causal transfer at 13B/70B, so 9B is where it should appear.
#
# Check the larger_than SINGLETON NIE first: if it is now clearly > 0, the
# antipodal test is live and the larger+smaller comparison is meaningful.
#
# Needs the model (forward passes with interventions), hence GPU + the same 176G
# host RAM that extraction needed — TransformerLens upcasts the state dict to
# fp32 during weight processing regardless of dtype.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

source cluster/lastping.sh
lp_init "Truth-Probes: 9B NIE"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

time python scripts/aux_nie_intervention.py
