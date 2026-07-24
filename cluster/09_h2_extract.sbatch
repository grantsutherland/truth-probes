#!/bin/bash
#SBATCH --job-name=tp-h2-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=176G
#SBATCH --output=cluster/logs/h2_extract_%j.out

# H2 Phase 1 — extract activations over the framed prompts, for the 392 rows
# that survived Phase 0's model_knows filter.
#
# 392 short prompts, all 42 layers in one forward pass (~236 MB cached). The
# 176G host RAM is the usual TransformerLens fp32-upcast requirement during
# weight processing, not a function of how much is being extracted.
#
# Everything after this is CPU-only: Phases 2-5 apply the FROZEN H1 probes to
# this cache and never load the model again. If a later phase needs the model,
# something has gone wrong with the no-retraining rule.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

source cluster/lastping.sh
lp_init "Truth-Probes: H2 Phase 1 extract"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

time python scripts/h2_01_extract.py
