#!/bin/bash
#SBATCH --job-name=tp-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|l40s|a100|h100|h200"
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=176G
#SBATCH --output=cluster/logs/extract_%j.out

# Phase 1 only: GPU forward passes, all 42 layers, 6 datasets -> sweep caches.
# The layer sweep that follows is CPU-only probe training and runs as a separate
# job (05_sweep.sbatch), so this job releases the card the moment the last
# forward pass lands.
#
# Sizing: 18.5 GiB of bf16 weights plus a small activation buffer (batch 16 x 42
# layers x 3584 dims is ~10 MB), so a 48 GiB a40/l40s is comfortable and far
# 128G of HOST ram is NOT slack: TransformerLens.process_weights upcasts the
# state dict to fp32 (weight_processing.py: state_dict[k] = v.float()), so a
# 9.24B model needs ~37GB there regardless of dtype=bfloat16. A 32G allocation
# OOMs during load, before a single forward pass. 7660 statements at batch 16 is ~480
# forward passes over short sequences — minutes of compute, so the hour of
# walltime is mostly model-load headroom, not a real estimate.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

# Job monitoring. Sets the EXIT trap and sends the start ping; the per-dataset
# heartbeats come from inside the extraction loop (src/lastping.py).
source cluster/lastping.sh
lp_init "Truth-Probes: 9B extract"

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

time python scripts/01_extract_activations.py --phase extract

echo
echo "== sweep caches written =="
du -sh results/activations/*/sweep
