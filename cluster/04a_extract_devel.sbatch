#!/bin/bash
#SBATCH --job-name=tp-extract-dev
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/extract_%j.out

# Same work as 04_extract.sbatch, sized to the gpu_devel QOS.
#
# WHY THIS EXISTS: the `gpu` partition is badly backlogged (462 pending vs 117
# running at time of writing) and projects a ~2-day wait for ANY card type —
# a40, l40s, a100 and h100 all returned the same start estimate, so the queue,
# not the constraint, is the binding factor. gpu_devel allows a 6-hour walltime
# and starts in about a minute.
#
# Extraction is ~480 short forward passes: minutes of compute, nowhere near the
# 6-hour cap. So this is a legitimate fit for gpu_devel, not an abuse of it.
#
# The QOS caps are cpu=4, gres/gpu=2, mem=32G per user — hence 4 CPUs and 32 GB
# here rather than the 8/64 in 04. The 32 GB is the only real constraint (the
# model briefly lands in host RAM before moving to the card), and step 03 —
# which loads the same model under the same 32 GB on this same partition — is
# what proves it fits. Run the gate first.
#
# Use 04_extract.sbatch instead if gpu_devel is busy or you want the roomier
# allocation and don't mind the wait.

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
