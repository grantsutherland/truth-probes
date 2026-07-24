#!/bin/bash
#SBATCH --job-name=tp-sweep
#SBATCH --partition=day
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/sweep_%j.out

# Phases 2-3: train MM + LR at every layer to pick the chosen layer, then commit
# the single-layer caches. No GPU — this reads the tensors 04 cached and never
# loads the model.
#
# This is the half that used to share a process with extraction. At 9B it is 42
# layers x 6 datasets x 2 probes, with LR at 2000 epochs on 3584-dim inputs, so
# it is a genuinely long CPU stretch that would otherwise pin an idle H100.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate

# LR training is the bottleneck and is pure BLAS; let torch use the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

time python scripts/01_extract_activations.py --phase sweep

echo
echo "== chosen layer =="
python -c "import json; d=json.load(open('results/layer_sweep.json')); print(d['model_name'], '-> layer', d['chosen_layer'])"
