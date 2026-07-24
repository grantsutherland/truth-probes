#!/bin/bash
#SBATCH --job-name=tp-analysis
#SBATCH --partition=day
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=cluster/logs/analysis_%j.out

# Everything that runs off the cached activations: probe training, the transfer
# matrix, combined-negation, and the CPU aux analyses. No GPU — none of these
# load the model, they all read results/activations/<model>/.
#
# Run in dependency order: 02 saves the probes that 03 reloads.
#
# Each step is allowed to fail without killing the rest (no `set -e` around the
# loop): these are independent analyses, and losing the whole batch because one
# aux script hits an edge case at 9B would waste the queue slot. Failures are
# summarised at the end.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

STEPS=(
  02_train_probes.py
  03_transfer_eval.py
  03b_combined_negation.py
  aux_negation_by_layer.py
  aux_transfer_by_layer.py
  aux_negation_mechanism.py
  aux_whitened_alignment.py
  aux_leakage_check.py
)

failed=()
for s in "${STEPS[@]}"; do
  echo
  echo "################################################################"
  echo "## $s"
  echo "################################################################"
  if time python "scripts/$s"; then
    echo "## OK: $s"
  else
    echo "## FAILED: $s (rc=$?)"
    failed+=("$s")
  fi
done

echo
echo "================================================================"
if [ ${#failed[@]} -eq 0 ]; then
  echo "all ${#STEPS[@]} steps OK"
else
  echo "${#failed[@]} FAILED: ${failed[*]}"
fi
echo "================================================================"
ls -la results/*.json
[ ${#failed[@]} -eq 0 ]
