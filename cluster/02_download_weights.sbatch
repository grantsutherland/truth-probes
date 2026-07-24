#!/bin/bash
#SBATCH --job-name=tp-download
#SBATCH --partition=day
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/download_%j.out

# Pre-fetch the Gemma-2-9B weights so the GPU job starts against a warm cache.
#
# 34.5 GiB across 8 safetensors shards — NOT the 18.5 GB that a bf16 checkpoint
# would be. google/gemma-2-9b ships fp32 weights (9.24B params x 4 bytes); the
# 18.5 GB figure is what it occupies on the GPU after the bf16 cast, not what
# comes over the wire or sits on disk. Hence 4h of walltime rather than 2.
#
# Doing this inside the extraction job would mean an accelerator sitting idle
# for the whole download, which is exactly what the idle-GPU monitors watch for.
# No GPU requested here.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate

# Home is 61/125 GiB used and already holds a 12 GiB HF cache; another 18.5 GiB
# would leave little room. Park the cache on project, which has ~1.2 TB free.
# Every later job sets the same HF_HOME so they all hit this copy.
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"

# Fail early and legibly if step 0 was skipped, rather than 401-ing mid-download.
echo "== auth check =="
hf auth whoami

echo
echo "== downloading google/gemma-2-9b =="
# TransformerLens loads the HF checkpoint by name; pre-populating the cache with
# the same repo id is all that is needed for it to resolve offline later.
hf download google/gemma-2-9b

echo
echo "== cache size =="
du -sh "${HF_HOME}"
echo "Done."
