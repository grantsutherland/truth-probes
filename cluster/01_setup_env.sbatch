#!/bin/bash
#SBATCH --job-name=tp-torch
#SBATCH --partition=day
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/setup_%j.out

# Replace the CUDA build of torch in the existing .venv.
#
# THE PROBLEM: `pip install -r requirements.txt` resolved torch 2.13.0 to the
# +cu130 wheel (the current default on PyPI). The cluster's driver is 570.211.01,
# which tops out at CUDA 12.8 — a cu130 build needs a 580+ driver. torch imports
# fine but torch.cuda.is_available() is False, so the extraction job would have
# fallen back to CPU (or crashed) after queueing.
#
# THE FIX: the cu126 index still carries torch 2.13.0 exactly, so the version
# pin in requirements.txt is preserved and only the CUDA build changes. (The
# cu128 index stops at 2.11.0, which would have meant breaking the pin.) A cu126
# runtime runs on any 12.x driver and covers every card in the gpu partition:
# a40 (sm_86), l40s (sm_89), a100 (sm_80), h100/h200 (sm_90).
#
# Runs as a batch job because YCRC's Python guidance is explicitly against
# pip-installing on login nodes. No GPU — pip is network-bound, and the wheel
# installs identically whether or not a card is visible.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs

source .venv/bin/activate
python -V

echo
echo "== before =="
python -c "import torch; print('  torch', torch.__version__, '| cuda build', torch.version.cuda)"

echo
echo "== installing torch 2.13.0+cu126 =="
pip install --force-reinstall "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126

echo
echo "== after =="
python -c "import torch; print('  torch', torch.__version__, '| cuda build', torch.version.cuda)"

echo
echo "== rest of the stack still intact =="
# The torch reinstall can move shared deps; confirm the pinned stack survived.
python - <<'PY'
import importlib.metadata as m
for p in ["torch", "transformer-lens", "transformers", "pandas", "numpy", "huggingface-hub"]:
    try:
        print(f"  {p:20s} {m.version(p)}")
    except Exception:
        print(f"  {p:20s} MISSING")
PY

echo
echo "Done. cuda availability is only testable on a GPU node — that is step 03."
