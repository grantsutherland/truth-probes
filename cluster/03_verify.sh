#!/bin/bash
# The gate. Run INSIDE an interactive gpu_devel allocation:
#
#   salloc -p gpu_devel -t 0:30:00 --gres=gpu:1 --mem=32G -c 4
#   bash cluster/03_verify.sh
#
# Confirms Gemma-2-9B loads under TransformerLens, that the residual hook is
# still named blocks.{n}.hook_resid_post at 9B, that the shape facts are what
# the analysis assumes (42 layers, d_model 3584), and that it fits in VRAM with
# room to spare. This is the same check that de-risked the 2B pipeline; nothing
# multi-hour should be queued until it passes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate
export HF_HOME=/gpfs/radev/project/krishnaswamy_smita/gss34/hf_cache

nvidia-smi --query-gpu=name,memory.total --format=csv

python - <<'PY'
import os, sys, torch
sys.path.insert(0, "src")
import config
from activations import load_model, verify_model, extract_activations, resid_hook_name

print(f"\n== loading {config.MODEL_NAME} ==", flush=True)
model = load_model(config.MODEL_NAME)
info = verify_model(model)

print("\n== model facts ==")
for k, v in info.items():
    print(f"  {k:12s} {v}")

n_layers, d_model = info["n_layers"], info["d_model"]
expect_layers, expect_dim = 42, 3584
ok = True
if n_layers != expect_layers:
    print(f"  !! n_layers {n_layers} != expected {expect_layers}")
    ok = False
if d_model != expect_dim:
    print(f"  !! d_model {d_model} != expected {expect_dim}")
    ok = False

# The last layer's hook matters as much as layer 0's: extraction asks for every
# layer at once, so a naming change anywhere in the stack breaks the sweep.
last = resid_hook_name(n_layers - 1)
if last not in model.hook_dict:
    print(f"  !! {last} missing from hook_dict")
    ok = False
else:
    print(f"  {'last hook':12s} {last} present")

# A real multi-layer extraction on a few statements — the actual code path 04
# will run, not just a bare forward pass.
print("\n== extraction smoke test (3 statements, all layers) ==")
acts = extract_activations(
    model,
    ["The city of Paris is in France.",
     "The city of Paris is in Japan.",
     "The Spanish word 'gato' means 'cat'."],
    layers=list(range(n_layers)),
    token_pos=-1,
    batch_size=3,
)
print(f"  shape {tuple(acts.shape)}  dtype {acts.dtype}  (expect (3, {n_layers}, {d_model}) float32)")
if tuple(acts.shape) != (3, n_layers, d_model):
    print("  !! unexpected shape")
    ok = False

# --------------------------------------------------------------------------- #
# Readout tokenization. The NIE experiment scores P(" TRUE") - P(" FALSE") on the
# token after "This statement is:". Two ways that silently corrupts every causal
# number, both invisible in the output (they look like a weak causal effect, not
# an error), and both cheap to check while the model is already loaded:
#   1. " TRUE"/" FALSE" not being single tokens in this vocab.
#   2. the model actually wanting to emit something else entirely there.
# Gemma-2-9B has its own vocab, so this is re-checked at 9B, not assumed from 2B.
# --------------------------------------------------------------------------- #
print("\n== readout tokenization ==")
for s in [" TRUE", " FALSE", "TRUE", "FALSE"]:
    toks = model.to_tokens(s, prepend_bos=False)[0]
    pieces = [model.to_string(t.unsqueeze(0)) for t in toks]
    flag = "" if len(toks) == 1 else "   <-- MULTI-TOKEN"
    print(f"  {s!r:10s} -> {len(toks)} token(s) {pieces}{flag}")
    if s.startswith(" ") and len(toks) != 1:
        ok = False

# The exact prompt aux_nie_intervention.build_prompt() constructs, so this shows
# what the model does at the position the NIE readout actually reads.
FEWSHOT = [("The Spanish word 'fruta' means 'goat'.", "FALSE"),
           ("The Spanish word 'carne' means 'meat'.", "TRUE")]
stmt = "The Spanish word 'gato' means 'cat'."
full = "\n".join(f"{s} This statement is: {lab}" for s, lab in FEWSHOT) \
       + "\n" + stmt + " This statement is:"

with torch.no_grad():
    logits = model(model.to_tokens(full))
probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
top = torch.topk(probs, 5)

print("\n== top-5 continuations after 'This statement is:' ==")
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {p:6.3f}  {model.to_string(torch.tensor([i]))!r}")

t_id, f_id = (model.to_tokens(s, prepend_bos=False)[0, 0].item() for s in (" TRUE", " FALSE"))
print(f"\n  P(' TRUE')  = {probs[t_id]:.4f}")
print(f"  P(' FALSE') = {probs[f_id]:.4f}")
print(f"  scored mass = {probs[t_id] + probs[f_id]:.4f} of the distribution")
# If the labels hold almost none of the mass, the readout is measuring a corner
# of the distribution and the NIE ceiling is low for that reason alone — the 2B
# run's weak PD+ (-0.065) is consistent with exactly this. Not fatal, but it
# decides whether a weak 9B NIE means "no causal effect" or "bad readout".
if probs[t_id] + probs[f_id] < 0.10:
    print("  !! labels hold <10% of the mass — the few-shot readout is weak here.")
    print("     Add exemplars (FEWSHOT in aux_nie_intervention.py) before reading")
    print("     anything into a small NIE. This is a documented M&T knob.")

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n== VRAM ==\n  peak {peak:.1f} GiB of {total:.1f} GiB ({100*peak/total:.0f}%)")

# Host RAM is the binding constraint on gpu_devel (32 GB cap), not VRAM.
# TransformerLens loads the HF checkpoint, applies weight processing, then moves
# to device — transiently holding close to two copies of 18.5 GB. If this run
# survived, extraction under the same allocation will too.
import resource
hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
print(f"\n== host RAM ==\n  peak RSS {hwm:.1f} GiB (gpu_devel caps at 32 GiB)")
if hwm > 26:
    print("  !! close to the cap — prefer the gpu partition for extraction.")

print("\nPASS - safe to queue extraction." if ok else "\nFAIL - do not queue extraction.")
sys.exit(0 if ok else 1)
PY
