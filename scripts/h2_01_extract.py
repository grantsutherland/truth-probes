"""H2 PHASE 1 — extract induced-lie activations.

Runs the FRAMED prompts (the `prompt` column: instruction sentence + statement)
through Gemma-2-9B and caches the residual stream at the end-of-prompt token,
for the base facts that survived Phase 0's model_knows filter.

Nothing is trained or scored here. The one job is to produce activations that
sit in exactly the same space as the ones H1's frozen probes were fitted on, so
that applying those probes in Phase 2/3 tests the frame rather than testing a
change of extraction convention.

SAME PATH AS H1, DELIBERATELY:
  * `verify_model` before anything, as the H1 runbook requires.
  * `blocks.{layer}.hook_resid_post` via the shared `extract_activations`.
  * token_pos = -1, the last real token. For H1 that was the end-of-statement
    token of a bare statement; here the framed prompt still ENDS with the
    statement ("Say something false. The city of Paris is in France."), so the
    gathered token is the same end-of-statement position with a prefix in front
    of it. That is the intended comparison, and Phase 2's aligned-cell canary is
    what confirms the prefix did not break it.

ALL 42 LAYERS, ONE FORWARD PASS. Phase 5 re-reads the conflict cells across the
plateau (L17-31) to check the result is not L17-specific, and the forward pass —
not the storage — is the cost. Extracting the committed layer alone would buy a
236 MB saving and charge another GPU job for it. This is the Option-1 flow
`01_extract_activations` already uses.

ROW IDENTITY TRAVELS WITH THE TENSOR. `data.save_induced_lies` stores base_fact,
condition, frame and label alongside the activations. H2 slices by condition
(aligned vs conflict) and scores against two different targets, so row i of the
tensor must be row i of the CSV; re-deriving that by re-reading the CSV would
silently break the moment Phase 0 is retuned at a different threshold and the
surviving row set changes.
"""

import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                            # noqa: E402
import data as dataio                                    # noqa: E402
from activations import load_model, verify_model, extract_activations  # noqa: E402

DATASET_CSV = os.path.join(dataio.DATA_DIR, "induced_lies.csv")
REPORT_JSON = os.path.join(config.RESULTS_DIR, "h2_extract_report.json")
BATCH_SIZE = 16
_TOKENIZATION_PROBES = 3


def load_surviving_rows():
    """The rows Phase 0 kept, with `frame` held as 'true'/'false' strings.

    Errors rather than proceeds if model_knows is unpopulated: extracting all
    400 rows would quietly include the facts the model does not know, which is
    the exact contamination Phase 0 exists to prevent.
    """
    df = pd.read_csv(DATASET_CSV, dtype={"frame": str, "model_knows": str},
                     keep_default_na=False)
    if (df["model_knows"] == "").any():
        raise ValueError(
            f"{DATASET_CSV} has {(df['model_knows'] == '').sum()} rows with an "
            f"empty model_knows. Run h2_00_model_knows.py --phase filter first.")

    kept = df[df["model_knows"].astype(int) == 1].reset_index(drop=True)
    if len(kept) == 0:
        raise ValueError("model_knows excluded every row.")

    # The 2x2 must stay balanced after filtering — the filter drops whole base
    # facts (all four of their cells), so any imbalance here means it dropped
    # cells instead, and every pooled comparison downstream would be confounded.
    cells = kept.groupby(["condition", "frame"]).size()
    if cells.nunique() != 1:
        raise ValueError(
            f"filtering left an unbalanced 2x2 — cell sizes {cells.to_dict()}. "
            f"model_knows is supposed to be a per-BASE-FACT decision.")
    return kept


def main():
    torch.manual_seed(0)
    rows = load_surviving_rows()
    n_facts = rows["base_fact"].nunique()
    print(f"{len(rows)} surviving rows across {n_facts} base facts")
    print(rows.groupby(["condition", "frame"]).size().to_string())

    model = load_model(config.MODEL_NAME)
    info = verify_model(model)
    print("\nverify_model:", json.dumps(info, indent=2, default=str))

    layers = list(range(model.cfg.n_layers))
    committed = config.chosen_layer()
    print(f"\nextracting all {len(layers)} layers "
          f"(committed layer {committed}, plateau {config.plateau_layers()})")

    # Confirm the gathered position is the end-of-statement token, not a
    # trailing artifact. The framed prompt ends with the statement's period, so
    # the last real token should be that period — printed rather than assumed,
    # because a silent off-by-one here would move every H2 number to a position
    # the frozen probes were never fitted on.
    print("\ntoken convention check (token_pos=-1 -> last real token):")
    for prompt in rows["prompt"].head(_TOKENIZATION_PROBES):
        toks = model.to_tokens(prompt)[0]
        pieces = [model.to_string(t.unsqueeze(0)) for t in toks[-4:]]
        print(f"  {prompt!r}\n    last 4 tokens: {pieces}  "
              f"-> gathering {pieces[-1]!r}")

    prompts = rows["prompt"].tolist()
    acts = extract_activations(model, prompts, layers=layers,
                               token_pos=-1, batch_size=BATCH_SIZE)
    print(f"\nacts: {tuple(acts.shape)}  dtype {acts.dtype}")

    # ---- acceptance checks (the plan's Phase 1 criteria) ----
    expected = (len(rows), len(layers), model.cfg.d_model)
    if tuple(acts.shape) != expected:
        raise RuntimeError(f"shape {tuple(acts.shape)} != expected {expected}")
    if not torch.isfinite(acts).all():
        raise RuntimeError("non-finite values in extracted activations")
    # Distinct prompts must give distinct activations. An all-identical tensor
    # is what a broken gather (e.g. every row reading a pad position) looks
    # like, and it would survive the shape and finiteness checks above.
    committed_idx = layers.index(committed)
    at_layer = acts[:, committed_idx, :]
    spread = at_layer.std(dim=0).mean().item()
    n_unique = len({tuple(r[:8].tolist()) for r in at_layer})
    print(f"acceptance: finite OK | shape OK | mean per-dim std {spread:.4f} | "
          f"{n_unique}/{len(rows)} distinct rows at layer {committed}")
    if n_unique < len(rows):
        raise RuntimeError(
            f"only {n_unique} distinct activation rows out of {len(rows)} — "
            f"the token gather is probably reading the same position for "
            f"multiple rows.")

    dataio.save_induced_lies(acts, rows, layers, token_pos=-1,
                             model_name=config.MODEL_NAME)
    path = dataio._cache_path(dataio.INDUCED_LIES)
    print(f"\nsaved -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")

    # Reload through the public loader: proves the saved payload actually
    # round-trips into the shapes Phases 2-5 expect, while the job is still the
    # thing that can be re-run cheaply.
    ra, rl, rf, rmeta = dataio.load_induced_lies()
    print(f"reload check: acts {tuple(ra.shape)}  labels {tuple(rl.shape)}  "
          f"frame {tuple(rf.shape)}  layers {len(rmeta['layers'])}  "
          f"model {rmeta['model_name']}")
    assert rmeta["model_name"] == config.MODEL_NAME
    assert ra.shape == acts.shape and len(rmeta["base_fact"]) == len(rows)
    assert rl.tolist() == rows["label"].astype(float).tolist()

    report = {
        "model": config.MODEL_NAME,
        "n_rows": len(rows),
        "n_base_facts": int(n_facts),
        "cell_sizes": {f"{c}/{f}": int(v) for (c, f), v
                       in rows.groupby(["condition", "frame"]).size().items()},
        "layers": layers,
        "committed_layer": committed,
        "plateau_layers": config.plateau_layers(),
        "token_pos": -1,
        "d_model": model.cfg.d_model,
        "acts_shape": list(acts.shape),
        "cache_path": path,
        "hook_name": info["hook_name"],
    }
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {REPORT_JSON}")


if __name__ == "__main__":
    main()
