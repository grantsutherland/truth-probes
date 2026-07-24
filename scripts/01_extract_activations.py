"""01 — Extract activations, sweep layers, commit the chosen layer (Option 1).

Three phases:
  1. EXTRACT  every layer for every dataset in one forward pass each, cached to
     results/activations/<model>/sweep/ (the slow step; skipped if already cached).
  2. SWEEP    train MM and LR probes at each layer on each SWEEP_DATASET, evaluate
     on a held-out split, and record both accuracy curves (averaged across those
     datasets). The chosen layer is selected on the MM curve — MM's direction is
     the one the induced-lie test leans on — a general truth direction, not one
     tuned to a single topic.
  3. COMMIT   slice every dataset's sweep cache at the chosen layer and save the
     canonical single-layer caches that 02/03/04 consume.

Phase 1 needs the GPU. Phases 2-3 are pure probe training off the cached tensors
and never touch the model — at 9B the sweep is 42 layers x 6 datasets x 2 probes
with LR at 2000 epochs, which is many minutes of holding an idle H100 if run in
the same process. So the phases are separately runnable:

    python scripts/01_extract_activations.py --phase extract   # GPU job
    python scripts/01_extract_activations.py --phase sweep     # CPU job
    python scripts/01_extract_activations.py                   # both (local dev)

After this, results/activations/<model>/<dataset>.pt holds the chosen layer, and
results/layer_sweep.json records the sweep so the choice is reproducible.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                     # noqa: E402
import data                       # noqa: E402
import lastping                   # noqa: E402
from activations import load_model, verify_model, extract_activations  # noqa: E402
from probes import MMProbe        # noqa: E402
from probes import LRProbe

# --------------------------------------------------------------------------- #
# CONFIG — split / position live here; the model is named in src/config.py so
# every script agrees on which model's caches they are reading.
# --------------------------------------------------------------------------- #
MODEL_NAME = config.MODEL_NAME
DATASETS = data.DATASETS
TOKEN_POS = -1                # end-of-statement token
BATCH_SIZE = 16
DEVICE = None                 # None -> cuda/mps/cpu auto
SWEEP_LAYERS = None           # None -> all layers; or e.g. range(8, 20)
# Pick the layer by mean accuracy across these datasets (structurally different
# topics + negations -> a general truth direction, not a topic-specific one).
SWEEP_DATASETS = DATASETS
# LR needs enough steps to converge on the ~2304-dim activations (the defaults
# under-train). Used only for the comparison curve; the chosen layer is MM's.
SWEEP_LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)
SELECTION_PROBE = "mm"        # curve the chosen layer is selected on
SPLIT_RATIO = 0.8
SEED = 0

_RESULTS = config.RESULTS_DIR
SWEEP_JSON = os.path.join(_RESULTS, "layer_sweep.json")


# --------------------------------------------------------------------------- #
def extract_phase(model, layers):
    """Extract all `layers` for every dataset; skip datasets already cached."""
    for i, name in enumerate(DATASETS, 1):
        if data.sweep_exists(name):
            print(f"  [skip] {name} (sweep cache exists)")
            continue
        texts, labels = data.load_statements(name)
        print(f"  extracting {name}: {len(texts)} statements x {len(layers)} layers ...",
              flush=True)
        acts = extract_activations(model, texts, layers=list(layers),
                                   token_pos=TOKEN_POS, batch_size=BATCH_SIZE)
        data.save_sweep(name, acts, labels, layers, TOKEN_POS, MODEL_NAME)
        print(f"    saved {tuple(acts.shape)}")
        # Progress for the job monitor. No-op when unconfigured, never raises.
        # IngestHeartbeat accepts only run_id / step / metric / text — the API
        # rejects anything else with a 422, so extra context goes in `text`.
        lastping.heartbeat(
            step=i,
            metric=str(len(texts)),
            text=f"extracted {name} ({i}/{len(DATASETS)}) @ {MODEL_NAME}",
        )


def sweep_phase(layers):
    """MM and LR accuracy per layer, averaged over SWEEP_DATASETS.

    Both probes train on the same seeded split at each (layer, dataset). Layer
    selection uses the SELECTION_PROBE curve (MM); the other is recorded for
    comparison.

    Returns:
        best_layer: layer with the highest mean SELECTION_PROBE accuracy.
        mm_table, lr_table: each {layer: {"mean": float, "per_dataset": {name: acc}}}.
    """
    # Load each sweep dataset's all-layer activations once, with its layer index.
    loaded = {}
    for name in SWEEP_DATASETS:
        acts, labels, meta = data.load_sweep(name)            # (n, L, d_model)
        loaded[name] = (acts, labels, {l: i for i, l in enumerate(meta["layers"])})

    mm_table, lr_table = {}, {}
    for l in layers:
        mm_pd, lr_pd = {}, {}
        for name, (acts, labels, layer_index) in loaded.items():
            col = acts[:, layer_index[l], :]                  # (n, d_model)
            tr_a, tr_l, te_a, te_l = data.train_test_split(col, labels, SPLIT_RATIO, SEED)
            mm_pd[name] = MMProbe.from_data(tr_a, tr_l).score(te_a, te_l)
            lr_pd[name] = LRProbe.from_data(tr_a, tr_l, **SWEEP_LR_KWARGS).score(te_a, te_l)
        mm_table[l] = {"mean": sum(mm_pd.values()) / len(mm_pd), "per_dataset": mm_pd}
        lr_table[l] = {"mean": sum(lr_pd.values()) / len(lr_pd), "per_dataset": lr_pd}

    selection = mm_table if SELECTION_PROBE == "mm" else lr_table
    best_layer = max(selection, key=lambda l: selection[l]["mean"])
    return best_layer, mm_table, lr_table


def commit_phase(best_layer):
    """Slice each dataset's sweep cache at best_layer -> canonical single-layer cache."""
    for name in DATASETS:
        acts, labels, meta = data.load_sweep(name)
        idx = meta["layers"].index(best_layer)
        data.save_activations(
            name, acts[:, idx, :], labels,
            layer=best_layer, token_pos=meta["token_pos"], model_name=meta["model_name"],
        )
        print(f"  committed {name} @ layer {best_layer}: {tuple(acts[:, idx, :].shape)}")


def run_extract():
    """Phase 1 (GPU): cache every layer for every dataset."""
    model = load_model(MODEL_NAME, device=DEVICE)
    info = verify_model(model)
    print(f"  verified: {info['n_layers']} layers, d_model={info['d_model']}, "
          f"device={info['device']}")

    layers = list(range(info["n_layers"])) if SWEEP_LAYERS is None else list(SWEEP_LAYERS)
    print("\n[1/3] extract")
    extract_phase(model, layers)


def run_sweep():
    """Phases 2-3 (CPU): sweep the cached layers, commit the chosen one.

    Takes the layer list from the cache metadata rather than from a loaded model,
    so this runs with no GPU and no model download.
    """
    missing = [n for n in DATASETS if not data.sweep_exists(n)]
    if missing:
        raise FileNotFoundError(
            f"No sweep cache for {missing} under model {MODEL_NAME!r}. "
            f"Run --phase extract first."
        )
    _, _, meta = data.load_sweep(DATASETS[0])
    layers = list(meta["layers"]) if SWEEP_LAYERS is None else list(SWEEP_LAYERS)

    print("\n[2/3] sweep")
    best_layer, mm_table, lr_table = sweep_phase(layers)
    print(f"    (mean accuracy across {len(SWEEP_DATASETS)} datasets)")
    print(f"    {'layer':>7}  {'mm':>6}  {'lr':>6}")
    for l in sorted(mm_table):
        mark = f"  <-- best ({SELECTION_PROBE})" if l == best_layer else ""
        print(f"    {l:7d}  {mm_table[l]['mean']:.4f}  {lr_table[l]['mean']:.4f}{mark}")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(SWEEP_JSON, "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "sweep_datasets": list(SWEEP_DATASETS),
                "token_pos": TOKEN_POS,
                "split_ratio": SPLIT_RATIO,
                "seed": SEED,
                "selection_probe": SELECTION_PROBE,
                "lr_hparams": SWEEP_LR_KWARGS,
                "chosen_layer": best_layer,
                "mm_accuracy_by_layer": {str(l): mm_table[l] for l in sorted(mm_table)},
                "lr_accuracy_by_layer": {str(l): lr_table[l] for l in sorted(lr_table)},
            },
            f, indent=2,
        )
    print(f"  chosen layer {best_layer} (by {SELECTION_PROBE}) -> {SWEEP_JSON}")

    print("\n[3/3] commit")
    commit_phase(best_layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["extract", "sweep", "all"], default="all",
                    help="extract = GPU forward passes; sweep = CPU probe "
                         "training + commit; all = both in one process.")
    args = ap.parse_args()

    print(f"== 01 [{args.phase}] | model={MODEL_NAME} ==")
    if args.phase in ("extract", "all"):
        run_extract()
    if args.phase in ("sweep", "all"):
        run_sweep()
    print("\nDone.")


if __name__ == "__main__":
    main()
