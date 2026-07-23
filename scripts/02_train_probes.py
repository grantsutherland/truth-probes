"""02 — Train LR and MM probes at the chosen layer (H1, in-distribution).

For each dataset, train both probe types on an 80% split at the committed layer,
report held-out accuracy, and record the cosine similarity between the LR and MM
directions (do they find the same axis?). Trained probes are saved so
03_transfer_eval and 04_induced_lie_test reuse them without retraining.

Reads the single-layer caches committed by 01_extract_activations. The layer is
whatever 01 committed (read from cache metadata, not hardcoded).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data                       # noqa: E402
from probes import LRProbe, MMProbe  # noqa: E402

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DATASETS = data.DATASETS
PROBE_TYPES = ["mm", "lr"]
LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)  # match the sweep
SPLIT_RATIO = 0.8
SEED = 0
CENTER = False                # subtract train-set mean before probing (M&T do)

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "probe_accuracy.json")


def train_one(probe_type, tr_a, tr_l):
    if probe_type == "mm":
        return MMProbe.from_data(tr_a, tr_l)
    if probe_type == "lr":
        return LRProbe.from_data(tr_a, tr_l, **LR_KWARGS)
    raise ValueError(f"unknown probe type {probe_type!r}")


def main():
    torch.manual_seed(SEED)  # deterministic LR init across the run
    print("== 02 train probes (in-distribution) ==")

    results = {}       # dataset -> {probe_type -> acc}
    directions = {}    # dataset -> {probe_type -> unit direction}
    layer = None

    for name in DATASETS:
        acts, labels, meta = data.load_activations(name)
        layer = meta["layer"]
        tr_a, tr_l, te_a, te_l = data.train_test_split(acts, labels, SPLIT_RATIO, SEED)

        offset = None
        if CENTER:
            offset = data.compute_mean_offset(tr_a)
            tr_a, te_a = data.center(tr_a, offset), data.center(te_a, offset)

        results[name], directions[name] = {}, {}
        for pt in PROBE_TYPES:
            probe = train_one(pt, tr_a, tr_l)
            results[name][pt] = probe.score(te_a, te_l)
            directions[name][pt] = probe.direction
            data.save_probe(probe, pt, name, layer, meta["model_name"], mean_offset=offset)

    # LR vs MM direction agreement per dataset
    cos = {
        name: torch.dot(directions[name]["mm"], directions[name]["lr"]).item()
        for name in DATASETS
        if "mm" in PROBE_TYPES and "lr" in PROBE_TYPES
    }

    # ---- report ----
    print(f"\nlayer {layer} | held-out accuracy (split {SPLIT_RATIO}, seed {SEED})"
          + ("  [centered]" if CENTER else ""))
    header = f"  {'dataset':18} " + "  ".join(f"{pt:>6}" for pt in PROBE_TYPES)
    if cos:
        header += "   cos(mm,lr)"
    print(header)
    for name in DATASETS:
        row = f"  {name:18} " + "  ".join(f"{results[name][pt]:.4f}" for pt in PROBE_TYPES)
        if cos:
            row += f"      {cos[name]:.3f}"
        print(row)

    for pt in PROBE_TYPES:
        mean_acc = sum(results[n][pt] for n in DATASETS) / len(DATASETS)
        print(f"  {'MEAN ' + pt:18} {mean_acc:.4f}")

    # ---- save ----
    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "layer": layer,
                "split_ratio": SPLIT_RATIO,
                "seed": SEED,
                "centered": CENTER,
                "lr_hparams": LR_KWARGS,
                "accuracy": results,
                "cos_mm_lr": cos,
            },
            f, indent=2,
        )
    print(f"\nsaved -> {OUT_JSON}")
    print(f"probes -> {data.PROBE_DIR}")


if __name__ == "__main__":
    main()
