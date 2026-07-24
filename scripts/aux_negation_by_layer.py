"""AUX (Task B) — Is the negation asymmetry a layer-13 artifact?

At layer 13, negation cleanly INVERTS translation (sp_en<->neg_sp AUROC ~0.00)
but DEGRADES cities (cities<->neg_cities ~0.42). M&T document the negation
relation rotating across layers (antipodal early -> ~orthogonal mid -> aligning
late), so both observed values are consistent with "same phenomenon, different
phase." This sweeps every layer to tell the two apart.

For each layer 0..25, both probe types, off the all-layer sweep caches:
  1. Signed transfer AUROC for 4 directed pairs:
       cities->neg_cities, neg_cities->cities,
       sp_en_trans->neg_sp_en_trans, neg_sp_en_trans->sp_en_trans
  2. Cosine between the A-trained and B-trained probe directions for the 2
     undirected pairs (cleaner rotation measure: ~-1 antipodal, ~0 orthogonal,
     ~+1 aligned).

Protocol matches 03: train on 80% split (seed 0), eval AUROC on the FULL target.
Saves results/negation_by_layer.json. No new forward passes.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                     # noqa: E402
import data                          # noqa: E402
from probes import LRProbe, MMProbe  # noqa: E402

LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)
SPLIT_RATIO = 0.8
SEED = 0
PAIRS = [("cities", "neg_cities"), ("sp_en_trans", "neg_sp_en_trans")]
INVOLVED = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans"]

_RESULTS = config.RESULTS_DIR
OUT_JSON = os.path.join(_RESULTS, "negation_by_layer.json")


def auroc(scores, labels):
    labels = labels.long()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_dir(probe_type, acts, labels):
    if probe_type == "mm":
        return MMProbe.from_data(acts, labels).direction
    return LRProbe.from_data(acts, labels, **LR_KWARGS).direction


def main():
    torch.manual_seed(SEED)
    # load all-layer sweep caches once
    sweep = {}
    for name in INVOLVED:
        acts, labels, meta = data.load_sweep(name)
        sweep[name] = (acts, labels, {l: i for i, l in enumerate(meta["layers"])})
    layers = sorted(sweep["cities"][2].keys())

    out = {"layers": layers, "pairs": [f"{a}<->{b}" for a, b in PAIRS],
           "auroc": {"mm": {}, "lr": {}}, "cosine": {"mm": {}, "lr": {}}}

    for pt in ["mm", "lr"]:
        for L in layers:
            # train a direction on 80% of each involved dataset at layer L
            dirs, full = {}, {}
            for name in INVOLVED:
                acts, labels, idx = sweep[name]
                col = acts[:, idx[L], :]
                tr_a, tr_l, _, _ = data.train_test_split(col, labels, SPLIT_RATIO, SEED)
                dirs[name] = train_dir(pt, tr_a, tr_l)
                full[name] = (col, labels)                # eval AUROC on full target

            # 4 directed AUROC + 2 cosine
            for a, b in PAIRS:
                for src, dst in [(a, b), (b, a)]:
                    col, labels = full[dst]
                    out["auroc"][pt].setdefault(f"{src}->{dst}", {})[str(L)] = \
                        auroc(col @ dirs[src], labels)
                out["cosine"][pt].setdefault(f"{a}<->{b}", {})[str(L)] = \
                    float(torch.dot(dirs[a], dirs[b]))

    # ---- report ----
    print("== Task B: negation relationship across layers ==\n")
    for pt in ["mm", "lr"]:
        print(f"[{pt}] COSINE between A-trained and B-trained directions "
              f"(~-1 antipodal, ~0 orthogonal, ~+1 aligned)")
        print(f"  {'layer':>5}  {'cities<->neg_cities':>20}  {'sp_en<->neg_sp':>16}")
        for L in layers:
            c1 = out["cosine"][pt]["cities<->neg_cities"][str(L)]
            c2 = out["cosine"][pt]["sp_en_trans<->neg_sp_en_trans"][str(L)]
            print(f"  {L:5d}  {c1:20.3f}  {c2:16.3f}")
        print()

    for pt in ["mm", "lr"]:
        print(f"[{pt}] SIGNED transfer AUROC (4 directed pairs)")
        cols = ["cities->neg_cities", "neg_cities->cities",
                "sp_en_trans->neg_sp_en_trans", "neg_sp_en_trans->sp_en_trans"]
        print("  layer  " + "  ".join(f"{c.split('->')[0][:6]}>{c.split('->')[1][:6]}" for c in cols))
        for L in layers:
            vals = "  ".join(f"{out['auroc'][pt][c][str(L)]:13.3f}" for c in cols)
            print(f"  {L:5d}  {vals}")
        print()

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
