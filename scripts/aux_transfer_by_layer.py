"""AUX — Transfer robustness across the accuracy plateau (layers 11-17).

Layer 13 was chosen as the argmax of in-distribution accuracy, which is SATURATED
across layers ~11-17 (all >0.976) and so cannot actually distinguish those layers.
But transfer is NOT flat over that band (e.g. cities->neg_cities is 0.006 at L11
vs 0.417 at L13). So Task A's banded counts and the 0.84 cross-topic figure could
be layer-13-contingent. This recomputes the full signed-AUROC transfer analysis at
each plateau layer to see whether the headlines are stable or depth-dependent.

Protocol matches 03: train on 80% split (seed 0) at layer L, eval signed AUROC on
the full target. No new forward passes (all off the sweep caches).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data                          # noqa: E402
from probes import LRProbe, MMProbe  # noqa: E402

LAYERS = list(range(11, 18))         # the accuracy plateau
LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)
SPLIT_RATIO, SEED = 0.8, 0
DATASETS = data.DATASETS
AFFIRMATIVE = ["cities", "sp_en_trans", "larger_than", "smaller_than"]
COMPLEMENT = {"larger_than->smaller_than", "smaller_than->larger_than"}

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "transfer_by_layer.json")


def auroc(scores, labels):
    labels = labels.long()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def train_dir(pt, a, l):
    return (MMProbe.from_data(a, l) if pt == "mm"
            else LRProbe.from_data(a, l, **LR_KWARGS)).direction


def analyze(matrix):
    """banded counts, cross-topic (aff->aff excl complements), key negation cell."""
    off = [matrix[a][b] for a in DATASETS for b in DATASETS if a != b]
    bands = {"ge0.9": sum(v >= 0.9 for v in off),
             "0.7-0.9": sum(0.7 <= v < 0.9 for v in off),
             "0.3-0.7": sum(0.3 <= v < 0.7 for v in off),
             "0.1-0.3": sum(0.1 <= v < 0.3 for v in off),
             "lt0.1": sum(v < 0.1 for v in off)}
    aff_nc = [matrix[a][b] for a in AFFIRMATIVE for b in AFFIRMATIVE
              if a != b and f"{a}->{b}" not in COMPLEMENT]
    return {
        "bands": bands,
        "crosstopic_aff_excl_complement": sum(aff_nc) / len(aff_nc),
        "cities_to_neg_cities": matrix["cities"]["neg_cities"],
    }


def main():
    torch.manual_seed(SEED)
    sweep = {}
    for name in DATASETS:
        acts, labels, meta = data.load_sweep(name)
        sweep[name] = (acts, labels, {l: i for i, l in enumerate(meta["layers"])})

    out = {"layers": LAYERS, "per_layer": {"mm": {}, "lr": {}}}
    for pt in ["mm", "lr"]:
        for L in LAYERS:
            dirs, full = {}, {}
            for name in DATASETS:
                acts, labels, idx = sweep[name]
                col = acts[:, idx[L], :]
                tr_a, tr_l, _, _ = data.train_test_split(col, labels, SPLIT_RATIO, SEED)
                dirs[name] = train_dir(pt, tr_a, tr_l)
                full[name] = (col, labels)
            matrix = {a: {b: auroc(full[b][0] @ dirs[a], full[b][1]) for b in DATASETS}
                      for a in DATASETS}
            out["per_layer"][pt][str(L)] = analyze(matrix)

    # report
    for pt in ["mm", "lr"]:
        print(f"\n[{pt}] transfer across the plateau (signed AUROC)")
        print(f"  {'layer':>5}  {'>=0.9':>5} {'<0.1':>5} {'0.3-0.7':>7}  "
              f"{'crosstopic':>10}  {'cities->neg_cities':>18}")
        for L in LAYERS:
            r = out["per_layer"][pt][str(L)]
            star = " *sel" if L == 13 else ""
            print(f"  {L:5d}  {r['bands']['ge0.9']:5d} {r['bands']['lt0.1']:5d} "
                  f"{r['bands']['0.3-0.7']:7d}  {r['crosstopic_aff_excl_complement']:10.3f}  "
                  f"{r['cities_to_neg_cities']:18.3f}{star}")

    # stability of the cross-topic headline
    print("\ncross-topic (aff excl complement) signed AUROC over L11-17:")
    for pt in ["mm", "lr"]:
        vals = [out["per_layer"][pt][str(L)]["crosstopic_aff_excl_complement"] for L in LAYERS]
        print(f"  {pt}: min {min(vals):.3f}  max {max(vals):.3f}  "
              f"range {max(vals)-min(vals):.3f}")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
