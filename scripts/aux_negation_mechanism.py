"""AUX — The averaging mechanism behind MM's combined-training failures.

For balanced concatenation, MM's combined direction is algebraically the average
of the constituent mass-mean vectors: theta_combo = 1/2 (theta_A + theta_B).
So MM combined training can only work when the constituents are roughly aligned:

    cos(theta_A, theta_B) = +1  -> combo ~= both            -> works
    cos = 0 (orthogonal)        -> combo at 45deg to both    -> works on own held-out
                                   (cos .707 each) but not a 3rd dataset
    cos = -1 (antipodal)        -> theta_A+theta_B ~= 0      -> combo is NOISE,
                                   fails even IN-DISTRIBUTION

This candidate-explains M&T's open 7.1 question: larger+smaller become ALIGNED at
70B (Fig 3) so MM combined works there; cities+neg_cities stay ORTHOGONAL (Fig 3c)
so MM combined fails. Both fall out of the same provable constraint.

Within-2B test at layer 13: three pairs with different cos(A,B). Report cos(A,B),
the norm-retention ratio |theta_A+theta_B| / (|theta_A|+|theta_B|), and the
combined-MM held-out AUROC on its OWN constituents. Prediction: larger+smaller
(antipodal) fails on its own held-out; cities+neg_cities (orthogonal) does not.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data                   # noqa: E402
from probes import MMProbe    # noqa: E402

LAYERS = [11, 13, 15]
SPLIT_RATIO, SEED = 0.8, 0
PAIRS = [("cities", "neg_cities"), ("sp_en_trans", "neg_sp_en_trans"),
         ("larger_than", "smaller_than")]

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "negation_mechanism.json")


def auroc(scores, labels):
    labels = labels.long()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def raw_theta(acts, labels):
    """Un-normalized mass-mean vector mu_true - mu_false."""
    return acts[labels == 1].mean(0) - acts[labels == 0].mean(0)


def main():
    torch.manual_seed(SEED)
    sweep = {}
    for name in {d for p in PAIRS for d in p}:
        acts, labels, meta = data.load_sweep(name)
        sweep[name] = (acts, labels, {l: i for i, l in enumerate(meta["layers"])})

    out = {"layers": LAYERS, "pairs": {}}
    print("== MM combined-training averaging mechanism (layer-by-pair) ==")
    print("cos_AB: direction similarity of constituents.  norm_ratio: |A+B|/(|A|+|B|)")
    print("combo AUROC on OWN held-out A / B (folded shown in parens).\n")

    for A, B in PAIRS:
        key = f"{A}+{B}"
        out["pairs"][key] = {}
        print(f"[{key}]")
        print(f"  {'layer':>5} {'cos_AB':>7} {'norm_ratio':>10}  "
              f"{'combo->A':>12}  {'combo->B':>12}")
        for L in LAYERS:
            def col(name):
                acts, labels, idx = sweep[name]
                return acts[:, idx[L], :], labels

            # 80/20 split each; train combined on 80% portions
            aA, lA = col(A); aB, lB = col(B)
            aA_tr, lA_tr, aA_te, lA_te = data.train_test_split(aA, lA, SPLIT_RATIO, SEED)
            aB_tr, lB_tr, aB_te, lB_te = data.train_test_split(aB, lB, SPLIT_RATIO, SEED)

            tA, tB = raw_theta(aA_tr, lA_tr), raw_theta(aB_tr, lB_tr)
            cos_ab = float(torch.dot(tA / tA.norm(), tB / tB.norm()))
            norm_ratio = float((tA + tB).norm() / (tA.norm() + tB.norm()))

            combo = MMProbe.from_data(torch.cat([aA_tr, aB_tr]),
                                      torch.cat([lA_tr, lB_tr]))
            d = combo.direction
            aur_a = auroc(aA_te @ d, lA_te)
            aur_b = auroc(aB_te @ d, lB_te)

            out["pairs"][key][str(L)] = {
                "cos_AB": cos_ab, "norm_ratio": norm_ratio,
                "combo_auroc_own_A": aur_a, "combo_auroc_own_B": aur_b,
            }
            fa, fb = max(aur_a, 1 - aur_a), max(aur_b, 1 - aur_b)
            print(f"  {L:5d} {cos_ab:7.3f} {norm_ratio:10.3f}  "
                  f"{aur_a:6.3f}({fa:.2f})  {aur_b:6.3f}({fb:.2f})")
        print()

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
