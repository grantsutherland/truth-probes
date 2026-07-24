"""Task C (fixed) — Complete H1: training on statements PLUS their negations.

M&T: training on a dataset with its negations improves generalization, because
the shared truth component survives while nuisance features that correlate with
truth in the affirmative set anti-correlate in the negated set and cancel.

FIXES over the first pass:
  * HELD-OUT split. Everything is evaluated on the seeded 20% held-out portion of
    the test dataset; combos train on the 80% portions. The earlier own-negation
    1.000s were train-on-test and are discarded.
  * SIGNED and FOLDED both reported. Folded = max(a,1-a) = information retained
    regardless of sign. AUROC 0.000 is FULL information (inverted); 0.5 is NO
    information. So MM 0.000->0.465 is an information COLLAPSE (folded 1.000->0.535),
    not a +0.464 gain. Only signed >0.5 with high folded is real usable transfer.
  * MECHANISM CHECK. For balanced concatenation MM's combined direction is
    algebraically ~= average of the two constituent directions,
    theta_combo ~= 1/2 (theta_A + theta_B). If theta_A _|_ theta_B (orthogonal),
    the average sits at ~45deg to both (cos ~0.707) and is good for neither. We
    verify cos(theta_combo, theta_A) and cos(theta_combo, theta_B) directly.
  * MULTIPLE LAYERS (11, 13, 15). Layer 13 is the worst layer for cities-negation
    by construction (global orthogonality extremum), so the MM/LR gap must be
    checked off-13 before it counts as a finding rather than a slice.

Off the all-layer sweep caches; no new forward passes.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                        # noqa: E402
import data                          # noqa: E402
from probes import LRProbe, MMProbe  # noqa: E402

LAYERS = config.spot_layers()
LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)
SPLIT_RATIO, SEED = 0.8, 0
DATASETS = data.DATASETS
COMBOS = {"cities+neg_cities": ("cities", "neg_cities"),
          "sp_en_trans+neg_sp_en_trans": ("sp_en_trans", "neg_sp_en_trans")}
BASELINE_OF = {"cities+neg_cities": "cities",
               "sp_en_trans+neg_sp_en_trans": "sp_en_trans"}
# The VALID, informative cells for each combo: cross-topic negation target
# (headline; M&T's specific claim) + own negation (now held-out).
CROSSTOPIC = {"cities+neg_cities": "neg_sp_en_trans",
              "sp_en_trans+neg_sp_en_trans": "neg_cities"}

_RESULTS = config.RESULTS_DIR
OUT_JSON = os.path.join(_RESULTS, "combined_negation.json")


def auroc(scores, labels):
    labels = labels.long()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fold(a):
    return max(a, 1 - a)


def train_dir(pt, acts, labels):
    return (MMProbe.from_data(acts, labels) if pt == "mm"
            else LRProbe.from_data(acts, labels, **LR_KWARGS)).direction


def main():
    torch.manual_seed(SEED)
    sweep = {}
    for name in DATASETS:
        acts, labels, meta = data.load_sweep(name)
        sweep[name] = (acts, labels, {l: i for i, l in enumerate(meta["layers"])})

    out = {"layers": LAYERS, "protocol": "held-out 20% (seed 0)", "per_layer": {}}

    for L in LAYERS:
        tr, te = {}, {}
        for name in DATASETS:
            acts, labels, idx = sweep[name]
            col = acts[:, idx[L], :]
            a_tr, l_tr, a_te, l_te = data.train_test_split(col, labels, SPLIT_RATIO, SEED)
            tr[name], te[name] = (a_tr, l_tr), (a_te, l_te)

        layer_out = {"mm": {}, "lr": {}}
        for pt in ["mm", "lr"]:
            base_dir = {b: train_dir(pt, *tr[b]) for b in set(BASELINE_OF.values())}
            combo_dir, cosines = {}, {}
            for combo, (A, B) in COMBOS.items():
                acts = torch.cat([tr[A][0], tr[B][0]], 0)
                labels = torch.cat([tr[A][1], tr[B][1]], 0)
                combo_dir[combo] = train_dir(pt, acts, labels)
                dA, dB = train_dir(pt, *tr[A]), train_dir(pt, *tr[B])
                cosines[combo] = {
                    "cos_combo_A": float(torch.dot(combo_dir[combo], dA)),
                    "cos_combo_B": float(torch.dot(combo_dir[combo], dB)),
                    "cos_A_B": float(torch.dot(dA, dB)),
                }

            def cell(direction, test_name):
                a_te, l_te = te[test_name]
                s = auroc(a_te @ direction, l_te)
                return {"signed": s, "folded": fold(s)}

            layer_out[pt] = {
                "baseline": {b: {t: cell(base_dir[b], t) for t in DATASETS}
                             for b in set(BASELINE_OF.values())},
                "combo": {c: {t: cell(combo_dir[c], t) for t in DATASETS}
                          for c in COMBOS},
                "direction_cosines": cosines,
            }
        out["per_layer"][str(L)] = layer_out

    # ---- report: valid cells only, cross-topic first ----
    print("== Task C (fixed): combined negation training, HELD-OUT 20% ==")
    print("cells shown signed/folded. folded=info retained (0.5=none, 1.0=full).")
    for L in LAYERS:
        print(f"\n================= LAYER {L} =================")
        for pt in ["mm", "lr"]:
            lo = out["per_layer"][str(L)][pt]
            print(f"\n[{pt}]")
            for combo, (A, B) in COMBOS.items():
                base = BASELINE_OF[combo]
                xt = CROSSTOPIC[combo]
                print(f"  {combo}")
                for label, test in [("CROSS-TOPIC->" + xt, xt), ("own-neg->" + B, B)]:
                    bc = lo["baseline"][base][test]
                    cc = lo["combo"][combo][test]
                    print(f"    {label:28} base {bc['signed']:.3f}/{bc['folded']:.3f}"
                          f"  ->  combo {cc['signed']:.3f}/{cc['folded']:.3f}")
                if pt == "mm":
                    c = lo["direction_cosines"][combo]
                    print(f"    theta cos: combo.A {c['cos_combo_A']:.3f}  "
                          f"combo.B {c['cos_combo_B']:.3f}  (A.B {c['cos_A_B']:.3f})"
                          f"  [~0.707 both => 45deg avg of orthogonal dirs]")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
