"""AUX — Why cos(theta_A, theta_B) fails to predict combined-MM transfer.

P1 predicted combined-MM cross-dataset generalization is monotone in
cos(theta_A, theta_B). The 9B run refutes that as worded: cos_AB for
cities+neg_cities is essentially unchanged from 2B (-0.023 -> -0.076) while the
cross-topic cell moved 0.399 -> 0.986. Same geometry, opposite outcome.

This script shows the identity is NOT what failed — the cos PROXY is. For a
balanced concatenation the RAW mass-mean directions satisfy

    theta_combo = 1/2 (theta_A + theta_B)          [exact]

so the combined score on any x decomposes into NORM-WEIGHTED unit-direction scores:

    theta_combo . x  =  1/2 ( |theta_A| (dA . x) + |theta_B| (dB . x) )

cos_AB describes only the ANGLE between dA and dB. It carries no information
about |theta_A| vs |theta_B|, and it is that ratio — together with each score's
spread on the TEST set — that decides which constituent wins the rank ordering
that AUROC reads. Two runs can share a cos and differ completely in who dominates.

Reports, per (combo, test set): each constituent's unit-direction AUROC on the
test set, its raw norm, its score spread on that test set, the resulting
effective weight, and the reconstructed combined AUROC checked against the value
03b_combined_negation.py reported. CPU only, reads cached activations.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                                   # noqa: E402
import data                                     # noqa: E402

SPLIT_RATIO, SEED = 0.8, 0                      # must match 03b_combined_negation.py
COMBOS = {"cities+neg_cities": ("cities", "neg_cities"),
          "sp_en_trans+neg_sp_en_trans": ("sp_en_trans", "neg_sp_en_trans")}
CROSSTOPIC = {"cities+neg_cities": "neg_sp_en_trans",
              "sp_en_trans+neg_sp_en_trans": "neg_cities"}
_RESULTS = config.RESULTS_DIR
OUT_JSON = os.path.join(_RESULTS, "score_decomposition.json")


def auroc(scores, labels):
    """Identical to 03b's, so reconstructed numbers are comparable to reported ones."""
    labels = labels.long()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    return (float(ranks[labels == 1].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def raw_mm(acts, labels):
    """Raw (un-normalized) mass-mean direction. The norm is the whole point here."""
    return acts[labels == 1].mean(0) - acts[labels == 0].mean(0)


def main():
    layer = config.chosen_layer()
    names = sorted({n for c in COMBOS.values() for n in c} | set(CROSSTOPIC.values()))
    tr, te = {}, {}
    for name in names:
        acts, labels, meta = data.load_sweep(name)
        col = acts[:, meta["layers"].index(layer), :]
        a_tr, l_tr, a_te, l_te = data.train_test_split(col, labels, SPLIT_RATIO, SEED)
        tr[name], te[name] = (a_tr, l_tr), (a_te, l_te)

    out = {"model": config.MODEL_NAME, "layer": layer, "combos": {}}
    print(f"{config.MODEL_NAME}  layer {layer}\n")

    for combo, (A, B) in COMBOS.items():
        T = CROSSTOPIC[combo]
        rA, rB = raw_mm(*tr[A]), raw_mm(*tr[B])

        # Verify the identity on THIS data rather than asserting it: the exact
        # form needs the concatenation to be balanced in both class and part.
        acts = torch.cat([tr[A][0], tr[B][0]], 0)
        labels = torch.cat([tr[A][1], tr[B][1]], 0)
        r_combo = raw_mm(acts, labels)
        half_sum = 0.5 * (rA + rB)
        id_err = float((r_combo - half_sum).norm() / half_sum.norm())

        dA, dB = rA / rA.norm(), rB / rB.norm()
        a_te, l_te = te[T]
        sA, sB = a_te @ dA, a_te @ dB

        # Effective weight = raw norm x spread of that score on THIS test set.
        # Norm alone is not enough: a large direction that barely varies over the
        # test set cannot control the ranking.
        wA, wB = float(rA.norm()), float(rB.norm())
        eA, eB = wA * float(sA.std()), wB * float(sB.std())

        s_combo = a_te @ (r_combo / r_combo.norm())
        rec = auroc(s_combo, l_te)
        rec_from_parts = auroc(0.5 * (wA * sA + wB * sB), l_te)

        rec_d = {
            "test_set": T,
            "identity_rel_error": id_err,
            "cos_A_B": float(torch.dot(dA, dB)),
            "auroc_A_alone": auroc(sA, l_te),
            "auroc_B_alone": auroc(sB, l_te),
            "raw_norm_A": wA, "raw_norm_B": wB, "norm_ratio_A_over_B": wA / wB,
            "score_sd_A": float(sA.std()), "score_sd_B": float(sB.std()),
            "effective_weight_A": eA, "effective_weight_B": eB,
            "dominance_A_over_B": eA / eB,
            "auroc_combo_reconstructed": rec,
            "auroc_combo_from_weighted_parts": rec_from_parts,
        }
        out["combos"][combo] = rec_d

        print(f"=== {combo} -> {T} ===")
        print(f"  identity |theta_combo - 1/2(A+B)| / |1/2(A+B)| = {id_err:.2e}")
        print(f"  cos(dA,dB) {rec_d['cos_A_B']:+.3f}   "
              f"(carries NO norm information — that is the point)")
        print(f"  A alone -> {T}: AUROC {rec_d['auroc_A_alone']:.3f}   "
              f"|theta_A| {wA:.2f}   score sd {sA.std():.2f}   eff weight {eA:.2f}")
        print(f"  B alone -> {T}: AUROC {rec_d['auroc_B_alone']:.3f}   "
              f"|theta_B| {wB:.2f}   score sd {sB.std():.2f}   eff weight {eB:.2f}")
        print(f"  dominance A/B = {eA / eB:.3f}  "
              f"(-> {'A' if eA > eB else 'B'} controls the ranking)")
        print(f"  combined AUROC reconstructed {rec:.3f}  "
              f"(from weighted parts {rec_from_parts:.3f})\n")

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
