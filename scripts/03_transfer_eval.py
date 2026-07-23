"""03 — Transfer evaluation (H1): does a probe find a GENERAL truth direction?

Take each probe trained by 02 (no retraining) and evaluate it on every OTHER
dataset. The question is about the DIRECTION, so the headline metric is AUROC:

  * AUROC is threshold-free and invariant to any constant shift of the
    activations, so it measures only whether the direction ranks true statements
    above false ones. It is immune to the bias-transfer artifact that makes plain
    accuracy read exactly 0.500 (constant prediction) when a probe's trained
    threshold lands off a new dataset's projection distribution.
  * AUROC ~1.0 = direction transfers; ~0.0 = transfers but sign-inverted (e.g.
    negation, or larger_than vs smaller_than); ~0.5 = genuinely no shared signal.
  * "informativeness" = max(auroc, 1-auroc) folds the sign away: how well the
    direction separates, regardless of which way it points.

We also report accuracy with each TEST set centered by its own mean (label-free,
M&T's protocol) and an unbiased zero threshold — this is the fair accuracy once
the bias-transfer artifact is removed. Raw (trained-bias) accuracy is kept in the
JSON only for the record.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data  # noqa: E402

DATASETS = data.DATASETS
PROBE_TYPES = ["mm", "lr"]

# Affirmative (no "not") vs negated. Used to isolate topic transfer from the
# negation confound. NOTE larger_than/smaller_than are affirmative but logical
# COMPLEMENTS of each other, so they invert even within aff->aff.
AFFIRMATIVE = ["cities", "sp_en_trans", "larger_than", "smaller_than"]

# Non-overlapping bands over signed AUROC, low->high.
BANDS = [("<0.1", -0.01, 0.1), ("0.1-0.3", 0.1, 0.3), ("0.3-0.7", 0.3, 0.7),
         ("0.7-0.9", 0.7, 0.9), (">=0.9", 0.9, 1.01)]

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "transfer.json")


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney). Ties negligible for continuous margins."""
    labels = labels.long()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def centered_accuracy(direction, acts, labels):
    """Accuracy with the TEST set centered by its own mean, unbiased 0 threshold."""
    proj = (acts - acts.mean(0)) @ direction
    pred = (proj >= 0).float()
    return float((pred == labels).float().mean())


def evaluate(probe_type):
    """Per (train, test): auroc, centered accuracy, raw (trained-bias) accuracy."""
    auroc_m, cacc_m, racc_m = {}, {}, {}
    for train_name in DATASETS:
        probe, meta = data.load_probe(probe_type, train_name)
        direction = probe.direction
        auroc_m[train_name], cacc_m[train_name], racc_m[train_name] = {}, {}, {}
        for test_name in DATASETS:
            acts, labels, _ = data.load_activations(test_name)
            proj = acts @ direction
            auroc_m[train_name][test_name] = auroc(proj, labels)
            cacc_m[train_name][test_name] = centered_accuracy(direction, acts, labels)
            racc_m[train_name][test_name] = probe.score(acts, labels)
    return auroc_m, cacc_m, racc_m


def off_diag(matrix, transform=lambda x: x):
    vals = [transform(matrix[a][b]) for a in DATASETS for b in DATASETS if a != b]
    return sum(vals) / len(vals)


def band_counts(matrix):
    """Count off-diagonal cells falling in each signed-AUROC band."""
    vals = [matrix[a][b] for a in DATASETS for b in DATASETS if a != b]
    counts = {label: 0 for label, _, _ in BANDS}
    for v in vals:
        for label, lo, hi in BANDS:
            if lo <= v < hi:
                counts[label] += 1
                break
    return counts, len(vals)


def aff_to_aff(matrix):
    """{'train->test': auroc} for affirmative->affirmative off-diagonal cells."""
    return {
        f"{a}->{b}": matrix[a][b]
        for a in AFFIRMATIVE for b in AFFIRMATIVE if a != b
    }


def print_matrix(title, matrix):
    short = {n: n[:9] for n in DATASETS}
    print(f"\n[{title}] rows=train, cols=test")
    print("  " + " " * 18 + "".join(f"{short[c]:>11}" for c in DATASETS))
    for a in DATASETS:
        print(f"  {a:18}" + "".join(f"{matrix[a][b]:11.3f}" for b in DATASETS))


def main():
    print("== 03 transfer eval (H1) — AUROC-first ==")
    out = {"datasets": list(DATASETS), "auroc": {}, "centered_acc": {},
           "raw_acc": {}, "summary": {}}

    for pt in PROBE_TYPES:
        auroc_m, cacc_m, racc_m = evaluate(pt)
        out["auroc"][pt], out["centered_acc"][pt], out["raw_acc"][pt] = auroc_m, cacc_m, racc_m

        fold = lambda a: max(a, 1 - a)
        print_matrix(f"{pt} AUROC", auroc_m)
        print_matrix(f"{pt} centered-acc", cacc_m)

        # Signed = operationally honest (probe used as-is). Folded uses TEST
        # labels to pick each direction's sign -> SIGN-ORACLE, not a real transfer.
        bands, n_off = band_counts(auroc_m)
        aff = aff_to_aff(auroc_m)
        aff_mean = sum(aff.values()) / len(aff)
        # larger_than<->smaller_than are logical complements (clean inversion),
        # not a topic-transfer failure; exclude them for pure topic transfer.
        complement = {"larger_than->smaller_than", "smaller_than->larger_than"}
        aff_nc = {k: v for k, v in aff.items() if k not in complement}
        aff_nc_mean = sum(aff_nc.values()) / len(aff_nc)
        out["summary"][pt] = {
            "auroc_signed_mean": off_diag(auroc_m),
            "auroc_folded_mean_SIGN_ORACLE": off_diag(auroc_m, fold),
            "centered_acc_signed_mean": off_diag(cacc_m),
            "centered_acc_folded_mean_SIGN_ORACLE": off_diag(cacc_m, fold),
            "raw_acc_signed_mean": off_diag(racc_m),
            "auroc_signed_bands_offdiag": bands,          # the honest structural summary
            "n_offdiag_cells": n_off,
            "auroc_signed_mean_aff_to_aff": aff_mean,     # topic transfer, negation removed
            "auroc_signed_mean_aff_to_aff_excl_complements": aff_nc_mean,  # pure topic transfer
            "aff_to_aff_cells": aff,
        }

    print("\n" + "=" * 78)
    print("SUMMARY (off-diagonal / transfer). SIGNED = probe used as-is (honest).")
    print("FOLDED = max(a,1-a), uses TEST labels to pick sign -> SIGN-ORACLE, NOT")
    print("real transfer. M&T report SIGNED transfer accuracy: only signed numbers")
    print("are comparable to their table.")
    print("=" * 78)
    for pt in PROBE_TYPES:
        s = out["summary"][pt]
        print(f"\n[{pt}]")
        print(f"  AUROC          signed {s['auroc_signed_mean']:.4f}   "
              f"| folded {s['auroc_folded_mean_SIGN_ORACLE']:.4f} [SIGN-ORACLE]")
        print(f"  centered-acc   signed {s['centered_acc_signed_mean']:.4f}   "
              f"| folded {s['centered_acc_folded_mean_SIGN_ORACLE']:.4f} [SIGN-ORACLE]")
        print(f"  raw-acc        signed {s['raw_acc_signed_mean']:.4f}   "
              f"(trained bias; confounded by bias-transfer)")

    print("\n" + "=" * 78)
    print("SIGNED AUROC — banded counts over the 30 off-diagonal cells")
    print("(the signed mean sits in the empty valley between the two spikes)")
    print("=" * 78)
    for pt in PROBE_TYPES:
        s = out["summary"][pt]
        b = s["auroc_signed_bands_offdiag"]
        cells = "  ".join(f"{label}: {b[label]:>2d}" for label, _, _ in BANDS)
        print(f"  [{pt}]  {cells}   (n={s['n_offdiag_cells']})")

    print("\naffirmative->affirmative signed AUROC (topic transfer, negation removed;")
    print("larger<->smaller are logical complements and still invert):")
    for pt in PROBE_TYPES:
        s = out["summary"][pt]
        print(f"  [{pt}] mean {s['auroc_signed_mean_aff_to_aff']:.4f}  "
              f"| excl. larger<->smaller complements: "
              f"{s['auroc_signed_mean_aff_to_aff_excl_complements']:.4f}")
        for cell, v in s["aff_to_aff_cells"].items():
            tag = "  <- complement inversion" if v < 0.1 else ""
            print(f"       {cell:34} {v:.3f}{tag}")

    out["metric_notes"] = (
        "Matrices (auroc/centered_acc/raw_acc) are UNFOLDED: raw signs preserved. "
        "Folding appears ONLY in summary keys tagged SIGN_ORACLE and uses test "
        "labels to choose each direction's sign. M&T report signed transfer "
        "accuracy; only *_signed_mean is comparable to their table."
    )
    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
