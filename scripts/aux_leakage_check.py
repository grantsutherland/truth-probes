"""AUX (Task D) — Leakage check on the perfect in-distribution accuracies.

LR is 1.0000 in-distribution on all six datasets. Plausible for simple facts, but
unverified. This checks, under the exact seeded split used everywhere (seed 0,
ratio 0.8), whether any statement string appears in BOTH train and test:
  * exact overlap (identical strings), and
  * near-duplicate overlap (identical after lowercasing, stripping punctuation,
    collapsing whitespace).
Also reports duplicate strings within the full dataset (the mechanism by which a
statement could land on both sides). No activations, no forward passes.
"""

import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config                     # noqa: E402
import data  # noqa: E402

SPLIT_RATIO, SEED = 0.8, 0
_RESULTS = config.RESULTS_DIR
OUT_JSON = os.path.join(_RESULTS, "leakage_check.json")


def normalize(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def split_indices(n):
    """Reproduce data.train_test_split's index split exactly."""
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    n_train = int(round(n * SPLIT_RATIO))
    return perm[:n_train].tolist(), perm[n_train:].tolist()


def main():
    out = {}
    print("== Task D: train/test leakage check (seed 0, ratio 0.8) ==\n")
    print(f"  {'dataset':18} {'n_tr':>5} {'n_te':>5} {'exact':>6} {'near-dup':>9} "
          f"{'dups_in_full':>13}")
    for name in data.DATASETS:
        texts, _ = data.load_statements(name)
        n = len(texts)
        tr_idx, te_idx = split_indices(n)
        tr = [texts[i] for i in tr_idx]
        te = [texts[i] for i in te_idx]

        tr_set, te_set = set(tr), set(te)
        exact = sum(1 for s in te_set if s in tr_set)

        tr_norm = {normalize(s) for s in tr}
        near = sum(1 for s in te if normalize(s) in tr_norm)

        dups_in_full = len(texts) - len(set(texts))

        out[name] = {
            "n_train": len(tr), "n_test": len(te),
            "exact_overlap_test_in_train": exact,
            "near_dup_overlap_test_in_train": near,
            "duplicate_strings_in_full_dataset": dups_in_full,
        }
        print(f"  {name:18} {len(tr):5d} {len(te):5d} {exact:6d} {near:9d} {dups_in_full:13d}")

    any_leak = any(v["exact_overlap_test_in_train"] or v["near_dup_overlap_test_in_train"]
                   for v in out.values())
    verdict = ("LEAKAGE DETECTED — the 1.0000 in-distribution accuracies are "
               "confounded; investigate before reporting."
               if any_leak else
               "NO LEAKAGE — no statement appears in both train and test (exact or "
               "near-duplicate) for any dataset. The 1.0000 in-distribution "
               "accuracies can be reported with confidence.")
    print("\n" + verdict)

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"split_ratio": SPLIT_RATIO, "seed": SEED,
                   "per_dataset": out, "verdict": verdict}, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
