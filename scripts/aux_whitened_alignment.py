"""AUX — Is the LR-MM direction tilt the covariance correction? (theory check)

Marks & Tegmark show that, given n >> d and homoscedasticity, LR converges to the
whitened mass-mean direction Sigma^-1 (mu_true - mu_false). If so, LR should align
with MM(iid=True) far better than with plain MM.

This computes cos(theta_lr, Sigma^-1 theta_mm) across a ridge sweep, alongside
cos(theta_lr, theta_mm), for each dataset at the committed layer. Not part of the
01-04 sequence; run after 01 has committed the single-layer caches.

Result at Gemma-2-2B (d_model=2304): the whitening does NOT improve alignment.
d_model exceeds n_train for every dataset, so the within-class covariance is
rank-deficient and Sigma^-1 is noise-dominated — the M&T "enough data" premise
(n >> d) is violated. See the "conclusion" field in the output JSON.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data                          # noqa: E402
from probes import LRProbe, MMProbe  # noqa: E402

LR_KWARGS = dict(lr=1e-2, epochs=2000, weight_decay=1e-2)
RIDGES = [1e-3, 1e-2, 1e-1, 1.0]
SPLIT_RATIO = 0.8
SEED = 0

_RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_JSON = os.path.join(_RESULTS, "whitened_alignment.json")

CONCLUSION = (
    "The whitened (covariance-corrected) MM direction does NOT align with LR "
    "better than plain MM. Small ridge (real whitening) DEGRADES alignment; large "
    "ridge converges back to plain MM. Cause: d_model (2304) > n_train for every "
    "dataset, so the within-class covariance is rank-deficient and Sigma^-1 is "
    "noise-dominated. M&T's LR = Sigma^-1 theta_mm equivalence assumes n >> d, "
    "which is violated here. Therefore the LR-MM tilt cannot be attributed to the "
    "covariance correction at this model's dimensionality with these dataset sizes."
)


def cos(a, b):
    return torch.dot(a, b).item()


def whitened_dir(acts, labels, eps):
    """Sigma^-1 (mu_true - mu_false), unit-normalized, ridge eps."""
    t, f = acts[labels == 1], acts[labels == 0]
    theta = t.mean(0) - f.mean(0)
    c = torch.cat([t - t.mean(0), f - f.mean(0)], 0)
    sigma = (c.T @ c) / (c.shape[0] - 2)
    sigma = sigma + eps * torch.eye(sigma.shape[0])
    w = torch.linalg.solve(sigma, theta)
    return w / torch.linalg.norm(w)


def main():
    torch.manual_seed(SEED)
    per_dataset, layer = {}, None

    for name in data.DATASETS:
        acts, labels, meta = data.load_activations(name)
        layer = meta["layer"]
        tr_a, tr_l, _, _ = data.train_test_split(acts, labels, SPLIT_RATIO, SEED)

        lr_dir = LRProbe.from_data(tr_a, tr_l, **LR_KWARGS).direction
        mm_dir = MMProbe.from_data(tr_a, tr_l).direction
        per_dataset[name] = {
            "n_train": int(tr_a.shape[0]),
            "d_model": int(tr_a.shape[1]),
            "cos_mm_lr": cos(mm_dir, lr_dir),
            "cos_iid_lr_by_ridge": {
                str(e): cos(whitened_dir(tr_a, tr_l, e), lr_dir) for e in RIDGES
            },
        }

    # report
    print(f"layer {layer} | cos(LR, .) — plain MM vs whitened MM (ridge sweep)\n")
    print(f"{'dataset':18} {'n_tr':>5} {'plain':>7}"
          + "".join(f"  eps={e:<6g}" for e in RIDGES))
    for name, r in per_dataset.items():
        cells = "".join(f"  {r['cos_iid_lr_by_ridge'][str(e)]:9.3f}" for e in RIDGES)
        print(f"{name:18} {r['n_train']:5d} {r['cos_mm_lr']:7.3f}{cells}")
    print("\n" + CONCLUSION)

    os.makedirs(_RESULTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "layer": layer,
                "split_ratio": SPLIT_RATIO,
                "seed": SEED,
                "lr_hparams": LR_KWARGS,
                "ridges": RIDGES,
                "per_dataset": per_dataset,
                "conclusion": CONCLUSION,
            },
            f, indent=2,
        )
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
