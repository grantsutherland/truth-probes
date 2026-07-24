"""Single source of truth for model choice and the layers the analysis runs at.

Before this module, MODEL_NAME lived in 01's config block and the chosen layer was
hardcoded as `LAYER = 13` (or `LAYERS = [11, 13, 15]`) in four separate aux
scripts. That was fine while 2B was the only model, but layer 13 is a *2B* fact:
Gemma-2-9B has 42 layers, not 26, and its sweep will pick a different one. Any
script still saying 13 would silently analyse the wrong depth.

So the layer is no longer written down anywhere. It is *derived* from the sweep
that 01 already commits to `results/layer_sweep.json`:

    chosen_layer()   the layer 01 selected (peak MM accuracy)
    plateau_layers() the contiguous band around it where accuracy is saturated
    spot_layers()    a 3-layer sample across that band

Changing models is now a one-line edit to MODEL_NAME here, plus a re-run of 01.
"""

import json
import os

# --------------------------------------------------------------------------- #
# The one place the model is named.
# --------------------------------------------------------------------------- #
# The env override exists for CROSS-MODEL COMPARISON: re-running a 2B analysis to
# sit beside the 9B one otherwise means editing this line, running, and editing it
# back, which is exactly the kind of stateful edit that produced mislabelled
# results before. TRUTH_PROBES_MODEL=gemma-2-2b python scripts/... instead.
MODEL_NAME = os.environ.get("TRUTH_PROBES_MODEL", "gemma-2-9b")  # 2B = local dev run

# In-distribution accuracy saturates over a band of layers (at 2B: >0.97 across
# L11-17), so the sweep's argmax is not meaningfully better than its neighbours.
# Analyses that want "the plateau" take every layer within this much of the peak.
PLATEAU_TOL = 0.02

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Findings are scoped by model for the same reason the activation caches are:
# every script writes a fixed filename (transfer.json, probe_accuracy.json, ...),
# so running the 9B pipeline over an unscoped results/ would overwrite the 2B
# numbers that the write-up cites and that the 9B run exists to be compared
# against. Both sets now sit side by side and stay tracked in git.
RESULTS_DIR = os.path.join(_ROOT, "results", MODEL_NAME)
SWEEP_JSON = os.path.join(RESULTS_DIR, "layer_sweep.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


def _sweep():
    """The layer sweep 01 committed, or a pointed error if it hasn't run."""
    if not os.path.exists(SWEEP_JSON):
        raise FileNotFoundError(
            f"No layer sweep at {SWEEP_JSON}. Run 01_extract_activations "
            f"(extract, then sweep) before any downstream script."
        )
    with open(SWEEP_JSON) as f:
        sweep = json.load(f)

    # A sweep from a different model would hand back that model's layer indices
    # against this model's activations. Fail loudly rather than analyse depth 13
    # of a 42-layer model because a stale 2B file was lying around.
    if sweep.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"{SWEEP_JSON} was built from {sweep.get('model_name')!r} but "
            f"config.MODEL_NAME is {MODEL_NAME!r}. Re-run 01 for this model "
            f"(or move the old sweep aside to keep it)."
        )
    return sweep


def chosen_layer():
    """The layer 01 selected as the peak of the MM accuracy curve."""
    return int(_sweep()["chosen_layer"])


def plateau_layers(tol=PLATEAU_TOL):
    """The contiguous run of layers around the peak whose accuracy is within
    `tol` of it — the band over which the layer choice is not really a choice.

    Contiguity matters: thresholding alone could admit an unrelated early layer
    that happens to clear the bar, which is not what "the plateau" means.
    """
    sweep = _sweep()
    curve = {int(l): v["mean"] for l, v in sweep["mm_accuracy_by_layer"].items()}
    peak = chosen_layer()
    floor = curve[peak] - tol

    lo = hi = peak
    while lo - 1 in curve and curve[lo - 1] >= floor:
        lo -= 1
    while hi + 1 in curve and curve[hi + 1] >= floor:
        hi += 1
    return list(range(lo, hi + 1))


def spot_layers(offset=2, n=3):
    """`n` layers around the peak, for analyses too slow to run at every layer
    (combined-negation, negation mechanism) that still need to show a result is
    not peak-specific.

    Naively taking {peak-offset, peak, peak+offset} and clipping to the plateau
    silently returns fewer than `n` when the peak sits at a band edge — which is
    exactly what happens at 9B, where the peak (L17) is the plateau's lower bound
    and clipping collapsed three layers to two. So when one side is unavailable,
    step further out on the other side instead of dropping the layer.
    """
    band = plateau_layers()
    peak = chosen_layer()
    lo, hi = band[0], band[-1]

    picked = [peak]
    step = offset
    # Alternate outward from the peak, skipping steps that fall outside the band,
    # until n distinct layers are collected or the band is exhausted.
    while len(picked) < n and step <= (hi - lo):
        for cand in (peak - step, peak + step):
            if lo <= cand <= hi and cand not in picked:
                picked.append(cand)
                if len(picked) == n:
                    break
        step += offset
    return sorted(picked)
