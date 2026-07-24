"""Dataset loading and the activation-cache contract.

This module sits at two points in the pipeline:

  1. Raw statements (text + label) from the Marks & Tegmark CSVs, consumed by
     `01_extract_activations` to feed the model (Gemma-2-2B in dev, Gemma-2-9B
     for final numbers).
  2. Cached residual-stream activations (tensors on disk), consumed by
     `02_train_probes`, `03_transfer_eval`, and `04_induced_lie_test`.

The cache schema written by `save_activations` is a contract shared by every
downstream script — change it here and re-run extraction, never patch it
piecemeal in a consumer.

Conventions fixed here so the rest of the codebase can stay dumb about them:
  * activations are returned as float32 (probes never see fp16),
  * labels are returned as float (matches `score` and BCEWithLogitsLoss),
  * splits are seeded (the README's reproducibility promise).
"""

import os

import pandas as pd
import torch

import config

# Repo root = parent of this file's directory (src/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
# Caches are scoped BY MODEL. Layer indices and d_model are model-specific, and
# `sweep_exists` is what 01 uses to skip already-extracted datasets — an unscoped
# path meant a 2B cache would satisfy that check during a 9B run, silently
# skipping extraction and leaving every downstream number computed on 2304-dim
# 2B activations under a 9B label. Scoping keeps both models' caches side by side.
CACHE_DIR = os.path.join(_ROOT, "results", "activations", config.MODEL_NAME)
# Intermediate all-layer caches from the extraction sweep (Option 1): extract
# every layer once here, pick the best layer offline, then commit that single
# layer to CACHE_DIR via save_activations. These are throwaway once a layer is chosen.
SWEEP_DIR = os.path.join(CACHE_DIR, "sweep")

# Base true/false datasets from saprmarks/geometry-of-truth. The induced-lie
# set is loaded by name too but kept clearly separate in analysis.
DATASETS = [
    "cities",
    "neg_cities",
    "sp_en_trans",
    "neg_sp_en_trans",
    "larger_than",
    "smaller_than",
]

# The current cache-schema version. Bump when the dict layout changes so a
# stale cache fails loudly instead of loading into the wrong shape.
CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# Raw statements  (step 01)
# --------------------------------------------------------------------------- #
def load_statements(name):
    """Load raw statements for one dataset.

    Reads `data/<name>.csv`, expecting at least a `statement` column and a
    `label` column (0/1), the Marks & Tegmark format. Extra columns are ignored.

    Returns:
        texts:  list[str], one statement per row.
        labels: torch.LongTensor of shape (n,), 1 = true, 0 = false.

    The order of `texts` and `labels` is preserved and must stay aligned with
    the activation rows produced downstream.
    """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No CSV for dataset '{name}' at {path}. "
            f"Download the geometry-of-truth data into {DATA_DIR}/."
        )

    df = pd.read_csv(path)
    missing = {"statement", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {sorted(missing)}; "
            f"found {list(df.columns)}."
        )
    if len(df) == 0:
        raise ValueError(f"{path} contained no rows.")

    texts = df["statement"].astype(str).tolist()
    labels = _coerce_labels(df["label"], path)
    return texts, torch.tensor(labels, dtype=torch.long)


def _coerce_labels(col, path):
    """Coerce a label column to a list of ints, erroring on non-0/1 values."""
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.isna().any():
        bad = col[numeric.isna()].unique()[:5]
        raise ValueError(f"{path}: non-numeric label(s) {list(bad)}.")
    if not numeric.isin([0, 1]).all():
        bad = numeric[~numeric.isin([0, 1])].unique()[:5]
        raise ValueError(f"{path}: labels must be 0 or 1, got {list(bad)}.")
    return numeric.astype(int).tolist()


# --------------------------------------------------------------------------- #
# Activation cache  (steps 02-04)
# --------------------------------------------------------------------------- #
def save_activations(name, acts, labels, layer, token_pos, model_name,
                     mean_offset=None):
    """Cache activations for one dataset to `results/activations/<name>.pt`.

    Forward passes are the slow step; everything downstream reads this cache.
    Stored as a single dict — the schema here is the contract for `load_activations`.

    Args:
        acts:        Tensor (n, d_model). Stored as-is; dtype is normalized on load.
        labels:      Tensor (n,), 1/0.
        layer:       int, residual-stream layer the activations came from.
        token_pos:   int, token position (e.g. -1 for end-of-statement).
        model_name:  str, e.g. "gemma-2-2b" (dev) or "gemma-2-9b" (final).
        mean_offset: optional Tensor (d_model,), the centering offset computed on
                     TRAINING data. Persisted so `03`/`04` apply the same offset
                     rather than recomputing a different mean on new data.
    """
    if acts.shape[0] != labels.shape[0]:
        raise ValueError(
            f"acts/labels length mismatch: {acts.shape[0]} vs {labels.shape[0]}."
        )
    os.makedirs(CACHE_DIR, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "dataset": name,
        "acts": acts.cpu(),
        "labels": labels.cpu(),
        "layer": layer,
        "token_pos": token_pos,
        "model_name": model_name,
        "mean_offset": None if mean_offset is None else mean_offset.cpu(),
    }
    torch.save(payload, _cache_path(name))


def load_activations(name, device=None):
    """Load cached activations for one dataset.

    Normalizes dtype at this boundary: activations -> float32, labels -> float,
    so probes and scoring never worry about it.

    Returns:
        acts:   Tensor (n, d_model), float32, on `device` (CPU if None).
        labels: Tensor (n,), float.
        meta:   dict with layer, token_pos, model_name, mean_offset, dataset.
    """
    path = _cache_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No activation cache for '{name}' at {path}. Run 01_extract_activations first."
        )
    # Our own trusted file; contains only tensors and primitives.
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)

    if payload.get("version") != CACHE_VERSION:
        raise ValueError(
            f"{path} has cache version {payload.get('version')}, "
            f"expected {CACHE_VERSION}. Re-run extraction."
        )

    acts = payload["acts"].to(dtype=torch.float32)
    labels = payload["labels"].to(dtype=torch.float32)
    meta = {
        "dataset": payload["dataset"],
        "layer": payload["layer"],
        "token_pos": payload["token_pos"],
        "model_name": payload["model_name"],
        "mean_offset": payload["mean_offset"],
    }
    return acts, labels, meta


def _cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}.pt")


def save_sweep(name, acts, labels, layers, token_pos, model_name):
    """Cache all-layer activations for the layer sweep (Option 1 intermediate).

    Args:
        acts:   Tensor (n, n_layers, d_model) — every swept layer, one forward pass.
        labels: Tensor (n,).
        layers: list[int], the layer index for each slot along dim 1.
        token_pos, model_name: as in save_activations.
    """
    if acts.shape[0] != labels.shape[0]:
        raise ValueError(
            f"acts/labels length mismatch: {acts.shape[0]} vs {labels.shape[0]}."
        )
    if acts.shape[1] != len(layers):
        raise ValueError(
            f"acts layer axis {acts.shape[1]} != len(layers) {len(layers)}."
        )
    os.makedirs(SWEEP_DIR, exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "dataset": name,
            "acts": acts.cpu(),
            "labels": labels.cpu(),
            "layers": list(layers),
            "token_pos": token_pos,
            "model_name": model_name,
        },
        _sweep_path(name),
    )


def load_sweep(name, device=None):
    """Load all-layer sweep activations.

    Returns:
        acts:   Tensor (n, n_layers, d_model), float32.
        labels: Tensor (n,), float.
        meta:   dict with layers, token_pos, model_name, dataset.
    """
    path = _sweep_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No sweep cache for '{name}' at {path}. Run 01_extract_activations first."
        )
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(
            f"{path} has cache version {payload.get('version')}, "
            f"expected {CACHE_VERSION}. Re-run extraction."
        )
    acts = payload["acts"].to(dtype=torch.float32)
    labels = payload["labels"].to(dtype=torch.float32)
    meta = {
        "dataset": payload["dataset"],
        "layers": payload["layers"],
        "token_pos": payload["token_pos"],
        "model_name": payload["model_name"],
    }
    return acts, labels, meta


def sweep_exists(name):
    return os.path.exists(_sweep_path(name))


def _sweep_path(name):
    return os.path.join(SWEEP_DIR, f"{name}.pt")


# --------------------------------------------------------------------------- #
# Trained probes  (02 saves; 03/04 reload without retraining)
# --------------------------------------------------------------------------- #
PROBE_DIR = os.path.join(_ROOT, "results", "probes", config.MODEL_NAME)


def save_probe(probe, probe_type, dataset, layer, model_name, mean_offset=None):
    """Persist a trained probe. `mean_offset` (if centering was used at train
    time) rides along so 03/04 center new data the same way the probe was trained."""
    os.makedirs(PROBE_DIR, exist_ok=True)
    torch.save(
        {
            "version": CACHE_VERSION,
            "state_dict": probe.state_dict(),
            "probe_type": probe_type,
            "dataset": dataset,
            "layer": layer,
            "model_name": model_name,
            "d_model": probe.d_model,
            "mean_offset": None if mean_offset is None else mean_offset.cpu(),
        },
        _probe_path(probe_type, dataset),
    )


def load_probe(probe_type, dataset, device=None):
    """Reload a trained probe (in eval mode) plus its metadata dict."""
    from probes import PROBES  # local import: data->probes is acyclic

    path = _probe_path(probe_type, dataset)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved {probe_type} probe for '{dataset}' at {path}. Run 02_train_probes first."
        )
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"{path} cache version {payload.get('version')} != {CACHE_VERSION}.")

    probe = PROBES[probe_type](payload["d_model"])
    probe.load_state_dict(payload["state_dict"])
    probe.eval()
    meta = {
        "probe_type": payload["probe_type"],
        "dataset": payload["dataset"],
        "layer": payload["layer"],
        "model_name": payload["model_name"],
        "mean_offset": payload["mean_offset"],
    }
    return probe, meta


def _probe_path(probe_type, dataset):
    return os.path.join(PROBE_DIR, f"{probe_type}_{dataset}.pt")


# --------------------------------------------------------------------------- #
# Splitting and centering  (step 02)
# --------------------------------------------------------------------------- #
def train_test_split(acts, labels, ratio=0.8, seed=0):
    """Seeded split of one dataset into train/test.

    Used for in-distribution evaluation. Transfer eval (train on A, test on B)
    doesn't call this — it uses whole datasets from `load_activations` directly.

    Args:
        ratio: fraction assigned to train.
        seed:  fixes the permutation for reproducibility.

    Returns:
        (train_acts, train_labels, test_acts, test_labels)
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}.")
    n = acts.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(round(n * ratio))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (
        acts[train_idx], labels[train_idx],
        acts[test_idx], labels[test_idx],
    )


def compute_mean_offset(acts):
    """Centering offset = mean over statements. Compute on TRAIN data only,
    persist via `save_activations(mean_offset=...)`, and apply everywhere with
    `center` so `03`/`04` use the same offset the probe was trained under."""
    return acts.mean(dim=0)


def center(acts, offset):
    """Subtract a precomputed offset. No-op if offset is None."""
    return acts if offset is None else acts - offset
