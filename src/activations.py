"""Residual-stream activation extraction via TransformerLens.

Forward passes are the slow, expensive step; this module runs them and hands
back activation tensors that `data.save_activations` caches. Everything else in
the pipeline works off that cache.

LEAD WITH THE LOAD CHECK. TransformerLens support — not raw model quality — is
the real constraint here. Gemma-2 is supported but has had tokenizer / rotary /
final-norm quirks, and the exact residual hook name is what makes extraction a
few lines instead of hand-written hooks. Before extracting anything, run:

    model = load_model("gemma-2-2b")
    print(verify_model(model))

`verify_model` does a tiny forward pass and confirms the residual-stream hook
exists and has the expected shape. If that passes, extraction is trivial; if it
fails, you want to know before running the whole dataset (or the slow 9B model).

Gemma-2 requires accepting the license and a HuggingFace token to download the
weights (`huggingface-cli login`).
"""

import torch
from transformer_lens import HookedTransformer


# The residual stream *after* a block — the standard localization for
# statement-level truth information (Marks & Tegmark).
def resid_hook_name(layer):
    """TransformerLens hook name for the post-block residual stream at `layer`."""
    return f"blocks.{layer}.hook_resid_post"


# --------------------------------------------------------------------------- #
# Load + verify  (run this FIRST)
# --------------------------------------------------------------------------- #
def load_model(model_name="gemma-2-2b", device=None, dtype=None):
    """Load a model via TransformerLens, in eval mode.

    Args:
        model_name: e.g. "gemma-2-2b" (dev) or "gemma-2-9b" (final).
        device:     defaults to cuda if available, else cpu.
        dtype:      defaults to bfloat16 on cuda (how these models run; 9B in
                    float32 would not fit), float32 on cpu. Activations are cast
                    to float32 at extraction time regardless, so the probes
                    always see float32.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        # bf16 on cuda (how these run; 9B won't fit in f32). float32 elsewhere:
        # MPS bf16 support is uneven and CPU has no bf16 speedup.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    return model


def verify_model(model, layer=0):
    """Smoke-test the load: run one forward pass and confirm the residual hook
    exists with the expected shape. Raises with a clear message otherwise.

    Returns a dict of model facts (n_layers, d_model, hook name/shape) worth
    printing before committing to a full extraction run.
    """
    hook = resid_hook_name(layer)
    tokens = model.to_tokens("The city of Paris is in France.")
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook, return_type=None
        )

    if hook not in cache:
        raise RuntimeError(
            f"Residual hook {hook!r} not found in cache for {model.cfg.model_name}. "
            f"TransformerLens may name this model's hooks differently — inspect "
            f"model.hook_dict before extracting."
        )

    shape = tuple(cache[hook].shape)  # (batch, seq, d_model)
    if shape[-1] != model.cfg.d_model:
        raise RuntimeError(
            f"Hook {hook} last dim {shape[-1]} != cfg.d_model {model.cfg.d_model}."
        )

    return {
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_ctx": model.cfg.n_ctx,
        "hook_name": hook,
        "hook_shape": shape,
        "device": str(next(model.parameters()).device),
        "pad_token_id": model.tokenizer.pad_token_id,
    }


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_activations(model, texts, layers, token_pos=-1, batch_size=16):
    """Extract residual-stream activations over a set of statements.

    Runs all requested layers in a single forward pass per batch (one forward,
    many layers) so the layer sweep costs one extraction, not one per layer.

    Args:
        model:      a loaded HookedTransformer.
        texts:      list[str] of statements.
        layers:     int or list[int]. Residual layer(s) to extract.
        token_pos:  -1 (default) = the last real (non-pad) token of each
                    statement, i.e. the end-of-statement token M&T probe over.
                    An explicit int uses that absolute position for every row.
        batch_size: statements per forward pass.

    Returns:
        Tensor, float32, on CPU:
          * (n, d_model)              if `layers` is a single int,
          * (n, len(layers), d_model) if `layers` is a list.
    """
    
    single = isinstance(layers, int)
    layer_list = [layers] if single else list(layers)
    hook_names = [resid_hook_name(l) for l in layer_list]
    wanted = set(hook_names)
    pad_id = model.tokenizer.pad_token_id
    assert pad_id is not None

    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = model.to_tokens(batch)  # (b, seq), padded to batch max
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in wanted, return_type=None
            )

            if token_pos == -1:
                pos = _last_real_token_index(tokens, pad_id)  # (b,)
            else:
                pos = torch.full(
                    (tokens.shape[0],), token_pos, device=tokens.device, dtype=torch.long
                )
            rows = torch.arange(tokens.shape[0], device=tokens.device)

            # gather the chosen token from each requested layer
            per_layer = [cache[h][rows, pos] for h in hook_names]  # each (b, d_model)
            stacked = torch.stack(per_layer, dim=1)  # (b, n_layers, d_model)
            chunks.append(stacked.to(device="cpu", dtype=torch.float32))

    acts = torch.cat(chunks, dim=0)  # (n, n_layers, d_model)
    return acts.squeeze(1) if single else acts


def _last_real_token_index(tokens, pad_token_id):
    """Index of the last non-pad token in each row. Robust to left- or
    right-padding (assumes no interior padding, which holds for standard
    tokenization). Falls back to the final position if there's no pad token."""
    seq = tokens.shape[1]
    if pad_token_id is None:
        return torch.full((tokens.shape[0],), seq - 1, device=tokens.device, dtype=torch.long)
    idx = torch.arange(seq, device=tokens.device).expand_as(tokens)
    real = torch.where(tokens != pad_token_id, idx, torch.full_like(idx, -1))
    return real.max(dim=1).values  # (b,)
