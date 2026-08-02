"""Impl-3 per-token weight signals + global normalization (PRD §3.2–3.3).

For each unmasked (loss-bearing) PEDAGOGY token t we compute a scalar "distance from
base" ``s_t`` and turn it into a per-token multiplier via a temperatured softmax of
``-s_t``. Two signal variants (recorded/compared per PRD §3.2):

  (a) base-surprise  : s_t = -log pi_0(y_t | context)          [one frozen-base pass]
  (b) forward-KL     : s_t = KL(pi_0(.|ctx_t) || pi_SFT(.|ctx_t))  [needs a vanilla SFT]

Global normalization (PRD §3.3, "cleaner equivalent"): a mean-1 multiplier over
pedagogy tokens, 1.0 for general tokens:

    m_t = N_ped * softmax_ped(-z(s_t)/T)     (pedagogy)
    m_t = 1                                   (general)

so the pedagogy:general ratio and overall LR are preserved. ``s_t`` is standardized
once, globally, with a robust z-score (median / MAD). ``T -> inf`` recovers vanilla
Impl 2.

The expensive part — computing ``s_t`` with a forward pass over the whole pedagogy
set — is cached to ``cache_dir`` keyed by (data content, variant), so a temperature
sweep reuses one precompute.
"""
from __future__ import annotations

import hashlib
import math
import os

import torch
import torch.nn.functional as F

from .chat import IGNORE
from .modeling import load_for_inference

NAN = float("nan")


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------
def _is_pedagogy(row):
    return row.get("kind", "pedagogy") != "general"


def _content_hash(train_tok, variant, base_model_id, sft_model_id):
    h = hashlib.md5()
    h.update(f"{variant}|{base_model_id}|{sft_model_id}".encode())
    for ids in train_tok["input_ids"]:
        h.update(bytes(memoryview(torch.tensor(ids, dtype=torch.int32).numpy())))
    return h.hexdigest()[:16]


@torch.no_grad()
def _row_signal(base_model, sft_model, input_ids, labels, variant, device):
    """Per-loss-token signal, returned as a full-length list (s at loss positions,
    NaN elsewhere). ``labels[i]`` is the target token at index i; the predicting
    logit is at position i-1."""
    L = len(input_ids)
    s_full = [NAN] * L
    loss_pos = [i for i in range(1, L) if labels[i] != IGNORE]  # i>=1: need i-1 logit
    if not loss_pos:
        return s_full
    ids = torch.tensor([input_ids], device=device)
    pred_pos = torch.tensor([p - 1 for p in loss_pos], device=device)
    tgt = torch.tensor([input_ids[p] for p in loss_pos], device=device)

    base_logits = base_model(ids).logits[0].index_select(0, pred_pos).float()  # (n_loss, V)
    lp0 = F.log_softmax(base_logits, dim=-1)
    if variant == "a":
        s = -lp0.gather(1, tgt.unsqueeze(1)).squeeze(1)                        # base NLL
    elif variant == "b":
        sft_logits = sft_model(ids).logits[0].index_select(0, pred_pos).float()
        lp1 = F.log_softmax(sft_logits, dim=-1)
        s = (lp0.exp() * (lp0 - lp1)).sum(dim=-1)                              # per-token fwd KL
    else:
        raise ValueError(f"unknown variant {variant!r} (expected 'a' or 'b')")
    for p, v in zip(loss_pos, s.tolist()):
        s_full[p] = v
    return s_full


def compute_signal_for_dataset(train_tok, tokenizer, variant, base_model_id,
                               sft_model_id=None, *, max_rows=0):
    """Return (s_rows, kind_rows): per-row full-length signal lists + kinds.

    General rows get all-NaN signal (they are never reweighted). ``max_rows`` caps
    the number of rows processed (0 = all).
    """
    if variant == "b" and not sft_model_id:
        raise ValueError("variant 'b' (forward-KL) needs a vanilla Impl-2 SFT via sft_model_id.")

    base_model, _, device = load_for_inference(base_model_id)
    sft_model = None
    if variant == "b":
        sft_model, _, _ = load_for_inference(base_model_id, adapter_dir=sft_model_id, merge=True)

    n = len(train_tok) if not max_rows else min(max_rows, len(train_tok))
    s_rows, kind_rows = [], []
    for i in range(n):
        row = train_tok[i]
        kind = row.get("kind", "pedagogy")
        kind_rows.append(kind)
        if kind == "general":
            s_rows.append([NAN] * len(row["input_ids"]))
        else:
            s_rows.append(_row_signal(base_model, sft_model, row["input_ids"], row["labels"], variant, device))
        if (i + 1) % 500 == 0:
            print(f"  signal {i + 1}/{n}")
    return s_rows, kind_rows


def get_or_compute_signal(train_tok, tokenizer, variant, base_model_id,
                          sft_model_id=None, *, cache_dir="weights"):
    """Load cached signal for this exact dataset+variant, or compute and cache it."""
    key = _content_hash(train_tok, variant, base_model_id, sft_model_id)
    path = os.path.join(cache_dir, f"signal_{variant}_{key}.pt")
    if os.path.exists(path):
        print(f"loading cached signal: {path}")
        blob = torch.load(path)
        return blob["s_rows"], blob["kind_rows"]
    print(f"computing signal (variant={variant}) -> {path}")
    s_rows, kind_rows = compute_signal_for_dataset(train_tok, tokenizer, variant, base_model_id, sft_model_id)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"s_rows": s_rows, "kind_rows": kind_rows, "variant": variant}, path)
    return s_rows, kind_rows


# ---------------------------------------------------------------------------
# Global normalization -> per-token multipliers
# ---------------------------------------------------------------------------
def robust_zscore_params(values):
    """(center, scale) via median / MAD (scaled to ~std). Falls back to mean/std."""
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return 0.0, 1.0
    n = len(vals)
    med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    devs = sorted(abs(v - med) for v in vals)
    mad = devs[n // 2] if n % 2 else 0.5 * (devs[n // 2 - 1] + devs[n // 2])
    scale = 1.4826 * mad
    if scale <= 1e-8:  # near-constant signal
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
        scale = math.sqrt(var) or 1.0
        return mean, scale
    return med, scale


def multipliers_from_signal(s_rows, kind_rows, T):
    """Per-row full-length multiplier lists.

    ``T -> inf`` => all pedagogy multipliers -> 1 (vanilla Impl 2). Non-loss positions
    (NaN signal, general rows) get multiplier 1.0 (harmless: masked in the loss).
    """
    ped_vals = [v for row, k in zip(s_rows, kind_rows) if k != "general" for v in row if v == v]
    center, scale = robust_zscore_params(ped_vals)

    inf_T = math.isinf(T)
    # global softmax denominator over pedagogy loss tokens
    U = 0.0
    N_ped = 0
    if not inf_T:
        for row, k in zip(s_rows, kind_rows):
            if k == "general":
                continue
            for v in row:
                if v == v:
                    U += math.exp(-((v - center) / scale) / T)
                    N_ped += 1
        U = U or 1.0

    weights_rows = []
    for row, k in zip(s_rows, kind_rows):
        w = [1.0] * len(row)
        if k != "general" and not inf_T:
            for i, v in enumerate(row):
                if v == v:
                    u = math.exp(-((v - center) / scale) / T)
                    w[i] = N_ped * u / U  # mean 1 over pedagogy tokens
        weights_rows.append(w)
    return weights_rows


def make_attach_weights(variant, T, base_model_id, *, sft_model_id=None, cache_dir="weights"):
    """Build an ``attach_weights(train_tok, tokenizer) -> ds`` for common.sft_train.run_sft."""
    def attach(train_tok, tokenizer):
        s_rows, kind_rows = get_or_compute_signal(
            train_tok, tokenizer, variant, base_model_id, sft_model_id, cache_dir=cache_dir)
        weights_rows = multipliers_from_signal(s_rows, kind_rows, T)
        assert len(weights_rows) == len(train_tok), "weight/row count mismatch"
        return train_tok.add_column("weights", weights_rows)

    return attach
