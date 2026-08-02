"""Forward-KL measurement (PRD "KL convention").

Whenever the PRD says "KL" it means the forward KL FROM the base model on the
NEW-TASK (pedagogy) inputs:  KL(pi_0 || pi), reference = base pi_0, evaluated on
held-out pedagogy prompts, reported in both conditions (kl_new_SI, kl_ped_noSI).

``item_kl`` computes the exact full-vocab forward KL per token, teacher-forced on a
base-sampled continuation (the expectation is over samples from the BASE policy, so
we generate the continuation with the base model — faithful to KL(pi_0 || pi)).

Use ``sweep_checkpoints`` to produce one (kl_new_SI, kl_ped_noSI) pair per checkpoint
for the KL–forgetting curve.
"""
from __future__ import annotations

import gc
import json
import os

import torch
import torch.nn.functional as F

from .modeling import load_for_inference
from .system_instructions import CANONICAL_SI


def pedagogy_contexts(rows, n=0):
    """Held-out pedagogy KL prompts: each dialogue truncated to just before its first tutor turn.

    The KL condition is "+SI vs no-SI on a new-task input", so the prompt must be a point where
    the SI is what decides whether the reply is Socratic — i.e. the student's opening problem,
    with no tutor turns yet. Passing the WHOLE dialogue instead (which is what happens if a row
    has no ``context`` field and the caller falls back to ``messages``) puts several gold Socratic
    turns in the context, which primes the tutor behaviour on its own and drives kl_new_SI and
    kl_ped_noSI together — the SI stops mattering because the transcript already did its job. It
    also asks the model to emit an assistant turn directly after an assistant turn, which no
    training example ever looks like.

    Mirrors the POC's construction so KL numbers stay comparable across the two setups.
    """
    out = []
    for r in rows:
        conv = [m for m in (r.get("context") or r.get("messages") or []) if m.get("role") != "system"]
        if r.get("context"):
            ctx = conv  # already a context: trust it as-is
        else:
            ai = next((i for i, m in enumerate(conv) if m.get("role") == "assistant"), None)
            if ai is None or ai == 0:
                continue  # no tutor turn to stop before -> not a usable KL prompt
            ctx = conv[:ai]
        if ctx:
            out.append(ctx)
        if n and len(out) >= n:
            break
    return out


@torch.no_grad()
def base_continuations(base_model, tokenizer, items, use_si, *, gen_max=200, device=None):
    """Pre-generate the base-policy continuations once, for reuse across many checkpoints.

    KL(pi_0 || pi) takes the expectation over samples from the BASE policy, so the continuation
    depends only on (base model, prompt, condition) — never on the checkpoint being scored. The
    per-checkpoint loop in ``sweep_checkpoints`` used to regenerate them every time, which is
    ~200 autoregressive steps per item versus the 2 forward passes the KL itself needs. Over a
    120-checkpoint sweep that is ~99% of the runtime thrown away.

    Returns a list of (full_ids, Lp) with the prompt+response token ids and the prompt length.
    """
    device = device or next(base_model.parameters()).device
    cached = []
    for messages in items:
        conv = ([{"role": "system", "content": CANONICAL_SI}] if use_si else []) + messages
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        Lp = enc.input_ids.shape[1]
        gen = base_model.generate(**enc, max_new_tokens=gen_max, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        if gen.shape[1] - Lp == 0:
            continue
        cached.append((gen, Lp))
    return cached


@torch.no_grad()
def mean_kl_cached(base_model, sft_model, cached):
    """Mean per-token forward KL over pre-generated base continuations (see ``base_continuations``).

    Identical arithmetic to ``item_kl``; only the generation is hoisted out of the loop.
    """
    vals = []
    for full, Lp in cached:
        b = base_model(full).logits[:, Lp - 1:-1, :].float()
        s = sft_model(full).logits[:, Lp - 1:-1, :].float()
        lp0 = F.log_softmax(b, dim=-1)
        lp1 = F.log_softmax(s, dim=-1)
        vals.append((lp0.exp() * (lp0 - lp1)).sum(dim=-1).mean().item())
    return (sum(vals) / len(vals)) if vals else float("nan")


@torch.no_grad()
def item_kl(base_model, sft_model, tokenizer, messages, use_si, *, gen_max=200, device=None):
    """Mean per-token forward KL(base||sft) over a base-generated continuation.

    ``messages`` is a chat history (no system message; the SI is added here per the
    condition). Returns (kl_per_token, n_response_tokens) or None if nothing generated.
    """
    device = device or next(base_model.parameters()).device
    conv = ([{"role": "system", "content": CANONICAL_SI}] if use_si else []) + messages
    text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    Lp = enc.input_ids.shape[1]

    gen = base_model.generate(**enc, max_new_tokens=gen_max, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
    resp = gen[:, Lp:]
    if resp.shape[1] == 0:
        return None

    full = torch.cat([enc.input_ids, resp], dim=1)
    # logits at positions Lp-1 .. end-1 predict the response tokens
    b = base_model(full).logits[:, Lp - 1:-1, :].float()
    s = sft_model(full).logits[:, Lp - 1:-1, :].float()
    lp0 = F.log_softmax(b, dim=-1)
    lp1 = F.log_softmax(s, dim=-1)
    kl = (lp0.exp() * (lp0 - lp1)).sum(dim=-1)  # exact full-vocab forward KL per position
    return kl.mean().item(), resp.shape[1]


def mean_kl(base_model, sft_model, tokenizer, items, use_si, *, gen_max=200):
    vals = [item_kl(base_model, sft_model, tokenizer, m, use_si, gen_max=gen_max) for m in items]
    vals = [v[0] for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else float("nan")


def sweep_checkpoints(base_model_id, checkpoints, pedagogy_items, *,
                      gen_max=200, out_path=None):
    """For each checkpoint, compute new-task forward KL in +SI and no-SI conditions.

    Args:
        base_model_id: HF id / path of the KL reference pi_0 (the Instruct model).
        checkpoints:   dict {label -> adapter_or_model_path}. Each is loaded, KL'd
                       against the shared base, then freed.
        pedagogy_items: list of chat histories (held-out pedagogy prompts). Provide
                       your own once eval data exists (data is blank for now).
    Returns {label: {kl_new_SI, kl_ped_noSI}} and optionally writes JSON.
    """
    base_model, tok, _ = load_for_inference(base_model_id)
    print(f"KL reference (base) loaded: {base_model_id}")

    # Generate the base continuations ONCE — they are checkpoint-independent, so doing this
    # inside the loop below made the sweep ~100x more expensive than it needed to be.
    print(f"pre-generating base continuations for {len(pedagogy_items)} prompts x 2 conditions ...")
    cached_si = base_continuations(base_model, tok, pedagogy_items, True, gen_max=gen_max)
    cached_no = base_continuations(base_model, tok, pedagogy_items, False, gen_max=gen_max)

    res = {}
    for label, path in checkpoints.items():
        print(f"\n=== {label} :: {path} ===")
        try:
            sft, _, _ = load_for_inference(base_model_id, adapter_dir=path, merge=True)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (load failed): {e}")
            continue
        kl_si = mean_kl_cached(base_model, sft, cached_si)
        kl_no = mean_kl_cached(base_model, sft, cached_no)
        res[label] = {"kl_new_SI": kl_si, "kl_ped_noSI": kl_no}
        print(f"  kl_new_SI={kl_si:.4f} | kl_ped_noSI={kl_no:.4f}")
        del sft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {out_path}")
    return res
