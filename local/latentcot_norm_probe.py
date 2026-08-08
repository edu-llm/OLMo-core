"""
Diagnostic: what is the scale of a continuous thought vs a real token embedding?

`Transformer.forward(return_hidden_states=True)` returns the POST-BLOCK, PRE-final-norm
residual stream (the final norm lives inside LMHead). latentcot feeds that back as the
next input embedding. This measures whether that vector lives in the same numeric range
as the real-token embeddings it gets concatenated with, and how it behaves over K steps.

Random init only (no S3 checkpoint locally) -> treat residual-stream numbers as a LOWER
BOUND; trained models grow the residual stream substantially more.

Run: .venv/bin/python local/latentcot_norm_probe.py
"""

import torch

from olmo_core.latentcot.cot import embed_tokens, run_continuous_thoughts
from olmo_core.latentcot.train_driver import build_model

RUNG = "olmo2_370M"  # olmo3_370M == this + sliding-window/flash_2 (needs CUDA); norm
# structure is identical, which is all this probe measures.
K = 10
SEQ = 24


def rms(t: torch.Tensor) -> float:
    return t.float().pow(2).mean().sqrt().item()


def main() -> None:
    torch.manual_seed(0)
    model = build_model(RUNG, init_seed=0, device="cpu")
    model.eval()

    d_model = model.embeddings.weight.shape[1]
    print(f"rung={RUNG}  d_model={d_model}  n_layers={len(model.blocks)}")
    print(f"embed_scale={getattr(model, 'embed_scale', None)}  "
          f"embedding_norm={type(getattr(model, 'embedding_norm', None)).__name__}")
    print(f"lm_head.norm={type(model.lm_head.norm).__name__}")
    print()

    ids = torch.randint(0, 100_000, (1, SEQ))

    with torch.no_grad():
        # 1. What a real token looks like on the way IN.
        emb = embed_tokens(model, ids)

        # 2. Per-layer residual-stream growth.
        h = emb.clone()
        per_layer = []
        for block in model.blocks.values():
            h = block(h)
            per_layer.append(rms(h))

        # 3. What the hook returns (pre-final-norm) vs what CODI's reference feeds
        #    (post-final-norm).
        h_pre = model(ids, return_hidden_states=True)
        h_post = model.lm_head.norm(h_pre)

        print(f"[1] real-token embedding RMS (what forward uses for tokens) : {rms(emb):9.4f}")
        print(f"[2] pre-final-norm  hidden RMS  <-- fed back as a THOUGHT    : {rms(h_pre):9.4f}")
        print(f"[3] post-final-norm hidden RMS  <-- Coconut/CODI reference   : {rms(h_post):9.4f}")
        print()
        print(f"    ratio thought / embedding      = {rms(h_pre) / rms(emb):8.1f}x")
        print(f"    ratio post-norm / embedding    = {rms(h_post) / rms(emb):8.1f}x")
        print()
        print("[4] residual-stream RMS by layer (growth with depth):")
        print("    " + "  ".join(f"{v:.3f}" for v in per_layer))
        print()

        # 4. Does the thought drift/explode over K autoregressive steps?
        prefix = emb
        thoughts, _ = run_continuous_thoughts(model, prefix, K)
        print(f"[5] thought RMS over K={K} steps (fed back autoregressively):")
        print("    " + "  ".join(f"{rms(thoughts[:, i]):.3f}" for i in range(K)))
        first, last = rms(thoughts[:, 0]), rms(thoughts[:, -1])
        print(f"    step1={first:.4f}  stepK={last:.4f}  drift={last / first:.3f}x")
        print()

        # 5. How much of the assembled sequence is out-of-range vs the embedding rows?
        emb_row_rms = rms(model.embeddings.weight)
        print(f"[6] embedding MATRIX row RMS (all {model.embeddings.weight.shape[0]} rows)"
              f": {emb_row_rms:.4f}")
        print(f"    thought is {rms(thoughts) / emb_row_rms:.1f}x the typical embedding row")


if __name__ == "__main__":
    main()
