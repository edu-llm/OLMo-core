"""
Pre-flight smoke test for the shared base checkpoint (PRD Phase 8, run this *first* on the GPU).

Before launching the A0-A4 x seeds sweep, this verifies the two things the sweep assumes but
nothing has exercised yet:

1. **The "best model" strict-loads into our config.** Builds ``olmo3_370M`` at the dolma2 padded
   vocab (100352) and strict-loads the base checkpoint (a plain ``.pt`` state_dict or a local/S3
   OLMo-core checkpoint dir). ``load_checkpoint`` uses ``strict=True``, so any vocab/shape/key
   mismatch hard-fails here instead of silently loading the wrong weights.

2. **The continuous-thought forward path runs on this model.** ``olmo3_370M`` adds ``flash_2`` +
   sliding-window attention; the continuous-thought loop drives the model through
   ``input_embeddings=`` + ``return_hidden_states=True`` + per-block hooks, a combination only
   tested on tiny ``llama_like`` models so far. This does one plain forward and one K-step
   continuous-thought forward+backward and checks the shapes and that gradients flow.

Usage::

    .venv/bin/python src/scripts/latentcot/verify_checkpoint.py \
        --init-checkpoint s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/ \
        --model olmo3_370M     # S3 needs AWS creds; --device auto-detects cuda

    # or dry-run the forward path with no checkpoint (random init):
    .venv/bin/python src/scripts/latentcot/verify_checkpoint.py --model olmo3_1M
"""

import argparse

import torch

from olmo_core.latentcot.cot import embed_tokens, run_continuous_thoughts
from olmo_core.latentcot.tokens import TOKENIZER_CONFIG
from olmo_core.latentcot.train_driver import load_checkpoint, resolve_device
from olmo_core.nn.transformer import TransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="olmo3_370M", help="TransformerConfig factory name")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="base ckpt to strict-load (.pt | dir | s3://…); omit to test random init",
    )
    parser.add_argument("--num-continuous-thoughts", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=16, help="tiny forward-pass length")
    parser.add_argument(
        "--device", default="auto", help="'auto' (cuda if available else cpu), 'cuda', or 'cpu'"
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    vocab = TOKENIZER_CONFIG.padded_vocab_size()
    print(f"building {args.model}(vocab_size={vocab}) on {device} ...")
    model = getattr(TransformerConfig, args.model)(vocab_size=vocab).build(init_device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  built: {n_params:,} params")

    if args.init_checkpoint:
        print(f"strict-loading base checkpoint: {args.init_checkpoint}")
        load_checkpoint(model, args.init_checkpoint, strict=True)
        print("  OK: state_dict matched the config (strict load succeeded)")
    else:
        print("no --init-checkpoint: testing the forward path on random init")

    model.to(device)

    # 1. plain forward
    model.eval()
    with torch.no_grad():
        ids = torch.randint(0, vocab, (1, args.seq_len), device=device)
        logits = model(ids)
    assert logits.shape == (1, args.seq_len, vocab), logits.shape
    print(f"  plain forward OK: logits {tuple(logits.shape)}")

    # 2. continuous-thought forward + backward (gradients must flow through the K-step loop)
    model.train()
    k = args.num_continuous_thoughts
    prefix_ids = torch.randint(0, vocab, (1, args.seq_len), device=device)
    prefix_embeds = embed_tokens(model, prefix_ids)
    thoughts, embeds = run_continuous_thoughts(model, prefix_embeds, k)
    d_model = prefix_embeds.shape[-1]
    assert thoughts.shape == (1, k, d_model), thoughts.shape
    assert embeds.shape == (1, args.seq_len + k, d_model), embeds.shape
    thoughts.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed through the continuous-thought loop"
    print(f"  continuous-thought forward+backward OK: thoughts {tuple(thoughts.shape)}, "
          f"{len(grads)} tensors got grads")

    print("\nVERIFY PASSED — base checkpoint loads and the continuous-thought path runs.")


if __name__ == "__main__":
    main()
