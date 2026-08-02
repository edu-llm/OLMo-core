"""End-to-end smoke test: does the machinery work, and do gradients actually move?

Runs on CPU in about a minute. Not a correctness proof of the science — it is the check
that the plumbing is connected, run before booking a GPU.

    python src/scripts/train/p3_math_split/smoke_test.py              # tiny model, fast
    python src/scripts/train/p3_math_split/smoke_test.py --real-model # the actual 494M Qwen2.5-0.5B (slow, ~1GB download)

Stages, each of which can fail independently:

  1  model      bias strip, embedding tie, parameter counts
  2  masks      label_mask -> labels through OLMo-core's own get_labels, so the
                convention is verified against the library rather than assumed
  3  gradients  loss is finite, grads are non-zero, and they REACH the embedding
  4  arms       dense and split produce different gradients on the same batch, and
                the split arm's fact tokens contribute exactly nothing
  5  training   a real AdamW loop: loss falls, parameters move
  6  divisor    the fixed divisor is actually substituted for OLMo-core's default

Stage 4 is the one that matters. Everything else can pass while the two arms silently
train identically, and that failure mode is invisible in a loss curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.nn.attention import AttentionConfig, AttentionType  # noqa: E402
from olmo_core.nn.feed_forward import FeedForwardConfig  # noqa: E402
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType  # noqa: E402
from olmo_core.nn.lm_head import LMHeadConfig  # noqa: E402
from olmo_core.nn.rope import RoPEConfig, RoPEType  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import (  # noqa: E402
    TransformerBlockConfig,
    TransformerBlockType,
)
from olmo_core.nn.transformer.qwen import strip_attn_out_bias  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    if not condition:
        _failures.append(name)
    return condition


def tiny_qwen_config(
    vocab: int, dtype: DType = DType.float32, tie: bool = True
) -> TransformerConfig:
    """A Qwen2-*shaped* model small enough to train on a laptop CPU.

    Same block type, same GQA ratio, same bias placement, same SwiGLU, same RoPE theta —
    only the sizes shrink. The point is to exercise the identical code path, so a bug in
    strip_attn_out_bias or the mask wiring shows up here rather than on the GPU.
    """
    ln = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    attention = AttentionConfig(
        name=AttentionType.default,
        n_heads=4,
        n_kv_heads=2,  # GQA, as in Qwen2
        bias=True,  # q/k/v need it; w_out's gets stripped
        rope=RoPEConfig(name=RoPEType.default, theta=1_000_000),
        dtype=dtype,
    )
    block = TransformerBlockConfig(
        name=TransformerBlockType.default,
        feed_forward=FeedForwardConfig(hidden_size=128, bias=False, dtype=dtype),
        layer_norm=ln,
        sequence_mixer=attention,
    )
    return TransformerConfig(
        d_model=64,
        vocab_size=vocab,
        n_layers=2,
        block=block,
        lm_head=LMHeadConfig(layer_norm=ln, bias=False, dtype=dtype),
        dtype=dtype,
        tie_word_embeddings=tie,
    )


def build_tiny(vocab: int, tie: bool = True):
    # Tying comes from the config, exactly as build_qwen2_0_5b does it, so this exercises
    # the same lifecycle (re-tie after to_empty, skip init_final_w_out) as the real model.
    model = tiny_qwen_config(vocab, tie=tie).build(init_device="cpu")
    # The real library function, not a reimplementation of it -- expected_layers matches
    # this scaled-down model so the production code path is what gets exercised.
    n_stripped = strip_attn_out_bias(model, expected_layers=2)
    return model, n_stripped


def make_batch(vocab: int, batch: int, seq: int, n_fact_tokens: int, seed: int = 0):
    """One synthetic batch shaped like the real corpus.

    Positions [0, n_fact_tokens) stand in for the fact block, the rest for
    `--- GOAL ... proof`. Last few positions are padding, unsupervised in both arms.
    """
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch, seq), generator=g)
    n_pad = 3
    dense = torch.ones(batch, seq, dtype=torch.bool)
    dense[:, seq - n_pad :] = False  # padding: unsupervised in both arms
    split = dense.clone()
    split[:, :n_fact_tokens] = False  # fact block: split scores nothing here
    return input_ids, dense, split, n_pad


# --------------------------------------------------------------------- stages
def stage_model(vocab):
    print("\n1. model construction")
    model, n_stripped = build_tiny(vocab)
    check(
        "attention output bias stripped from every block",
        n_stripped == 2,
        f"stripped {n_stripped}/2",
    )
    check(
        "no w_out.bias remains",
        all(getattr(b.attention.w_out, "bias", None) is None for b in model.blocks.values()),
    )
    check(
        "q/k/v biases retained",
        all(
            getattr(b.attention, p).bias is not None
            for b in model.blocks.values()
            for p in ("w_q", "w_k", "w_v")
        ),
    )
    check("embeddings tied to lm_head", model.lm_head.w_out.weight is model.embeddings.weight)
    check(
        "no w_out.bias in state_dict",
        not any(k.endswith("attention.w_out.bias") for k in model.state_dict()),
    )

    untied, _ = build_tiny(vocab, tie=False)

    def uniq(m):
        return sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())

    check(
        "untied model is larger by exactly the embedding matrix",
        uniq(untied) - uniq(model) == vocab * 64,
        f"{uniq(untied) - uniq(model):,} == {vocab * 64:,}",
    )
    return model


def stage_masks(vocab):
    print("\n2. mask semantics, verified against olmo_core.data.utils.get_labels")
    seq, n_fact = 16, 6
    input_ids, dense, split, n_pad = make_batch(vocab, 2, seq, n_fact)

    lab_d = get_labels({"input_ids": input_ids, "label_mask": dense}, label_ignore_index=-100)
    lab_s = get_labels({"input_ids": input_ids, "label_mask": split}, label_ignore_index=-100)

    # get_labels masks then shifts left, so label position j scores input token j+1.
    scored_d = (lab_d[0] != -100).nonzero().flatten().tolist()
    scored_s = (lab_s[0] != -100).nonzero().flatten().tolist()
    check(
        "dense scores every real token except the last",
        scored_d == list(range(seq - n_pad - 1)),
        f"{scored_d}",
    )
    check(
        "split scores only positions after the fact block",
        scored_s == list(range(n_fact - 1, seq - n_pad - 1)),
        f"{scored_s}",
    )
    check(
        "split is a strict subset of dense",
        set(scored_s) < set(scored_d),
        f"split drops {len(set(scored_d) - set(scored_s))} positions",
    )
    check("padding is unscored in both arms", all(p < seq - n_pad - 1 for p in scored_d + scored_s))
    check("the two arms differ", lab_d.ne(lab_s).any().item())
    return input_ids, dense, split


def loss_and_grads(model, input_ids, mask, divisor):
    model.zero_grad(set_to_none=True)
    labels = get_labels({"input_ids": input_ids, "label_mask": mask}, label_ignore_index=-100)
    out = model(
        input_ids,
        labels=labels,
        ignore_index=-100,
        loss_reduction="sum",
        loss_div_factor=divisor,
        return_logits=False,
    )
    loss = out[1] if isinstance(out, (tuple, list)) else out
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


def stage_gradients(model, input_ids, dense, split, divisor):
    print("\n3. gradients flow")
    loss, grads = loss_and_grads(model, input_ids, dense, divisor)
    check("loss is finite", torch.isfinite(loss).item(), f"loss = {loss.item():.4f}")
    check("loss is positive", loss.item() > 0)

    total = sum(int(p.numel()) for p in grads.values())
    nonzero = sum(int((g != 0).sum()) for g in grads.values())
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads.values())).item()
    check(
        "gradient norm is non-zero and finite",
        gnorm > 0 and torch.isfinite(torch.tensor(gnorm)),
        f"||g|| = {gnorm:.4f}",
    )
    check(
        "gradients are non-zero across parameters",
        nonzero > 0.5 * total,
        f"{nonzero:,}/{total:,} entries non-zero",
    )
    check(
        "every parameter received a gradient",
        len(grads) == sum(1 for _ in model.parameters()),
        f"{len(grads)} params with grad",
    )
    # The tie means one shared matrix; a gradient here proves it reaches both roles.
    check(
        "gradient reaches the (tied) embedding",
        "embeddings.weight" in grads and grads["embeddings.weight"].abs().sum().item() > 0,
    )
    return grads


def stage_arms(model, input_ids, dense, split, divisor):
    print("\n4. the arms differ — the check that matters")
    loss_d, grads_d = loss_and_grads(model, input_ids, dense, divisor)
    loss_s, grads_s = loss_and_grads(model, input_ids, split, divisor)

    check(
        "dense loss > split loss (dense sums over more tokens)",
        loss_d.item() > loss_s.item(),
        f"dense {loss_d.item():.4f} vs split {loss_s.item():.4f}",
    )

    diffs = {n: (grads_d[n] - grads_s[n]).abs().max().item() for n in grads_d}
    max_diff = max(diffs.values())
    n_differ = sum(1 for v in diffs.values() if v > 1e-9)
    check(
        "gradients differ between arms",
        max_diff > 1e-6,
        f"max |Δgrad| = {max_diff:.3e} across {n_differ}/{len(diffs)} tensors",
    )
    check(
        "the difference is broad, not one stray tensor",
        n_differ > len(diffs) // 2,
        f"{n_differ}/{len(diffs)} tensors differ",
    )

    # Number of fact tokens = positions dense supervises and split does not.
    n_fact = int((~split[0] & dense[0]).sum())

    # get_labels masks, then shifts left, so label position j predicts input token j+1.
    # Fact tokens occupy inputs [0, n_fact), so label positions [0, n_fact-1) must all
    # be ignored. Position n_fact-1 predicts the SEPARATOR, which is not fact content
    # and is supervised in both arms on purpose — the split arm still has to learn to
    # stop listing facts and start proving.
    lab_s = get_labels({"input_ids": input_ids, "label_mask": split}, label_ignore_index=-100)
    check(
        "split ignores every fact-token prediction target",
        bool((lab_s[:, : n_fact - 1] == -100).all()),
        f"label positions 0..{n_fact - 2} all -100",
    )
    check(
        "split DOES supervise the separator right after the block",
        bool((lab_s[:, n_fact - 1] != -100).all()),
        f"label position {n_fact - 1} predicts the first post-block token",
    )

    # Perturb exactly the targets that only dense scores: input tokens 1..n_fact-1.
    perturbed = input_ids.clone()
    perturbed[:, 1:n_fact] = (perturbed[:, 1:n_fact] + 7919) % model.embeddings.weight.shape[0]

    l_d0, _ = loss_and_grads(model, input_ids, dense, divisor)
    l_d1, _ = loss_and_grads(model, perturbed, dense, divisor)
    l_s0, _ = loss_and_grads(model, input_ids, split, divisor)
    l_s1, _ = loss_and_grads(model, perturbed, split, divisor)

    check(
        "dense loss responds to changing fact-token targets",
        abs(l_d1.item() - l_d0.item()) > 1e-4,
        f"Δ = {abs(l_d1.item() - l_d0.item()):.4f}",
    )
    # Split still conditions on the facts, so its loss moves too; what it never does is
    # credit gradient to *predicting* them, which the two label checks above establish.
    check(
        "split loss still changes when the facts change (it conditions on them)",
        abs(l_s1.item() - l_s0.item()) > 1e-6,
        f"Δ = {abs(l_s1.item() - l_s0.item()):.4f}",
    )


def stage_training(vocab, divisor, steps=30):
    print(f"\n5. real training loop, {steps} steps per arm")
    results: dict = {}
    for arm in ("dense", "split"):
        torch.manual_seed(1234)  # identical init for both arms
        model, _ = build_tiny(vocab)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)

        input_ids, dense, split, _ = make_batch(vocab, 4, 24, 8, seed=7)
        mask = dense if arm == "dense" else split
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            labels = get_labels(
                {"input_ids": input_ids, "label_mask": mask}, label_ignore_index=-100
            )
            out = model(
                input_ids,
                labels=labels,
                ignore_index=-100,
                loss_reduction="sum",
                loss_div_factor=divisor,
                return_logits=False,
            )
            loss = out[1] if isinstance(out, (tuple, list)) else out
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        moved = sum(
            1 for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])
        )
        total = sum(1 for _ in model.named_parameters())
        results[arm] = {
            "losses": losses,
            "moved": moved,
            "total": total,
            "final": {n: p.detach().clone() for n, p in model.named_parameters()},
        }

        check(
            f"[{arm}] loss decreases",
            losses[-1] < losses[0],
            f"{losses[0]:.4f} -> {losses[-1]:.4f} ({100 * (1 - losses[-1] / losses[0]):.0f}% down)",
        )
        check(f"[{arm}] all losses finite", all(torch.isfinite(torch.tensor(x)) for x in losses))
        check(f"[{arm}] parameters moved", moved == total, f"{moved}/{total} tensors changed")

    d = results["dense"]["final"]
    s = results["split"]["final"]
    delta = max((d[n] - s[n]).abs().max().item() for n in d)
    check(
        "the two trained models are different",
        delta > 1e-5,
        f"max |Δparam| after {steps} steps = {delta:.3e}",
    )
    check(
        "same init, so the difference came from the mask alone",
        results["dense"]["losses"][0] != results["split"]["losses"][0],
        "step-0 losses differ only because the masks do",
    )


def stage_divisor(vocab):
    print("\n6. fixed divisor is substituted for OLMo-core's default")
    import inspect

    from train_module import FixedDivisorTransformerTrainModule

    from olmo_core.train.train_module import TransformerTrainModule

    src = inspect.getsource(TransformerTrainModule.train_batch)
    check(
        "OLMo-core's default divisor is still the live-token count",
        "loss_div_factor=batch_num_tokens_for_loss" in src,
        "if this fails, the override may no longer be needed",
    )
    check(
        "the override targets a real method",
        hasattr(TransformerTrainModule, "model_forward")
        and "model_forward" in FixedDivisorTransformerTrainModule.__dict__,
    )

    # Exercise the substitution without building a full train module.
    captured = {}

    class Spy(FixedDivisorTransformerTrainModule):
        def __init__(self):  # bypass the heavy __init__
            self.fixed_loss_div_factor = 4096.0

        def record_metric(self, *a, **k):
            pass

    def fake_super_forward(self, input_ids, labels=None, **kwargs):
        captured.update(kwargs)
        return None

    spy = Spy()

    # Patch the parent so we can observe exactly what the override forwards upward.
    orig = TransformerTrainModule.model_forward
    try:
        TransformerTrainModule.model_forward = fake_super_forward  # type: ignore[method-assign]
        FixedDivisorTransformerTrainModule.model_forward(
            spy,
            torch.zeros(1, 1, dtype=torch.long),
            labels=None,
            loss_div_factor=torch.tensor(999.0),
        )
        check(
            "live-token divisor replaced by the fixed constant",
            captured.get("loss_div_factor") == 4096.0,
            f"forwarded {captured.get('loss_div_factor')}",
        )
        captured.clear()
        FixedDivisorTransformerTrainModule.model_forward(
            spy, torch.zeros(1, 1, dtype=torch.long), labels=None
        )
        check("eval path (no divisor) is left alone", "loss_div_factor" not in captured)
    finally:
        TransformerTrainModule.model_forward = orig  # type: ignore[method-assign]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--real-model",
        action="store_true",
        help="also run 2 steps of the actual Qwen2.5-0.5B (slow, downloads ~1GB)",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"torch {torch.__version__}  device cpu  vocab {args.vocab}")

    model = stage_model(args.vocab)
    input_ids, dense, split = stage_masks(args.vocab)
    divisor = float(input_ids.numel())  # the fixed divisor: B x S, constant
    stage_gradients(model, input_ids, dense, split, divisor)
    stage_arms(model, input_ids, dense, split, divisor)
    stage_training(args.vocab, divisor, steps=args.steps)
    stage_divisor(args.vocab)

    if args.real_model:
        stage_real_model()

    print("\n" + "=" * 66)
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


def stage_real_model():
    """Two steps of the genuine 494M model, to prove the tiny config was not the only
    thing holding this together."""
    print("\n7. real Qwen2.5-0.5B (494M, CPU — slow)")
    from olmo_core.nn.transformer.qwen import build_qwen2_0_5b, parameter_report

    model = build_qwen2_0_5b(dtype=DType.float32, tie=True)
    rep = parameter_report(model)
    check(
        "parameter count matches tied Qwen2.5-0.5B",
        4.8e8 < rep.unique_params < 5.1e8,
        f"{rep.unique_params:,}",
    )

    try:
        from olmo_core.nn.transformer.qwen import load_hf_weights

        load_hf_weights(model)
        check("HF weights load strictly into the port", True)
    except Exception as e:
        check("HF weights load strictly into the port", False, f"{type(e).__name__}: {e}")
        return

    input_ids, dense, split, _ = make_batch(151936, 1, 64, 24, seed=3)
    divisor = float(input_ids.numel())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    losses = []
    for arm_mask in (dense, split):
        opt.zero_grad(set_to_none=True)
        labels = get_labels(
            {"input_ids": input_ids, "label_mask": arm_mask}, label_ignore_index=-100
        )
        out = model(
            input_ids,
            labels=labels,
            ignore_index=-100,
            loss_reduction="sum",
            loss_div_factor=divisor,
            return_logits=False,
        )
        loss = out[1] if isinstance(out, (tuple, list)) else out
        loss.backward()
        losses.append(loss.item())
    check(
        "real model produces finite loss on both arms",
        all(torch.isfinite(torch.tensor(x)) for x in losses),
        f"dense {losses[0]:.4f}  split {losses[1]:.4f}",
    )
    check("real model: dense loss > split loss", losses[0] > losses[1])


if __name__ == "__main__":
    main()
