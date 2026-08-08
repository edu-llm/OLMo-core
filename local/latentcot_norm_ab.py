"""
A/B: does feeding the PRE-final-norm hidden state back as a continuous thought
(current behavior) actually change training, vs the POST-final-norm state that
Coconut/CODI's reference implementations feed?

Variant A = current code (h straight from return_hidden_states=True).
Variant B = same, with model.lm_head.norm(h) applied before feeding back.

Uses a deeper-than-unit-test tiny model (8 layers) and the real K=10, because the
drift scales with layers x K -- the 2-layer/K=2 unit-test config is precisely the
regime where it is invisible.

Run: .venv/bin/python local/latentcot_norm_ab.py
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from olmo_core.latentcot import cot, tokens as T
from olmo_core.latentcot import loss as loss_mod
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.train_driver import train_arm
from olmo_core.nn.transformer import TransformerConfig

K = 10
N_LAYERS = 8
D_MODEL = 128
STEPS = 60


def build_dataset(tmp: Path) -> LatentCotDataset:
    path = tmp / "conversations" / "train-00000.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        for s in range(12):
            ex = generate(num_nodes=14, branching=2, depth=3, seed=s, reachable=bool(s % 2))
            f.write(json.dumps(to_sft_record(ex)) + "\n")
    return LatentCotDataset(path, num_continuous_thoughts=K)


def tiny():
    torch.manual_seed(0)
    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
    )
    return cfg.build(init_device="cpu")


def postnorm_run_continuous_thoughts(model, prefix_embeds, num_thoughts):
    """Variant B: normalize each thought with the LM head's final norm before feeding back."""
    embeds = prefix_embeds
    thoughts = []
    for _ in range(num_thoughts):
        hidden = cot._forward_hidden(model, embeds)
        thought = model.lm_head.norm(hidden[:, -1:, :])
        thoughts.append(thought)
        embeds = torch.cat([embeds, thought], dim=1)
    return torch.cat(thoughts, dim=1), embeds


def thought_rms(model, dataset):
    """RMS of each of the K thoughts for one example, under the currently-patched path."""
    ex = dataset[0]
    ids = torch.tensor([ex["input_ids"][: ex["bot_pos"] + 1]], dtype=torch.long)
    with torch.no_grad():
        prefix = cot.embed_tokens(model, ids)
        thoughts, _ = loss_mod.run_continuous_thoughts(model, prefix, K)
    return [thoughts[:, i].float().pow(2).mean().sqrt().item() for i in range(K)]


def run(label, dataset, patched):
    original = loss_mod.run_continuous_thoughts
    if patched:
        loss_mod.run_continuous_thoughts = postnorm_run_continuous_thoughts
    try:
        model = tiny()
        pre = thought_rms(model, dataset)
        history = train_arm(
            model, ARMS["A2"], dataset, steps=STEPS, batch_size=2, lr=3e-4,
            warmup_steps=10, seed=0, log_every=10,
        )
        post = thought_rms(model, dataset)
        print(f"\n===== {label} =====")
        print(f"thought RMS before training (step 1..K): "
              f"{'  '.join(f'{v:.2f}' for v in pre)}")
        print(f"thought RMS after  training (step 1..K): "
              f"{'  '.join(f'{v:.2f}' for v in post)}")
        print(f"{'step':>6} {'total':>10} {'ce_student':>11} {'distill':>10} {'grad_norm':>10}")
        for h in history:
            print(f"{h.get('step', -1):>6} {h.get('loss', float('nan')):>10.4f} "
                  f"{h.get('ce_student', float('nan')):>11.4f} "
                  f"{h.get('distill', float('nan')):>10.4f} "
                  f"{h.get('grad_norm', float('nan')):>10.4f}")
        return history, pre, post
    finally:
        loss_mod.run_continuous_thoughts = original


def main():
    with TemporaryDirectory() as td:
        dataset = build_dataset(Path(td))
        print(f"model: {N_LAYERS} layers, d_model={D_MODEL}, K={K}, steps={STEPS}")
        print(f"history keys: {sorted(train_arm.__doc__ is not None and [] or [])}")
        a, a_pre, a_post = run("A: current (pre-final-norm thought)", dataset, patched=False)
        b, b_pre, b_post = run("B: post-final-norm thought (CODI reference)", dataset, patched=True)

        print("\n===== summary =====")
        print(f"{'':<34}{'A (current)':>14}{'B (post-norm)':>16}")
        print(f"{'final total loss':<34}{a[-1]['loss']:>14.4f}{b[-1]['loss']:>16.4f}")
        print(f"{'final ce_student':<34}{a[-1].get('ce_student', float('nan')):>14.4f}"
              f"{b[-1].get('ce_student', float('nan')):>16.4f}")
        print(f"{'final distill':<34}{a[-1].get('distill', float('nan')):>14.4f}"
              f"{b[-1].get('distill', float('nan')):>16.4f}")
        print(f"{'loss drop (first->last)':<34}"
              f"{a[0]['loss'] - a[-1]['loss']:>14.4f}{b[0]['loss'] - b[-1]['loss']:>16.4f}")
        print(f"{'thought RMS at K, pre-train':<34}{a_pre[-1]:>14.2f}{b_pre[-1]:>16.2f}")
        print(f"{'thought RMS at K, post-train':<34}{a_post[-1]:>14.2f}{b_post[-1]:>16.2f}")


if __name__ == "__main__":
    main()
