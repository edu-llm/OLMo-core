"""
MFU audit for the latent-CoT Phase-8 loop: how many forwards per optimizer step, at what
batch size and sequence length, and how much of it is recomputation.

Run: .venv/bin/python local/latentcot_mfu_audit.py
"""

from collections import Counter

from olmo_core.latentcot.arms import DEFAULT_K
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate

K = DEFAULT_K
BATCH = 16          # runbook --batch-size
N_LAYERS, D_MODEL = 16, 1024
PARAMS = 474e6      # olmo3_370M


def main():
    # Match the runbook's data shape: a depth sweep, both reachable and not.
    examples = []
    for depth in range(2, 9):
        for s in range(12):
            ex = generate(num_nodes=24, branching=3, depth=depth, seed=s * 100 + depth,
                          reachable=bool(s % 2))
            examples.append(encode_example(ex, K))

    teacher_len = [len(e["teacher_input_ids"]) for e in examples]
    student_len = [len(e["input_ids"]) for e in examples]
    prefix_len = [e["bot_pos"] + 1 for e in examples]

    def stats(name, xs):
        xs = sorted(xs)
        print(f"  {name:<16} min={xs[0]:4d}  median={xs[len(xs)//2]:4d}  max={xs[-1]:4d}  "
              f"mean={sum(xs)/len(xs):7.1f}")

    print(f"n={len(examples)} examples, K={K}")
    stats("teacher seq", teacher_len)
    stats("student seq", student_len)
    stats("prefix (->bot)", prefix_len)

    print("\nprefix-length distribution (does bucketing work without padding/masks?)")
    counts = Counter(prefix_len)
    print(f"  {len(counts)} distinct prefix lengths over {len(examples)} examples")
    top = counts.most_common(8)
    print("  most common: " + ", ".join(f"len{l}x{c}" for l, c in top))
    print(f"  a batch of {BATCH} drawn at random would split into ~"
          f"{min(BATCH, len(counts))} buckets on average")

    # ---- forwards per optimizer step, as written (batch dim 1 per example) ----
    print("\nforwards per optimizer step (batch_size=16), CODI arm A2/A3/A4:")
    per_ex_thought_tokens = sum(p + i for p, i in
                                [(sum(prefix_len) / len(prefix_len), i) for i in range(K)])
    print(f"  teacher forward            : 1 per example  -> {BATCH:4d} forwards @ batch=1")
    print(f"  thought loop (K={K})        : {K} per example -> {BATCH * K:4d} forwards @ batch=1")
    print(f"  final student forward      : 1 per example  -> {BATCH:4d} forwards @ batch=1")
    print(f"  TOTAL                      : {BATCH * (K + 2):4d} sequential forwards, "
          f"every one at batch=1")
    print(f"  over 5000 steps            : {BATCH * (K + 2) * 5000:,} batch-1 forward passes")

    # ---- recomputation in the thought loop ----
    mean_prefix = sum(prefix_len) / len(prefix_len)
    with_recompute = sum(mean_prefix + i for i in range(K))       # what we do now
    with_kv_cache = mean_prefix + K                              # what a cache would cost
    print(f"\nthought-loop token-positions processed per example (mean prefix {mean_prefix:.0f}):")
    print(f"  as written (full re-forward each step): {with_recompute:8.0f}")
    print(f"  with a KV cache                       : {with_kv_cache:8.0f}")
    print(f"  -> {with_recompute / with_kv_cache:.1f}x redundant compute in the latent path")

    # ---- rough MFU ceiling ----
    mean_student = sum(student_len) / len(student_len)
    tokens_per_step = BATCH * (mean_student + sum(mean_prefix + i for i in range(K)))
    flops_per_step = 6 * PARAMS * tokens_per_step  # 6ND fwd+bwd
    print(f"\nrough FLOPs accounting (6*N*D):")
    print(f"  token-positions per step (all branches): {tokens_per_step:,.0f}")
    print(f"  FLOPs per step                         : {flops_per_step:.3e}")
    for name, tflops in [("A100 bf16 (312 TF)", 312e12), ("A100 fp32 no-TF32 (19.5 TF)", 19.5e12)]:
        ideal_ms = flops_per_step / tflops * 1e3
        print(f"  {name:<28} ideal step time @100% MFU: {ideal_ms:7.2f} ms")
    print("\n  ^ the gap between that and the real step time IS the MFU headroom; at batch=1")
    print("    per forward, a 370M model is launch/bandwidth-bound, not compute-bound.")


if __name__ == "__main__":
    main()
