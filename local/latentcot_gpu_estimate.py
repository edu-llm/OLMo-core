"""
GPU sizing + wall-clock estimate for the Phase-8 latent-CoT campaign (1 seed x 5 arms).

Uses REAL token counts from the actual gen_graph_data.py grid, not guesses. Everything is
an estimate from FLOPs and an assumed MFU -- the point is to bound which GPUs are eligible
and roughly how long, so a scheduling agent can shop. Calibrate on the real box with the
10-minute measurement in the runbook and replace these numbers.

Run: .venv/bin/python local/latentcot_gpu_estimate.py
"""

from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate

# --- config as the runbook will run it -------------------------------------------------
K = 10
STEPS = 5_000
BATCH = 16
SEEDS = 1                    # <-- 1 seed per arm (pilot)
ARMS = ["A0", "A1", "A2", "A3", "A4"]
CODI_ARMS = {"A2", "A3", "A4"}
PARAMS = 474e6
N_LAYERS, D_MODEL = 16, 1024
BEST_EVAL_SIZE, N_CHECKPOINTS = 200, 10   # --best-eval-size, ~steps/--save-every
HELDOUT = 960                             # 4 depths x 2 branchings x 40 x 2, + OOD 2 x 2 x 40 x 2
GEN_TOKENS = 80                           # A0 greedy CoT length (cap 128, stops at <distill>)

# Dense bf16 tensor-core peaks (TFLOPS). Vendor "with sparsity" figures are halved here.
GPUS = [("A100-40GB", 312, 40), ("A100-80GB", 312, 80),
        ("H100-80GB", 989, 80), ("L40S-48GB", 181, 48), ("RTX-A6000-48GB", 155, 48)]
MFUS = [0.02, 0.05, 0.10]


def sample_tokens():
    """Mean token counts over the real difficulty grid (gen_graph_data.py)."""
    teacher, student, direct, prefix = [], [], [], []
    for depth in [2, 3, 4, 6, 5, 8]:
        for branching in [3, 4]:
            for s in range(6):
                ex = generate(num_nodes=max(depth + 1, 6 * depth), branching=branching,
                              depth=depth, seed=s, reachable=bool(s % 2))
                e = encode_example(ex, K)
                teacher.append(len(e["teacher_input_ids"]))
                student.append(len(e["input_ids"]))
                direct.append(len(e["direct_input_ids"]))
                prefix.append(e["bot_pos"] + 1)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return mean(teacher), mean(student), mean(direct), mean(prefix)


def main():
    t_len, s_len, d_len, p_len = sample_tokens()
    print(f"mean token counts (real grid): teacher={t_len:.0f} student={s_len:.0f} "
          f"direct={d_len:.0f} prefix={p_len:.0f}\n")

    # token-positions processed per example, per arm
    thought_loop = sum(p_len + i for i in range(K))   # full re-forward each of K steps
    per_ex = {
        "A0": t_len,                                  # teacher CE only
        "A1": d_len,                                  # direct CE only
        "A2": t_len + thought_loop + s_len,            # teacher + K loop + final student
    }
    per_ex["A3"] = per_ex["A4"] = per_ex["A2"]

    # eval cost per full pass (no KV cache anywhere: A0 regenerates its CoT token by token)
    eval_per_ex = {"A0": GEN_TOKENS * t_len, "A1": d_len, "A2": thought_loop + s_len}
    eval_per_ex["A3"] = eval_per_ex["A4"] = eval_per_ex["A2"]

    print(f"{'arm':>4} {'fwd/step':>9} {'tok-pos/step':>13} {'train PFLOPs':>13} "
          f"{'eval PFLOPs':>12} {'total PFLOPs':>13}")
    totals = {}
    for arm in ARMS:
        fwd = BATCH * (K + 2 if arm in CODI_ARMS else 1)
        tok = BATCH * per_ex[arm]
        train = 6 * PARAMS * tok * STEPS                       # 6ND fwd+bwd
        evals = 2 * PARAMS * eval_per_ex[arm] * (              # 2ND fwd only
            BEST_EVAL_SIZE * N_CHECKPOINTS + HELDOUT)
        totals[arm] = train + evals
        print(f"{arm:>4} {fwd:>9} {tok:>13,.0f} {train/1e15:>13.1f} {evals/1e15:>12.1f} "
              f"{totals[arm]/1e15:>13.1f}")

    campaign = sum(totals.values()) * SEEDS
    slowest = max(totals.values())
    print(f"\ncampaign total ({SEEDS} seed x {len(ARMS)} arms): {campaign/1e15:,.0f} PFLOPs")
    print(f"slowest single arm:                    {slowest/1e15:,.0f} PFLOPs")

    print("\n--- wall-clock estimates (hours) ---")
    print("arms are INDEPENDENT jobs -> 5 GPUs runs them concurrently; wall-clock = slowest arm")
    hdr = "".join(f"{f'MFU {int(m*100)}%':>22}" for m in MFUS)
    print(f"{'GPU':>16}{hdr}")
    print(f"{'':>16}" + "".join(f"{'serial / parallel':>22}" for _ in MFUS))
    for name, tflops, mem in GPUS:
        row = ""
        for mfu in MFUS:
            rate = tflops * 1e12 * mfu
            row += f"{campaign/rate/3600:>10.1f} /{slowest/rate/3600:>10.1f}"
        print(f"{name:>16}{row}")

    # --- memory floor -------------------------------------------------------------------
    print("\n--- memory (per GPU, one arm) ---")
    p_gb = PARAMS * 4 / 1e9
    opt_gb = PARAMS * 8 / 1e9      # AdamW exp_avg + exp_avg_sq, fp32
    grad_gb = PARAMS * 4 / 1e9
    # Activations: codi_loss accumulates all BATCH examples' graphs, then backprops ONCE, so
    # every example's whole K-chain is retained simultaneously. bf16 autocast halves it.
    act_per_fwd = s_len * D_MODEL * N_LAYERS * 10 * 2 / 1e9    # ~10 saved tensors/layer, bf16
    act_gb = act_per_fwd * (K + 2) * BATCH
    print(f"  params fp32            {p_gb:6.2f} GB")
    print(f"  AdamW states           {opt_gb:6.2f} GB")
    print(f"  grads                  {grad_gb:6.2f} GB")
    print(f"  retained activations  ~{act_gb:6.2f} GB   <-- all {BATCH} examples' K-chains at once")
    print(f"  TOTAL                 ~{p_gb+opt_gb+grad_gb+act_gb:6.2f} GB (CODI arm; A0/A1 far less)")
    print(f"\n  eligible: {', '.join(n for n, _, m in GPUS if m > p_gb+opt_gb+grad_gb+act_gb)}")
    print("  activations scale LINEARLY with --batch-size; halve it to halve that term.")


if __name__ == "__main__":
    main()
