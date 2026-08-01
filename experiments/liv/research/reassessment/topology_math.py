"""Re-derivation from scratch of the LIV TOPOLOGY decode-traffic claim, 350M geometry.

Pure integer/float arithmetic. No torch, no imports beyond stdlib. Runs on a login node.

Primary sources for the geometry (all cross-checked, see 02_topology_claim.md §0):
  https://huggingface.co/LiquidAI/LFM2-350M/raw/main/config.json  (fetched 2026-08-01)
  Brainlifts/liv_experiment_research/01_lfm2_architecture.md §6.1-6.2 (formula + 6 ckpts)

Nothing here is trusted from a doc; every number is recomputed and asserted.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# §A. Geometry, straight from LFM2-350M config.json
# --------------------------------------------------------------------------------------
D = 1024  # hidden_size
L = 16  # num_hidden_layers
H = 16  # num_attention_heads
G = 8  # num_key_value_heads
HD = 64  # head_dim = hidden_size / num_attention_heads
V = 65536  # vocab_size, TIED (no separate lm_head tensor in the checkpoint)
K = 3  # conv_L_cache
ATTN_IDX = (2, 5, 8, 10, 12, 14)  # full_attn_idxs
N_ATTN = len(ATTN_IDX)  # 6
N_LIV = L - N_ATTN  # 10

# block_ff_dim=6656 with block_auto_adjust_ff_dim=true, block_multiple_of=256:
#   ff = 256 * ceil(int(2/3 * 6656) / 256) = 256 * ceil(4437/256) = 256*18 = 4608
FF_CFG = 6656
FF = 256 * -(-int(2 / 3 * FF_CFG) // 256)
assert FF == 4608, FF

BF16 = 2  # bytes/element


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# §B. Parameter ledger, from first principles
# --------------------------------------------------------------------------------------
rule("B. PARAMETER LEDGER — 350M, derived not quoted")


def liv_mixer(d=D, k=K):
    """in_proj d->3d, out_proj d->d, depthwise conv k*d, no bias, no norm."""
    return 3 * d * d + d * d + k * d  # == 4d^2 + kd


def gqa_mixer(d=D, h=H, g=G, hd=HD):
    """q,k,v,out + per-head q/k RMSNorm of size head_dim each."""
    return d * (h * hd) + 2 * d * (g * hd) + (h * hd) * d + 2 * hd


def mlp(d=D, ff=FF):
    return 3 * d * ff  # SwiGLU w1,w3 up + w2 down


def total_params(d=D, ff=FF, n_attn=N_ATTN, n_liv=N_LIV, k=K, g=G, h=H, hd=HD, v=V):
    return (
        v * d  # tied embeddings
        + d  # final embedding_norm
        + L * (mlp(d, ff) + 2 * d)  # MLP + operator_norm + ffn_norm
        + n_liv * liv_mixer(d, k)
        + n_attn * gqa_mixer(d, h, g, hd)
    )


L0_PARAMS = total_params()
print(f"  LIV mixer  4d^2+kd        = {liv_mixer():>13,}   ({liv_mixer()/D**2:.3f} d^2)")
print(f"  GQA mixer  2d^2+2d(Ghd)+2hd = {gqa_mixer():>11,}   ({gqa_mixer()/D**2:.3f} d^2)")
print(f"  MLP/layer  3dF            = {mlp():>13,}")
print(f"  embeddings V*d (tied)     = {V*D:>13,}")
print(f"  L0 TOTAL                  = {L0_PARAMS:>13,}")
assert L0_PARAMS == 354_483_968, f"L0 ledger MISMATCH: {L0_PARAMS}"
print("  -> matches HF safetensors total 354,483,968 EXACTLY.  [VERIFIED]")
print(f"  NOTE at d=1024 the LIV mixer is {liv_mixer()/gqa_mixer():.4f}x the GQA mixer")
print("       (1.33x, not the 1.60x that holds at d=2048 -- coefficient is d-dependent)")

# A16-P: all 16 layers attention, SwiGLU width solved to re-match L0's params.
FF_A16 = 4820
A16_PARAMS = total_params(ff=FF_A16, n_attn=16, n_liv=0)
print(f"\n  A16-P (16 GQA, ff={FF_A16})    = {A16_PARAMS:>13,}")
print(
    f"  delta vs L0 = {A16_PARAMS-L0_PARAMS:>+13,}  "
    f"({100*(A16_PARAMS-L0_PARAMS)/L0_PARAMS:+.4f}%)"
)
assert A16_PARAMS == 354_388_992, A16_PARAMS
print("  -> matches the arm builder's declared 354,388,992.  [VERIFIED]")

# Is 4820 actually the best width on the multiple-of-4 grid?
best = min(
    range(4000, 5600, 4),
    key=lambda w: abs(total_params(ff=w, n_attn=16, n_liv=0) - L0_PARAMS),
)
print(f"  independent re-solve of the width on the /4 grid -> {best} " f"({'MATCH' if best==FF_A16 else 'MISMATCH'})")

# --------------------------------------------------------------------------------------
# §C. Decode traffic — the actual claim
# --------------------------------------------------------------------------------------
rule("C. DECODE TRAFFIC — bytes/token")


def kv_per_token(n_attn, g=G, hd=HD, b=BF16):
    """K and V, one vector each per attention layer per token."""
    return n_attn * 2 * g * hd * b


KV_L0 = kv_per_token(N_ATTN)
KV_A16 = kv_per_token(16)
DKV = KV_A16 - KV_L0
W_L0 = L0_PARAMS * BF16
W_A16 = A16_PARAMS * BF16

print(f"  ASSUMED dtype: bf16 for BOTH weights and KV (config torch_dtype=bfloat16)")
print(f"  KV/token  L0    (6 attn) = {KV_L0:>8,} B = {KV_L0/1024:>5.2f} KiB")
print(f"  KV/token  A16-P (16 attn)= {KV_A16:>8,} B = {KV_A16/1024:>5.2f} KiB")
print(f"  dKV                      = {DKV:>8,} B = {DKV/1024:>5.2f} KiB   " f"[claim: 20 KiB]")
print(f"  weight bytes/token L0    = {W_L0:>13,} B")
print(f"  weight bytes/token A16-P = {W_A16:>13,} B  (delta {W_A16-W_L0:+,})")

print(f"\n  KV read == weight read (L0):    T = {W_L0/KV_L0:>10,.1f}   [claim: 57,690]")
print(f"  KV read == weight read (A16-P): T = {W_A16/KV_A16:>10,.1f}")

print("\n  KV share of L0 decode traffic:")
for T in (1024, 2048, 4096, 8192, 16384, 32768):
    kv = KV_L0 * T
    print(f"    T={T:>7,}  {100*kv/(W_L0+kv):>6.2f}%")

# --------------------------------------------------------------------------------------
# §D. The 10% threshold — exact, both arms' real weights
# --------------------------------------------------------------------------------------
rule("D. THE 10% DECODE-TRAFFIC THRESHOLD")


def ratio(T, w_a=W_A16, kv_a=KV_A16, w_l=W_L0, kv_l=KV_L0):
    """A16-P traffic / L0 traffic at cache occupancy T."""
    return (w_a + kv_a * T) / (w_l + kv_l * T)


def solve_ratio(target, w_a=W_A16, kv_a=KV_A16, w_l=W_L0, kv_l=KV_L0):
    """(w_a + kv_a T) = target (w_l + kv_l T)  ->  T (kv_a - target kv_l) = target w_l - w_a"""
    num = target * w_l - w_a
    den = kv_a - target * kv_l
    return num / den if den > 0 else float("inf")


# The docs' formula: saving fraction relative to the A16-P baseline, W assumed equal.
doc_T = 0.10 * W_L0 / (DKV - 0.10 * KV_A16)
print(f"  (i)  docs' formula  T = f*W/(dKV - f*KV_A16), W=W_L0, f=0.10")
print(f"         T = {doc_T:,.2f}      [claim: 4,121]  -> {'CONFIRMED' if abs(doc_T-4121)<2 else 'REFUTED'}")
print(f"  (ii) exact, using each arm's own weight bytes, ratio = 1.10")
print(f"         T = {solve_ratio(1.10):,.2f}")
print(f"  (iii) same but expressed as a saving fraction 1 - L0/A16P = 0.10  (=> ratio 1/0.9)")
print(f"         T = {solve_ratio(1/0.9):,.2f}")
print(f"\n  traffic ratio A16-P/L0 at fixed T:")
for T in (512, 1024, 2048, 4096, 8192, 16384, 32768, 131072):
    r = ratio(T)
    print(f"    T={T:>7,}  ratio {r:.4f}   L0 saves {100*(1-1/r):>5.2f}% of A16-P traffic")
print(f"  ceiling as T->inf: ratio {KV_A16/KV_L0:.4f}, saving {100*(1-KV_L0/KV_A16):.2f}%")

# --------------------------------------------------------------------------------------
# §E. dtype sensitivity — the thing the docs treat too lightly
# --------------------------------------------------------------------------------------
rule("E. DTYPE SENSITIVITY (the measured ONNX build was q4 weights)")
print("  KEY ALGEBRAIC FACT: T = f*W/(dKV - f*KV_A16) is INVARIANT under a UNIFORM")
print("  rescaling of both W and KV. Quantizing weights only moves the threshold.\n")
print(f"  {'weight dtype':>14} {'KV dtype':>10} {'W bytes':>14} {'KV/tok L0':>10} {'T(10%)':>10}")
print("  " + "-" * 62)
for wname, wb in (("bf16", 2.0), ("int8", 1.0), ("q4 (+scales)", 0.5625), ("q4 (raw)", 0.5)):
    for kname, kb in (("bf16", 2.0), ("fp32", 4.0), ("int8", 1.0)):
        w = L0_PARAMS * wb
        kv_l = N_ATTN * 2 * G * HD * kb
        kv_a = 16 * 2 * G * HD * kb
        dkv = kv_a - kv_l
        t = 0.10 * w / (dkv - 0.10 * kv_a)
        print(f"  {wname:>14} {kname:>10} {w:>14,.0f} {kv_l:>10,.0f} {t:>10,.0f}")

# --------------------------------------------------------------------------------------
# §F. Scale-invariance of KV bytes/token — how far does it actually go?
# --------------------------------------------------------------------------------------
rule("F. IS KV BYTES/TOKEN REALLY SCALE-INVARIANT?")
FAM = [
    # name, d, n_attn, G, hd, params
    ("LFM2-350M", 1024, 6, 8, 64, 354_483_968),
    ("LFM2-700M", 1536, 6, 8, 64, 742_489_344),
    ("LFM2-1.2B", 2048, 6, 8, 64, 1_170_340_608),
    ("LFM2-2.6B", 2048, 8, 8, 64, 2_569_272_320),
    ("LFM2-8B-A1B", 2048, 6, 8, 64, 8_339_929_856),
    ("LFM2-24B-A2B", 2048, 10, 8, 64, None),
]
print(f"  {'model':>14} {'d':>5} {'n_attn':>7} {'KV/tok':>10} {'W bytes':>15} {'T(KV==W)':>10}")
print("  " + "-" * 68)
for n, d, na, g, hd, p in FAM:
    kv = na * 2 * g * hd * BF16
    if p:
        print(f"  {n:>14} {d:>5} {na:>7} {kv:>8,} B {p*2:>15,} {p*2/kv:>10,.0f}")
    else:
        print(f"  {n:>14} {d:>5} {na:>7} {kv:>8,} B {'(n/a)':>15} {'':>10}")
print("\n  VERDICT: constant at 12,288 B only across the 16-layer 10/6 family (350M/700M/")
print("  1.2B/8B-A1B). 2.6B is 16 KiB, 24B-A2B is 20 KiB. The invariance is in *d*, not in")
print("  scale: KV = n_attn*2*G*hd*b has no d in it, and Liquid happens to hold G*hd=512")
print("  fixed. Correct statement: 'KV bytes/token is independent of d_model'.")

# --------------------------------------------------------------------------------------
# §G. FLOPs/token — reproducing (and correcting) the arm builder's cost table BY HAND
# --------------------------------------------------------------------------------------
rule("G. FLOPS/TOKEN — arm builder's convention vs a consistent one")


def flops_olmo(T, ff, n_attn, n_liv, d=D, k=K):
    """Exactly what olmo_core computes today.

    Attention:   6*params + 12*H*hd*T          (6x = fwd+bwd; 12x = 2 matmuls*2ops*3)
    FeedForward: 6*params
    LMHead:      6*params  (tied weight still registered on the head)
    ShortConv:   2*(in_proj+out_proj) + 2*k*d + 2*d      <-- 2x, i.e. FORWARD ONLY
    Norms contribute nothing; the embedding lookup contributes nothing.
    """
    lm_head = 6 * (V * d + d)
    ff_f = L * 6 * 3 * d * ff
    at_f = n_attn * (6 * gqa_mixer(d) + 12 * H * HD * T)
    sc_f = n_liv * (2 * (3 * d * d + d * d) + 2 * k * d + 2 * d)
    return lm_head + ff_f + at_f + sc_f


def flops_consistent(T, ff, n_attn, n_liv, d=D, k=K, causal=False):
    """ShortConv counted at 6x params like every other module. Optionally causal-correct
    the attention score term (the true causal cost is half the T^2 rectangle)."""
    lm_head = 6 * (V * d + d)
    ff_f = L * 6 * 3 * d * ff
    tmul = 6 if causal else 12
    at_f = n_attn * (6 * gqa_mixer(d) + tmul * H * HD * T)
    sc_f = n_liv * (6 * (3 * d * d + d * d + k * d) + 6 * d)
    return lm_head + ff_f + at_f + sc_f


print(f"  {'T':>8} {'convention':>28} {'L0':>16} {'A16-P':>16} {'ratio':>8}")
print("  " + "-" * 80)
for T in (4096, 32768):
    for label, fn, kw in (
        ("olmo as-committed", flops_olmo, {}),
        ("ShortConv fixed to 6x", flops_consistent, {"causal": False}),
        ("6x + causal attn score", flops_consistent, {"causal": True}),
    ):
        a = fn(T, FF, N_ATTN, N_LIV, **kw)
        b = fn(T, FF_A16, 16, 0, **kw)
        print(f"  {T:>8,} {label:>28} {a:>16,} {b:>16,} {b/a:>7.3f}x")
print("\n  Declared in HANDOFF / liv_arms.py docstring: 1.297x @4K, 1.959x @32K")

# --------------------------------------------------------------------------------------
# §H. Training cost of the minimum experiment
# --------------------------------------------------------------------------------------
rule("H. TRAINING COST — 6ND, 8xA100, 40% MFU")
A100_BF16 = 312e12  # dense bf16 TFLOP/s, no sparsity
L40S_BF16 = 362e12  # L40S dense bf16 (no-sparsity) TFLOP/s


def gpu_hours(tokens, n_params, n_gpu, peak, mfu, flops_per_tok=None):
    total = (flops_per_tok * tokens) if flops_per_tok else (6 * n_params * tokens)
    return total / (n_gpu * peak * mfu) / 3600


for label, toks in (("Chinchilla 20x (7.1B)", 20 * L0_PARAMS), ("40x (14.2B)", 40 * L0_PARAMS), ("30B fixed", 30e9)):
    h = gpu_hours(toks, L0_PARAMS, 8, A100_BF16, 0.40)
    print(f"  L0, {label:>22}: {toks/1e9:>6.2f}B tok -> {h:>7.2f} GPU-h (8xA100@40% MFU)")
print()
for label, fpt in (
    ("L0 @4K   (olmo)", flops_olmo(4096, FF, N_ATTN, N_LIV)),
    ("L0 @4K   (6x fix)", flops_consistent(4096, FF, N_ATTN, N_LIV)),
    ("A16-P @4K", flops_olmo(4096, FF_A16, 16, 0)),
):
    h = gpu_hours(30e9, L0_PARAMS, 8, A100_BF16, 0.40, flops_per_tok=fpt)
    print(f"  30B tok, {label:>20}: fpt={fpt/1e9:>5.3f} GF -> {h:>7.2f} GPU-h")
print(f"\n  6ND check: 6*N = {6*L0_PARAMS/1e9:.3f} GF/token vs measured-convention " f"{flops_olmo(4096,FF,N_ATTN,N_LIV)/1e9:.3f} GF/token")
print("  (6ND overcounts here: it charges the tied 67.1M embedding matrix twice-ish and")
print("   ignores that the embedding *lookup* is not a matmul. Use num_flops_per_token.)")
print()
print(f"  Same on L40S (FarmShare's actual GPU, {L40S_BF16/1e12:.0f} TFLOP/s dense bf16):")
for ng in (1, 4, 8):
    h = gpu_hours(30e9, L0_PARAMS, ng, L40S_BF16, 0.35, flops_per_tok=flops_olmo(4096, FF, N_ATTN, N_LIV))
    print(f"    30B tok on {ng}x L40S @35% MFU: {h:>8.2f} GPU-h  ({h/ng:>7.2f} wall-h)")

rule("DONE")
