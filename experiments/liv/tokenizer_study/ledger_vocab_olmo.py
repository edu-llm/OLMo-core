"""Route-2 cross-check: build the REAL olmo-core configs and count parameters.

The olmo-core installed at /scratch/users/ericrcwu/kda/olmo is an OLDER revision
whose ShortConv.num_flops_per_token uses the 2x convention; num_params is
byte-identical to the worktree, so PARAMS are a valid cross-check and FLOPS are
checked against the SC_MULT=2 column.
"""
import sys, os
sys.path.insert(0, "/scratch/users/ericrcwu/liv/tok")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import ledger_vocab as LV

from olmo_core.config import DType
from olmo_core.nn.attention.short_conv import ShortConvConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.config import TransformerConfig
from dataclasses import replace

N_LAYERS, N_HEADS, HEAD_DIM = 16, 16, 64


def build(a, vocab):
    cfg = TransformerConfig.llama_like(
        d_model=a.d_model, vocab_size=vocab, n_layers=N_LAYERS, n_heads=N_HEADS,
        n_kv_heads=a.n_kv_heads, head_dim=HEAD_DIM,
        feed_forward=FeedForwardConfig(hidden_size=a.swiglu_width, bias=False,
                                       dtype=DType.float32),
        qk_norm=True, use_head_qk_norm=True, dtype=DType.float32,
    )
    cfg.tie_word_embeddings = True
    liv = replace(cfg.block, sequence_mixer=ShortConvConfig(
        kernel_size=a.kernel_size, gate_structure=a.gate_structure,
        gate_rank=a.gate_rank, gate_groups=a.gate_groups, dtype=DType.float32))
    ov = {i: liv for i in range(N_LAYERS) if i not in a.attention_layers}
    cfg.block_overrides = ov or None
    return cfg


def count(cfg):
    m = cfg.build(init_device="meta")
    seen, tot = set(), 0
    for p in m.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            tot += p.numel()
    return tot, m


print("ROUTE-2 CROSS-CHECK: olmo-core build vs closed form")
hdr = ("arm".ljust(11) + "V".rjust(8) + "closed-form".rjust(15) + "olmo-core".rjust(15)
       + "par".rjust(5) + "cf f@4K".rjust(16) + "olmo f@4K".rjust(16) + "ok".rjust(4)
       + "cf f@32K".rjust(16) + "olmo f@32K".rjust(16) + "ok".rjust(4))
print(hdr)
bad = 0
for _, v in LV.VOCABS:
    arms = LV.resolved_arms(v)
    for n in LV.ARM_ORDER + ["A16-P*", "N-narrow*"]:
        a = arms[n]
        cf = LV.total_params(a, v)
        oc, m = count(build(a, v))
        cf4 = LV.flops_per_token(a, v, 4096, LV.SC_MULT_OLD)
        cf32 = LV.flops_per_token(a, v, 32768, LV.SC_MULT_OLD)
        o4 = m.num_flops_per_token(4096)
        o32 = m.num_flops_per_token(32768)
        bad += (cf != oc) + (cf4 != o4) + (cf32 != o32)
        print(n.ljust(11) + str(v).rjust(8) + format(cf, ",").rjust(15)
              + format(oc, ",").rjust(15) + ("OK" if cf == oc else "X").rjust(5)
              + format(cf4, ",").rjust(16) + format(o4, ",").rjust(16)
              + ("OK" if cf4 == o4 else "X").rjust(4)
              + format(cf32, ",").rjust(16) + format(o32, ",").rjust(16)
              + ("OK" if cf32 == o32 else "X").rjust(4))
print("")
print("mismatches: " + str(bad))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "liv_arms_ref", "/scratch/users/ericrcwu/liv/tok/liv_arms_ref.py")
ref = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ref)
    print("")
    print("REAL liv_arms.py solvers, per vocab:")
    for _, v in LV.VOCABS:
        l0 = ref._count_params(ref.build_arm("L0", vocab_size=v))
        f = ref._count_params(ref.build_arm("F-r128", vocab_size=v))
        w, pw = ref.solve_swiglu_width("A16-P", target_params=l0, vocab_size=v)
        d, pd = ref.solve_d_model("N-narrow", target_params=f, vocab_size=v)
        print("  V=" + str(v) + "  L0=" + format(l0, ",") + "  F-r128=" + format(f, ",")
              + "  solve_swiglu_width(A16-P)=" + str(w) + " -> " + format(pw, ",")
              + " (" + format(100 * (pw - l0) / l0, "+.4f") + "%)"
              + "  solve_d_model(N-narrow,w=4668)=" + str(d) + " -> " + format(pd, ",")
              + " (" + format(100 * (pd - f) / f, "+.4f") + "%)")
        st = replace(ref.ARMS["N-narrow"], swiglu_width=4608)
        d2, p2 = ref.solve_d_model(st, target_params=f, vocab_size=v)
        w2, p3 = ref.solve_swiglu_width(replace(st, d_model=d2), target_params=f, vocab_size=v)
        print("        two-stage: d=" + str(d2) + " (resid "
              + format(100 * (p2 - f) / f, "+.4f") + "%) then w=" + str(w2) + " -> "
              + format(p3, ",") + " (resid " + format(p3 - f, "+,") + " = "
              + format(100 * (p3 - f) / f, "+.4f") + "%)")
except Exception as e:
    print("")
    print("[liv_arms_ref failed: " + type(e).__name__ + ": " + str(e) + "]")
