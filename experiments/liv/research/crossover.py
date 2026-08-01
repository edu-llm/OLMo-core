"""Decode-traffic crossover: when does KV-cache read overtake weight read?

Geometry: LFM2-1.2B-like. d=2048, 16 layers (10 LIV + 6 GQA), 32 q heads,
8 kv heads, head_dim 64, vocab 65536 tied, SwiGLU ff=8192, bf16.
"""
d, L, n_liv, n_gqa = 2048, 16, 10, 6
hq, hkv, hd = 32, 8, 64
vocab, ff, B = 65536, 8192, 2  # B = bytes/elt (bf16)

liv   = 3*d*d + d*d + 3*d                    # in_proj d->3d, out_proj d->d, depthwise k=3
gqa   = d*(hq*hd) + 2*d*(hkv*hd) + (hq*hd)*d  # q,k,v,o
mlp   = 3*d*ff
emb   = vocab*d                               # tied
total = emb + n_liv*liv + n_gqa*gqa + L*mlp

print(f"LIV mixer      {liv:>14,}   ({liv/d**2:.2f} d^2)")
print(f"GQA mixer      {gqa:>14,}   ({gqa/d**2:.2f} d^2)")
print(f"  -> brainlift claims 16.783M LIV vs 10.486M GQA: "
      f"{'CONFIRMED' if (liv,gqa)==(16783360,10485760) else 'MISMATCH'}")
print(f"MLP/layer      {mlp:>14,}")
print(f"embeddings     {emb:>14,}")
print(f"TOTAL          {total:>14,}  ({total/1e9:.3f}B)\n")

W = total*B                                   # weight bytes read per decode token
kv_tok  = n_gqa*2*hkv*hd*B                    # 6-GQA hybrid
kv_tok16 = L*2*hkv*hd*B                       # all-16-GQA control
print(f"weight bytes/decode-token   {W/1e9:.3f} GB")
print(f"KV bytes/token  6 GQA       {kv_tok:>8,} B ({kv_tok/1024:.0f} KiB)")
print(f"KV bytes/token 16 GQA       {kv_tok16:>8,} B ({kv_tok16/1024:.0f} KiB)")
print(f"\nCROSSOVER (KV read == weight read):")
print(f"   6-GQA hybrid : T = {W/kv_tok:>10,.0f} tokens")
print(f"  16-GQA control: T = {W/kv_tok16:>10,.0f} tokens\n")

hdr = f"{'ctx T':>8} {'KV MB':>9} {'%traffic':>9} {'6->3 saves':>11} {'% total':>8}"
print(hdr); print('-'*len(hdr))
for T in (4096, 8192, 16384, 32768, 131072, 262144):
    kv = kv_tok*T
    tot = W + kv
    save = kv/2                               # pairing 6 banks -> 3
    print(f"{T:>8,} {kv/1e6:>9.1f} {100*kv/tot:>8.1f}% {save/1e6:>10.1f}MB {100*save/tot:>7.2f}%")

print("\nPrefill FLOP crossover (attention vs everything else), per token:")
# non-attention-score FLOPs per token (fwd, 2 FLOP/MAC), all layers
dense = 2*(n_liv*liv + n_gqa*gqa + L*mlp)
# attention scores+AV per token across 6 GQA layers, causal (~T/2 avg): 2*2*hq*hd*(T/2)
for T in (4096, 8192, 16384, 32768, 65536, 131072):
    attn = n_gqa*2*2*hq*hd*(T/2)
    print(f"  T={T:>7,}  dense={dense/1e9:>6.3f} GF  attn-score={attn/1e9:>6.3f} GF  "
          f"attn share={100*attn/(dense+attn):>5.1f}%")
import math
# solve dense == attn  =>  T
Tstar = dense/(n_gqa*2*2*hq*hd/2)
print(f"\n  attention scores == all dense FLOPs at T = {Tstar:,.0f} tokens")
