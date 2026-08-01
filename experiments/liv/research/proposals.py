"""Effect of each brainlift proposal on the SAME decode-traffic budget."""
d, L, n_liv, n_gqa = 2048, 16, 10, 6
hq, hkv, hd, vocab, ff, B = 32, 8, 64, 65536, 8192, 2

liv  = 3*d*d + d*d + 3*d
gqa  = d*(hq*hd) + 2*d*(hkv*hd) + (hq*hd)*d
mlp  = 3*d*ff
emb  = vocab*d
base = emb + n_liv*liv + n_gqa*gqa + L*mlp
Wb   = base*B
kv_tok = n_gqa*2*hkv*hd*B

print("=== P1: low-rank gates (value+out stay full width) ===")
for r in (64, 128, 256, 512):
    liv_r = (d*d + 2*(d*r + r*d)) + d*d + 3*d   # value d->d, 2 gates d->r->d, out d->d
    tot_r = emb + n_liv*liv_r + n_gqa*gqa + L*mlp
    print(f" r={r:>4}: LIV {liv_r/1e6:>6.3f}M ({liv_r/d**2:.2f} d^2)  "
          f"model {tot_r/1e9:.3f}B  weight-read {tot_r*B/1e9:.3f}GB  "
          f"delta {100*(tot_r-base)/base:>6.2f}%")
print(f" r=128 vs brainlift 9.443M: ", end="")
liv128 = (d*d + 2*(d*128+128*d)) + d*d + 3*d
print(f"{liv128:,} -> {'CONFIRMED' if liv128==9443328 else 'MISMATCH ('+str(liv128)+')'}")

print("\n=== P2: cross-layer KV sharing, 6 banks -> 3 ===")
for T in (4096, 16384, 32768, 131072):
    kv = kv_tok*T; save = kv/2
    print(f" T={T:>7,}: saves {save/1e6:>7.1f}MB = {100*save/(Wb+kv):>5.2f}% of decode traffic")
# consumer can also drop its own K/V projections
kvproj = 2*d*(hkv*hd)
print(f" + dropping 3 consumers' K/V projections: {3*kvproj/1e6:.3f}M params "
      f"({100*3*kvproj/base:.2f}% of model)")

print("\n=== P3: multiscale conv — parameters are trivial, STATE is not ===")
print(f" taps/channel: dense 3,5,9,15 = 32   dilated 4x3taps = 12   (stock = 3)")
for name, taps, maxlag in (("stock k=3",3,2), ("dilated 1,2,4,7",12,14), ("dense 3..15",32,14)):
    p = taps*d
    print(f" {name:>16}: conv params {p:>8,}  state = maxlag*d = {maxlag*d:>7,} elt "
          f"= {maxlag*d*B/1024:>6.1f} KiB/layer  ({10*maxlag*d*B/1024:.0f} KiB for 10 LIV layers)")
print(f"\n stock LIV state (10 layers): {10*2*d*B/1024:.0f} KiB")
print(f" vs 6-GQA KV at 32K:          {kv_tok*32768/1024/1024:.0f} MiB   "
      f"-> conv state is {100*10*2*d*B/(kv_tok*32768):.3f}% of KV")
print(f" dilated raises conv state 7x, still {100*10*14*d*B/(kv_tok*32768):.2f}% of KV at 32K")
