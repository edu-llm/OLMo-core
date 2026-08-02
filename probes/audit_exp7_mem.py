"""EXP7: backward workspace memory accounting, from the actual allocations in
`kda_householder_bwd`. Compares against the torch backend's measured 5.65 GiB at B2/T8192/R4."""

GiB = 1024**3


def acct(B, T, H, K, V, R, BV=8):
    NV = -(-V // BV)
    s_dq = B * T * H * K
    s_dk = B * T * R * H * K
    s_db = B * T * R * H
    a = {
        "hs (B*T*H*K*V fp32)": B * T * H * K * V * 4,
        f"dq_p (NV={NV} x fp32)": NV * s_dq * 4,
        f"dg_p (NV={NV} x fp32)": NV * s_dq * 4,
        f"dk_p (NV={NV} x fp32)": NV * s_dk * 4,
        f"db_p (NV={NV} x fp32)": NV * s_db * 4,
        "dv (bf16)": B * T * R * H * V * 2,
        "dh0 (fp32, always alloc)": B * H * K * V * 4,
    }
    # reduction temporaries: .view(NV,-1).sum(0) allocates the fp32 result, then .to(bf16)
    a["reduce temps (dq+dg+dk+db fp32 out)"] = (s_dq * 2 + s_dk + s_db) * 4
    return a, NV


print("=" * 96)
print("EXP7  kda_householder_bwd workspace, per layer, per backward call")
print("=" * 96)
for label, cfg in [
    ("LM scale  B4 T8192 H8 K64 V64 R4", dict(B=4, T=8192, H=8, K=64, V=64, R=4)),
    ("LM scale  B2 T8192 H8 K64 V64 R4", dict(B=2, T=8192, H=8, K=64, V=64, R=4)),
    ("LM scale  B1 T8192 H8 K64 V64 R4", dict(B=1, T=8192, H=8, K=64, V=64, R=4)),
    ("B4 T8192 H8 K64 V64 R2", dict(B=4, T=8192, H=8, K=64, V=64, R=2)),
    ("B4 T2048 H8 K64 V64 R4", dict(B=4, T=2048, H=8, K=64, V=64, R=4)),
    ("accept test B2 T64 H2 K64 V64 R2", dict(B=2, T=64, H=2, K=64, V=64, R=2)),
]:
    a, NV = acct(**cfg)
    tot = sum(a.values())
    print(f"\n {label}   NV={NV}")
    for kk, vv in a.items():
        print(f"    {kk:42s} {vv / GiB:9.4f} GiB")
    print(f"    {'TOTAL transient workspace':42s} {tot / GiB:9.4f} GiB")

print("\n" + "=" * 96)
print(" comparison to the torch backend it replaces")
print("=" * 96)
tot4, _ = acct(B=4, T=8192, H=8, K=64, V=64, R=4)
tot2, _ = acct(B=2, T=8192, H=8, K=64, V=64, R=4)
print("  torch backend MEASURED activations, B2/T8192/R4 : 5.6500 GiB")
print(f"  triton bwd workspace,             B2/T8192/R4 : {sum(tot2.values()) / GiB:.4f} GiB"
      f"   ratio {sum(tot2.values()) / GiB / 5.65:.2f}x")
print(f"  triton bwd workspace,             B4/T8192/R4 : {sum(tot4.values()) / GiB:.4f} GiB")
print("  torch backend saved activations scale as O(B*T*R*H*K*V); triton hs is O(B*T*H*K*V),")
print("  i.e. R-times smaller for hs -- but the NV=8 partial buffers are NOT in the torch path.")
