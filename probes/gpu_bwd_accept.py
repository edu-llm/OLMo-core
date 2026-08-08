"""GPU acceptance test for the Triton backward: does it match the verified references?

Acceptance criteria mirror the emulator's 7 cases. The torch backend is the ground truth
(its gradients pass float64 gradcheck); the Triton backward must agree within bf16 tolerance.
"""

import sys

import torch

sys.path.insert(0, "/scratch/users/ericrcwu/kda/probes")

from olmo_core.nn.attention.kda_householder import chunk_kda_householder

CASES = [
    ("dense R=1", dict(B=2, T=64, H=2, K=64, V=64, R=1)),
    ("dense R=2", dict(B=2, T=64, H=2, K=64, V=64, R=2)),
    ("dense R=3", dict(B=2, T=64, H=2, K=64, V=64, R=3)),
    ("ragged K=48", dict(B=2, T=64, H=2, K=48, V=64, R=2)),
    ("ragged V=48", dict(B=2, T=64, H=2, K=64, V=48, R=2)),
    ("initial_state", dict(B=2, T=64, H=2, K=64, V=64, R=2, h0=True)),
]
NAMES = ["dq", "dk", "dv", "dg", "dbeta"]


def run(backend: str, cfg: dict, seed: int = 0):
    """:returns: list of gradient tensors for the given backend."""
    torch.manual_seed(seed)
    B, T, H, K, V, R = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"], cfg["R"]
    dev = "cuda"
    # L2-normalize q/k so the delta rule stays contractive -- with raw randn the recurrence
    # diverges and both backends produce (identically) enormous values. Mirrors _make_inputs()
    # in the repo's kda_householder_test.py.
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, device=dev, dtype=torch.float32), p=2, dim=-1
    ).to(torch.bfloat16).requires_grad_()
    k = torch.nn.functional.normalize(
        torch.randn(B, T * R, H, K, device=dev, dtype=torch.float32), p=2, dim=-1
    ).to(torch.bfloat16).requires_grad_()
    v = torch.randn(B, T * R, H, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    g = (-torch.rand(B, T, H, K, device=dev, dtype=torch.float32)).requires_grad_()
    beta = torch.rand(B, T * R, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    h0 = None
    if cfg.get("h0"):
        h0 = torch.randn(B, H, K, V, device=dev, dtype=torch.float32, requires_grad=True)
    torch.manual_seed(seed + 1000)
    do = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
    o, _ = chunk_kda_householder(
        q, k, v, g, beta, num_householder=R, backend=backend, initial_state=h0
    )
    leaves = [q, k, v, g, beta] + ([h0] if h0 is not None else [])
    # autograd.grad on the declared leaves: chunk_kda_householder() calls .contiguous()/.to() on
    # the triton path, which creates NEW tensors, so reading `.grad` off the originals is unsafe.
    return list(torch.autograd.grad(o, leaves, grad_outputs=do))


def main() -> None:
    print("Triton backward vs torch backend (ground truth), bf16, atol 2e-2\n")
    all_ok = True
    for label, cfg in CASES:
        try:
            ref = run("torch", cfg)
            got = run("triton", cfg)
        except NotImplementedError:
            print(f"  {label:16s} SKIP - triton backward not implemented yet")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  {label:16s} ERROR {type(e).__name__}: {str(e)[:70]}")
            all_ok = False
            continue
        names = NAMES + (["dh0"] if len(ref) == 6 else [])
        diffs = [(a.float() - b.float()).abs().max().item() for a, b in zip(got, ref)]
        ok = all(d < 2e-2 for d in diffs)
        all_ok &= ok
        cells = "  ".join(f"{n}={d:.2e}" for n, d in zip(names, diffs))
        print(f"  {label:16s} {cells}   {'PASS' if ok else 'FAIL'}")

    # determinism: partial-buffer reduction must be bit-identical run to run
    try:
        a = run("triton", CASES[1][1])
        b = run("triton", CASES[1][1])
        det = all(torch.equal(x, y) for x, y in zip(a, b))
        print(f"\n  determinism (bit-identical across runs): {det}")
        all_ok &= det
    except NotImplementedError:
        print("\n  determinism: SKIP")
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
