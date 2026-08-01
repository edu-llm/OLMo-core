"""Measure the torch backend's fwd+bwd cost, to size the Triton backward's payoff."""

import time

import torch

from olmo_core.nn.attention.kda_householder import chunk_kda_householder


def bench(B: int, T: int, H: int, K: int, V: int, R: int, backend: str, iters: int = 3):
    """:returns: (seconds per fwd+bwd iteration, peak GiB)."""
    dev = "cuda"

    def mk(*s):
        return torch.randn(*s, device=dev, dtype=torch.bfloat16, requires_grad=True)

    q, k, v = mk(B, T, H, K), mk(B, T * R, H, K), mk(B, T * R, H, V)
    g = (-torch.rand(B, T, H, K, device=dev, dtype=torch.float32)).requires_grad_()
    beta = torch.rand(B, T * R, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R, backend=backend)
        o.sum().backward()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters, torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
    print("TRITON backend -- fwd+bwd (the new kernel):")
    for B, T in [(4, 512), (4, 2048), (2, 8192)]:
        for R in (1, 4):
            torch.cuda.reset_peak_memory_stats()
            try:
                dt, mem = bench(B, T, 4, 64, 64, R, "triton")
                print(f"  B{B} T{T:5d} R{R}: {dt*1000:9.1f} ms/iter   peak {mem:5.2f} GiB")
            except Exception as e:  # noqa: BLE001
                print(f"  B{B} T{T:5d} R{R}: FAILED {type(e).__name__}: {str(e)[:60]}")
    print()
    print("torch backend (Python loop over T*R) -- fwd+bwd:")
    for B, T in [(4, 512), (4, 2048), (2, 8192)]:
        for R in (1, 4):
            torch.cuda.reset_peak_memory_stats()
            try:
                dt, mem = bench(B, T, 4, 64, 64, R, "torch")
                print(f"  B{B} T{T:5d} R{R}: {dt*1000:9.1f} ms/iter   peak {mem:5.2f} GiB")
            except Exception as e:  # noqa: BLE001
                print(f"  B{B} T{T:5d} R{R}: FAILED {type(e).__name__}: {str(e)[:60]}")
