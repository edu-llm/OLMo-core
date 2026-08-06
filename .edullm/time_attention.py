"""Time one attention block forward and backward, across sequence lengths.

Answers "how long does attention take on this card", which is a question about the
card and the shape rather than about a corpus, so this reads no data and needs none.

The shape is olmo2_190M's: 12 heads of 64, hidden 768. Sequence length is what moves,
because attention is the term that grows with the square of it and everything else in
the block does not.

Timed with CUDA events rather than wall clock, because a kernel launch returns before
the kernel runs and time.time() around an un-synchronised call measures the launch.
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F


def time_one(*, batch: int, seq: int, heads: int, head_dim: int, steps: int, device: str):
    """Median forward and backward milliseconds for one attention block at this shape."""
    hidden = heads * head_dim
    qkv = torch.nn.Linear(hidden, 3 * hidden, device=device, dtype=torch.float16)
    out = torch.nn.Linear(hidden, hidden, device=device, dtype=torch.float16)
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16, requires_grad=True)

    def block() -> torch.Tensor:
        q, k, v = qkv(x).chunk(3, dim=-1)
        shape = (batch, seq, heads, head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out(attended.transpose(1, 2).reshape(batch, seq, hidden))

    for _ in range(3):  # warm up: the first call compiles and allocates
        block().sum().backward()
    torch.cuda.synchronize()

    forward, backward = [], []
    for _ in range(steps):
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        y = block()
        mid.record()
        y.sum().backward()
        end.record()
        torch.cuda.synchronize()
        forward.append(start.elapsed_time(mid))
        backward.append(mid.elapsed_time(end))

    forward.sort()
    backward.sort()
    return forward[len(forward) // 2], backward[len(backward) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", nargs="?", default="local")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device, so there is nothing here worth measuring", flush=True)
        return 64

    name = torch.cuda.get_device_name(0)
    print(f"run           {args.run_name}", flush=True)
    print(f"card          {name}", flush=True)
    print(f"shape         batch {args.batch}, {args.heads} heads of {args.head_dim}", flush=True)
    print(f"median of     {args.steps} timed iterations after 3 warm-up\n", flush=True)
    print(f"{'seq':>6}  {'forward ms':>11}  {'backward ms':>12}  {'total ms':>9}  {'tok/s':>10}")

    for seq in args.sequence_lengths:
        try:
            fwd, bwd = time_one(
                batch=args.batch,
                seq=seq,
                heads=args.heads,
                head_dim=args.head_dim,
                steps=args.steps,
                device="cuda",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{seq:>6}  {'out of memory on this card':>48}", flush=True)
            torch.cuda.empty_cache()
            continue
        total = fwd + bwd
        print(
            f"{seq:>6}  {fwd:>11.3f}  {bwd:>12.3f}  {total:>9.3f}"
            f"  {args.batch * seq / (total / 1000):>10,.0f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
