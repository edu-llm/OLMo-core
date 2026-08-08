"""Prove the grouped_mm substitution matches the loop it replaces, before it runs on the block.

    python .edullm/verify_grouped_mm.py

Exits 0 when every case agrees and non-zero on the first that does not, so it can be a gate
rather than something somebody reads. Runs on CPU or GPU; it says which, and a CPU pass is not
a GPU pass -- the kernel that backs ``torch._grouped_mm`` differs by device and the whole point
of the substitution is the CUDA one.

WHAT IS BEING COMPARED. ``DroplessMoEMLP.gmm`` calls a module-level ``gmm`` when the
``grouped_gemm`` package imported and otherwise runs a Python loop over experts. That loop is
transcribed here as ``reference_loop`` rather than imported, deliberately: the point is to
compare against what the library *does*, and importing the method would mean comparing the
patch against itself once the patch is installed.

WHY BIT-EXACT AND NOT A TOLERANCE. A tolerance answers "are these close", and the failure this
is guarding against does not look like drift. Getting the offsets convention wrong by one, or
leaving a transpose undone, produces numbers that are the right shape and the right magnitude
and wrong -- a run that trains, reports a falling loss, and is a different model. Exact
equality is available here because both paths do the same multiplications in the same order
per expert, so anything less would be accepting a difference nobody can explain.
"""

from __future__ import annotations

import sys

import torch


def reference_loop(a, b, batch_sizes, trans_b: bool = False):
    """What ``olmo_core.nn.moe.mlp.DroplessMoEMLP.gmm`` does with no ``grouped_gemm``."""
    out = []
    start = 0
    for i, size in enumerate(batch_sizes.cpu().numpy()):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


def cases(full_size: bool):
    """Shapes to check, each named for the thing it would catch.

    ``full_size`` adds the run's real 4 local experts x 4096 tokens at 2048 x 2048. It is off
    unless asked for because this is ordinarily run under emulation on a laptop, where that
    one case is around 140 GFLOP per call and does not finish. A wrong transpose or a
    misplaced offset is caught at any size; what the large case adds is the kernel's own
    tiling and alignment behaviour at the shapes the run will actually use, and that is only
    meaningful on the GPU anyway.
    """
    yield "even split", 4, [16, 16, 16, 16], 32, 48
    yield "ragged split", 4, [1, 37, 5, 21], 32, 48
    yield "one expert takes everything", 4, [0, 64, 0, 0], 32, 48
    yield "every expert empty but one", 8, [0, 0, 0, 7, 0, 0, 0, 0], 32, 48
    yield "unaligned counts", 4, [3, 11, 29, 21], 64, 64
    yield "single local expert", 1, [64], 32, 48
    yield "the run's widths, few tokens", 4, [8, 24, 0, 16], 2048, 2048
    if full_size:
        yield "the run's real shapes", 4, [4096, 4096, 4096, 4096], 2048, 2048


def check(name, device, num_experts, counts, k, n, trans_b, grouped_mm_gmm) -> bool:
    torch.manual_seed(0)
    m = sum(counts)
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device, requires_grad=True)
    weight_shape = (num_experts, n, k) if trans_b else (num_experts, k, n)
    b = torch.randn(*weight_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    batch_sizes = torch.tensor(counts, dtype=torch.long, device=device)

    expected = reference_loop(a, b, batch_sizes, trans_b=trans_b)
    actual = grouped_mm_gmm(a, b, batch_sizes, trans_b=trans_b)

    label = f"{name} / trans_b={trans_b}"
    if expected.shape != actual.shape:
        print(f"  FAIL {label}: shape {tuple(actual.shape)} != {tuple(expected.shape)}", flush=True)
        return False
    if not torch.equal(expected, actual):
        worst = (expected.float() - actual.float()).abs().max().item()
        print(f"  FAIL {label}: forward differs, largest element {worst:.3e}", flush=True)
        return False

    # Backward matters as much as forward and is the half a shape check cannot see. A
    # transpose that is wrong in a way the forward tolerates shows up here as a gradient of
    # the right shape flowing to the wrong weights.
    seed = torch.randn_like(expected)
    (grad_a_expected, grad_b_expected) = torch.autograd.grad(expected, (a, b), seed)
    (grad_a_actual, grad_b_actual) = torch.autograd.grad(actual, (a, b), seed)
    for what, want, got in (
        ("grad wrt activations", grad_a_expected, grad_a_actual),
        ("grad wrt weights", grad_b_expected, grad_b_actual),
    ):
        if not torch.equal(want, got):
            worst = (want.float() - got.float()).abs().max().item()
            print(f"  FAIL {label}: {what} differs, largest element {worst:.3e}", flush=True)
            return False

    print(f"  ok   {label}", flush=True)
    return True


def main() -> int:
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from train_on_corpus import grouped_mm_gmm, install_grouped_mm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Full size on a GPU always, and on CPU only when asked, because emulated it does not
    # finish. `--full` is there so the case can still be forced somewhere slow and patient.
    full_size = device == "cuda" or "--full" in sys.argv
    print(f"torch {torch.__version__} on {device}, full_size={full_size}", flush=True)
    if device == "cpu":
        print("NOTE: a CPU pass is not a GPU pass. The kernel differs by device.", flush=True)

    print(f"install_grouped_mm says: {install_grouped_mm()}", flush=True)
    if not hasattr(torch, "_grouped_mm"):
        print("FAIL: torch has no _grouped_mm, so there is nothing to verify", flush=True)
        return 1

    failures = 0
    for name, num_experts, counts, k, n in cases(full_size):
        for trans_b in (False, True):
            if not check(name, device, num_experts, counts, k, n, trans_b, grouped_mm_gmm):
                failures += 1

    print(f"\n{failures} failing case(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
