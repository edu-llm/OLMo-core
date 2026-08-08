import os
import subprocess
import sys
import textwrap

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import olmo_core.nn.memory.counterfactual as counterfactual_module
from olmo_core.nn.memory.counterfactual import counterfactual_lookup
from olmo_core.ops import lngram as lngram_ops
from olmo_core.testing import requires_gpu


def test_lngram_imports_and_runs_without_triton() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    raise ImportError("Triton is unavailable")
                return None

        sys.meta_path.insert(0, BlockTriton())

        import torch
        from olmo_core.nn.memory.counterfactual import counterfactual_lookup
        from olmo_core.ops.lngram import has_lngram_triton

        assert not has_lngram_triton()
        z = torch.randn(1, 3, 4, requires_grad=True)
        tables = (
            torch.randn(16**2, 2, requires_grad=True),
            torch.randn(16**3, 2, requires_grad=True),
        )
        outputs = counterfactual_lookup(z, tables, (2, 3), bits_per_route=4)
        sum(output.sum() for output in outputs).backward()
        assert z.grad is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_triton_dispatch_keeps_cpu_reference_path() -> None:
    z = torch.randn(1, 3, 4)
    codes = counterfactual_module._pack_route_codes(z, 4)
    tables = (torch.randn(16**2, 2), torch.randn(16**3, 2))
    upstreams = (torch.randn(1, 3, 2), torch.randn(1, 3, 2))

    assert (
        lngram_ops._try_counterfactual_grad_z(
            z,
            codes,
            tables,
            (2, 3),
            upstreams,
            bits_per_route=4,
            temperature=1.0,
            scale=1.0,
        )
        is None
    )


@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
def test_triton_interpreter_matches_reference() -> None:
    code = textwrap.dedent(
        """
        import torch

        from olmo_core.kernels.lngram import counterfactual_grad_z_kernel
        from olmo_core.nn.memory.counterfactual import counterfactual_lookup

        torch.manual_seed(7)
        batch_size = 2
        num_routes = 6
        memory_dim = 7
        tables = (
            torch.randn(num_routes * 16**2, memory_dim),
            torch.randn(num_routes * 16**3, memory_dim),
        )
        bit_weights = torch.tensor([1, 2, 4, 8])
        for sequence_length in (1, 2, 3, 5):
            z = torch.randn(batch_size, sequence_length, num_routes * 4)
            upstreams = (
                torch.randn(
                    batch_size,
                    sequence_length,
                    num_routes * memory_dim,
                ),
                torch.randn(
                    batch_size,
                    sequence_length,
                    num_routes * memory_dim,
                ),
            )
            reference_z = z.clone().requires_grad_()
            outputs = counterfactual_lookup(
                reference_z,
                tables,
                (2, 3),
                bits_per_route=4,
                temperature=0.8,
                scale=1.4,
            )
            torch.autograd.backward(outputs, upstreams)

            codes = (
                (z > 0).reshape(
                    batch_size,
                    sequence_length,
                    num_routes,
                    4,
                )
                * bit_weights
            ).sum(-1).to(torch.uint8)
            actual = torch.empty_like(z)
            counterfactual_grad_z_kernel[
                (batch_size * sequence_length, 2)
            ](
                z,
                codes,
                tables[0],
                tables[1],
                upstreams[0],
                upstreams[1],
                actual,
                batch_size,
                sequence_length,
                num_routes,
                memory_dim,
                0.8,
                1.4,
                *z.stride(),
                *codes.stride(),
                *tables[0].stride(),
                *tables[1].stride(),
                *upstreams[0].stride(),
                *upstreams[1].stride(),
                *actual.stride(),
                BLOCK_R=4,
                BLOCK_D=64,
                num_warps=4,
            )
            torch.testing.assert_close(
                actual,
                reference_z.grad,
                rtol=2e-5,
                atol=2e-6,
            )
        """
    )
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
def test_triton_operator_supports_fake_cuda_tensors() -> None:
    assert lngram_ops._counterfactual_grad_z_triton is not None
    with FakeTensorMode():
        z = torch.empty(2, 5, 8, device="cuda")
        codes = torch.empty(2, 5, 2, device="cuda", dtype=torch.long)
        table_order_2 = torch.empty(2 * 16**2, 7, device="cuda")
        table_order_3 = torch.empty(2 * 16**3, 7, device="cuda")
        upstream_order_2 = torch.empty(2, 5, 14, device="cuda")
        upstream_order_3 = torch.empty(2, 5, 14, device="cuda")
        output = lngram_ops._counterfactual_grad_z_triton(
            z,
            codes,
            table_order_2,
            table_order_3,
            upstream_order_2,
            upstream_order_3,
            0.8,
            1.4,
        )

    assert output.shape == z.shape
    assert output.device.type == "cuda"


def _fixture(dtype: torch.dtype):
    torch.manual_seed(17)
    z = torch.randn(2, 5, 8, device="cuda", dtype=dtype)
    tables = (
        torch.randn(2 * 16**2, 7, device="cuda", dtype=dtype),
        torch.randn(2 * 16**3, 7, device="cuda", dtype=dtype),
    )
    upstreams = (
        torch.randn(2, 5, 14, device="cuda", dtype=dtype),
        torch.randn(2, 5, 14, device="cuda", dtype=dtype),
    )
    return z, tables, upstreams


def _reference_grad_z(
    monkeypatch: pytest.MonkeyPatch,
    z: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    upstreams: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        lambda *args, **kwargs: None,
    )
    reference_z = z.detach().clone().requires_grad_()
    outputs = counterfactual_lookup(
        reference_z,
        tuple(table.detach() for table in tables),
        (2, 3),
        bits_per_route=4,
        temperature=0.8,
        scale=1.4,
    )
    torch.autograd.backward(outputs, upstreams)
    assert reference_z.grad is not None
    return reference_z.grad


@requires_gpu
@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
def test_triton_counterfactual_grad_z_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    z, tables, upstreams = _fixture(dtype)
    expected = _reference_grad_z(monkeypatch, z, tables, upstreams)
    codes = counterfactual_module._pack_route_codes(z, 4)

    actual = lngram_ops._try_counterfactual_grad_z(
        z,
        codes,
        tables,
        (2, 3),
        upstreams,
        bits_per_route=4,
        temperature=0.8,
        scale=1.4,
    )

    assert actual is not None
    if dtype is torch.float32:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_gpu
@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
def test_triton_counterfactual_grad_z_supports_strides_and_int32_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    z = torch.randn(2, 5, 16, device="cuda")[..., ::2]
    tables = (
        torch.randn(2 * 16**2, 14, device="cuda")[:, ::2],
        torch.randn(2 * 16**3, 14, device="cuda")[:, ::2],
    )
    upstreams = (
        torch.randn(2, 5, 28, device="cuda")[..., ::2],
        torch.randn(2, 5, 28, device="cuda")[..., ::2],
    )
    expected = _reference_grad_z(monkeypatch, z, tables, upstreams)
    codes = counterfactual_module._pack_route_codes(z, 4).to(torch.int32)

    actual = lngram_ops._try_counterfactual_grad_z(
        z,
        codes,
        tables,
        (2, 3),
        upstreams,
        bits_per_route=4,
        temperature=0.8,
        scale=1.4,
    )

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@requires_gpu
@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
def test_triton_counterfactual_grad_z_compiles() -> None:
    z, tables, upstreams = _fixture(torch.float32)
    codes = counterfactual_module._pack_route_codes(z, 4)
    assert lngram_ops._counterfactual_grad_z_triton is not None

    compiled = torch.compile(
        lngram_ops._counterfactual_grad_z_triton,
        fullgraph=True,
    )
    eager = lngram_ops._counterfactual_grad_z_triton(
        z,
        codes,
        tables[0],
        tables[1],
        upstreams[0],
        upstreams[1],
        0.8,
        1.4,
    )
    actual = compiled(
        z,
        codes,
        tables[0],
        tables[1],
        upstreams[0],
        upstreams[1],
        0.8,
        1.4,
    )

    torch.testing.assert_close(actual, eager)


@requires_gpu
@pytest.mark.skipif(
    not lngram_ops.has_lngram_triton(),
    reason="Triton is not available",
)
def test_triton_counterfactual_grad_z_operator_contract() -> None:
    z, tables, upstreams = _fixture(torch.float32)
    codes = counterfactual_module._pack_route_codes(z, 4)
    assert lngram_ops._counterfactual_grad_z_triton is not None

    torch.library.opcheck(
        lngram_ops._counterfactual_grad_z_triton,
        (
            z,
            codes,
            tables[0],
            tables[1],
            upstreams[0],
            upstreams[1],
            0.8,
            1.4,
        ),
        test_utils=(
            "test_schema",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )
