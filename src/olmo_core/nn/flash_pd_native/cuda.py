"""Ahead-of-time CUDA binding and custom linear-work backward."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional

import torch

from .contracts import NativePDMode
from .reference import _validate_values
from .routes import prove_selected_maps_bijective


@dataclass(frozen=True)
class NativeCUDACapability:
    """Result of probing the ahead-of-time native extension."""

    available: bool
    reason: str


def _load_extension() -> tuple[Optional[Any], Optional[BaseException]]:
    try:
        return import_module("_flash_pd_native_cuda"), None
    except ModuleNotFoundError as error:
        if error.name != "_flash_pd_native_cuda":
            raise
        return None, error


_EXTENSION, _EXTENSION_ERROR = _load_extension()


@torch.library.custom_op("flash_pd_native::mamba3_siso_backward", mutates_args=())
def _mamba3_siso_backward_op(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    output_real: torch.Tensor,
    output_imag: torch.Tensor,
    grad_output_real: torch.Tensor,
    grad_output_imag: torch.Tensor,
    dictionary_temperature: float,
    router_temperature: float,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if _EXTENSION is None or not diagonal_real.is_cuda:
        raise RuntimeError("Mamba-3 SISO custom backward requires the native CUDA extension")
    gradients = _EXTENSION.paper_backward(
        dictionary_logits,
        selector_logits,
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        output_real,
        output_imag,
        grad_output_real.contiguous(),
        grad_output_imag.contiguous(),
        value_real,
        value_imag,
        beta,
        gamma,
        dictionary_temperature,
        router_temperature,
        chunk_size,
    )
    return tuple(gradients)


@_mamba3_siso_backward_op.register_fake
def _mamba3_siso_backward_fake(
    dictionary_logits,
    selector_logits,
    destination,
    routes,
    diagonal_real,
    diagonal_imag,
    value_real,
    value_imag,
    beta,
    gamma,
    output_real,
    output_imag,
    grad_output_real,
    grad_output_imag,
    dictionary_temperature,
    router_temperature,
    chunk_size,
):
    del (
        destination,
        routes,
        output_real,
        output_imag,
        grad_output_real,
        grad_output_imag,
        dictionary_temperature,
        router_temperature,
        chunk_size,
    )
    return (
        torch.empty_like(dictionary_logits),
        torch.empty_like(selector_logits),
        torch.empty_like(diagonal_real),
        torch.empty_like(diagonal_imag),
        torch.empty_like(value_real),
        torch.empty_like(value_imag),
        torch.empty_like(beta),
        torch.empty_like(gamma),
    )


@torch.library.custom_op("flash_pd_native::mamba3_siso", mutates_args=())
def _mamba3_siso_op(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    destination: torch.Tensor,
    inverse_destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    dictionary_temperature: float,
    router_temperature: float,
    chunk_size: int,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dictionary_logits, selector_logits, dictionary_temperature, router_temperature
    if diagonal_real.is_cuda:
        if _EXTENSION is None:
            raise RuntimeError("native CUDA extension is not installed")
        return tuple(
            _EXTENSION.mamba3_forward(
                destination,
                inverse_destination,
                routes,
                diagonal_real,
                diagonal_imag,
                value_real,
                value_imag,
                beta,
                gamma,
                chunk_size,
                mode,
            )
        )
    from .reference import trapezoidal_reference_scan

    return trapezoidal_reference_scan(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        chunk_size=chunk_size,
        mode=(NativePDMode.PERMUTATION_GATHER if mode == 1 else NativePDMode.GENERAL_SCATTER),
    )


@_mamba3_siso_op.register_fake
def _mamba3_siso_fake(
    dictionary_logits,
    selector_logits,
    destination,
    inverse_destination,
    routes,
    diagonal_real,
    diagonal_imag,
    value_real,
    value_imag,
    beta,
    gamma,
    dictionary_temperature,
    router_temperature,
    chunk_size,
    mode,
):
    del (
        dictionary_logits,
        selector_logits,
        destination,
        inverse_destination,
        routes,
        beta,
        gamma,
        dictionary_temperature,
        router_temperature,
        chunk_size,
        mode,
    )
    del diagonal_real, diagonal_imag
    return torch.empty_like(value_real), torch.empty_like(value_imag)


def _mamba3_siso_setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
    (
        dictionary_logits,
        selector_logits,
        destination,
        _,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        dictionary_temperature,
        router_temperature,
        chunk_size,
        _,
    ) = inputs
    ctx.save_for_backward(
        dictionary_logits,
        selector_logits,
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        *output,
    )
    ctx.dictionary_temperature = dictionary_temperature
    ctx.router_temperature = router_temperature
    ctx.chunk_size = chunk_size


def _mamba3_siso_autograd_backward(
    ctx: Any,
    grad_output_real: torch.Tensor,
    grad_output_imag: torch.Tensor,
):
    gradients = _mamba3_siso_backward_op(
        *ctx.saved_tensors,
        grad_output_real,
        grad_output_imag,
        ctx.dictionary_temperature,
        ctx.router_temperature,
        ctx.chunk_size,
    )
    return (
        gradients[0],
        gradients[1],
        None,
        None,
        None,
        gradients[2],
        gradients[3],
        gradients[4],
        gradients[5],
        gradients[6],
        gradients[7],
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    _mamba3_siso_op,
    _mamba3_siso_autograd_backward,
    setup_context=_mamba3_siso_setup_context,
)


def native_cuda_capability(
    destination: Optional[torch.Tensor] = None,
    routes: Optional[torch.Tensor] = None,
    diagonal_real: Optional[torch.Tensor] = None,
    diagonal_imag: Optional[torch.Tensor] = None,
    bias_real: Optional[torch.Tensor] = None,
    bias_imag: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = 128,
    allow_mixed_diagonal_payload: bool = False,
) -> NativeCUDACapability:
    """Probe extension and tensor constraints without dispatching or falling back."""
    if _EXTENSION is None:
        return NativeCUDACapability(False, "native CUDA extension is not installed")
    if not torch.cuda.is_available():
        return NativeCUDACapability(False, "CUDA is not available")
    if chunk_size < 1 or chunk_size > 128:
        return NativeCUDACapability(False, "chunk_size must be in [1, 128]")
    if destination is None:
        return NativeCUDACapability(True, "native CUDA extension is installed")
    if any(
        value is None
        for value in (
            routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
        )
    ):
        return NativeCUDACapability(False, "all maps, routes, and split values are required")
    assert routes is not None
    assert diagonal_real is not None
    assert diagonal_imag is not None
    assert bias_real is not None
    assert bias_imag is not None
    try:
        _, _, time, state = _validate_values(
            destination,
            routes,
            (diagonal_real, diagonal_imag, bias_real, bias_imag),
            allow_mixed_diagonal_payload=allow_mixed_diagonal_payload,
        )
    except (TypeError, ValueError) as error:
        return NativeCUDACapability(False, str(error))
    if time < 1:
        return NativeCUDACapability(False, "native CUDA kernel requires non-empty time")
    if state >= 1024:
        return NativeCUDACapability(False, "native CUDA kernel requires state below 1024")
    if diagonal_real.dtype not in (torch.float32, torch.bfloat16):
        return NativeCUDACapability(False, "native CUDA kernel supports float32 and bfloat16")
    if bias_real.dtype not in (torch.float32, torch.bfloat16):
        return NativeCUDACapability(
            False,
            "native CUDA payload supports float32 and bfloat16",
        )
    tensors = (
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        return NativeCUDACapability(False, "CUDA tensors are required")
    return NativeCUDACapability(True, "supported by native Flash PD CUDA")


class _NativeFlashPD(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        destination: torch.Tensor,
        inverse_destination: torch.Tensor,
        routes: torch.Tensor,
        diagonal_real: torch.Tensor,
        diagonal_imag: torch.Tensor,
        bias_real: torch.Tensor,
        bias_imag: torch.Tensor,
        chunk_size: int,
        mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert _EXTENSION is not None
        destination_contiguous = destination.contiguous()
        inverse_destination_contiguous = inverse_destination.contiguous()
        routes_contiguous = routes.contiguous()
        diagonal_real_contiguous = diagonal_real.contiguous()
        diagonal_imag_contiguous = diagonal_imag.contiguous()
        bias_real_contiguous = bias_real.contiguous()
        bias_imag_contiguous = bias_imag.contiguous()
        output_real, output_imag = _EXTENSION.forward(
            destination_contiguous,
            inverse_destination_contiguous,
            routes_contiguous,
            diagonal_real_contiguous,
            diagonal_imag_contiguous,
            bias_real_contiguous,
            bias_imag_contiguous,
            chunk_size,
            mode,
        )
        ctx.save_for_backward(
            destination_contiguous,
            routes_contiguous,
            diagonal_real_contiguous,
            diagonal_imag_contiguous,
            output_real,
            output_imag,
        )
        return output_real, output_imag

    @staticmethod
    def backward(
        ctx: Any,
        grad_output_real: torch.Tensor,
        grad_output_imag: torch.Tensor,
    ):
        assert _EXTENSION is not None
        (
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            output_real,
            output_imag,
        ) = ctx.saved_tensors
        gradients = _EXTENSION.backward(
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            output_real,
            output_imag,
            grad_output_real.contiguous(),
            grad_output_imag.contiguous(),
        )
        return (None, None, None, *gradients, None, None)


class _NativeFlashPDPaperTraining(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        dictionary_logits: torch.Tensor,
        selector_logits: torch.Tensor,
        destination: torch.Tensor,
        inverse_destination: torch.Tensor,
        routes: torch.Tensor,
        diagonal_real: torch.Tensor,
        diagonal_imag: torch.Tensor,
        bias_real: torch.Tensor,
        bias_imag: torch.Tensor,
        temperature: float,
        chunk_size: int,
        mode: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert _EXTENSION is not None
        destination_contiguous = destination.contiguous()
        inverse_destination_contiguous = inverse_destination.contiguous()
        routes_contiguous = routes.contiguous()
        diagonal_real_contiguous = diagonal_real.contiguous()
        diagonal_imag_contiguous = diagonal_imag.contiguous()
        bias_real_contiguous = bias_real.contiguous()
        bias_imag_contiguous = bias_imag.contiguous()
        output_real, output_imag = _EXTENSION.forward(
            destination_contiguous,
            inverse_destination_contiguous,
            routes_contiguous,
            diagonal_real_contiguous,
            diagonal_imag_contiguous,
            bias_real_contiguous,
            bias_imag_contiguous,
            chunk_size,
            mode,
        )
        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            destination_contiguous,
            routes_contiguous,
            diagonal_real_contiguous,
            diagonal_imag_contiguous,
            bias_real_contiguous,
            bias_imag_contiguous,
            output_real,
            output_imag,
        )
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        return output_real, output_imag

    @staticmethod
    def backward(
        ctx: Any,
        grad_output_real: torch.Tensor,
        grad_output_imag: torch.Tensor,
    ):
        assert _EXTENSION is not None
        empty = ctx.saved_tensors[4].new_empty((0,))
        gradients = _EXTENSION.paper_backward(
            *ctx.saved_tensors,
            grad_output_real.contiguous(),
            grad_output_imag.contiguous(),
            empty,
            empty,
            empty,
            empty,
            ctx.temperature,
            ctx.temperature,
            ctx.chunk_size,
        )
        return (*gradients[:2], None, None, None, *gradients[2:], None, None, None)


def native_cuda_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    chunk_size: int,
    mode: NativePDMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the three-launch forward and one-launch analytic backward."""
    capability = native_cuda_capability(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size=chunk_size,
    )
    if not capability.available:
        raise RuntimeError(capability.reason)
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven or proof.inverse_destination is None:
            raise ValueError(
                "permutation_gather requires every selected dictionary map to be bijective"
            )
        inverse_destination = proof.inverse_destination
        mode_id = 1
    elif mode == NativePDMode.GENERAL_SCATTER:
        inverse_destination = destination
        mode_id = 0
    else:
        raise ValueError("native_cuda_scan requires a resolved transition mode")
    return _NativeFlashPD.apply(
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size,
        mode_id,
    )


def native_cuda_paper_surrogate_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    mode: NativePDMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run hard native forward with the activated Appendix-C CUDA backward."""
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven or proof.inverse_destination is None:
            raise ValueError("permutation_gather requires bijective selected maps")
        inverse_destination = proof.inverse_destination
        mode_id = 1
    elif mode == NativePDMode.GENERAL_SCATTER:
        inverse_destination = destination
        mode_id = 0
    else:
        raise ValueError("paper training requires a resolved transition mode")
    return _NativeFlashPDPaperTraining.apply(
        dictionary_logits,
        selector_logits,
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        temperature,
        chunk_size,
        mode_id,
    )


def native_cuda_mamba3_siso_surrogate_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    dictionary_temperature: float,
    router_temperature: float,
    chunk_size: int,
    mode: NativePDMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused trapezoidal preprocessing and chunkwise native training."""
    capability = native_cuda_capability(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        chunk_size=chunk_size,
        allow_mixed_diagonal_payload=True,
    )
    if not capability.available:
        raise RuntimeError(capability.reason)
    if beta.shape != diagonal_real.shape[:3] or gamma.shape != beta.shape:
        raise ValueError("beta and gamma must have shape (batch, heads, time)")
    if beta.dtype != value_real.dtype or gamma.dtype != value_real.dtype:
        raise TypeError("beta, gamma, and split payload values must use one dtype")
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven or proof.inverse_destination is None:
            raise ValueError("permutation_gather requires bijective selected maps")
        inverse_destination = proof.inverse_destination
        mode_id = 1
    elif mode == NativePDMode.GENERAL_SCATTER:
        inverse_destination = destination
        mode_id = 0
    else:
        raise ValueError("Mamba-3 SISO training requires a resolved transition mode")
    return _mamba3_siso_op(
        dictionary_logits,
        selector_logits,
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        dictionary_temperature,
        router_temperature,
        chunk_size,
        mode_id,
    )
