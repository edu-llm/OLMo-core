"""Triton kernels for packed TWN training."""

from typing import Tuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _pack_twn_kernel(
    weight,
    packed,
    alpha,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    w = tl.load(weight + row * K + offsets, mask=mask, other=0.0).to(tl.float32)
    absw = tl.abs(w)
    delta = 0.7 * tl.sum(absw, axis=0) / K
    survives = (absw > delta) & mask
    count = tl.maximum(tl.sum(survives.to(tl.float32), axis=0), 1.0)
    row_alpha = tl.sum(tl.where(survives, absw, 0.0), axis=0) / count
    tl.store(alpha + row, row_alpha)

    # DeepGrove/MLX affine codes: -1 -> 0, 0 -> 1, +1 -> 2. Invalid tail lanes are
    # encoded as zero trits (code 1); consumers still mask the logical K.
    codes = tl.where(mask, tl.where(w > delta, 2, tl.where(w < -delta, 0, 1)), 1)
    code_matrix = tl.reshape(codes.to(tl.uint32), (BLOCK_WORDS, 16))
    shifts = 2 * tl.arange(0, 16)[None, :]
    words = tl.sum(code_matrix << shifts, axis=1)
    word_offsets = tl.arange(0, BLOCK_WORDS)
    tl.store(
        packed + row * ((K + 15) // 16) + word_offsets,
        words,
        mask=word_offsets < ((K + 15) // 16),
    )


@triton.jit
def _transpose_packed_kernel(
    packed,
    packed_t,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    PACKED_IN: tl.constexpr,
    PACKED_OUT: tl.constexpr,
):
    expert = tl.program_id(0)
    in_idx = tl.program_id(1)
    out_word = tl.program_id(2)
    lanes = tl.arange(0, 16)
    out_idx = out_word * 16 + lanes
    valid = out_idx < OUT_FEATURES
    source_word = tl.load(
        packed + (expert * OUT_FEATURES + out_idx) * PACKED_IN + in_idx // 16,
        mask=valid,
        other=0,
    )
    code = (source_word >> (2 * (in_idx % 16))) & 0x3
    # Invalid rows are represented by the zero-trit code.
    code = tl.where(valid, code, 1)
    word = tl.sum(code.to(tl.uint32) << (2 * lanes), axis=0)
    tl.store(
        packed_t + (expert * IN_FEATURES + in_idx) * PACKED_OUT + out_word,
        word,
    )


def _logical_weight(weight: torch.Tensor, in_dim: int) -> Tuple[torch.Tensor, int, int, int]:
    in_dim %= weight.ndim
    if weight.ndim not in (2, 3):
        raise ValueError(f"packed TWN expects a 2-D or 3-D weight, got {weight.ndim}-D")
    if weight.ndim == 3 and in_dim == 0:
        raise ValueError("the expert axis cannot be the packed TWN input-feature axis")
    logical = weight.movedim(in_dim, -1).contiguous()
    experts = logical.shape[0] if logical.ndim == 3 else 1
    return logical, experts, logical.shape[-2], logical.shape[-1]


def pack_twn(weight: torch.Tensor, in_dim: int):
    """
    Pack a latent BF16 weight with exact FP32 row statistics.

    The returned object is imported lazily to avoid an import cycle with the optional-kernel
    loader in :mod:`olmo_core.ops.ternary`.
    """
    from olmo_core.ops.ternary import PackedTWN

    if not weight.is_cuda:
        raise RuntimeError("the Triton TWN packer requires a CUDA weight")
    if weight.dtype is not torch.bfloat16:
        raise RuntimeError(f"the Triton TWN packer requires BF16, got {weight.dtype}")

    logical, experts, out_features, in_features = _logical_weight(weight, in_dim)
    rows = experts * out_features
    block_k = triton.next_power_of_2(in_features)
    if block_k < 16:
        block_k = 16
    packed_in = triton.cdiv(in_features, 16)
    packed_out = triton.cdiv(out_features, 16)
    flat_weight = logical.reshape(rows, in_features)
    codes = torch.empty(
        (experts, out_features, packed_in), device=weight.device, dtype=torch.uint32
    )
    alpha = torch.empty((experts, out_features), device=weight.device, dtype=torch.bfloat16)
    _pack_twn_kernel[(rows,)](
        flat_weight,
        codes,
        alpha,
        K=in_features,
        BLOCK_K=block_k,
        BLOCK_WORDS=block_k // 16,
    )
    codes_t = torch.empty(
        (experts, in_features, packed_out), device=weight.device, dtype=torch.uint32
    )
    _transpose_packed_kernel[(experts, in_features, packed_out)](
        codes,
        codes_t,
        OUT_FEATURES=out_features,
        IN_FEATURES=in_features,
        PACKED_IN=packed_in,
        PACKED_OUT=packed_out,
    )
    if logical.ndim == 2:
        codes = codes.squeeze(0)
        codes_t = codes_t.squeeze(0)
        alpha = alpha.squeeze(0)
    return PackedTWN(
        codes=codes,
        codes_t=codes_t,
        alpha=alpha,
        in_features=in_features,
        out_features=out_features,
        num_experts=experts,
    )


@triton.jit
def _packed_matmul_kernel(
    x,
    packed,
    alpha,
    expert_ids,
    output,
    M: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    PACKED_IN: tl.constexpr,
    ROWS_PER_EXPERT: tl.constexpr,
    GROUPED: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    out_idx = tl.program_id(1)
    valid_m = offsets_m < M
    if GROUPED:
        if JAGGED:
            expert = tl.load(expert_ids + offsets_m, mask=valid_m, other=0)
        else:
            expert = offsets_m // ROWS_PER_EXPERT
    else:
        expert = tl.zeros((BLOCK_M,), dtype=tl.int32)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in tl.range(0, IN_FEATURES, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offsets_k < IN_FEATURES
        activations = tl.load(
            x + offsets_m[:, None] * IN_FEATURES + offsets_k[None, :],
            mask=valid_m[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.float32)
        words = tl.load(
            packed
            + (expert[:, None] * OUT_FEATURES + out_idx) * PACKED_IN
            + offsets_k[None, :] // 16,
            mask=valid_m[:, None] & valid_k[None, :],
            other=1,
        )
        codes = (words >> (2 * (offsets_k[None, :] % 16))) & 0x3
        signed = tl.where(codes == 2, activations, tl.where(codes == 0, -activations, 0.0))
        acc += tl.sum(signed, axis=1)

    row_alpha = tl.load(alpha + expert * OUT_FEATURES + out_idx, mask=valid_m, other=0.0).to(
        tl.float32
    )
    tl.store(output + offsets_m * OUT_FEATURES + out_idx, acc * row_alpha, mask=valid_m)


@triton.jit
def _packed_matmul_transpose_kernel(
    grad_output,
    packed_t,
    alpha,
    expert_ids,
    grad_input,
    M: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    PACKED_OUT: tl.constexpr,
    ROWS_PER_EXPERT: tl.constexpr,
    GROUPED: tl.constexpr,
    JAGGED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    in_idx = tl.program_id(1)
    valid_m = offsets_m < M
    if GROUPED:
        if JAGGED:
            expert = tl.load(expert_ids + offsets_m, mask=valid_m, other=0)
        else:
            expert = offsets_m // ROWS_PER_EXPERT
    else:
        expert = tl.zeros((BLOCK_M,), dtype=tl.int32)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for out_start in tl.range(0, OUT_FEATURES, BLOCK_O):
        offsets_o = out_start + tl.arange(0, BLOCK_O)
        valid_o = offsets_o < OUT_FEATURES
        grad = tl.load(
            grad_output + offsets_m[:, None] * OUT_FEATURES + offsets_o[None, :],
            mask=valid_m[:, None] & valid_o[None, :],
            other=0.0,
        ).to(tl.float32)
        row_alpha = tl.load(
            alpha + expert[:, None] * OUT_FEATURES + offsets_o[None, :],
            mask=valid_m[:, None] & valid_o[None, :],
            other=0.0,
        ).to(tl.float32)
        words = tl.load(
            packed_t
            + (expert[:, None] * IN_FEATURES + in_idx) * PACKED_OUT
            + offsets_o[None, :] // 16,
            mask=valid_m[:, None] & valid_o[None, :],
            other=1,
        )
        codes = (words >> (2 * (offsets_o[None, :] % 16))) & 0x3
        scaled_grad = grad * row_alpha
        signed = tl.where(codes == 2, scaled_grad, tl.where(codes == 0, -scaled_grad, 0.0))
        acc += tl.sum(signed, axis=1)

    tl.store(grad_input + offsets_m * IN_FEATURES + in_idx, acc, mask=valid_m)


@triton.jit
def _packed_dot_kernel(
    x,
    packed,
    alpha,
    output,
    M: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    PACKED_IN: tl.constexpr,
    ROWS_PER_EXPERT: tl.constexpr,
    BLOCKS_M_PER_EXPERT: tl.constexpr,
    GROUPED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tensor-core packed forward for dense and fixed-capacity expert matrices.

    The previous add/sub kernel assigned one program to one output feature. That reloaded every
    activation once per output and reduced K in scalar ALUs, leaving Hopper tensor cores idle.
    This kernel decodes only the current weight tile into registers and feeds BF16 trits to
    ``tl.dot``; the full quantized matrix is never materialized.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if GROUPED:
        expert = pid_m // BLOCKS_M_PER_EXPERT
        local_block_m = pid_m % BLOCKS_M_PER_EXPERT
        local_m = local_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_m = expert * ROWS_PER_EXPERT + local_m
        valid_m = local_m < ROWS_PER_EXPERT
    else:
        expert = 0
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_m = offsets_m < M

    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = offsets_n < OUT_FEATURES
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, IN_FEATURES, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offsets_k < IN_FEATURES
        activations = tl.load(
            x + offsets_m[:, None] * IN_FEATURES + offsets_k[None, :],
            mask=valid_m[:, None] & valid_k[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        words = tl.load(
            packed
            + (expert * OUT_FEATURES + offsets_n[:, None]) * PACKED_IN
            + offsets_k[None, :] // 16,
            mask=valid_n[:, None] & valid_k[None, :],
            other=1,
        )
        codes = (words >> (2 * (offsets_k[None, :] % 16))) & 0x3
        trits = tl.where(codes == 2, 1.0, tl.where(codes == 0, -1.0, 0.0)).to(
            tl.bfloat16
        )
        acc += tl.dot(activations, tl.trans(trits))

    row_alpha = tl.load(
        alpha + expert * OUT_FEATURES + offsets_n, mask=valid_n, other=0.0
    ).to(tl.float32)
    tl.store(
        output + offsets_m[:, None] * OUT_FEATURES + offsets_n[None, :],
        acc * row_alpha[None, :],
        mask=valid_m[:, None] & valid_n[None, :],
    )


@triton.jit
def _packed_dot_transpose_kernel(
    grad_output,
    packed_t,
    alpha,
    grad_input,
    M: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    PACKED_OUT: tl.constexpr,
    ROWS_PER_EXPERT: tl.constexpr,
    BLOCKS_M_PER_EXPERT: tl.constexpr,
    GROUPED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    """Tensor-core input-gradient for dense and fixed-capacity packed matrices."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    if GROUPED:
        expert = pid_m // BLOCKS_M_PER_EXPERT
        local_block_m = pid_m % BLOCKS_M_PER_EXPERT
        local_m = local_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_m = expert * ROWS_PER_EXPERT + local_m
        valid_m = local_m < ROWS_PER_EXPERT
    else:
        expert = 0
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_m = offsets_m < M

    offsets_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    valid_i = offsets_i < IN_FEATURES
    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for o_start in tl.range(0, OUT_FEATURES, BLOCK_O):
        offsets_o = o_start + tl.arange(0, BLOCK_O)
        valid_o = offsets_o < OUT_FEATURES
        grad = tl.load(
            grad_output + offsets_m[:, None] * OUT_FEATURES + offsets_o[None, :],
            mask=valid_m[:, None] & valid_o[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        words = tl.load(
            packed_t
            + (expert * IN_FEATURES + offsets_i[None, :]) * PACKED_OUT
            + offsets_o[:, None] // 16,
            mask=valid_i[None, :] & valid_o[:, None],
            other=1,
        )
        codes = (words >> (2 * (offsets_o[:, None] % 16))) & 0x3
        trits = tl.where(codes == 2, 1.0, tl.where(codes == 0, -1.0, 0.0)).to(
            tl.bfloat16
        )
        row_alpha = tl.load(
            alpha + expert * OUT_FEATURES + offsets_o, mask=valid_o, other=0.0
        ).to(tl.bfloat16)
        scaled_trits = (trits * row_alpha[:, None]).to(tl.bfloat16)
        acc += tl.dot(grad, scaled_trits)

    tl.store(
        grad_input + offsets_m[:, None] * IN_FEATURES + offsets_i[None, :],
        acc,
        mask=valid_m[:, None] & valid_i[None, :],
    )


def _forward(
    x: torch.Tensor,
    packed: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    in_features: int,
    *,
    grouped: bool,
    jagged: bool,
    rows_per_expert: int,
) -> torch.Tensor:
    x2 = x.reshape(-1, in_features)
    if packed.ndim == 2:
        out_features = packed.shape[0]
        packed3 = packed.unsqueeze(0)
        alpha2 = alpha.unsqueeze(0)
    else:
        out_features = packed.shape[1]
        packed3 = packed
        alpha2 = alpha
    out = torch.empty((x2.shape[0], out_features), device=x.device, dtype=x.dtype)
    if x2.shape[0] == 0:
        return out.reshape(*x.shape[:-1], out_features)
    if jagged:
        _packed_matmul_kernel[(triton.cdiv(x2.shape[0], 16), out_features)](
            x2,
            packed3,
            alpha2,
            expert_ids,
            out,
            M=x2.shape[0],
            OUT_FEATURES=out_features,
            IN_FEATURES=in_features,
            PACKED_IN=triton.cdiv(in_features, 16),
            ROWS_PER_EXPERT=rows_per_expert,
            GROUPED=grouped,
            JAGGED=True,
            BLOCK_M=16,
            BLOCK_K=64,
        )
    else:
        block_m = 64
        blocks_m_per_expert = triton.cdiv(rows_per_expert, block_m) if grouped else 0
        grid_m = (
            packed3.shape[0] * blocks_m_per_expert
            if grouped
            else triton.cdiv(x2.shape[0], block_m)
        )
        _packed_dot_kernel[(grid_m, triton.cdiv(out_features, 64))](
            x2,
            packed3,
            alpha2,
            out,
            M=x2.shape[0],
            OUT_FEATURES=out_features,
            IN_FEATURES=in_features,
            PACKED_IN=triton.cdiv(in_features, 16),
            ROWS_PER_EXPERT=rows_per_expert,
            BLOCKS_M_PER_EXPERT=blocks_m_per_expert,
            GROUPED=grouped,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
    return out.reshape(*x.shape[:-1], out_features)


def _backward_input(
    grad_output: torch.Tensor,
    packed_t: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    in_features: int,
    *,
    grouped: bool,
    jagged: bool,
    rows_per_expert: int,
) -> torch.Tensor:
    if packed_t.ndim == 2:
        packed_t3 = packed_t.unsqueeze(0)
        alpha2 = alpha.unsqueeze(0)
    else:
        packed_t3 = packed_t
        alpha2 = alpha
    grad2 = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input = torch.empty((grad2.shape[0], in_features), device=grad2.device, dtype=grad2.dtype)
    if grad2.shape[0] == 0:
        return grad_input.reshape(*grad_output.shape[:-1], in_features)
    if jagged:
        _packed_matmul_transpose_kernel[(triton.cdiv(grad2.shape[0], 16), in_features)](
            grad2,
            packed_t3,
            alpha2,
            expert_ids,
            grad_input,
            M=grad2.shape[0],
            OUT_FEATURES=grad2.shape[-1],
            IN_FEATURES=in_features,
            PACKED_OUT=triton.cdiv(grad2.shape[-1], 16),
            ROWS_PER_EXPERT=rows_per_expert,
            GROUPED=grouped,
            JAGGED=True,
            BLOCK_M=16,
            BLOCK_O=64,
        )
    else:
        block_m = 64
        blocks_m_per_expert = triton.cdiv(rows_per_expert, block_m) if grouped else 0
        grid_m = (
            packed_t3.shape[0] * blocks_m_per_expert
            if grouped
            else triton.cdiv(grad2.shape[0], block_m)
        )
        _packed_dot_transpose_kernel[(grid_m, triton.cdiv(in_features, 64))](
            grad2,
            packed_t3,
            alpha2,
            grad_input,
            M=grad2.shape[0],
            OUT_FEATURES=grad2.shape[-1],
            IN_FEATURES=in_features,
            PACKED_OUT=triton.cdiv(grad2.shape[-1], 16),
            ROWS_PER_EXPERT=rows_per_expert,
            BLOCKS_M_PER_EXPERT=blocks_m_per_expert,
            GROUPED=grouped,
            BLOCK_M=block_m,
            BLOCK_I=64,
            BLOCK_O=32,
            num_warps=4,
        )
    return grad_input.reshape(*grad_output.shape[:-1], in_features)


def _unused_expert_ids(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.int32)


def dense_packed_matmul(
    x: torch.Tensor, packed: torch.Tensor, alpha: torch.Tensor, in_features: int
) -> torch.Tensor:
    """Dense ``x @ W_q.T`` without materializing ``W_q``."""
    return _forward(
        x,
        packed,
        alpha,
        _unused_expert_ids(x.device),
        in_features,
        grouped=False,
        jagged=False,
        rows_per_expert=0,
    )


def dense_packed_matmul_transpose(
    grad_output: torch.Tensor,
    packed_t: torch.Tensor,
    alpha: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Dense ``grad_output @ W_q`` without materializing ``W_q``."""
    return _backward_input(
        grad_output,
        packed_t,
        alpha,
        _unused_expert_ids(grad_output.device),
        in_features,
        grouped=False,
        jagged=False,
        rows_per_expert=0,
    )


def fixed_grouped_packed_matmul(
    x: torch.Tensor, packed: torch.Tensor, alpha: torch.Tensor, in_features: int
) -> torch.Tensor:
    """Fixed-capacity expert ``x[e] @ W_q[e].T``."""
    return _forward(
        x,
        packed,
        alpha,
        _unused_expert_ids(x.device),
        in_features,
        grouped=True,
        jagged=False,
        rows_per_expert=x.shape[1],
    )


def fixed_grouped_packed_matmul_transpose(
    grad_output: torch.Tensor,
    packed_t: torch.Tensor,
    alpha: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Fixed-capacity expert input gradient."""
    return _backward_input(
        grad_output,
        packed_t,
        alpha,
        _unused_expert_ids(grad_output.device),
        in_features,
        grouped=True,
        jagged=False,
        rows_per_expert=grad_output.shape[1],
    )


def jagged_grouped_packed_matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Dropless expert forward for contiguous expert-grouped tokens."""
    return _forward(
        x,
        packed,
        alpha,
        expert_ids,
        in_features,
        grouped=True,
        jagged=True,
        rows_per_expert=0,
    )


def jagged_grouped_packed_matmul_transpose(
    grad_output: torch.Tensor,
    packed_t: torch.Tensor,
    alpha: torch.Tensor,
    expert_ids: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Dropless expert input gradient for contiguous expert-grouped tokens."""
    return _backward_input(
        grad_output,
        packed_t,
        alpha,
        expert_ids,
        in_features,
        grouped=True,
        jagged=True,
        rows_per_expert=0,
    )
