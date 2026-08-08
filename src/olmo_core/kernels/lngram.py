"""Triton kernels for Lngram counterfactual gradients."""

import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _load_code(
    codes,
    batch_idx,
    token_idx,
    routes,
    route_mask,
    sequence_length,
    stride_batch,
    stride_sequence,
    stride_route,
):
    token_valid = (token_idx >= 0) & (token_idx < sequence_length)
    safe_token = tl.maximum(0, tl.minimum(token_idx, sequence_length - 1))
    pointers = (
        codes + batch_idx * stride_batch + safe_token * stride_sequence + routes * stride_route
    )
    return tl.load(pointers, mask=route_mask & token_valid, other=0).to(tl.int32) & 15


@triton.jit
def _window_scores(
    table,
    upstream,
    hard_addresses,
    current_codes,
    batch_idx,
    output_token,
    routes,
    valid_window,
    memory_dim,
    table_stride_row,
    table_stride_dim,
    upstream_stride_batch,
    upstream_stride_sequence,
    upstream_stride_feature,
    BIT_SHIFT: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    dimensions = tl.arange(0, BLOCK_D)
    bits = tl.arange(0, 4)
    route_mask = valid_window & (routes >= 0)
    value_mask = route_mask[:, None] & (dimensions[None, :] < memory_dim)

    hard_pointers = (
        table + hard_addresses[:, None] * table_stride_row + dimensions[None, :] * table_stride_dim
    )
    hard_values = tl.load(hard_pointers, mask=value_mask, other=0.0).to(tl.float32)

    upstream_pointers = (
        upstream
        + batch_idx * upstream_stride_batch
        + output_token * upstream_stride_sequence
        + (routes[:, None] * memory_dim + dimensions[None, :]) * upstream_stride_feature
    )
    upstream_values = tl.load(upstream_pointers, mask=value_mask, other=0.0).to(tl.float32)

    flip_deltas = 1 << (bits + BIT_SHIFT)
    flipped_addresses = hard_addresses[:, None] ^ flip_deltas[None, :]
    flipped_pointers = (
        table
        + flipped_addresses[:, :, None] * table_stride_row
        + dimensions[None, None, :] * table_stride_dim
    )
    flipped_mask = route_mask[:, None, None] & (dimensions[None, None, :] < memory_dim)
    flipped_values = tl.load(flipped_pointers, mask=flipped_mask, other=0.0).to(tl.float32)

    differences = flipped_values - hard_values[:, None, :]
    scores = tl.sum(
        differences * upstream_values[:, None, :],
        axis=2,
    )
    current_bits = (current_codes[:, None] >> bits[None, :]) & 1
    signs = 1.0 - 2.0 * current_bits.to(tl.float32)
    return scores * signs


@triton.jit
def counterfactual_grad_z_kernel(
    z,
    codes,
    table_order_2,
    table_order_3,
    upstream_order_2,
    upstream_order_3,
    grad_z,
    batch_size,
    sequence_length,
    num_routes,
    memory_dim,
    temperature,
    scale,
    z_stride_batch,
    z_stride_sequence,
    z_stride_channel,
    codes_stride_batch,
    codes_stride_sequence,
    codes_stride_route,
    table_2_stride_row,
    table_2_stride_dim,
    table_3_stride_row,
    table_3_stride_dim,
    upstream_2_stride_batch,
    upstream_2_stride_sequence,
    upstream_2_stride_feature,
    upstream_3_stride_batch,
    upstream_3_stride_sequence,
    upstream_3_stride_feature,
    output_stride_batch,
    output_stride_sequence,
    output_stride_channel,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_program = tl.program_id(0)
    route_program = tl.program_id(1)
    batch_idx = token_program // sequence_length
    token_idx = token_program - batch_idx * sequence_length

    routes = route_program * BLOCK_R + tl.arange(0, BLOCK_R)
    route_mask = routes < num_routes
    current_codes = _load_code(
        codes,
        batch_idx,
        token_idx,
        routes,
        route_mask,
        sequence_length,
        codes_stride_batch,
        codes_stride_sequence,
        codes_stride_route,
    )
    score_matrix = tl.zeros((BLOCK_R, 4), dtype=tl.float32)

    # Order 2, current token at address offset 0: [t, t + 1].
    valid = route_mask & (token_idx + 1 < sequence_length)
    next_code = _load_code(
        codes,
        batch_idx,
        token_idx + 1,
        routes,
        route_mask,
        sequence_length,
        codes_stride_batch,
        codes_stride_sequence,
        codes_stride_route,
    )
    address = routes * 256 + current_codes + next_code * 16
    score_matrix += _window_scores(
        table_order_2,
        upstream_order_2,
        address,
        current_codes,
        batch_idx,
        token_idx + 1,
        routes,
        valid,
        memory_dim,
        table_2_stride_row,
        table_2_stride_dim,
        upstream_2_stride_batch,
        upstream_2_stride_sequence,
        upstream_2_stride_feature,
        BIT_SHIFT=0,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )

    # Order 2, current token at address offset 1: [t - 1, t].
    valid = route_mask & (token_idx >= 1)
    previous_code = _load_code(
        codes,
        batch_idx,
        token_idx - 1,
        routes,
        route_mask,
        sequence_length,
        codes_stride_batch,
        codes_stride_sequence,
        codes_stride_route,
    )
    address = routes * 256 + previous_code + current_codes * 16
    score_matrix += _window_scores(
        table_order_2,
        upstream_order_2,
        address,
        current_codes,
        batch_idx,
        token_idx,
        routes,
        valid,
        memory_dim,
        table_2_stride_row,
        table_2_stride_dim,
        upstream_2_stride_batch,
        upstream_2_stride_sequence,
        upstream_2_stride_feature,
        BIT_SHIFT=4,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )

    # Order 3, current token at address offset 0: [t, t + 1, t + 2].
    valid = route_mask & (token_idx + 2 < sequence_length)
    next_next_code = _load_code(
        codes,
        batch_idx,
        token_idx + 2,
        routes,
        route_mask,
        sequence_length,
        codes_stride_batch,
        codes_stride_sequence,
        codes_stride_route,
    )
    address = routes * 4096 + current_codes + next_code * 16 + next_next_code * 256
    score_matrix += _window_scores(
        table_order_3,
        upstream_order_3,
        address,
        current_codes,
        batch_idx,
        token_idx + 2,
        routes,
        valid,
        memory_dim,
        table_3_stride_row,
        table_3_stride_dim,
        upstream_3_stride_batch,
        upstream_3_stride_sequence,
        upstream_3_stride_feature,
        BIT_SHIFT=0,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )

    # Order 3, current token at address offset 1: [t - 1, t, t + 1].
    valid = route_mask & (token_idx >= 1) & (token_idx + 1 < sequence_length)
    address = routes * 4096 + previous_code + current_codes * 16 + next_code * 256
    score_matrix += _window_scores(
        table_order_3,
        upstream_order_3,
        address,
        current_codes,
        batch_idx,
        token_idx + 1,
        routes,
        valid,
        memory_dim,
        table_3_stride_row,
        table_3_stride_dim,
        upstream_3_stride_batch,
        upstream_3_stride_sequence,
        upstream_3_stride_feature,
        BIT_SHIFT=4,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )

    # Order 3, current token at address offset 2: [t - 2, t - 1, t].
    valid = route_mask & (token_idx >= 2)
    previous_previous_code = _load_code(
        codes,
        batch_idx,
        token_idx - 2,
        routes,
        route_mask,
        sequence_length,
        codes_stride_batch,
        codes_stride_sequence,
        codes_stride_route,
    )
    address = routes * 4096 + previous_previous_code + previous_code * 16 + current_codes * 256
    score_matrix += _window_scores(
        table_order_3,
        upstream_order_3,
        address,
        current_codes,
        batch_idx,
        token_idx,
        routes,
        valid,
        memory_dim,
        table_3_stride_row,
        table_3_stride_dim,
        upstream_3_stride_batch,
        upstream_3_stride_sequence,
        upstream_3_stride_feature,
        BIT_SHIFT=8,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )

    bits = tl.arange(0, 4)
    channels = routes[:, None] * 4 + bits[None, :]
    output_mask = route_mask[:, None]
    z_pointers = (
        z + batch_idx * z_stride_batch + token_idx * z_stride_sequence + channels * z_stride_channel
    )
    logits = tl.load(z_pointers, mask=output_mask, other=0.0).to(tl.float32)
    probabilities = tl.sigmoid(temperature * logits)
    slopes = scale * temperature * probabilities * (1.0 - probabilities)

    output_pointers = (
        grad_z
        + batch_idx * output_stride_batch
        + token_idx * output_stride_sequence
        + channels * output_stride_channel
    )
    tl.store(output_pointers, slopes * score_matrix, mask=output_mask)
