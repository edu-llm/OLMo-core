"""
Dispatch to the chunked Kimi Delta Attention kernel from ``flash-linear-attention``.

Companion to :mod:`olmo_core.nn.attention.flash_linear_attn_api`, which holds the equivalent
dispatchers for the gated delta rule, GDN-2 and the causal convolution.
"""

import torch

from olmo_core.nn.attention.flash_linear_attn_api import has_fla

__all__ = ["dispatch_chunk_kda"]


def dispatch_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Dispatch to the chunked Kimi Delta Attention (KDA) kernel from
    `flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_.

    :param q: Queries of shape ``(batch_size, seq_len, n_heads, head_k_dim)``.
    :param k: Keys of shape ``(batch_size, seq_len, n_heads, head_k_dim)``.
    :param v: Values of shape ``(batch_size, seq_len, n_heads, head_v_dim)``.
    :param g: The channel-wise forget gate of shape ``(batch_size, seq_len, n_heads, head_k_dim)``.
        When ``use_gate_in_kernel=True`` this is the *raw* (pre-activation) gate input and the
        kernel computes ``-exp(A_log) * softplus(g + dt_bias)`` internally, otherwise it's
        expected to already be the log-space decay.
    :param beta: The delta-rule step sizes of shape ``(batch_size, seq_len, n_heads)``.
    :param A_log: The per-head log decay scale of shape ``(n_heads,)``. Only used (and required)
        when ``use_gate_in_kernel=True``.
    :param dt_bias: The per-channel gate bias of shape ``(n_heads * head_k_dim,)``. Only used
        when ``use_gate_in_kernel=True``.
    :param scale: The attention scale. Defaults to ``head_k_dim ** -0.5``.
    :param initial_state: An optional initial recurrent state.
    :param output_final_state: Whether to also return the final recurrent state.
    :param use_qk_l2norm_in_kernel: Whether to L2-normalize ``q`` and ``k`` inside the kernel.
    :param use_gate_in_kernel: Whether to compute the KDA gate activation inside the kernel.
    :param cu_seqlens: Cumulative sequence lengths for variable-length inputs. Requires
        ``q.shape[0] == 1``.

    :returns: A tuple of the output with shape ``(batch_size, seq_len, n_heads, head_v_dim)`` and
        the final recurrent state (``None`` unless ``output_final_state=True``).

    :raises RuntimeError: If flash-linear-attention isn't installed.
    :raises ValueError: If ``use_gate_in_kernel=True`` but ``A_log`` wasn't given.
    """
    if not has_fla():
        raise RuntimeError(
            "Kimi Delta Attention requires flash-linear-attention, "
            "which can be installed with 'pip install ai2-olmo-core[fla]'"
        )

    try:
        from fla.ops.kda import chunk_kda
    except ImportError as e:
        raise RuntimeError(
            "Failed to import 'chunk_kda' from flash-linear-attention. "
            "Kimi Delta Attention requires flash-linear-attention >= 0.4.1."
        ) from e

    # NOTE: 'A_log' and 'dt_bias' are consumed from '**kwargs' by 'chunk_kda' and are only
    # honored when 'use_gate_in_kernel=True', so we only forward them in that case.
    gate_kwargs: dict[str, torch.Tensor | None] = {}
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("'A_log' is required when 'use_gate_in_kernel=True'")
        gate_kwargs["A_log"] = A_log
        gate_kwargs["dt_bias"] = dt_bias

    # NOTE: 'chunk_kda' always returns a '(output, final_state)' tuple.
    return chunk_kda(  # type: ignore[reportCallIssue]
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        cu_seqlens=cu_seqlens,
        **gate_kwargs,
    )
