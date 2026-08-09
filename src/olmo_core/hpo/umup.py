"""Same-depth u-μP proxy construction for the three-arm HPO study."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

from ..optim import AdamWConfig

__all__ = [
    "UMUP_BACKEND",
    "UMUP_FORWARD_SUPPORTED",
    "UMuPAdamWConfig",
    "UMuPScaleMetadata",
    "apply_umup_model",
    "apply_umup_parameter_metadata",
    "build_same_depth_umup_proxy",
    "require_official_umup_forward",
    "validate_umup_parity",
]

UMUP_BACKEND = "unit-scaling"
UMUP_FORWARD_SUPPORTED = True
UMUP_EXECUTION_BACKEND = "unit-scaling-public-functional"


@dataclass
class UMuPAdamWConfig(AdamWConfig):
    """AdamW with official u-μP per-parameter learning-rate scaling."""

    def build_groups(self, model, strict: bool = True):
        from unit_scaling.optim import lr_scale_func_adam, scaled_parameters

        groups = super().build_groups(model, strict=strict)
        return scaled_parameters(
            groups,
            lr_scale_func_adam,
            lr=self.lr,
            weight_decay=self.weight_decay,
            independent_weight_decay=True,
        )


def require_official_umup_forward() -> None:
    """Require the official public operations used by the explicit OLMo integration."""

    import unit_scaling
    import unit_scaling.functional as U

    required = (
        "cross_entropy",
        "embedding",
        "linear",
        "linear_readout",
        "residual_add",
        "residual_split",
        "rms_norm",
        "scaled_dot_product_attention",
        "silu_glu",
    )
    missing = [name for name in required if not callable(getattr(U, name, None))]
    if missing:
        raise RuntimeError(
            f"unit-scaling {unit_scaling.__version__} lacks required public operations: {missing}"
        )


@dataclass(frozen=True)
class UMuPScaleMetadata:
    """Auditable width-scaling decision derived by parameter counting."""

    backend: str
    source_architecture: str
    reference_architecture: str
    source_depth: int
    proxy_depth: int
    source_d_model: int
    proxy_d_model: int
    source_n_heads: int
    proxy_n_heads: int
    width_factor: float
    source_non_embedding_params: int
    target_non_embedding_params: int
    proxy_non_embedding_params: int
    relative_parameter_error: float
    parity_tolerance: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata for experiment artifacts."""

        return asdict(self)


def _closest_head_count(d_model: int, desired_heads: float) -> int:
    candidates = [
        heads
        for heads in range(1, d_model + 1)
        if d_model % heads == 0 and 32 <= d_model // heads <= 128
    ]
    if not candidates:
        raise ValueError(f"no valid attention head count for d_model={d_model}")
    return min(candidates, key=lambda heads: (abs(heads - desired_heads), -heads))


def build_same_depth_umup_proxy(
    vocab_size: int,
    *,
    d_model_granularity: int = 32,
    parity_tolerance: float = 0.05,
):
    """Derive an approximately-190M, 16-layer proxy from the 370M architecture.

    The target is the stock 190M model's non-embedding parameter count, which is the naming
    convention used by these OLMo2 size labels. Architecture is *not* copied from the stock
    12-layer model: candidates are rebuilt from the 370M recipe at its original depth and the
    closest candidate is selected by explicit parameter counting.

    :returns: ``(TransformerConfig, UMuPScaleMetadata)``.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if d_model_granularity <= 0:
        raise ValueError("d_model_granularity must be positive")
    if not math.isfinite(parity_tolerance) or not 0.0 <= parity_tolerance < 1.0:
        raise ValueError("parity_tolerance must be finite and in [0, 1)")

    from ..nn.transformer import TransformerConfig
    from ..nn.transformer.config import TransformerBlockType

    source = TransformerConfig.olmo2_370M(vocab_size=vocab_size)
    reference = TransformerConfig.olmo2_190M(vocab_size=vocab_size)
    target_params = reference.num_non_embedding_params

    best = None
    for d_model in range(d_model_granularity, source.d_model + 1, d_model_granularity):
        width_factor = d_model / source.d_model
        n_heads = _closest_head_count(d_model, source.block.sequence_mixer.n_heads * width_factor)
        candidate = TransformerConfig.llama_like(
            d_model=d_model,
            hidden_size_multiplier=1.5,
            n_layers=source.n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            block_name=TransformerBlockType.reordered_norm,
            qk_norm=True,
            rope_theta=500_000,
            layer_norm_eps=1e-6,
        )
        error = abs(candidate.num_non_embedding_params - target_params) / target_params
        estimated_width = source.d_model * math.sqrt(
            target_params / source.num_non_embedding_params
        )
        score = (error, abs(d_model - estimated_width))
        if best is None or score < best[0]:
            best = (score, candidate, n_heads, width_factor)

    assert best is not None
    _, proxy, proxy_heads, width_factor = best
    relative_error = abs(proxy.num_non_embedding_params - target_params) / target_params
    metadata = UMuPScaleMetadata(
        backend=UMUP_BACKEND,
        source_architecture="olmo2_370M",
        reference_architecture="olmo2_190M_parameter_count_only",
        source_depth=source.n_layers,
        proxy_depth=proxy.n_layers,
        source_d_model=source.d_model,
        proxy_d_model=proxy.d_model,
        source_n_heads=source.block.sequence_mixer.n_heads,
        proxy_n_heads=proxy_heads,
        width_factor=width_factor,
        source_non_embedding_params=source.num_non_embedding_params,
        target_non_embedding_params=target_params,
        proxy_non_embedding_params=proxy.num_non_embedding_params,
        relative_parameter_error=relative_error,
        parity_tolerance=parity_tolerance,
    )
    validate_umup_parity(proxy, metadata)
    return proxy, metadata


def validate_umup_parity(model_config, metadata: UMuPScaleMetadata) -> None:
    """Fail closed if the derived proxy violates same-depth or count parity."""

    if metadata.backend != UMUP_BACKEND:
        raise RuntimeError(f"unsupported u-μP backend: {metadata.backend}")
    if metadata.source_depth != 16 or metadata.proxy_depth != metadata.source_depth:
        raise RuntimeError("u-μP proxy must preserve the 370M model's 16-layer depth")
    if model_config.n_layers != metadata.proxy_depth:
        raise RuntimeError("u-μP metadata depth does not match model config")
    if model_config.d_model != metadata.proxy_d_model:
        raise RuntimeError("u-μP metadata width does not match model config")
    if model_config.num_non_embedding_params != metadata.proxy_non_embedding_params:
        raise RuntimeError("u-μP metadata parameter count does not match model config")
    if metadata.relative_parameter_error > metadata.parity_tolerance:
        raise RuntimeError(
            "same-depth u-μP proxy misses the 190M parameter target: "
            f"relative error {metadata.relative_parameter_error:.3%} > "
            f"{metadata.parity_tolerance:.3%}"
        )


def apply_umup_parameter_metadata(model, *, n_layers: int) -> None:
    """Tag every parameter with official ``unit_scaling`` u-μP metadata.

    OLMo owns the module implementations, so the configurator annotates their parameters with
    the protocol consumed by :mod:`unit_scaling.optim`. Transformer-block parameters receive
    the preserved depth, readout parameters are marked ``output``, and normalization/bias
    parameters receive their official categories.
    """

    from unit_scaling.parameter import has_parameter_data

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    for name, parameter in model.named_parameters():
        if name in ("lm_head.weight", "lm_head.w_out.weight"):
            mup_type = "output"
        elif name.endswith(".bias"):
            mup_type = "bias"
        elif "norm" in name.lower() or parameter.ndim == 1:
            mup_type = "norm"
        else:
            mup_type = "weight"
        parameter.mup_type = mup_type
        parameter.mup_scaling_depth = n_layers if name.startswith("blocks.") else None
        if not has_parameter_data(parameter):
            raise RuntimeError(f"failed to attach unit_scaling metadata to parameter {name}")


def _umup_apply_init(init_fun, weight, *, generator=None) -> None:
    """Initialize sharded u-μP weights without illegal inplace writes to autograd views."""

    import torch
    from torch.distributed.tensor import DTensor

    from ..distributed.utils import distribute_like

    def initialize(tensor) -> None:
        if generator is None:
            init_fun(tensor)
        else:
            init_fun(tensor, generator=generator)

    with torch.no_grad():
        if not isinstance(weight, DTensor):
            initialize(weight)
            return
        full = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
        initialize(full)
        weight.copy_(distribute_like(weight, full))


def apply_umup_model(model, *, n_layers: int) -> None:
    """Install explicit official unit-scaled operations on a dense OLMo2 transformer.

    The official documentation recommends manually substituting public unit-scaled operations
    as the standard integration path. This avoids the experimental ``unit_scale()`` Dynamo
    transform, keeps OLMo's module identity intact for FSDP, and supports meta-device
    construction. Unsupported model features fail closed instead of silently executing stock
    operations.
    """

    require_official_umup_forward()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import unit_scaling as uu
    import unit_scaling.functional as U

    from ..distributed.utils import get_local_tensor
    from ..nn.attention.backend import TorchAttentionBackend
    from ..nn.feed_forward import FeedForward
    from ..nn.layer_norm import RMSNorm
    from ..nn.lm_head import LMHead, LMLossImplementation, LMOutputWithLoss
    from ..nn.transformer.block import ReorderedNormTransformerBlock
    if getattr(model, "_umup_execution_backend", None) == UMUP_EXECUTION_BACKEND:
        return
    if n_layers <= 0 or len(model.blocks) != n_layers:
        raise ValueError("n_layers must match the OLMo transformer block count")
    if not isinstance(model.lm_head, LMHead):
        raise TypeError("u-muP currently requires OLMo's default LMHead")
    if model.lm_head.loss_implementation is not LMLossImplementation.default:
        raise RuntimeError("u-muP requires the default, unfused LM loss")
    if model.embeddings is None or not isinstance(model.embeddings, nn.Embedding):
        raise TypeError("u-muP requires a standard OLMo token embedding")
    if model.embed_scale is not None:
        raise RuntimeError("u-muP does not support an additional OLMo embedding scale")

    apply_umup_parameter_metadata(model, n_layers=n_layers)
    residual_rule = uu.transformer_residual_scaling_rule()

    def linear_forward(module, input):
        constraint = getattr(module, "_umup_constraint", "to_output_scale")
        return U.linear(input, module.weight, module.bias, constraint=constraint)

    def readout_forward(module, input):
        return U.linear_readout(input, module.weight, module.bias)

    def embedding_forward(module, input):
        return U.embedding(
            input,
            module.weight,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.sparse,
        )

    def rms_norm_forward(module, x):
        if module.bias is not None:
            raise RuntimeError("unit_scaling.functional.rms_norm does not support an affine bias")
        with torch.autocast(enabled=False, device_type=x.device.type):
            output_dtype = x.dtype
            if module.full_precision:
                x = x.float()
            weight = None if module.weight is None else module.weight.type_as(x)
            return U.rms_norm(x, module.normalized_shape, weight, eps=module.eps).to(output_dtype)

    def feed_forward(module, x):
        if module.activation_fn is not F.silu:
            raise RuntimeError("u-muP OLMo integration currently supports only SwiGLU")
        return module.w2(U.silu_glu(module.w3(x), module.w1(x)))

    def attention_backend_forward(
        module,
        qkv,
        cu_doc_lens=None,
        cu_doc_lens_q=None,
        cu_doc_lens_k=None,
        max_doc_len=None,
        max_doc_len_q=None,
        max_doc_len_k=None,
        local_k_slice=None,
        kv_cache_manager=None,
    ):
        del local_k_slice
        if isinstance(qkv, torch.Tensor):
            raise RuntimeError("unit-scaled Torch attention requires unpacked Q/K/V tensors")
        if module.cp_enabled or kv_cache_manager is not None:
            raise RuntimeError("unit-scaled HPO attention does not support CP or KV caching")
        if module.window_size != (-1, -1):
            raise RuntimeError("unit-scaled HPO attention does not support sliding windows")
        if module.scale is not None:
            raise RuntimeError("u-muP owns attention scaling; softmax_scale must be unset")
        if any(
            value is not None
            for value in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
            )
        ):
            raise RuntimeError("unit-scaled HPO attention does not support intra-document masking")

        q, k, v = qkv
        repetitions = module.n_heads // module.n_kv_heads
        if repetitions > 1:
            k = k.repeat_interleave(repetitions, dim=2)
            v = v.repeat_interleave(repetitions, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = U.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=module.dropout_p,
            is_causal=True,
        )
        return output.transpose(1, 2).contiguous()

    def block_forward(module, x, *, loss_div_factor=None, **kwargs):
        del loss_div_factor
        attention_tau = residual_rule(2 * module.block_idx, 2 * n_layers)
        residual, skip = U.residual_split(x, tau=attention_tau)
        residual = module.attention_norm(module.attention(residual, **kwargs))
        hidden = U.residual_add(residual, skip, tau=attention_tau)

        feed_forward_tau = residual_rule(2 * module.block_idx + 1, 2 * n_layers)
        residual, skip = U.residual_split(hidden, tau=feed_forward_tau)
        residual = module.feed_forward_norm(module.feed_forward(residual))
        return U.residual_add(residual, skip, tau=feed_forward_tau)

    def lm_head_forward(
        module,
        x,
        *,
        labels=None,
        ignore_index=-100,
        loss_reduction="mean",
        z_loss_multiplier=None,
        loss_div_factor=None,
        return_logits=None,
        logits_to_keep=0,
    ):
        batch_size = x.shape[0]
        hidden = module.norm(x) if module.norm is not None else x
        if isinstance(logits_to_keep, int):
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:, :]
                if labels is not None:
                    labels = labels[:, -logits_to_keep:]
        else:
            hidden = hidden.gather(1, logits_to_keep.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
            if labels is not None:
                labels = labels.gather(1, logits_to_keep)
        if labels is None:
            if return_logits is False:
                raise RuntimeError("'return_logits=False' is only valid when labels are provided")
            return module.w_out(hidden)
        if loss_reduction not in ("mean", "sum", "none"):
            raise RuntimeError("unsupported unit-scaled cross entropy reduction")

        logits = module.w_out(hidden)
        local_logits = get_local_tensor(logits).float().view(-1, module.vocab_size)
        local_labels = get_local_tensor(labels).contiguous().view(-1)
        if loss_reduction == "none":
            # unit_scaling.functional.cross_entropy hard-codes a summed PyTorch loss and
            # therefore only accepts "mean" or "sum". Evaluation needs one loss per token.
            # Reproduce that public implementation's unit-scaling transforms, then leave
            # PyTorch's cross entropy unreduced.
            import torch.nn.functional as F

            vocab_size = local_logits.shape[-1]
            scaled_logits = U.scale_bwd(
                local_logits,
                vocab_size / math.sqrt(vocab_size - 1),
            )
            scaled_logits = U.scale_fwd(scaled_logits, 1.0)
            ce_loss = F.cross_entropy(
                scaled_logits,
                local_labels,
                ignore_index=ignore_index,
                reduction="none",
            )
        else:
            ce_loss = U.cross_entropy(
                local_logits,
                local_labels,
                ignore_index=ignore_index,
                reduction=loss_reduction,
            )
        z_loss = None
        if z_loss_multiplier is not None:
            z_squared = local_logits.logsumexp(-1).pow(2)
            mask = local_labels != ignore_index
            if loss_reduction == "mean":
                z_squared = (z_squared * mask).sum() / mask.sum()
            elif loss_reduction == "sum":
                z_squared = (z_squared * mask).sum()
            else:
                z_squared = z_squared * mask
            z_loss = z_loss_multiplier * z_squared
        loss = ce_loss if z_loss is None else ce_loss + z_loss
        if return_logits is False:
            logits = None
        return LMOutputWithLoss(
            logits=logits,
            loss=module._finalize_loss(
                loss,
                batch_size,
                loss_reduction=loss_reduction,
                loss_div_factor=loss_div_factor,
            ),
            ce_loss=module._finalize_loss(
                ce_loss.detach(),
                batch_size,
                loss_reduction=loss_reduction,
                loss_div_factor=loss_div_factor,
                reduce_across_tp_group=False,
            ),
            z_loss=(
                None
                if z_loss is None
                else module._finalize_loss(
                    z_loss.detach(),
                    batch_size,
                    loss_reduction=loss_reduction,
                    loss_div_factor=loss_div_factor,
                    reduce_across_tp_group=False,
                )
            ),
        )

    for block in model.blocks.values():
        if not isinstance(block, ReorderedNormTransformerBlock):
            raise TypeError("u-muP HPO supports only OLMo2 reordered-norm transformer blocks")
        if not isinstance(block.feed_forward, FeedForward):
            raise TypeError("u-muP HPO supports only dense OLMo feed-forward blocks")
        if not isinstance(block.attention.backend, TorchAttentionBackend):
            raise TypeError("u-muP HPO requires OLMo's Torch attention backend")
        block.forward = MethodType(block_forward, block)
        block.feed_forward.forward = MethodType(feed_forward, block.feed_forward)
        for linear in (
            block.feed_forward.w1,
            block.feed_forward.w2,
            block.feed_forward.w3,
        ):
            linear._umup_constraint = None
        block.attention.backend.forward = MethodType(
            attention_backend_forward, block.attention.backend
        )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.forward = MethodType(
                readout_forward if name == "lm_head.w_out" else linear_forward,
                module,
            )
        elif isinstance(module, nn.Embedding):
            module.forward = MethodType(embedding_forward, module)
        elif isinstance(module, RMSNorm):
            module.forward = MethodType(rms_norm_forward, module)
    model.lm_head.forward = MethodType(lm_head_forward, model.lm_head)

    original_init_weights = model.init_weights

    def init_weights(module, *args, **kwargs):
        generator = original_init_weights(*args, **kwargs)
        initialized = set()
        for child in module.modules():
            if isinstance(child, (nn.Linear, nn.Embedding)) and id(child.weight) not in initialized:
                _umup_apply_init(nn.init.normal_, child.weight, generator=generator)
                initialized.add(id(child.weight))
            if isinstance(child, nn.Linear) and child.bias is not None:
                _umup_apply_init(nn.init.zeros_, child.bias)
        # ``to_empty()`` and FSDP materialization replace Parameter objects, so restore the
        # official optimizer protocol on the final parameters rather than trusting meta tensors.
        apply_umup_parameter_metadata(module, n_layers=n_layers)
        return generator

    model.init_weights = MethodType(init_weights, model)
    model._umup_execution_backend = UMUP_EXECUTION_BACKEND
