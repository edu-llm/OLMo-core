"""Selection masks, frozen weights, EMA history, and attention scoring."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import torch
from torch import Tensor, nn


def _local(tensor: Tensor) -> Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _write(parameter: Tensor, value: Tensor) -> None:
    destination = _local(parameter)
    source = _local(value).detach()
    if destination.shape == source.shape:
        destination.copy_(source.to(destination))
        return
    if hasattr(parameter, "device_mesh") and tuple(source.shape) == tuple(parameter.shape):
        from torch.distributed.tensor import distribute_tensor

        sharded = distribute_tensor(
            source.to(device=destination.device, dtype=destination.dtype),
            parameter.device_mesh,
            parameter.placements,
        )
        destination.copy_(_local(sharded))
        return
    raise ValueError(
        f"reference shape {tuple(source.shape)} cannot populate parameter "
        f"global={tuple(parameter.shape)} local={tuple(destination.shape)}"
    )


def _snapshot(parameter: Tensor) -> Tensor:
    full_tensor = getattr(parameter, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor().detach().clone()
    return _local(parameter).detach().clone()


def per_row_topk(scores: Tensor, fraction: float, valid: Tensor) -> Tensor:
    fraction = min(max(float(fraction), 1e-8), 1.0)
    shape = scores.shape
    values = scores.reshape(-1, shape[-1]).masked_fill(~valid.reshape(-1, shape[-1]), -torch.inf)
    validity = valid.reshape_as(values)
    count = validity.sum(-1)
    keep_count = torch.minimum(torch.clamp((count.float() * fraction).round().long(), min=1), count)
    order = values.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(values.shape[1], device=values.device).expand_as(order),
    )
    return ((ranks < keep_count[:, None]) & validity).reshape(shape)


def per_row_middle(scores: Tensor, fraction: float, valid: Tensor) -> Tensor:
    fraction = min(max(float(fraction), 1e-8), 1.0)
    shape = scores.shape
    values = scores.reshape(-1, shape[-1]).masked_fill(~valid.reshape(-1, shape[-1]), torch.inf)
    validity = valid.reshape_as(values)
    count = validity.sum(-1)
    keep_count = torch.minimum(torch.clamp((count.float() * fraction).round().long(), min=1), count)
    lower = (count - keep_count) // 2
    order = values.argsort(dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(values.shape[1], device=values.device).expand_as(order),
    )
    return ((ranks >= lower[:, None]) & (ranks < (lower + keep_count)[:, None]) & validity).reshape(
        shape
    )


def selection_weights(
    method: str,
    *,
    valid: Tensor,
    keep_fraction: float,
    step: int,
    seed: int,
    current: Optional[Tensor] = None,
    history: Optional[Tensor] = None,
    reference: Optional[Tensor] = None,
    early: Optional[Tensor] = None,
    late: Optional[Tensor] = None,
    attention: Optional[Tensor] = None,
) -> Tensor:
    """Return float weights; all README methods reduce to a deterministic 0/1 mask."""
    if method == "full":
        return valid.float()
    if method == "random":
        generator = torch.Generator(device=valid.device)
        generator.manual_seed(int(seed) + int(step) * 1_000_003)
        scores = torch.rand(valid.shape, device=valid.device, generator=generator)
        mask = per_row_topk(scores, keep_fraction, valid)
    elif method == "rho_excess":
        if current is None or reference is None:
            raise ValueError("RHO-1 requires current and reference losses")
        mask = per_row_topk(current - reference, keep_fraction, valid)
    elif method == "rel_ema":
        if current is None or history is None:
            raise ValueError("relative EMA requires current and history losses")
        mask = per_row_topk(history - current, keep_fraction, valid)
    elif method == "middle_ppl":
        if reference is None:
            raise ValueError("middle-PPL requires frozen reference losses")
        mask = per_row_middle(reference, keep_fraction, valid)
    elif method == "learnability":
        if early is None or late is None:
            raise ValueError("learnability requires early and late reference losses")
        mask = per_row_topk(early - late, keep_fraction, valid)
    elif method == "attention_topk":
        if attention is None:
            raise ValueError("attention selection requires received-attention scores")
        mask = per_row_topk(attention, keep_fraction, valid)
    elif method == "blade":
        if current is None or reference is None:
            raise ValueError("BLADE requires proxy and dynamic-reference losses")
        mask = per_row_topk(reference - current, keep_fraction, valid)
    else:
        raise ValueError(f"unsupported token-selection method {method!r}")
    return mask.float()


class WeightShadow:
    """Full immutable weights temporarily swapped into a sharded training model."""

    VERSION = 1

    def __init__(self, weights: Mapping[str, Tensor]):
        self.weights = {name: value.detach().cpu().clone() for name, value in weights.items()}
        if not self.weights:
            raise ValueError("a weight shadow cannot be empty")

    @classmethod
    def from_state_dict(cls, model: nn.Module, state: Mapping[str, Tensor]) -> "WeightShadow":
        missing = [name for name, _ in model.named_parameters() if name not in state]
        if missing:
            raise KeyError(f"reference is missing model parameters: {missing[:8]}")
        return cls({name: state[name] for name, _ in model.named_parameters()})

    @contextlib.contextmanager
    def swap_to(self, model: nn.Module) -> Iterator[nn.Module]:
        saved: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.weights:
                        saved[name] = _snapshot(parameter)
                        _write(parameter, self.weights[name])
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in saved:
                        _write(parameter, saved[name])

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.VERSION, "weights": self.weights}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION or not isinstance(state.get("weights"), Mapping):
            raise ValueError("invalid frozen-reference state")
        self.weights = {
            str(name): value.detach().cpu().clone()
            for name, value in state["weights"].items()
            if isinstance(value, Tensor)
        }


class EMAHistory:
    """Bias-corrected local-shard EMA, optionally initialized from RefHQ."""

    VERSION = 2

    def __init__(self, model: nn.Module, *, seed: Optional[Mapping[str, Tensor]] = None):
        self.shadow = {
            name: _local(parameter).detach().clone().zero_()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.correction = 0.0
        if seed is not None:
            parameters = dict(model.named_parameters())
            for name, destination in self.shadow.items():
                if name not in seed:
                    raise KeyError(f"EMA seed missing parameter {name!r}")
                source = seed[name]
                if source.shape != destination.shape:
                    parameter = parameters[name]
                    saved = _snapshot(parameter)
                    _write(parameter, source)
                    source = _local(parameter).detach().clone()
                    _write(parameter, saved)
                destination.copy_(source.to(destination))
            self.correction = 1.0

    @property
    def has_history(self) -> bool:
        return self.correction > 0

    @torch.no_grad()
    def update(self, model: nn.Module, alpha: float) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(alpha).add_(_local(parameter).detach(), alpha=1.0 - alpha)
        self.correction = alpha * self.correction + 1.0 - alpha

    @contextlib.contextmanager
    def swap_to(self, model: nn.Module) -> Iterator[nn.Module]:
        if not self.has_history:
            yield model
            return
        saved: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        saved[name] = _local(parameter).detach().clone()
                        _local(parameter).copy_(self.shadow[name] / self.correction)
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in saved:
                        _local(parameter).copy_(saved[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "correction": self.correction,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("unsupported EMA state; pre-v2 states are methodologically invalid")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise ValueError("EMA state parameter set differs")
        for name, value in shadow.items():
            self.shadow[name].copy_(value.to(self.shadow[name]))
        self.correction = float(state["correction"])


def ema_alpha(step: int, *, tau: Optional[float], constant: Optional[float]) -> float:
    if tau is not None:
        if tau <= 0:
            raise ValueError("EMA tau must be positive")
        return 1.0 - math.exp(-float(step) / tau)
    if constant is None:
        raise ValueError("relative EMA requires tau or constant alpha")
    return float(constant)


def attention_received_from_qk(
    query: Tensor, key: Tensor, *, scale: Optional[float] = None, chunk: int = 256
) -> Tensor:
    """Mean-head causal column mass: ``mean_h sum_{j>=i} A[j,i]``."""
    batch, length, heads, dim = query.shape
    if key.shape[2] != heads:
        key = key.repeat_interleave(heads // key.shape[2], dim=2)
    result = torch.zeros(batch, heads, length, device=query.device, dtype=torch.float32)
    q, k = query.float(), key.float()
    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        logits = torch.einsum("bchd,bthd->bhct", q[:, start:stop], k)
        logits.mul_(float(scale if scale is not None else dim**-0.5))
        invalid = (
            torch.arange(length, device=q.device)[None, :]
            > torch.arange(start, stop, device=q.device)[:, None]
        )
        result.add_(logits.masked_fill(invalid[None, None], -torch.inf).softmax(-1).sum(-2))
    return result.mean(1)


@dataclass
class AttentionCapture:
    x: Optional[Tensor] = None
    module: Optional[nn.Module] = None
    kwargs: Optional[dict[str, Any]] = None


@contextlib.contextmanager
def capture_last_attention(model: nn.Module) -> Iterator[AttentionCapture]:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    blocks = list(model.blocks.values()) if hasattr(model.blocks, "values") else list(model.blocks)
    attention = blocks[-1].attention
    capture = AttentionCapture(module=attention)

    def hook(_module, args, kwargs):
        capture.x = args[0].detach()
        capture.kwargs = dict(kwargs)

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield capture
    finally:
        handle.remove()


@torch.no_grad()
def scores_from_capture(capture: AttentionCapture) -> Tensor:
    if capture.x is None or capture.module is None:
        raise RuntimeError("last-layer attention input was not captured")
    attention, x = capture.module, capture.x
    q, k = attention.w_q(x), attention.w_k(x)
    if attention.clip_qkv is not None:
        q = q.clamp(min=-attention.clip_qkv, max=attention.clip_qkv)
        k = k.clamp(min=-attention.clip_qkv, max=attention.clip_qkv)
    head_dim = attention.head_dim
    head_norm = bool(getattr(attention, "use_head_qk_norm", False))
    if not head_norm:
        if attention.q_norm is not None:
            q = attention.q_norm(q)
        if attention.k_norm is not None:
            k = attention.k_norm(k)
    q = q.view(x.shape[0], x.shape[1], -1, head_dim)
    k = k.view(x.shape[0], x.shape[1], -1, head_dim)
    if head_norm:
        if attention.q_norm is not None:
            q = attention.q_norm(q)
        if attention.k_norm is not None:
            k = attention.k_norm(k)
    if getattr(attention, "rope", None) is not None:
        kwargs = capture.kwargs or {}
        q, k = attention._apply_rope(
            q,
            k,
            kwargs.get("start_pos"),
            kwargs.get("pos_sin"),
            kwargs.get("pos_cos"),
            kwargs.get("freqs_cis"),
            kwargs.get("cu_doc_lens"),
        )
    return attention_received_from_qk(q, k, scale=getattr(attention, "softmax_scale", None))
