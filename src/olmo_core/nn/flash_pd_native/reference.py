"""Dense oracle and linear-state PyTorch recurrence for native Flash PD-SSM."""

import torch
from torch.nn import functional as F

from .contracts import NativePDMode, SISOScanCache
from .routes import _validate_compact_shapes, prove_selected_maps_bijective


def _active_hardmax_surrogate_gradient(
    logits: torch.Tensor,
    active_index: torch.Tensor,
    active_gradient: torch.Tensor,
    *,
    dim: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply Appendix-C's selected softmax-Jacobian row."""
    probabilities = torch.softmax(logits / temperature, dim=dim)
    index = active_index.unsqueeze(dim)
    active_probability = torch.gather(probabilities, dim, index)
    one_hot = torch.zeros_like(logits).scatter(dim, index, 1)
    return (
        active_gradient.unsqueeze(dim)
        * active_probability
        * (one_hot - probabilities)
        / temperature
    )


def _validate_values(
    destination: torch.Tensor,
    routes: torch.Tensor,
    values: tuple[torch.Tensor, ...],
) -> tuple[int, int, int, int]:
    _validate_compact_shapes(destination, routes)
    if len(values) != 4:
        raise ValueError("expected split diagonal and bias real/imag tensors")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("diagonal and bias real/imag tensors must have identical shapes")
    if len(shape) != 4:
        raise ValueError("value tensors must have shape (batch, heads, time, state)")
    batch, heads, time, state = shape
    if routes.shape != (batch, heads, time):
        raise ValueError(
            f"routes must have shape {(batch, heads, time)}, got {tuple(routes.shape)}"
        )
    if destination.shape[0] != heads or destination.shape[-1] != state:
        raise ValueError("destination heads/state dimensions must match value tensors")
    if any(not value.is_floating_point() for value in values):
        raise TypeError("split real/imag tensors must use floating-point dtypes")
    if any(value.dtype != values[0].dtype for value in values):
        raise TypeError("split real/imag tensors must use one common dtype")
    if any(value.device != values[0].device for value in values):
        raise ValueError("split real/imag tensors must be on one device")
    if destination.device != values[0].device or routes.device != values[0].device:
        raise ValueError("maps, routes, and values must be on one device")
    return batch, heads, time, state


def _selected_destination(
    destination: torch.Tensor,
    routes: torch.Tensor,
    token: int,
) -> torch.Tensor:
    heads, _, state = destination.shape
    batch = routes.shape[0]
    dictionary = destination.unsqueeze(0).expand(batch, -1, -1, -1)
    index = routes[:, :, token].long().view(batch, heads, 1, 1).expand(-1, -1, 1, state)
    return torch.gather(dictionary, dim=2, index=index).squeeze(2).long()


def dense_scan_oracle(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    mode: NativePDMode | str = NativePDMode.GENERAL_SCATTER,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the recurrence through explicit dense matrices."""
    _, _, time, state = _validate_values(
        destination,
        routes,
        (diagonal_real, diagonal_imag, bias_real, bias_imag),
    )
    mode = NativePDMode(mode)
    if mode == NativePDMode.AUTO:
        mode = NativePDMode.GENERAL_SCATTER
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven:
            raise ValueError(
                "permutation_gather requires every selected dictionary map to be bijective"
            )

    diagonal = torch.complex(diagonal_real, diagonal_imag)
    bias = torch.complex(bias_real, bias_imag)
    current = torch.zeros(
        diagonal.shape[:2] + (state,),
        dtype=diagonal.dtype,
        device=diagonal.device,
    )
    states = []
    for token in range(time):
        token_destination = _selected_destination(destination, routes, token)
        matrix = F.one_hot(token_destination, num_classes=state).movedim(-1, -2)
        matrix = matrix.to(diagonal.dtype) * diagonal[:, :, token].unsqueeze(-2)
        current = torch.einsum("bhij,bhj->bhi", matrix, current) + bias[:, :, token]
        states.append(current)
    if not states:
        return bias_real.clone(), bias_imag.clone()
    output = torch.stack(states, dim=2)
    return output.real, output.imag


def reference_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    mode: NativePDMode | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate with O(N) state and explicit scatter/gather semantics."""
    _, _, time, state = _validate_values(
        destination,
        routes,
        (diagonal_real, diagonal_imag, bias_real, bias_imag),
    )
    mode = NativePDMode(mode)
    proof = None
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven:
            raise ValueError(
                "permutation_gather requires every selected dictionary map to be bijective"
            )
    elif mode == NativePDMode.AUTO:
        proof = prove_selected_maps_bijective(destination, routes)
        mode = NativePDMode.PERMUTATION_GATHER if proof.proven else NativePDMode.GENERAL_SCATTER

    current_real = torch.zeros_like(diagonal_real[:, :, 0]) if time else diagonal_real[:, :, :0]
    current_imag = torch.zeros_like(diagonal_imag[:, :, 0]) if time else diagonal_imag[:, :, :0]
    output_real = []
    output_imag = []
    for token in range(time):
        token_destination = _selected_destination(destination, routes, token)
        product_real = (
            diagonal_real[:, :, token] * current_real - diagonal_imag[:, :, token] * current_imag
        )
        product_imag = (
            diagonal_real[:, :, token] * current_imag + diagonal_imag[:, :, token] * current_real
        )
        if mode == NativePDMode.GENERAL_SCATTER:
            next_real = torch.zeros_like(current_real).scatter_add(
                -1, token_destination, product_real
            )
            next_imag = torch.zeros_like(current_imag).scatter_add(
                -1, token_destination, product_imag
            )
        else:
            assert proof is not None and proof.inverse_destination is not None
            inverse = _selected_destination(proof.inverse_destination, routes, token)
            next_real = torch.gather(product_real, -1, inverse)
            next_imag = torch.gather(product_imag, -1, inverse)
        current_real = next_real + bias_real[:, :, token]
        current_imag = next_imag + bias_imag[:, :, token]
        output_real.append(current_real)
        output_imag.append(current_imag)

    if not output_real:
        return bias_real.clone(), bias_imag.clone()
    return torch.stack(output_real, dim=2), torch.stack(output_imag, dim=2)


def trapezoidal_reference_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    chunk_size: int,
    mode: NativePDMode | str,
    initial_cache: SISOScanCache | None = None,
    return_cache: bool = False,
):
    """
    Evaluate the exact Mamba-3 SISO PD recurrence with linear state storage.

    The update is
    ``h_t = A_t (h_{t-1} + beta_t v_{t-1}) + gamma_t v_t``
    with ``A_t = P_t diag(d_t)``. The prior input ``v`` is part of the
    recurrent cache, so one-token calls are exactly equivalent to prefill.
    """
    batch, heads, time, state = _validate_values(
        destination,
        routes,
        (diagonal_real, diagonal_imag, value_real, value_imag),
    )
    if chunk_size not in (32, 64, 128):
        raise ValueError(f"chunk_size must be one of (32, 64, 128), got {chunk_size}")
    if beta.shape != (batch, heads, time) or gamma.shape != beta.shape:
        raise ValueError(
            f"beta and gamma must have shape {(batch, heads, time)}, got "
            f"{tuple(beta.shape)} and {tuple(gamma.shape)}"
        )
    if beta.dtype != diagonal_real.dtype or gamma.dtype != diagonal_real.dtype:
        raise TypeError("beta, gamma, and split complex values must use one common dtype")
    if beta.device != diagonal_real.device or gamma.device != diagonal_real.device:
        raise ValueError("beta, gamma, and split complex values must be on one device")

    mode = NativePDMode(mode)
    proof = None
    if mode == NativePDMode.PERMUTATION_GATHER:
        proof = prove_selected_maps_bijective(destination, routes)
        if not proof.proven:
            raise ValueError(
                "permutation_gather requires every selected dictionary map to be bijective"
            )
    elif mode == NativePDMode.AUTO:
        proof = prove_selected_maps_bijective(destination, routes)
        mode = NativePDMode.PERMUTATION_GATHER if proof.proven else NativePDMode.GENERAL_SCATTER

    cache_shape = (batch, heads, state)
    if initial_cache is None:
        current_real = diagonal_real.new_zeros(cache_shape)
        current_imag = diagonal_imag.new_zeros(cache_shape)
        previous_value_real = value_real.new_zeros(cache_shape)
        previous_value_imag = value_imag.new_zeros(cache_shape)
    else:
        cache_tensors = tuple(initial_cache)
        if any(tensor.shape != cache_shape for tensor in cache_tensors):
            raise ValueError(f"cache tensors must have shape {cache_shape}")
        if any(tensor.dtype != diagonal_real.dtype for tensor in cache_tensors):
            raise TypeError("cache and scan values must use one common dtype")
        if any(tensor.device != diagonal_real.device for tensor in cache_tensors):
            raise ValueError("cache and scan values must be on one device")
        (
            current_real,
            current_imag,
            previous_value_real,
            previous_value_imag,
        ) = cache_tensors

    output_real = []
    output_imag = []
    for token in range(time):
        token_destination = _selected_destination(destination, routes, token)
        transition_input_real = current_real + beta[:, :, token, None] * previous_value_real
        transition_input_imag = current_imag + beta[:, :, token, None] * previous_value_imag
        product_real = (
            diagonal_real[:, :, token] * transition_input_real
            - diagonal_imag[:, :, token] * transition_input_imag
        )
        product_imag = (
            diagonal_real[:, :, token] * transition_input_imag
            + diagonal_imag[:, :, token] * transition_input_real
        )
        if mode == NativePDMode.GENERAL_SCATTER:
            next_real = torch.zeros_like(current_real).scatter_add(
                -1, token_destination, product_real
            )
            next_imag = torch.zeros_like(current_imag).scatter_add(
                -1, token_destination, product_imag
            )
        else:
            assert proof is not None and proof.inverse_destination is not None
            inverse = _selected_destination(proof.inverse_destination, routes, token)
            next_real = torch.gather(product_real, -1, inverse)
            next_imag = torch.gather(product_imag, -1, inverse)
        current_real = next_real + gamma[:, :, token, None] * value_real[:, :, token]
        current_imag = next_imag + gamma[:, :, token, None] * value_imag[:, :, token]
        previous_value_real = value_real[:, :, token]
        previous_value_imag = value_imag[:, :, token]
        output_real.append(current_real)
        output_imag.append(current_imag)

    cache = SISOScanCache(
        h_real=current_real,
        h_imag=current_imag,
        v_real=previous_value_real,
        v_imag=previous_value_imag,
    )
    if time:
        outputs = torch.stack(output_real, dim=2), torch.stack(output_imag, dim=2)
    else:
        outputs = value_real.clone(), value_imag.clone()
    if return_cache:
        return outputs[0], outputs[1], cache
    return outputs


class _TrapezoidalProposition2Reference(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dictionary_logits,
        selector_logits,
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
        from .routes import compact_hard_selection

        selection = compact_hard_selection(dictionary_logits, selector_logits)
        output = trapezoidal_reference_scan(
            selection.destination,
            selection.routes,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
            chunk_size=chunk_size,
            mode=mode,
        )
        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            selection.destination,
            selection.routes,
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
        return output

    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        (
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
        ) = ctx.saved_tensors
        batch, heads, time, state = diagonal_real.shape
        active_dictionary = torch.zeros(
            destination.shape,
            dtype=dictionary_logits.dtype,
            device=dictionary_logits.device,
        )
        selector_score = torch.empty(
            (batch, heads, time),
            dtype=selector_logits.dtype,
            device=selector_logits.device,
        )
        gradients = [
            torch.zeros_like(value)
            for value in (
                diagonal_real,
                diagonal_imag,
                value_real,
                value_imag,
                beta,
                gamma,
            )
        ]
        carry_real = torch.zeros_like(diagonal_real[:, :, 0])
        carry_imag = torch.zeros_like(diagonal_imag[:, :, 0])
        value_carry_real = torch.zeros_like(value_real[:, :, 0])
        value_carry_imag = torch.zeros_like(value_imag[:, :, 0])
        dictionary = destination.unsqueeze(0).expand(batch, -1, -1, -1)

        for token in range(time - 1, -1, -1):
            total_real = grad_output_real[:, :, token] + carry_real
            total_imag = grad_output_imag[:, :, token] + carry_imag
            route = routes[:, :, token].long()
            selected = (
                torch.gather(
                    dictionary,
                    2,
                    route[..., None, None].expand(-1, -1, 1, state),
                )
                .squeeze(2)
                .long()
            )
            destination_real = torch.gather(total_real, -1, selected)
            destination_imag = torch.gather(total_imag, -1, selected)
            if token:
                previous_real = output_real[:, :, token - 1]
                previous_imag = output_imag[:, :, token - 1]
                previous_value_real = value_real[:, :, token - 1]
                previous_value_imag = value_imag[:, :, token - 1]
            else:
                previous_real = torch.zeros_like(total_real)
                previous_imag = torch.zeros_like(total_imag)
                previous_value_real = torch.zeros_like(total_real)
                previous_value_imag = torch.zeros_like(total_imag)
            token_beta = beta[:, :, token, None]
            transition_input_real = previous_real + token_beta * previous_value_real
            transition_input_imag = previous_imag + token_beta * previous_value_imag
            token_diagonal_real = diagonal_real[:, :, token]
            token_diagonal_imag = diagonal_imag[:, :, token]
            transformed_input_real = (
                token_diagonal_real * transition_input_real
                - token_diagonal_imag * transition_input_imag
            )
            transformed_input_imag = (
                token_diagonal_real * transition_input_imag
                + token_diagonal_imag * transition_input_real
            )
            active = (
                destination_real * transformed_input_real
                + destination_imag * transformed_input_imag
            )
            route_one_hot = torch.nn.functional.one_hot(route, num_classes=destination.shape[1]).to(
                active.dtype
            )
            active_dictionary.add_(torch.einsum("bhk,bhn->hkn", route_one_hot, active))
            selector_score[:, :, token] = active.sum(-1)
            gradients[0][:, :, token] = (
                destination_real * transition_input_real + destination_imag * transition_input_imag
            )
            gradients[1][:, :, token] = (
                -destination_real * transition_input_imag + destination_imag * transition_input_real
            )
            gradients[2][:, :, token] = gamma[:, :, token, None] * total_real + value_carry_real
            gradients[3][:, :, token] = gamma[:, :, token, None] * total_imag + value_carry_imag
            gradients[5][:, :, token] = (
                total_real * value_real[:, :, token] + total_imag * value_imag[:, :, token]
            ).sum(-1)
            carry_real = (
                destination_real * token_diagonal_real + destination_imag * token_diagonal_imag
            )
            carry_imag = (
                -destination_real * token_diagonal_imag + destination_imag * token_diagonal_real
            )
            gradients[4][:, :, token] = (
                carry_real * previous_value_real + carry_imag * previous_value_imag
            ).sum(-1)
            value_carry_real = token_beta * carry_real
            value_carry_imag = token_beta * carry_imag

        dictionary_gradient = _active_hardmax_surrogate_gradient(
            dictionary_logits,
            destination.long(),
            active_dictionary,
            dim=-2,
            temperature=ctx.dictionary_temperature,
        )
        selector_gradient = _active_hardmax_surrogate_gradient(
            selector_logits,
            routes.permute(0, 2, 1).long(),
            selector_score.permute(0, 2, 1),
            dim=-1,
            temperature=ctx.router_temperature,
        )
        return (
            dictionary_gradient,
            selector_gradient,
            *gradients,
            None,
            None,
            None,
            None,
        )


def trapezoidal_proposition2_reference_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
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
    """Apply the hard trapezoidal recurrence with Proposition-2 selection gradients."""
    return _TrapezoidalProposition2Reference.apply(
        dictionary_logits,
        selector_logits,
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
    )


class _PaperSurrogateReference(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dictionary_logits,
        selector_logits,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        temperature,
        mode,
    ):
        from .routes import compact_hard_selection

        selection = compact_hard_selection(dictionary_logits, selector_logits)
        output = reference_scan(
            selection.destination,
            selection.routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            mode=mode,
        )
        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            selection.destination,
            selection.routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            *output,
        )
        ctx.temperature = temperature
        return output

    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        (
            dictionary_logits,
            selector_logits,
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            output_real,
            output_imag,
        ) = ctx.saved_tensors
        batch, heads, time, state = diagonal_real.shape
        active_dictionary = torch.zeros(
            destination.shape, dtype=dictionary_logits.dtype, device=dictionary_logits.device
        )
        selector_score = torch.empty(
            (batch, heads, time), dtype=selector_logits.dtype, device=selector_logits.device
        )
        gradients = [
            torch.empty_like(value)
            for value in (diagonal_real, diagonal_imag, bias_real, bias_imag)
        ]
        carry_real = torch.zeros_like(diagonal_real[:, :, 0])
        carry_imag = torch.zeros_like(diagonal_imag[:, :, 0])
        dictionary = destination.unsqueeze(0).expand(batch, -1, -1, -1)

        for token in range(time - 1, -1, -1):
            total_real = grad_output_real[:, :, token] + carry_real
            total_imag = grad_output_imag[:, :, token] + carry_imag
            gradients[2][:, :, token] = total_real
            gradients[3][:, :, token] = total_imag
            route = routes[:, :, token].long()
            selected = (
                torch.gather(
                    dictionary,
                    2,
                    route[..., None, None].expand(-1, -1, 1, state),
                )
                .squeeze(2)
                .long()
            )
            destination_real = torch.gather(total_real, -1, selected)
            destination_imag = torch.gather(total_imag, -1, selected)
            if token:
                previous_real = output_real[:, :, token - 1]
                previous_imag = output_imag[:, :, token - 1]
            else:
                previous_real = torch.zeros_like(total_real)
                previous_imag = torch.zeros_like(total_imag)
            token_diagonal_real = diagonal_real[:, :, token]
            token_diagonal_imag = diagonal_imag[:, :, token]
            value_real = token_diagonal_real * previous_real - token_diagonal_imag * previous_imag
            value_imag = token_diagonal_real * previous_imag + token_diagonal_imag * previous_real
            active = destination_real * value_real + destination_imag * value_imag
            route_one_hot = torch.nn.functional.one_hot(route, num_classes=destination.shape[1]).to(
                active.dtype
            )
            active_dictionary.add_(torch.einsum("bhk,bhn->hkn", route_one_hot, active))
            selector_score[:, :, token] = (
                total_real * (output_real[:, :, token] - bias_real[:, :, token])
                + total_imag * (output_imag[:, :, token] - bias_imag[:, :, token])
            ).sum(-1)
            gradients[0][:, :, token] = (
                destination_real * previous_real + destination_imag * previous_imag
            )
            gradients[1][:, :, token] = (
                -destination_real * previous_imag + destination_imag * previous_real
            )
            carry_real = (
                destination_real * token_diagonal_real + destination_imag * token_diagonal_imag
            )
            carry_imag = (
                -destination_real * token_diagonal_imag + destination_imag * token_diagonal_real
            )

        dictionary_gradient = _active_hardmax_surrogate_gradient(
            dictionary_logits,
            destination.long(),
            active_dictionary,
            dim=-2,
            temperature=ctx.temperature,
        )
        selector_gradient = _active_hardmax_surrogate_gradient(
            selector_logits,
            routes.permute(0, 2, 1).long(),
            selector_score.permute(0, 2, 1),
            dim=-1,
            temperature=ctx.temperature,
        )
        return (
            dictionary_gradient,
            selector_gradient,
            *gradients,
            None,
            None,
        )


def paper_surrogate_reference_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    temperature: float,
    mode: NativePDMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal Equations 9--12 diagnostic oracle."""
    return _PaperSurrogateReference.apply(
        dictionary_logits,
        selector_logits,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        temperature,
        mode,
    )
