"""
Kernel dispatch and reference implementation for the Mamba-3 state-space mixer.

This mirrors the pattern in :mod:`olmo_core.nn.attention.flash_linear_attn_api`: a
:func:`has_mamba3` capability probe plus a :func:`dispatch_mamba3_ssd` entry point that
prefers a fast kernel when available and otherwise falls back to the in-repo
:func:`mamba3_ssd_reference` implementation.

The reference implements the three Mamba-3 innovations from
`Mamba-3: Improved Sequence Modeling using State Space Principles
<https://arxiv.org/abs/2603.15569>`_ (§3):

1. **Exponential-trapezoidal discretization** (§3.1): the recurrence
   ``h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t`` with
   ``α_t = exp(Δ_t A_t)``, ``β_t = (1-λ_t) Δ_t α_t``, ``γ_t = λ_t Δ_t``. This is a
   width-2 convolution on the state-input ``B_t x_t`` inside the recurrence.
2. **Complex-valued state via the RoPE trick** (§3.2): a data-dependent 2x2 block
   rotation applied as a cumulative product across timesteps to both ``B`` and ``C``.
   Generalized here to ``b x b`` orthogonal blocks via ``rotation_block_size``: ``b == 2``
   is the paper's complex diagonal, while ``b >= 3`` makes the transition monoid
   non-solvable (``SO(3)`` contains the icosahedral group ``A_5``), lifting the layer out
   of TC^0. The factorization needs only associativity and orthogonality, not the
   commutativity the paper's Prop. 3 assumes, so the scan below is shared by every ``b``.
3. **MIMO** (§3.3): rank-``R`` input/output projections. Here MIMO is realized as ``R``
   parallel rank-1 SSMs whose outputs are summed, which reduces exactly to SISO at
   ``R == 1``. This is a correctness-first reference; the official ``mamba-ssm`` kernel
   uses the state-size-preserving matmul form and is the fast path.
"""

from typing import Optional

import torch

__all__ = [
    "has_mamba3",
    "kernel_padded_width",
    "mamba3_ssd_reference",
    "dispatch_mamba3_ssd",
]


def kernel_padded_width(dim: int, *, min_width: int = 16) -> int:
    """
    The width the official ``mamba-ssm`` kernel will actually run ``dim`` at.

    ``mamba3_siso_combined`` needs a power-of-two head dimension for TMA, so
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_fast.mamba3_ssd_fast` zero-pads up to one. The padding
    is numerically exact -- a zero column of ``B`` never enters the state -- but it is *wasted
    work*, and it collides with :func:`~olmo_core.nn.mamba3.mixer.admissible_block_sizes`: no
    power of two is divisible by 3, so every ``b=3`` configuration pays some. ``d_state=192`` runs
    at 256, a quarter of the ``Q``/``K`` lanes carrying zeros.

    Only the official/fast path pads; the chunked and reference paths use ``dim`` as given.

    This is the single source of the padding rule. The fast and official adapters, and the
    mixer's diagnostics, all import it rather than reimplementing it, so the value the kernel
    pads to and the value the FLOP/waste accounting assumes cannot drift apart.

    :param dim: The logical width (``d_state`` or ``head_dim``).
    :param min_width: Floor imposed by the kernel's ``tl.dot`` contraction.
    """
    out = min_width
    while out < dim:
        out *= 2
    return out


def has_mamba3() -> bool:
    """
    Check if a Mamba-3 fast kernel is installed.

    Unlike a plain ``import mamba_ssm`` check, this probes the specific Mamba-3 module,
    since older ``mamba-ssm`` releases only ship Mamba-1/2. The Mamba-3 kernels currently
    live on ``mamba-ssm`` ``main`` and require a source build.
    """
    try:
        import mamba_ssm.modules.mamba3  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def _rotate_bc(bc: torch.Tensor, theta_cumulative: torch.Tensor) -> torch.Tensor:
    """
    Apply a data-dependent rotary embedding to the last (state) dimension of ``B`` or ``C``.

    :param bc: Tensor of shape ``(batch, seq_len, n_groups, rank, d_state)`` with ``d_state``
        even.
    :param theta_cumulative: Cumulative rotation angles of shape
        ``(batch, seq_len, n_groups, d_state // 2)``.
    """
    *lead, d_state = bc.shape
    half = d_state // 2
    bc_pairs = bc.reshape(*lead, half, 2)
    x1 = bc_pairs[..., 0]
    x2 = bc_pairs[..., 1]
    # theta_cumulative: (B, T, G, half) -> broadcast over the rank dim.
    cos = torch.cos(theta_cumulative).unsqueeze(-2)
    sin = torch.sin(theta_cumulative).unsqueeze(-2)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack((rot1, rot2), dim=-1).reshape(*lead, d_state)


def _skew_from_angles(theta: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Build skew-symmetric ``b x b`` generators from ``b*(b-1)//2`` angles per block.

    The angles fill the strict upper triangle in row-major order and are mirrored with the
    opposite sign into the lower triangle, so ``S = sum_{i<j} theta_ij (E_ij - E_ji)``.

    The sign convention is fixed by backward compatibility, not by taste. At ``block_size == 2``
    this gives ``S = [[0, theta], [-theta, 0]]`` and ``exp(S) = R(-theta)``; the cumulative
    product is then transposed before it is applied (:func:`_rotate_bc_blocks`), which restores
    ``R(+theta_cumulative)`` -- exactly what the legacy 2x2 path in :func:`_rotate_bc` applies.
    Flipping this sign would leave the model equally expressive, since ``theta`` is an
    unconstrained linear projection that would simply learn ``-theta``, but it would break the
    bit-exact ``b == 2`` regression gate.

    :param theta: Angles of shape ``(..., b*(b-1)//2)``.
    :param block_size: The rotation block size ``b``.

    :returns: Skew-symmetric matrices of shape ``(..., b, b)``.
    """
    *lead, n_angles = theta.shape
    expected = block_size * (block_size - 1) // 2
    if n_angles != expected:
        raise ValueError(
            f"expected {expected} angles for rotation_block_size={block_size}, got {n_angles}"
        )
    rows, cols = torch.triu_indices(block_size, block_size, offset=1, device=theta.device)
    skew = theta.new_zeros(*lead, block_size, block_size)
    skew[..., rows, cols] = theta
    return skew - skew.transpose(-1, -2)


def _block_rotations(theta: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Map per-step angles to per-step rotation matrices in ``SO(b)``.

    ``matrix_exp`` of a skew-symmetric matrix is *surjective* onto ``SO(b)``, so every element
    of ``A_5`` is representable. This is the whole point of the parameterization: the cheaper
    Cayley transform ``(I - S)(I + S)^-1`` misses every rotation with a ``-1`` eigenvalue, which
    silently excludes all 15 order-2 elements of the icosahedral group and would cap the layer
    below ``A_5`` with no visible symptom. See ``rotation_test.py`` for the guarding test.

    :param theta: Angles of shape ``(..., b*(b-1)//2)``.
    :param block_size: The rotation block size ``b``.

    :returns: Rotation matrices of shape ``(..., b, b)``.
    """
    return torch.matrix_exp(_skew_from_angles(theta, block_size))


def _cumulative_block_rotation(rot: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """
    Inclusive prefix product ``Q_t = R_t R_{t-1} ... R_1`` over the sequence axis.

    Iterated *non-abelian* group product is the operation that carries the NC^1-hardness, so
    this replaces the ``cumsum`` of the ``b == 2`` path (``R(a) R(b) = R(a+b)`` is precisely the
    abelian-ness of ``SO(2)`` written as code, and iterated addition is in TC^0).

    The scan is chunked rather than a flat Hillis-Steele: a flat scan stores all
    ``ceil(log2(T))`` intermediates for the backward pass, which at ``T=2048`` costs hundreds of
    MB per layer and OOMs a deep stack. Running the product sequentially within a chunk and
    Hillis-Steele only across the ``T/chunk_size`` chunk boundaries cuts the stored levels from
    ``log2(T)`` to ``log2(T/chunk_size)`` at the cost of ``chunk_size`` sequential steps.

    :param rot: Per-step rotations of shape ``(batch, seq_len, n_groups, n_blocks, b, b)``.
    :param chunk_size: Sequential-product chunk length.

    :returns: Inclusive prefix products, same shape as ``rot``.
    """
    batch, seq_len, n_groups, n_blocks, block_size, _ = rot.shape
    if seq_len == 1:
        return rot

    def identity(length: int) -> torch.Tensor:
        return torch.eye(block_size, dtype=rot.dtype, device=rot.device).expand(
            batch, length, n_groups, n_blocks, block_size, block_size
        )

    chunk = min(chunk_size, seq_len)
    n_chunks = (seq_len + chunk - 1) // chunk
    padded = n_chunks * chunk
    if padded != seq_len:
        # Pad with identities past the end; those positions are sliced off before returning.
        rot = torch.cat([rot, identity(padded - seq_len)], dim=1)
    chunked = rot.view(batch, n_chunks, chunk, n_groups, n_blocks, block_size, block_size)

    # Sequential inclusive prefix product within each chunk, newest-left.
    running = [chunked[:, :, 0]]
    for i in range(1, chunk):
        running.append(chunked[:, :, i] @ running[-1])
    local = torch.stack(running, dim=2)

    if n_chunks > 1:
        # Hillis-Steele inclusive scan over the per-chunk totals, then shift right by one to
        # turn it into the exclusive carry-in for each chunk.
        totals = local[:, :, -1]
        step = 1
        while step < n_chunks:
            totals = totals @ torch.cat([identity(step), totals[:, :-step]], dim=1)
            step *= 2
        carry = torch.cat([identity(1), totals[:, :-1]], dim=1)
        local = local @ carry.unsqueeze(2)

    return local.reshape(batch, padded, n_groups, n_blocks, block_size, block_size)[:, :seq_len]


def _rotate_bc_blocks(bc: torch.Tensor, cumulative_rot: torch.Tensor) -> torch.Tensor:
    """
    Apply the transposed cumulative block rotation ``Q_t^T`` to ``B`` or ``C``.

    Transposed, not plain: the readout contracts ``C~_t`` against ``B~_s``, and
    ``(Q_t^T C_t)^T (Q_s^T B_s) = C_t^T (Q_t Q_s^-1) B_s`` recovers the accumulated forward
    transition ``R_t R_{t-1} ... R_{s+1}``. Applying ``Q`` instead would run the rotation
    backwards. This is the step that keeps the scan itself unchanged (§3.1.4 of the design
    note): once ``B`` and ``C`` are pre-rotated the recurrence is a plain scalar-transition SSM.

    :param bc: Tensor of shape ``(batch, seq_len, n_groups, rank, d_state)``.
    :param cumulative_rot: Prefix products of shape
        ``(batch, seq_len, n_groups, n_blocks, b, b)``.
    """
    *lead, d_state = bc.shape
    block_size = cumulative_rot.shape[-1]
    bc_blocks = bc.reshape(*lead, d_state // block_size, block_size)
    rotated = torch.einsum("btgkji,btgrkj->btgrki", cumulative_rot, bc_blocks)
    return rotated.reshape(*lead, d_state)


def mamba3_ssd_reference(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int = 2,
) -> torch.Tensor:
    """
    Correctness-first, autograd-friendly reference for the Mamba-3 exponential-trapezoidal,
    complex (RoPE-trick), rank-R MIMO SSD recurrence.

    :param x: SSM input ("values"), shape ``(batch, seq_len, n_heads, head_dim)``.
    :param B: Input projection, shape ``(batch, seq_len, n_groups, rank, d_state)``.
    :param C: Output projection, shape ``(batch, seq_len, n_groups, rank, d_state)``.
    :param dt: Discretization step ``Δ_t`` (already positive), shape ``(batch, seq_len, n_heads)``.
    :param A: State-transition log-decay term (negative), shape ``(n_heads,)``.
    :param lam: Trapezoidal mixing coefficient ``λ_t`` in ``(0, 1)``, shape
        ``(batch, seq_len, n_heads)``.
    :param theta: Per-step rotation angles (pre-accumulation), shape
        ``(batch, seq_len, n_groups, d_state // block_size, block_size*(block_size-1)//2)``.
        At ``block_size == 2`` the legacy 4-D shape ``(batch, seq_len, n_groups, d_state // 2)``
        is also accepted.
    :param heads_per_group: Number of heads that share each ``(B, C)`` group.
    :param block_size: Rotation block size ``b``. ``2`` is the paper's complex diagonal;
        ``b >= 3`` gives a non-abelian ``SO(b)`` transition monoid.

    :returns: The SSM output, shape ``(batch, seq_len, n_heads, head_dim)``.
    """
    batch, seq_len, n_heads, head_dim = x.shape
    n_groups = B.shape[2]

    # Run the scan in float32 for numerical stability regardless of input dtype.
    orig_dtype = x.dtype
    x = x.float()
    B = B.float()
    C = C.float()
    dt = dt.float()
    A = A.float()
    lam = lam.float()
    theta = theta.float()

    # Data-dependent RoPE trick: cumulative rotation applied to both B and C (§3.2).
    if block_size == 2:
        # Abelian fast path: SO(2) prefix products collapse to a cumsum of angles. Kept
        # separate so the default configuration stays bit-identical to the pre-blocked code.
        theta_cumulative = torch.cumsum(theta.squeeze(-1) if theta.dim() == 5 else theta, dim=1)
        B = _rotate_bc(B, theta_cumulative)
        C = _rotate_bc(C, theta_cumulative)
    else:
        if theta.dim() != 5:
            raise ValueError(
                f"theta must be 5-D (batch, seq_len, n_groups, n_blocks, angles_per_block) "
                f"for block_size={block_size}, got shape {tuple(theta.shape)}"
            )
        cumulative_rot = _cumulative_block_rotation(_block_rotations(theta, block_size))
        B = _rotate_bc_blocks(B, cumulative_rot)
        C = _rotate_bc_blocks(C, cumulative_rot)

    # Broadcast groups to heads: (B, T, G, R, N) -> (B, T, H, R, N).
    if heads_per_group != 1:
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
    assert B.shape[2] == n_heads, (n_groups, heads_per_group, n_heads)

    # Exponential-trapezoidal coefficients (§3.1).
    alpha = torch.exp(dt * A)  # (B, T, H), in (0, 1) since A < 0, dt > 0
    gamma = lam * dt  # (B, T, H)
    beta = (1.0 - lam) * dt * alpha  # (B, T, H)

    rank = B.shape[3]
    d_state = B.shape[4]

    # State: (batch, n_heads, rank, d_state, head_dim).
    h = x.new_zeros((batch, n_heads, rank, d_state, head_dim))
    v_prev = x.new_zeros((batch, n_heads, rank, d_state, head_dim))
    outputs = []
    for t in range(seq_len):
        # State-input outer product v_t = B_t ⊗ x_t: (batch, H, R, N, P).
        v_t = B[:, t].unsqueeze(-1) * x[:, t].unsqueeze(2).unsqueeze(2)
        # Width-2 convolution on the state-input (trapezoidal), then decay.
        a_t = alpha[:, t].view(batch, n_heads, 1, 1, 1)
        g_t = gamma[:, t].view(batch, n_heads, 1, 1, 1)
        b_t = beta[:, t].view(batch, n_heads, 1, 1, 1)
        w_t = g_t * v_t + b_t * v_prev
        h = a_t * h + w_t
        # Read out: y_t = sum over (rank, d_state) of C_t * h.
        y_t = (C[:, t].unsqueeze(-1) * h).sum(dim=(2, 3))  # (batch, H, P)
        outputs.append(y_t)
        v_prev = v_t

    y = torch.stack(outputs, dim=1)  # (batch, T, H, P)
    return y.to(orig_dtype)


def _reduced_precision_requested(x: torch.Tensor) -> bool:
    """Whether the caller has already accepted bf16/fp16 for this op."""
    if x.dtype in (torch.bfloat16, torch.float16):
        return True
    device_type = x.device.type
    return torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) in (
        torch.bfloat16,
        torch.float16,
    )


def _official_kernel_eligible(x: torch.Tensor, B: torch.Tensor) -> bool:
    """Whether the official Triton kernel can run this call at all."""
    if not x.is_cuda or B.shape[3] != 1:  # CUDA-only, and SISO-only (mimo_rank == 1)
        return False
    # Imported here rather than at module scope: the official adapter imports the rotation
    # helpers from this one, so a top-level import would be circular.
    from .mamba3_ssd_official import official_mamba3_is_available

    return official_mamba3_is_available()


def dispatch_mamba3_ssd(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int = 2,
    prefer_fast_kernel: bool = True,
    chunk_size: int = 256,
    prefer_official_kernel: Optional[bool] = None,
    prefer_fast_rotation: bool = True,
) -> torch.Tensor:
    """
    Run the Mamba-3 SSD recurrence, preferring the fastest form that fits the call.

    :func:`~olmo_core.nn.mamba3.mamba3_ssd_chunked.mamba3_ssd_chunked` computes the same
    function as :func:`mamba3_ssd_reference` but as a handful of matmuls instead of one
    dependent step per token, which measures 15-23x faster at 7x less activation memory. The
    sequential reference remains reachable via ``prefer_fast_kernel=False`` and is the
    numerical oracle both fast forms are tested against.

    On CUDA with ``mimo_rank == 1`` and the official ``mamba-ssm`` Mamba-3 kernels installed,
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_official.mamba3_ssd_official` takes over. It works
    at *every* ``block_size``, not just the paper's ``b == 2``: the rotation is applied to
    ``B``/``C`` before the kernel is called and zero ``Angles`` make the kernel's own internal
    rotation the identity, so the non-abelian ``SO(b >= 3)`` prefix product rides along
    unchanged and the kernel only ever sees the per-head scalar decay it was written for.

    :param prefer_fast_kernel: Use one of the fast forms. Set ``False`` to force the
        sequential reference, e.g. when validating numerics.
    :param chunk_size: Chunk length for the chunked form; ignored by the other two.
    :param prefer_official_kernel: ``True``/``False`` force or forbid the official kernel;
        ``None`` (the default) arms it only when the caller has already accepted reduced
        precision, since ``mamba3_siso_combined`` hard-casts to bf16 internally and would
        otherwise silently throw away accuracy the chunked form still delivers.
    :param prefer_fast_rotation: Route the official-kernel path through
        :func:`~olmo_core.nn.mamba3.mamba3_ssd_fast.mamba3_ssd_fast`, which calls the same
        upstream kernel but computes the ``SO(b)`` rotation with a closed form at ``b == 3`` and
        a shorter prefix-product scan. Set ``False`` to reach ``mamba3_ssd_official`` unchanged,
        which is what the parity tests compare against.
    """
    if not prefer_fast_kernel:
        return mamba3_ssd_reference(
            x, B, C, dt, A, lam, theta, heads_per_group=heads_per_group, block_size=block_size
        )

    if prefer_official_kernel and not _official_kernel_eligible(x, B):
        # An explicit request must not be silently downgraded. `mamba3_ssd_official` itself
        # raises when it cannot run, so swallowing the same condition one level up meant
        # `prefer_official_kernel=True` could quietly return a chunked result -- exactly the
        # failure mode where a benchmark or a parity test believes it exercised the kernel and
        # did not. `None` (the default) still falls through to the chunked path silently,
        # because that is a preference rather than a request.
        raise RuntimeError(
            "prefer_official_kernel=True but the official kernel cannot run this call: it "
            f"needs CUDA (got {x.device.type}), mimo_rank == 1 (got {B.shape[3]}), and an "
            "installed mamba-ssm Mamba-3 build. Pass prefer_official_kernel=None to allow the "
            "chunked fallback."
        )

    if prefer_official_kernel is not False and _official_kernel_eligible(x, B):
        if prefer_official_kernel or _reduced_precision_requested(x):
            if prefer_fast_rotation:
                from .mamba3_ssd_fast import mamba3_ssd_fast

                return mamba3_ssd_fast(
                    x,
                    B,
                    C,
                    dt,
                    A,
                    lam,
                    theta,
                    heads_per_group=heads_per_group,
                    block_size=block_size,
                )

            from .mamba3_ssd_official import mamba3_ssd_official

            return mamba3_ssd_official(
                x,
                B,
                C,
                dt,
                A,
                lam,
                theta,
                heads_per_group=heads_per_group,
                block_size=block_size,
            )

    # Imported here rather than at module scope: the chunked module imports the rotation
    # helpers from this one, so a top-level import would be circular.
    from .mamba3_ssd_chunked import mamba3_ssd_chunked

    return mamba3_ssd_chunked(
        x,
        B,
        C,
        dt,
        A,
        lam,
        theta,
        heads_per_group=heads_per_group,
        block_size=block_size,
        chunk_size=chunk_size,
    )


def _maybe_fast_kernel_available() -> Optional[str]:
    """Return an identifier for the fast kernel if importable, else ``None`` (diagnostics)."""
    return "mamba_ssm.modules.mamba3" if has_mamba3() else None
