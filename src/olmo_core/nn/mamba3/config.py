import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Optional

from olmo_core.config import DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from ..attention import AttentionBackendName, AttentionConfig, AttentionType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig
from ..rope import RoPEConfig, RoPEType
from ..transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from .mixer import DEFAULT_D_STATE, Mamba3MixerConfig

if TYPE_CHECKING:
    from ..transformer.block import TransformerBlockBase
    from .model import Mamba3

log = logging.getLogger(__name__)

__all__ = ["Mamba3Type", "Mamba3BlockType", "Mamba3BlockConfig", "Mamba3Config"]


class Mamba3Type(StrEnum):
    """An enumeration of Mamba-3 model implementations."""

    hybrid = "hybrid"
    """
    ➡️ :class:`~olmo_core.nn.mamba3.model.Mamba3` - an attention + Mamba-3 hybrid.
    """


class Mamba3BlockType(StrEnum):
    """
    An enumeration of Mamba-3 hybrid block implementations.

    These share string values with :class:`~olmo_core.nn.transformer.config.TransformerBlockType`
    so the two are interchangeable at the config level.
    """

    default = "default"
    """➡️ :class:`~olmo_core.nn.mamba3.block.Mamba3Block`"""

    reordered_norm = "reordered_norm"
    """➡️ :class:`~olmo_core.nn.mamba3.block.ReorderedNormMamba3Block`"""


@dataclass
class Mamba3BlockConfig(TransformerBlockConfig):
    """
    A block config for the Mamba-3 hybrid. Identical to
    :class:`~olmo_core.nn.transformer.config.TransformerBlockConfig` except that
    :meth:`build` returns the :mod:`olmo_core.nn.mamba3.block` subclasses for the ``default`` and
    ``reordered_norm`` types (any other type falls back to the transformer blocks).
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        from .block import Mamba3Block, ReorderedNormMamba3Block

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormMamba3Block(**kwargs)
            elif self.name == TransformerBlockType.default:
                return Mamba3Block(**kwargs)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e

        # Fall back to the transformer block types for anything else.
        return super().build(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )


@dataclass
class Mamba3Config(TransformerConfig):
    """
    A config for building :class:`~olmo_core.nn.mamba3.model.Mamba3` hybrid models, mirroring
    :class:`~olmo_core.nn.transformer.config.TransformerConfig`.

    The 1:3 attention-to-Mamba-3 hybrid is assembled with named blocks + ``block_pattern`` (the
    same mechanism ``TransformerConfig.qwen3_5_like`` uses for the GatedDeltaNet + attention
    hybrid). Use :meth:`mamba3_hybrid_like` or the size presets to build one.
    """

    name: Mamba3Type = Mamba3Type.hybrid  # type: ignore[assignment]
    """The Mamba-3 implementation."""

    def build(self, *, init_device: str = "cpu") -> "Mamba3":
        """
        Build the :class:`~olmo_core.nn.mamba3.model.Mamba3` model corresponding to this config.

        :param init_device: The device to put parameters on during initialization (e.g. "meta").
        """
        from .model import Mamba3

        log.info(
            f"Building Mamba-3 hybrid with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )

        model = Mamba3(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            n_layers=self.n_layers,
            block=self.block,
            embedding_norm=self.embedding_norm,
            lm_head=self.lm_head,
            dtype=self.dtype.as_pt(),
            init_method=self.init_method,
            init_device=init_device,
            init_seed=self.init_seed,
            init_std=self.init_std,
            embedding_init_std=self.embedding_init_std,
            block_overrides=self.block_overrides,
            block_pattern=self.block_pattern,
            embed_scale=self.embed_scale,
            tie_word_embeddings=self.tie_word_embeddings,
        )

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        return model

    @classmethod
    def mamba3_hybrid_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        intermediate_size: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        mamba_n_heads: Optional[int] = None,
        mamba_head_dim: Optional[int] = None,
        d_state: int = DEFAULT_D_STATE,
        n_groups: int = 1,
        mimo_rank: int = 4,
        rotation_block_size: int = 2,
        exempt_timescale_params_from_weight_decay: bool = True,
        a_log_init_min: float = 0.05,
        a_log_init_max: float = 16.0,
        prefer_official_kernel: Optional[bool] = None,
        rotation_scan_impl: Optional[str] = None,
        theta_max: Optional[float] = None,
        block_pattern: Optional[list[str]] = None,
        block_name: Mamba3BlockType = Mamba3BlockType.reordered_norm,
        use_rope: bool = False,
        rope_theta: int = 500_000,
        layer_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        attn_backend: Optional[AttentionBackendName] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "Mamba3Config":
        """
        Build a hybrid config interleaving attention and Mamba-3 layers.

        The default ``block_pattern`` is ``["mamba3", "mamba3", "mamba3", "attn"]`` - a 1:3
        attention-to-Mamba-3 ratio (so ``n_layers`` should be divisible by 4). Attention layers
        default to **NoPE** (no position embeddings), matching Nemotron-H / Jamba, since the
        Mamba-3 layers supply positional information; set ``use_rope=True`` for RoPE (Bamba-style).

        :param d_model: Model hidden size.
        :param vocab_size: Vocabulary size.
        :param n_layers: Total number of layers.
        :param n_heads: Number of attention heads (for the attention layers).
        :param intermediate_size: Feed-forward hidden size.
        :param n_kv_heads: Number of key/value heads for attention (defaults to ``n_heads``).
        :param head_dim: Attention head dimension (defaults to ``d_model // n_heads``).
        :param mamba_n_heads: Number of Mamba-3 SSM heads (defaults to ``n_heads``).
        :param mamba_head_dim: Mamba-3 per-head dim (defaults to ``d_model // mamba_n_heads``).
        :param d_state: Mamba-3 SSM state size. Must be divisible by ``rotation_block_size``.
        :param n_groups: Mamba-3 ``(B, C)`` groups. Note the default of 1 shares a single
            rotation schedule across every head, which limits state tracking; set it to
            ``mamba_n_heads`` to give each head its own.
        :param mimo_rank: Mamba-3 MIMO rank (``1`` == SISO). MIMO widens the read/write rank but
            leaves the transition monoid untouched, so it buys no state-tracking power while
            still scaling the cost of applying the rotation.
        :param rotation_block_size: Size ``b`` of the orthogonal transition blocks (``2`` keeps
            the paper's abelian complex diagonal; ``b >= 3`` is non-solvable). Must be one of
            :func:`~olmo_core.nn.mamba3.admissible_block_sizes` for the chosen ``d_state``.
        :param exempt_timescale_params_from_weight_decay: Tag ``A_log`` and ``dt_bias`` so an
            optimizer group can exclude them from weight decay; see
            :attr:`~olmo_core.nn.mamba3.mixer.Mamba3MixerConfig.exempt_timescale_params_from_weight_decay`.
        :param a_log_init_min: Lower bound of the ``A_log`` init distribution, i.e. the floor on
            the decay rate. Must be ``> 0``. The default of 0.05 is intentionally below
            ``mamba_ssm``'s ``A_init_range`` lower bound of 1.0; see
            :attr:`~olmo_core.nn.mamba3.mixer.Mamba3MixerConfig.a_log_init_min`.
        :param a_log_init_max: Upper bound of the ``A_log`` init distribution.
        :param rotation_scan_impl: Which of
            :data:`~olmo_core.nn.mamba3.mamba3_ssd_fast.ROTATION_SCAN_IMPLS` computes the
            ``b >= 3`` prefix product, or ``None`` (the default) to defer to
            ``MAMBA3_ROTATION_SCAN_IMPL``. Setting it here is what records the choice in the
            saved config; left in the environment alone it is invisible to the checkpoint and a
            resume that loses the export silently drops to ``chunked``.
        :param theta_max: Bound on the per-step rotation angle, applied at every
            ``rotation_block_size`` (see
            :attr:`~olmo_core.nn.mamba3.mixer.Mamba3MixerConfig.theta_max`). ``None`` leaves it
            unbounded. Set to about ``1/sqrt(sequence_length)`` so the random walk's mixing time
            stays past the sequence length.
        :param block_pattern: Override the repeating block pattern. Pass ``["mamba3"]`` for a
            pure Mamba-3 stack; the default hybrid's attention layers can memorize short
            sequences and confound state-tracking evaluations.
        :param block_name: The block type for both layer kinds.
        :param use_rope: Whether attention layers use RoPE (default ``False`` = NoPE).
        :param rope_theta: RoPE theta (only used when ``use_rope=True``).
        :param layer_norm_eps: Layer norm epsilon.
        :param qk_norm: Whether attention layers use QK norm.
        :param attn_backend: Backend for the attention layers. Defaults to ``None``, which lets
            :class:`~olmo_core.nn.attention.AttentionConfig` select one.
        :param dtype: Default parameter dtype.
        """
        mamba_n_heads = mamba_n_heads if mamba_n_heads is not None else n_heads
        block_pattern = block_pattern or ["mamba3", "mamba3", "mamba3", "attn"]

        layer_norm = LayerNormConfig(
            name=LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        mamba_block = Mamba3BlockConfig(
            name=block_name,  # type: ignore[arg-type]
            sequence_mixer=Mamba3MixerConfig(
                n_heads=mamba_n_heads,
                head_dim=mamba_head_dim,
                d_state=d_state,
                n_groups=n_groups,
                mimo_rank=mimo_rank,
                rotation_block_size=rotation_block_size,
                norm_eps=layer_norm_eps,
                exempt_timescale_params_from_weight_decay=exempt_timescale_params_from_weight_decay,
                a_log_init_min=a_log_init_min,
                a_log_init_max=a_log_init_max,
                prefer_official_kernel=prefer_official_kernel,
                rotation_scan_impl=rotation_scan_impl,
                theta_max=theta_max,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        attn_block = Mamba3BlockConfig(
            name=block_name,  # type: ignore[arg-type]
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=(RoPEConfig(name=RoPEType.default, theta=rope_theta) if use_rope else None),
                qk_norm=layer_norm if qk_norm else None,
                backend=attn_backend,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block={"mamba3": mamba_block, "attn": attn_block},
            block_pattern=block_pattern,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def mamba3_hybrid_190M(
        cls, vocab_size: int, *, d_state: int = DEFAULT_D_STATE, **kwargs
    ) -> "Mamba3Config":
        """
        A small ~190M hybrid config (useful for tests / smoke runs). 12 layers = 9 Mamba-3 + 3
        attention.

        :param d_state: Mamba-3 SSM state size. Overridable because it has to be:
            :data:`~olmo_core.nn.mamba3.DEFAULT_D_STATE` cannot express
            ``rotation_block_size=3``. See :func:`~olmo_core.nn.mamba3.admissible_block_sizes`.
        """
        return cls.mamba3_hybrid_like(
            d_model=768,
            vocab_size=vocab_size,
            n_layers=12,
            n_heads=12,
            intermediate_size=3072,
            mamba_n_heads=12,
            d_state=d_state,
            **kwargs,
        )

    @classmethod
    def mamba3_hybrid_1B(
        cls, vocab_size: int, *, d_state: int = DEFAULT_D_STATE, **kwargs
    ) -> "Mamba3Config":
        """
        A ~1B hybrid config. 16 layers = 12 Mamba-3 + 4 attention.

        :param d_state: Mamba-3 SSM state size. Overridable because it has to be:
            :data:`~olmo_core.nn.mamba3.DEFAULT_D_STATE` cannot express
            ``rotation_block_size=3``. See :func:`~olmo_core.nn.mamba3.admissible_block_sizes`.
        """
        return cls.mamba3_hybrid_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=16,
            n_heads=16,
            intermediate_size=8192,
            mamba_n_heads=16,
            d_state=d_state,
            **kwargs,
        )

    @classmethod
    def mamba3_olmo3_370M(
        cls, vocab_size: int, *, d_state: int = DEFAULT_D_STATE, **kwargs
    ) -> "Mamba3Config":
        """
        OLMo-3-370M with Mamba-3 in place of the sliding-window attention layers.

        A parameter-matched ablation against :meth:`TransformerConfig.olmo3_370M
        <olmo_core.nn.transformer.TransformerConfig.olmo3_370M>`. OLMo-3's attention pattern is
        ``[4096, 4096, 4096, -1]`` -- three sliding-window layers, then one full-attention layer
        -- and this preset's ``["mamba3", "mamba3", "mamba3", "attn"]`` has the same 3:1 shape.
        The substitution is therefore exactly "replace the sub-quadratic layer", and the
        surviving attention layers are the ones OLMo-3 also leaves global. Nothing here carries
        a sliding window: windowing the full-attention layers would change the very layers the
        ablation holds fixed.

        Width, depth, head count, feed-forward size, RoPE (theta 500k), QK-norm and the 1e-6
        norm epsilon all follow the reference. ``use_rope=True`` overrides the hybrid default of
        NoPE: the Mamba-3 layers do carry position, but leaving RoPE off would be a second
        difference from the reference and confound the comparison.

        **This is the TC^0 baseline.** ``rotation_block_size`` defaults to 2, whose cumulative
        rotation is a cumulative *sum* of angles -- the abelian ``SO(2)`` case. It is set up so
        that an NC^1 arm differs from this run in exactly one config field: SISO
        (``mimo_rank=1``) with a single ``(B, C)`` group, so nothing but the block size moves.

        Parameters land at 363.0M active non-embedding against the reference's 371.3M, i.e.
        **-2.2%**. That gap is the price of the clean switch, and it is entirely
        ``mimo_rank``: rank only widens ``in_B``/``in_C`` via
        ``bc_out = n_groups * mimo_rank * d_state``, so 4 -> 1 drops 787k parameters per Mamba
        layer, 9.45M across the twelve. ``n_groups=4`` would restore the match to +0.95% and
        cost nothing on the scan, but it adds a second axis of difference to every later
        comparison. Match on this metric, not the "370M" label: both models leave the LM head
        untied, so the reference is 474M total and only ``num_active_non_embedding_params``
        recovers the 371M the name refers to.

        ``A_log`` uses the library default range ``(0.05, 16)``. An earlier version of this
        preset lowered ``a_log_init_max`` to 0.1 on the reasoning that 16 gives an 11-44 token
        decay horizon against a 4096-token window. That reasoning ignores ``dt in [0.001, 0.1]``,
        which multiplies ``A`` in ``alpha = exp(dt * A)`` and spreads the realized horizon across
        three orders of magnitude. Trained end to end at 4.8B tokens, 0.1 put every head above a
        1000-token horizon and the model plateaued near CE 8.1, behaving as a document-mean
        accumulator; the same config with an upper bound of 16 trained normally to CE 2.67-2.83.
        Do not lower the upper bound again without re-checking that plateau.

        The 0.05 lower bound is deliberately below ``mamba_ssm``'s 1.0. Inspecting the trained
        checkpoints of both 4.8B arms showed the only long-horizon heads either one had were
        inherited from the old ``Uniform(0, 16)`` init's accidental sub-1.0 tail (b=3's longest
        was ``|A| = 0.062``, a 3048-token horizon); the bulk of the distribution barely moved,
        so training did not manufacture them.

        SISO also makes the run eligible for the official SISO Triton kernel, roughly 3x
        cheaper on the scan than the chunked PyTorch path that ``mimo_rank > 1`` forces.

        :param d_state: Mamba-3 SSM state size. The default admits ``b`` in ``{2, 3, 4}``
            (:func:`~olmo_core.nn.mamba3.admissible_block_sizes`), which is what lets the TC^0
            baseline and the NC^1 arm share one state size so ``rotation_block_size`` is the only
            field that differs between them.

            The intended pairing is ``b=2`` against ``b=3``. ``b=4`` is expressible but is not a
            useful third arm: it is NC^1-hard by exactly the same argument as ``b=3``
            (``A_5 subset SO(3) subseteq SO(b)``), so it buys no additional hardness, while
            costing ~6x more in the rotation because ``SO(4)`` has no closed-form exponential
            here and falls back to ``matrix_exp``. It has also been observed to be sensitive to
            learning rate and seed on the ``A_5`` task where ``b=3`` was not.
        """
        return cls.mamba3_hybrid_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=16,
            n_heads=16,
            intermediate_size=4096,
            mamba_n_heads=16,
            d_state=d_state,
            mimo_rank=kwargs.pop("mimo_rank", 1),
            n_groups=kwargs.pop("n_groups", 1),
            a_log_init_min=kwargs.pop("a_log_init_min", 0.05),
            a_log_init_max=kwargs.pop("a_log_init_max", 16.0),
            use_rope=kwargs.pop("use_rope", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            **kwargs,
        )
