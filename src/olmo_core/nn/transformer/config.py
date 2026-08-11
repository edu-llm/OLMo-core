import logging
import math
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from fnmatch import fnmatch
from itertools import cycle, islice
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple, cast

from olmo_core.config import UNSET, DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.utils import ensure_multiple_of

from ..attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GateConfig,
    GateGranularity,
    SlidingWindowAttentionConfig,
)
from ..attention.recurrent import GatedDeltaNetConfig
from ..buffer_cache import BufferCache
from ..config import ModelConfig, ModuleConfig
from ..feed_forward import ActivationFunction, FeedForwardConfig, FeedForwardType
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig, LMHeadType
from ..moe import MoEConfig, MoERouterConfig, MoEType
from ..rope import RoPEConfig, RoPEScalingConfig, RoPEType
from .init import InitMethod

if TYPE_CHECKING:
    from .block import TransformerBlockBase
    from .model import Transformer

log = logging.getLogger(__name__)


def _maple_uniform_attn_backend() -> AttentionBackendName:
    """
    Pick ONE attention backend for every layer of a Maple model.

    The Maple factory mixes sliding-window and global layers, and OLMo-core resolves an unset
    backend *per layer* from whether that layer has a window — flash_2 for the sliding ones,
    torch SDPA for the global ones (``nn/attention/__init__.py:454-462``). That would make the
    3:1 SWA layout a 3:1 kernel split: it biases MFU, which is the E-sweep's dependent
    variable, and torch SDPA additionally raises on intra-document masking.

    Preferring flash_2 unconditionally is not an option either — it is absent from the platform
    image and from FarmShare, and its ``assert_supported`` raises rather than degrading. So
    probe once and return a single backend that every layer can actually use.
    """
    from ..attention import flash_attn_api

    if flash_attn_api.has_flash_attn_2():
        return AttentionBackendName.flash_2
    log.info(
        "flash-attn 2 is unavailable, so all Maple layers will use the torch SDPA backend. "
        "This is uniform across sliding and global layers, which is what keeps MFU "
        "comparable; it is slower than flash_2 and does not support intra-document masking."
    )
    return AttentionBackendName.torch


class TransformerDataParallelWrappingStrategy(StrEnum):
    """
    An enumeration of the different wrapping strategy for the data parallel implementations.
    """

    full = "full"
    """
    Wrap each block and the LM head (only applies to FSDP).
    """

    blocks = "blocks"
    """
    Like full but the LM head is not wrapped separately (only applies to FSDP).
    """

    fine_grained = "fine_grained"
    """
    Wrap certain modules within each block in addition to wrapping each block (only applies to FSDP).
    """


@beta_feature
class TransformerActivationCheckpointingMode(StrEnum):
    """
    An enumeration of the different activation checkpointing modes.
    """

    full = "full"
    """Checkpoint every block."""
    selected_blocks = "selected_blocks"
    """Checkpoint only selected blocks."""
    selected_modules = "selected_modules"
    """Checkpoint only selected modules."""
    selected_ops = "selected_ops"
    """Checkpoint only a specific set of operations."""
    budget = "budget"
    """Checkpoint based on a budget."""


class TransformerType(StrEnum):
    """
    An enumeration of transformer implementations.
    """

    default = "default"
    """
    ➡️ :class:`Transformer`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformer` (nGPT)
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformer`
    """


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    ➡️ :class:`TransformerBlock`
    """

    default_scaled = "default_scaled"
    """
    ➡️ :class:`LayerNormScaledTransformerBlock` (applies LayerNorm Scaling)
    """

    reordered_norm = "reordered_norm"
    """
    ➡️ :class:`ReorderedNormTransformerBlock`
    """

    peri_norm = "peri_norm"
    """
    ➡️ :class:`PeriNormTransformerBlock`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformerBlock`
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe_reordered_norm"
    """
    ➡️ :class:`MoEReorderedNormTransformerBlock`
    """

    moe_hybrid = "moe_hybrid"
    """
    ➡️ :class:`MoEHybridTransformerBlock`
    """

    moe_hybrid_reordered_norm = "moe_hybrid_reordered_norm"
    """
    ➡️ :class:`MoEHybridReorderedNormTransformerBlock`
    """


@dataclass
class TransformerBlockConfig(ModuleConfig):
    """
    A configuration class for easily building transformer blocks.
    """

    sequence_mixer: SequenceMixerConfig = field(default=UNSET)
    """
    The sequence mixer config (e.g. attention, recurrent, convolution, etc.).
    """
    attention: InitVar[Optional[AttentionConfig]] = None
    """
    .. deprecated::
        Use :data:`sequence_mixer` instead. This field is only kept for backwards compatibility
        with old configs that used ``attention: AttentionConfig``.
    """
    layer_norm: Optional[LayerNormConfig] = None
    """
    The layer norm config.
    """
    feed_forward: Optional[FeedForwardConfig] = None
    """
    The feed-forward config, required for non-MoE blocks.
    """
    feed_forward_moe: Optional[MoEConfig] = None
    """
    The config for the MoE feed-forward layer. Required for MoE blocks.
    """
    name: TransformerBlockType = TransformerBlockType.default
    """
    The block type.
    """
    dropout: Optional[float] = None
    """
    Dropout probability.
    """
    attention_residual_alpha: Optional[float] = None
    """
    A scaling factor applied to the attention/recurrent output before adding it to the residual stream.
    """
    feed_forward_residual_alpha: Optional[float] = None
    """
    A scaling factor applied to the feed-forward (MLP) output before adding it to the residual stream.
    """

    def __post_init__(self, attention: Optional[AttentionConfig] = None):
        # Handle backwards compatibility: old configs used `attention` instead of `sequence_mixer`.
        if attention is not None:
            if self.sequence_mixer is not UNSET:
                raise OLMoConfigurationError(
                    "Cannot specify both 'attention' and 'sequence_mixer' in TransformerBlockConfig. "
                    "Use 'sequence_mixer' only (the 'attention' field is deprecated)."
                )
            self.sequence_mixer = attention
        if self.sequence_mixer is UNSET:
            raise OLMoConfigurationError(
                "TransformerBlockConfig requires 'sequence_mixer' to be set."
            )

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        from .block import (
            LayerNormScaledTransformerBlock,
            MoEHybridReorderedNormTransformerBlock,
            MoEHybridTransformerBlock,
            MoEReorderedNormTransformerBlock,
            MoETransformerBlock,
            NormalizedTransformerBlock,
            PeriNormTransformerBlock,
            ReorderedNormTransformerBlock,
            TransformerBlock,
        )

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
            if self.name == TransformerBlockType.default:
                return TransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.default_scaled:
                return LayerNormScaledTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.peri_norm:
                return PeriNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.normalized:
                return NormalizedTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe:
                return MoETransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_reordered_norm:
                return MoEReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid:
                return MoEHybridTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid_reordered_norm:
                return MoEHybridReorderedNormTransformerBlock(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e

    def num_params(self, d_model: int) -> int:
        block_params = 0

        # Block attn and MLP scaling factors.
        if self.name == TransformerBlockType.normalized:
            block_params += 2 * d_model

        # Block attention params.
        block_params += self.sequence_mixer.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)

        # Block feed forward (dense and/or sparse).
        if self.feed_forward is not None:
            block_params += self.feed_forward.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)
        if self.feed_forward_moe is not None:
            block_params += self.feed_forward_moe.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)

        # Two extra norms for Peri-LN block type.
        if self.name == TransformerBlockType.peri_norm:
            assert self.layer_norm is not None
            block_params += 2 * self.layer_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        num_params = self.num_params(d_model)
        if self.feed_forward_moe is None:
            return num_params

        num_inactive_params = self.feed_forward_moe.num_params(
            d_model
        ) - self.feed_forward_moe.num_active_params(d_model)
        return num_params - num_inactive_params


@dataclass
class TransformerConfig(ModelConfig):
    """
    A config for easily building transformer models.

    :param name: The name of the implementation.

    See :class:`Transformer` for a description of the other parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig]
    lm_head: LMHeadConfig
    embedding_norm: Optional[LayerNormConfig] = None
    name: TransformerType = TransformerType.default
    dtype: DType = DType.float32
    init_method: InitMethod = InitMethod.normal
    init_seed: int = 0
    init_std: float = 0.02
    embedding_init_std: Optional[float] = None
    freeze_params: Optional[List[str]] = None
    block_pattern: Optional[List[str]] = None
    block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None
    embed_scale: Optional[float] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.tie_word_embeddings and self.name == TransformerType.normalized:
            raise OLMoConfigurationError(
                "Tying word embeddings is not supported with the normalized transformer"
            )
        validate_block_resolution_config(
            n_layers=self.n_layers,
            block=self.block,
            block_pattern=self.block_pattern,
            block_overrides=self.block_overrides,
        )
        if self.block_pattern is not None and self.n_layers % len(self.block_pattern) != 0:
            log.warning(
                "`n_layers` (%d) is not divisible by the length of `block_pattern` (%d). "
                "The pattern will be cycled and truncated to fit `n_layers`, so the last "
                "cycle will be incomplete.",
                self.n_layers,
                len(self.block_pattern),
            )

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> "Transformer":
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        from .model import MoETransformer, NormalizedTransformer, Transformer

        log.info(
            f"Building transformer with {self.num_params:,d} total params, "
            f"{self.num_non_embedding_params:,d} non-embedding params"
        )
        model: Transformer
        if self.name == TransformerType.default:
            model = Transformer(
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
        elif self.name == TransformerType.normalized:
            assert self.embedding_norm is None
            model = NormalizedTransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                embedding_init_std=self.embedding_init_std,
                block_overrides=self.block_overrides,
                block_pattern=self.block_pattern,
            )
        elif self.name == TransformerType.moe:
            model = MoETransformer(
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
                tie_word_embeddings=self.tie_word_embeddings,
            )
        else:
            raise NotImplementedError(self.name)

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        log.info("%s", model)
        log.info(
            f"Built model with:\n"
            f"- {model.num_params:,d} total params\n"
            f"- {model.num_non_embedding_params:,d} non-embedding params\n"
            f"- {model.num_trainable_params:,d} trainable params"
        )

        return model

    @property
    def resolved_block_configs(self) -> list[TransformerBlockConfig]:
        return resolve_block_configs(
            n_layers=self.n_layers,
            block=self.block,
            block_pattern=self.block_pattern,
            block_overrides=self.block_overrides,
        )

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """
        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size
        if self.embedding_norm is not None:
            num_params += self.embedding_norm.num_params(self.d_model)

        # All block params.
        for block_config in self.resolved_block_configs:
            num_params += block_config.num_params(self.d_model)

        # LM head.
        num_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        # The LM head weight is shared with the embeddings when tied.
        if self.tie_word_embeddings:
            num_params -= self.d_model * self.vocab_size

        return num_params

    @property
    def num_active_params(self) -> int:
        """
        The total number of active parameters that a model from this config would have.
        """
        num_active_params = 0

        # Embedding params.
        num_active_params += self.d_model * self.vocab_size
        if self.embedding_norm is not None:
            num_active_params += self.embedding_norm.num_params(self.d_model)

        # All block active params.
        for block_config in self.resolved_block_configs:
            num_active_params += block_config.num_active_params(self.d_model)

        # LM head.
        num_active_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        # The LM head weight is shared with the embeddings when tied.
        if self.tie_word_embeddings:
            num_active_params -= self.d_model * self.vocab_size

        return num_active_params

    @property
    def num_non_embedding_params(self) -> int:
        """
        The number of parameters excluding embedding parameters.
        """
        return self.num_params - self.d_model * self.vocab_size

    @property
    def num_active_non_embedding_params(self) -> int:
        """
        The number of active parameters excluding embedding parameters.
        """
        return self.num_active_params - self.d_model * self.vocab_size

    @classmethod
    def olmo2_1M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=12,
            hidden_size_multiplier=1.0,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 4),
            head_dim=kwargs.pop("head_dim", 4),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_14M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=128,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_30M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=256,
            n_layers=kwargs.pop("n_layers", 4),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_60M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=384,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 8),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_100M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 100M OLMo2 model config.
        """
        return cls.llama_like(
            d_model=512,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 8),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=768,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=kwargs.pop("d_model", 1344),
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1536,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo2 model config.

        This is different from the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def olmo2_1B_v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo2 model config.

        This matches the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            n_layers=kwargs.pop("n_layers", 16),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.5),
            **kwargs,
        )

    @classmethod
    def olmo2_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 3B OLMo2 model config.
        """
        return cls.llama_like(
            d_model=3328,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo2 model config.
        """
        return cls.llama2_7B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo2 model config.
        """
        return cls.llama2_13B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo2 model config.
        """
        d_model = 5120
        return cls.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 512),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 27648 / (8 * d_model / 3)),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo3_1M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_1M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_14M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_14M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_30M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_30M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_60M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        config = cls.olmo2_60M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_100M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 100M OLMo3 model config.
        """
        config = cls.olmo2_100M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 190M OLMo3 model config.
        """
        config = cls.olmo2_190M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 370M OLMo3 model config.
        """
        config = cls.olmo2_370M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 600M OLMo3 model config.
        """
        config = cls.olmo2_600M(
            vocab_size=vocab_size,
            d_model=kwargs.pop("d_model", 1280),
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 760M OLMo3 model config.
        """
        config = cls.olmo2_760M(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo3 model config.
        """
        config = cls.olmo2_1B_v2(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 3B OLMo3 model config.
        """
        config = cls.olmo2_3B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo3 model config.
        """
        config = cls.olmo2_7B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo3 model config.
        """
        config = cls.olmo2_13B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    @classmethod
    def olmo3_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo3 model config.
        """
        config = cls.olmo2_32B(
            vocab_size=vocab_size,
            sliding_window=kwargs.pop(
                "sliding_window",
                SlidingWindowAttentionConfig(
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=True,
                    pattern=[4096, 4096, 4096, -1],
                ),
            ),
            attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
            **kwargs,
        )
        return config

    # -------------------------------------------------------------------------------------
    # Maple-Preview scale-down ladder
    # -------------------------------------------------------------------------------------
    #
    # See `maple/agents/contracts/ladder-and-factory.md` for the ratified ladder and the
    # reasoning behind every constant below. Two facts about how this factory is *reached*
    # shape its whole shape, and both are load-bearing:
    #
    #   1. `.edullm/train_on_corpus.py` dispatches with
    #      `getattr(TransformerConfig, opts.model_factory)` and then calls
    #      `factory(vocab_size=corpus.tokenizer.padded_vocab_size())` -- passing *only*
    #      `vocab_size`. So `vocab_size` must stay the only required argument, and no other
    #      knob is reachable from the command line.
    #   2. The only override channel is `config.merge(overrides)`, which runs *after*
    #      `build_config` has already built the model config. A dotlist override edits the
    #      serialized dict and rebuilds through `Config.from_dict` -- it never re-enters this
    #      factory, so it BYPASSES every assertion in `_maple_assert_ladder`.
    #
    # (2) is why rung selection is per-rung wrapper classmethods (`maple_r0` .. `maple_r3`)
    # rather than a `rung=` override: a rung passed as a dotlist would silently skip the
    # param-count check that is this factory's reason for existing.

    #: The Maple scale-down ladder. One entry per rung; `maple_scaled` reads nothing else.
    #:
    #: Ratio identities that make this Maple-faithful, preserved at every rung:
    #: ``f_e/d == 1/4``, ``k*f_e/d == 2.0``, ``n_heads*head_dim == d_model`` (attention width
    #: equals the residual stream, as in Maple), GQA 4:1, and ``L % 4 == 0`` for the 3:1 SWA
    #: pattern. `k/E == 1/32` holds at R3 only -- R1/R2 vary E deliberately, which is the
    #: E-sweep, and is why that identity is asserted only where it is claimed.
    #:
    #: The `total`/`active` figures are at ``V=100352`` (padded dolma2) and are asserted to
    #: within 1%. Active params are *exactly* equal across R1-R3 because active cost depends
    #: on `k`, not `E` -- that identity is what makes any throughput delta across the E-sweep
    #: attributable to kernel and routing overhead rather than arithmetic, so it is a
    #: correctness property of this table and not a coincidence.
    #: NOTE the `ClassVar`. `TransformerConfig` is a `@dataclass`, so a bare annotated
    #: class attribute becomes a FIELD -- and a dict default is a mutable default, which makes
    #: `@dataclass` raise at import time and takes the whole module with it. `ClassVar` is what
    #: keeps this a constant instead of a field. (Found the hard way: ruff, black and isort all
    #: passed on a module that would not import at all.)
    MAPLE_RUNGS: ClassVar[Dict[str, Dict[str, int]]] = {
        # R0 is a code-path smoke rung for `gpu-1xa10g`, never compared for quality. n_kv=1
        # (MQA rather than GQA 4:1) is accepted there for that reason.
        "R0": dict(d_model=512, n_layers=8, num_experts=64, n_heads=4, n_kv_heads=1),
        "R1": dict(d_model=1024, n_layers=12, num_experts=64, n_heads=8, n_kv_heads=2),
        "R2": dict(d_model=1024, n_layers=12, num_experts=128, n_heads=8, n_kv_heads=2),
        "R3": dict(d_model=1024, n_layers=12, num_experts=256, n_heads=8, n_kv_heads=2),
        # X2's low anchor: E=8 at R3's geometry. Same active params as R1-R3 by construction.
        "E8": dict(d_model=1024, n_layers=12, num_experts=8, n_heads=8, n_kv_heads=2),
        # M20 -- THE MISSION DELIVERABLE. **Maple-Preview's own shape, not a scale-down.**
        #
        # Read straight off `maple/evidence/config.json`, field for field, rather than derived
        # by scaling R3: `hidden_size=2048`, `num_hidden_layers=24`, `num_experts=256`,
        # `num_experts_per_tok=8`, `moe_intermediate_size=512`, `num_attention_heads=16`,
        # `num_key_value_heads=4`, `head_dim=128`, `num_shared_experts=0`,
        # `tie_word_embeddings=false`. Every ratio identity the ladder asserts holds here by
        # construction, which is the point: it is the fixed point the ladder scales *down* from.
        #   f_e/d   = 512/2048   = 1/4    OK
        #   k*f_e/d = 8*512/2048 = 2.0    OK
        #   k/E     = 8/256      = 1/32   OK  (claimed at M20 and R3, the two faithful points)
        #   n_h*h_d = 16*128     = 2048 = d  OK (1.0x, the D-012/D-014 assertion)
        #   GQA     = 16/4       = 4:1    OK
        #   L % 4   = 24 % 4     = 0      OK
        #
        # This differs from Maple ONLY in vocabulary: V=100,352 (padded dolma2) against Maple's
        # 151,936. That is a frozen decision (`maple/CLAUDE.md`), so **the total is 20.00B and
        # NOT DeepGrove's published 20.2B, and that is correct rather than an error.** The whole
        # 211,288,064 gap is `2*d*(151936 - 100352)`; see `MAPLE_EXPECTED_PARAMS` below.
        "M20": dict(d_model=2048, n_layers=24, num_experts=256, n_heads=16, n_kv_heads=4),
    }

    #: Expected ``(total, active)`` params per rung at ``V=100352``, asserted to within 1%.
    #: Keyed by vocab size because these numbers are dominated by the embedding tables --
    #: at R1, ``2 * 1024 * 100352`` is 205.5M of an 841.8M total, so a different vocab moves
    #: the total by more than the tolerance and the assertion must not fire spuriously.
    #:
    #: **These are MEASURED, not hand-derived.** Every figure below was read off
    #: ``config.num_params`` / ``config.num_active_params`` for the built config on FarmShare
    #: (job 1676541, L40S) and independently reproduced bit-for-bit by closed-form arithmetic.
    #: Two of them differ from the numbers first ratified in the contract -- see
    #: ``agents/lanes/L1-model-factory/STATUS.md``:
    #:   * R0 was 213.9M/125.8M, which is a **2.0x-width** R0 (n_heads=8 at d=512). At the
    #:     ruled 1.0x width (n_heads=4) R0 is 208.9M/120.9M.
    #:   * "active exactly constant across R1-R3" is true of active params **excluding
    #:     routers**, not of active params. Routers are active at every token and scale with E.
    MAPLE_EXPECTED_PARAMS: ClassVar[Dict[int, Dict[str, Tuple[int, int]]]] = {
        100352: {
            "R0": (208_939_520, 120_859_136),
            "R1": (841_773_056, 313_290_752),
            "R2": (1_446_539_264, 314_077_184),
            "R3": (2_656_071_680, 315_650_048),
            # M20 is DERIVED, NOT MEASURED -- the only row in this table that is, and it is
            # labelled so deliberately. Every other figure here was read off a built model on
            # FarmShare. These two were computed two independent ways and have never been built:
            #
            #   (a) the closed form in `contracts/ladder-and-factory.md`, and
            #   (b) an adversarial walk of `num_params`/`num_active_params` term by term as the
            #       interpreter executes them, which never consulted the closed form and which
            #       reproduced the MEASURED R3 and R1 rows bit-for-bit as its control
            #       (`maple/agents/lanes/P-M20/verify/m20-param-count-via-config-tree.md`).
            #
            # Both landed on exactly these integers. **That is two methods agreeing, which D-076
            # established is not proof** -- two derivations there agreed and were both wrong
            # because they shared a premise. The premise these two share is that the factory
            # constructs the config tree the verifier read. The ~$1.43 CPU dry-run in
            # `agents/lanes/P-M20/STATUS.md` closes that by printing `PARAM_LEDGER` off a built
            # config. **Until it has, treat this row as a falsifiable prediction**: if it is
            # wrong the assertion below FAILS THE RUN at config-build time for $1.43 and no GPU,
            # which is the cheapest possible place to find out.
            "M20": (20_002_742_272, 1_279_369_216),
        }
    }

    #: Active params EXCLUDING router params, at ``V=100352``. This is the quantity that is
    #: *exactly* invariant across the E-sweep (312,504,320 at R1, R2 and R3 alike), and it is
    #: therefore the one the FLOPs-per-token argument actually rests on. Asserted exactly.
    #:
    #: Router params are ``L * d * E`` -- 0.79M at R1, 1.57M at R2, 3.15M at R3. Every token
    #: traverses the full router, so they are active by definition and cannot be constant while
    #: E is the swept axis. Quoting plain "active params" as constant across the ladder would
    #: overstate the invariance by up to 1.01%, which is why this second table exists.
    MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS: ClassVar[Dict[int, Dict[str, int]]] = {
        100352: {
            "R0": 120_596_992,
            "R1": 312_504_320,
            "R2": 312_504_320,
            "R3": 312_504_320,
            # DERIVED, not measured -- see the note in MAPLE_EXPECTED_PARAMS. Routers at M20 are
            # L*d*E = 24*2048*256 = 12,582,912, so active-minus-routers is
            # 1,279,369,216 - 12,582,912. M20 is not part of the E-sweep (it is a single point,
            # not a rung of it), so this row exists for the ledger's completeness and to keep M20
            # out of the `expected is None` branch below -- NOT because any invariance is claimed
            # between M20 and R1-R3, which have a different d and L entirely.
            "M20": 1_266_786_304,
        }
    }

    @classmethod
    def maple_scaled(
        cls,
        vocab_size: int,
        *,
        rung: str = "R3",
        quantize: Optional[bool] = None,
        cache_quantized_weight: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        """
        A ratio-faithful scale-down of DeepGrove's Maple-Preview (20.2B total / 1.49B active).

        Maple-Preview is 24 layers at ``d=2048``, 256 experts routed top-8 with expert FFN 512
        and zero shared experts, 3:1 SWA-512:global with NoPE on the global layers, partial
        rotary 0.5, per-head QK-norm, untied embeddings. Preserving ``f_e/d = 1/4`` and
        ``k*f_e/d = 2.0`` forces ``k=8``, and ``k/E = 1/32`` then forces ``E=256``, so a
        ratio-faithful Maple is a one-parameter family in ``d``.

        :param vocab_size: The vocabulary size. Must stay the only required argument -- see
            the module comment above; the platform's dispatcher passes nothing else.
        :param rung: Which rung of the ladder, a key of :data:`MAPLE_RUNGS`. Not reachable
            from the platform command line; use the ``maple_r0`` .. ``maple_r3`` wrappers.
        :param quantize: Ternary QAT on the expert and attention projections. **Three states,
            not two, and the distinction is load-bearing for X4a:**

            * ``None`` (default) -- stock ``nn.Linear``. Use for any run that is *not* part of
              the ternary-vs-bf16 paired comparison.
            * ``False`` -- builds :class:`~olmo_core.nn.quantization.QuantLinear` with the
              quantizer **bypassed**. Bitwise identical to ``nn.Linear`` (it is the same
              ``F.linear`` call; ``QuantLinear`` subclasses ``nn.Linear``), but it keeps the
              same module graph and state-dict keys. **This is the bf16 CONTROL arm of X4a.**
            * ``True`` -- the ternary arm.

            ``None`` and ``False`` produce identical *numbers* but different *module trees*. If
            the control arm were built with ``None``, the two X4a arms would have different
            state dicts and the comparison would no longer be paired -- which is exactly the
            property `contracts/quant-surface.md` calls the cheapest thing to verify. So
            ``False`` is not a synonym for ``None``, and picking the wrong one produces a
            comparison that looks fine and is not paired.
        :param cache_quantized_weight: Reuse the quantized weight across the microbatches of a
            gradient-accumulation window instead of recomputing it per forward. The values are
            identical either way -- the latent weights do not move until the optimizer steps --
            so this is a pure throughput setting. See
            :attr:`~olmo_core.nn.quantization.QuantConfig.cache_quantized_weight` for the
            memory trade-off that keeps it off by default.
        """
        if rung not in cls.MAPLE_RUNGS:
            raise OLMoConfigurationError(
                f"unknown Maple rung {rung!r}; known rungs: {sorted(cls.MAPLE_RUNGS)}"
            )
        spec = cls.MAPLE_RUNGS[rung]

        d_model = kwargs.pop("d_model", spec["d_model"])
        n_layers = kwargs.pop("n_layers", spec["n_layers"])
        num_experts = kwargs.pop("num_experts", spec["num_experts"])
        n_heads = kwargs.pop("n_heads", spec["n_heads"])
        n_kv_heads = kwargs.pop("n_kv_heads", spec["n_kv_heads"])
        head_dim = kwargs.pop("head_dim", 128)
        top_k = kwargs.pop("top_k", 8)
        # f_e = d/4. Maple is d=2048 -> 512.
        expert_hidden_size = kwargs.pop("expert_hidden_size", d_model // 4)

        # Deferred import, deliberately. L4 owns `nn/quantization.py` and the merge order is
        # L1 -> ... -> L4, so a module-level import of it would make THIS branch unimportable
        # until L4 lands. Imported at call time instead, and only when quantization was asked
        # for -- so `quantize=None` (every bf16 run) has no dependency on L4 at all.
        quant = None
        if quantize is not None:
            try:
                from ..quantization import QuantConfig
            except ImportError as e:
                raise OLMoConfigurationError(
                    "`quantize` was requested but `olmo_core.nn.quantization` is not present "
                    "in this tree -- it is L4's C1-C5 and lands after L1 in the merge order. "
                    "This refuses rather than silently building an unquantized model, because "
                    "a ternary arm that quietly ran in bf16 would look like a *successful* "
                    "paired comparison, which is the worst outcome for X4a."
                ) from e
            quant = QuantConfig(
                enabled=quantize, cache_quantized_weight=cache_quantized_weight
            )

        config = cls._maple_config(
            quant=quant,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_size=expert_hidden_size,
            **kwargs,
        )
        cls._maple_assert_ladder(
            config,
            rung=rung,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_size=expert_hidden_size,
            vocab_size=vocab_size,
        )
        return config

    @classmethod
    def _maple_config(
        cls,
        *,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        **kwargs,
    ) -> "TransformerConfig":
        """Build the Maple config. Every non-default knob here is deliberate; see comments."""
        dtype = kwargs.pop("dtype", DType.float32)
        layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-6)
        # None -> stock nn.Linear. QuantConfig(enabled=False) -> QuantLinear, bypassed, bitwise
        # identical to nn.Linear but sharing the module graph. See `maple_scaled`'s docstring.
        quant = kwargs.pop("quant", None)
        # `AttentionConfig.quant`, `MoEConfig.quant` and `MoEConfig.swiglu_limit` are L4's
        # fields and do not exist until L4 merges (L1 lands first). Pass them only when they
        # are actually available, so this branch builds standalone AND picks them up
        # automatically once L4 is in, with no second edit and no silent divergence.
        quant_kwargs = {} if quant is None else {"quant": quant}
        # The gate/up clamp. Maple applies it in `MapleMLP.forward` unconditionally, in BOTH
        # precision regimes, so its absence is a faithfulness divergence and not a quant detail.
        #
        # This used to fall back silently to "no clamp" when L4's field was absent, which the
        # independent audit correctly flagged (F5): a silently-absent architectural feature is
        # the exact failure class this factory exists to prevent, and it was inconsistent with
        # how `quantize` behaves one screen up -- that RAISES. So this warns loudly instead of
        # passing in silence, and `require_swiglu_limit=True` turns it into a hard error for
        # anyone who wants the guarantee rather than the notice.
        require_swiglu_limit = kwargs.pop("require_swiglu_limit", False)
        try:
            from ..feed_forward import MAPLE_SWIGLU_LIMIT

            swiglu_kwargs = {"swiglu_limit": kwargs.pop("swiglu_limit", MAPLE_SWIGLU_LIMIT)}
        except ImportError as e:
            swiglu_kwargs = {}
            kwargs.pop("swiglu_limit", None)
            msg = (
                "MoEConfig.swiglu_limit is unavailable in this tree, so Maple's gate/up clamp "
                "(gate max=7.0, up [-7,7]) is ABSENT from every expert built by this factory. "
                "Maple applies it unconditionally in both precision regimes, so this model is "
                "not clamp-faithful. It lands with L4's C4."
            )
            if require_swiglu_limit:
                raise OLMoConfigurationError(msg) from e
            log.warning("%s Pass require_swiglu_limit=True to make this fatal.", msg)

        layer_norm = LayerNormConfig(
            name=LayerNormType.rms, eps=layer_norm_eps, bias=False, dtype=dtype
        )

        # A3 + A4 + A5 IN ONE CONFIG, NOT TWELVE.
        #
        # Per-layer window size is NOT per-block config in this tree. `AttentionConfig.build`
        # receives `layer_idx`/`n_layers` and computes the window from the pattern itself; and
        # -- this is the part worth knowing -- the same `else` branch that handles a global
        # layer is what drops RoPE when `no_global_rope` is set. So one shared
        # `SlidingWindowAttentionConfig` realizes the 3:1 SWA layout *and* NoPE-on-globals
        # together.
        #
        # DO NOT hand-place this with `block_overrides`: 12 deep copies would each still
        # resolve from their own `layer_idx`, giving identical behavior with 12x the config
        # surface and a second place for the layout to disagree with itself.
        #
        # DO NOT use `block_pattern` either: it `cycle()`s and would silently give a wrong
        # layout that trains happily.
        #
        # Both `force_full_attention_on_*` are False because Maple's own `layer_types` puts
        # globals at 3, 7, ..., 23 with layer 0 sliding and layer 23 global. Leaving
        # `force_full_attention_on_first_layer` at its default `True` would both make layer 0
        # global AND shift the whole pattern by one (`_get_window_size` decrements the
        # effective index), i.e. two errors, neither of which raises.
        sliding_window = kwargs.pop(
            "sliding_window",
            SlidingWindowAttentionConfig(
                pattern=[512, 512, 512, -1],
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=False,
            ),
        )

        attention = AttentionConfig(
            name=AttentionType.default,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bias=False,
            rope=RoPEConfig(
                name=RoPEType.default,
                # 10,000 -- NOT this tree's 500,000 default. Maple's config.json says 10000,
                # and a wrong theta on partial-rotary SWA layers is a faithfulness bug that
                # trains happily.
                theta=kwargs.pop("rope_theta", 10_000),
                full_precision=True,
                # Realizes NoPE on the global layers, via the branch described above.
                no_global_rope=True,
                # Rotate only the leading 64 of 128 head dims.
                partial_rotary_factor=0.5,
            ),
            # Per-HEAD QK-norm, matching Maple's `MapleRMSNorm(self.head_dim)`. The default
            # (`use_head_qk_norm=False`) norms over the whole concatenated projection instead,
            # which is a DIFFERENT OPERATOR, not just a different parameter count.
            qk_norm=layer_norm,
            use_head_qk_norm=True,
            sliding_window=sliding_window,
            # PINNED to ONE backend for every layer, and the value is resolved at call time.
            #
            # Left as `None`, the backend is chosen PER LAYER from whether that layer has a
            # window: a sliding layer takes the `if backend is None and has_flash_attn_2()`
            # branch and gets flash_2, while a global layer skips it and falls through to
            # `backend = torch` (`nn/attention/__init__.py:454-462`). On a flash-attn host that
            # silently turns our 3:1 SWA layout into a 3:1 *kernel* split -- 9 layers on
            # FlashAttention-2 and layers {3,7,11} on torch SDPA. Two consequences, neither
            # visible in the config:
            #   1. Throughput. Three layers on a slower kernel biases absolute MFU downward and
            #      makes it non-comparable to the `moe/` track's baseline, which had no
            #      global/sliding split. MFU is the E-sweep's dependent variable.
            #   2. Correctness, latently: torch SDPA raises on intra-document masking, so a
            #      packed-corpus run with `generate_doc_lengths=True` would die at the first
            #      global layer. flash_2 supports it.
            #
            # WHY THIS IS NOT A HARD `flash_2`, which is what the sibling `olmo3_*` factories
            # do: flash-attn 2 is **not installed in the platform image**. `.edullm/Dockerfile`
            # installs `torch==2.9.0` and `.[wandb]`, and flash-attn appears in neither
            # `pyproject.toml`'s core dependencies nor its extras (only `fa4`, a different
            # package). `AttentionBackendName.flash_2.assert_supported()` RAISES when the
            # package is missing (`nn/attention/backend.py:452-457`), so a hard pin would make
            # every rung unbuildable on the platform and on FarmShare. Measured: it did --
            # FarmShare job 1676576 died with "'FlashAttention2Backend' is missing the
            # flash-attn package".
            #
            # So: use flash_2 where it exists, torch everywhere else, and either way use the
            # SAME backend on all layers -- which is the property that actually matters. The
            # uniformity is asserted below; the identity of the backend is not.
            backend=kwargs.pop("attn_backend", None) or _maple_uniform_attn_backend(),
            # q/k/v/o. Maple ternarizes every matmul; norms stay full precision, and there is
            # no code path from here to the QK-norms.
            **quant_kwargs,
            dtype=dtype,
        )

        moe = MoEConfig(
            name=MoEType.default,
            num_experts=num_experts,
            hidden_size=expert_hidden_size,
            # Explicit, and NOT left to `None`: `MoEConfig.build` calls
            # `as_dict(exclude_none=True)`, which drops the key, so `None` silently becomes
            # `MoE.__init__`'s own default of 1.2. At R3 (mean load 256) a factor of 1.2 is
            # known-wrong -- headroom collapses to ~3.5 sigma -- and dropless is unavailable
            # because `grouped_gemm` is unreachable in the image.
            capacity_factor=kwargs.pop("capacity_factor", 2.0),
            router=MoERouterConfig(
                # Stock default is 1. Left unset this is silently a top-1 model.
                top_k=top_k,
                # Maple's `norm_topk_prob`. Measured gate mass 0.161 vs 1.000 when unset --
                # a 6.2x error that trains happily. Zero of five shipped recipes set it.
                normalize_expert_weights=1.0,
                # Maple has no expert bias (`moe_router_enable_expert_bias: false`). Do NOT
                # set this as a load-balancing fallback -- it would change the architecture
                # under test.
                bias_gamma=None,
            ),
            # Maple has zero shared experts. MoTE measured a ternary shared expert at 48.2 vs
            # 57.3 BF16, and it would cost +151 MB/token at Maple scale.
            shared_mlp=None,
            lb_loss_weight=kwargs.pop("lb_loss_weight", 0.01),
            # Explicit: stock default is off.
            z_loss_weight=kwargs.pop("z_loss_weight", 0.001),
            # EXPERT projections only. The router is structurally unreachable from this flag,
            # and that is load-bearing: routing is discrete, so quantizing the router would
            # change *which* experts fire rather than how accurately they compute.
            **quant_kwargs,
            # gpt-oss's asymmetric SwiGLU outlier guard (gate `max=7.0`, up `[-7,7]`), which
            # Maple carries in `MapleMLP.forward` unconditionally -- NOT gated on `quantize`.
            # At ~52x the measured pre-activation RMS it never fires, so it costs nothing; but
            # gating it on the quant flag would confound the clamp with the precision change
            # across X4a's two arms, which is the one thing X4a is trying to measure.
            **swiglu_kwargs,
            dtype=dtype,
        )

        block = TransformerBlockConfig(
            name=TransformerBlockType.moe,
            sequence_mixer=attention,
            feed_forward_moe=moe,
            layer_norm=layer_norm,
        )

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            name=TransformerType.moe,
            dtype=dtype,
            # Maple is untied (`tie_word_embeddings: false`). Untied is also this tree's
            # default, but it is pinned here so an upstream change cannot move our ledger.
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )

    @classmethod
    def _maple_assert_ladder(
        cls,
        config: "TransformerConfig",
        *,
        rung: str,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        vocab_size: int,
    ) -> None:
        """
        A2. Assert the ratio identities and the param ledger, and **raise** rather than warn.

        These are cheap insurance against a class of failure this project has already paid
        for: a config that serializes correctly, trains happily, and is not the model anyone
        intended. Every check here is an identity someone has claimed in writing.
        """
        problems: List[str] = []

        # --- Ratio identities -----------------------------------------------------------
        # f_e/d == 1/4 and k*f_e/d == 2.0 are what force k=8; they are the reason this
        # scale-down is a one-parameter family in d rather than a free choice.
        if expert_hidden_size * 4 != d_model:
            problems.append(
                f"f_e/d must be 1/4: got f_e={expert_hidden_size}, d={d_model} "
                f"(ratio {expert_hidden_size / d_model:.4f})"
            )
        if top_k * expert_hidden_size != 2 * d_model:
            problems.append(
                f"k*f_e/d must be 2.0: got k={top_k}, f_e={expert_hidden_size}, d={d_model} "
                f"(ratio {top_k * expert_hidden_size / d_model:.4f})"
            )
        # k/E == 1/32 is claimed at the two Maple-faithful points, R3 and M20. R1/R2/E8 vary E
        # deliberately -- that variation IS the E-sweep -- so asserting it everywhere would
        # forbid the experiment. M20 is Maple's literal config, so if this ever fails there the
        # transcription from `evidence/config.json` is wrong, which is the more valuable catch.
        if rung in ("R3", "M20") and top_k * 32 != num_experts:
            problems.append(
                f"k/E must be 1/32 at R3: got k={top_k}, E={num_experts} "
                f"(ratio {top_k / num_experts:.5f})"
            )

        # --- Geometry -------------------------------------------------------------------
        # THE assertion that catches a mixed-geometry ladder. Attention width must equal the
        # residual stream, as it does in Maple (16 * 128 == 2048 == d). A ladder whose rows
        # were computed at different widths reproduces its own totals row by row and is still
        # wrong; only checking one identity against every row catches it.
        if n_heads * head_dim != d_model:
            problems.append(
                f"attention width must equal the residual stream: n_heads*head_dim = "
                f"{n_heads}*{head_dim} = {n_heads * head_dim} != d_model={d_model} "
                f"({n_heads * head_dim / d_model:.2f}x)"
            )
        # The 3:1 SWA pattern has period 4, so a layer count not divisible by 4 would leave a
        # truncated final cycle and put the globals somewhere nobody chose.
        if n_layers % 4 != 0:
            problems.append(f"L % 4 must be 0 for the 3:1 SWA pattern: got L={n_layers}")
        if d_model % head_dim != 0:
            problems.append(f"d_model {d_model} must be divisible by head_dim {head_dim}")
        # GQA 4:1, except at R0, which is a code-path smoke rung and is MQA (4/1) by design.
        expected_gqa = 4 if rung != "R0" else n_heads
        if n_heads != expected_gqa * n_kv_heads:
            problems.append(
                f"GQA ratio must be {expected_gqa}:1 at {rung}: got n_heads={n_heads}, "
                f"n_kv_heads={n_kv_heads}"
            )

        # --- Known-broken defaults that must have been set explicitly -------------------
        # Re-read off the built config rather than off the local variables, so this checks
        # what was actually constructed and not what we meant to construct.
        block = config.block
        assert not isinstance(block, dict)
        moe = block.feed_forward_moe
        if moe is None:
            problems.append("expected an MoE block, got a dense one")
        else:
            if moe.router.top_k != top_k:
                problems.append(f"router top_k is {moe.router.top_k}, expected {top_k}")
            if moe.router.normalize_expert_weights != 1.0:
                problems.append(
                    "normalize_expert_weights must be 1.0 (Maple's `norm_topk_prob`); got "
                    f"{moe.router.normalize_expert_weights!r}. Unset, gate mass measures "
                    "0.161 against 1.000 -- a 6.2x error that trains happily."
                )
            if moe.router.bias_gamma is not None:
                problems.append(
                    f"bias_gamma must be unset (Maple has no expert bias); got "
                    f"{moe.router.bias_gamma!r}"
                )
            # Two separate checks, because they catch two different mistakes. `None` is the
            # silent-default bug: `as_dict(exclude_none=True)` drops the key, so `MoE.__init__`'s
            # own `capacity_factor: float = 1.2` wins and "unset" silently means 1.2 -- the
            # known-wrong value at R3. The value check then pins the funded choice (D-009), so an
            # explicit 1.2 is rejected too rather than coinciding with the default and passing.
            if moe.capacity_factor is None:
                problems.append(
                    "capacity_factor must be set explicitly; `None` is dropped by "
                    "`exclude_none` and silently becomes 1.2"
                )
            elif moe.capacity_factor != 2.0:
                problems.append(
                    f"capacity_factor must be 2.0 (D-009: the funded path is MoEType.default at "
                    f"2.0, dropless descoped because grouped_gemm does not build in the image); "
                    f"got {moe.capacity_factor}. At cf=2.0 the `ensure_multiple_of(..., 8)` "
                    f"quantization vanishes and effective capacity is exactly 2.0000 at every "
                    f"rung, which is what keeps the E-sweep unconfounded."
                )
            if moe.z_loss_weight is None:
                problems.append("z_loss_weight must be set explicitly")
            if moe.shared_mlp is not None:
                problems.append("Maple has zero shared experts; got a shared MLP")

        mixer = block.sequence_mixer
        if isinstance(mixer, AttentionConfig):
            if mixer.qk_norm is None:
                problems.append("QK-norm must be on (Maple `use_qk_norm: true`)")
            if not mixer.use_head_qk_norm:
                problems.append(
                    "QK-norm must be PER-HEAD (`use_head_qk_norm=True`) to match Maple's "
                    "`MapleRMSNorm(head_dim)`; the default norms the whole concatenated "
                    "projection, which is a different operator"
                )
            if mixer.sliding_window is None:
                problems.append("expected a sliding-window config for the 3:1 SWA pattern")
            else:
                swa = mixer.sliding_window
                if swa.force_full_attention_on_first_layer:
                    problems.append(
                        "force_full_attention_on_first_layer must be False: it makes layer 0 "
                        "global AND shifts the whole pattern by one"
                    )
                if swa.force_full_attention_on_last_layer:
                    problems.append("force_full_attention_on_last_layer must be False")
            # The backend must be PINNED, not resolved per layer. Unpinned, sliding layers get
            # flash_2 and global layers fall through to torch SDPA, making the 3:1 SWA layout a
            # 3:1 kernel split that biases MFU and crashes on intra-document masking. Neither
            # is visible in the config, and MFU is the E-sweep's dependent variable.
            if mixer.backend is None:
                problems.append(
                    "attention backend must be pinned to a single value; left None it is "
                    "resolved PER LAYER from the presence of a window, so sliding layers get "
                    "flash_2 while global layers fall through to torch SDPA "
                    "(nn/attention/__init__.py:454-462) -- a 3:1 kernel split that biases MFU "
                    "and breaks intra-document masking on the global layers"
                )
            rope = mixer.rope
            if rope is None:
                problems.append("expected a RoPE config")
            else:
                if not rope.no_global_rope:
                    problems.append("no_global_rope must be True (NoPE on the global layers)")
                if rope.partial_rotary_factor != 0.5:
                    problems.append(
                        f"partial_rotary_factor must be 0.5; got {rope.partial_rotary_factor}"
                    )
                if rope.theta != 10_000:
                    problems.append(
                        f"rope_theta must be 10000 to match Maple; got {rope.theta}. This "
                        "tree defaults to 500000."
                    )

        if config.tie_word_embeddings:
            problems.append("Maple has untied embeddings; got tied")

        # --- The param ledger -----------------------------------------------------------
        # Printed unconditionally and BEFORE any raise, so a mismatch is diagnosable rather
        # than merely fatal. `PARAM_LEDGER ` is the agreed stdout protocol.
        total = config.num_params
        active = config.num_active_params
        embed = d_model * vocab_size
        # Router params are active at every token and scale with E, so they are the reason
        # plain `active` is not constant across the E-sweep. Report them separately.
        routers = n_layers * d_model * num_experts
        print(
            f"PARAM_LEDGER rung={rung} V={vocab_size} d={d_model} L={n_layers} "
            f"E={num_experts} k={top_k} f_e={expert_hidden_size} "
            f"n_heads={n_heads} n_kv={n_kv_heads} head_dim={head_dim} "
            f"total={total} active={active} embed={embed} "
            f"non_embed_total={total - embed} non_embed_active={active - embed} "
            f"routers={routers} active_minus_routers={active - routers}",
            flush=True,
        )

        expected = cls.MAPLE_EXPECTED_PARAMS.get(vocab_size, {}).get(rung)
        if expected is None:
            # Say so. A rung or vocab with no ratified figures skips BOTH ledger checks, and a
            # skipped assertion that announces nothing is indistinguishable from a passing one --
            # which is how `E8` (X2's low anchor) could have been submitted un-gated.
            log.warning(
                "no ratified param figures for rung=%s at V=%d, so the total/active ledger "
                "assertions were SKIPPED, not passed. Computed: total=%d active=%d "
                "active_minus_routers=%d. File these in MAPLE_EXPECTED_PARAMS before relying "
                "on this rung.",
                rung,
                vocab_size,
                total,
                active,
                active - routers,
            )
        else:
            exp_total, exp_active = expected
            # 1% because the published table is rounded, not because the count is uncertain.
            if abs(total - exp_total) > 0.01 * exp_total:
                problems.append(
                    f"total params {total:,} is outside 1% of the ladder's {exp_total:,} "
                    f"for {rung} at V={vocab_size} ({100 * (total - exp_total) / exp_total:+.2f}%)"
                )
            if abs(active - exp_active) > 0.01 * exp_active:
                problems.append(
                    f"active params {active:,} is outside 1% of the ladder's {exp_active:,} "
                    f"for {rung} at V={vocab_size} "
                    f"({100 * (active - exp_active) / exp_active:+.2f}%)"
                )

        # The E-sweep's load-bearing invariant, asserted EXACTLY rather than to a tolerance:
        # active-params-excluding-routers must be identical across R1/R2/R3, because that is
        # what makes FLOPs/token constant and therefore what makes a measured throughput delta
        # attributable to kernel and routing overhead rather than to arithmetic. A 1% band here
        # would admit exactly the drift it is supposed to forbid.
        exp_amr = cls.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS.get(vocab_size, {}).get(rung)
        if exp_amr is not None and active - routers != exp_amr:
            problems.append(
                f"active params excluding routers is {active - routers:,}, expected exactly "
                f"{exp_amr:,} for {rung} at V={vocab_size}. This quantity must be identical "
                f"across R1-R3 or the E-sweep's throughput attribution does not hold."
            )

        if problems:
            raise OLMoConfigurationError(
                f"maple_scaled({rung}) violates the ratified ladder "
                f"(contracts/ladder-and-factory.md):\n  - " + "\n  - ".join(problems)
            )

    @classmethod
    def maple_r0(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """R0, the code-path smoke rung: d=512, L=8, E=64.

        Sized for `gpu-1xa10g` at ~$1/hr. Never compared for quality -- it is MQA rather than
        GQA 4:1 and exists only to prove the code path runs before A100 time is spent.

        Param counts are deliberately NOT quoted here. They live in
        :data:`MAPLE_EXPECTED_PARAMS`, which is the authoritative table; every one of the three
        errors this ladder has had entered through a hand-copied figure, including a stale
        "~214M / ~126M" that sat in this docstring after the geometry was corrected.
        """
        return cls.maple_scaled(vocab_size, rung="R0", **kwargs)

    @classmethod
    def maple_r1(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """R1: d=1024, L=12, E=64. Counts in :data:`MAPLE_EXPECTED_PARAMS`, not here."""
        return cls.maple_scaled(vocab_size, rung="R1", **kwargs)

    @classmethod
    def maple_r2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """R2: d=1024, L=12, E=128. Counts in :data:`MAPLE_EXPECTED_PARAMS`, not here."""
        return cls.maple_scaled(vocab_size, rung="R2", **kwargs)

    @classmethod
    def maple_r3(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """R3, the flagship: d=1024, L=12, E=256.

        The unique sub-Maple point that preserves every ratio: f_e/d = 1/4 and k*f_e/d = 2.0
        force k=8, then k/E = 1/32 forces E=256, and d/L with head_dim 128 and L % 4 == 0
        forces d=1024/L=12.

        Counts in :data:`MAPLE_EXPECTED_PARAMS`, not here.
        """
        return cls.maple_scaled(vocab_size, rung="R3", **kwargs)

    @classmethod
    def maple_m20(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """M20, the mission deliverable: Maple-Preview's OWN shape. d=2048, L=24, E=256.

        Not a scale-down -- this is DeepGrove's published config transcribed from
        `maple/evidence/config.json`, at our vocabulary instead of theirs. It is the fixed point
        that R0-R3 scale down *from*, and every ratio identity the ladder preserves holds here by
        construction rather than by choice.

        **The total is 20.00B, not the published 20.2B**, because V=100,352 (padded dolma2) and
        Maple is 151,936. The entire difference is the untied embedding pair. Do not quote this
        model as reproducing DeepGrove's headline parameter count, and **report bits-per-byte
        rather than loss** -- ln(100352) vs ln(151936) is a 0.415-nat offset before fertility.

        **Active params are ~1.28B, not the ~1B a scale target might suggest, and this is the one
        place where the mission's stated numbers and Maple-faithfulness cannot both hold.** See
        `agents/lanes/P-M20/NOTES.md`: at Maple's own geometry `k` and `f_e` are fully determined
        by the ratio identities, so active params are not a free parameter. Nothing was adjusted
        to hit a round number.

        Counts in :data:`MAPLE_EXPECTED_PARAMS`, not here -- and note that M20's row there is the
        only **derived** row in the table. It has never been built.
        """
        return cls.maple_scaled(vocab_size, rung="M20", **kwargs)

    @classmethod
    def smallmoe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                shared_mlp=FeedForwardConfig(hidden_size=d_model * 2),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def small_hybrid_moe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_hybrid_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def olmoe_1B_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 2048)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless,
                num_experts=64,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=8),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def ngpt_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M nGPT model config.
        """
        return cls.ngpt_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def ngpt_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B nGPT model config.
        """
        return cls.ngpt_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def llama2_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M Llama2-like model config.
        """
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama2-like model config.

        Note: Llama2 doesn't have a 1B. We made this up.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 40),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_26B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 26B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=8192,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def llama3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=1024,
            **kwargs,
        )

    @classmethod
    def llama3_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=8196,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_405B(
        cls,
        vocab_size: int,
        **kwargs,
    ) -> "TransformerConfig":
        """
        A 405B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=16384,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 126),
            n_heads=kwargs.pop("n_heads", 128),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def gemma3_1B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 1B model config.
        """
        return cls.gemma3_like(
            d_model=2304,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 26),
            n_heads=kwargs.pop("n_heads", 8),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            hidden_size=kwargs.pop("hidden_size", 9216),
            **kwargs,
        )

    @classmethod
    def qwen3_0_6B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 28),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=3072, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            **kwargs,
        )

    @classmethod
    def gemma3_4B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 4B model config.
        """
        return cls.gemma3_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 34),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            hidden_size=kwargs.pop("hidden_size", 10240),
            **kwargs,
        )

    @classmethod
    def qwen3_1_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 28),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=6144, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            **kwargs,
        )

    @classmethod
    def gemma3_12B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 12B model config.
        """
        return cls.gemma3_like(
            d_model=3840,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 48),
            n_heads=kwargs.pop("n_heads", 24),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            hidden_size=kwargs.pop("hidden_size", 15360),
            **kwargs,
        )

    @classmethod
    def qwen3_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 36),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=9728, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            **kwargs,
        )

    @classmethod
    def gemma3_27B(cls, vocab_size: int = 262208, **kwargs) -> "TransformerConfig":
        """
        Gemma 3 27B model config.
        """
        return cls.gemma3_like(
            d_model=5376,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 62),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 16),
            hidden_size=kwargs.pop("hidden_size", 21504),
            **kwargs,
        )

    @classmethod
    def qwen3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 36),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=12288, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_14B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 48),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=17408, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            head_dim=kwargs.pop("head_dim", 128),
            rope_theta=kwargs.pop("rope_theta", 1_000_000),
            rope_full_precision=kwargs.pop("rope_full_precision", False),
            layer_norm_eps=1e-6,
            layer_norm_name=LayerNormType.qwen_rms,
            qk_norm=kwargs.pop("qk_norm", True),
            use_head_qk_norm=kwargs.pop("use_head_qk_norm", True),
            feed_forward=FeedForwardConfig(
                hidden_size=25600, bias=False, dtype=kwargs.get("dtype", DType.float32)
            ),
            **kwargs,
        )

    @classmethod
    def qwen3_5_0_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 24),
            n_heads=kwargs.pop("n_heads", 8),
            n_kv_heads=kwargs.pop("n_kv_heads", 2),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 3584),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 16),
            **kwargs,
        )

    @classmethod
    def qwen3_5_4B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=2560,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 9216),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 32),
            **kwargs,
        )

    @classmethod
    def qwen3_5_9B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 16),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 12288),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 32),
            **kwargs,
        )

    @classmethod
    def qwen3_5_27B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.qwen3_5_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 24),
            n_kv_heads=kwargs.pop("n_kv_heads", 4),
            head_dim=kwargs.pop("head_dim", 256),
            intermediate_size=kwargs.pop("intermediate_size", 17408),
            linear_num_key_heads=kwargs.pop("linear_num_key_heads", 16),
            linear_num_value_heads=kwargs.pop("linear_num_value_heads", 48),
            **kwargs,
        )

    @classmethod
    def qwen3_5_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        rope_theta: int = 10_000_000,
        partial_rotary_factor: float = 0.25,
        layer_norm_eps: float = 1e-6,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,
        attn_backend: Optional[AttentionBackendName] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Qwen3.5-like hybrid model configuration.

        Qwen3.5 dense models combine Gated DeltaNet (linear attention) layers with
        full attention layers in a 3:1 ratio. Both layer types use pre-norm blocks with
        Qwen-style RMS normalization, per-head QK norm and output gating on full-attention
        layers, and partial RoPE (25% of head dimension by default).
        """
        layer_norm = LayerNormConfig(
            name=LayerNormType.qwen_rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        gdn_block = TransformerBlockConfig(
            name=TransformerBlockType.default,
            sequence_mixer=GatedDeltaNetConfig(
                n_heads=linear_num_key_heads,
                n_v_heads=linear_num_value_heads,
                head_dim=linear_key_head_dim,
                expand_v=linear_value_head_dim / linear_key_head_dim,
                allow_neg_eigval=False,
                conv_size=linear_conv_kernel_dim,
                norm_eps=layer_norm_eps,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype),
            layer_norm=layer_norm,
        )

        attn_block = TransformerBlockConfig(
            name=TransformerBlockType.default,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(
                    name=RoPEType.default,
                    theta=rope_theta,
                    full_precision=kwargs.pop("rope_full_precision", False),
                    partial_rotary_factor=partial_rotary_factor,
                ),
                gate=GateConfig(granularity=GateGranularity.elementwise),
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                use_flash=use_flash,
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
            block={"gdn": gdn_block, "attn": attn_block},
            block_pattern=["gdn", "gdn", "gdn", "attn"],
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            **kwargs,
        )

    @classmethod
    def llama_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        gate: Optional[GateConfig] = None,
        qk_norm: bool = False,
        use_head_qk_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        layer_norm_name: Optional[LayerNormType] = None,
        rope_theta: int = 500_000,
        rope_type: Optional[RoPEType] = None,
        rope_full_precision: bool = True,
        no_global_rope: bool = False,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,
        attn_backend: Optional[AttentionBackendName] = None,
        sliding_window: Optional[SlidingWindowAttentionConfig] = None,
        block_name: TransformerBlockType = TransformerBlockType.default,
        block_mods: Optional[
            Dict[int, Callable[[TransformerBlockConfig], TransformerBlockConfig]]
        ] = None,
        dtype: DType = DType.float32,
        rope_scaling: Optional[RoPEScalingConfig] = None,
        feed_forward: Optional[FeedForwardConfig] = None,
        feed_forward_moe: Optional[MoEConfig] = None,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Llama-like model configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible.
        :param layer_norm_name: Override the layer norm implementation. Defaults to
            :data:`LayerNormType.fused_rms` when ``fused_ops=True``, otherwise
            :data:`LayerNormType.rms`.
        :param block_mods: A dictionary of block indices to functions that take the base block config and return a modified block config.
        :param dtype: The default data type to use for all parameters.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure global layer norm.
        if layer_norm_name is None:
            layer_norm_name = LayerNormType.fused_rms if fused_ops else LayerNormType.rms
        layer_norm = LayerNormConfig(
            name=layer_norm_name,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        # Decide on attention/rope implementations.
        att_type = AttentionType.default
        if rope_type is None:
            rope_type = RoPEType.default
            if fused_ops and n_kv_heads is None:  # fused attention not compatible with MQA/GQA.
                att_type = AttentionType.fused
                rope_type = RoPEType.fused

        # Feed-forward.
        if feed_forward is None and feed_forward_moe is None:
            feed_forward = FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=block_name,
            sequence_mixer=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(
                    name=rope_type,
                    theta=rope_theta,
                    full_precision=rope_full_precision,
                    no_global_rope=no_global_rope,
                    scaling=rope_scaling,
                ),
                gate=gate,
                qk_norm=layer_norm if qk_norm else None,
                use_head_qk_norm=use_head_qk_norm if qk_norm else None,
                use_flash=use_flash,
                backend=attn_backend,
                sliding_window=sliding_window,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
        )

        if block_mods and kwargs.get("block_overrides"):
            raise OLMoConfigurationError(
                "`block_mods` and `block_overrides` cannot be used together."
            )
        block_overrides = None
        if block_mods:
            block_overrides = {i: block_mods[i](block.copy()) for i in block_mods}
        elif kwargs.get("block_overrides"):
            block_overrides = kwargs.get("block_overrides")

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_overrides=block_overrides,
            **kwargs,
        )

    @classmethod
    def llama_like_moe(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        shared_expert_hidden_size: Optional[int] = None,
        dropless: bool = False,
        capacity_factor: Optional[float] = None,
        lb_loss_weight: float = 0.01,
        z_loss_weight: Optional[float] = 0.001,
        reordered_norm: bool = False,
        hybrid: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        block_name: TransformerBlockType
        if reordered_norm:
            block_name = (
                TransformerBlockType.moe_hybrid_reordered_norm
                if hybrid
                else TransformerBlockType.moe_reordered_norm
            )
        else:
            block_name = TransformerBlockType.moe_hybrid if hybrid else TransformerBlockType.moe
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            name=TransformerType.moe,
            block_name=block_name,
            qk_norm=kwargs.pop("qk_norm", reordered_norm),
            feed_forward_moe=MoEConfig(
                name=MoEType.default if not dropless else MoEType.dropless,
                num_experts=num_experts,
                hidden_size=expert_hidden_size,
                capacity_factor=capacity_factor,
                router=MoERouterConfig(top_k=top_k),
                shared_mlp=None
                if shared_expert_hidden_size is None
                else FeedForwardConfig(hidden_size=shared_expert_hidden_size, bias=False),
                lb_loss_weight=lb_loss_weight,
                z_loss_weight=z_loss_weight,
            ),
            **kwargs,
        )

    @classmethod
    def ngpt_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        rope_theta: int = 500_000,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        use_flash: bool = False,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create an nGPT-like model configuration.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=TransformerBlockType.normalized,
            sequence_mixer=AttentionConfig(
                name=AttentionType.normalized,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                qk_norm=None if not qk_norm else LayerNormConfig(name=LayerNormType.l2_norm),
                rope=RoPEConfig(name=RoPEType.default, theta=rope_theta),
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                name=FeedForwardType.normalized, hidden_size=hidden_size, dtype=dtype
            ),
        )

        return cls(
            name=TransformerType.normalized,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(name=LMHeadType.normalized, dtype=dtype),
            dtype=dtype,
            init_method=InitMethod.normalized,
            **kwargs,
        )

    @classmethod
    def gemma3_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        head_dim: Optional[int] = None,
        gate: Optional[GateConfig] = None,
        activation: ActivationFunction = ActivationFunction.gelu_tanh,
        local_window_size: int = 1024,
        local_rope_theta: int = 10_000,
        global_rope_theta: int = 1_000_000,
        global_layer_interval: int = 6,
        layer_norm_eps: float = 1e-6,
        fused_ops: bool = False,
        use_flash: Optional[bool] = None,
        attn_backend: Optional[AttentionBackendName] = None,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Gemma 3-like model configuration.

        Gemma 3 features:
        - Hybrid local/global attention: 5 local layers with sliding window, then 1 global layer
        - Dual RoPE frequencies: local layers use 10K, global layers use 1M
        - QK-norm for attention score stabilization
        - GeGLU activation (GELU with tanh approximation)

        :param local_window_size: Sliding window size for local attention layers.
        :param local_rope_theta: RoPE base frequency for local attention layers.
        :param global_rope_theta: RoPE base frequency for global attention layers.
        :param global_layer_interval: Number of layers per pattern cycle (default 6 = 5 local + 1 global).
        """
        layer_norm = LayerNormConfig(
            name=LayerNormType.fused_rms if fused_ops else LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        local_block = TransformerBlockConfig(
            name=TransformerBlockType.peri_norm,
            sequence_mixer=AttentionConfig(
                name=AttentionType.default,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                bias=False,
                rope=RoPEConfig(name=RoPEType.default, theta=local_rope_theta),
                gate=gate,
                qk_norm=layer_norm,
                use_head_qk_norm=True,
                use_flash=use_flash,
                backend=attn_backend,
                sliding_window=SlidingWindowAttentionConfig(
                    pattern=[local_window_size],  # Always apply SWA on local_block
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=False,
                ),
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                hidden_size=hidden_size,
                bias=False,
                dtype=dtype,
                activation=activation,
            ),
            layer_norm=layer_norm,
        )

        global_block = local_block.copy()
        sequence_mixer = cast(AttentionConfig, global_block.sequence_mixer.copy())
        sequence_mixer.rope = RoPEConfig(name=RoPEType.default, theta=global_rope_theta)
        sequence_mixer.sliding_window = None
        global_block.sequence_mixer = sequence_mixer

        blocks = {"local": local_block, "global": global_block}
        block_pattern = ["local"] * (global_layer_interval - 1) + ["global"]

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=blocks,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_pattern=block_pattern,
            embed_scale=math.sqrt(d_model),
            **kwargs,
        )

    def with_rope_scaling(
        self, rope_scaling: RoPEScalingConfig, full_attn_layers_only: bool = True
    ) -> "TransformerConfig":
        """
        Return a copy of this config with the given RoPE scaling scheme applied.
        """
        new_config = self.copy()
        if isinstance(new_config.block, dict):
            raise OLMoConfigurationError(
                "Cannot use `with_rope_scaling` with a hybrid model with named blocks."
            )
        assert isinstance(
            new_config.block.sequence_mixer, AttentionConfig
        ), "Sequence mixer must be an attention config for RoPE scaling"
        if new_config.block.sequence_mixer.rope is None:
            raise ValueError("Cannot apply RoPE scaling to a model without RoPE.")
        if new_config.block_overrides:
            raise ValueError("Cannot apply RoPE scaling when block_overrides are already set.")

        def apply_scaling(block_config: TransformerBlockConfig) -> None:
            assert isinstance(block_config.sequence_mixer, AttentionConfig)
            rope_config = block_config.sequence_mixer.rope
            if rope_config is None:
                raise ValueError("Cannot apply RoPE scaling to a layer without RoPE.")
            rope_config = rope_config.copy()
            rope_config.scaling = rope_scaling
            block_config.sequence_mixer.rope = rope_config

        if not full_attn_layers_only:
            apply_scaling(new_config.block)
            return new_config

        # Add rope scaling only to layers that do not use sliding window attention
        # We supply "block_overrides" for the layers we want to scale.
        overrides: Dict[int, TransformerBlockConfig] = {}
        for i in range(new_config.n_layers):
            sliding_window_cfg = new_config.block.sequence_mixer.sliding_window
            if sliding_window_cfg and sliding_window_cfg.should_use_swa(i, new_config.n_layers):
                continue
            block_copy = new_config.block.copy()
            apply_scaling(block_copy)
            overrides[i] = block_copy

        new_config.block_overrides = overrides or None
        return new_config


def validate_block_resolution_config(
    n_layers: int,
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
    block_pattern: list[str] | None = None,
    block_overrides: dict[int, TransformerBlockConfig] | None = None,
) -> None:
    if not isinstance(block, dict):
        if block_pattern is not None:
            raise OLMoConfigurationError(
                "`block_pattern` is not supported when `block` is not a dict of named blocks."
            )
        return

    if not block_pattern:
        raise OLMoConfigurationError(
            "`block_pattern` must be provided and non-empty when `block` is a dict of named blocks."
        )
    if block_overrides is not None:
        raise OLMoConfigurationError(
            "`block_overrides` is not supported when `block` is a dict of named blocks; "
            "use `block_pattern` to control per-layer block selection."
        )

    available_block_names = set(block.keys())
    missing_block_names = set(block_pattern) - available_block_names
    if missing_block_names:
        raise OLMoConfigurationError(
            "Every name in `block_pattern` must exist in `block`. "
            f"Unknown names: {missing_block_names}. Available names: {available_block_names}."
        )


def resolve_block_configs(
    n_layers: int,
    block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
    block_pattern: list[str] | None = None,
    block_overrides: dict[int, TransformerBlockConfig] | None = None,
) -> list[TransformerBlockConfig]:
    """Resolve the block configuration for each layer."""
    validate_block_resolution_config(
        n_layers=n_layers,
        block=block,
        block_pattern=block_pattern,
        block_overrides=block_overrides,
    )

    block_configs: list[TransformerBlockConfig]
    if isinstance(block, dict):
        # Named-block configuration.
        assert block_pattern is not None
        assert block_overrides is None
        full_pattern = list(islice(cycle(block_pattern), n_layers))
        block_configs = [block[name] for name in full_pattern]
    else:
        # Single-block with manual override configuration.
        assert block_pattern is None
        block_configs = [block] * n_layers
        if block_overrides is not None:
            for block_idx, override in block_overrides.items():
                block_configs[block_idx] = override

    assert len(block_configs) == n_layers
    return block_configs
