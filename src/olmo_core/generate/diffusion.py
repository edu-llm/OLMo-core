"""
Block-wise decoding for a masked diffusion language model, and ES-dLLM early-skipping on top.

TWO SEPARATE THINGS LIVE HERE AND THE DISTINCTION MATTERS. :class:`DiffusionSampler` is how a
masked diffusion model generates at all: there is no next-token loop, so nothing in
:mod:`olmo_core.generate.sampling` applies. :class:`EarlySkippingPolicy` is
`ES-dLLM <https://openreview.net/forum?id=O2WvMkJbws>`_ (ICLR 2026), which is a **training-free**
accelerator layered over that sampler -- it changes how much of the network each denoising
iteration touches and nothing about the model or the objective. A diffusion model trained without
ever hearing of ES-dLLM is exactly what it expects to be given.

HOW THE SAMPLER WORKS. A canvas of ``MASK`` follows the prompt and is denoised in blocks, which is
the arrangement DiffusionGemma calls block-autoregressive multi-canvas sampling and Fast-dLLM and
BD3-LM reach for as well. Within a block, each iteration runs the model over the whole sequence,
reads the logits at that block's still-masked positions, and commits the most confident of them --
`LLaDA <https://arxiv.org/abs/2502.09992>`_'s low-confidence remasking, kept because the
alternative (commit everything, once) is what
`the ICLR-2026 experimental analysis <https://arxiv.org/html/2606.19475v1>`_ measures degrading
monotonically as steps fall, with no saturation up to 1024 steps.

WHAT ES-dLLM SKIPS, AND WHAT IT CANNOT SKIP HERE. The paper's observation is that key, value and
hidden states change only subtly between successive iterations, so shallow layers may skip
unimportant tokens and reuse the previous iteration's result. Its importance signal is the
variation of those intermediate tensors combined with the previous iteration's confidence, and it
refreshes the cache periodically so error cannot accumulate. All of that transfers. What does not
is the assumption that every layer is attention:

* **Feed-forward and MoE sublayers are position-independent.** Skipping a token there is exact --
  the arithmetic for position *i* never consults position *j*. This is valid on **every** block,
  recurrent ones included, and on this architecture it is where most of the skippable work is.
* **Attention is skippable on the query side only.** A skipped position must still contribute its
  key and value, or the positions that did not skip would attend to a shorter sequence.
* **A recurrent scan cannot be skipped at all.** :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet2`
  advances a state along the sequence, so position *i*'s output depends on every position before
  it; dropping one does not save that position's work, it changes every later position's answer.

On the 3:1 hybrid this branch trains, that means early-skipping reaches the FFN of all 16 blocks
and the query side of 4, and none of the 12 recurrent scans. **So this will not reproduce the
paper's 226/308 tokens per second on LLaDA and Dream, which are pure-attention models.** The
offsetting gain is that the backbone is already linear-time. The two are not multiplicative and
:meth:`EarlySkippingPolicy.describe_budget` reports the split rather than a product, because a
product is the number somebody would otherwise quote.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)

__all__ = [
    "RemaskingStrategy",
    "DiffusionSamplingConfig",
    "EarlySkippingConfig",
    "EarlySkippingPolicy",
    "block_schedule",
    "commit_counts",
    "select_commits",
]


class RemaskingStrategy(StrEnum):
    """
    How an iteration chooses which of a block's masked positions to commit.
    """

    low_confidence = "low_confidence"
    """
    Commit the most confident positions and leave the rest masked, which is LLaDA's strategy and
    the default. "Low confidence" names what gets *re*-masked, which is the convention the
    literature uses and reads backwards at first.
    """
    random = "random"
    """
    Commit a random subset. Only useful as the control that shows confidence ordering is doing
    something.
    """


@dataclass
class DiffusionSamplingConfig(Config):
    """
    Configuration for :class:`DiffusionSampler`.
    """

    mask_token_id: int
    """
    The ``MASK`` id the model was trained with. Getting this wrong does not raise: the model is
    handed a sequence of some ordinary token and asked to denoise it, and it will produce fluent
    continuations that ignore the canvas.
    """
    max_new_tokens: int = 256
    """
    Length of the canvas. DiffusionGemma denoises 256 at a time.
    """
    block_length: int = 32
    """
    Tokens per block. Committed blocks are fixed, so this trades parallelism against how much
    context each decision has. The experimental analysis finds block size nearly free to tune
    under fixed compute, with model rankings unchanged, so it is not a sensitive knob -- except
    that DiffuMamba finds the best block size differs by architecture under caching.
    """
    steps_per_block: int = 8
    """
    Denoising iterations per block. With ``block_length`` 32 this commits ~4 tokens per forward
    pass. Fewer steps is faster and monotonically worse on reasoning and code.
    """
    remasking: RemaskingStrategy = RemaskingStrategy.low_confidence
    """
    See :class:`RemaskingStrategy`.
    """
    temperature: float = 0.0
    """
    0.0 is greedy per position. Note that a base diffusion model may repeat under greedy decoding
    -- RND1's own release says so of itself -- which is a property of the checkpoint and not of
    this sampler.
    """

    def __post_init__(self):
        if self.block_length <= 0 or self.max_new_tokens <= 0 or self.steps_per_block <= 0:
            raise OLMoConfigurationError(
                "block_length, max_new_tokens and steps_per_block must all be positive"
            )
        if self.max_new_tokens % self.block_length != 0:
            raise OLMoConfigurationError(
                f"max_new_tokens ({self.max_new_tokens}) must be a multiple of block_length "
                f"({self.block_length}), otherwise the last block is a different shape than the "
                "ones whose behaviour was tuned"
            )
        if self.steps_per_block > self.block_length:
            raise OLMoConfigurationError(
                f"steps_per_block ({self.steps_per_block}) exceeds block_length "
                f"({self.block_length}), so some iterations would have nothing left to commit"
            )


def block_schedule(prompt_len: int, config: DiffusionSamplingConfig) -> List[Tuple[int, int]]:
    """
    The ``(start, end)`` span of each block, in absolute sequence positions.

    :param prompt_len: How many tokens of prompt precede the canvas.
    :param config: The sampling configuration.

    :returns: One half-open span per block, covering the canvas exactly once.
    """
    n_blocks = config.max_new_tokens // config.block_length
    return [
        (
            prompt_len + i * config.block_length,
            prompt_len + (i + 1) * config.block_length,
        )
        for i in range(n_blocks)
    ]


def commit_counts(block_length: int, steps: int) -> List[int]:
    """
    How many positions each iteration of a block commits.

    Spread as evenly as integers allow, with the remainder on the earliest iterations rather than
    the latest. Front-loading is deliberate: the last iterations are the ones deciding between the
    hardest remaining positions, and they should have the most context and the fewest choices.

    :param block_length: Positions in the block.
    :param steps: Iterations to spend on it.

    :returns: A list of length ``steps`` summing to ``block_length``.
    """
    base, remainder = divmod(block_length, steps)
    return [base + (1 if i < remainder else 0) for i in range(steps)]


def select_commits(
    logits: torch.Tensor,
    still_masked: torch.Tensor,
    n_commit: int,
    *,
    strategy: RemaskingStrategy = RemaskingStrategy.low_confidence,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Choose which masked positions to commit this iteration, and to what.

    :param logits: Logits over the block, shape ``(batch_size, block_length, vocab_size)``.
    :param still_masked: Which of those positions are unresolved, shape
        ``(batch_size, block_length)``.
    :param n_commit: How many positions to commit per row.
    :param strategy: See :class:`RemaskingStrategy`.
    :param temperature: 0.0 selects the argmax per position.
    :param generator: Optional generator, so a test can fix the draw.

    :returns: ``(commit_mask, token_ids, confidence)``. ``commit_mask`` is ``True`` at the chosen
        positions, ``token_ids`` holds the selection at every position (only meaningful where
        committed), and ``confidence`` is the per-position probability of the selected token,
        which the next iteration reads as part of ES-dLLM's importance signal.
    """
    probs = F.softmax(logits.float(), dim=-1)
    if temperature > 0.0:
        # Gumbel-max, which samples from softmax(logits / T) without materialising it.
        noise = torch.rand_like(probs, dtype=torch.float32)
        if generator is not None:
            noise = torch.rand(probs.shape, generator=generator, device=probs.device)
        gumbel = -torch.log(-torch.log(noise.clamp_min(1e-20)).clamp_min(1e-20))
        token_ids = ((logits.float() / temperature) + gumbel).argmax(dim=-1)
    else:
        token_ids = probs.argmax(dim=-1)

    confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    if strategy == RemaskingStrategy.random:
        score = torch.rand(confidence.shape, generator=generator, device=confidence.device)
    else:
        score = confidence

    # Resolved positions must never be reselected, and -inf keeps them out of topk regardless of
    # how many positions are still masked.
    score = score.masked_fill(~still_masked, float("-inf"))

    n_commit = min(n_commit, int(still_masked.sum(dim=-1).max().item()) or 1)
    chosen = score.topk(n_commit, dim=-1).indices
    commit_mask = torch.zeros_like(still_masked)
    commit_mask.scatter_(-1, chosen, True)
    # A row with fewer remaining positions than `n_commit` would otherwise pick up -inf entries.
    commit_mask &= still_masked

    return commit_mask, token_ids, confidence


@dataclass
class EarlySkippingConfig(Config):
    """
    Configuration for ES-dLLM early-skipping.

    Off by default. Every field here trades exactness for speed, and the honest baseline for any
    speed claim is a run with ``enabled=False``.
    """

    enabled: bool = False
    """
    Whether to skip anything at all.
    """
    skip_layers: int = 8
    """
    How many of the shallowest blocks may skip. ES-dLLM skips in early layers because that is
    where its measurement found intermediate states most stable between iterations.
    """
    skip_fraction: float = 0.5
    """
    Fraction of the *unimportant* positions to skip in those layers.
    """
    refresh_interval: int = 4
    """
    Run a full, unskipped iteration every this many iterations.

    This is the paper's guard against error accumulation and it is not optional: reused states
    drift, and the drift compounds silently because a diffusion sampler has no per-step target to
    check against.
    """
    variation_weight: float = 0.5
    """
    Weight on intermediate-state variation against previous-iteration confidence when ranking
    importance. The paper builds importance from both; 0.5 weights them equally.
    """

    def __post_init__(self):
        if not 0.0 <= self.skip_fraction < 1.0:
            raise OLMoConfigurationError(
                f"skip_fraction must be in [0, 1), got {self.skip_fraction}"
            )
        if self.refresh_interval < 1:
            raise OLMoConfigurationError("refresh_interval must be at least 1")
        if self.skip_layers < 0:
            raise OLMoConfigurationError("skip_layers must not be negative")


@dataclass
class EarlySkippingPolicy:
    """
    The decision half of ES-dLLM: which positions may be skipped, in which layers, this iteration.

    Kept separate from any model surgery so that the ranking, the refresh schedule and the budget
    accounting can be tested without a GPU, a checkpoint or ``flash-linear-attention``.

    :param config: See :class:`EarlySkippingConfig`.
    :param n_layers: Total blocks in the model.
    :param recurrent_layers: Indices of blocks whose sequence mixer is a recurrence. Their scans
        can never be skipped -- see this module's docstring -- so the policy reports them
        separately rather than pretending otherwise.
    """

    config: EarlySkippingConfig
    n_layers: int
    recurrent_layers: Tuple[int, ...] = ()
    _previous_hidden: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    _iteration: int = 0

    def reset(self) -> None:
        """Forget the previous iteration's states, at the start of a new block or sequence."""
        self._previous_hidden.clear()
        self._iteration = 0

    def is_refresh_iteration(self) -> bool:
        """Whether this iteration must run unskipped."""
        return self._iteration % self.config.refresh_interval == 0

    def skippable_layers(self) -> Tuple[int, ...]:
        """Which blocks may skip this iteration."""
        if not self.config.enabled or self.is_refresh_iteration():
            return ()
        return tuple(range(min(self.config.skip_layers, self.n_layers)))

    def importance(
        self, layer_idx: int, hidden: torch.Tensor, confidence: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Rank positions by how much this iteration needs to recompute them.

        :param layer_idx: Which block's hidden states these are.
        :param hidden: Hidden states, shape ``(batch_size, seq_len, d_model)``.
        :param confidence: The previous iteration's per-position confidence, shape
            ``(batch_size, seq_len)``, or ``None`` on the first iteration.

        :returns: Importance per position, shape ``(batch_size, seq_len)``, higher meaning
            "recompute this".
        """
        previous = self._previous_hidden.get(layer_idx)
        if previous is None or previous.shape != hidden.shape:
            # Nothing to compare against, so everything is maximally important. This is what makes
            # the first iteration after a reset implicitly a full one.
            variation = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
        else:
            variation = (hidden - previous).float().norm(dim=-1)
            largest = variation.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            variation = variation / largest

        if confidence is None:
            return variation

        # An uncertain position is one whose answer is still moving, so it is important. High
        # confidence is what licenses reuse.
        uncertainty = (1.0 - confidence.float()).clamp(0.0, 1.0)
        w = self.config.variation_weight
        return w * variation + (1.0 - w) * uncertainty

    def positions_to_skip(
        self, importance: torch.Tensor, *, protected: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Turn an importance ranking into a skip mask.

        :param importance: From :meth:`importance`, shape ``(batch_size, seq_len)``.
        :param protected: Positions that must never be skipped, shape ``(batch_size, seq_len)``.
            The current block is always protected by the caller: those are the positions being
            decided this iteration, and reusing a stale state for them is reusing the answer.

        :returns: ``True`` where a position may be skipped.
        """
        seq_len = importance.shape[1]
        n_skip = int(seq_len * self.config.skip_fraction)
        if n_skip == 0:
            return torch.zeros_like(importance, dtype=torch.bool)

        score = importance.clone()
        if protected is not None:
            score = score.masked_fill(protected, float("inf"))

        lowest = score.topk(n_skip, dim=-1, largest=False).indices
        skip = torch.zeros_like(importance, dtype=torch.bool)
        skip.scatter_(-1, lowest, True)
        if protected is not None:
            skip &= ~protected
        return skip

    def record(self, layer_idx: int, hidden: torch.Tensor) -> None:
        """Remember a block's output so the next iteration can measure variation against it."""
        self._previous_hidden[layer_idx] = hidden.detach().clone()

    def advance(self) -> None:
        """Move to the next denoising iteration."""
        self._iteration += 1

    def describe_budget(self) -> Dict[str, float]:
        """
        What fraction of each kind of work early-skipping can reach, on this architecture.

        Reported as a split rather than one number on purpose. A single "speedup" over a hybrid
        backbone invites multiplying ES-dLLM's gain by the linear-attention gain, and the two are
        not multiplicative: the recurrent scans that make the backbone cheap are exactly the part
        early-skipping cannot touch.

        :returns: The share of blocks whose feed-forward is skippable, the share whose sequence
            mixer is query-skippable, and the share whose mixer is a scan and is not skippable.
        """
        n = max(self.n_layers, 1)
        skippable = self.skippable_layers()
        recurrent_in_skip = [i for i in skippable if i in self.recurrent_layers]
        return {
            # Position-independent, so exact wherever it is applied.
            "feed_forward_skippable": len(skippable) / n,
            # Query side only; keys and values are still needed by the positions that did not skip.
            "attention_query_skippable": (len(skippable) - len(recurrent_in_skip)) / n,
            # The scans. Not skippable at any fraction.
            "recurrent_scan_unskippable": len(self.recurrent_layers) / n,
        }
