"""Train a masked-diffusion language model on a published eduLLM corpus.

    python .edullm/train_diffusion.py "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR" [FLAGS...]

WHAT THIS IS. `train_gdn2.py` with three changes, and nothing else: the sequence mixers are laid
out as a 3:1 hybrid of bidirectional GDN-2 and non-causal attention instead of GDN-2 everywhere,
the train module is the masked-diffusion one instead of the autoregressive one, and the model can
be initialised from an autoregressive checkpoint. Everything `train_on_corpus.py` earned -- corpus
resolution through the reader, the dtype assertions, the precision refusal that exits before a T4
is billed, the torn-checkpoint repair a Batch retry needs, the stage-numbered exit codes that are
the only thing visible from outside a dead container -- and everything `train_gdn2.py` added on
top of it -- the GDN-2 geometry flags, the source mixture -- is inherited rather than re-typed.

WHY A HYBRID AND NOT GDN-2 EVERYWHERE, WHICH IS THE LARGEST DECISION IN THIS FILE. A causal
recurrence cannot carry a masked diffusion objective. DeltaFlow (https://arxiv.org/abs/2608.01240)
measures unidirectional GDN under a diffusion objective and reports *entropy collapse*: the model
trains, the loss descends, and what comes out generates degenerately. So every GDN-2 block here is
bidirectional, in DeltaFlow's *alternating scan* arrangement -- one scan per layer, direction
flipping from layer to layer, which costs nothing over the causal layer because it is still one
scan. The alternation leaves each individual layer one-directional, and what corrects that is the
attention every fourth block, which DeltaFlow describes as giving "exact bidirectional
correction". Their own stack is [GDN, GDN, GDN, Attn] x 3 and this is the same 3:1 at 16 layers.

Two independent reasons not to make it DeltaFlow's *parallel* variant, which runs both directions
in every layer and reports the better perplexity (21.2 vs 24.7). It costs ~40% more memory, and
this shape is `gpu-8xa100` with 40 GB cards where the loss path already pins
`--rank-microbatch-size 8192`. Lowering that microbatch to buy the second scan would not merely be
slower: `expert_capacity` derives from `max_local_microbatch_size`, so for a MoE it changes the
MODEL, and the comparison against the autoregressive baseline stops being a comparison. The
parallel variant is the follow-up on a shape with more memory per card, and `gpu-8xh100` is
documented unobtainable in this account.

WHAT THE SCALE RISK IS, STATED HERE BECAUSE A NEGATIVE RESULT SHOULD NOT BE A SURPRISE. DiffuMamba
(https://arxiv.org/abs/2511.15927) finds linear-attention diffusion *losing* to plain attention at
240M -- its Mamba variants "struggle to generalize effectively" at that size -- and only winning
from 0.5B up. 370M is inside that gap. The mitigating evidence is that DeltaFlow's own models were
104-110M parameters in this exact configuration (3:1 hybrid, noise-adaptive gates) and did not
collapse, so the arrangement is attested below this scale even though the pure-recurrence one is
not. If this run underperforms the baseline, that is a result and not necessarily a bug.

WHAT IS MISSING, NAMED HERE RATHER THAN LEFT TO BE DISCOVERED. Every frontier diffusion LM is
converted from an autoregressive checkpoint rather than trained from scratch -- RND1, Dream and
DiffusionGemma all are -- because it reuses the tokens the AR run already paid for, and that is by
some distance the largest efficiency lever available to this experiment. `--init-from` is where it
would go and it is NOT IMPLEMENTED: it refuses. Writing it needs the source checkpoint's own key
names, to map one dense feed-forward onto 32 expert rows and to decide what happens to the four
layers that stop being GDN-2, and this repository may not read the bucket that checkpoint lives in.
A state dict whose keys silently do not match loads nothing and trains to completion looking like
the run it was meant to improve on, so this refuses rather than warns. `edullm run` puts a shell on
a machine that may read it; that is the first step, not more code here.
"""

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

# Both files live in `.edullm/`, which is not a package. Insert the directory so that the
# siblings import here and also re-import cleanly in a dataloader worker.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_gdn2 as gdn2  # noqa: E402
import train_on_corpus as toc  # noqa: E402

from olmo_core.nn.attention import AttentionConfig, GatedDeltaNet2Config  # noqa: E402
from olmo_core.nn.moe import MoEConfig, MoERouterConfig, MoEType  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerBlockConfig  # noqa: E402
from olmo_core.train.train_module.transformer import (  # noqa: E402
    DiffusionSchedule,
    DiffusionTransformerTrainModuleConfig,
    MaskedDiffusionConfig,
)

log = toc.log

#: How many blocks per repeat, and which one of them is attention. 4 and "the last" is DeltaFlow's
#: [GDN, GDN, GDN, Attn], and also the linear-to-full ratio Qwen3-Next and Kimi Linear ship.
HYBRID_PERIOD = 4


def hybrid_block_overrides(
    model_config: TransformerConfig,
    mixer: GatedDeltaNet2Config,
    *,
    period: int = HYBRID_PERIOD,
    noise_conditioned: bool = True,
) -> Tuple[Dict[int, TransformerBlockConfig], int, int]:
    """Lay out the 3:1 hybrid as one ``block_overrides`` entry per layer.

    Every layer gets an explicit entry rather than only the ones that differ from the shared
    ``block``. It is more lines in the saved config and it is the point: which layer got which
    mixer, and which direction it scans, is then a fact in the record instead of something a
    reader has to re-derive from a period and an off-by-one.

    :param model_config: The factory's config, read for ``n_layers`` and its block template.
    :param mixer: The GDN-2 geometry to use, from ``train_gdn2.mixer_config``.
    :param period: Layers per repeat. The last layer of each repeat is the attention one.
    :param noise_conditioned: Whether the GDN-2 layers read the diffusion noise level.

    :returns: ``(block_overrides, n_recurrent, n_attention)``.

    :raises Refusal: If the factory exposes no single block config to build the overrides from,
        or if ``n_layers`` is not a multiple of ``period``.
    """
    block = model_config.block
    if not isinstance(block, TransformerBlockConfig):
        # A dict of named blocks is the shape `block_overrides` is documented not to support.
        raise toc.Refusal(
            toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "this model factory exposes a dict of named blocks rather than one block config, so "
            "the hybrid schedule cannot be expressed as block_overrides",
        )

    n_layers = model_config.n_layers
    if n_layers % period != 0:
        # Not fatal in principle, but it would silently give the last repeat a different shape,
        # and an unbalanced forward/reverse split is a confound nobody would look for.
        raise toc.Refusal(
            toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"n_layers ({n_layers}) is not a multiple of the hybrid period ({period}), so the "
            "last repeat would have a different mixer layout than the others",
        )

    overrides: Dict[int, TransformerBlockConfig] = {}
    recurrent = attention = 0
    for layer_idx in range(n_layers):
        entry = block.copy()
        if layer_idx % period == period - 1:
            # The attention block. Non-causal, because in a diffusion model this is what gives
            # every position a view of the whole partially-masked sequence; it is also the only
            # block that carries a KV cache, which is what an early-skipping decoder skips
            # against.
            assert isinstance(entry.sequence_mixer, AttentionConfig), type(entry.sequence_mixer)
            attn = entry.sequence_mixer.copy()
            attn.causal = False
            entry.sequence_mixer = attn
            attention += 1
        else:
            layer_mixer = mixer.copy()
            # Alternate direction across the recurrent layers, counting only those -- counting
            # over all layers would put every reverse scan on the same side of the attention
            # blocks and the two directions would not be balanced.
            layer_mixer.reverse_scan = recurrent % 2 == 1
            layer_mixer.noise_conditioned = noise_conditioned
            entry.sequence_mixer = layer_mixer
            recurrent += 1
        overrides[layer_idx] = entry

    return overrides, recurrent, attention


def moe_config(opts) -> Optional[MoEConfig]:
    """The MoE shape, or ``None`` to leave the factory's own.

    ``--moe-top-k`` defaults above ``olmo2_370M_moe``'s own 4 on purpose; see the flag's help for
    the active-parameter arithmetic that sets it.
    """
    return MoEConfig(
        name=MoEType.default,
        num_experts=opts.moe_num_experts,
        hidden_size=opts.moe_hidden_size,
        router=MoERouterConfig(top_k=opts.moe_top_k),
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
    )


@contextmanager
def factory_with_hybrid(
    factory_name: str,
    mixer: GatedDeltaNet2Config,
    *,
    noise_conditioned: bool = True,
    moe: Optional[MoEConfig] = None,
) -> Iterator[None]:
    """Temporarily wrap one ``TransformerConfig`` factory so its blocks come back as the hybrid.

    Wrapping the factory rather than post-processing the built config is deliberate, for the
    reason `train_gdn2.py` gives: ``train_on_corpus.build_config`` ends with
    ``config.merge(overrides)``, and a dotted override naming a GDN-2 field is only valid once the
    mixer is already a ``GatedDeltaNet2Config``.
    """
    original = getattr(TransformerConfig, factory_name, None)
    if original is None:
        # Left to train_on_corpus, which raises this as a staged Refusal with an exit code.
        yield
        return

    def wrapped(**kwargs):
        if moe is not None:
            # Passed into the factory rather than edited onto the built config, because
            # `olmo2_370M_moe` derives the expert hidden size from the `d_model` it also chooses.
            kwargs.setdefault("feed_forward_moe", moe)
        model_config = original(**kwargs)
        overrides, recurrent, attention = hybrid_block_overrides(
            model_config, mixer, noise_conditioned=noise_conditioned
        )
        if model_config.block_overrides:
            raise toc.Refusal(
                toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"model factory {factory_name} already sets block_overrides, which the hybrid "
                "schedule would overwrite",
            )
        model_config.block_overrides = overrides
        log.info(
            "hybrid mixer schedule: %d bidirectional GDN-2 blocks (%d forward, %d reverse) and "
            "%d non-causal attention blocks over %d layers",
            recurrent,
            (recurrent + 1) // 2,
            recurrent // 2,
            attention,
            model_config.n_layers,
        )
        return model_config

    setattr(TransformerConfig, factory_name, wrapped)
    try:
        yield
    finally:
        setattr(TransformerConfig, factory_name, original)


@contextmanager
def train_module_as_diffusion(diffusion: MaskedDiffusionConfig) -> Iterator[None]:
    """Temporarily make ``train_on_corpus``'s train-module config the diffusion one.

    ``train_on_corpus.build_config`` names ``TransformerTrainModuleConfig`` off its own module, so
    rebinding that attribute puts the diffusion train module inside the whole of its existing
    configuration -- FSDP, the scheduler, the grad-norm clip, the precision refusal -- without
    this file restating any of it.
    """
    original = toc.TransformerTrainModuleConfig

    def wrapped(**kwargs):
        return DiffusionTransformerTrainModuleConfig(diffusion=diffusion, **kwargs)

    toc.TransformerTrainModuleConfig = wrapped  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        toc.TransformerTrainModuleConfig = original  # type: ignore[assignment,misc]


def diffusion_config(opts, tokenizer) -> MaskedDiffusionConfig:
    """Build the corruption config, defaulting the mask id to the first free vocabulary slot.

    ``padded_vocab_size`` rounds up to a multiple of 128, so the ids from ``vocab_size`` to the
    padded size are already allocated and unreachable by the tokenizer. Taking the first of them
    means the embedding matrix does not have to grow, which in turn means an autoregressive
    checkpoint of the same architecture loads without a shape mismatch.
    """
    mask_token_id = opts.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.vocab_size
        padded = tokenizer.padded_vocab_size()
        if mask_token_id >= padded:
            raise toc.Refusal(
                toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"this tokenizer's vocab_size ({mask_token_id}) leaves no free slot below its "
                f"padded size ({padded}), so --mask-token-id has to name one explicitly",
            )
        log.info(
            "MASK token id %d, taken from the padding between vocab_size %d and padded size %d",
            mask_token_id,
            tokenizer.vocab_size,
            padded,
        )

    return MaskedDiffusionConfig(
        mask_token_id=mask_token_id,
        schedule=DiffusionSchedule(opts.noise_schedule),
        min_mask_probability=opts.min_mask_probability,
        antithetic_sampling=opts.antithetic_sampling,
    )


def build_parser() -> argparse.ArgumentParser:
    """`train_gdn2`'s parser -- so the GDN-2 geometry and the mixture -- plus the diffusion flags."""
    parser = gdn2.build_parser()
    parser.prog = "train_diffusion"
    parser.description = (
        "Train a masked-diffusion language model with a bidirectional GDN-2 / attention hybrid "
        "on a published eduLLM corpus."
    )

    group = parser.add_argument_group("masked diffusion")
    group.add_argument(
        "--noise-schedule",
        choices=[str(s) for s in DiffusionSchedule],
        default=str(DiffusionSchedule.linear),
        help="Maps a uniform draw to a per-sequence masking probability. linear is Quokka's "
        "strongest and lowest-variance choice and cosine its worst, so this is not a knob to "
        "turn without a reason.",
    )
    group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Token id written at masked positions. Defaults to the tokenizer's vocab_size, "
        "which is the first of the already-allocated slots between it and the padded size, so "
        "the embedding matrix does not grow.",
    )
    group.add_argument(
        "--min-mask-probability",
        type=float,
        default=1e-3,
        help="Lower clamp on the masking probability. A sequence drawn at zero contributes no "
        "loss and wastes its whole forward and backward pass; DiffuMamba uses this same floor.",
    )
    group.add_argument(
        "--antithetic-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the second half of the batch's noise levels as 1 - t of the first half. Free "
        "variance reduction on a draw that is otherwise a real part of the gradient noise.",
    )
    group.add_argument(
        "--noise-conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the GDN-2 decay and gate logits read the noise level, through zero-initialised "
        "per-head projections. DeltaFlow finds the bidirectional core alone still produces "
        "overly concentrated generations, so switching this off is switching off the half of "
        "their method that restores diversity.",
    )
    moe = parser.add_argument_group("MoE active-parameter match")
    moe.add_argument(
        "--moe-top-k",
        type=int,
        default=8,
        help="Experts activated per token. 8 rather than olmo2_370M_moe's own 4, and this is a "
        "science parameter: at top_k=4 this arm has 299.6M active non-embedding parameters "
        "against the AR baseline's 409.2M, so a loss gap would be mostly the 27%% size "
        "difference rather than the objective. 8 lands at 400.3M, within 2.2%%. Total parameters "
        "are unchanged either way, so MuonH's per-expert blocking is unaffected.",
    )
    moe.add_argument(
        "--moe-num-experts",
        type=int,
        default=32,
        help="Experts per layer. 32 is what olmo2_370M_moe defines and what the MuonH work on "
        "this repository measured against; changing it changes total parameters.",
    )
    moe.add_argument(
        "--moe-hidden-size",
        type=int,
        default=512,
        help="Hidden size of each expert. 512 is olmo2_370M_moe's own (0.5 * d_model).",
    )

    group.add_argument(
        "--init-from",
        default=None,
        help="NOT IMPLEMENTED, and refused rather than ignored. Naming an autoregressive "
        "checkpoint to convert from is the right thing to want -- see this file's header -- but "
        "the remapping cannot be written without first reading the checkpoint's own keys, and "
        "nothing on a laptop may read that bucket. Use `edullm run` to inspect one, then "
        "implement the mapping this flag would use.",
    )

    return parser


def build_config(opts, overrides: List[str]):
    """`train_on_corpus.build_config`, with the factory and the train module wrapped."""
    gdn2._MIXTURE_OPTS = opts

    if opts.init_from is not None:
        # Refused loudly instead of warned about, because the failure mode of a half-written
        # conversion is the expensive one: a state dict whose keys do not match loads nothing,
        # every parameter stays at its random initialisation, and the run trains to completion
        # looking exactly like the from-scratch run it was supposed to improve on.
        #
        # THREE REMAPPINGS ARE NEEDED AND NONE OF THEM IS GUESSABLE FROM HERE.
        #   1. dense -> MoE. The baseline's feed-forward is one MLP; this model's is 32 experts
        #      stored with the expert dimension folded into rows. Upcycling means broadcasting
        #      that MLP into every expert row.
        #   2. all-GDN-2 -> hybrid. The baseline is GDN-2 in all 16 layers. Here layers 3, 7, 11
        #      and 15 are attention, so there is nothing to carry into them and they start random
        #      while their neighbours start trained -- which is a real effect on the result, not
        #      a detail.
        #   3. causal -> bidirectional GDN-2. Sound in principle, since a reversed scan shares
        #      every parameter with the forward one, and untested.
        #
        # What blocks writing it is 1 and 2 needing the checkpoint's actual key names, and
        # AGENTS.md forbidding this repository from reading S3. `edullm run` puts a shell on a
        # machine that may; inspect the checkpoint there first.
        raise toc.Refusal(
            toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "--init-from is not implemented. AR-to-diffusion conversion needs a state-dict "
            "remapping (dense feed-forward to upcycled experts, all-GDN-2 to the hybrid layout) "
            "that cannot be written without reading the source checkpoint's keys, which this "
            "repository may not do. Train from scratch, or implement the mapping first.",
        )

    # The tokenizer is resolved twice -- once here for the MASK id, once inside the base
    # build_config -- rather than threaded, because the base function owns corpus resolution and
    # reaching into it to borrow a half-built value is the more fragile of the two couplings.
    corpus = gdn2.resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )

    with factory_with_hybrid(
        opts.model_factory,
        gdn2.mixer_config(opts),
        noise_conditioned=opts.noise_conditioned,
        moe=moe_config(opts),
    ):
        with train_module_as_diffusion(diffusion_config(opts, corpus.tokenizer)):
            config = gdn2._base_build_config(opts, overrides)

    # Context parallelism reshapes the sequence across ranks, which is exactly what a reversed
    # scan needs whole. The layer raises on this too; refusing at config time means it costs a
    # second rather than a machine.
    if getattr(config.train_module, "cp_config", None) is not None:
        raise toc.Refusal(
            toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "context parallelism is not supported with a reversed recurrent scan: each rank "
            "holds a slice of the sequence, so reversing within the slice is not reversing "
            "within the document",
        )

    return config


_base_build_parser = gdn2.build_parser
_base_build_config = gdn2.build_config

# The seams, re-pointed from `train_gdn2`'s to this file's. `toc.main` and `toc.build_config` call
# these by name off their own module, so patching the module attributes is what puts this file's
# behaviour inside the whole of train_on_corpus's error handling, precision refusal, dry-run path
# and exit-code contract. `resolve_corpus` and `show` are left as train_gdn2 set them: this file
# changes neither corpus resolution nor what a dry run prints.
toc.build_parser = build_parser
toc.build_config = build_config


if __name__ == "__main__":
    sys.exit(toc.cli())
