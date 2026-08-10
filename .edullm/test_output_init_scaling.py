"""The sqrt(n) output-init numbers, measured against the real arm configs rather than quoted.

Run with ``PYTHONPATH=src pytest -v .edullm/test_output_init_scaling.py``.

WHY THIS FILE EXISTS. The same measurement was quoted at three magnitudes in three documents --
0.80, 0.73 and 0.115 relative in the logits -- and read interchangeably, as though the number
were a property of the method. It is not. It is a property of the model *and the protocol it was
measured under*, and none of the three named its protocol. Two of them were measured on a
64-wide toy model and the third at a sequence length of 32, against the 4,096 every cell of the
tranche trained at.

So the numbers are asserted here, at the shape and the length in :data:`STAGE_PINNED`, and the
document that quotes them is parsed rather than trusted. A change to the scaling, to the model
shape, to the sequence length, or to the prose then fails loudly instead of leaving a document
saying something that is no longer true.

It is also asserted here rather than in ``src/test/`` because the shape under test is
``hc_370M``, which lives in this directory and is not part of the library.

THIS FILE IS SLOW -- about a minute, nine 370M models built on the CPU. That is the price of
pinning the number at the configuration that ran rather than at a proxy. The structural claims
it rests on are pinned again, for pennies, at the toy shape in
``src/test/nn/transformer/hyper_connection_test.py``.

CAVEAT ON PRECISION. Everything here is float32 on the CPU; the tranche trained with
``param_dtype`` bfloat16. RMSNorm computes its variance in float32 either way, which is the
operation the result turns on, but these are not bit-for-bit the run's activations.
"""

import gc
import os
import pathlib
import sys
from dataclasses import replace
from typing import Dict, Optional, Tuple

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms as arms  # noqa: E402

from olmo_core.nn.residual_stream import HyperConnectionConfig  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockType,
    TransformerConfig,
)

#: dolma2, which is what every cell tokenized with.
VOCAB_SIZE = 100_278

#: One initialization seed. Over seeds 12,536-12,538 the headline moves between 0.36194 and
#: 0.37313, which is why the tolerances below are percentages and not decimals.
INIT_SEED = 12_536

#: Uniform token ids. A Zipfian draw, which is closer to real text, gives 0.334 rather than
#: 0.362 -- a tenth of the value, and the reason the protocol is stated wherever it is quoted.
DATA_SEED = 0

PRE_REGISTRATION = pathlib.Path(_HERE) / "hyper-connections.md"

#: What this file measures, at ``hc_370M``, batch 1, at the tranche's own sequence length. The
#: reproduction scripts and the derivation live in the research note; these are the figures the
#: documents are allowed to quote.
MEASURED: Dict[str, float] = {
    # `faithful` against `baseline`: the scaling on, at the paper's exponent of 0.5.
    "relative_on": 0.362,
    # `no-output-init` against `baseline`: the same model, to floating point.
    "relative_off_ceiling": 1e-6,
    # The magnitude the scaling exists to correct, which the final norm then deletes.
    "hidden_ratio_off": 4.0,
    # The stream the second block reads. Block 0 reads the embeddings and cannot move.
    "block_1_read": 0.918,
    # With the norm's epsilon taken to 1e-12 the scaling is a no-op, which is what says the
    # distortion is a numerical artifact and not a reweighting of depth.
    "relative_on_tiny_eps_ceiling": 1e-4,
    # Scaling block 0's attention output alone reproduces essentially all of it.
    "relative_block_0_attention_only": 0.362,
    # ... and scaling everything else reproduces essentially none of it.
    "relative_everything_but_block_0_attention_ceiling": 0.01,
    # Why that one module and no other: its pre-norm variance is below the norm's epsilon, so
    # the norm that follows it is not normalizing and the factor passes straight through.
    "attention_0_pre_norm_rms": 5.607e-4,
    # The counterfactual the mechanism claim rests on. Same width, same depth, norms moved in
    # front of the sublayers: there the rescale is real, and the epsilon has nothing to do
    # with it. Measured at PROBE_SEQ_LEN, not at the run length.
    "relative_pre_norm": 0.578,
}

#: The pre-norm counterfactual is measured here rather than at the run length. It is a statement
#: about an architecture we did not run, so paying four times over for it is not worth it.
PROBE_SEQ_LEN = 512


def _config(
    exponent: Optional[float],
    *,
    eps: Optional[float] = None,
    n_layers: Optional[int] = None,
    pre_norm: bool = False,
) -> TransformerConfig:
    """
    The 370M config an arm runs, or the ordinary residual baseline.

    :param exponent: ``None`` for ``baseline``, ``0.0`` for ``no-output-init``, ``0.5`` for
        ``faithful``.
    :param eps: Override every layer norm's epsilon. Used only by the counterfactual.
    :param n_layers: Override the depth. Used only by the depth check.
    :param pre_norm: Normalize before each sublayer rather than after. **Not an arm** --
        ``Arm.apply`` refuses this block type and is right to. It exists here only to price the
        counterfactual the mechanism claim in the write-up rests on.
    """
    extra = {} if n_layers is None else {"n_layers": n_layers}
    config = arms.hc_370M(vocab_size=VOCAB_SIZE, **extra)
    assert not isinstance(config.block, dict)
    if exponent is None:
        config.block.name = (
            TransformerBlockType.default if pre_norm else TransformerBlockType.reordered_norm
        )
    else:
        config.block.name = (
            TransformerBlockType.hyper_connection
            if pre_norm
            else TransformerBlockType.hyper_connection_reordered_norm
        )
        config.block.hyper_connections = replace(
            HyperConnectionConfig(n_lanes=arms.N_LANES), output_init_exponent=exponent
        )
    config.__post_init__()
    if eps is not None:
        config.block.layer_norm.eps = eps
        assert config.lm_head.layer_norm is not None
        config.lm_head.layer_norm.eps = eps
    return config


def _build(config: TransformerConfig, seq_len: int) -> torch.nn.Module:
    config.init_seed = INIT_SEED
    model = config.build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=seq_len)
    model.eval()
    return model


def _probe(model: torch.nn.Module, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    One forward pass.

    :returns: The logits, the RMS of the stream each block's sequence mixer reads, and a dict
        holding the RMS of the hidden state entering the final norm (``hidden``) and the RMS of
        block 0's attention output *before* the norm that follows it (``attention_0``). The
        mixer's input is the stream itself and not a normalized copy of it, because the
        reordered-norm block normalizes *after* its sublayer -- which is also why the second
        scalar is the one that decides everything else in this file.
    """
    stream: Dict[int, float] = {}
    scalars: Dict[str, float] = {}

    def rms(tensor: torch.Tensor) -> float:
        return float(tensor.detach().double().pow(2).mean().sqrt())

    handles = [
        model.lm_head.norm.register_forward_pre_hook(
            lambda _m, args: scalars.__setitem__("hidden", rms(args[0]))
        ),
        model.blocks["0"].attention.register_forward_hook(
            lambda _m, _a, out: scalars.__setitem__("attention_0", rms(out))
        ),
    ]
    handles += [
        block.attention.register_forward_pre_hook(
            lambda _m, args, key=int(index): stream.__setitem__(key, rms(args[0]))
        )
        for index, block in model.blocks.items()
    ]
    try:
        with torch.no_grad():
            logits = model(input_ids).detach()
    finally:
        for handle in handles:
            handle.remove()
    return logits, stream, scalars


def _relative(actual: torch.Tensor, reference: torch.Tensor, chunk: int = 256) -> float:
    """
    ``||actual - reference|| / ||reference||``, accumulated in float64 over slices so that a
    4,096-token logit tensor is never held twice at double width.
    """
    numerator = denominator = 0.0
    for start in range(0, actual.shape[1], chunk):
        a = actual[:, start : start + chunk].double()
        b = reference[:, start : start + chunk].double()
        numerator += float((a - b).pow(2).sum())
        denominator += float(b.pow(2).sum())
    return (numerator / denominator) ** 0.5


def _tokens(seq_len: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(DATA_SEED)
    return torch.randint(0, VOCAB_SIZE, (1, seq_len), generator=generator)


@pytest.fixture(scope="module")
def run_length() -> int:
    """The sequence length the tranche trained at, read from the pinned stage options."""
    return int(arms.STAGE_PINNED["sequence_length"])  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def measured(run_length: int) -> Dict[str, float]:
    """
    Every figure in :data:`MEASURED`, in one pass, so that the baseline is built once.

    Only scalars survive the fixture. A 4,096-token logit tensor at this vocabulary is 1.6 GB and
    there are six of them.
    """
    input_ids = _tokens(run_length)
    out: Dict[str, float] = {}

    baseline_logits, baseline_stream, baseline_scalars = _probe(
        _build(_config(None), run_length), input_ids
    )
    out["attention_0_pre_norm_rms"] = baseline_scalars["attention_0"]
    gc.collect()

    for exponent, key in ((0.0, "off"), (0.5, "on")):
        model = _build(_config(exponent), run_length)
        logits, stream, scalars = _probe(model, input_ids)
        out[f"relative_{key}"] = _relative(logits, baseline_logits)
        out[f"hidden_ratio_{key}"] = scalars["hidden"] / baseline_scalars["hidden"]
        if exponent == 0.5:
            out.update(
                {f"block_{index}_read": stream[index] / baseline_stream[index] for index in stream}
            )
        del model, logits
        gc.collect()

    # The scaling applied to one module, and to every module except that one.
    for factor, key in ((0.5, "block_0_attention_only"), (2.0, "everything_but_block_0_attention")):
        model = _build(_config(0.0 if factor == 0.5 else 0.5), run_length)
        with torch.no_grad():
            model.blocks["0"].attention.w_out.weight.mul_(factor)
        logits, _, _ = _probe(model, input_ids)
        out[f"relative_{key}"] = _relative(logits, baseline_logits)
        del model, logits
        gc.collect()

    # The same block-1 read at half the depth. Nothing else about the model changes.
    shallow_logits, shallow_stream, _ = _probe(
        _build(_config(None, n_layers=8), run_length), input_ids
    )
    del shallow_logits
    gc.collect()
    model = _build(_config(0.5, n_layers=8), run_length)
    logits, stream, _ = _probe(model, input_ids)
    out["block_1_read_at_8_layers"] = stream[1] / shallow_stream[1]
    del model, logits
    gc.collect()

    # And with the norm's epsilon effectively removed.
    tiny = 1e-12
    tiny_logits, tiny_stream, _ = _probe(_build(_config(None, eps=tiny), run_length), input_ids)
    gc.collect()
    model = _build(_config(0.5, eps=tiny), run_length)
    logits, stream, _ = _probe(model, input_ids)
    out["relative_on_tiny_eps"] = _relative(logits, tiny_logits)
    out["block_1_read_tiny_eps"] = stream[1] / tiny_stream[1]
    del model, logits, tiny_logits, baseline_logits
    gc.collect()

    # The counterfactual the mechanism claim rests on, at PROBE_SEQ_LEN to keep this affordable:
    # the same width and depth with the norms moved in front of the sublayers, which is what a
    # LLaMA-shaped reimplementation would have. Not an arm -- `Arm.apply` refuses to build it.
    probe_ids = _tokens(PROBE_SEQ_LEN)
    pre_norm_logits, _, _ = _probe(_build(_config(None, pre_norm=True), PROBE_SEQ_LEN), probe_ids)
    gc.collect()
    for eps, key in ((None, "pre_norm"), (1e-12, "pre_norm_tiny_eps")):
        if eps is not None:
            pre_norm_logits, _, _ = _probe(
                _build(_config(None, eps=eps, pre_norm=True), PROBE_SEQ_LEN), probe_ids
            )
            gc.collect()
        model = _build(_config(0.5, eps=eps, pre_norm=True), PROBE_SEQ_LEN)
        logits, _, _ = _probe(model, probe_ids)
        out[f"relative_{key}"] = _relative(logits, pre_norm_logits)
        del model, logits
        gc.collect()
    del pre_norm_logits
    gc.collect()

    return out


def test_the_shape_these_numbers_were_measured_at_is_the_shape_that_ran():
    """
    Every figure below is a property of one model. If the model moves, they are figures about
    nothing, and the documents quoting them say so in the same breath -- so the shape is asserted
    before the numbers are.
    """
    config = arms.hc_370M(vocab_size=VOCAB_SIZE)
    assert not isinstance(config.block, dict)
    mixer = config.block.attention or config.block.sequence_mixer
    assert mixer is not None

    assert (config.d_model, config.n_layers, mixer.n_heads) == (1024, 16, 16)
    assert config.d_model // mixer.n_heads == 64
    assert config.block.name == TransformerBlockType.reordered_norm
    assert config.block.layer_norm.eps == 1e-6

    faithful = arms.ARMS["faithful"].apply(arms.hc_370M(vocab_size=VOCAB_SIZE))
    assert not isinstance(faithful.block, dict)
    assert faithful.block.hyper_connections is not None
    assert faithful.block.hyper_connections.output_init_exponent == 0.5
    assert faithful.block.hyper_connections.n_lanes == 4

    arm_four = arms.ARMS["no-output-init"].apply(arms.hc_370M(vocab_size=VOCAB_SIZE))
    assert not isinstance(arm_four.block, dict)
    assert arm_four.block.hyper_connections is not None
    assert arm_four.block.hyper_connections.output_init_exponent == 0.0

    assert arms.STAGE_PINNED["model_factory"] == "hc_370M"
    assert arms.STAGE_PINNED["sequence_length"] == 4096


def test_with_the_scaling_off_the_arm_is_the_baseline_at_the_length_it_ran(measured):
    """
    The paper's section 2.3 equivalence, at the configuration that ran and not at a proxy. Four
    lanes summed make the pre-unembedding hidden state exactly four times the baseline's, and a
    scale-invariant RMSNorm sits between that sum and the unembedding, so the model is the
    baseline anyway.
    """
    assert measured["hidden_ratio_off"] == pytest.approx(MEASURED["hidden_ratio_off"], rel=1e-4)
    assert measured["relative_off"] < MEASURED["relative_off_ceiling"], (
        "with the output-init scaling off the model is no longer the baseline at init, so arm 4 "
        "is not the paper's equivalence claim and its framing needs rewriting"
    )


def test_the_scaling_moves_the_model_by_the_measured_amount(measured):
    """
    THE NUMBER THREE DOCUMENTS GOT WRONG. 0.362 at ``hc_370M`` on a 4,096-token sequence, not
    the 0.80 of a 64-wide toy model, the 0.73 of the same toy, or the 0.115 of the right model
    read at a sequence length of 32.
    """
    assert measured["relative_on"] == pytest.approx(MEASURED["relative_on"], rel=0.05)
    assert measured["block_1_read"] == pytest.approx(MEASURED["block_1_read"], rel=0.01)
    assert measured["block_0_read"] == pytest.approx(
        1.0, rel=1e-6
    ), "block 0 reads the embeddings, which no output-module scaling can touch"


def test_the_distortion_is_the_norms_epsilon_and_not_a_reweighting_of_depth(measured):
    """
    WHAT THE DOCUMENTS SAID WAS HAPPENING IS NOT WHAT IS HAPPENING, and this is the assertion
    that says so. They described the scaling as a per-block ``n**-0.5`` that "reweights depth
    against depth" and cannot be absorbed by a norm.

    It is absorbed by a norm. This block normalizes *after* its sublayer, so a constant factor on
    the sublayer's output is divided straight back out -- exactly, except where the pre-norm
    variance is small enough for the ``eps`` in ``rsqrt(variance + eps)`` to survive. Take the
    epsilon to 1e-12 and the scaling does nothing at all.

    This matters beyond bookkeeping: it is why H1b measures an optimization difference rather
    than a difference in the function the two arms start from.
    """
    assert measured["relative_on_tiny_eps"] < MEASURED["relative_on_tiny_eps_ceiling"], (
        "the scaling now survives a norm that is exactly scale-invariant, so it is no longer a "
        "numerical artifact and every document's reading of arm 4 has to be revisited"
    )
    assert measured["block_1_read_tiny_eps"] == pytest.approx(1.0, rel=1e-5)
    assert measured["relative_on"] > 100 * measured["relative_on_tiny_eps"]


def test_one_module_carries_the_whole_distortion(measured):
    """
    At this shape exactly one sublayer in the model has a pre-norm variance at or below the
    norm's epsilon -- block 0's attention output, whose input is the raw embedding stream. Scale
    that one weight matrix and the whole effect appears; scale all thirty-one others and almost
    none of it does.
    """
    assert measured["relative_block_0_attention_only"] == pytest.approx(
        MEASURED["relative_block_0_attention_only"], rel=0.05
    )
    assert (
        measured["relative_everything_but_block_0_attention"]
        < MEASURED["relative_everything_but_block_0_attention_ceiling"]
    )

    # And the reason it is that module: its variance is below the epsilon it is added to.
    variance = measured["attention_0_pre_norm_rms"] ** 2
    assert measured["attention_0_pre_norm_rms"] == pytest.approx(
        MEASURED["attention_0_pre_norm_rms"], rel=0.05
    )
    assert variance < 1e-6, (
        "block 0's attention output is no longer below the norm's epsilon, so the mechanism this "
        "file documents is not the one operating and the numbers above are unexplained"
    )


def test_the_same_rescale_is_a_real_intervention_in_a_pre_norm_stack(measured):
    """
    THE CLAIM THE WRITE-UP LEADS ON, AND THE ONE MOST LIKELY TO BE CHECKED. Everything else here
    says the paper's prescription costs almost nothing in our architecture. That is only
    interesting because it costs a great deal in the architecture a reimplementation is most
    likely to have: move the norms in front of the sublayers and the sublayer output reaches the
    residual stream unnormalized, so halving it really does halve every block's contribution.

    The tell is the epsilon. In our stack, removing it removes the whole effect. Here it changes
    nothing, because nothing was relying on it.

    This is not an arm and no cell was ever run against it. It is measured so that a number the
    documents quote is a number something checks.
    """
    assert measured["relative_pre_norm"] == pytest.approx(MEASURED["relative_pre_norm"], rel=0.05)
    assert measured["relative_pre_norm_tiny_eps"] == pytest.approx(
        measured["relative_pre_norm"], rel=0.01
    ), "the pre-norm effect has become epsilon-dependent too, which would collapse the contrast"
    assert measured["relative_pre_norm"] > 1.5 * measured["relative_on"]


def test_depth_is_not_what_moves_the_number(measured):
    """
    The claim being retired. Halving the depth leaves the block-1 read where it was, to four
    decimal places, because the pre-norm variance that decides the effect is set by the width and
    the sequence length and not by the number of blocks. The residual-stream profile the
    documents plotted against depth was a comparison across three variables at once.
    """
    assert measured["block_1_read_at_8_layers"] == pytest.approx(measured["block_1_read"], abs=1e-4)


def test_the_taper_is_a_diluted_early_perturbation_and_not_a_gradient_across_blocks(measured):
    """
    A single perturbation injected at block 0 is diluted as the stream grows, which looks like a
    taper if it is read as one. The distinguishing feature is that it is monotone and lands at
    parity, rather than being spread across the stack.
    """
    reads = [measured[f"block_{index}_read"] for index in range(16)]
    assert reads[0] == pytest.approx(1.0, rel=1e-6)
    assert reads[1] == min(reads) < 0.95
    # It recovers, monotonically, over the blocks where there is anything left to recover.
    assert reads[1] < reads[2] < reads[3] < reads[4]
    # And then sits at the baseline for the whole rest of the stack, within a percent.
    assert all(read > 0.985 for read in reads[5:])


def test_the_document_quotes_the_numbers_this_file_measured():
    """
    THE FAILURE MODE THIS WHOLE FILE IS ABOUT was a document quoting a figure nothing asserted.
    So the document is parsed. If a number here moves, the prose fails with it rather than
    quietly becoming false.
    """
    text = PRE_REGISTRATION.read_text()
    for quoted in ("0.362", "4.0000×", "0.918×", "4,096"):
        assert quoted in text, f"{quoted!r} is no longer in {PRE_REGISTRATION.name}"
    for retired in ("| 2.618× | 0.73 |", "reweights depth against depth"):
        assert retired not in text, (
            f"{retired!r} is back in {PRE_REGISTRATION.name}; it was measured on a 64-wide toy "
            "model and describes a mechanism this file shows is not the one operating"
        )
