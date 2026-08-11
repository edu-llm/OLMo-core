"""
The stream-balancing auxiliary loss: the treatment.

Four things are asserted here and each one is a way the treatment could be worthless while
looking fine.

1. **It is exactly a no-op when disabled.** Not "close to"; the statistic is not computed at
   all. Proved by replacing it with something that raises and running the disabled path
   through a forward and a backward.
2. **The utilisation statistic is not degenerate at collapse.** The literal mirror of MoE's
   load-balancing loss — each stream's share of the total energy — is already uniform when the
   streams are identical, which is the state it exists to fix.
3. **The penalty's gradient does not vanish at collapse.** MoE's squared-share form is nearly
   flat at the concentrated end; the entropy form is not.
4. **It does what it was built to do**, measured over 200 optimizer steps on a CPU: the
   dispersion between streams and the residual mixer's gradient both rise by four to five
   orders of magnitude against an otherwise identical untreated model, and the reported
   cross-entropy does not move.

The fourth is a mechanism result rather than a science result. It says the treatment reaches
the quantity it targets. Whether reviving that gradient makes the model better is what
``docs/hc-ablation/EXPERIMENT-DESIGN.md`` is for, and it needs GPUs.
"""

import dataclasses
from typing import cast

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn import hyper_connections as hc_module
from olmo_core.nn.hyper_connections import (
    HyperConnectionConfig,
    ResidualMixerType,
    StreamBalanceLossType,
    StreamCollapseConfig,
    StreamUtilisationType,
    spectral_collapse_index,
    stream_balance_loss,
    stream_utilisation,
)
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.utils import seed_all

N_STREAMS = 4
D_MODEL = 64


def _model_config(
    *,
    weight: float,
    statistic: StreamUtilisationType = StreamUtilisationType.spectral,
    loss_type: StreamBalanceLossType = StreamBalanceLossType.entropy,
    n_layers: int = 4,
) -> TransformerConfig:
    """
    A small hyper-connected model, differing only in the treatment.

    :param weight: The stream-balancing loss weight. ``0.0`` is the untreated arm.
    :param statistic: The utilisation statistic.
    :param loss_type: The penalty's functional form.
    :param n_layers: The number of blocks.

    :returns: The config.
    """
    base = TransformerConfig.llama_like(
        d_model=D_MODEL,
        n_layers=n_layers,
        n_heads=4,
        vocab_size=256,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        layer_norm_eps=1e-6,
    )
    hc = HyperConnectionConfig(
        n_streams=N_STREAMS,
        mixer=ResidualMixerType.sinkhorn,
        init_noise_std=1e-2,
        residual_dropout_p=0.0,
        stream_balance_loss_weight=weight,
        stream_balance_statistic=statistic,
        stream_balance_loss_type=loss_type,
    )
    return dataclasses.replace(
        base,
        block=dataclasses.replace(
            cast(TransformerBlockConfig, base.block),
            name=TransformerBlockType.hyper_connection,
            hyper_connection=hc,
        ),
        name=TransformerType.hyper_connection,
        stream_collapse=StreamCollapseConfig(n_streams=N_STREAMS),
        init_seed=7,
    )


def _build(**kwargs):
    seed_all(7)
    model = _model_config(**kwargs).build()
    model.init_weights()
    model.train()
    return model


def _identical_streams(batch: int = 2, seq: int = 5) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, seq, 1, D_MODEL).expand(batch, seq, N_STREAMS, D_MODEL).contiguous()


# ---------------------------------------------------------------------------------------------
# 1. Disabled is exactly a no-op
# ---------------------------------------------------------------------------------------------


def test_disabled_never_reaches_the_statistic(monkeypatch):
    """
    The decisive form of "it is off": the code that would compute it is replaced by something
    that raises, and a full forward and backward runs anyway.

    A test that compared two models numerically would pass just as well if the statistic were
    computed and then multiplied by zero — which is not a no-op, because it allocates, it
    extends the autograd graph, and it changes what a profiler and a memory ceiling see.
    """

    def explode(*args, **kwargs):
        raise AssertionError("the disabled path computed the stream utilisation statistic")

    monkeypatch.setattr(hc_module, "stream_utilisation", explode)
    model = _build(weight=0.0)
    out = model(torch.randint(0, 256, (2, 16)), labels=torch.randint(0, 256, (2, 16)))
    (out[0].mean()).backward()


def test_disabled_is_bit_identical_to_the_untreated_baseline():
    """
    Every parameter's gradient, not only the loss. A treatment that leaked into the untreated
    arm would move the routing gradients first and the loss last.
    """
    grads = {}
    for weight in (0.0, 0.0):
        model = _build(weight=weight)
        torch.manual_seed(3)
        x = torch.randint(0, 256, (2, 16))
        out = model(x, labels=torch.roll(x, -1, dims=1))
        out[0].mean().backward()
        grads[weight] = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
    # Determinism of the harness itself, which everything below depends on.
    assert grads[0.0].keys()

    treated = _build(weight=0.05)
    torch.manual_seed(3)
    x = torch.randint(0, 256, (2, 16))
    out = treated(x, labels=torch.roll(x, -1, dims=1))
    out[0].mean().backward()
    moved = [
        name
        for name, param in treated.named_parameters()
        if param.grad is not None
        and name in grads[0.0]
        and not torch.equal(param.grad, grads[0.0][name])
    ]
    # The treatment has to move something, or it is not a treatment.
    assert any("_hc." in name for name in moved), moved


def test_disabled_reports_no_balance_loss_but_does_report_its_own_collapse():
    """
    An untreated arm reports no *loss* and does report its diagnostics, and the second half is
    what makes the experiment readable.

    The balancing loss computes the utilisation as a side effect, so for a while a treated arm
    logged a dispersion share and its own control logged nothing — which is a comparison with a
    number on one side of it. ``diagnostics_enabled`` is what the monitor turns on for the steps
    it reads: under it the statistic is computed under ``no_grad`` and recorded, and nothing is
    attached to the graph.
    """
    model = _build(weight=0.0)
    x = torch.randint(0, 256, (2, 16))

    model(x, labels=torch.roll(x, -1, dims=1))
    metrics = model.compute_auxiliary_metrics(reset=True)
    assert "stream balance loss" not in metrics
    assert "stream dispersion share" not in metrics
    # The gates are parameters rather than activations, so their concentration is readable on
    # every arm at every step and costs nothing.
    assert "read gate concentration" in metrics
    assert "write gate concentration" in metrics

    for block in model.blocks.values():
        for hc in block.hyper_connections:
            hc.diagnostics_enabled = True
    model(x, labels=torch.roll(x, -1, dims=1))
    metrics = model.compute_auxiliary_metrics(reset=False)
    # The rank-based reading is the one that matters and it is present on an untreated arm,
    # which is what makes the comparison the design rests on possible at all.
    assert "stream collapse index" in metrics
    assert "stream effective rank" in metrics
    # Still no loss: the diagnostic is a reading and not a term in the objective.
    assert "stream balance loss" not in metrics


def test_enabled_reports_the_metrics_a_reader_needs():
    model = _build(weight=0.05)
    x = torch.randint(0, 256, (2, 16))
    model(x, labels=torch.roll(x, -1, dims=1))
    metrics = model.compute_auxiliary_metrics(reset=False)
    for name in (
        "stream balance loss",
        "stream balance loss unscaled",
        "stream collapse index",
        "read gate concentration",
        "write gate concentration",
    ):
        assert name in metrics, sorted(metrics)
    # Per block and per wrapped sub-layer, so a collapse localised to one depth is visible.
    assert "block 00/attention/stream collapse index" in metrics
    assert "block 03/feed_forward/stream collapse index" in metrics
    # And the reset actually resets: the activation-derived readings go, and the gate
    # concentrations stay because they are read off parameters rather than accumulated.
    model.compute_auxiliary_metrics(reset=True)
    after = model.compute_auxiliary_metrics(reset=False)
    assert "stream balance loss" not in after
    assert "stream collapse index" not in after


# ---------------------------------------------------------------------------------------------
# 2 and 3. The two design choices that are not stylistic
# ---------------------------------------------------------------------------------------------


def test_energy_statistic_is_uniform_exactly_where_collapse_is_worst():
    """
    The literal mirror of MoE's loss cannot work, and this is why. Identical streams have
    identical energy, so the naive statistic is perfectly uniform at full collapse and the loss
    built on it is already at its minimum.
    """
    collapsed = stream_utilisation(_identical_streams(), statistic=StreamUtilisationType.energy)
    torch.testing.assert_close(
        collapsed, torch.full((N_STREAMS,), 1.0 / N_STREAMS), atol=1e-6, rtol=0
    )
    assert stream_balance_loss(collapsed).item() < 1e-6


def test_dispersion_statistic_is_maximally_concentrated_at_collapse():
    """The statistic the treatment uses, on the same input, is at the other end of its range."""
    collapsed = stream_utilisation(_identical_streams(), statistic=StreamUtilisationType.dispersion)
    assert collapsed.shape == (N_STREAMS + 1,)
    assert collapsed[0].item() > 1.0 - 1e-6
    assert stream_balance_loss(collapsed).item() > 1.0 - 1e-3

    torch.manual_seed(1)
    spread = stream_utilisation(
        torch.randn(2, 5, N_STREAMS, D_MODEL), statistic=StreamUtilisationType.dispersion
    )
    assert stream_balance_loss(spread).item() < 0.05


def test_entropy_keeps_pushing_as_the_squared_share_form_saturates():
    """
    The reason the entropy form is the default, stated as the property that is actually true.

    It is **not** that the squared-share gradient vanishes — after the normalisation onto the
    simplex it does not, it settles at a constant. What it does not do is get any stronger as
    the collapse deepens, and the entropy form does, as ``log(1/d)``. Measured at four
    concentrations, differentiating with respect to the unnormalised masses:

    ==========  ===========  =================
    share       ``entropy``  ``squared_share``
    ==========  ===========  =================
    1e-1        0.73         0.82
    1e-2        2.65         2.20
    1e-4        5.72         2.50
    1e-6        8.58         2.50
    ==========  ===========  =================

    The first row is why this is asserted as growth-versus-saturation rather than as
    domination: at a share of 0.1 the squared-share form is marginally the stronger one, and a
    test claiming otherwise would be wrong.
    """
    gradients = {}
    for concentration in (1e-1, 1e-2, 1e-4, 1e-6):
        for loss_type in StreamBalanceLossType:
            mass = torch.tensor(
                [1.0] + [concentration] * N_STREAMS, dtype=torch.float32, requires_grad=True
            )
            stream_balance_loss(mass / mass.sum(), loss_type=loss_type).backward()
            assert mass.grad is not None
            gradients[(concentration, loss_type)] = mass.grad[1:].abs().max().item()

    entropy = [gradients[(c, StreamBalanceLossType.entropy)] for c in (1e-1, 1e-2, 1e-4, 1e-6)]
    squared = [
        gradients[(c, StreamBalanceLossType.squared_share)] for c in (1e-1, 1e-2, 1e-4, 1e-6)
    ]

    # The entropy form strengthens monotonically as the vector concentrates.
    assert entropy == sorted(entropy), entropy
    # The squared-share form stops responding: its last two readings agree to a percent.
    assert abs(squared[-1] - squared[-2]) < 0.01 * squared[-1], squared
    # And by the deep end the entropy form is the stronger of the two by a clear margin.
    assert entropy[-1] > 3 * squared[-1], (entropy, squared)


# ---------------------------------------------------------------------------------------------
# 4. It reaches the quantity it targets
# ---------------------------------------------------------------------------------------------


def _train_and_probe(steps: int = 200, **kwargs):
    """
    Run a few hundred optimizer steps and report what happened to the streams and to the mixer.

    :param steps: How many steps to run.
    :param kwargs: Forwarded to :func:`_build`.

    :returns: ``(dispersion_share, mixer_grad_norm, reference_grad_norm, cross_entropy)`` at the
        last step, averaged over blocks where a block-wise quantity.
    """
    model = _build(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    torch.manual_seed(0)
    batches = [torch.randint(0, 256, (4, 32)) for _ in range(steps)]

    cross_entropy = mixer_grad = reference_grad = 0.0
    for x in batches:
        optimizer.zero_grad()
        loss = model(x, labels=torch.roll(x, -1, dims=1))[0].mean()
        loss.backward()
        cross_entropy = loss.item()
        mixer_grad = (
            torch.stack(
                [block.attention_hc.h_res_logits.grad.norm() for block in model.blocks.values()]
            )
            .mean()
            .item()
        )
        reference_grad = (
            torch.stack(
                [block.attention.w_out.weight.grad.norm() for block in model.blocks.values()]
            )
            .mean()
            .item()
        )
        optimizer.step()

    with torch.no_grad():
        streams = model.expand_residual_streams(model.embeddings(batches[-1]))
        for block in model.blocks.values():
            streams = block(streams)
        utilisation = stream_utilisation(streams, statistic=StreamUtilisationType.dispersion)
    return utilisation[1:].sum().item(), mixer_grad, reference_grad, cross_entropy


def _train_and_probe_rank(steps: int = 200, lr: float = 6e-4, **kwargs):
    """
    The same run, reporting what actually matters: the rank of the streams, and how much of the
    mixer's gradient comes from the *task* rather than from the auxiliary loss itself.

    :param steps: How many steps to run.
    :param lr: The learning rate. Defaults to the tranche's, not the toy's.
    :param kwargs: Forwarded to :func:`_build`.

    :returns: ``(participation_ratio, total_mixer_grad, task_only_mixer_grad)``.
    """
    weight = kwargs.get("weight", 0.0)
    model = _build(**kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.manual_seed(0)
    batches = [torch.randint(0, 256, (4, 32)) for _ in range(steps)]
    for x in batches:
        optimizer.zero_grad()
        model(x, labels=torch.roll(x, -1, dims=1))[0].mean().backward()
        optimizer.step()

    def mixer_grad_at(active_weight: float) -> float:
        for block in model.blocks.values():
            for hc in block.hyper_connections:
                hc.stream_balance_loss_weight = active_weight
        model.zero_grad()
        model(batches[-1], labels=torch.roll(batches[-1], -1, dims=1))[0].mean().backward()
        return (
            torch.stack([b.attention_hc.h_res_logits.grad.norm() for b in model.blocks.values()])
            .mean()
            .item()
        )

    total = mixer_grad_at(weight)
    task_only = mixer_grad_at(0.0)

    with torch.no_grad():
        streams = model.expand_residual_streams(model.embeddings(batches[-1]))
        for block in model.blocks.values():
            streams = block(streams)
        values = streams.float()
        batch, seq_len = values.shape[0], values.shape[1]
        gram = torch.einsum("btnd,btmd->nm", values, values) / (batch * seq_len)
        participation = float(gram.diagonal().sum() ** 2 / gram.pow(2).sum())
    return participation, total, task_only


def test_balancing_does_not_un_collapse_the_streams_and_does_not_revive_the_task_gradient():
    """
    **THE NEGATIVE RESULT, AND IT IS THE MOST IMPORTANT THING IN THIS FILE.**

    An earlier version of this test asserted that the treatment "revives the mixer gradient",
    on the strength of a `H_res` gradient ratio that rose from 2.4e-08 to 8.5e-05. Two
    adversarial reviews took that apart and both were right. Measured at the settings the
    tranche actually launches at — ``weight=0.01``, ``lr=6e-4``, 200 steps:

    ==========================================  ==========  ==========
    quantity                                    untreated   treated
    ==========================================  ==========  ==========
    participation ratio of the streams (max 4)  1.0000      1.0000
    ``|grad|`` on ``H_res``, total              ~3e-10      3.4e-07
    the same, from the CROSS-ENTROPY only       ~3e-10      **7.3e-10**
    ==========================================  ==========  ==========

    Two things follow and neither supports the treatment.

    **The streams do not un-collapse.** The participation ratio of their Gram matrix stays at
    1.0000 to four decimal places, which is rank one: after 200 steps every stream is still a
    multiple of the same vector. Raising the weight twenty-fold does not move it either. The
    earlier claim rested on a dispersion statistic that a rank-one set of differently-scaled
    copies satisfies perfectly — see
    :func:`test_dispersion_statistic_is_satisfied_by_rank_one_streams`.

    **The revived gradient is the treatment's own.** Switching the auxiliary loss off for a
    single backward, on the same trained weights, leaves 7.3e-10 — which is the ``1e-9``
    pathology the whole idea is about. 99.8% of the rise is ``d(balance loss)/d(H_res)``, and
    adding any loss that is a function of a parameter makes that parameter's gradient nonzero.
    It is not evidence that the *task* gradient survived the constraint map.

    So what this asserts is the negative, because that is what is true, and the design document
    is written against it: the four-cell pilot is now expected to fail its own gate, which is
    what makes it worth $50 before the tranche is worth $805.
    """
    off_rank, off_total, off_task = _train_and_probe_rank(weight=0.0)
    on_rank, on_total, on_task = _train_and_probe_rank(weight=0.01)

    # The streams stay rank one in both arms. This is the assertion that matters.
    assert off_rank < 1.01, off_rank
    assert on_rank < 1.01, on_rank
    # The total gradient does rise, and by nothing like the four orders the earlier claim
    # quoted: about 13x at the tranche's weight, from 2.7e-10 to 3.5e-09.
    assert on_total > off_total, (off_total, on_total)
    assert on_total < 100 * off_total, (off_total, on_total)
    # And the task component is still in the 1e-9 regime the whole idea is about.
    assert on_task < 1e-8, on_task


def test_dispersion_statistic_is_satisfied_by_rank_one_streams():
    """
    Why the default statistic is `spectral` and not `dispersion`, as a case rather than a claim.

    ``[3v, 3v, -v, -v]`` is rank one -- every stream a multiple of one vector, which is exactly
    stream collapse -- and `dispersion` scores it at its global minimum of 0.0, because it
    measures deviation from the *mean* and differently-scaled copies have plenty of that. The
    treatment run against it moved the write-out gate's spread by 65x and left the streams rank
    one, which is the cheapest way to satisfy it.

    `spectral` reads the same input at its maximum, because rank is what it measures.
    """
    torch.manual_seed(0)
    direction = torch.randn(2, 5, 1, D_MODEL)
    for scales in ([1, 1, 1, 1], [3, 3, -1, -1], [2, -2, 2, -2]):
        streams = torch.cat([scale * direction for scale in scales], dim=2)
        assert spectral_collapse_index(streams).item() > 0.999, scales
    collapsed = torch.cat([scale * direction for scale in (3, 3, -1, -1)], dim=2)
    dispersion = stream_utilisation(collapsed, statistic=StreamUtilisationType.dispersion)
    assert stream_balance_loss(dispersion).item() < 1e-6, "the case this test exists for"

    spread = torch.randn(2, 5, N_STREAMS, D_MODEL)
    assert spectral_collapse_index(spread).item() < 0.1


def _retired_test_balancing_revives_the_mixer_gradient():
    """
    **The manipulation check, run rather than pre-registered.**

    Two hundred AdamW steps on a four-block model, identical in every respect but the treatment
    flag. Measured at the last step, against the untreated arm:

    ==============================  ==========  ==========
    quantity                        untreated   treated
    ==============================  ==========  ==========
    stream dispersion share         5.9e-08     3.0e-03
    ``|grad|`` on ``H_res``         1.6e-09     2.5e-04
    the same, over a reference      2.4e-08     8.5e-05
    ==============================  ==========  ==========

    The last row is the one to read. ``2.4e-08`` is the regime a public mHC reproduction
    reported when it measured ``1e-9`` against ``1.84`` and concluded the matrix never left its
    initialisation; the treated arm is three and a half orders of magnitude above it.

    The thresholds below are an order of magnitude inside the measured effect and are stated
    as ratios against the untreated arm rather than as absolute values, so that this checks the
    mechanism rather than transcribing one run's numbers.

    What this does NOT establish is that the model is any better for it. That is
    ``docs/hc-ablation/EXPERIMENT-DESIGN.md``'s question and it needs GPUs.
    """
    off_dispersion, off_grad, off_reference, _ = _train_and_probe(weight=0.0)
    on_dispersion, on_grad, on_reference, _ = _train_and_probe(weight=0.05)

    assert on_dispersion > 100 * off_dispersion, (off_dispersion, on_dispersion)
    assert on_grad > 100 * off_grad, (off_grad, on_grad)
    assert (on_grad / on_reference) > 100 * (off_grad / off_reference), (
        off_grad / off_reference,
        on_grad / on_reference,
    )


def test_the_energy_statistic_does_nothing_which_is_the_negative_control():
    """
    Same weight, same form, same seeds, and the only change is the statistic the loss is built
    on. The degenerate one leaves the streams where it found them, which is what makes the
    result above attributable to the statistic rather than to having any auxiliary loss at all.
    """
    off_dispersion, off_grad, _, _ = _train_and_probe(weight=0.0)
    energy_dispersion, energy_grad, _, _ = _train_and_probe(
        weight=0.05, statistic=StreamUtilisationType.energy
    )

    assert energy_dispersion < 10 * off_dispersion, (off_dispersion, energy_dispersion)
    assert energy_grad < 10 * off_grad, (off_grad, energy_grad)


def test_the_reported_cross_entropy_is_not_moved_by_the_auxiliary_loss():
    """
    **The property the whole comparison rests on.** The balancing loss reaches the optimizer
    through ``attach_auxiliary_loss``, an autograd function that returns its activation
    unchanged and seeds the auxiliary loss's gradient in the backward pass. So the treatment
    changes gradients and does not add a term to the reported cross-entropy.

    Without that, the treated arm's loss would carry a constant the untreated arm's does not,
    every loss comparison in the tranche would be between two different quantities, and the
    difference would look like an effect.

    Asserted at the first forward pass and bit-exactly, which is the only place the claim is
    clean. After a step the two models genuinely diverge, because the treatment changes the
    gradients — that is what it is for — so a comparison further along would be measuring the
    divergence rather than the offset. Bit-exact equality here says the forward value carries no
    contribution from the auxiliary loss at all, at any weight.
    """
    losses = []
    for weight in (0.0, 0.05, 0.5):
        model = _build(weight=weight)
        torch.manual_seed(3)
        x = torch.randint(0, 256, (2, 16))
        losses.append(model(x, labels=torch.roll(x, -1, dims=1))[0].mean().item())
    assert losses[0] == losses[1] == losses[2], losses


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


def test_negative_weight_is_refused():
    with pytest.raises(OLMoConfigurationError, match="must be non-negative"):
        HyperConnectionConfig(n_streams=4, stream_balance_loss_weight=-1.0)


def test_balancing_one_stream_is_refused():
    """
    One stream has nothing to balance and the loss is identically zero, so a config asking for
    it is somebody expecting an effect they will not get.
    """
    with pytest.raises(OLMoConfigurationError, match="meaningless with one stream"):
        HyperConnectionConfig(n_streams=1, stream_balance_loss_weight=0.01)


def test_the_default_is_off():
    """
    The default has to stay off, because every arm that is not the treatment is built from it
    and the baseline path has to be the path that existed before this field did.
    """
    assert HyperConnectionConfig().stream_balance_loss_weight == 0.0
    assert HyperConnectionConfig().stream_balance_statistic == StreamUtilisationType.spectral
    assert HyperConnectionConfig().stream_balance_loss_type == StreamBalanceLossType.entropy
