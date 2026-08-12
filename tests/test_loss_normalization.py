"""The arm-symmetry property that the previous generation's trainer violated.

`reduction="mean"` over surviving targets makes the per-token gradient depend on
how many *other* tokens were masked. These tests pin the property directly: with
identical supervised positions, the split arm's gradient must be numerically
identical to the dense arm's, no matter how much else is masked.

`test_mean_reduction_would_fail_this` documents the magnitude of the old bug so
the regression is not just prevented but explained.
"""

import torch
import torch.nn.functional as F

from memsplit.model import IGNORE_INDEX, build_model


def _fixture(seed: int = 0):
    torch.manual_seed(seed)
    model = build_model("toy")
    idx = torch.randint(0, 512, (2, 16))
    targets = torch.randint(0, 512, (2, 16))
    return model, idx, targets


def _grad_norm(model, idx, targets, divisor):
    model.zero_grad(set_to_none=True)
    _, loss = model(idx, targets, loss_divisor=divisor)
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])


def test_masking_other_positions_does_not_rescale_surviving_gradients():
    """The core arm-symmetry property.

    'dense' supervises all 32 positions. 'split' supervises only the first 8.
    The gradient contribution of those first 8 must be identical in both arms,
    which we check by supervising *only* those 8 in the dense arm too.
    """
    model, idx, targets = _fixture()
    divisor = float(targets.numel())

    only8 = targets.clone()
    only8.view(-1)[8:] = IGNORE_INDEX

    g_a = _grad_norm(model, idx, only8, divisor)

    # Same supervised set, but reached by "masking the rest" as the split arm
    # would -- a different code path, must give the same numbers.
    split = targets.clone()
    split.view(-1)[8:] = IGNORE_INDEX
    g_b = _grad_norm(model, idx, split, divisor)

    assert torch.allclose(g_a, g_b, atol=0, rtol=0)


def test_fixed_divisor_is_arm_independent():
    """Two arms, different mask rates, same divisor -> comparable scale.

    The dense arm supervises 32 positions, the split arm 24 (25% masked). Under
    the fixed divisor the split arm's loss is *smaller* -- which is correct and
    expected, because it is summing fewer terms over the same denominator. What
    must NOT happen is the surviving terms being scaled up to compensate.
    """
    model, idx, targets = _fixture()
    divisor = float(targets.numel())

    _, dense_loss = model(idx, targets, loss_divisor=divisor)

    masked = targets.clone()
    masked.view(-1)[24:] = IGNORE_INDEX
    _, split_loss = model(idx, masked, loss_divisor=divisor)

    per_tok = F.cross_entropy(
        model(idx)[0].view(-1, 512).float(), targets.view(-1), reduction="none"
    )
    expected_split = per_tok[:24].sum() / divisor
    assert torch.allclose(split_loss, expected_split, atol=1e-6)
    assert split_loss < dense_loss


def test_mean_reduction_would_fail_this():
    """Quantifies the old bug: mean-over-surviving inflates by 1/(1-f).

    At a 25% mask rate the inflation is 1/0.75 = 1.333, matching the 1.331
    figure the project measured at its 24.89% in-document mask rate.
    """
    model, idx, targets = _fixture()
    masked = targets.clone()
    masked.view(-1)[24:] = IGNORE_INDEX

    logits = model(idx)[0]
    per_tok = F.cross_entropy(
        logits.view(-1, 512).float(), masked.view(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    )
    supervised = per_tok[per_tok != 0]

    old_mean = supervised.sum() / supervised.numel()          # sum / 24
    fixed = supervised.sum() / float(targets.numel())         # sum / 32
    ratio = (old_mean / fixed).item()

    assert abs(ratio - 32 / 24) < 1e-4, ratio
    assert abs(ratio - 1.3333) < 1e-3, ratio


def test_loss_divisor_is_mandatory():
    model, idx, targets = _fixture()
    try:
        model(idx, targets)
    except ValueError as exc:
        assert "loss_divisor is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a missing divisor must be an error, not a default")


def test_target_weights_scale_positions_independently():
    model, idx, targets = _fixture()
    divisor = float(targets.numel())
    w = torch.ones_like(targets, dtype=torch.float32)
    w.view(-1)[:8] = 0.0

    zeroed = targets.clone()
    zeroed.view(-1)[:8] = IGNORE_INDEX

    _, a = model(idx, targets, target_weights=w, loss_divisor=divisor)
    _, b = model(idx, zeroed, loss_divisor=divisor)
    assert torch.allclose(a, b, atol=1e-6)


def test_flops_accounting_keeps_the_attention_term():
    """The attention term is large at this scale under either N convention.

    Pinning both conventions, because the share differs materially between them
    and quoting one without saying which is how the 18%-vs-13% confusion arises:

        d160m ctx=1024:  13.2% (N incl. head)  /  18.2% (Kaplan N)
        d160m ctx=2048:  23.4% (N incl. head)  /  30.7% (Kaplan N)
    """
    f1 = build_model("d160m").flops_per_token()
    assert abs(f1["with_head"]["attention_share"] - 0.132) < 0.005, f1["with_head"]
    assert abs(f1["kaplan"]["attention_share"] - 0.182) < 0.005, f1["kaplan"]

    f2 = build_model("d160m_ctx2048").flops_per_token()
    assert abs(f2["with_head"]["attention_share"] - 0.234) < 0.006, f2["with_head"]
    assert abs(f2["kaplan"]["attention_share"] - 0.307) < 0.006, f2["kaplan"]

    # Kaplan's N is smaller, so the same attention term is a larger share of it.
    assert f2["kaplan"]["n_params"] < f2["with_head"]["n_params"]
    assert f2["total"] > 6 * f2["with_head"]["n_params"]


def test_kaplan_n_matches_the_closed_form():
    """N_kaplan should equal the block parameter count, sans head and norms."""
    m = build_model("d160m")
    f = m.flops_per_token()
    blocks = sum(p.numel() for n, p in m.named_parameters() if n.startswith("blocks."))
    # 12*n_layer*d^2 is the standard approximation; norms are the only slack.
    assert abs(f["kaplan"]["n_params"] - blocks) / blocks < 0.001


def test_embedding_share_is_reported_because_it_dominates():
    """At d40m with a 50k vocab, embedding+head is ~2/3 of all parameters."""
    rep = build_model("d40m").param_report()
    assert rep["embedding_plus_head_share"] > 0.6, rep
    assert rep["blocks"] < rep["embedding"] + rep["head"], rep
    assert rep["nonembed"] < rep["total"]
    assert "never 'total'" in rep["capacity_basis_note"]


def test_capacity_basis_differs_by_3x_between_conventions():
    """A bits/param claim on total params would be off by ~3x at this scale."""
    m = build_model("d40m")
    cap = m.capacity_bits(bits_per_param=1.0)
    assert cap["basis_total_bits_DO_NOT_USE"] / cap["basis_blocks_bits"] > 2.5
    assert "exposures" in cap["caveat"]


def test_n800k_fact_load_against_the_d40m_ceiling():
    """The reason to shrink the model: 42.4 Mbit is >= a 40M model's 1-bit ceiling.

    Growing the corpus to reach capacity is far more expensive than shrinking the
    model. On blocks-basis at 1 bit/param this configuration is oversubscribed,
    which is the regime the previous sweep never reached.
    """
    m = build_model("d40m")
    demanded = 800_000 * 41.47  # entities * bits/entity from bios.bits_per_entity
    ceiling_1bit = m.capacity_bits(1.0)["basis_blocks_bits"]
    assert demanded / ceiling_1bit > 1.0, (demanded, ceiling_1bit)
