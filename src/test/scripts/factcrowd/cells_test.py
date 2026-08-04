"""
What a cell guarantees: one stated quantity, a derived one, and a grid whose labels are true.

The refusals matter most. A cell that states both its demand and its entity count can have them
disagree, and then it lands on the trend plot at the wrong x while every number in the run stays
plausible. So stating both is refused rather than reconciled, and every committed config is resolved
here so a hand edit cannot ship a cell that does not add up.

``train_cell.py`` is imported too. It needs no ``torch`` until it builds a trainer, which is what lets
``--dry-run`` catch config and arithmetic errors on any machine -- and which is the arrangement that a
torch-gated bug earlier in this branch argued for.
"""

from pathlib import Path

import pytest
from factcrowd import cells as C
from factcrowd.corpus import values as V
from factcrowd.ladder import rho, sizes

from olmo_core.exceptions import OLMoConfigurationError

CONFIG_ROOT = Path("src/scripts/train/factcrowd/configs/cells")


# --- the grid ---------------------------------------------------------------------------------------


def test_the_first_run_is_seventeen_cells_across_three_rows():
    """
    13M and 28M at all five demands, 64M at four, and a reasoning-only control on each row.

    The 113M row and 64M's highest demand are omitted. The controls are what the rest of each row is
    compared against, so a row that generated without one would have no zero-fact reference point.
    """
    cells = C.first_run_cells()
    assert len(cells) == 17
    per_row = {row: sum(1 for c in cells if c.row == row) for row in ("13M", "28M", "64M")}
    assert per_row == {"13M": 6, "28M": 6, "64M": 5}
    assert [c.cell_id for c in cells if c.is_control] == ["13m_ctrl", "28m_ctrl", "64m_ctrl"]
    assert not any(c.row == "113M" for c in cells)


# --- the reasoning-only control ----------------------------------------------------------------------


def test_the_control_has_no_facts_and_states_it_by_demand():
    """
    Zero facts is stated as demand 0, and resolves to zero entities without going near the solver.

    ``rho.solve`` refuses a zero target and should keep refusing it: its linear path divides by
    bits_per_entity and its name-term path is non-monotone there, so a solver that answered would be
    answering by accident. The control is the one cell whose entity count is stated rather than solved.
    """
    control = C.CellSpec(
        cell_id="13m_ctrl", row="13M", demand_bits_per_param=0.0, reasoning_tokens=1_000_000_000
    )
    assert control.is_control
    resolved = control.resolve()
    assert resolved.n_entities == 0
    assert resolved.demand_per_non_embedding_param == 0.0
    assert resolved.attribute_bits == 0 and resolved.name_bits == 0
    assert resolved.fact_tokens(69.2) == 0
    assert resolved.total_tokens(69.2) == 1_000_000_000

    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        rho.solve(
            control.non_embedding_params,
            0.0,
            bits_per_entity=V.bios_bits_per_entity(),
            name_space=V.NAME_SPACE,
        )


def test_a_cell_with_neither_facts_nor_reasoning_is_refused():
    """Demand 0 and no reasoning is a run with no data at all, which should not reach a GPU."""
    with pytest.raises(OLMoConfigurationError, match="no facts and no reasoning"):
        C.CellSpec(cell_id="empty", row="13M", demand_bits_per_param=0.0)
    with pytest.raises(OLMoConfigurationError, match="must not be negative"):
        C.CellSpec(cell_id="neg", row="13M", demand_bits_per_param=-1.0)


def test_the_control_gets_the_same_unrelated_exposure_as_the_cells_it_anchors():
    """
    The confound this arithmetic exists to prevent, and it was live for one commit.

    The budget is per *slice*. Splitting a fixed total between however many slices a cell carries gave
    the control -- which carries only the unrelated slice -- twice the unrelated exposure of every cell
    it is the reference for. Its score would then have beaten theirs for a reason that has nothing to do
    with fact load, which is the one inference the control exists to support.
    """
    cells = {cell.cell_id: cell for cell in C.first_run_cells()}
    control, demand = cells["13m_ctrl"], cells["13m_d1p2"]

    assert control.reasoning_slice_names == ("mano",)
    assert demand.reasoning_slice_names == ("mano", "compare")
    # The same *unrelated* budget in both, which is the exposure the comparison rests on.
    assert control.slice_budget("mano") == demand.slice_budget("mano") == C.REASONING_TOKENS
    # The totals differ, which is the correct consequence rather than a violated invariant: the related
    # slice carries its own budget, sized on per-entity coverage rather than on parity.
    assert control.resolve().reasoning_total == C.REASONING_TOKENS
    assert demand.resolve().reasoning_total == C.REASONING_TOKENS + C.RELATED_REASONING_TOKENS
    assert demand.slice_budget("compare") == C.RELATED_REASONING_TOKENS
    with pytest.raises(OLMoConfigurationError, match="carries"):
        control.slice_budget("compare")


def test_the_related_slice_does_not_out_supervise_the_facts_it_depends_on():
    """
    Its budget is sized on per-entity coverage, because 1.0B would have taught the ranks outright.

    The slice names two of the fixed 25,000 probe entities per item, so at the unrelated slice's budget
    each entity's birth-year rank is supervised 4,211 times against the 200 exposures 3.3 fixes for the
    fact slice -- 21x. At that rate the slice supplies its own answer and "needs two facts" stops being
    true, so a decline would say nothing about fact access. This pins the ratio near one.
    """
    cell = [c for c in C.first_run_cells() if c.cell_id == "28m_d1p2"][0]
    items = cell.slice_budget("compare") // 19  # the compare item width
    mentions_per_entity = 2 * items / 25_000
    assert 150 < mentions_per_entity < 400, mentions_per_entity
    assert mentions_per_entity < 3 * cell.exposures, mentions_per_entity


def test_the_entropy_axis_carries_the_unrelated_slice_alone():
    """
    Its attributes are positional composites with no ordinal field, so there is nothing to compare.

    Checked here rather than only where the task is built, because the token arithmetic reads this and
    a cell whose declared slices differ from its built ones has an unreproducible step count.
    """
    for cell in C.entropy_sweep_cells("28M"):
        assert cell.reasoning_slice_names == ("mano",)
        assert not cell.is_control


def test_a_facts_only_cell_declares_no_slices():
    """Zero reasoning tokens means no slices, whatever axis the cell is on."""
    cell = C.CellSpec(cell_id="facts", row="13M", demand_bits_per_param=1.2, reasoning_tokens=0)
    assert cell.reasoning_slice_names == ()
    assert cell.resolve().reasoning_total == 0


def test_every_count_cell_lands_on_its_intended_rho():
    """
    Demand 0.30/0.60/1.20/2.40/4.80 reads as rho 0.25/0.5/1/2/4 at the declared constant.

    The cell is placed by the demand -- rho is a presentation transform -- but the mapping has to hold
    or the grid cannot be read against the literature.
    """
    expected = [0.25, 0.5, 1.0, 2.0, 4.0]
    for row in ("13M", "28M"):
        got = [
            c.resolve().rho
            for c in C.first_run_cells()
            if c.row == row
            and not c.is_control  # the control sits at 0 by definition, not by demand
        ]
        # 1e-4 rather than exact: n_entities is an integer, so each cell carries a rounding residual
        # of order 1/n. Far inside the 1% tolerance rho.check enforces.
        assert got == pytest.approx(expected, abs=1e-4)


def test_the_entropy_sweep_holds_entity_count_and_token_count_fixed():
    """
    The axis's whole justification, checked at the cell layer rather than only in the schema.

    Six cells, one entity count, one token count, one step count -- so tokens, steps, schedule position
    and cumulative weight decay are identical while demand sweeps over a 25x range.
    """
    cells = C.entropy_sweep_cells("28M")
    assert len(cells) == 6
    resolved = [cell.resolve() for cell in cells]

    assert len({r.n_entities for r in resolved}) == 1
    assert len({r.total_tokens(69.2) for r in resolved}) == 1
    assert len({r.steps(69.2) for r in resolved}) == 1

    demands = [r.demand_per_non_embedding_param for r in resolved]
    assert demands == sorted(demands)
    assert demands[-1] / demands[0] > 20


def test_the_entropy_midpoint_sits_at_the_bios_anchor():
    """``b=8`` is 48 bits/entity against bioS's 47.592, so its demand reads as rho about 1."""
    midpoint = next(c for c in C.entropy_sweep_cells("28M") if c.bits_per_attribute == 8)
    assert midpoint.resolve().rho == pytest.approx(1.0, abs=0.02)


def test_the_entropy_sweep_pins_its_entity_count_to_the_demand_one_cell():
    """
    So that the sweep straddles the bioS anchor rather than sitting beside it.

    The count comes from solving the count axis at demand 1.20, which is what makes ``b=8`` land there.
    """
    cells = C.entropy_sweep_cells("28M")
    expected = rho.solve(
        sizes.non_embedding_params(sizes.row("28M").d_model),
        1.20,
        bits_per_entity=V.bios_bits_per_entity(),
        name_space=V.NAME_SPACE,
    ).n_entities
    assert {c.n_entities for c in cells} == {expected}


# --- the stated-versus-derived rule -----------------------------------------------------------------


def test_a_count_cell_stating_its_entity_count_is_refused():
    """
    Stating both is how a cell comes to sit at a demand other than its label.

    Derivation is the point: one number is in the config and the other is arithmetic.
    """
    with pytest.raises(OLMoConfigurationError, match="derived from the demand"):
        C.CellSpec(cell_id="x", row="28M", demand_bits_per_param=1.2, n_entities=500_000)


def test_a_count_cell_without_a_demand_is_refused():
    """There would be nothing to place it by."""
    with pytest.raises(OLMoConfigurationError, match="placed by"):
        C.CellSpec(cell_id="x", row="28M")


def test_a_count_cell_stating_bits_per_attribute_is_refused():
    """That field belongs to the entropy axis; the count axis uses the bioS schema's own bits."""
    with pytest.raises(OLMoConfigurationError, match="belongs to the entropy axis"):
        C.CellSpec(cell_id="x", row="28M", demand_bits_per_param=1.2, bits_per_attribute=8)


def test_an_entropy_cell_needs_both_its_stated_quantities():
    """Entity count is what it holds fixed; entropy is what it sweeps."""
    with pytest.raises(OLMoConfigurationError, match="states both"):
        C.CellSpec(cell_id="x", row="28M", sweep="entropy", bits_per_attribute=8)
    with pytest.raises(OLMoConfigurationError, match="states both"):
        C.CellSpec(cell_id="x", row="28M", sweep="entropy", n_entities=1000)


def test_an_entropy_cell_stating_a_demand_is_refused():
    """Demand is derived there, and a stated one could disagree with the entropy."""
    with pytest.raises(OLMoConfigurationError, match="derived on the entropy axis"):
        C.CellSpec(
            cell_id="x",
            row="28M",
            sweep="entropy",
            bits_per_attribute=8,
            n_entities=1000,
            demand_bits_per_param=1.2,
        )


def test_an_unknown_row_is_refused_and_the_ladder_is_listed():
    """The message fires in front of somebody editing a config."""
    with pytest.raises(OLMoConfigurationError, match="no ladder row"):
        C.CellSpec(cell_id="x", row="60M", demand_bits_per_param=1.2)


def test_an_unknown_sweep_is_refused():
    """Silently defaulting would run the wrong axis under the right name."""
    with pytest.raises(OLMoConfigurationError, match="'sweep' must be"):
        C.CellSpec(cell_id="x", row="28M", sweep="widthwise", demand_bits_per_param=1.2)


# --- hyperparameters are part of the cell -----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"exposures": 0}, "'exposures' must be positive"),
        ({"sequence_length": 0}, "'sequence_length' must be positive"),
        ({"global_batch_size": 0}, "'global_batch_size' must be positive"),
        ({"learning_rate": 0.0}, "'learning_rate' must be positive"),
        ({"warmup_steps": 0}, "'warmup_steps' must be positive"),
        ({"decay_fraction": 0.0}, "'decay_fraction' must be in"),
        ({"decay_fraction": 1.0}, "'decay_fraction' must be in"),
        ({"weight_decay": -1.0}, "must not be negative"),
        ({"seed": -1}, "must not be negative"),
        ({"global_batch_size": 1000}, "must be a multiple of"),
    ],
)
def test_degenerate_hyperparameters_are_refused(overrides, match):
    """
    Every one of these would produce a run, and a wrong one.

    They live on the cell rather than on the launcher because cumulative weight decay already varies
    14.6x across a count row; if the values themselves also drifted between cells there would be
    nothing left to attribute a difference to.
    """
    with pytest.raises(OLMoConfigurationError, match=match):
        C.CellSpec(cell_id="x", row="28M", demand_bits_per_param=1.2, **overrides)


def test_the_grid_shares_one_set_of_hyperparameters():
    """One learning rate, one batch size, one schedule across every cell."""
    cells = C.first_run_cells() + C.entropy_sweep_cells("28M")
    for field in (
        "learning_rate",
        "weight_decay",
        "global_batch_size",
        "sequence_length",
        "warmup_steps",
        "decay_fraction",
        "exposures",
    ):
        assert len({getattr(cell, field) for cell in cells}) == 1, field


def test_cumulative_weight_decay_varies_across_a_count_row():
    """
    The confound the entropy axis exists to remove, quantified at the cell layer.

    Steps scale with demand on the count axis, so ``sum(lr * wd)`` does too -- about 16x across the
    five 28M cells. The entropy axis holds it fixed instead, which is the comparison that matters.
    """
    count = [
        c.resolve().summary(69.2)["cumulative_lr_times_wd"]
        for c in C.first_run_cells()
        if c.row == "28M"
    ]
    assert max(count) / min(count) > 10

    entropy = [
        c.resolve().summary(69.2)["cumulative_lr_times_wd"] for c in C.entropy_sweep_cells("28M")
    ]
    assert len(set(entropy)) == 1


# --- serialisation, and the committed configs -------------------------------------------------------


def test_a_cell_round_trips_through_a_dictionary():
    """A config file is the unit a person edits and a run reproduces."""
    original = C.first_run_cells()[3]
    assert C.CellSpec.from_dict(original.to_dict()) == original


def test_unknown_config_keys_are_refused_rather_than_ignored():
    """
    A typo'd override that silently does nothing is how a cell runs at settings nobody chose.
    """
    raw = C.first_run_cells()[0].to_dict()
    raw["learning_rat"] = 1e-3
    with pytest.raises(OLMoConfigurationError, match="unknown config keys"):
        C.CellSpec.from_dict(raw)


def test_a_config_missing_required_keys_is_refused():
    """``cell_id`` and ``row`` have no sensible default."""
    with pytest.raises(OLMoConfigurationError, match="missing required keys"):
        C.CellSpec.from_dict({"demand_bits_per_param": 1.2})


def test_every_committed_config_loads_and_resolves():
    """
    The test that stops a hand edit shipping a cell that does not add up.

    Resolution runs ``rho.check``, so a config whose demand and entity count disagree fails here rather
    than on a GPU.
    """
    for axis in ("count", "entropy"):
        cells = C.load_cells(CONFIG_ROOT / axis)
        assert cells, axis
        for cell in cells:
            resolved = cell.resolve()
            assert cell.sweep == axis
            if cell.is_control:
                # The one cell with no facts, so the only one whose demand is legitimately zero.
                assert resolved.n_entities == 0
                assert resolved.demand_per_non_embedding_param == 0
                assert cell.reasoning_tokens > 0
            else:
                assert resolved.n_entities > 0
                assert resolved.demand_per_non_embedding_param > 0


def test_the_committed_grid_matches_the_generator():
    """
    Regenerating is a command, so the configs and the code cannot drift.

    Compares cell ids rather than whole objects, since ``reasoning_tokens`` is expected to be edited.
    """
    assert {c.cell_id for c in C.load_cells(CONFIG_ROOT / "count")} == {
        c.cell_id for c in C.first_run_cells()
    }
    assert {c.cell_id for c in C.load_cells(CONFIG_ROOT / "entropy")} == {
        c.cell_id for c in C.entropy_sweep_cells("28M")
    }


def test_the_two_axes_live_in_separate_directories():
    """
    A fan-out maps an array index to a cell by position, so its size must be what ``ls`` says.

    Sharing one directory would let ``--row 28M`` pick up twelve cells where the submission asked for
    six, and run the wrong cell under the right name.
    """
    count_rows = {c.row for c in C.load_cells(CONFIG_ROOT / "count")}
    entropy_rows = {c.row for c in C.load_cells(CONFIG_ROOT / "entropy")}
    assert "28M" in count_rows and "28M" in entropy_rows
    # Six: five demands plus the row's reasoning-only control. This number is the submission's
    # fanout_size, so it is asserted rather than derived.
    assert len([c for c in C.load_cells(CONFIG_ROOT / "count") if c.row == "28M"]) == 6
    assert len([c for c in C.load_cells(CONFIG_ROOT / "entropy") if c.row == "28M"]) == 6


def test_load_cells_is_sorted_so_a_fanout_index_is_stable():
    """
    The index-to-cell mapping must be a function of the directory and nothing else.

    A mapping that changed between submission and execution would run a different cell under the name
    the approver saw.
    """
    ids = [cell.cell_id for cell in C.load_cells(CONFIG_ROOT / "count")]
    assert ids == sorted(ids)


def test_writing_and_reloading_a_grid_round_trips(tmp_path):
    """Regeneration has to be idempotent, or the committed configs drift every time it runs."""
    cells = C.first_run_cells(reasoning_tokens=5_000)
    C.write_cells(cells, tmp_path)
    assert C.load_cells(tmp_path) == cells


def test_load_cell_refuses_a_file_that_is_not_a_mapping(tmp_path):
    """A YAML list or scalar would otherwise reach ``from_dict`` as a type error."""
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(OLMoConfigurationError, match="must hold a mapping"):
        C.load_cell(path)


def test_load_cells_refuses_an_empty_directory(tmp_path):
    """Otherwise a fan-out of size zero submits and does nothing."""
    with pytest.raises(OLMoConfigurationError, match="no cell configs"):
        C.load_cells(tmp_path)


# --- the entry point, without torch -----------------------------------------------------------------


def test_train_cell_imports_and_resolves_a_fanout_index_without_torch():
    """
    ``--dry-run`` has to work on a laptop, which is what makes it worth running before a submission.

    Only ``build_trainer`` needs the GPU stack, and it imports inside the function. An earlier module
    in this branch put its only real logic behind a module-level ``torch`` import; its tests skipped,
    and a call that raised ``TypeError`` for every input passed review.
    """
    import argparse
    import importlib.util
    import sys

    path = Path("src/scripts/train/factcrowd/train_cell.py")
    spec = importlib.util.spec_from_file_location("factcrowd_train_cell", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Index 0 is the control: 'ctrl' sorts before 'd0p3', so adding it shifted every demand cell by
    # one. That shift is why the mapping is asserted here at all -- a submission that named a
    # fanout_size from an older directory would run a different cell under the name it was approved as.
    args = argparse.Namespace(cell=None, row="28M", sweep="count", cell_index="0")
    assert module.resolve_cell(args).cell_id == "28m_ctrl"

    args = argparse.Namespace(cell=None, row="28M", sweep="count", cell_index="3")
    chosen = module.resolve_cell(args)
    assert chosen.row == "28M"
    assert chosen.cell_id == "28m_d1p2"

    args.cell_index = "99"
    with pytest.raises(OLMoConfigurationError, match="out of range"):
        module.resolve_cell(args)

    args.row = None
    with pytest.raises(OLMoConfigurationError, match="either --cell"):
        module.resolve_cell(args)


def test_every_committed_cell_yields_all_ten_checkpoints():
    """
    The log-spaced schedule collapses on a short run, and a bits curve read off five points is not
    the one the design specifies.

    ``train_cell.py`` warns when it collapses and refuses below three. This asserts no real cell is
    anywhere near either.
    """
    fractions = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.50, 0.75, 1.00)
    for cell in C.load_cells(CONFIG_ROOT / "count") + C.load_cells(CONFIG_ROOT / "entropy"):
        steps = cell.resolve().steps(69.2)
        distinct = {min(steps, max(1, int(fraction * steps))) for fraction in fractions}
        assert len(distinct) == len(fractions), (cell.cell_id, steps, sorted(distinct))


def test_the_total_parameter_basis_is_real_only_when_a_vocabulary_is_supplied():
    """
    Without one, both bases report the non-embedding count and the dual-basis reporting is one basis
    printed twice.

    Worth a test because the failure is silent and looks like agreement: PRD 3 argues at length that a
    design which quietly picks one basis loses the cross-size comparability the size axis exists for,
    and the code satisfied that argument with ``total_params = params + 0``.
    """
    cell = [c for c in C.first_run_cells() if c.cell_id == "13m_d1p2"][0]

    blind = cell.resolve()
    assert blind.demand_per_total_param == blind.demand_per_non_embedding_param

    seeing = cell.resolve(vocab_size=3_584)
    assert seeing.n_entities == blind.n_entities  # the vocabulary moves the basis, not the corpus
    assert seeing.demand_per_total_param < seeing.demand_per_non_embedding_param
    ratio = seeing.demand_per_non_embedding_param / seeing.demand_per_total_param
    # 1.073x with our closed word-level vocabulary, against the 1.650x a tied 32k BPE would give. The
    # magnitude matters: it is why this is a precaution rather than a correction that moves a result.
    assert ratio == pytest.approx(1.073, abs=0.005), ratio


def test_the_basis_gap_stays_monotone_in_model_size():
    """
    The shape PRD 3 rests on, checked at our own vocabulary rather than at the 32k it was derived for.

    The embedding table is one tied matrix of ``d_model x vocab``, so it grows linearly in width while
    the body grows quadratically -- the gap must therefore shrink as the rows get wider, whatever the
    vocabulary. A ladder where it did not would mean the widths were wrong.
    """
    ratios = []
    for row in ("13M", "28M", "64M", "113M"):
        cell = C.CellSpec(
            cell_id=f"{row.lower()}_x", row=row, demand_bits_per_param=1.2, reasoning_tokens=1
        ).resolve(vocab_size=3_584)
        ratios.append(cell.demand_per_non_embedding_param / cell.demand_per_total_param)
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[0] == pytest.approx(1.073, abs=0.005)
    assert ratios[-1] == pytest.approx(1.024, abs=0.005)
