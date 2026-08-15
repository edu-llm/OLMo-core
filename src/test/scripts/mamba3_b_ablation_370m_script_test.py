"""
Tests for the Mamba-3 ``b=2`` vs ``b=3`` ablation entrypoint
(``OLMo3-370M-mamba3-b-ablation.py``).

The script's job is not to train -- ``OLMo3-370M-dolma2mix.py`` already does that -- it is to make
the *comparison* auditable. So most of what is pinned here is the contract that the two arms differ
in exactly one architectural field and in nothing else that could explain a difference in loss:
same learning rate, same data, same seeds, same optimizer, same everything.

Everything is offline: no S3 (the dataset is never built), no GPU, no distributed. Model configs
are built on ``meta``, which allocates nothing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path("src/scripts/train/OLMo3/OLMo3-370M-mamba3-b-ablation.py")

#: The one field the treatment is allowed to move. Anything else in the diff is a confounder.
TREATMENT_FIELD = "block.mamba3.sequence_mixer.rotation_block_size"

#: ``b=3`` needs three angles per block where ``b=2`` needs one: 96 against 48 columns of
#: ``theta_proj``, over twelve Mamba layers of width 1024.
INTRINSIC_B3_PARAMETER_COST = (96 - 48) * 1024 * 12


@pytest.fixture(scope="module")
def script() -> ModuleType:
    assert SCRIPT_PATH.exists(), f"{SCRIPT_PATH} not found; run pytest from the repo root"
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("olmo3_370m_mamba3_b_ablation", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------------------------------------
# The arms
# ------------------------------------------------------------------------------------------


def test_there_are_exactly_two_arms_and_b2_is_the_control(script):
    """
    Two arms, and the baseline is named as the control.

    Which arm is the control is not cosmetic: it decides whose parameter count sets the shared
    learning rate, so the treatment inherits the baseline's recipe rather than the other way round.
    """
    assert script.ARMS == ("b2", "b3")
    assert script.CONTROL_ARM == "b2"
    assert script.ARM_BLOCK_SIZE == {"b2": 2, "b3": 3}


def test_each_arm_builds_its_own_block_size(script):
    """The failure this guards is an arm that silently trains as the other one."""
    for arm, expected in script.ARM_BLOCK_SIZE.items():
        config = script.build_model_config(arm)
        assert isinstance(config.block, dict)
        assert config.block["mamba3"].sequence_mixer.rotation_block_size == expected


def test_an_unknown_arm_is_refused(script):
    with pytest.raises(SystemExit):
        script.build_model_config("b4")


# ------------------------------------------------------------------------------------------
# The single-field contract -- the whole point of the script
# ------------------------------------------------------------------------------------------


def test_the_two_arms_differ_in_exactly_one_config_field(script):
    """
    A flattened diff of the two model configs must contain the block size and nothing else.

    This is the strongest statement the comparison can make and it is cheap to check, so it is
    checked mechanically rather than asserted in prose. The extra ``theta_proj`` columns ``b=3``
    carries do not appear here because they are *derived* at build time from this one field.
    """
    difference = script.arm_config_difference(
        script.build_model_config("b2"), script.build_model_config("b3")
    )

    assert set(difference) == {TREATMENT_FIELD}
    assert difference[TREATMENT_FIELD] == (2, 3)


def test_parameter_matching_moves_the_ffn_and_says_so(script):
    """
    Under ``--param-match ffn`` the FFN width becomes a second difference, by design.

    That is a legitimate choice -- it is what the Mamba-3 paper itself does to parameter-match its
    MIMO variants -- but it must show up in the diff rather than being smuggled in.
    """
    difference = script.arm_config_difference(
        script.build_model_config("b2", param_match="ffn"),
        script.build_model_config("b3", param_match="ffn"),
    )

    assert set(difference) == {
        TREATMENT_FIELD,
        "block.mamba3.feed_forward.hidden_size",
        "block.attn.feed_forward.hidden_size",
    }


def test_the_two_arms_train_identically_apart_from_the_treatment(script):
    """
    The same check against the *whole* experiment config, not just the model.

    The model diff cannot see the learning rate, the seed, the optimizer groups, the schedule, the
    token budget or the trainer -- and those are where a confounder is most likely to be
    introduced by accident, because the dense builder derives some of them from the model. Built
    against an in-repo mixture YAML so it stays offline; the dataset itself is never built.
    """
    local_mix = Path("src/scripts/b200/olmo150b-scaled-nowiki.yaml")
    assert local_mix.exists(), "the offline mixture YAML this test builds against has moved"

    def experiment(arm: str) -> dict:
        opts = script.parse_args(
            ["train", f"t-{arm}", "--arm", arm, "--dry-run", "--data-config", str(local_mix)]
        )
        return script._flatten(script.build_config(opts, []).as_config_dict())

    # Where the two cells write and what they are called. Not model, data or optimization.
    run_identity = {"trainer.save_folder", "trainer.callbacks.wandb.name"}

    b2, b3 = experiment("b2"), experiment("b3")
    differing = {key for key in set(b2) | set(b3) if b2.get(key) != b3.get(key)} - run_identity

    assert differing == {
        "model." + TREATMENT_FIELD,
        # The same field restated, so the sentinel can refuse a run that trains as the other arm.
        "trainer.callbacks.mamba3_sentinel.expected_rotation_block_size",
    }, f"unexpected differences between the arms: {sorted(differing)}"


def test_verify_arms_rejects_a_tampered_arm(script):
    """
    The contract check has to actually fail when the contract is broken.

    A checker that only ever passes is worse than no checker, so break one field deliberately and
    require a refusal naming it.
    """
    b2 = script.build_model_config("b2")
    b3 = script.build_model_config("b3")
    b3.block["mamba3"].sequence_mixer.d_state = 96

    with pytest.raises(script.ArmContractError, match="d_state"):
        script.verify_arms({"b2": b2, "b3": b3}, param_match="off")


# ------------------------------------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------------------------------------


def test_default_arms_land_on_the_370m_reference(script):
    """
    Both arms must be a 370M model, or the label on the experiment is wrong.

    ``olmo3_370M`` reports 371,262,464 non-embedding parameters; the FFN width is solved so the
    Mamba arms land within a small tolerance of it despite the published expand factor of 2 making
    the mixer far wider than the preset it replaces.
    """
    reference = 371_262_464
    for arm in script.ARMS:
        count = script.build_model_config(arm).num_non_embedding_params
        drift = abs(count - reference) / reference
        assert drift < 0.005, f"{arm} is {count:,}, {drift:.2%} from the 370M reference"


def test_without_param_matching_b3_carries_only_its_intrinsic_surcharge(script):
    """
    The default keeps the FFN identical, so ``b=3`` is heavier by exactly the angle projection.

    Reported rather than hidden: 0.16% of the model, worth about 1e-4 nats under standard loss
    scaling, and irreducible -- SO(3) cannot be parameterized with fewer angles.
    """
    b2 = script.build_model_config("b2", param_match="off")
    b3 = script.build_model_config("b3", param_match="off")

    assert b3.num_params - b2.num_params == INTRINSIC_B3_PARAMETER_COST
    assert (b3.num_params - b2.num_params) / b2.num_params < 0.002


def test_param_matching_equalizes_the_two_arms_exactly(script):
    """
    ``--param-match ffn`` must be exact, not approximate.

    The surcharge is 12 units of FFN width to the last parameter, so an exact match is available
    and a near-match would mean the solver is wrong.
    """
    b2 = script.build_model_config("b2", param_match="ffn")
    b3 = script.build_model_config("b3", param_match="ffn")

    assert b2.num_params == b3.num_params
    assert b2.num_non_embedding_params == b3.num_non_embedding_params


def test_param_matching_defaults_to_off(script):
    """
    The default is the cleaner statement: one field differs, and the parameters it forces.

    Equalizing instead trades a 0.16% parameter difference for a 0.35% FFN width difference, which
    also makes the two arms' GEMM shapes differ and so perturbs the throughput endpoint.
    """
    assert script.DEFAULT_PARAM_MATCH == "off"
    assert script.PARAM_MATCH_MODES == ("off", "ffn")


# ------------------------------------------------------------------------------------------
# Everything that is held fixed
# ------------------------------------------------------------------------------------------


def test_learning_rate_is_pinned_and_identical_across_arms(script):
    """
    The ladder formula reads the parameter count, so leaving the LR auto-derived would hand the
    two arms different learning rates off the back of the 0.16% parameter gap. That is a silent
    confounder in a comparison whose whole claim is that one field differs.
    """
    opts = script.parse_args(["plan"])
    rates = {arm: script.resolve_learning_rate(opts, arm) for arm in script.ARMS}

    assert len(set(rates.values())) == 1, f"arms must share one learning rate, got {rates}"
    assert rates["b2"] == script.DEFAULT_LEARNING_RATE


def test_learning_rate_does_not_move_when_the_parameter_count_does(script):
    """The pinned rate must be independent of the arm's size, including under param matching."""
    plain = script.parse_args(["plan"])
    matched = script.parse_args(["plan", "--param-match", "ffn"])

    assert script.resolve_learning_rate(plain, "b3") == script.resolve_learning_rate(matched, "b3")


def test_the_data_recipe_is_ten_billion_dolma_tokens_from_s3(script):
    """The experiment as specified: 10B tokens of the dolma2 source mixture on S3."""
    opts = script.parse_args(["plan"])

    assert opts.token_budget == 10_000_000_000
    assert opts.data_config.startswith("s3://")
    assert "dolma2" in opts.data_config


def test_both_arms_of_a_replicate_share_a_seed(script):
    """
    Paired design: the two arms of a replicate see the same tokens in the same order.

    The dense recipe drives data order, the mixture draw and initialization from one ``--seed``, so
    pairing on it pairs all three. Otherwise the comparison spends part of its signal on data-order
    noise, which at two runs it cannot afford.
    """
    cells = script.build_plan(script.parse_args(["plan", "--replicates", "3"]))

    by_replicate: dict[int, set[int]] = {}
    for cell in cells:
        by_replicate.setdefault(cell["replicate"], set()).add(cell["seed"])
    assert by_replicate, "the plan must produce cells"
    for replicate, seeds in by_replicate.items():
        assert len(seeds) == 1, f"replicate {replicate} has seeds {seeds}"


def test_the_plan_covers_every_arm_for_every_replicate(script):
    cells = script.build_plan(script.parse_args(["plan", "--replicates", "2"]))

    assert len(cells) == 4
    assert sorted(cell["arm"] for cell in cells) == ["b2", "b2", "b3", "b3"]
    assert [cell["index"] for cell in cells] == [0, 1, 2, 3]


def test_the_plan_defaults_to_one_replicate_per_arm(script):
    """Two 10B-token runs is the stated experiment; more seeds are opt-in."""
    assert len(script.build_plan(script.parse_args(["plan"]))) == 2


def test_replicate_seeds_are_distinct_and_deterministic(script):
    """A plan that changes between invocations cannot be pre-registered."""
    first = script.build_plan(script.parse_args(["plan", "--replicates", "3"]))
    second = script.build_plan(script.parse_args(["plan", "--replicates", "3"]))

    assert first == second
    assert len({cell["seed"] for cell in first}) == 3, "each replicate needs its own seed"


def test_a_passed_through_seed_is_refused(script):
    """
    ``--seed`` belongs to the replicate, not to the caller.

    It is not one of this script's own flags, so argparse hands it through to the dense parser --
    which takes the *last* ``--seed`` it sees and would silently override the pinned one. That
    unpairs the two arms of a replicate, and neither the config diff nor the manifest can see it,
    because by then the seed looks like an ordinary part of the recipe.
    """
    for form in (["--seed", "1234"], ["--seed=1234"]):
        with pytest.raises(SystemExit):
            script.parse_args(["train", "my-run", "--arm", "b2", *form])


def test_the_replicate_decides_the_seed_that_reaches_the_trainer(script):
    """End to end: the seed the plan reserves is the seed the run is built with."""
    local_mix = Path("src/scripts/b200/olmo150b-scaled-nowiki.yaml")
    for replicate in (0, 1):
        opts = script.parse_args(
            [
                "train",
                "r",
                "--arm",
                "b2",
                "--replicate",
                str(replicate),
                "--dry-run",
                "--data-config",
                str(local_mix),
            ]
        )
        assert script.build_config(opts, []).init_seed == script.SEEDS[replicate]


def test_replicate_zero_reuses_the_ladder_seed(script):
    """
    So the first pair sits on the same data order as the repository's other 370M ladder runs, and
    can be read beside them rather than only against each other.
    """
    assert script.SEEDS[0] == script.dolma2.DEFAULT_SEED


def test_more_replicates_than_reserved_seeds_is_refused(script):
    """Better to refuse than to invent a seed nobody wrote down."""
    with pytest.raises(SystemExit):
        script.parse_args(["plan", "--replicates", str(len(script.SEEDS) + 1)])


# ------------------------------------------------------------------------------------------
# The manifest a write-up has to be able to quote
# ------------------------------------------------------------------------------------------


def test_the_manifest_records_what_a_writeup_needs(script):
    manifest = script.verify_arms(
        {arm: script.build_model_config(arm) for arm in script.ARMS}, param_match="off"
    )

    assert manifest["param_match"] == "off"
    assert set(manifest["arms"]) == set(script.ARMS)
    assert manifest["config_difference"] == {TREATMENT_FIELD: [2, 3]}
    for arm, row in manifest["arms"].items():
        assert row["rotation_block_size"] == script.ARM_BLOCK_SIZE[arm]
        assert row["num_params"] > 0
        assert row["num_non_embedding_params"] > 0
        assert row["intermediate_size"] > 0


def test_the_manifest_is_json_serializable(script):
    """It is written next to the checkpoints and read by whoever writes the results up."""
    import json

    manifest = script.verify_arms(
        {arm: script.build_model_config(arm) for arm in script.ARMS}, param_match="off"
    )

    assert json.loads(json.dumps(manifest)) == manifest


# ------------------------------------------------------------------------------------------
# CLI surface
# ------------------------------------------------------------------------------------------


def test_training_requires_an_explicit_arm(script):
    """No default arm: a run that forgets to say which one it is would be unattributable."""
    with pytest.raises(SystemExit):
        script.parse_args(["train", "my-run"])


def test_training_accepts_an_arm_and_a_replicate(script):
    opts = script.parse_args(["train", "my-run", "--arm", "b3", "--replicate", "0"])

    assert opts.command == "train" and opts.arm == "b3" and opts.replicate == 0


def test_an_unknown_arm_is_refused_at_the_cli(script):
    with pytest.raises(SystemExit):
        script.parse_args(["train", "my-run", "--arm", "b4"])


def test_the_rotation_scan_is_named_rather_than_left_to_the_environment(script):
    """
    Left in ``MAMBA3_ROTATION_SCAN_IMPL`` the choice is invisible to the checkpoint, and a relaunch
    that forgets the export silently drops to the chunked scan -- 2.2x slower, and it raises
    nothing. Naming it in the config puts it in the saved config and the startup banner.
    """
    for arm in script.ARMS:
        mixer = script.build_model_config(arm).block["mamba3"].sequence_mixer
        assert mixer.rotation_scan_impl == "quaternion"
