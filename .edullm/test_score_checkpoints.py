"""The downstream scoring path, asserted rather than eyeballed.

Run with ``pytest -v .edullm/test_score_checkpoints.py``.

WHAT THESE ARE PROTECTING. A scoring job's failures are quiet. Loading the wrong arm's
weights raises, so that one is loud; scoring the right arm under the wrong replicate's name,
averaging a trivial canary into a headline, reading the tokenizer off the public internet,
or reporting an accuracy that is pinned at chance all produce a number and no error, and the
number goes into a write-up. Every test here is one of those.

The tests that need ``ai2-olmo-eval`` skip without it, and say so rather than passing
vacuously. It is the ``eval`` extra in ``pyproject.toml``.
"""

import json
import os
import pathlib
import shlex
import socket
import sys

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms as arms  # noqa: E402
import score_checkpoints as score  # noqa: E402
import train_hyper_connections  # noqa: E402
import train_on_corpus  # noqa: E402


def has_olmo_eval() -> bool:
    """
    Whether the harness that carries the task data is installed here.

    :returns: ``True`` if ``olmo_eval`` imports.
    """
    try:
        import olmo_eval  # noqa: F401
    except ImportError:
        return False
    return True


needs_olmo_eval = pytest.mark.skipif(
    not has_olmo_eval(),
    reason="ai2-olmo-eval is not installed; it is the 'eval' extra and it carries the tasks",
)


# ---------------------------------------------------------------------------------------
# Which cell is which, which is the mapping the whole job rests on.
# ---------------------------------------------------------------------------------------


def test_the_scoring_fanout_is_the_same_cells_the_tranche_trained():
    """
    Cell 7 has to be the same ``(arm, seed)`` here that it was during training, and the way
    that is guaranteed is that it is the same function reading the same table rather than
    two implementations that happen to agree today.

    NOTHING HERE PINS A CELL COUNT, WHICH IS THE POINT. The tranche has been nine cells,
    fifteen and twenty inside one module, and each move was a two-line edit to the arm table.
    A test asserting fifteen would have been the thing that broke rather than the thing that
    noticed.
    """
    assert score.FANOUT_INDEX_PARAMETER_CELL == train_hyper_connections.FANOUT_INDEX_PARAMETER_CELL

    for index, (arm, seed) in enumerate(arms.TRANCHE_CELLS):
        environ = {
            train_hyper_connections.FANOUT_INDEX_VARIABLE: str(index),
            train_hyper_connections.FANOUT_PARAMETER_VARIABLE: score.FANOUT_INDEX_PARAMETER_CELL,
        }
        opts = score.build_parser().parse_args(["run"])
        assert score.resolve_target(opts, environ)[:3] == (arm, seed, score.FINAL_STEP)


def test_the_ladder_is_every_checkpoint_of_every_cell_and_nothing_else():
    """
    One rung per checkpoint per cell. A 6,000-step run at ``--save-interval 500`` writes
    thirteen: one at step zero and twelve on the interval, the last of which is the final
    step, which is the count ``train_hyper_connections``'s parser comment states.
    """
    rungs = 1 + arms.TRANCHE_STEPS // arms.TRANCHE_SAVE_INTERVAL
    assert score.LADDER_STEPS[0] == 0
    assert score.LADDER_STEPS[-1] == arms.TRANCHE_STEPS == score.FINAL_STEP
    assert len(score.LADDER_STEPS) == rungs == 13
    assert len(score.LADDER_CELLS) == len(arms.TRANCHE_CELLS) * rungs
    assert score.cell_of_the_ladder(0) == ("baseline", 0, 0)
    assert score.cell_of_the_ladder(len(score.LADDER_CELLS) - 1) == (
        *arms.TRANCHE_CELLS[-1],
        arms.TRANCHE_STEPS,
    )
    with pytest.raises(IndexError):
        score.cell_of_the_ladder(len(score.LADDER_CELLS))


@pytest.mark.parametrize("flag,value", [("--arm", "faithful"), ("--seed", "3"), ("--step", "500")])
def test_a_flag_the_ladder_index_owns_is_refused_rather_than_honoured(flag, value):
    """
    Every cell of a fan-out is handed one command. A ``--step`` written into the spec would
    score one checkpoint 195 times and the output would read as a ladder that never moved.
    """
    environ = {
        train_hyper_connections.FANOUT_INDEX_VARIABLE: "4",
        train_hyper_connections.FANOUT_PARAMETER_VARIABLE: score.FANOUT_INDEX_PARAMETER_LADDER,
    }
    opts = score.build_parser().parse_args(["run", flag, value])
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.resolve_target(opts, environ)
    assert flag in refusal.value.explanation


def test_a_cell_index_under_the_wrong_label_is_refused():
    """
    An index that means a checkpoint sweep read as a replicate number is the same accident
    ``resolve_seed`` is shaped around, wearing a different hat.
    """
    environ = {
        train_hyper_connections.FANOUT_INDEX_VARIABLE: "2",
        train_hyper_connections.FANOUT_PARAMETER_VARIABLE: "something-else",
    }
    opts = score.build_parser().parse_args(["run"])
    with pytest.raises(train_on_corpus.Refusal):
        score.resolve_target(opts, environ)


def test_where_a_cell_looks_is_where_the_platform_put_it():
    """
    ``edullm_platform.execution.FANOUT_PROLOGUE`` appends ``cell-$AWS_BATCH_JOB_ARRAY_INDEX/``
    to the output prefix and defines ``$EDULLM_CHECKPOINT_DIR`` as that plus ``checkpoints/``.
    Each training stage fanned out on the seed, so a stage cell's index is its seed.
    """
    root = "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_0199/"
    assert score.checkpoint_uri(root, 3, 6000) == (
        "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_0199/"
        "cell-3/checkpoints/step6000"
    )
    # A trailing slash on the root must not double up, because S3 treats "//" as a real
    # empty path segment and the object would simply not be there.
    assert "//" not in score.checkpoint_uri(root, 0, 0).removeprefix("s3://")


@pytest.mark.parametrize(
    "value", ["baseline", "nosucharm=s3://x", "baseline=s3://x,faithful=s3://y"]
)
def test_an_arm_run_that_is_not_arm_equals_uri_is_refused(value):
    with pytest.raises(train_on_corpus.Refusal):
        score.parse_arm_runs([value])


def test_naming_one_arm_twice_is_refused_rather_than_last_one_winning():
    with pytest.raises(train_on_corpus.Refusal):
        score.parse_arm_runs(["baseline=s3://a", "baseline=s3://b"])


# ---------------------------------------------------------------------------------------
# Which model a cell builds, which is the thing the task asked to be explicit and tested.
# ---------------------------------------------------------------------------------------


def test_each_funded_arm_builds_the_architecture_its_row_claims():
    """
    The three funded arms do not share an architecture, so a loader that guessed would put
    a baseline's weights into a hyper-connection model or the reverse.
    """
    from olmo_core.nn.residual_stream import HyperConnectionMode
    from olmo_core.nn.transformer import TransformerBlockType

    baseline = score.model_config_for("baseline", 0)
    assert baseline.block.name == TransformerBlockType.reordered_norm
    assert baseline.block.hyper_connections is None

    faithful = score.model_config_for("faithful", 0)
    assert faithful.block.name == TransformerBlockType.hyper_connection_reordered_norm
    assert faithful.block.hyper_connections.mode == HyperConnectionMode.full
    assert faithful.block.hyper_connections.n_lanes == arms.N_LANES

    output_only = score.model_config_for("output-only", 0)
    assert output_only.block.hyper_connections.mode == HyperConnectionMode.output


def test_the_vocabulary_is_the_one_the_arms_were_built_at():
    """
    ``build_config`` pads dolma2's 100,278 to a multiple of 128. A scoring cell that built
    100,278 would fail the load on a shape mismatch, which is loud -- but deriving it means
    a tokenizer change moves both together instead of leaving a literal behind.
    """
    assert score.PADDED_VOCAB_SIZE == 100_352


def test_the_replicate_moves_the_init_seed_exactly_as_training_moved_it():
    """
    This is what lets the cross-check catch a seed collapse after the fact. Three training
    cells that all resolved to seed 0 wrote three configs carrying the base seed; a scoring
    cell that believes it is seed 2 then finds one that says otherwise.
    """
    base = score.model_config_for("faithful", 0).init_seed
    for seed in range(5):
        assert score.model_config_for("faithful", seed).init_seed == base + seed


def test_the_default_factory_is_the_one_the_tranche_trained_and_not_the_platforms():
    """
    The platform's ``--model-factory`` default is ``olmo2_190M``. A scoring cell that
    quietly built one would report a number about a model nobody trained.
    """
    assert score.DEFAULT_MODEL_FACTORY == "hc_370M"
    assert score.build_parser().parse_args(["run"]).model_factory == "hc_370M"


def test_an_unknown_arm_or_factory_is_refused_with_a_stage():
    with pytest.raises(train_on_corpus.Refusal):
        score.model_config_for("no-such-arm", 0)
    with pytest.raises(train_on_corpus.Refusal):
        score.model_config_for("baseline", 0, factory="no_such_factory")


# ---------------------------------------------------------------------------------------
# The cross-check against the config the run itself wrote.
# ---------------------------------------------------------------------------------------


def test_a_config_matching_itself_has_no_differences():
    config = score.model_config_for("faithful", 2)
    assert score.config_differences(config.as_config_dict(), config.as_config_dict()) == []


def test_the_cross_check_catches_an_arm_swap():
    """
    ``--arm-run faithful=<the baseline's run>`` is a single wrong word in a spec file, and
    the load would raise on it. This makes the refusal say what went wrong instead.
    """
    expected = score.model_config_for("faithful", 0)
    saved = score.model_config_for("output-only", 0).as_config_dict()
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.check_against_saved_config(expected, saved, arm="faithful", seed=0)
    assert (
        refusal.value.stage
        is score.ScoringStage.THE_CHECKPOINT_DOES_NOT_DESCRIBE_THE_ARM_THIS_CELL_IS
    )
    assert "hyper_connections.mode" in refusal.value.explanation


def test_the_cross_check_catches_a_seed_swap_which_the_load_would_not():
    """
    THE QUIET ONE. Loading seed 3's weights while believing they are seed 4 succeeds -- the
    shapes are identical -- and produces five numbers of which two are one run. A noise
    floor computed from those is too small, and every contrast divided by it looks
    significant.
    """
    expected = score.model_config_for("faithful", 4)
    saved = score.model_config_for("faithful", 3).as_config_dict()
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.check_against_saved_config(expected, saved, arm="faithful", seed=4)
    assert "init_seed" in refusal.value.explanation


def test_the_cross_check_catches_the_wrong_size():
    expected = score.model_config_for("baseline", 0)
    saved = score.model_config_for("baseline", 0, factory="hc_rehearsal").as_config_dict()
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.check_against_saved_config(expected, saved, arm="baseline", seed=0)
    assert "d_model" in refusal.value.explanation


def test_a_checkpoint_with_no_config_warns_and_does_not_refuse():
    """
    A checkpoint from something other than ``train_hyper_connections.py`` has no
    ``config.json``, and refusing would make this program unusable on one. It says so
    instead, and the warning travels in the output document.
    """
    warnings = score.check_against_saved_config(
        score.model_config_for("baseline", 0), None, arm="baseline", seed=0
    )
    assert len(warnings) == 1 and "config.json" in warnings[0]


def test_the_dtype_is_the_one_field_the_two_configs_may_disagree_on():
    """
    An allowlist rather than a list of things to compare, for the reason
    ``STAGE_CONTRAST_EXEMPT`` gives: the failure is a field nobody thought about, and a
    checked list silently permits whatever is not on it.
    """
    assert set(score.CONFIG_COMPARISON_EXEMPT) == {"dtype"}
    expected = score.model_config_for("baseline", 0).as_config_dict()
    saved = dict(expected, dtype="bfloat16")
    assert score.config_differences(expected, saved) == []


# ---------------------------------------------------------------------------------------
# The metric and the suite, which are the two decisions the write-up rests on.
# ---------------------------------------------------------------------------------------


def test_the_primary_metric_is_continuous_and_is_not_accuracy():
    """
    THE DECISION THIS FILE EXISTS TO PIN. At 370M over 4.72B tokens a model is at or near
    chance on multiple-choice accuracy, and a metric pinned at chance has no variance for a
    five-against-five contrast to divide by.
    """
    assert score.PRIMARY_METRIC == "bpb_v2"
    assert "acc" not in score.PRIMARY_METRIC
    assert "len_norm" not in score.PRIMARY_METRIC


def test_the_softmax_metric_is_reported_and_is_not_primary():
    """
    ``soft_log_v2`` is DataDecide's NORM CORRECT PROB -- the probability of the gold answer
    conditioned on the choice set -- and Sec. 3.3 finds it trends with Accuracy rather than
    beating it, unlike CORRECT PROB, which is what ``bpb_v2`` and ``ce_loss_v2`` are. It is
    recorded because a number that is recorded is one the write-up can show behaving the
    way the paper says it behaves.
    """
    assert "soft_log_v2" in score.SECONDARY_METRICS
    assert score.PRIMARY_METRIC not in score.SECONDARY_METRICS


def test_no_multiple_choice_task_is_in_the_headline_suite():
    """
    The ``*_mc_*`` spelling of a task presents the choices as lettered options and scores a
    single letter token. It is the format the metric decision above says is uninformative
    here, so having one in the suite would be the decision reversed by a task label.
    """
    for task in score.SUITE_H2B:
        if task.group == "canary":
            continue
        assert "_mc_" not in task.label, task.label


def test_the_canary_is_the_one_multiple_choice_reading_and_it_is_out_of_the_headline():
    """
    ``copycolors_10way_fast`` is a hundred trivial ten-way items and it is what turns "MC
    accuracy is uninformative at this scale" from an assertion into a measurement. Averaging
    it into the headline would flatter every arm equally and dilute the contrast, so it is
    the reason :data:`HEADLINE_GROUPS` exists.
    """
    canaries = [task for task in score.SUITE_H2B if task.group == "canary"]
    assert [task.label for task in canaries] == ["copycolors_10way_fast"]
    assert "canary" not in score.HEADLINE_GROUPS


def test_every_headline_group_is_populated_and_every_task_says_why_it_is_here():
    groups = {task.group for task in score.SUITE_H2B}
    assert set(score.HEADLINE_GROUPS) <= groups
    assert groups - set(score.HEADLINE_GROUPS) == {"canary"}
    for task in score.SUITE_H2B:
        assert task.why.strip(), task.label
    assert len({task.label for task in score.SUITE_H2B}) == len(score.SUITE_H2B)


def test_the_suite_version_moves_when_the_suite_does():
    """
    Two scoring runs a month apart have to be tellable apart by something other than their
    date, and a pre-registered suite that changed silently would be a researcher degree of
    freedom. The pin is the labels; changing them fails here until the version moves with
    them.
    """
    assert score.SUITE_VERSION == "h2b-rc-2026-08-a"
    assert [task.label for task in score.SUITE_H2B] == [
        "arc_challenge_test_rc_5shot",
        "arc_easy_test_rc_5shot",
        "hellaswag_rc_5shot",
        "piqa_val_rc_5shot",
        "csqa_val_rc_5shot",
        "socialiqa_val_rc_5shot",
        "winogrande_val_rc_5shot",
        "mmlu_stem_val_rc_5shot",
        "mmlu_humanities_val_rc_5shot",
        "mmlu_social_sciences_val_rc_5shot",
        "mmlu_other_val_rc_5shot",
        "lambada_bpb_0shot",
        "copycolors_10way_fast",
    ]


def test_the_headline_averages_groups_and_not_tasks():
    """
    MMLU is four labels of one benchmark. A flat mean over tasks would give it four votes
    against HellaSwag's one and the headline would be an MMLU number wearing a suite's name.
    """
    results = [
        score.TaskResult("a", "olmes", {"bpb_v2": 1.0}),
        score.TaskResult("m1", "mmlu", {"bpb_v2": 2.0}),
        score.TaskResult("m2", "mmlu", {"bpb_v2": 4.0}),
        score.TaskResult("c", "canary", {"bpb_v2": 100.0}),
    ]
    aggregate = score.aggregate(results)
    assert aggregate["groups"] == {"olmes": 1.0, "mmlu": 3.0, "canary": 100.0}
    # (1.0 + 3.0) / 2, with the canary left out of the headline and lambada absent.
    assert aggregate["headline"] == pytest.approx(2.0)
    assert aggregate["headline_groups"] == ["olmes", "mmlu"]


def test_a_task_that_does_not_report_a_metric_is_skipped_rather_than_counted_as_zero():
    """
    ``lambada_bpb_0shot`` reports only bits-per-byte; the RC tasks report accuracy as well.
    Averaging a missing accuracy in as zero would make the suite's accuracy depend on how
    many BPB-only tasks are in it.
    """
    results = [
        score.TaskResult("rc", "olmes", {"bpb_v2": 1.0, "len_norm_v2": 0.4}),
        score.TaskResult("bpb", "lambada", {"bpb_v2": 3.0}),
    ]
    accuracy = score.aggregate(results, "len_norm_v2")
    assert accuracy["groups"] == {"olmes": 0.4}
    assert accuracy["tasks"] == 1


# ---------------------------------------------------------------------------------------
# The offline path, which is the reason this job may run beside a sealed corpus at all.
# ---------------------------------------------------------------------------------------


def test_a_missing_tokenizer_is_refused_rather_than_fetched():
    """
    THE ONE FILE THE WHEEL DOES NOT CARRY. ``olmo_eval.HFTokenizer`` resolves an identifier
    that is neither a local file nor package data by calling ``Tokenizer.from_pretrained``,
    which is a fetch from huggingface.co. A fallback here would put a public-internet read
    in the middle of a run whose whole claim is that it made none, and its failure would
    look like a scoring failure.
    """
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.resolve_tokenizer_path("/nonexistent/tokenizer.json", environ={})
    assert refusal.value.stage is score.ScoringStage.THE_IMAGE_HAS_NO_TASK_DATA
    assert "public internet" in refusal.value.explanation


def test_the_environment_variable_beats_the_image_default(tmp_path):
    handed = tmp_path / "tokenizer.json"
    handed.write_text("{}")
    resolved = score.resolve_tokenizer_path(None, environ={score.TOKENIZER_VARIABLE: str(handed)})
    assert resolved == str(handed)


def test_a_tokenizer_that_is_not_the_pinned_one_is_refused(tmp_path):
    """
    Scoring under the wrong tokenizer does not fail -- every id is in range -- it reports a
    worse model, identically for every arm, which is the kind of error that survives a
    contrast and ruins an absolute number.
    """
    impostor = tmp_path / "tokenizer.json"
    impostor.write_text("{}")
    with pytest.raises(train_on_corpus.Refusal) as refusal:
        score.build_tokenizer(str(impostor))
    assert "sha256" in refusal.value.explanation


@needs_olmo_eval
def test_the_harness_ships_the_task_data_and_reads_it_without_a_network(tmp_path):
    """
    THE CRUX, AND IT IS ASSERTED WITH THE NETWORK TAKEN AWAY RATHER THAN ARGUED FROM THE
    SOURCE. ``ai2-olmo-eval`` 0.9.0 ships ``olmo_eval/oe_eval_tasks`` and
    ``olmo_eval/hf_datasets`` as package data -- read through ``importlib_resources`` and
    ``datasets.load_from_disk`` -- so every task in the suite builds from bytes already
    inside the image. ``train_on_corpus.py`` declines to wire the stock downstream evaluator
    into training on the grounds that it "pulls HellaSwag from Hugging Face"; at this
    version that is no longer true of the task data, and this is the test that says so.

    ``socket.socket.connect`` is replaced for the duration, so anything reaching for a
    network raises instead of succeeding slowly on a machine that happens to have one.
    """
    import olmo_eval

    tokenizer_path = os.environ.get(score.TOKENIZER_VARIABLE)
    if not tokenizer_path or not os.path.isfile(tokenizer_path):
        pytest.skip(
            f"${score.TOKENIZER_VARIABLE} does not point at a dolma2 tokenizer.json, so the "
            "offline claim cannot be tested here. The research image sets it."
        )

    connect = socket.socket.connect
    create_connection = socket.create_connection

    def refuse_to_connect(*args, **kwargs):
        raise AssertionError(f"the offline path reached for a network: {args[1:2]}")

    socket.socket.connect = refuse_to_connect  # type: ignore[method-assign]
    socket.create_connection = refuse_to_connect  # type: ignore[assignment]
    try:
        tokenizer = score.build_tokenizer(tokenizer_path)
        # One cheap task rather than all thirteen: the point is that the read path reaches
        # no network, and every task in the suite goes through the same two loaders.
        dataset = olmo_eval.build_task("copycolors_10way_fast", tokenizer, model_ctx_len=2048)
        assert len(dataset.samples) == 100
    finally:
        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.create_connection = create_connection  # type: ignore[assignment]


@needs_olmo_eval
def test_every_task_in_the_suite_is_one_the_installed_harness_has():
    """
    ``build_task`` raises on an unknown label, and it raises inside a container after the
    image has been pulled and the checkpoint fetched. This is the same check on a laptop.
    """
    from olmo_eval import list_tasks

    known = set(list_tasks())
    for suite in score.SUITES.values():
        for task in suite:
            assert task.label in known, task.label


# ---------------------------------------------------------------------------------------
# The image, which is where the offline path is actually built.
# ---------------------------------------------------------------------------------------


def dockerfile() -> str:
    """
    The research image's Dockerfile.

    :returns: Its text.
    """
    return (pathlib.Path(_HERE) / "Dockerfile").read_text()


def test_the_image_installs_the_harness_that_carries_the_task_data():
    """
    Without the ``eval`` extra the image has no ``olmo_eval``, so a scoring cell pulls a
    5 GB image, fetches a checkpoint and refuses on the import. It is one word in two places
    in the Dockerfile and both are checked, because installing the project with the extra
    while extracting the dependency layer without it would leave pip re-resolving 240 MB
    below the source copy -- which is the exact defect that file is arranged to prevent.
    """
    text = dockerfile()
    assert "optional-dependencies']['eval']" in text
    assert '".[wandb,eval]"' in text


def test_the_image_and_the_program_agree_on_where_the_tokenizer_is():
    """
    The Dockerfile sets the variable and installs the file; this program reads the variable
    and falls back to the literal path. Two literals that drifted apart would be a scoring
    cell refusing on an image that has exactly what it is asking for.
    """
    text = dockerfile()
    assert f"ENV {score.TOKENIZER_VARIABLE}={score.TOKENIZER_IN_THE_IMAGE}" in text


def test_the_image_and_the_program_agree_on_which_tokenizer_it_is():
    """
    The digest is asserted twice on purpose: at build time, where it catches a moved
    HuggingFace revision, and at run time, where it catches an image that has something else
    at that path. Two copies of a constant are worth having only while they are the same
    constant.
    """
    assert score.TOKENIZER_SHA256 in dockerfile()


def test_the_tokenizer_url_is_pinned_to_a_revision_rather_than_to_a_branch():
    """
    ``resolve/main/`` names whatever the tip is on the day of the build, so two builds of one
    commit would carry different tokenizers -- and that does not fail, it reports a worse
    model. The pin is forty hex digits.
    """
    text = dockerfile()
    assert "allenai/dolma2-tokenizer/resolve/" in text
    assert "dolma2-tokenizer/resolve/main/" not in text


# ---------------------------------------------------------------------------------------
# The staged spec, which is the other place a wrong word costs a submission.
# ---------------------------------------------------------------------------------------

SPEC = "run.score-stage.yaml"


def spec_of(name: str = SPEC) -> dict:
    """
    The committed spec.

    :param name: The file, relative to ``.edullm/``.

    :returns: The parsed YAML.
    """
    return yaml.safe_load((pathlib.Path(_HERE) / name).read_text())


def committed_argv(name: str = SPEC) -> list:
    """
    Everything the committed command passes to this program, launcher stripped.

    :param name: The file, relative to ``.edullm/``.

    :returns: The argv the entry point will see.

    :raises AssertionError: If the command runs something other than this program.
    """
    wrapper = shlex.split(spec_of(name)["command"])
    assert wrapper[:2] == ["bash", "-lc"], wrapper[:2]
    tokens = shlex.split(wrapper[2])
    for index, token in enumerate(tokens):
        if token.endswith("score_checkpoints.py"):
            return tokens[index + 1 :]
    raise AssertionError(f"the committed command runs no known entry point: {tokens}")


def test_the_committed_command_parses_through_the_real_parser():
    """
    ``main`` reads argv with ``parse_args``, so an unknown flag is a container that pulls an
    image, starts, and exits 2 on argparse's usage message. Parsing it here costs nothing.
    """
    opts = score.build_parser().parse_args(committed_argv())
    assert opts.suite == "h2b"
    assert opts.model_factory == score.DEFAULT_MODEL_FACTORY
    assert opts.step == score.FINAL_STEP
    assert opts.param_dtype == "bfloat16"


def test_the_committed_command_names_every_funded_arm():
    """
    Every cell of the fan-out is handed this one command and any cell may turn out to be any
    arm, so an arm the command does not name is five cells that pull an image and refuse.

    A SUPERSET RATHER THAN AN EQUALITY, AND THE ASYMMETRY IS THE REAL RISK. An arm in the
    command that the table does not fund costs nothing: no cell asks for it, and the URI is
    never read. An arm the table funds and the command does not is a fifth of the tranche
    missing from the downstream average -- and not missing at random, but missing exactly the
    arm somebody has just decided to buy. ``mhc`` arrived that way while this file was being
    written.
    """
    opts = score.build_parser().parse_args(committed_argv())
    named = set(score.parse_arm_runs(opts.arm_run))
    assert set(arms.FUNDED) <= named, sorted(set(arms.FUNDED) - named)


def test_every_arm_the_committed_command_names_is_a_real_arm():
    """
    ``parse_arm_runs`` refuses an unknown name, so this is the typo check: a spec saying
    ``output_only`` rather than ``output-only`` would otherwise be found by a container.
    """
    opts = score.build_parser().parse_args(committed_argv())
    assert set(score.parse_arm_runs(opts.arm_run)) <= set(arms.ARMS)


def test_the_submission_command_derives_its_cell_count_rather_than_writing_one_down():
    """
    THE NUMBER THAT HAS MOVED THREE TIMES. Nine cells, then fifteen, then twenty, each from a
    two-line edit to the arm table. A ``--fanout-size`` below the table's length is a
    submission that silently does not score the last arm -- too high refuses, too low does
    not -- so the header tells a reader to ask the program instead of copying a literal.
    """
    header = pathlib.Path(_HERE, SPEC).read_text()
    assert "--fanout-size 15" not in header
    assert "--fanout-size 20" not in header
    assert "score_checkpoints.py --fanout-size" in header


def test_the_program_can_be_asked_how_many_cells_a_fanout_has(capsys):
    with pytest.raises(SystemExit) as exit_code:
        score.build_parser().parse_args(["run", "--fanout-size"])
    assert exit_code.value.code == 0
    printed = dict(
        (line.split()[1], int(line.split()[0]))
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    )
    assert printed[score.FANOUT_INDEX_PARAMETER_CELL] == len(arms.TRANCHE_CELLS)
    assert printed[score.FANOUT_INDEX_PARAMETER_LADDER] == len(score.LADDER_CELLS)


def test_the_committed_command_carries_no_arm_and_no_seed():
    """
    They are what the fan-out index owns. Passing either would score one checkpoint fifteen
    times, which is not an error and not a visibly wrong number -- it is fifteen identical
    rows and a measured downstream noise floor of zero.
    """
    opts = score.build_parser().parse_args(committed_argv())
    assert opts.arm is None and opts.seed is None


def test_the_dtype_is_in_the_command_text_and_not_only_in_code():
    """
    The platform's precision guard reads the words of the command and cannot see a dtype set
    in code, so a command that does not name one is accepted onto a shape with no bfloat16
    in hardware and dies on the first kernel that needs it, after the machine is billed.
    """
    assert "--param-dtype" in committed_argv()


def test_the_spec_asks_for_a_shape_and_a_profile_this_repository_can_actually_submit():
    """
    ``olmo-eval-sweep`` is the contract this job wants and it belongs to ``olmo-eval-full``,
    so ``check`` refuses it from here with ``workload_profile_repository_mismatch``. The
    header says so at length; this is the assertion that the file did not quietly acquire it.
    """
    spec = spec_of()
    assert spec["schema_version"] == 1
    assert spec["workload_profile"] in {"olmo-core-train", "olmo-core-check"}
    assert spec["suggested_compute"] == "gpu-1xl4"


def test_a_single_device_shape_runs_one_process_and_does_not_go_through_torchrun():
    """
    ``require_a_process_for_every_device`` reads the device count out of the platform's
    ``CONTAINER_SHAPES`` against ``--nproc-per-node``. One L4 is one process, and a bare
    ``python`` is one process, so the launcher is absent rather than set to one.
    """
    command = shlex.split(shlex.split(spec_of()["command"])[2])
    assert "torch.distributed.run" not in command
    assert not [token for token in command if token.startswith("--nproc-per-node")]


def test_the_spec_still_has_its_placeholder_run_ids():
    """
    THE GUARD AGAINST SUBMITTING THIS BEFORE IT CAN WORK. Two of the three training stages
    have not been submitted -- ``STAGE_SPECS`` carries ``run_id=None`` for both -- and the
    one that has is recorded abbreviated. A spec carrying a plausible-looking URI that is
    not a run would fan out fifteen cells that each pull an image and refuse.

    When the ids are real this test is what has to be deleted, deliberately, in the same
    commit that fills them in.
    """
    opts = score.build_parser().parse_args(committed_argv())
    runs = score.parse_arm_runs(opts.arm_run)
    assert runs, "the spec names no arm runs at all"
    filled = {arm: uri for arm, uri in runs.items() if "RUN_ID_OF_THE" not in uri}
    assert not filled, (
        f"{sorted(filled)} carry what look like real run ids, so this spec claims to know "
        "where those training runs wrote their checkpoints. Delete this test in the commit "
        "that fills them in, and check the ids against `edullm status --json`."
    )


def test_the_checkpoint_contract_is_waived_out_loud_rather_than_faked():
    """
    ``olmo-core-train`` declares a checkpoint contract, so
    ``require_a_save_folder_a_retry_can_find`` refuses a command that does not expand
    ``$EDULLM_CHECKPOINT_DIR``. This run writes no checkpoint, so the two honest answers are
    the token -- which the refusal's own detail names for exactly this case, and which puts
    a paragraph on the approver's page -- or a ``--save-folder`` that exists only to quiet a
    check. The second is the thing AGENTS.md says not to do.
    """
    command = spec_of()["command"]
    assert "EDULLM_CHECKPOINT_CHECK=waived" in command
    assert "$EDULLM_CHECKPOINT_DIR" not in command
    assert "--save-folder" not in command


def test_the_output_goes_to_this_runs_own_prefix():
    """
    A cell writes into ``$EDULLM_OUTPUT_PREFIX``, which the fan-out prologue has already
    given a ``cell-<index>/`` of its own. Fifteen cells sharing one prefix is last-writer-wins
    with nothing reporting it, and it is a shape the platform has already been bitten by.
    """
    argv = committed_argv()
    assert "--output-dir" in argv
    assert argv[argv.index("--output-dir") + 1] == "$EDULLM_OUTPUT_PREFIX"


# ---------------------------------------------------------------------------------------
# The document fifteen of these produce, which is what the analysis reads.
# ---------------------------------------------------------------------------------------


def test_the_output_document_round_trips_and_names_its_schema(tmp_path, capsys):
    document = {
        "schema": score.OUTPUT_SCHEMA,
        "arm": "faithful",
        "seed": 2,
        "downstream": {score.PRIMARY_METRIC: {"headline": 1.25}},
    }
    score.write_document(document, str(tmp_path), arm="faithful", seed=2, step=6000)
    written = tmp_path / "downstream-faithful-seed2-step6000.json"
    assert json.loads(written.read_text()) == document
    # Always on stdout as well, which is the channel that works when the object write does
    # not -- the argument `train_on_corpus.summarise` makes. And stdout is EXACTLY the
    # document with nothing around it, so twenty of these can be read by something that does
    # not have to learn to skip a preamble.
    assert json.loads(capsys.readouterr().out) == document


def test_a_failed_write_does_not_lose_the_numbers(capsys):
    """
    The document is on stdout before anything is written anywhere, so a cell whose S3 write
    was denied still reported its result into the log stream the platform reads back.
    """
    score.write_document({"schema": score.OUTPUT_SCHEMA}, "/proc/nowhere", arm="a", seed=0, step=1)
    captured = capsys.readouterr()
    assert score.OUTPUT_SCHEMA in captured.out
    assert "could not write" in captured.err


def test_the_human_readable_lines_stay_off_stdout(capsys):
    """
    A preamble on stdout is a preamble every reader of these documents has to skip, and the
    one that forgets reads a truncated object rather than failing.
    """
    score.say("cell         arm faithful, seed 2")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "faithful" in captured.err


def test_the_scoring_stages_continue_the_training_ones_and_do_not_collide():
    """
    The exit code is the only channel out of a container that dies before W&B exists, so a
    number that means two things is the one diagnostic that works, broken.
    """
    training = {int(stage) for stage in train_on_corpus.Stage}
    scoring = {int(stage) for stage in score.ScoringStage}
    assert not training & scoring
    assert min(scoring) == max(training) + 1
    # Clear of the codes the shell and the signal convention own.
    assert max(scoring) < 126
