"""
What the submission plan must get right, since getting it wrong costs a run rather than a test.

Four things, each of which has cost somebody a submission in this project's history: the fan-out size must
equal the config directory's file count, the index must map to the cell the approval names, the dtype must
appear in the *text* of the command, and a job whose depth comes from calibration must refuse to be staged
without one.
"""

import shlex
from typing import Any, Dict

import pytest
import yaml
from factcrowd import cells as C
from factcrowd import plan

from olmo_core.exceptions import OLMoConfigurationError


def test_every_job_answers_a_question_and_names_its_blockers():
    """A job whose purpose is not stated is a job nobody can decide to skip."""
    assert plan.JOBS
    for job in plan.JOBS:
        assert job.answers.strip() and job.answers[0].isupper() or job.answers.startswith("THE")
        assert job.scoring or job.config_dir, job.name
        for blocker in job.blocked_by:
            assert blocker in plan.JOBS_BY_NAME, blocker
    # Exactly the jobs that need calibration's depth say so.
    assert {job.name for job in plan.JOBS if job.needs_lengths} == {"ladder", "entropy", "count"}
    # Runnable with nothing else finished. `score-m0` qualifies because it scores runs that already
    # exist -- it is orthogonal to the phase-2 sequence rather than part of it, and listed last.
    ready = {job.name for job in plan.JOBS if not job.blocked_by}
    assert ready == {"smoke", "calibration", "score-m0"}
    assert plan.JOBS[-1].name == "score-m0", "the optional job belongs at the end of the listing"
    assert plan.JOBS[0].name == "smoke" and plan.JOBS[1].name == "calibration"


def test_the_m0_prefixes_survive_being_overwritten_in_run_yaml():
    """
    ``.edullm/run.yaml`` holds one command, so staging a phase-2 job overwrites whatever was there. The
    three phase-1 run prefixes are not recoverable from a run id without ``edullm status``, so they are kept.
    """
    assert len(plan.M0_PREFIXES) == 3
    assert all(one.startswith("s3://") and one.endswith("/") for one in plan.M0_PREFIXES)
    command = plan.render(plan.JOBS_BY_NAME["score-m0"], endpoint="mano")
    for prefix in plan.M0_PREFIXES:
        assert prefix in command


@pytest.mark.parametrize("name", ["smoke", "calibration"])
def test_a_staged_training_job_is_valid_yaml_holding_one_shell_line(name):
    """
    The command is a folded scalar, so YAML rejoins it with single spaces. A literal block would keep the
    newlines and hand ``bash -lc`` a script whose first line is incomplete.
    """
    document = yaml.safe_load(plan.render(plan.JOBS_BY_NAME[name]))
    command = document["command"]
    assert "\n" not in command
    parts = shlex.split(command)
    assert parts[:2] == ["bash", "-lc"]
    assert len(parts) == 3, "the whole program must be one argument to -lc"
    assert document["schema_version"] == 1
    # NESTED. `RunSpec.fanout` is a `SpecFanOut` of `size` and `index_parameter`; the flat spelling belongs
    # to `SubmissionInputs` and is `extra_forbidden` here, however many places in this repository's own
    # documentation show it (PRD 8.4 and the factcrowd README both do, and both are stale).
    assert document["fanout"]["index_parameter"] == "cell"
    assert "fanout_size" not in document
    assert "fanout_index_parameter" not in document


@pytest.mark.parametrize("name", ["smoke", "calibration"])
def test_the_dtype_is_in_the_command_text_not_only_in_the_code(name):
    """
    The precision guard reads the words of the command. A command that does not name a dtype is accepted
    onto a card with no bfloat16 in hardware and dies on the first kernel that needs it -- after billing.
    """
    document = yaml.safe_load(plan.render(plan.JOBS_BY_NAME[name]))
    assert "param_dtype=bfloat16" in document["command"]


@pytest.mark.parametrize("name", ["smoke", "calibration"])
def test_the_fanout_size_is_counted_from_the_directory_not_declared(name):
    """
    A size from an older directory runs a different cell under the name the approval was granted for.

    Counted, and reported in the rendered comment and the printed invocation rather than as a field, since
    the file's schema rejects it.
    """
    job = plan.JOBS_BY_NAME[name]
    assert job.config_dir is not None
    on_disk = len(C.load_cells(plan.CONFIG_ROOT / job.config_dir))
    assert plan.fanout_size(job.config_dir) == on_disk
    assert yaml.safe_load(plan.render(job))["fanout"]["size"] == on_disk


def test_the_index_mapping_is_printed_because_filenames_sort_as_strings():
    """
    ``b16`` sorts before ``b4`` and ``113m`` before ``13m``. Any bijection is a correct submission, but a
    reader who assumes the indices ascend with demand reads the wrong cell's result.
    """
    text = plan.mapping("calibration")
    lines = [line.strip() for line in text.splitlines()[1:-1]]
    assert lines[0].startswith("0  p2_28m_ctxmanoL02"), lines[0]
    assert lines[4].startswith("4  p2_28m_manoL02"), lines[4]
    assert len(lines) == len(C.load_cells(plan.CONFIG_ROOT / "calibration"))
    assert "refused" in text.splitlines()[-1]


def test_a_job_that_needs_a_calibrated_depth_refuses_to_be_staged_without_one():
    """
    Defaulting the depth is how eighteen cells came to be trained where the endpoint had no dynamic range.
    """
    with pytest.raises(OLMoConfigurationError, match="needs both"):
        plan.main(["stage", "entropy", "--print"])
    with pytest.raises(OLMoConfigurationError, match="needs both"):
        plan.main(["stage", "ladder", "--print", "--ctxmano-length", "4"])


def test_the_scoring_job_keeps_the_step_zero_checkpoint():
    """
    ``--last-only`` drops step 0, which is G2's only evidence, so the scoring job must not pass it.
    """
    command = plan.scoring_command(["s3://bucket/runs/run_a/"], endpoint="ctxmano")
    assert "--last-only" not in command
    assert "--gate-endpoint ctxmano" in command
    assert "--write-gate-report" in command
    with pytest.raises(OLMoConfigurationError, match="at least one --prefix"):
        plan.scoring_command([])


def test_staging_an_unknown_directory_says_how_to_make_it():
    """The three post-calibration jobs have no configs until they are generated, and the error says so."""
    with pytest.raises(OLMoConfigurationError, match="no config directory"):
        plan.fanout_size("does_not_exist")


def test_the_plan_quotes_no_price():
    """
    Prices, runtime bounds and approver counts live in reviewed configuration that changes without anybody
    being told. A number here would be a stale copy of one, so there are none -- and the listing says where
    to get them.
    """
    import re

    listing = plan.describe()
    assert "edullm check --json" in listing
    source = (plan.__file__ or "").replace(".pyc", ".py")
    with open(source) as handle:
        text = handle.read()
    # No currency, and no "$N per hour" style figure.
    assert not re.search(r"\$\s*\d", text), "a price in the source is a price that goes stale"
    for word in ("approval_class", "cost"):
        assert word in text, f"{word} should be pointed at, just not quoted"


# --- validated against the platform's own schema, not against a reading of it ---------------------


def _platform_spec_module():
    """
    The platform's ``RunSpec``, or ``None`` when its checkout is not beside this one.

    **This is the test that should have existed first.** ``run.yaml``'s schema was guessed twice from
    documentation -- once from this repository's PRD 8.4 and README, which show the flat
    ``fanout_size:``/``fanout_index_parameter:`` keys, and once from
    ``schemas/submission-inputs.schema.json``, which carries those names for a different model entirely.
    Both readings were wrong and the second cost a refused submission. The authority is
    ``src/edullm_platform/cli/spec.py``, and when the platform is checked out beside this repository there
    is no reason to read documentation about it at all.

    Skipped rather than failed when absent: the checkout is a convenience of one machine, and a test that
    demands it would fail in CI for a reason unrelated to this repository.
    """
    import importlib
    import sys
    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "platform" / "src"
        if (candidate / "edullm_platform" / "cli" / "spec.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                return importlib.import_module("edullm_platform.cli.spec")
            except ImportError:  # pragma: no cover - pydantic absent
                return None
    return None


@pytest.mark.parametrize("name", ["smoke", "calibration", "score-calibration", "score-m0"])
def test_a_staged_spec_validates_against_the_platforms_own_run_spec(name):
    """Not against a reading of the schema. Against the model the CLI actually loads."""
    spec_module = _platform_spec_module()
    if spec_module is None:
        pytest.skip("the platform checkout is not beside this repository")
    job = plan.JOBS_BY_NAME[name]
    text = plan.render(job, prefixes=["s3://bucket/runs/run_a/"] if job.scoring else ())
    spec = spec_module.RunSpec.model_validate(yaml.safe_load(text))
    assert spec.schema_version == 1
    assert spec.argv[0] == "bash"
    if job.scoring:
        assert spec.fanout is None, "a scoring pass is one process, and size has a floor of 2"
    else:
        assert spec.fanout is not None
        assert spec.fanout.index_parameter == "cell"
        assert spec.fanout.size == plan.fanout_size(job.config_dir or "")


def test_the_fanout_is_nested_and_the_flat_spelling_is_refused():
    """
    The exact mistake, pinned. ``RunSpec.fanout`` is a ``SpecFanOut`` of ``size`` and ``index_parameter``;
    the flat names belong to ``SubmissionInputs``, which the CLI derives, and in this file they are
    ``extra_forbidden``.
    """
    spec_module = _platform_spec_module()
    if spec_module is None:
        pytest.skip("the platform checkout is not beside this repository")
    # Annotated, or mypy reads each `{**base, ...}` merge as updating with an incompatible value type.
    base: Dict[str, Any] = {
        "schema_version": 1,
        "workload_profile": "olmo-core-check",
        "command": "bash -lc 'true'",
    }
    nested = spec_module.RunSpec.model_validate(
        {**base, "fanout": {"size": 5, "index_parameter": "cell"}}
    )
    assert nested.fanout is not None and nested.fanout.size == 5

    from pydantic import ValidationError

    with pytest.raises(ValidationError) as caught:
        spec_module.RunSpec.model_validate(
            {**base, "fanout_size": 5, "fanout_index_parameter": "cell"}
        )
    message = str(caught.value)
    assert "fanout_size" in message and "extra_forbidden" in message
    # And a one-cell fan-out is refused, which is why the scoring jobs carry none at all.
    with pytest.raises(ValidationError):
        spec_module.RunSpec.model_validate(
            {**base, "fanout": {"size": 1, "index_parameter": "cell"}}
        )


# --- the refusals that cost a submission each ------------------------------------------------------


@pytest.mark.parametrize("name", ["smoke", "calibration", "ladder", "entropy", "count"])
def test_a_training_command_starts_one_process_per_device(name):
    """
    **The refusal that cost a submission.** ``edullm submit`` refuses ``process_per_device`` when the number
    of processes the command starts differs from the number of cards the profile bills for -- in *both*
    directions, since two ranks on a four-GPU shape idle two cards at $2.84/hour and four ranks on a
    one-GPU shape is an ``invalid device ordinal``.

    Nothing wraps the command, so the launcher has to be in it. Asserted against the device count read from
    the platform's own ``config/accelerators.yaml`` rather than a number written here.
    """
    job = plan.JOBS_BY_NAME[name]
    devices = plan.devices_for(job.compute)
    command = plan.train_command(job.config_dir or "", compute_profile=job.compute)
    if devices > 1:
        assert f"--nproc-per-node={devices}" in command
        assert "-m torch.distributed.run" in command
        assert "--standalone" in command
    else:
        # A single card needs no rendezvous, and a launcher declaring one rank would still have to agree.
        assert "--nproc-per-node" not in command


@pytest.mark.parametrize("name", ["smoke", "calibration", "ladder", "entropy", "count"])
def test_a_training_command_saves_to_the_checkpoint_directory_the_platform_reads(name):
    """
    ``checkpoint_path_not_in_command``: the platform reads the *text* of the command to check that a run
    promising a checkpoint will write one, because it cannot see inside the program.

    ``$EDULLM_OUTPUT_PREFIX`` is a different prefix and does not satisfy it. OLMo-core's own default is
    ``/tmp`` on a machine that stops existing, so a run that takes it exits zero having saved nothing.
    """
    job = plan.JOBS_BY_NAME[name]
    command = plan.train_command(job.config_dir or "", compute_profile=job.compute)
    assert '--save-folder "$EDULLM_CHECKPOINT_DIR"' in command
    assert "EDULLM_OUTPUT_PREFIX" not in command
    # And under a shell, or the variable arrives as literal characters rather than a path.
    assert command.startswith("bash -lc ")


def test_the_workload_profile_matches_what_the_job_promises():
    """
    The two presets differ in what they promise, not in what they charge. A run that trains needs the one
    with a checkpoint contract; a scoring pass needs the one without, since more than one attempt on a
    workload that checkpoints nothing earns ``retry_without_a_checkpoint_contract``.
    """
    for job in plan.JOBS:
        if job.scoring:
            assert job.workload == plan.CHECK_WORKLOAD, job.name
            assert plan.devices_for(job.compute) == 1, job.name
        elif job.name == "smoke":
            assert job.workload == plan.CHECK_WORKLOAD  # seconds of work, nothing to resume
        else:
            assert job.workload == plan.TRAIN_WORKLOAD, job.name


def test_a_scoring_command_neither_launches_nor_claims_a_checkpoint():
    """It writes a table and a gate report on one device, so both would be wrong."""
    command = plan.scoring_command(["s3://bucket/runs/run_a/"], endpoint="ctxmano")
    assert "--nproc-per-node" not in command
    assert "EDULLM_CHECKPOINT_DIR" not in command
    assert command.startswith("bash -lc ")


def test_the_device_count_comes_from_configuration_not_from_a_guess():
    """
    An unknown profile raises rather than defaulting, because a default here idles cards silently.
    """
    assert plan.devices_for("gpu-4xa10g") == 4
    assert plan.devices_for("gpu-1xa10g") == 1
    assert plan.devices_for("gpu-8xa100") == 8
    assert plan.devices_for("cpu-32vcpu") == 0
    with pytest.raises(OLMoConfigurationError, match="unknown compute profile"):
        plan.devices_for("gpu-3xa10g")


def test_the_dtype_is_named_even_though_the_program_sets_it():
    """
    ``bfloat16_not_in_the_hardware`` reads the words of the command. Both multi-card shapes that place
    instantly are T4, which has no bfloat16 at all, so a command that does not name a dtype is accepted onto
    one and dies on the first kernel that needs the format -- after the machine is billed.
    """
    for job in plan.JOBS:
        if job.scoring:
            continue
        command = plan.train_command(job.config_dir or "", compute_profile=job.compute)
        assert "param_dtype=bfloat16" in command
