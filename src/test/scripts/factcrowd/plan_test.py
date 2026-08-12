"""
What the submission plan must get right, since getting it wrong costs a run rather than a test.

Four things, each of which has cost somebody a submission in this project's history: the fan-out size must
equal the config directory's file count, the index must map to the cell the approval names, the dtype must
appear in the *text* of the command, and a job whose depth comes from calibration must refuse to be staged
without one.
"""

import shlex

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
    # And the first two are runnable with nothing else finished.
    ready = {job.name for job in plan.JOBS if not job.blocked_by}
    assert ready == {"smoke", "calibration"}


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
    assert document["fanout_index_parameter"] == "cell"


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
    """A size from an older directory runs a different cell under the name the approval was granted for."""
    job = plan.JOBS_BY_NAME[name]
    assert job.config_dir is not None
    document = yaml.safe_load(plan.render(job))
    on_disk = len(C.load_cells(plan.CONFIG_ROOT / job.config_dir))
    assert document["fanout_size"] == on_disk


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
