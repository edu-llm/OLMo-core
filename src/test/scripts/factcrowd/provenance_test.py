"""
Recording which code produced an artefact, and why "unknown" must not look like "clean".

Every other digest in this project answers "is this the same corpus?". This one answers "is this the same
code?", which is the question a reader of a finished table has and could not previously ask: two
checkpoints with identical schema, vocabulary and renderer fingerprints can come from revisions that score
the endpoint differently.

The tests are mostly about *absence*. Bookkeeping that raises on a machine without git would take down a
training run for a reason unrelated to training, and a dirty tree recorded as clean would make a commit
hash a false claim about what ran.
"""

import subprocess

import pytest
from factcrowd import provenance


def test_the_record_names_the_commit_of_this_checkout():
    """
    The useful 90%: a hash a reader can check out.

    Compared against ``git`` rather than a fixture, because a hard-coded hash would pass on the commit it
    was written at and never again.
    """
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    if expected.returncode != 0:  # pragma: no cover - only when tests run outside a checkout
        pytest.skip("not a git checkout")
    assert provenance.commit() == expected.stdout.strip()
    assert provenance.record()["commit"] == expected.stdout.strip()


def test_a_dirty_tree_is_recorded_separately_from_the_commit():
    """
    A dirty tree means the commit does **not** identify the code that ran.

    Reachable rather than hypothetical: the launch path supports ``allow_dirty=True``, so a run can
    legitimately carry a hash whose tree it does not match. Recording only the hash would turn that into a
    false claim, and the two together are still honest.

    Skipped rather than failed when git cannot answer. This checkout is a worktree whose ``gitdir``
    pointer has been rewritten to a foreign path four times now; each time, every git command in it
    returns "not a git repository". A test asserting that provenance is *populated* then fails for a
    reason that has nothing to do with provenance -- and the module's actual contract, that an
    unanswerable git records as absent rather than as clean, is what the next two tests pin.
    """
    dirty = provenance.is_dirty()
    if dirty is None:  # pragma: no cover - only when git cannot read this tree
        pytest.skip("git cannot read this checkout, so there is no dirty state to record")
    assert dirty in (True, False)
    assert provenance.record()["dirty"] is dirty


def test_git_failure_is_recorded_as_absence_not_as_an_exception(monkeypatch):
    """
    A machine without git must not fail a training run over bookkeeping.

    And the key is *omitted* rather than set to None, so a reader can tell "not recorded" from "recorded
    as absent" -- which also means adding a key later does not retroactively give old records empty
    columns.
    """

    def explode(*args, **kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", explode)
    assert provenance.commit() is None
    assert provenance.is_dirty() is None
    record = provenance.record()
    assert "commit" not in record and "dirty" not in record


def test_a_nonzero_git_exit_is_absence_and_not_a_clean_tree(monkeypatch):
    """
    Not every git failure raises: outside a work tree, git exits non-zero.

    The return code is what separates "could not tell" from "told me nothing", and the two differ most for
    ``is_dirty``: ignoring the code makes a failed ``git status`` return empty output, which reads as
    ``False`` -- a clean tree. That would pair a commit hash with a false claim that it describes the code
    that ran, which is the exact thing recording the dirty flag exists to prevent.
    """

    class Failed:
        returncode = 128
        stdout = "some output git printed anyway"
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
    assert provenance.commit() is None
    assert provenance.is_dirty() is None  # not False
    record = provenance.record()
    assert "commit" not in record and "dirty" not in record


def test_git_succeeding_with_no_output_is_not_a_commit_but_is_a_clean_tree(monkeypatch):
    """
    Empty output on success means opposite things for the two commands.

    ``git status --porcelain`` prints nothing for a clean tree, so empty is the *answer* -- collapsing it
    to "unknown" would leave every clean run without a dirty flag. ``rev-parse`` printing nothing is not a
    hash, and recording ``""`` would give a reader a column that looks populated and identifies nothing.
    """

    class Empty:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Empty())
    assert provenance.commit() is None
    assert provenance.is_dirty() is False
    record = provenance.record()
    assert "commit" not in record and record["dirty"] is False


def test_platform_variables_are_recorded_when_set_and_skipped_when_blank(monkeypatch):
    """
    The fan-out index is the only thing separating two cells of one submission in the platform's logs.

    Blank is treated as unset: the platform exports these variables unconditionally, so an unexpanded or
    empty value is common and recording ``""`` would look like a value.
    """
    monkeypatch.setenv("EDULLM_RUN_ID", "factcrowd-m0-a")
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "3")
    monkeypatch.setenv("EDULLM_IMAGE_DIGEST", "")
    record = provenance.record()
    assert record["run_id"] == "factcrowd-m0-a"
    assert record["fanout_index"] == "3"
    assert "image_digest" not in record


def test_the_record_is_json_serialisable():
    """It is written into a checkpoint's config record, which is JSON."""
    import json

    assert json.loads(json.dumps(provenance.record())) == provenance.record()
