"""
Shared fixtures for the factcrowd tests.

The only thing here is a disk guard, and it earned its place. The ``slow`` tests each train a cell and
write ten sharded checkpoints -- roughly a gigabyte per test -- so a run that fills the disk reports
``CheckpointException`` from deep inside ``torch.distributed.checkpoint``, ten times over, with the real
cause (``OSError: [Errno 28]``) buried in a subprocess's captured stdout. That reads exactly like ten
broken tests. It cost a full verification cycle to tell apart once; this makes the difference obvious.
"""

import shutil

import pytest

SLOW_TEST_HEADROOM_BYTES = 8 * 1024**3
"""
Free space required before a ``slow`` test will run.

Twelve slow tests at about a gigabyte each, plus the checkpoint each one writes before it deletes the
previous -- 8 GiB is comfortable rather than tight, and being wrong in the tight direction reintroduces
the failure this guard exists to explain.
"""


@pytest.fixture(autouse=True)
def _refuse_to_run_slow_tests_without_disk(request: pytest.FixtureRequest) -> None:
    """
    Skip a ``slow`` test when the disk is too full to hold its checkpoints.

    Autouse and marker-gated, so the fast tests -- which write almost nothing -- are unaffected and pay no
    ``statvfs`` call each.

    :param request: The test's request object, used to read its markers.
    """
    if request.node.get_closest_marker("slow") is None:
        return
    free = shutil.disk_usage("/tmp").free
    if free < SLOW_TEST_HEADROOM_BYTES:
        pytest.skip(
            f"only {free / 1024 ** 3:.1f} GiB free on /tmp, under the "
            f"{SLOW_TEST_HEADROOM_BYTES / 1024 ** 3:.0f} GiB a slow test needs for its checkpoints. "
            f"This is a full disk, not a broken test: killed runs leave their scratch behind, so clear "
            f"/tmp/pytest-of-* and any orphaned /tmp/tmp* directories."
        )
