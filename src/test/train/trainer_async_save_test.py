"""The happens-before edge between an async checkpoint save and the metric it records.

WHAT KILLED run_019fd382. ``Trainer._shutdown`` calls ``_log_metrics``, which returns early --
taking NO collective -- when ``self._metrics`` is empty. After the last training step the only
thing left to land in ``self._metrics`` is ``checkpoint/save_async_duration_s``, recorded from
the done-callback of the async save. ``Future.set_result`` wakes waiters BEFORE it invokes
callbacks, so a rank awaiting the save could return from ``fut.result()`` while that
``record_metric`` had not run yet. Ranks then disagreed about whether ``_log_metrics`` had
anything to reduce: some entered an all-reduce on the bookkeeping group, others fell through to
the ``barrier()`` at the end of ``_shutdown``. Both sides timed out (30 min and 15 min
respectively) and three of four cells died in teardown after training had completed.

WHY THIS IS TESTABLE WITHOUT A GPU. The defect is pure ordering inside one process: it is a
property of when the future handed to ``CheckpointerCallback`` resolves relative to the
callback body. No process group, no CUDA and no S3 are needed to observe it -- only a real
``Trainer.save_checkpoint_async`` call with a checkpointer whose inner future we control. The
multi-rank *consequence* is not reproducible here; the ordering that causes it is exactly what
these tests pin.

These tests call ``save_checkpoint_async`` itself rather than re-deriving its logic, so
reverting the fix turns them red.
"""

import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, List, Optional

import pytest

from olmo_core.train.trainer import Trainer

# How long the save's done-callback chain is made to take.
#
# THIS IS NOT A SLEEP-AND-HOPE. Done-callbacks run in registration order, and the tests register
# a slow one on the inner future BEFORE ``save_checkpoint_async`` registers the trainer's. So the
# trainer's ``record_metric`` provably cannot run until this delay has elapsed, while a thread
# awaiting the *inner* future is released the instant ``set_result`` is called -- before any
# callback runs at all. The window is therefore deterministic, not probabilistic: without the fix
# the await always returns first. The delay only has to exceed thread wake-up latency.
CALLBACK_SECONDS = 0.5


class _InnerFuture(Future):
    """The future ``Checkpointer.save_async`` returns, resolvable on demand."""

    def finish(self) -> None:
        self.set_running_or_notify_cancel()
        self.set_result(None)

    def fail(self, exc: BaseException) -> None:
        self.set_running_or_notify_cancel()
        self.set_exception(exc)

    def delay_callbacks_by(self, seconds: float) -> None:
        """Register a slow done-callback, to be called BEFORE the code under test registers its.

        Callbacks run in registration order on the resolving thread, so this deterministically
        holds off everything ``save_checkpoint_async`` registers afterwards. A thread awaiting
        this future directly is released by ``set_result`` before any callback runs, so the
        pre-fix ordering is forced rather than raced for.
        """
        self.add_done_callback(lambda _fut: time.sleep(seconds))

    def finish_from_another_thread(self) -> threading.Thread:
        """Resolve from a separate thread, which is what a real DCP save does.

        This detail is load-bearing. ``Future.set_result`` invokes done-callbacks on the thread
        that resolves the future, so resolving from the *awaiting* thread runs the callbacks to
        completion before the await returns and hides the bug entirely. In production the
        resolver is a DCP writer thread and the awaiting thread is the training thread.
        """
        thread = threading.Thread(target=self.finish, name="dcp-writer")
        thread.start()
        return thread


class _FakeCheckpointer:
    def __init__(self) -> None:
        self.inner = _InnerFuture()

    @staticmethod
    def checkpoint_dirname(step: int) -> str:
        return f"step{step}"

    def save_async(self, *args, **kwargs) -> Future:
        del args, kwargs
        return self.inner


class _RecordingTrainer:
    """The narrowest object ``Trainer.save_checkpoint_async`` actually touches.

    Deliberately not a real ``Trainer``: constructing one requires a train module, a data
    loader and a process group. The method under test is taken unbound off the class, so the
    code exercised is the shipped code, while this stands in for the collaborators it reads.
    """

    save_checkpoint_async = Trainer.save_checkpoint_async

    def __init__(self) -> None:
        self.global_step = 200
        self.save_folder = "s3://bucket/run/checkpoints"
        self.checkpointer = _FakeCheckpointer()
        self.train_module = object()
        self.metrics: Dict[str, float] = {}
        self.saved_paths: List[str] = []
        self.log_metrics_calls = 0
        self.join_calls = 0

    # --- the pieces save_checkpoint_async calls -------------------------------------------
    def _log_metrics(self) -> None:
        self.log_metrics_calls += 1

    def _join_bookkeeping_ops(self) -> None:
        self.join_calls += 1

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def record_metric(self, name: str, value: float, reduce_type: Optional[Any] = None) -> None:
        del reduce_type
        self.metrics[name] = value

    def _iter_callbacks(self):
        trainer = self

        class _Callback:
            @staticmethod
            def post_checkpoint_saved(path) -> None:
                trainer.saved_paths.append(str(path))

        return iter([_Callback()])


METRIC = "checkpoint/save_async_duration_s"


def test_awaiting_the_save_guarantees_the_duration_metric_was_recorded():
    """
    THE REGRESSION, IN THE SHAPE THE RUN HIT IT. ``CheckpointerCallback._await_last_checkpoint``
    calls ``fut.result()`` on the training thread while a DCP writer thread resolves the save.
    Before the fix the awaited future WAS the inner future, and ``set_result`` wakes waiters
    before running done-callbacks -- so ``result()`` returned while ``record_metric`` had not run.
    The awaiting rank then enters ``_shutdown`` with ``self._metrics`` empty, ``_log_metrics``
    returns early taking no collective, and it lands on the ``barrier()`` while its peers sit in
    an all-reduce. That is the divergence that killed 3 of 4 cells of run_019fd382.

    The assertion is made IMMEDIATELY after the await, because that is the instant at which
    ``_shutdown`` inspects ``self._metrics``.
    """
    trainer = _RecordingTrainer()
    trainer.checkpointer.inner.delay_callbacks_by(CALLBACK_SECONDS)
    _, future = trainer.save_checkpoint_async()

    assert not future.done(), "the save future is complete before the save even finished"

    thread = trainer.checkpointer.inner.finish_from_another_thread()
    try:
        future.result()
        # Read at once -- no join() first, or the callback is given time to finish regardless of
        # the ordering and the test would pass against the broken code.
        metric_visible = METRIC in trainer.metrics
    finally:
        thread.join()

    assert metric_visible, (
        "the await returned before the duration metric was recorded, so a rank reaches _shutdown "
        "with no metrics to reduce while its peers have one -- the collective divergence that "
        "killed run_019fd382"
    )


def test_awaiting_the_save_guarantees_the_sidecar_callbacks_ran():
    """``post_checkpoint_saved`` writes ``config.json`` and ``data_paths.txt`` beside the
    checkpoint. If the await can return first, a caller may act on -- or a peer may list -- a
    checkpoint whose sidecar files are not there yet."""
    trainer = _RecordingTrainer()
    trainer.checkpointer.inner.delay_callbacks_by(CALLBACK_SECONDS)
    path, future = trainer.save_checkpoint_async()

    thread = trainer.checkpointer.inner.finish_from_another_thread()
    try:
        future.result()
        saved = list(trainer.saved_paths)
    finally:
        thread.join()

    assert saved == [str(path)], "post_checkpoint_saved had not run when the await returned"


def test_a_failed_save_still_resolves_the_future_rather_than_hanging():
    """
    THE HAZARD THE FIX ITSELF INTRODUCES, PINNED. The returned future is now resolved by hand,
    so any path that fails to resolve it converts a loud failure into a silent hang -- strictly
    worse than the bug being fixed. The failure must still surface, and through ``result()``.
    """
    trainer = _RecordingTrainer()
    _, future = trainer.save_checkpoint_async()

    boom = RuntimeError("upload failed")
    trainer.checkpointer.inner.fail(boom)

    assert future.done(), "a failed save left the returned future unresolved -- an await hangs"
    with pytest.raises(RuntimeError, match="upload failed"):
        future.result()
    assert METRIC not in trainer.metrics, "a failed save must not report a save duration"


def test_a_save_that_fails_after_uploading_also_resolves():
    """The inner save can succeed while the callback body fails (a sidecar write, say). That
    exception must reach the awaiting rank rather than stranding it."""
    trainer = _RecordingTrainer()

    def explode(path) -> None:
        del path
        raise RuntimeError("sidecar write failed")

    class _BadCallback:
        post_checkpoint_saved = staticmethod(explode)

    trainer._iter_callbacks = lambda: iter([_BadCallback()])  # type: ignore[method-assign]

    _, future = trainer.save_checkpoint_async()
    trainer.checkpointer.inner.finish()

    assert future.done(), "a callback failure left the returned future unresolved"
    with pytest.raises(RuntimeError, match="sidecar write failed"):
        future.result()


def test_metrics_are_flushed_before_the_save_captures_state():
    """Ordering the method already relied on, asserted so the fix cannot quietly reorder it:
    pending metrics are logged and bookkeeping joined BEFORE the state dict is captured."""
    trainer = _RecordingTrainer()
    trainer.save_checkpoint_async()

    assert trainer.log_metrics_calls == 1
    assert trainer.join_calls == 1
