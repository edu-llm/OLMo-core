from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks import CheckpointerCallback


def pending_callback(**kwargs) -> tuple[CheckpointerCallback, Future]:
    """A callback with an async save in flight, which is the state ``pre_train`` leaves."""
    callback = CheckpointerCallback(**kwargs)
    callback._trainer = Mock()
    future: Future = Future()
    callback._future = future
    callback._latest_checkpoint_path = "s3://bucket/run/checkpoints/step0"
    return callback, future


def test_close_waits_for_an_in_flight_checkpoint():
    """The one that matters. Without this the writer thread outlives the process group.

    ``post_train`` is the only other place that awaits the future, and the error path in
    ``Trainer.fit`` never reaches it: it calls ``_shutdown(gracefully=False)`` and re-raises.
    ``close`` is called on both paths, so it is where the wait has to be.
    """
    callback, future = pending_callback()
    future.set_result(None)

    callback.close()

    assert callback._future is None, "the save was not awaited, so nothing waits for it at all"


def test_close_gives_up_rather_than_hanging_when_the_save_cannot_finish():
    """A bound, because an ungraceful shutdown is where ranks disagree.

    ``_shutdown(gracefully=False)`` exists so that ranks already in a bad state do not
    deadlock on further collectives. If one rank raised and the others did not, the save can
    never complete, and an open-ended wait here would turn a crash into a twelve-hour hang.
    """
    callback, _ = pending_callback(shutdown_timeout=0.01)

    callback.close()

    assert callback._future is None


def test_close_does_not_replace_the_exception_that_is_on_its_way_up():
    """A save that failed is not the report anybody needs.

    ``close`` runs while the original exception is propagating. Raising out of it would
    replace the traceback that says what actually went wrong with one about checkpointing,
    which is the confusion this whole change exists to remove.
    """
    callback, future = pending_callback()
    future.set_exception(RuntimeError("the write failed"))

    callback.close()

    assert callback._future is None


def test_close_does_nothing_when_no_save_is_in_flight():
    """The success path already awaited in ``post_train``, so this must be a cheap no-op."""
    callback = CheckpointerCallback()
    callback._trainer = Mock()

    callback.close()

    assert callback._future is None


def test_a_disabled_checkpointer_does_not_wait():
    callback, future = pending_callback(enabled=False)

    callback.close()

    assert not future.done()


def test_a_shutdown_timeout_that_cannot_expire_is_refused():
    with pytest.raises(OLMoConfigurationError, match="shutdown_timeout"):
        CheckpointerCallback(shutdown_timeout=0)
