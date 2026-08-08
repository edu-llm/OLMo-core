"""
Weights & Biases tracking for the latent-CoT arms.

The Phase-8 driver is a direct training loop rather than the framework
:class:`~olmo_core.train.Trainer`, because the CODI student is processed one variable-length
example at a time. That means it never builds the callback list, so
:class:`~olmo_core.train.callbacks.WandBCallback` — the way everything else in this repository
reaches W&B — is not available to it. This module is the equivalent for that loop.

**It follows the platform's convention rather than inventing one.** ``.edullm/train_on_corpus.py``
on ``main`` establishes it, and the pieces matter:

- **``EDULLM_WANDB_PROJECT``** is set by the platform when a submission passes
  ``--wandb-project``. Tracking is enabled *if and only if* that variable (or an explicit
  ``--wandb-project``) names a project, which is what keeps running the image by hand from
  failing on a missing ``WANDB_API_KEY``.
- **``WANDB_RUN_GROUP``** is also set by the platform, and the ``wandb`` client reads it by
  itself. So ``group`` is deliberately **not** passed here: forwarding an environment variable
  that may not exist would set it to ``None`` and look like a decision.
- **``WANDB_INIT_TIMEOUT``** gets a 60 s default, since ``init`` reaches the network and the
  default is short enough to bite on a busy container.

**Every failure is swallowed, and that is the whole design.** A five-arm run costs a day of
A100 time; losing it because a metrics sidecar could not resolve DNS would be absurd. So a
missing package, an unset key, a failed ``init`` and a failed ``log`` all degrade to printing
one line on stderr and continuing untracked. :attr:`ArmTracker.active` says which happened, and
the caller prints it, so "not tracked" is visible in the log rather than silent.

The other half of the design is that ``train_arm`` knows nothing about any of this. It takes an
``on_log`` callable, so the training core stays dependency-free and unit-testable, and this is
the only file in the module that imports ``wandb``.
"""

import os
import sys
from typing import Any, Dict, Optional, Sequence

__all__ = ["resolve_project", "ArmTracker"]


def resolve_project(explicit: Optional[str] = None) -> Optional[str]:
    """
    The W&B project to log to, or ``None`` to run untracked.

    :param explicit: A project named on the command line, which wins. Use it for local runs;
        on the platform, leave it unset and pass ``--wandb-project`` to ``edullm submit``, which
        is what sets ``EDULLM_WANDB_PROJECT`` in the container.
    :returns: The project name, or ``None`` if neither source named one.
    """
    return explicit or os.environ.get("EDULLM_WANDB_PROJECT") or None


class ArmTracker:
    """
    A W&B run for one experiment arm, or an inert stand-in when tracking is off.

    Construct with :meth:`start`. Every method is a no-op when inactive, so callers need no
    ``if tracker is not None`` branches.
    """

    def __init__(self, run: Any = None, reason: str = "") -> None:
        self._run = run
        self.reason = reason
        """Why tracking is off, for the caller to print. Empty when it is on."""

    @property
    def active(self) -> bool:
        """Whether metrics are actually going anywhere."""
        return self._run is not None

    @property
    def url(self) -> str:
        """The run's W&B page, or ``""`` when inactive."""
        try:
            return str(getattr(self._run, "url", "") or "")
        except BaseException:  # noqa: BLE001 -- a URL is never worth an exception
            return ""

    @classmethod
    def start(
        cls,
        *,
        project: Optional[str],
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> "ArmTracker":
        """
        Begin tracking, or return an inert tracker explaining why not.

        :param project: From :func:`resolve_project`. ``None`` means run untracked.
        :param name: The W&B run name — use ``"<arm>-seed<n>"`` so the five arms of one
            experiment are told apart inside the group the platform sets.
        :param config: Hyperparameters to record. The arm-defining fields belong here; that is
            what makes the confound control legible in the UI.
        :param dir: Where the ``wandb/`` state directory goes. Point it at the arm's own staging
            directory: five arms run as five concurrent processes in one container, and giving
            each its own keeps them from sharing scratch.
        :param tags: Optional tags.

        :returns: An :class:`ArmTracker`, active or not. Never raises.
        """
        if not project:
            return cls(
                reason="no project (pass --wandb-project, or --wandb-project to edullm submit)"
            )
        try:
            import wandb

            # Through the environment rather than Settings, whose accepted fields move between
            # wandb versions. setdefault, so an explicitly-set value still wins.
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
            if dir is not None:
                os.makedirs(dir, exist_ok=True)
            # No `group=`: the platform exports WANDB_RUN_GROUP and the client reads it itself.
            run = wandb.init(
                project=project,
                name=name,
                config=config or {},
                dir=dir,
                tags=list(tags) if tags else None,
            )
            return cls(run=run)
        except BaseException as exc:  # noqa: BLE001 -- see the module docstring
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[wandb] init failed, continuing untracked -- {reason}", file=sys.stderr)
            return cls(reason=reason)

    def log(self, entry: Dict[str, Any]) -> None:
        """
        Log one training-history entry.

        Non-numeric values are dropped, because ``entry`` is the same dict the loop prints and a
        stray string would be rejected by W&B for the whole call. ``step`` is passed as the
        W&B step rather than logged as a metric.

        :param entry: A ``train_history`` entry (must contain numeric values; ``step`` optional).
        """
        if self._run is None:
            return
        try:
            step = entry.get("step")
            values = {
                key: value
                for key, value in entry.items()
                if key != "step" and isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            self._run.log(values, step=int(step) if step is not None else None)
        except BaseException as exc:  # noqa: BLE001 -- a metric is never worth the run
            print(f"[wandb] log failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def summarize(self, values: Dict[str, Any]) -> None:
        """
        Write end-of-run values into the run summary (the columns a runs table compares on).

        :param values: Scalars and small dicts, e.g. ``overall_acc`` and ``solve_rate_by_depth``.
        """
        if self._run is None:
            return
        try:
            for key, value in values.items():
                self._run.summary[key] = value
        except BaseException as exc:  # noqa: BLE001
            print(f"[wandb] summary failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def finish(self, exit_code: int = 0) -> None:
        """
        Close the run, flushing anything buffered.

        :param exit_code: Non-zero marks the run failed in the UI.
        """
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except BaseException as exc:  # noqa: BLE001
            print(f"[wandb] finish failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            self._run = None
