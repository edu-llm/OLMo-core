"""Run the group-word state-tracking probe on the eduLLM platform.

Usage (as the platform ``command``)::

    python .edullm/probe_group_words.py "$EDULLM_RUN_ID" \
        --task a5_words --arm Reflection --bundle-id 1101 \
        --output-dir "$EDULLM_OUTPUT_PREFIX"

This is a *probe*, not a language-model training run. It trains a ~1M-parameter model on a
synthetic group-word task for a few hundred steps and reports held-out accuracy by sequence
length. That is why it declares no checkpoint contract and why it does not touch
``EDULLM_DATASET_ID``: the task is generated from a seed, so there is no corpus to open.

Why this file exists rather than a copy of ``train_on_corpus.py``
----------------------------------------------------------------
``train_on_corpus.py`` is the base for *corpus* training: it resolves a dataset release through
``edullm_data.read``, checks the manifest's dtype/byte-order/header, and hands OLMo-core a
``NumpyDatasetConfig``. None of that applies to a synthetic task. What this probe needs instead
already exists in ``probes/train_probe.py`` -- the four-way seed plumbing, the frozen and
checksummed evaluation bank, the arm registry, the beta-regime contract and the manifest guards.
Re-implementing those here would fork them, and the fork would drift.

So this module is a thin adapter: it translates the platform's environment contract into
``train_probe``'s argv contract and delegates. The scientific code has one home.

The platform contract this file is responsible for
--------------------------------------------------
* **Run name.** First positional argument, used as the run id in the emitted record.
* **Outputs.** Written under ``$EDULLM_OUTPUT_PREFIX``, passed on the command line. Nothing is
  written to local disk expecting to survive -- the machine goes away.
* **No shell assumptions.** The container ``exec``s one command. This module appends nothing and
  spawns nothing.
* **No dataset variables.** Deliberately unread; a synthetic task has no corpus. Submit with
  ``dataset_release=none``.

:raises SystemExit: on a missing run name, an unknown task, or a delegation failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# probes/ is vendored at the repository root (see the repo's own probes/ directory). The
# platform builds an image from the whole checkout, so the harness ships with this file rather
# than needing $KDA_PROBES_DIR the way the test suite does.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBES = _REPO_ROOT / "probes"


def _require_probes() -> None:
    """Put the vendored ``probes/`` on ``sys.path``, failing loudly if it is absent.

    Deliberately not a warning. A probe run whose harness is missing has nothing to measure,
    and the platform records a zero exit as an unqualified success -- so a soft failure here
    would produce a green run with an empty output prefix.

    :raises SystemExit: if ``probes/`` is not in the image.
    """
    if not (_PROBES / "train_probe.py").is_file():
        raise SystemExit(
            f"probes/train_probe.py is not in this image (looked in {_PROBES}). The probe "
            f"harness is vendored at the repository root; if it is missing, the image was "
            f"built from a commit that predates it."
        )
    if str(_PROBES) not in sys.path:
        sys.path.insert(0, str(_PROBES))


def build_parser() -> argparse.ArgumentParser:
    """Build the adapter's own argument parser.

    Only the arguments the platform command needs to set are declared here. Everything else is
    passed through to ``train_probe`` verbatim, so its defaults stay the single source of truth
    and this file does not silently shadow them.

    :returns: the parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", help='Run id; pass "$EDULLM_RUN_ID".')
    parser.add_argument(
        "--output-dir",
        default=None,
        help='Where to write the result record. Pass "$EDULLM_OUTPUT_PREFIX".',
    )
    parser.add_argument("--task", required=True, help="Task id from probes/tasks.py TASKS.")
    parser.add_argument(
        "--arm",
        required=True,
        help="Canonical arm id (R1, R1-P, DP2-strict, Reflection). Sets mixer, R and regime.",
    )
    parser.add_argument("--bundle-id", type=int, required=True, help="Seed bundle.")
    # Forwarded as train_probe's '--param-ledger-only'. The names differ because this flag is
    # the platform-facing verb and that one is the harness's; keeping '--dry-run' here means a
    # submission can be rehearsed with the same word the rest of the platform uses.
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print; train nothing.")
    return parser


def main() -> None:
    """Translate the platform contract into ``train_probe``'s argv and delegate."""
    logging.basicConfig(level=logging.INFO)
    opts, passthrough = build_parser().parse_known_args()

    _require_probes()

    import tasks as probe_tasks  # type: ignore[import-not-found]
    import train_probe  # type: ignore[import-not-found]

    if opts.task not in probe_tasks.TASKS:
        raise SystemExit(f"unknown task {opts.task!r}. Available: {sorted(probe_tasks.TASKS)}")

    out = opts.output_dir or os.environ.get("EDULLM_OUTPUT_PREFIX")
    if not out:
        raise SystemExit(
            'no output location: pass --output-dir "$EDULLM_OUTPUT_PREFIX". Writing to '
            "container-local disk would lose the result when the machine goes away."
        )

    # $EDULLM_OUTPUT_PREFIX is an s3:// prefix, and train_probe writes its record with a plain
    # `open()` (train_probe.py:633,724) which cannot address s3://. Handing it the prefix
    # directly would either raise, or -- worse -- create a local directory literally named
    # "s3:" and still exit 0, which the platform records as success with an empty prefix.
    #
    # So: write to a local staging file, then upload. Uploading is deliberately explicit rather
    # than hidden inside train_probe, because the harness is also run on non-AWS hosts where
    # boto3 and credentials are absent and a local path is the correct behaviour.
    name = f"probe-{opts.arm}-{opts.task}-b{opts.bundle_id}.json"
    staged = Path("/tmp") / name
    record = str(staged)
    destination = f"{out.rstrip('/')}/{name}" if out.startswith("s3://") else None
    if destination is None:
        # A local output directory was given (or the var held a local path): write there
        # directly and skip the upload.
        target_dir = Path(out)
        target_dir.mkdir(parents=True, exist_ok=True)
        record = str(target_dir / name)

    # '--run-id' carries the platform's run id into the record. Without it train_probe writes
    # 'run_id': null on the free-form path, and a null id makes a result impossible to tie back
    # to the submission, the S3 prefix or the W&B run that produced it -- all three of which are
    # named by exactly this string.
    argv = [
        "--run-id",
        opts.run_name,
        "--arm",
        opts.arm,
        "--task",
        opts.task,
        "--bundle-id",
        str(opts.bundle_id),
        "--out",
        record,
        *passthrough,
    ]
    if opts.dry_run:
        # NOT '--dry-run'. train_probe spells this '--param-ledger-only' (train_probe.py:673),
        # and forwarding the platform's spelling made argparse exit 2 with 'unrecognized
        # arguments' -- after the image pull, on a paid instance, from a flag whose whole
        # purpose is to cost nothing.
        argv.append("--param-ledger-only")

    log.info("run_name=%s", opts.run_name)
    log.info("delegating to train_probe with argv: %s", " ".join(argv))
    log.info("result record -> %s", record)
    if destination:
        log.info("will upload to %s", destination)

    # train_probe.main() reads sys.argv, so hand it the translated argv rather than importing
    # its internals piecemeal. Its own manifest/seed/regime guards then apply unchanged.
    sys.argv = ["train_probe.py", *argv]
    train_probe.main()

    if destination:
        _upload(Path(record), destination)


def _upload(local: Path, destination: str) -> None:
    """Copy the finished record to ``destination`` (an ``s3://`` URI).

    Failing to upload is fatal. The probe has by this point already done its work, so a swallowed
    upload error would leave a run that trained correctly, exited zero, and produced nothing an
    analyst can read -- the exact shape of failure the platform's checkpoint guard exists to stop
    for training runs, and which nothing guards for a probe's outputs.

    :param local: The staged local file.
    :param destination: An ``s3://bucket/key`` URI.

    :raises SystemExit: if the file is missing or the upload fails.
    """
    if not local.is_file():
        raise SystemExit(
            f"train_probe reported success but wrote no record at {local}. Nothing to upload."
        )
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the image
        raise SystemExit(
            f"cannot upload the result to {destination}: boto3 is not installed in this image "
            f"({exc}). The record is at {local} but will be lost when this machine goes away."
        ) from exc

    bucket, _, key = destination.removeprefix("s3://").partition("/")
    try:
        boto3.client("s3").upload_file(str(local), bucket, key)
    except Exception as exc:
        raise SystemExit(f"upload to {destination} failed: {type(exc).__name__}: {exc}") from exc
    log.info("uploaded %s -> %s (%d bytes)", local, destination, local.stat().st_size)


if __name__ == "__main__":
    main()
