"""
What ``edullm check`` cannot ask: does the command in a run spec, once the platform has
wrapped it and a shell has run it, hand the trainer the arguments it was meant to?

``edullm check`` reads the command's *words*. It answers whether a launcher is in command
position, whether the process count matches the shape, whether a dtype the hardware lacks is
named, and whether the checkpoint variable is written somewhere a shell would expand. It
cannot answer the question that actually killed a run, which is whether
``train_on_corpus.py`` receives ``--data-seed 3`` when the third cell of a fan-out starts.

This script answers that one, on a CPU, without a network:

1. Reads a spec and splits its command exactly as the submission workflow does (``shlex``).
2. Wraps it in :func:`edullm_platform.execution.fanout_cell_command`'s prologue when the spec
   declares a fan-out, which is the text a cell's container is actually handed. When the
   ``edullm`` CLI is not installed the prologue is reproduced from the platform's own source
   and the run says which of the two it used.
3. Runs that text in a real ``bash`` with the environment Batch sets, against a stub
   ``python`` that records its argv rather than training anything.
4. Feeds the recorded argv to ``train_on_corpus.build_parser().parse_known_args`` and merges
   the leftover dotted overrides into a real :class:`TransformerConfig`, then asserts that
   every seed the spec meant to move has moved.

What it cannot do is run the model, reach S3, or say anything about throughput. Those need a
GPU and a corpus; see ``docs/hc-ablation/AGENT-STATUS.md``.

    python src/scripts/ablations/hc_launch_check.py
    python src/scripts/ablations/hc_launch_check.py --spec .edullm/run.hc-baseline.yaml
"""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The specs this checks when it is given none. Every spec this repository asks a human to
#: submit belongs here, so that adding one and forgetting to check it is a failing run of this
#: script rather than a discovery on the platform.
DEFAULT_SPECS: Tuple[str, ...] = (
    ".edullm/run.hc-smoke.yaml",
    ".edullm/run.hc-baseline.yaml",
    ".edullm/run.hc-treatment.yaml",
)

#: Reproduced from ``edullm_platform.execution.FANOUT_PROLOGUE`` for the case where the CLI is
#: not installed beside this checkout. The import is tried first and this is the fallback, so a
#: machine with ``edullm`` present is always checking against the platform's own text; the copy
#: exists so that CI without the CLI still checks something, and the report says which was used.
FALLBACK_FANOUT_PROLOGUE = (
    'export EDULLM_OUTPUT_PREFIX="${EDULLM_OUTPUT_PREFIX}cell-${AWS_BATCH_JOB_ARRAY_INDEX}/"; '
    'export EDULLM_CHECKPOINT_DIR="${EDULLM_OUTPUT_PREFIX}checkpoints/"; '
    'export WANDB_RUN_ID="${WANDB_RUN_ID}-cell-${AWS_BATCH_JOB_ARRAY_INDEX}"; '
    "exec "
)

#: What the platform exports into a training container, with values chosen so that a value
#: leaking into the wrong field is recognisable in the output rather than plausible.
CONTAINER_ENVIRONMENT: Dict[str, str] = {
    "EDULLM_RUN_ID": "run_0199aaaa-bbbb-7ccc-8ddd-eeeeffff0000",
    "EDULLM_OUTPUT_PREFIX": "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/RUNID/",
    "EDULLM_CHECKPOINT_DIR": (
        "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/RUNID/checkpoints/"
    ),
    "EDULLM_DATASET_ID": "pretrain/regmix-10b",
    "EDULLM_DATASET_VERSION": "v1",
    "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
    "WANDB_RUN_ID": "run_0199aaaa-bbbb-7ccc-8ddd-eeeeffff0000",
}

#: A stand-in for ``python`` that writes its own argv out and exits. ``os.execv``-free and
#: dependency-free on purpose: it is invoked by whatever interpreter the outer shell finds, and
#: the only thing it must not do is import this repository.
STUB_PYTHON = """#!/bin/sh
# Records the argv the shell built, then exits 0 so the rest of the command text runs.
printf '%s\\n' "$@" > "$EDULLM_ARGV_CAPTURE"
exit 0
"""


@dataclass
class Finding:
    """One thing that is wrong, or one thing that was checked and held."""

    ok: bool
    what: str
    detail: str = ""


@dataclass
class SpecReport:
    """Everything this script concluded about one spec."""

    path: str
    prologue_source: str = ""
    argv: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)

    @property
    def failures(self) -> List[Finding]:
        return [finding for finding in self.findings if not finding.ok]


def fanout_prologue() -> Tuple[str, str]:
    """
    The prologue a fan-out cell's container runs before the submitted command.

    :returns: The prologue text and where it came from, so a report can say whether it was the
        platform's own or this file's copy of it.
    """
    try:
        from edullm_platform.execution import FANOUT_PROLOGUE  # type: ignore[import-not-found]
    except ImportError:
        return FALLBACK_FANOUT_PROLOGUE, "this file's copy (edullm CLI not installed)"
    return FANOUT_PROLOGUE, "edullm_platform.execution.FANOUT_PROLOGUE"


def load_train_on_corpus():
    """
    Import ``.edullm/train_on_corpus.py`` by path.

    It is not on ``sys.path`` and is not a package, and it is the file whose argument parser
    this script exists to check, so importing it by path is the only honest option.

    :returns: The imported module.

    :raises FileNotFoundError: If the entrypoint is not where the specs say it is.
    """
    path = REPO_ROOT / ".edullm" / "train_on_corpus.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_edullm_train_on_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def container_command(spec: dict, *, array_index: Optional[int]) -> Tuple[List[str], str]:
    """
    The argv a container is handed for one cell of this spec.

    :param spec: The parsed run spec.
    :param array_index: The fan-out index, or ``None`` for a single-container run.

    :returns: The container's argv and a note on where the fan-out prologue came from.
    """
    words = shlex.split(spec["command"])
    if array_index is None:
        return words, "n/a (no fan-out)"
    prologue, source = fanout_prologue()
    return ["bash", "-c", prologue + shlex.join(words)], source


def capture_argv(command: Sequence[str], *, array_index: Optional[int]) -> List[str]:
    """
    Run a container command with a stub ``python`` and return the argv that stub received.

    :param command: The container's argv.
    :param array_index: The fan-out index Batch would set, or ``None``.

    :returns: The arguments the training entrypoint would have been called with, the program
        name included.

    :raises RuntimeError: If the command exits nonzero or never invokes ``python``.
    """
    with tempfile.TemporaryDirectory() as workspace:
        stub_dir = Path(workspace) / "bin"
        stub_dir.mkdir()
        for name in ("python", "python3"):
            stub = stub_dir / name
            stub.write_text(STUB_PYTHON, encoding="utf-8")
            stub.chmod(0o755)

        capture = Path(workspace) / "argv.txt"
        environment = dict(os.environ)
        environment.update(CONTAINER_ENVIRONMENT)
        environment["EDULLM_ARGV_CAPTURE"] = str(capture)
        # In front of the real interpreter so the stub wins. A login shell may rewrite PATH,
        # which is why the specs use `bash -lc` and why this has to survive that: BASH_ENV is
        # not read by an interactive-less login shell, so the guard is that the stub directory
        # is first here and `-l` prepends rather than replaces on every image this runs on.
        environment["PATH"] = f"{stub_dir}{os.pathsep}{environment.get('PATH', '')}"
        if array_index is not None:
            environment["AWS_BATCH_JOB_ARRAY_INDEX"] = str(array_index)
            environment["EDULLM_FANOUT_INDEX_PARAMETER"] = "seed"

        finished = subprocess.run(
            list(command),
            env=environment,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
        )
        if finished.returncode != 0:
            raise RuntimeError(
                f"the container command exited {finished.returncode}\n"
                f"stdout: {finished.stdout}\nstderr: {finished.stderr}"
            )
        if not capture.is_file():
            raise RuntimeError(
                "the command never ran python, so there is no argv to check. "
                f"stdout: {finished.stdout}\nstderr: {finished.stderr}"
            )
        return capture.read_text(encoding="utf-8").splitlines()


def entrypoint_argv(argv: Sequence[str]) -> List[str]:
    """
    Strip the launcher off a recorded argv, leaving what the training script is handed.

    :param argv: The argv the ``python`` stub recorded, ``-m torch.distributed.run`` included.

    :returns: The arguments from the training script's path onward.

    :raises RuntimeError: If the training script is not in the argv at all.
    """
    for position, word in enumerate(argv):
        if word.endswith("train_on_corpus.py") or word.endswith("train_hc_moe.py"):
            return list(argv[position:])
    raise RuntimeError(f"no training entrypoint found in argv: {argv}")


def check_spec(path: Path, *, cells: Sequence[int]) -> SpecReport:
    """
    Check one spec end to end.

    :param path: The spec's path.
    :param cells: The fan-out indices to exercise. Ignored for a spec with no fan-out.

    :returns: The report.
    """
    report = SpecReport(path=str(path.relative_to(REPO_ROOT)))
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    train_on_corpus = load_train_on_corpus()

    fanout = spec.get("fanout")
    indices: List[Optional[int]] = (
        [index for index in cells if index < fanout["size"]] if fanout else [None]
    )
    if fanout and not indices:
        indices = [0]

    seeds_seen: Dict[Optional[int], Tuple[int, int, int]] = {}
    for index in indices:
        command, source = container_command(spec, array_index=index)
        report.prologue_source = source
        try:
            recorded = capture_argv(command, array_index=index)
        except (RuntimeError, subprocess.TimeoutExpired) as failure:
            report.findings.append(
                Finding(False, f"cell {index}: the command runs and reaches python", str(failure))
            )
            continue
        argv = entrypoint_argv(recorded)
        if index in (None, indices[0]):
            report.argv = argv

        opts, extras = train_on_corpus.build_parser().parse_known_args(argv[1:])

        report.findings.append(
            Finding(
                opts.run_name == CONTAINER_ENVIRONMENT["EDULLM_RUN_ID"],
                f"cell {index}: $EDULLM_RUN_ID expanded into the run name",
                f"got {opts.run_name!r}",
            )
        )
        # The whole reason the checkpoint guard exists: a save folder that is not the platform's
        # per-run prefix is a run whose retry starts from nothing and whose checkpoints nobody
        # can reach. Under a fan-out the prologue moves the prefix per cell, so the expected
        # value is the cell's and not the job's.
        expected_save = CONTAINER_ENVIRONMENT["EDULLM_CHECKPOINT_DIR"]
        if index is not None:
            expected_save = (
                f"{CONTAINER_ENVIRONMENT['EDULLM_OUTPUT_PREFIX']}cell-{index}/checkpoints/"
            )
        report.findings.append(
            Finding(
                opts.save_folder == expected_save,
                f"cell {index}: --save-folder is this cell's own checkpoint prefix",
                f"got {opts.save_folder!r}, expected {expected_save!r}",
            )
        )
        report.findings.append(
            Finding(
                opts.param_dtype == "bfloat16",
                f"cell {index}: --param-dtype reaches the parser as bfloat16",
                f"got {opts.param_dtype!r}",
            )
        )

        # The dotted overrides have to survive argparse and then land on a real config. Both
        # halves have failed independently before: argparse can swallow an unrecognised
        # positional, and a merge can accept a key that no field reads.
        from olmo_core.data import TokenizerConfig
        from olmo_core.nn.transformer import TransformerConfig

        vocab_size = TokenizerConfig.dolma2().padded_vocab_size()
        model = getattr(TransformerConfig, opts.model_factory)(vocab_size=vocab_size)
        model_overrides = [
            override[len("model.") :] for override in extras if override.startswith("model.")
        ]
        merged = model.merge(model_overrides) if model_overrides else model
        experiment_seed = next(
            (
                int(override.split("=", 1)[1])
                for override in extras
                if override.startswith("init_seed=")
            ),
            None,
        )
        seeds = (opts.data_seed, merged.init_seed, experiment_seed if experiment_seed else 0)
        seeds_seen[index] = seeds

        expected = 0 if index is None else index
        report.findings.append(
            Finding(
                seeds == (expected, expected, expected),
                f"cell {index}: all three seeds are {expected}",
                f"got data={seeds[0]} model.init={seeds[1]} experiment.init={seeds[2]}",
            )
        )
        report.findings.append(
            Finding(
                any(override.startswith("train_module.") for override in extras)
                or "train_module" not in spec["command"],
                f"cell {index}: train_module overrides survive argparse",
                f"extras: {extras}",
            )
        )

    if len(seeds_seen) > 1:
        distinct = len(set(seeds_seen.values()))
        report.findings.append(
            Finding(
                distinct == len(seeds_seen),
                f"the {len(seeds_seen)} cells checked draw {len(seeds_seen)} distinct seed "
                "triples",
                # The failure this line exists for is the expensive one: identical cells
                # measure a noise floor of zero and make every later arm significant.
                f"got {distinct} distinct out of {len(seeds_seen)}: {seeds_seen}",
            )
        )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the CLI.

    :param argv: Arguments, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code: 0 if every spec checked held, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="a run spec to check; repeatable. Defaults to every spec this repository ships.",
    )
    parser.add_argument(
        "--cells",
        default="0,1,4",
        help="which fan-out indices to exercise, comma separated (default: 0,1,4)",
    )
    parser.add_argument("--json", action="store_true", help="print one JSON document instead")
    args = parser.parse_args(argv)

    cells = [int(entry) for entry in args.cells.split(",") if entry.strip()]
    wanted = args.spec if args.spec else list(DEFAULT_SPECS)

    reports: List[SpecReport] = []
    missing: List[str] = []
    for entry in wanted:
        path = REPO_ROOT / entry
        if not path.is_file():
            # A default spec that does not exist yet is not a failure; a spec somebody named
            # explicitly is.
            if args.spec:
                missing.append(entry)
            continue
        reports.append(check_spec(path, cells=cells))

    failed = sum(len(report.failures) for report in reports) + len(missing)

    if args.json:
        print(
            json.dumps(
                {
                    "specs": [
                        {
                            "path": report.path,
                            "prologue_source": report.prologue_source,
                            "entrypoint_argv": report.argv,
                            "findings": [
                                {"ok": f.ok, "what": f.what, "detail": f.detail}
                                for f in report.findings
                            ],
                        }
                        for report in reports
                    ],
                    "missing": missing,
                    "failures": failed,
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    for entry in missing:
        print(f"MISSING  {entry}")
    for report in reports:
        print(f"\n=== {report.path}")
        print(f"    fan-out prologue: {report.prologue_source}")
        print(f"    entrypoint argv: {shlex.join(report.argv)}")
        for finding in report.findings:
            print(f"    [{'ok  ' if finding.ok else 'FAIL'}] {finding.what}")
            if not finding.ok:
                print(f"           {finding.detail}")
    print(
        f"\n{len(reports)} spec(s) checked, {failed} problem(s).\n"
        "Nothing here trained, reached a network, or allocated a GPU. What it establishes is "
        "that\nthe command text produces the arguments it was written to produce, in the "
        "container the\nplatform builds around it."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
