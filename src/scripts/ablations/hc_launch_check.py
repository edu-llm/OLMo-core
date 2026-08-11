"""
What ``edullm check`` cannot ask: does the command in a run spec, once the platform has
wrapped it and a shell has run it, build the config it was meant to build?

``edullm check`` reads the command's *words*. It answers whether a launcher is in command
position, whether the process count matches the shape, whether a dtype the hardware lacks is
named, and whether the checkpoint variable is written somewhere a shell would expand. It
cannot answer the questions that actually cost a tranche: whether the third cell of a fan-out
hands ``train_on_corpus.py`` a seed of 3, and whether ``train_module.compile_model=false``
reaches the field it names.

So this runs the whole path:

1. Reads a spec and splits its command exactly as the submission workflow does (``shlex``).
2. Wraps it in :func:`edullm_platform.execution.fanout_cell_command`'s prologue when the spec
   declares a fan-out, which is the text a cell's container is actually handed.
3. Runs that text in a real ``bash`` with the environment Batch sets, against a stub ``python``
   that records its argv rather than training anything.
4. Feeds the recorded argv to **``train_on_corpus.build_config``**, the container's own config
   constructor, with only the corpus resolution stubbed out — and asserts every value in
   :data:`SPEC_EXPECTATIONS` against the resulting config.

**Step 4 is the whole of why this file was rewritten.** An earlier version checked that some
leftover argument began with ``train_module.`` and that the seeds parsed, which is a check
that cannot fail: deleting ``train_module.compile_model=false`` outright made the second
disjunct true and the finding stayed green, while the run it cleared had ``torch.compile``
silently back on for every cell. Two independent audits found that, along with three more
mutations it passed. The expectations table below is the remedy and it is deliberately a
maintenance burden: changing a spec's shape now requires changing this file, which is what
makes a silent change impossible.

What it still cannot do is run the model, reach S3, or say anything about throughput. Those
need a GPU and a corpus; see ``docs/hc-ablation/AGENT-STATUS.md``.

    python src/scripts/ablations/hc_launch_check.py
    python src/scripts/ablations/hc_launch_check.py --spec .edullm/run.hc-baseline.yaml
    python src/scripts/ablations/hc_launch_check.py --json
"""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

#: What every spec's command has to build, checked against the config
#: ``train_on_corpus.build_config`` actually produces. A spec whose shape changes without this
#: table changing is a failing run of this script rather than a discovery on the platform.
#:
#: ``seed`` is ``None`` where the seed comes from the fan-out index, in which case each cell is
#: expected to draw its own index; anything else is a literal the command hard-codes.
SPEC_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    ".edullm/run.hc-smoke.yaml": {
        "entrypoint": "train_on_corpus.py",
        "model_factory": "smallmoe",
        "sequence_length": 2048,
        "global_batch_size": 262_144,
        "rank_microbatch_size": 8_192,
        "steps": 40,
        "save_interval": 20,
        "warmup_steps": 10,
        "learning_rate": 6e-4,
        "param_dtype": "bfloat16",
        "compile_model": False,
        "seed": 0,
        "launcher_processes": 1,
        "expects_fanout": False,
    },
    ".edullm/run.hc-baseline.yaml": {
        "entrypoint": "train_on_corpus.py",
        "model_factory": "smallmoe",
        "sequence_length": 2048,
        "global_batch_size": 262_144,
        "rank_microbatch_size": 8_192,
        "steps": 3_000,
        "save_interval": 125,
        "warmup_steps": 200,
        "learning_rate": 6e-4,
        "param_dtype": "bfloat16",
        "compile_model": False,
        "seed": None,
        "launcher_processes": 1,
        "expects_fanout": True,
    },
    ".edullm/run.hc-treatment.yaml": {
        # A different entrypoint, so `--model-factory` is not a flag it takes: the shape comes
        # from `train_hc_moe.build_model_config`, and the arm comes from the cell index.
        "model_factory": None,
        "entrypoint": "train_hc_moe.py",
        "arms_by_cell": {0: "mhc_moe", 1: "mhc_moe", 4: "mhc_moe"},
        "sequence_length": 2048,
        "global_batch_size": 262_144,
        "rank_microbatch_size": 8_192,
        "steps": 3_000,
        "save_interval": 125,
        "warmup_steps": 200,
        "learning_rate": 6e-4,
        "param_dtype": "bfloat16",
        "compile_model": False,
        "seed": None,
        "launcher_processes": 1,
        "expects_fanout": True,
        "fanout_size": 20,
    },
}

#: The launcher every spec is required to name, and the flag whose value has to match the
#: compute profile's device count. Asserted exactly rather than pattern-matched: a mutation to
#: ``torch.distributed.runn`` used to pass, and it is a container that dies on `No module named`.
REQUIRED_LAUNCHER_MODULE = "torch.distributed.run"

#: Reproduced from ``edullm_platform.execution.FANOUT_PROLOGUE`` for the case where the CLI is
#: not installed. The import is tried first, including into the ``uv`` tool's own virtual
#: environment, and this is the fallback; the report says which was used and the run fails if it
#: had to fall back while the CLI is on the PATH, because that is drift the copy would hide.
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

#: A stand-in for ``python`` that writes its own argv out and exits.
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

    def record(self, ok: bool, what: str, detail: str = "") -> None:
        self.findings.append(Finding(bool(ok), what, detail))


def _import_edullm_platform():
    """
    Import ``edullm_platform``, looking inside the ``uv`` tool venv if it is not importable.

    ``edullm`` is installed as a uv tool, which puts it in its own virtual environment rather
    than on this interpreter's path, so a plain import fails on the very machines where the CLI
    is present. That made the drift check below report "not installed" on a machine with the
    CLI installed, so it never once compared anything.

    :returns: The module, or ``None``.
    """
    try:
        import edullm_platform  # type: ignore[import-not-found]

        return edullm_platform
    except ImportError:
        pass
    home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        home / "uv" / "tools" / "edullm" / "lib" / version / "site-packages",
        home / "uv" / "tools" / "edullm" / "lib" / sysconfig.get_python_version(),
    ]
    for candidate in candidates:
        if (candidate / "edullm_platform").is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import edullm_platform  # type: ignore[import-not-found]

                return edullm_platform
            except ImportError:
                sys.path.pop(0)
    return None


def _cli_is_on_the_path() -> bool:
    """Whether ``edullm`` is runnable, which is what makes a failed import drift rather than
    an absence."""
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / "edullm"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def fanout_prologue() -> Tuple[str, str]:
    """
    The prologue a fan-out cell's container runs before the submitted command.

    :returns: The prologue text and where it came from.
    """
    module = _import_edullm_platform()
    if module is not None:
        from edullm_platform.execution import FANOUT_PROLOGUE  # type: ignore[import-not-found]

        return FANOUT_PROLOGUE, "edullm_platform.execution.FANOUT_PROLOGUE"
    return FALLBACK_FANOUT_PROLOGUE, "this file's copy (edullm_platform not importable)"


def load_entrypoint(filename: str = "train_on_corpus.py"):
    """
    Import one of the ``.edullm/`` entrypoints by path.

    Which one matters: the tranche's arms run ``train_hc_moe.py``, whose parser takes ``--cell``
    and whose ``build_config`` resolves an arm from it. Checking every spec against
    ``train_on_corpus``'s parser reported the treatment spec's own flags as unrecognised
    overrides, which is this script being wrong rather than the spec.

    :param filename: The entrypoint's file name under ``.edullm/``.

    :returns: The imported module.

    :raises FileNotFoundError: If the entrypoint is not where the specs say it is.
    """
    path = REPO_ROOT / ".edullm" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"_edullm_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed. ``ExperimentConfig`` is a dataclass whose field types
    # are resolved by name out of its defining module, so a module that is not in
    # ``sys.modules`` builds fine and then raises ``No module named`` from inside
    # ``build_config`` -- which reads as the command being wrong rather than as this file being
    # wrong, which is the worst way for a checker to fail.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub_corpus(module):
    """
    Replace ``resolve_corpus`` with something that needs no network and no credential.

    Everything downstream of it — the model factory lookup, the dataset config, the train
    module and the trainer, and the dotted-override merge that is the point of this script — is
    the container's own code, unmodified.

    :param module: The imported ``train_on_corpus`` module.
    """
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig

    def resolve(*, dataset_id: str, version: str, tokenizer_id: str):
        return module.Corpus(
            dataset_id=dataset_id or "pretrain/regmix-10b",
            version=version or "v1",
            paths=[f"s3://edullm-data/pretrain/regmix-10b/v1/shard-{i:03d}.npy" for i in range(4)],
            dtype=NumpyDatasetDType.uint32,
            tokenizer=TokenizerConfig.dolma2(),
            rows=None,
        )

    module.resolve_corpus = resolve


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

    :returns: The arguments the training entrypoint would have been called with.

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


def split_launcher(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Separate the launcher's own arguments from the training script's.

    :param argv: The argv the ``python`` stub recorded.

    :returns: ``(launcher_argv, entrypoint_argv)``.

    :raises RuntimeError: If no training entrypoint is present.
    """
    for position, word in enumerate(argv):
        if word.endswith(".py") and "/" in word:
            return list(argv[:position]), list(argv[position:])
    raise RuntimeError(f"no training entrypoint found in argv: {argv}")


def _expectations_for(path: str) -> Dict[str, Any]:
    """
    The expectations for one spec.

    :param path: The spec's repository-relative path.

    :returns: The expectations.

    :raises KeyError: If the spec has no entry, which is the point of the table.
    """
    return SPEC_EXPECTATIONS[path]


def check_spec(path: Path, *, cells: Sequence[int]) -> SpecReport:
    """
    Check one spec end to end.

    :param path: The spec's path.
    :param cells: The fan-out indices to exercise. Ignored for a spec with no fan-out.

    :returns: The report.
    """
    relative = str(path.relative_to(REPO_ROOT))
    report = SpecReport(path=relative)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = _expectations_for(relative)
    entrypoint = load_entrypoint(expected["entrypoint"])
    # The arm entrypoint delegates every corpus decision to the sibling it imports, so the stub
    # goes on whichever module owns `resolve_corpus`.
    _stub_corpus(getattr(entrypoint, "TOC", entrypoint))

    fanout = spec.get("fanout")
    if "fanout_size" in expected:
        report.record(
            bool(fanout) and fanout["size"] == expected["fanout_size"],
            f"the fan-out declares {expected['fanout_size']} cells",
            f"got {fanout}. It has to equal arms x --seeds-per-arm, and nothing on the "
            "platform checks that.",
        )
    report.record(
        bool(fanout) == expected["expects_fanout"],
        "the fan-out block matches what the experiment design expects",
        f"spec fanout={fanout}, expected_fanout={expected['expects_fanout']}",
    )
    indices: List[Optional[int]] = (
        [index for index in cells if index < fanout["size"]] if fanout else [None]
    )
    if fanout and not indices:
        indices = [0]

    # A fan-out command must refuse to start without its index, or five cells silently run one
    # replicate and the measured noise floor is exactly zero. Checked by running it with the
    # variable unset and requiring a nonzero exit.
    if fanout:
        try:
            bare, _ = container_command(spec, array_index=None)
            capture_argv(bare, array_index=None)
            ran_without_index = True
        except (RuntimeError, subprocess.TimeoutExpired):
            ran_without_index = False
        report.record(
            not ran_without_index,
            "the command refuses to start when the fan-out index is unset",
            "it started anyway, so an unset or renamed index runs every cell on one seed",
        )

    seeds_seen: Dict[Optional[int], Tuple[int, int, int]] = {}
    for index in indices:
        command, source = container_command(spec, array_index=index)
        report.prologue_source = source
        try:
            recorded = capture_argv(command, array_index=index)
        except (RuntimeError, subprocess.TimeoutExpired) as failure:
            report.record(False, f"cell {index}: the command runs and reaches python", str(failure))
            continue
        launcher, argv = split_launcher(recorded)
        if index in (None, indices[0]):
            report.argv = argv

        # THE LAUNCHER, EXACTLY. `torch.distributed.runn` used to pass here and is a container
        # that dies on `No module named`.
        report.record(
            launcher[:2] == ["-m", REQUIRED_LAUNCHER_MODULE],
            f"cell {index}: the launcher is python -m {REQUIRED_LAUNCHER_MODULE}",
            f"got {launcher}",
        )
        report.record(
            f"--nproc-per-node={expected['launcher_processes']}" in launcher,
            f"cell {index}: --nproc-per-node is {expected['launcher_processes']}",
            f"got {launcher}",
        )
        # `exec` in front of the launcher is what lets a SIGTERM from `edullm cancel` or from
        # the attempt timeout reach the trainer instead of killing a wrapper shell that leaves
        # it running and billing.
        report.record(
            "exec " in spec["command"],
            f"cell {index}: the launcher is exec'd rather than run as a child of the wrapper",
            "without exec, SIGTERM kills the wrapper and the trainer keeps running",
        )

        opts, extras = entrypoint.build_parser().parse_known_args(argv[1:])

        # `init_seed=` absent is NOT the same as `init_seed=0`: the field's default is 12536, so
        # a spec that dropped the override would run a seed the checker reported as 0. Only
        # asserted where the command is what sets the seed; the arm entrypoint sets all three
        # from the cell index instead, and the seed assertions below cover it either way.
        if expected["entrypoint"] == "train_on_corpus.py":
            report.record(
                len([o for o in extras if o.startswith("init_seed=")]) == 1,
                f"cell {index}: exactly one init_seed= override is present",
                f"extras: {extras}",
            )

        # THE CONFIG THE CONTAINER WOULD BUILD, from the container's own constructor.
        try:
            config = entrypoint.build_config(opts, extras)
            built = True
            detail = ""
        except Exception as failure:  # noqa: BLE001 - a bad override is exactly what this finds
            config, built, detail = None, False, repr(failure)
        report.record(
            built, f"cell {index}: train_on_corpus.build_config accepts the command", detail
        )
        if config is None:
            continue

        # The seed a cell draws. For the fan-out-over-replicates specs it is the index; for
        # the 2x2 the index also picks an arm, so the replicate is the remainder.
        if expected["seed"] is not None:
            expected_seed = expected["seed"]
        elif "arms_by_cell" in expected:
            expected_seed = (index or 0) % opts.seeds_per_arm
        else:
            expected_seed = index
        checks: List[Tuple[str, Any, Any]] = [
            ("model factory", opts.model_factory, expected["model_factory"]),
            ("sequence length", config.dataset.sequence_length, expected["sequence_length"]),
            (
                "global batch size",
                config.data_loader.global_batch_size,
                expected["global_batch_size"],
            ),
            (
                "rank microbatch size",
                config.train_module.rank_microbatch_size,
                expected["rank_microbatch_size"],
            ),
            ("steps", opts.steps, expected["steps"]),
            ("save interval", opts.save_interval, expected["save_interval"]),
            ("warmup steps", opts.warmup_steps, expected["warmup_steps"]),
            ("learning rate", config.train_module.optim.lr, expected["learning_rate"]),
            (
                "param dtype",
                str(config.train_module.dp_config.param_dtype),
                expected["param_dtype"],
            ),
            ("compile_model", config.train_module.compile_model, expected["compile_model"]),
            ("experiment init seed", config.init_seed, expected_seed),
            ("model init seed", config.model.init_seed, expected_seed),
            ("data loader seed", config.data_loader.seed, expected_seed),
        ]
        if expected["model_factory"] is None:
            checks = [entry for entry in checks if entry[0] != "model factory"]
        # The 2x2's whole design is that a cell index picks an arm. A modulus or an off-by-one
        # here gives twenty cells one arm, or two cells one replicate reported as two.
        for cell_index, arm in expected.get("arms_by_cell", {}).items():
            if cell_index == index:
                resolved, resolved_seed = entrypoint.resolve_cell(
                    index, seeds_per_arm=opts.seeds_per_arm, arm=None
                )
                checks.append((f"cell {index} resolves to an arm", resolved, arm))
                checks.append((f"cell {index} resolves to a seed", resolved_seed, expected_seed))
        for name, actual, want in checks:
            report.record(actual == want, f"cell {index}: {name} is {want!r}", f"got {actual!r}")

        expected_save = CONTAINER_ENVIRONMENT["EDULLM_CHECKPOINT_DIR"]
        if index is not None:
            expected_save = (
                f"{CONTAINER_ENVIRONMENT['EDULLM_OUTPUT_PREFIX']}cell-{index}/checkpoints/"
            )
        report.record(
            config.trainer.save_folder == expected_save,
            f"cell {index}: the save folder is this cell's own checkpoint prefix",
            f"got {config.trainer.save_folder!r}, expected {expected_save!r}",
        )
        seeds_seen[index] = (config.data_loader.seed, config.model.init_seed, config.init_seed)

    if len(seeds_seen) > 1:
        distinct = len(set(seeds_seen.values()))
        report.record(
            distinct == len(seeds_seen),
            f"the {len(seeds_seen)} cells checked draw {len(seeds_seen)} distinct seed triples",
            f"got {distinct} distinct out of {len(seeds_seen)}: {seeds_seen}",
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
    parser.add_argument("--spec", action="append", default=None, help="a run spec; repeatable")
    parser.add_argument(
        "--cells", default="0,1,4", help="which fan-out indices to exercise (default: 0,1,4)"
    )
    parser.add_argument("--json", action="store_true", help="print one JSON document instead")
    args = parser.parse_args(argv)

    cells = [int(entry) for entry in args.cells.split(",") if entry.strip()]
    wanted = args.spec if args.spec else list(SPEC_EXPECTATIONS)

    reports: List[SpecReport] = []
    missing: List[str] = []
    for entry in wanted:
        path = REPO_ROOT / entry
        if not path.is_file():
            # A MISSING SPEC IS A FAILURE, WHICH IT USED NOT TO BE. Skipping one silently meant
            # deleting a spec reduced this script's coverage while leaving it green.
            missing.append(entry)
            continue
        reports.append(check_spec(path, cells=cells))

    drift = []
    if _cli_is_on_the_path() and _import_edullm_platform() is None:
        drift.append(
            "edullm is on the PATH and edullm_platform is not importable, so the fan-out "
            "prologue was checked against this file's copy rather than the platform's."
        )

    failed = sum(len(report.failures) for report in reports) + len(missing) + len(drift)

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
                    "drift": drift,
                    "failures": failed,
                },
                indent=2,
            )
        )
        return 1 if failed else 0

    for entry in missing:
        print(f"MISSING  {entry} -- it is in SPEC_EXPECTATIONS and not on disk")
    for note in drift:
        print(f"DRIFT    {note}")
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
        "that\nthe command text builds the config it was written to build, inside the "
        "container the\nplatform assembles around it."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
