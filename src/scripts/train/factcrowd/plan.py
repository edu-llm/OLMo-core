"""
The submission plan: which jobs to run, in what order, and what each one answers.

**Why this is a program and not a section of a document.** ``.edullm/run.yaml`` holds one command per
repository, so every submission is an edit to that file, and the edit has to get four things right at once:
the fan-out size must match the config directory's file count, the fan-out index must map to the cell the
approval was granted for, the dtype must appear in the *text* of the command, and the whole thing must be
committed to an ``edullm/`` branch because the platform builds from the commit rather than the working tree.
A prose description of that is a thing to copy by hand at 2am. ``stage`` writes it.

It stops where the platform starts. This writes a file; it reaches no network, dispatches nothing, and
**quotes no price** -- ``edullm check --json`` is the only thing that knows what a job costs and who has to
approve it, those live in reviewed configuration that changes without anybody being told, and this program
would be a stale copy of both. Run ``check`` before ``submit``, match refusals on ``code``, and read ``cost``
and ``approval_class`` out of its output.

Usage::

    python src/scripts/train/factcrowd/plan.py list
    python src/scripts/train/factcrowd/plan.py stage smoke
    python src/scripts/train/factcrowd/plan.py stage calibration
    python src/scripts/train/factcrowd/plan.py stage entropy --ctxmano-length 4 --mano-length 6

The three jobs after calibration need the depth it selects, so they refuse to stage without it rather than
defaulting to a length nobody measured -- which is how eighteen cells came to be trained at a depth where
the endpoint had no dynamic range.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factcrowd import cells as cells_module  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = Path(__file__).resolve().parent / "configs" / "cells"
RUN_YAML = REPO_ROOT / ".edullm" / "run.yaml"

#: Proven available, and 8x returned 1.06x of it at 28M -- these sizes are launch- and communication-bound
#: rather than FLOP-bound, so more devices buy almost nothing. Measured, not assumed.
DEFAULT_COMPUTE = "gpu-4xa10g"

#: A scoring pass loads checkpoints and runs forward passes; one device is enough and the queue is shorter.
SCORING_COMPUTE = "gpu-1xa10g"

#: 1h and one attempt, and it promises no checkpoint. Right for a smoke run and for scoring.
CHECK_WORKLOAD = "olmo-core-check"

#: 24h, two attempts, and it promises a checkpoint a retry resumes from. Right for anything that trains.
TRAIN_WORKLOAD = "olmo-core-train"

#: Device counts, read from the platform's own ``config/accelerators.yaml`` when its checkout is beside this
#: repository and falling back to this table when it is not.
#:
#: **The fallback is the part worth reading.** ``edullm submit`` refuses ``process_per_device`` when the
#: number of processes the command starts differs from the number of cards the profile bills for -- in either
#: direction, since two ranks on a four-GPU shape idle two cards and four ranks on a one-GPU shape is an
#: invalid device ordinal. The count therefore has to be right, and the authority is
#: ``edullm_platform.launchers.CONTAINER_SHAPES``, which cannot be imported under Python 3.11 because the
#: package uses PEP 695 generics. So the config file is read directly, and this table exists only for a
#: machine with neither.
_FALLBACK_DEVICES: Dict[str, int] = {
    "cpu-32vcpu": 0,
    "gpu-1xt4": 1,
    "gpu-1xl4": 1,
    "gpu-1xa10g": 1,
    "gpu-1xl40s": 1,
    "gpu-4xt4": 4,
    "gpu-4xl4": 4,
    "gpu-4xa10g": 4,
    "gpu-4xl40s": 4,
    "gpu-8xt4": 8,
    "gpu-8xl4": 8,
    "gpu-8xa10g": 8,
    "gpu-8xa100": 8,
    "gpu-8xl40s": 8,
}

#: What the platform splices in when it corrects a command itself, so the shape matches its own suggestion.
_LAUNCHER = "-m torch.distributed.run --nproc-per-node={devices} --standalone"

#: Written into the *text* of every training command. The precision guard reads the words of the command and
#: cannot see a dtype the program sets in code, so a command that omits it is accepted onto a card with no
#: bfloat16 in hardware and dies on the first kernel that needs the format -- after being billed.
DTYPE_OVERRIDE = "train_module.dp_config.param_dtype=bfloat16"


def devices_for(compute_profile: str) -> int:
    """
    How many cards a compute profile bills for.

    :param compute_profile: e.g. ``"gpu-4xa10g"``.

    :returns: The device count; ``0`` for a CPU profile, which is not checked.

    :raises OLMoConfigurationError: If the profile is unknown to both the platform's config and the
        fallback, since guessing a device count is how a run comes to idle three cards.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent.parent / "platform" / "config" / "accelerators.yaml"
        if candidate.is_file():
            import yaml

            for entry in yaml.safe_load(candidate.read_text())["profiles"]:
                if entry["profile"] == compute_profile:
                    return int(entry["devices"])
            break
    if compute_profile not in _FALLBACK_DEVICES:
        raise OLMoConfigurationError(
            f"unknown compute profile {compute_profile!r}, so the number of processes the command must "
            f"start is unknown. `edullm submit` refuses a mismatch as 'process_per_device'. Known here: "
            f"{sorted(_FALLBACK_DEVICES)}"
        )
    return _FALLBACK_DEVICES[compute_profile]


@dataclass(frozen=True)
class Job:
    """
    One submission.

    :param name: What to pass to ``stage``.
    :param answers: The question this job exists to settle, in one line.
    :param config_dir: Directory under ``configs/cells`` whose files are the fan-out.
    :param compute: ``suggested_compute``, which also decides how many processes the launcher starts.
    :param workload: The policy preset, and it is not a cost knob -- the two differ in what they promise,
        not in what they charge. ``olmo-core-check`` is 1h, one attempt and no checkpoint contract;
        ``olmo-core-train`` is 24h, two attempts and promises a checkpoint a retry resumes from. Anything
        that trains for real needs the second; a scoring pass needs the first, since more than one attempt
        on a workload that checkpoints nothing earns ``retry_without_a_checkpoint_contract``.
    :param needs_lengths: Whether the configs have to be generated from calibration's result first.
    :param blocked_by: Jobs that must finish first, and why.
    :param scoring: A scoring job rather than a training one.
    """

    name: str
    answers: str
    config_dir: Optional[str] = None
    compute: str = DEFAULT_COMPUTE
    workload: str = TRAIN_WORKLOAD
    needs_lengths: bool = False
    blocked_by: Tuple[str, ...] = ()
    scoring: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)


JOBS: Tuple[Job, ...] = (
    Job(
        name="smoke",
        answers="Does the pipeline run at all, including the in-context endpoint that has never left a CPU?",
        config_dir="smoke",
        # One card and the check preset: seconds of work, nothing worth resuming, and `gpu-1xa10g` places
        # reliably where `gpu-4xa10g` waits in a queue. One device also means no launcher to get wrong.
        compute="gpu-1xa10g",
        workload=CHECK_WORKLOAD,
        notes=(
            "Seconds per cell. Worth it before the 11B-token sweep: <ctxmano> items are 256 tokens with a "
            "fresh operator table each and padding that tiles the instance, none of which had run on a GPU.",
        ),
    ),
    Job(
        name="calibration",
        answers="Is either reasoning endpoint learnable at 28M, and at which depth? G1's evidence.",
        config_dir="calibration",
        notes=(
            "GATES EVERYTHING. Phase 1 trained eighteen confirmatory cells at a depth where <mano> was a "
            "constant function; this is the experiment that would have caught it, and it is 1/11 the cost.",
            "Indices 0-3 are <ctxmano> at lengths 2-5 (confirmatory); 4-10 are <mano> at 2-10 (secondary).",
        ),
    ),
    Job(
        name="score-calibration",
        answers="Which depth clears its admission band, and does the table probe separate the confound?",
        compute=SCORING_COMPUTE,
        workload=CHECK_WORKLOAD,
        scoring=True,
        blocked_by=("calibration",),
        notes=(
            "Reads the sweep and writes a gate report. Do NOT pass --last-only: it drops the step-0 "
            "checkpoint, which is G2's only evidence.",
            "Choose the hardest depth that reaches its endpoint's bound -- 28.4% for <ctxmano>, 23.5% for "
            "<mano> -- with a >=15pp spread across depths.",
        ),
    ),
    Job(
        name="ladder",
        answers="Does the endpoint respond to reasoning-token share at all? G8's positive control.",
        config_dir="ladder_p2",
        needs_lengths=True,
        blocked_by=("score-calibration",),
        notes=(
            "Iso-token: zero-bit biographies backfill what the dose removes, so every arm trains the same "
            "steps on the same tokens and only reasoning share moves.",
            "Must carry the confirmatory endpoint's variant, or a --gate-endpoint ctxmano report finds no "
            "ladder and G8 comes back owed.",
        ),
    ),
    Job(
        name="entropy",
        answers="THE PRIMARY RESULT: does reasoning decline as demanded fact entropy rises, iso-token?",
        config_dir="entropy_p2",
        needs_lengths=True,
        blocked_by=("score-calibration",),
        notes=(
            "The identified axis: entity count, token budget and mixture are held, only entropy moves.",
            "Three replicates, differing in initialisation and data order only. df=2, so report the "
            "interval from the measured sigma rather than a margin chosen in advance.",
        ),
    ),
    Job(
        name="count",
        answers="Does the same pattern hold on the confounded count axis? Descriptive sensitivity.",
        config_dir="count_p2",
        needs_lengths=True,
        blocked_by=("entropy",),
        notes=(
            "NOT a decomposition of the entropy axis. The two differ in schema, vocabulary, entity count, "
            "tokens, steps and mixture, so subtracting their slopes isolates nothing.",
            "A gate report from the entropy architecture will not admit these rows, correctly. Either "
            "calibrate the count architecture too or state that they are not confirmatory.",
        ),
    ),
    Job(
        name="score-m0",
        answers="Re-score phase 1 and write the gate report that never reached S3. Optional; phase 2 supersedes it.",
        compute=SCORING_COMPUTE,
        workload=CHECK_WORKLOAD,
        scoring=True,
        notes=(
            "The command that was in .edullm/run.yaml before this program overwrote it, preserved so the "
            "three run prefixes are not lost. Its verdict is already known -- G4 refused the endpoint for "
            "having no dynamic range -- but the report itself never survived the container.",
            "It is worth more now than when it was written: G2 can close off the step-0 checkpoints these "
            "runs already wrote, which nothing could read before.",
        ),
    ),
)

#: The three phase-1 submissions the M0 gate report was assembled from. Kept because ``.edullm/run.yaml``
#: holds one command and staging a phase-2 job overwrites it -- and these are not recoverable from a run id
#: without ``edullm status``. The sigma block has no dilution ladder, the ladder has one replicate where G7
#: needs three, and round two carries the re-runs of three cells that crashed; ``select_complete`` resolves
#: each to the finished run and drops the crash, giving 14 unique cells.
M0_PREFIXES: Tuple[str, ...] = (
    "s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/"
    "run_019fd3c0-8d2c-70ce-873e-4c2e333856b6/",
    "s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/"
    "run_019fd3bf-ece6-708d-9706-08967ddbd557/",
    "s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/"
    "run_019fdd84-9e11-707b-bcd3-adbadb4468ea/",
)

JOBS_BY_NAME: Dict[str, Job] = {job.name: job for job in JOBS}


def train_command(config_dir: str, *, compute_profile: str) -> str:
    """
    The training command for a fan-out over one config directory.

    Three things here are refusals when they are wrong, and each cost a submission to learn:

    - **One process per device.** Nothing wraps what you type, so the launcher goes in the command. A
      four-card profile that starts one process idles three cards and is refused as ``process_per_device``
      -- in both directions, since four ranks on one card is an ``invalid device ordinal``. Omitted at one
      device or fewer: a CPU profile is not checked and a single card needs no rendezvous.
    - **``$EDULLM_CHECKPOINT_DIR``, on the command line.** Checkpoints must go there, and the platform reads
      the *text* of the command to check that a run promising one will write one -- it cannot see inside the
      program. ``${EDULLM_OUTPUT_PREFIX}ckpt`` is a different prefix and earns
      ``checkpoint_path_not_in_command``. OLMo-core's own default is ``/tmp``, on a machine that stops
      existing, so a run that takes it exits zero having saved nothing.
    - **``bash -lc``.** The container runs the command directly with no shell, so without it
      ``$EDULLM_RUN_ID`` arrives as eighteen literal characters rather than a run id.

    :param config_dir: Directory name under ``configs/cells``.
    :param compute_profile: Decides how many processes the launcher starts.

    :returns: A single-line shell command.
    """
    devices = devices_for(compute_profile)
    launcher = f"{_LAUNCHER.format(devices=devices)} " if devices > 1 else ""
    return (
        f"bash -lc 'python {launcher}"
        f'src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" '
        f"--config-dir src/scripts/train/factcrowd/configs/cells/{config_dir} "
        f'--cell-index "$AWS_BATCH_JOB_ARRAY_INDEX" '
        f'--save-folder "$EDULLM_CHECKPOINT_DIR" '
        f"{DTYPE_OVERRIDE}'"
    )


def scoring_command(prefixes: Sequence[str], *, endpoint: str = "ctxmano") -> str:
    """
    The scoring command for one or more run output prefixes.

    :param prefixes: Output prefixes of finished runs, e.g. ``s3://.../runs/run_x/``.
    :param endpoint: Which endpoint the gate report is about.

    :returns: A single-line shell command.

    :raises OLMoConfigurationError: If no prefix is given, since scoring nothing is not a submission.
    """
    if not prefixes:
        raise OLMoConfigurationError(
            "scoring needs at least one --prefix, which is the EDULLM_OUTPUT_PREFIX of a finished run. "
            "`edullm status --json` names your recent submissions and is free."
        )
    joined = " ".join(prefixes)
    return (
        f"bash -lc 'python src/scripts/train/factcrowd/score_run.py --prefix {joined} "
        f"--out ${{EDULLM_OUTPUT_PREFIX}}scores.csv --work-dir /tmp/score "
        f"--device cuda --dtype float32 --batch-size 256 "
        f"--write-gate-report ${{EDULLM_OUTPUT_PREFIX}}gates-{endpoint}.json "
        f"--gate-endpoint {endpoint} --json'"
    )


def fanout_size(config_dir: str) -> int:
    """
    How many cells a directory holds, which is the fan-out size.

    Counted rather than declared: a size from an older directory runs a different cell under the name the
    approval was granted for, and the index maps by *filename*.

    :param config_dir: Directory name under ``configs/cells``.

    :returns: The count.

    :raises OLMoConfigurationError: If the directory is missing or empty.
    """
    target = CONFIG_ROOT / config_dir
    if not target.is_dir():
        raise OLMoConfigurationError(
            f"no config directory at {target}. If this job needs lengths from calibration, generate them "
            f"first: `plan.py stage <job> --ctxmano-length N --mano-length M`."
        )
    found = sorted(target.glob("*.yaml"))
    if not found:
        raise OLMoConfigurationError(f"{target} holds no cell configs")
    return len(found)


def render(job: Job, *, prefixes: Sequence[str] = (), endpoint: str = "ctxmano") -> str:
    """
    The full ``run.yaml`` text for one job.

    :param job: The job.
    :param prefixes: For a scoring job, the run prefixes to score.
    :param endpoint: For a scoring job, the endpoint to admit.

    :returns: YAML text.
    """
    lines = [
        f"# {job.name}: {job.answers}",
        "#",
    ]
    for note in job.notes:
        for wrapped in _wrap(note):
            lines.append(f"# {wrapped}")
        lines.append("#")
    lines += [
        "# Written by src/scripts/train/factcrowd/plan.py -- edit freely, but keep the dtype in the",
        "# command text: the precision guard reads the words of the command and cannot see one set in code.",
        "#",
        "# Commit this to a branch named edullm/<something> and push. The platform builds the image from the",
        "# last commit, so nothing uncommitted is part of the run. Then `edullm check --json`, then submit.",
        "schema_version: 1",
        f"workload_profile: {job.workload}",
        f"suggested_compute: {job.compute}",
    ]
    if job.scoring:
        resolved = tuple(prefixes) or (M0_PREFIXES if job.name == "score-m0" else ())
        lines.append("command: >-")
        lines += [f"  {part}" for part in _fold(scoring_command(resolved, endpoint=endpoint))]
    else:
        assert job.config_dir is not None
        size = fanout_size(job.config_dir)
        # NESTED, AND THIS IS THE FIELD I GOT WRONG TWICE. `RunSpec` in the platform's own
        # `src/edullm_platform/cli/spec.py` declares `fanout: SpecFanOut | None`, and `SpecFanOut` is
        # `size` (ge=2) and `index_parameter`. The flat `fanout_size:` / `fanout_index_parameter:` spelling
        # belongs to `SubmissionInputs`, the record the CLI derives, and in this file it is refused with
        # "Extra inputs are not permitted".
        #
        # This repository's PRD 8.4 and factcrowd README both show the flat form here. Both are stale, and
        # believing two committed documents over the schema is what cost a refusal -- the platform checkout
        # sits at ../platform and settles it in one grep.
        lines += [
            "fanout:",
            f"  size: {size}",
            "  index_parameter: cell",
            "command: >-",
        ]
        lines += [
            f"  {part}"
            for part in _fold(train_command(job.config_dir, compute_profile=job.compute))
        ]
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 106) -> List[str]:
    """Wrap a comment line without importing textwrap for one call."""
    out: List[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def _fold(command: str, width: int = 104) -> List[str]:
    """
    Break a command across lines for YAML's folded scalar, which rejoins them with single spaces.

    Folded rather than literal because the command is one shell line; a literal block would keep the
    newlines and hand ``bash -lc`` a script whose first line is incomplete.
    """
    return _wrap(command, width)


def mapping(config_dir: str) -> str:
    """
    The fan-out index for every cell, printed so it can be checked rather than assumed.

    **The index maps by filename, and filenames sort as strings.** ``b16`` sorts before ``b4`` because
    ``"1" < "4"``, and ``113m`` before ``13m`` for the same reason -- the phase-1 calibration directory did
    exactly that. Any bijection is a correct submission, but a reader who assumes the indices ascend with
    demand will read the wrong cell's result, and a size taken from an older directory runs a different cell
    under the name the approval was granted for. So this prints what the platform will actually do.

    :param config_dir: Directory name under ``configs/cells``.

    :returns: Text for stdout.
    """
    target = CONFIG_ROOT / config_dir
    found = sorted(target.glob("*.yaml"))
    out = [f"fan-out index -> cell, as the platform will resolve it ({len(found)} cells):"]
    for index, path in enumerate(found):
        out.append(f"  {index:3d}  {path.stem}")
    out.append(f"  index {len(found)} and above are refused by train_cell.py.")
    return "\n".join(out)


def describe() -> str:
    """
    The plan as a table, with what is ready and what is waiting on what.

    :returns: Text for stdout.
    """
    rows = []
    for job in JOBS:
        if job.scoring:
            shape = "scoring"
        elif job.needs_lengths:
            shape = "needs lengths"
        else:
            try:
                shape = f"{fanout_size(job.config_dir or '')} cells"
            except OLMoConfigurationError:
                shape = "configs missing"
        blocked = f"after {', '.join(job.blocked_by)}" if job.blocked_by else "READY NOW"
        rows.append((job.name, shape, blocked, job.answers))
    width = max(len(r[0]) for r in rows)
    shape_width = max(len(r[1]) for r in rows)
    block_width = max(len(r[2]) for r in rows)
    out = [
        f"{'job'.ljust(width)}  {'shape'.ljust(shape_width)}  {'when'.ljust(block_width)}  answers",
        f"{'-' * width}  {'-' * shape_width}  {'-' * block_width}  {'-' * 7}",
    ]
    for name, shape, blocked, answers in rows:
        out.append(
            f"{name.ljust(width)}  {shape.ljust(shape_width)}  {blocked.ljust(block_width)}  {answers}"
        )
    out += [
        "",
        "Read cost and approval_class out of `edullm check --json`. This program quotes neither.",
    ]
    return "\n".join(out)


def generate_dependent_configs(
    job: Job, *, ctxmano_length: int, mano_length: int
) -> Tuple[int, Path]:
    """
    Write the configs for a job whose depth comes from calibration.

    :param job: The job.
    :param ctxmano_length: Depth selected for the in-context endpoint.
    :param mano_length: Depth selected for the memorised endpoint.

    :returns: How many cells were written, and where.

    :raises OLMoConfigurationError: If the job does not take lengths.
    """
    assert job.config_dir is not None
    target = CONFIG_ROOT / job.config_dir
    # Annotated, because a literal mixing str and int infers Dict[str, object] and every `**shared`
    # unpack then reads as passing an object where a str or an int goes.
    shared: Dict[str, Any] = dict(
        mano_variant="both",
        ctxmano_length=ctxmano_length,
        mano_length=mano_length,
        mano_pad_to=cells_module.MANO_PAD_TO,
        ctxmano_pad_to=cells_module.IN_CONTEXT_PAD_TO,
        phase="p2",
    )
    if job.name == "ladder":
        built = cells_module.dilution_ladder_cells("13M", iso_token=True, **shared)
    elif job.name == "entropy":
        built = cells_module.replicate_block(
            cells_module.entropy_sweep_cells(row="28M", **shared), 3
        )
    elif job.name == "count":
        built = cells_module.replicate_block(
            tuple(
                cells_module.CellSpec(
                    cell_id=f"28m_d{str(demand).replace('.', 'p')}",
                    row="28M",
                    sweep="count",
                    demand_bits_per_param=demand,
                    reasoning_tokens=cells_module.REASONING_TOKENS,
                    related_reasoning_tokens=(
                        0 if demand == 0 else cells_module.RELATED_REASONING_TOKENS
                    ),
                    **shared,
                )
                for demand in (0.0, 0.6, 2.4)
            ),
            3,
        )
    else:
        raise OLMoConfigurationError(f"job {job.name!r} does not take lengths")
    if target.exists():
        for stale in target.glob("*.yaml"):
            stale.unlink()
    written = cells_module.write_cells(built, target)
    return len(written), target


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point.

    :param argv: Arguments, for testing.

    :returns: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__ or "")
    sub = parser.add_subparsers(dest="verb", required=True)
    sub.add_parser("list", help="Print the plan and what is ready")
    stage = sub.add_parser("stage", help="Write .edullm/run.yaml for one job")
    stage.add_argument("job", choices=sorted(JOBS_BY_NAME))
    stage.add_argument(
        "--ctxmano-length", type=int, help="Depth calibration selected for <ctxmano>"
    )
    stage.add_argument("--mano-length", type=int, help="Depth calibration selected for <mano>")
    stage.add_argument(
        "--prefix",
        nargs="+",
        default=(),
        help="For score-calibration: finished run output prefixes",
    )
    stage.add_argument("--gate-endpoint", default="ctxmano", help="Endpoint a scoring job admits")
    stage.add_argument("--print", action="store_true", help="Write nothing; print the YAML")
    args = parser.parse_args(argv)

    if args.verb == "list":
        print(describe())
        return 0

    job = JOBS_BY_NAME[args.job]
    if job.needs_lengths:
        if args.ctxmano_length is None or args.mano_length is None:
            raise OLMoConfigurationError(
                f"job {job.name!r} runs at the depth calibration selected, so it needs both "
                f"--ctxmano-length and --mano-length. Defaulting them would mean choosing a depth before "
                f"measuring it, which is how eighteen cells came to be trained where the endpoint had no "
                f"dynamic range. Run the calibration job and score it first."
            )
        count, where = generate_dependent_configs(
            job, ctxmano_length=args.ctxmano_length, mano_length=args.mano_length
        )
        print(f"wrote {count} cell config(s) to {where}")
        print()
        print(mapping(job.config_dir or ""))
        print()

    text = render(job, prefixes=args.prefix, endpoint=args.gate_endpoint)
    if args.print:
        print(text, end="")
        return 0
    RUN_YAML.parent.mkdir(parents=True, exist_ok=True)
    RUN_YAML.write_text(text)
    print(f"wrote {RUN_YAML}")
    print()
    print("Next, in order:")
    print(
        "  1. git add -A && git commit  (the platform builds from the commit, not the working tree)"
    )
    print("  2. git push origin edullm/<branch>")
    print(
        "  3. edullm check --json --experiment <slug> --dataset none   # free, no network, lists refusals"
    )
    print("  4. read `cost` and `approval_class` from that output, then submit")
    if not job.scoring and job.config_dir:
        size = fanout_size(job.config_dir)
        print()
        print(
            f"Fan-out: {size} cells, written into run.yaml as a nested `fanout:` mapping -- which is"
        )
        print(
            "what RunSpec declares in the platform's cli/spec.py. The flat fanout_size spelling belongs"
        )
        print(
            "to SubmissionInputs and is refused in this file; PRD 8.4 and the README are stale on it."
        )
        print()
        print(mapping(job.config_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OLMoConfigurationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
