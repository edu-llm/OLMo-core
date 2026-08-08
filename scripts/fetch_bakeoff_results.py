#!/usr/bin/env python3
"""
Pull run 2's per-cell mixer-bakeoff results out of CloudWatch, programmatically, so no cell is
ever hand-transcribed the way 16 of run 1's 18 committed ``cell-*.json`` files were.

WHY THIS EXISTS. Run 1's adversarial audit (docs/mixer-bakeoff/HANDOFF.md, /tmp/council/E1.md)
found that the exact identity ``throughput_tok_s_steady_per_device * world_size ==
throughput_tok_s_steady`` -- which cannot fail in the source, because the per-device figure is
computed as ``steady_total / world_size`` (train_core6_arm.py:1756) and every world_size in this
project is a power of two, so multiplying back is bit-exact with no rounding -- failed in 13 of
18 committed files, by up to 111 tok/s. The corroborating tell: ``first_loss`` was exactly
``11.7124`` in 16 cells. That value does NOT round-trip through float32
(``struct.unpack('f', struct.pack('f', 11.7124))[0] != 11.7124``), and ``first_loss`` is read
from a loss tensor via ``.item()`` (LossWatcher), so a genuine reading is a float32 value
promoted to float64 and MUST round-trip cleanly. A value that fails to round-trip could not have
come from that tensor read -- it was typed by hand. This did not change run 1's conclusions
(val_ce, parameters and both memory fields were byte-exact in all 18 cells; the throughput
rounding was <=0.026% against arm gaps of 2-29%) but "that was luck, not design."

WHERE THE RESULTS LIVE. The training entrypoint (.edullm/train_core6_arm.py, function
``summarise()`` at line 1906, the ``print(json.dumps({...}, indent=2), flush=True)`` call at
lines ~1906-1990) prints one JSON object per cell on rank 0 only. It is printed with
``indent=2`` -- confirmed empirically against run 1's real CloudWatch stream
(gpu-8xa100-run/default/8a107943799d4b8d91802fceb8826586) on 2026-08-08, which is the OPPOSITE
of what a plain reading of the call site suggests (no explicit single-line assumption holds).
Docker's log driver splits stdout into one CloudWatch log event per newline, so the block is
NOT one event -- it is ~51 consecutive events, one per JSON line, starting with a bare ``{``
event and ending with a bare ``}`` event. Reassembling those lines in stream order, in the
correct span, is the actual hard part of this script; see ``extract_summary_blocks()``.

The results are NOT written to S3 (only checkpoints are, at ``checkpoint_uri``); CloudWatch is
the only channel.

HARD RULES THIS SCRIPT OBEYS.
  - Pure stdlib. No AWS SDK. Talks to AWS by shelling out to the real ``aws`` CLI as a
    subprocess (this script is meant to run on a machine that HAS AWS credentials, e.g. an
    engineer's shell or CI -- it does not run through this agent's own read-only MCP tool).
  - Read-only: ``_ALLOWED_AWS_CALLS`` is a hardcoded allowlist of exactly the
    (service, operation) pairs this script is permitted to invoke. Batch ``describe-jobs`` /
    ``list-jobs`` and CloudWatch Logs ``filter-log-events`` -- nothing that creates, submits,
    cancels or mutates anything. ``run_aws()`` refuses anything outside that list before ever
    building a command line, as a structural guard independent of how carefully the rest of the
    script is written.
  - Never reads stdin. A prior CLI in this exact repo (``scripts/analyse_bakeoff.py``) has an
    ``if not sys.stdin.isatty(): sys.stdin.read()`` path that blocks forever under an agent
    shell, where stdin is neither a TTY nor closed. This script takes no positional stdin input
    at all, and every subprocess call passes ``stdin=subprocess.DEVNULL`` so a stray
    credential/MFA prompt from the ``aws`` CLI cannot block on OUR stdin either. Safe to run
    with ``< /dev/null`` always; see the module-level ``EXAMPLES`` string below.
  - Byte-faithful output: a cell's ``cell-<i>.json`` is the literal text the process printed
    (CloudWatch line messages rejoined with ``\\n``, in order), not a Python object re-serialized
    through ``json.dumps()``. No key is dropped, renamed, or reformatted.

USAGE (never blocks on stdin; safe under an agent shell or in CI):

    python3 scripts/fetch_bakeoff_results.py \\
        --run-id run_019f.... \\
        --job-queue sbsandbox-intern-edullm-gpu-8xa100 \\
        --out-dir docs/mixer-bakeoff/run2 \\
        < /dev/null

    # Direct array-job-id path (skip run-id resolution), dry run, no files written:
    python3 scripts/fetch_bakeoff_results.py \\
        --array-job-id d568724d-8e89-4771-8caa-5fbce2dcc977 --cell-count 18 \\
        --out-dir /tmp/p6/inspect --dry-run < /dev/null

Exit codes (mirrors the ``0 clean / 1 hard errors / 2 usage-or-discovery error`` convention
already used by scripts/analyse_bakeoff.py in this repo):
  0 - every cell fetched, schema-complete, and every validation passed.
  1 - at least one cell was fetched but failed validation (ERROR-severity finding). Files for
      passing cells are still written; the manifest records exactly which cells and fields
      failed. Do not hand-fix a failed cell; find out why it failed.
  2 - could not even get to validation: AWS discovery failed, the array job does not exist,
      cell count could not be resolved, or a usage error.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ==================== Constants: the authoritative schema ====================
#
# Every key ``summarise()`` prints, transcribed from a real run-1 CloudWatch stream
# (log group /aws/batch/sbsandbox-intern-edullm-gpu, stream
# gpu-8xa100-run/default/8a107943799d4b8d91802fceb8826586, cell 0, fetched read-only via
# `aws logs get-log-events` on 2026-08-08) cross-checked against .edullm/train_core6_arm.py's
# summarise() dict literal (~lines 1906-1990) and throughput_report() (line 1706). 51 keys.
# PRESENCE is required; several of these are legitimately None (val_* when no held-out split
# was scored, throughput_tok_s_whole_run* when wall clock was unavailable, sliced_eval when no
# slice directories were passed) -- "None" is a real, intended value here, not a missing key.
REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "dataset_id",
        "dataset_version",
        "data_seed",
        "init_seed",
        "gpu",
        "torch",
        "cuda",
        "parameters",
        "steps",
        "first_loss",
        "last_loss",
        "seconds",
        "world_size",
        "throughput_tok_s_steady",
        "throughput_tok_s_steady_per_device",
        "throughput_tok_s_whole_run",
        "throughput_tok_s_whole_run_per_device",
        "throughput_tok_s_all_steps",
        "steps_measured",
        "steady_state_steps",
        "warmup_steps_excluded",
        "tokens_in_steady_window",
        "step_time_s_p50",
        "step_time_s_p90",
        "steady_window_seconds",
        "training_seconds_excluding_startup",
        "mfu_pct",
        "mfu_basis",
        "device_peak_bf16_flops",
        "flops_per_token",
        "peak_memory_gib",
        "peak_memory_reserved_gib",
        "peak_memory_source",
        "peak_memory_samples",
        "tps_device_avg",
        "tps_device_last",
        "tps_total_avg",
        "tps_naive_wall_clock",
        "checkpoint_uri",
        "wandb_project",
        "wandb_url",
        "arm",
        "tokens_trained",
        "val_ce",
        "val_tokens",
        "val_tokens_present",
        "val_tokens_declared",
        "val_nll_sum",
        "val_shards",
        "sliced_eval",
    }
)

# Fields whose value MUST be a float32 value promoted to float64 -- i.e. it came from a
# `.item()` read on a loss tensor (LossWatcher.first / LossWatcher.last) rather than from pure
# Python float arithmetic. A genuine reading always round-trips through float32 unchanged. A
# value that does NOT round-trip could not have come from that tensor and is the exact
# fingerprint that caught run 1's fabricated cells (first_loss == 11.7124 in 16/18 files).
# Every OTHER float in this schema (seconds, throughput_*, mfu_pct, val_ce, peak_memory_*, ...)
# is computed via plain Python float64 arithmetic over ints/floats -- for those, failing to
# round-trip through float32 is the NORMAL case and proves nothing either way, so this check is
# deliberately NOT applied to them.
FLOAT32_NATIVE_FIELDS = ("first_loss", "last_loss")

# Fields that are run-wide constants and must be byte-identical across every cell of one run.
# `steps`/`world_size`/`warmup_steps_excluded` are literals in .edullm/run-bakeoff.yaml (or its
# run-2 equivalent) shared by the whole array; `device_peak_bf16_flops` is a table lookup keyed
# only on `gpu`, so it is constant iff the hardware is (which it must be, for arms to be
# comparable); `dataset_id`/`dataset_version` fix the corpus; `torch`/`cuda` fix the image.
# Deliberately EXCLUDES `parameters`, which legitimately differs *between* arms (see
# IDENTICAL_WITHIN_ARM_KEYS) because solve_widths gives each mixer a different FFN width.
IDENTICAL_ACROSS_ALL_CELLS_KEYS = (
    "steps",
    "world_size",
    "warmup_steps_excluded",
    "device_peak_bf16_flops",
    "dataset_id",
    "dataset_version",
    "torch",
    "cuda",
    "gpu",
)

# Fields that must be identical across every seed of the SAME arm (same architecture -> same
# parameter count) but are expected to differ ACROSS arms.
IDENTICAL_WITHIN_ARM_KEYS = ("parameters",)

# Read-only AWS CLI surface this script is permitted to invoke, as (service, operation) pairs.
# `run_aws()` checks every call against this before building a command line. Nothing here
# creates, submits, cancels, tags or otherwise mutates a resource.
_ALLOWED_AWS_CALLS = frozenset(
    {
        ("batch", "list-jobs"),
        ("batch", "describe-jobs"),
        ("logs", "filter-log-events"),
    }
)

# ln(vocab) for the shared eduLLM tokenizer, same constant scripts/analyse_bakeoff.py uses
# (VOCAB_SIZE = 100352) for its FIRST_LOSS_BAND plausibility check. Inlined rather than
# imported: this script does not import analyse_bakeoff.py (out of scope; not owned here), and
# duplicating one integer constant is cheaper than coupling the two tools' import graphs.
_VOCAB_SIZE = 100352
_FIRST_LOSS_BAND: Tuple[float, float] = (
    __import__("math").log(_VOCAB_SIZE) - 0.5,
    __import__("math").log(_VOCAB_SIZE) + 0.5,
)


# ==================== Small data types ====================


@dataclasses.dataclass
class Finding:
    """One validation result. ERROR fails the run (exit 1); WARN is printed but does not."""

    severity: str  # "ERROR" | "WARN"
    cell: str  # "cell-3" or "cross-cell" for whole-run checks
    field: str
    message: str

    def line(self) -> str:
        return f"[{self.severity}] {self.cell}: {self.field}: {self.message}"


@dataclasses.dataclass
class LogEvent:
    timestamp_ms: int
    message: str


@dataclasses.dataclass
class SummaryBlock:
    """One successfully-parsed ``{...}`` JSON object found in a log stream."""

    start_event_index: int
    end_event_index: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    raw_text: str  # exact reconstructed text, byte-faithful to what was printed
    parsed: Dict[str, Any]


@dataclasses.dataclass
class ExtractionOutcome:
    """What came out of scanning one cell's log stream."""

    accepted_block: Optional[SummaryBlock]
    duplicate_blocks: List[SummaryBlock]  # any FURTHER blocks found after the accepted one
    rejected_false_starts: int  # `{`-lines that parsed but lacked required keys
    truncated: bool  # a `{` was seen but no valid JSON was ever completed
    error: Optional[str]  # human-readable reason accepted_block is None, if it is


@dataclasses.dataclass
class CellJob:
    """AWS Batch's view of one array-child job."""

    job_id: str  # "<array_job_id>:<index>"
    index: int
    status: str
    exit_code: Optional[int]
    log_stream: Optional[str]  # the LATEST attempt's stream (== last attempts[] entry)
    attempt_log_streams: List[str]  # one per Batch attempt, in order
    attempts: int  # len(attempt_log_streams); >1 means Batch itself retried this cell


# ==================== AWS access: subprocess wrapper around the real `aws` CLI ====================


class AwsCallRefused(RuntimeError):
    """Raised when code asks run_aws() to make a call outside _ALLOWED_AWS_CALLS."""


class AwsCallFailed(RuntimeError):
    """Raised when the `aws` CLI itself exits non-zero."""


def run_aws(service: str, operation: str, args: Sequence[str], *, timeout_s: float = 60.0) -> Any:
    """Invoke ``aws <service> <operation> <args...> --output json`` and return the parsed body.

    Refuses anything not in ``_ALLOWED_AWS_CALLS`` before touching subprocess at all -- a
    structural read-only guard that holds even if a future edit to this file gets careless.
    Always passes ``stdin=subprocess.DEVNULL``: an interactive prompt from the `aws` CLI (an
    expired SSO session, an MFA challenge) must fail loudly and immediately, never hang waiting
    on input this script never intends to supply.
    """
    if (service, operation) not in _ALLOWED_AWS_CALLS:
        raise AwsCallRefused(
            f"refusing to call `aws {service} {operation}`: not in the read-only allowlist "
            f"({sorted(_ALLOWED_AWS_CALLS)}). This script may only inspect Batch jobs and "
            f"CloudWatch logs; it must never submit, cancel or mutate anything."
        )
    cmd = ["aws", service, operation, *args, "--output", "json"]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AwsCallFailed(f"`{' '.join(cmd)}` timed out after {timeout_s}s") from exc
    if proc.returncode != 0:
        raise AwsCallFailed(
            f"`{' '.join(cmd)}` exited {proc.returncode}\nstderr:\n{proc.stderr}"
        )
    text = proc.stdout.strip()
    if not text:
        return None
    return json.loads(text)


def resolve_array_job(run_id: str, job_queue: str) -> Tuple[str, int]:
    """Map a platform run id (``run_019f...``) to its Batch array job id and declared size.

    Per the edullm-platform-runs skill: "the run id ... is the Batch job name" -- confirmed
    empirically on 2026-08-08 against run 1 (`aws batch describe-jobs --jobs
    d568724d-8e89-4771-8caa-5fbce2dcc977` returned ``jobName ==
    "run_019fe0f9-1bbd-702c-b141-6d58e128bda6"`` exactly), and the resolution direction (name ->
    id) via ``list-jobs --filters name=JOB_NAME,values=<run_id>`` was verified the same way.
    """
    body = run_aws(
        "batch",
        "list-jobs",
        ["--job-queue", job_queue, "--filters", f"name=JOB_NAME,values={run_id}"],
    )
    summaries = (body or {}).get("jobSummaryList", [])
    matches = [j for j in summaries if j.get("jobName") == run_id]
    if not matches:
        raise AwsCallFailed(
            f"no Batch job named {run_id!r} found on queue {job_queue!r}. Either the run "
            f"has not been submitted, the queue is wrong, or the run id was mistyped."
        )
    if len(matches) > 1:
        raise AwsCallFailed(
            f"{len(matches)} Batch jobs named {run_id!r} found on queue {job_queue!r} "
            f"(job ids: {[m.get('jobId') for m in matches]}). Job names should be unique; "
            f"resolve this by hand before trusting either one."
        )
    job = matches[0]
    array_props = job.get("arrayProperties") or {}
    size = array_props.get("size")
    if not size:
        raise AwsCallFailed(
            f"Batch job {job.get('jobId')} (name {run_id!r}) is not an array job "
            f"(arrayProperties.size={size!r}). A bake-off fan-out must be an array job."
        )
    return job["jobId"], int(size)


def describe_cell(array_job_id: str, index: int) -> CellJob:
    """Describe one array-child job: status, exit code, and every attempt's log stream.

    The per-attempt list matters on its own, independent of anything inside the logs: a cell
    with more than one Batch attempt has a different token order than its declared seed would
    otherwise imply (steps_after_warmup filters on step INDEX for exactly this reason -- see
    .edullm/train_core6_arm.py:822-826 -- but a full container-level Batch retry is a coarser
    event than that filter was built for), which would break the seed pairing run 2 depends on.
    """
    job_id = f"{array_job_id}:{index}"
    body = run_aws("batch", "describe-jobs", ["--jobs", job_id])
    jobs = (body or {}).get("jobs", [])
    if not jobs:
        raise AwsCallFailed(f"describe-jobs returned no job for {job_id!r}")
    job = jobs[0]
    attempts = job.get("attempts") or []
    attempt_streams = [
        a.get("container", {}).get("logStreamName")
        for a in attempts
        if a.get("container", {}).get("logStreamName")
    ]
    # Fall back to the top-level convenience field (== the latest attempt's stream) for a job
    # that has not recorded an `attempts` entry yet, e.g. one attempt still in flight.
    top_level_stream = job.get("container", {}).get("logStreamName")
    if not attempt_streams and top_level_stream:
        attempt_streams = [top_level_stream]
    return CellJob(
        job_id=job_id,
        index=index,
        status=job.get("status", "UNKNOWN"),
        exit_code=job.get("container", {}).get("exitCode"),
        log_stream=top_level_stream or (attempt_streams[-1] if attempt_streams else None),
        attempt_log_streams=attempt_streams,
        attempts=len(attempt_streams) if attempt_streams else (1 if top_level_stream else 0),
    )


def fetch_log_events(log_group: str, log_stream: str, *, page_limit: int = 10_000) -> List[LogEvent]:
    """Fetch an ENTIRE log stream's events in chronological order via `filter-log-events`.

    Chosen over `get-log-events` deliberately: `get-log-events`'s pagination contract ("you have
    reached the end of the stream when nextForwardToken equals the token you passed in") is easy
    to get wrong and silently truncate a stream one page early. `filter-log-events` returns
    events for a single named stream in ascending timestamp order and simply omits `nextToken`
    once exhausted, which is unambiguous to loop on.
    """
    events: List[LogEvent] = []
    next_token: Optional[str] = None
    pages = 0
    while True:
        args = ["--log-group-name", log_group, "--log-stream-names", log_stream]
        if next_token:
            args += ["--next-token", next_token]
        body = run_aws("logs", "filter-log-events", args)
        for e in (body or {}).get("events", []):
            events.append(LogEvent(timestamp_ms=e["timestamp"], message=e["message"]))
        next_token = (body or {}).get("nextToken")
        pages += 1
        if not next_token or pages >= page_limit:
            break
    return events


# ==================== Log parsing: pure functions, no AWS, fully unit-testable ====================


def extract_summary_blocks(events: Sequence[LogEvent], *, max_block_lines: int = 400) -> ExtractionOutcome:
    """Scan a stream's events for the ``summarise()`` JSON block(s).

    ALGORITHM. A block starts at an event whose message, stripped of surrounding whitespace, is
    exactly ``"{"`` -- the bare top-level open brace `json.dumps(..., indent=2)` always emits as
    its own line, with no rank/timestamp prefix (unlike the framework log lines around it, which
    are tab-delimited ``timestamp\\thost:rank\\tlogger:line\\tLEVEL\\tmessage``). From there,
    messages are appended one at a time (joined with ``\\n``, exactly reconstructing the
    original multi-line `print()` argument) and `json.loads()` is retried after every new line.

    This is correct rather than a brace-counter for a reason: `json.loads` understands strings,
    nesting and escaping, so a value that itself contains a brace character, or a nested object
    (`sliced_eval` is a dict when slice directories were passed), is handled for free. And it
    cannot succeed prematurely: every non-final line of a `json.dumps(..., indent=2)` dict
    literal ends in a trailing comma or an open value, which is syntactically invalid JSON on
    its own, so the ONLY line at which `json.loads` can start succeeding is the true final `}`.

    A completed parse that is missing REQUIRED_KEYS is treated as a false start (some unrelated
    JSON blob that happened to also start with a bare `{`) and scanning continues past it rather
    than aborting -- this is theoretical belt-and-suspenders, not a case observed in practice.

    Scanning does NOT stop at the first accepted block. It continues to the end of the stream so
    a SECOND valid block -- a retry that reprinted the whole summary into the same stream without
    Batch itself creating a new attempt -- is caught and reported, rather than the last one
    silently winning.
    """
    accepted: Optional[SummaryBlock] = None
    duplicates: List[SummaryBlock] = []
    false_starts = 0
    saw_any_start = False
    truncated_after_accept = False

    i = 0
    n = len(events)
    while i < n:
        if events[i].message.strip() != "{":
            i += 1
            continue
        saw_any_start = True
        start = i
        lines = [events[i].message]
        parsed: Optional[Dict[str, Any]] = None
        end = i
        j = i + 1
        while j < n and (j - start) < max_block_lines:
            lines.append(events[j].message)
            candidate_text = "\n".join(lines)
            try:
                obj = json.loads(candidate_text)
            except json.JSONDecodeError:
                j += 1
                continue
            # Valid JSON reached. Only a dict counts as a candidate summary block.
            if isinstance(obj, dict):
                parsed = obj
                end = j
            break
        if parsed is None:
            # Never became valid JSON within the stream/line cap: either genuinely truncated
            # (stream ended mid-block) or corrupted by interleaved noise that broke JSON syntax
            # for good. Either way this candidate start produced nothing usable.
            if accepted is None:
                return ExtractionOutcome(
                    accepted_block=None,
                    duplicate_blocks=[],
                    rejected_false_starts=false_starts,
                    truncated=True,
                    error=(
                        f"a `{{` block starting at event {start} "
                        f"(timestamp {events[start].timestamp_ms}) never completed as valid "
                        f"JSON within {max_block_lines} lines or before the stream ended -- "
                        f"truncated log, or noise corrupted the block."
                    ),
                )
            # Already have an accepted block; a broken second start does not itself invalidate
            # the first one, but it is suspicious enough to surface.
            truncated_after_accept = True
            i = n
            continue
        block = SummaryBlock(
            start_event_index=start,
            end_event_index=end,
            start_timestamp_ms=events[start].timestamp_ms,
            end_timestamp_ms=events[end].timestamp_ms,
            raw_text="\n".join(lines),
            parsed=parsed,
        )
        missing = REQUIRED_KEYS - set(parsed.keys())
        if missing:
            false_starts += 1
            i = end + 1
            continue
        if accepted is None:
            accepted = block
        else:
            duplicates.append(block)
        i = end + 1

    if accepted is None:
        if saw_any_start:
            return ExtractionOutcome(
                accepted_block=None,
                duplicate_blocks=[],
                rejected_false_starts=false_starts,
                truncated=False,
                error=(
                    f"found {false_starts} `{{`-starting JSON block(s) but none contained all "
                    f"{len(REQUIRED_KEYS)} required keys -- none of them is the summary block."
                ),
            )
        return ExtractionOutcome(
            accepted_block=None,
            duplicate_blocks=[],
            rejected_false_starts=0,
            truncated=False,
            error="no line matching a bare `{` was found in this stream at all.",
        )
    return ExtractionOutcome(
        accepted_block=accepted,
        duplicate_blocks=duplicates,
        rejected_false_starts=false_starts,
        truncated=truncated_after_accept,
        error=None,
    )


def float32_roundtrips(value: float) -> bool:
    """True iff ``value`` survives a float32 pack/unpack round trip unchanged.

    A value that originated as a float32 (or bf16, which promotes losslessly to float32) tensor
    read is ALWAYS exactly representable in float32 -- that is where it came from -- so packing
    it down and back up is a no-op. A value that was instead typed as a decimal literal by a
    human (e.g. "11.7124") almost never has this property, because an arbitrary base-10 decimal's
    nearest float64 essentially never also happens to be exactly representable in float32's
    narrower 23-bit mantissa. Failure to round-trip is therefore proof of non-float32-origin;
    success is consistent with (not proof of) genuine origin.
    """
    packed = struct.pack("f", value)
    return struct.unpack("f", packed)[0] == value


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ==================== Per-cell validation ====================


def validate_cell(cell_label: str, obj: Dict[str, Any]) -> List[Finding]:
    """Every identity/consistency check this script knows how to run on ONE cell's object.

    Findings are tiered by how the check was derived, stated in each message:
      - EXACT (bit-for-bit): identities read directly out of the source (power-of-two divisor
        multiplications, or the literal `model_flops_utilisation` formula) are asserted with
        `==`, no tolerance. A source-derived exact identity failing is not noise -- it is a
        rounded or fabricated number, by construction.
      - float32-native fields (first_loss, last_loss): asserted with `==` via float32_roundtrips,
        for the reason given on FLOAT32_NATIVE_FIELDS above.
      - Everything else that is only PLAUSIBLE (val_ce vs val_nll_sum/val_tokens_present, whose
        exact computation lives outside this file and was not read) is checked with a tight
        relative tolerance and reported as WARN, not ERROR, and the message says so explicitly.
    """
    findings: List[Finding] = []

    def err(field: str, message: str) -> None:
        findings.append(Finding("ERROR", cell_label, field, message))

    def warn(field: str, message: str) -> None:
        findings.append(Finding("WARN", cell_label, field, message))

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        err("<schema>", f"missing {len(missing)} required key(s): {sorted(missing)}")
        # Every other check below assumes the keys it touches exist; bail out rather than
        # raising a wall of KeyErrors on top of the one finding that actually matters.
        return findings

    world_size = obj.get("world_size")

    # --- EXACT: per-device throughput figures back-multiply to the total. ------------------
    # Source: train_core6_arm.py:1756 `steady_per_device = steady_total / divisor` and :1757-1759
    # the identical pattern for whole_run. Dividing then multiplying back by the SAME integer
    # divisor is bit-exact IEEE-754 whenever that divisor is a power of two (multiplying/dividing
    # by a power of two only shifts the exponent; no mantissa bits are touched, so nothing
    # rounds). Every provisioned GPU shape in this project (1/2/4/8 GPUs) is a power of two.
    for total_key, per_device_key in (
        ("throughput_tok_s_steady", "throughput_tok_s_steady_per_device"),
        ("throughput_tok_s_whole_run", "throughput_tok_s_whole_run_per_device"),
    ):
        total = obj.get(total_key)
        per_device = obj.get(per_device_key)
        if total is None or per_device is None:
            if (total is None) != (per_device is None):
                err(
                    total_key,
                    f"{total_key}={total!r} but {per_device_key}={per_device!r} -- these are "
                    f"computed together (train_core6_arm.py:1756-1759) and must be None/non-None "
                    f"together.",
                )
            continue
        if not isinstance(world_size, int) or not _is_power_of_two(world_size):
            warn(
                total_key,
                f"world_size={world_size!r} is not a power of two -- the "
                f"per_device*world_size==total identity is not guaranteed bit-exact here, "
                f"skipping the EXACT check (would need a tolerance-based fallback).",
            )
            continue
        recomputed = per_device * world_size
        if recomputed != total:
            err(
                total_key,
                f"EXACT identity violated: {per_device_key} ({per_device!r}) * world_size "
                f"({world_size}) = {recomputed!r}, but {total_key} = {total!r} "
                f"(residual {total - recomputed!r}). This is the exact defect that caught run "
                f"1's 13 corrupted cells.",
            )

    # --- EXACT: tps_total_avg == tps_device_avg * world_size. -------------------------------
    # Source: train_core6_arm.py summarise(), `"tps_total_avg": (None if tps_device_avg is None
    # else tps_device_avg * get_world_size())`. Same power-of-two exactness argument as above.
    tps_device_avg = obj.get("tps_device_avg")
    tps_total_avg = obj.get("tps_total_avg")
    if (tps_device_avg is None) != (tps_total_avg is None):
        err(
            "tps_total_avg",
            f"tps_device_avg={tps_device_avg!r} but tps_total_avg={tps_total_avg!r} -- must be "
            f"None/non-None together (summarise() computes one directly from the other).",
        )
    elif tps_device_avg is not None:
        if not isinstance(world_size, int) or not _is_power_of_two(world_size):
            warn("tps_total_avg", f"world_size={world_size!r} not a power of two; skipping EXACT check.")
        else:
            recomputed = tps_device_avg * world_size
            if recomputed != tps_total_avg:
                err(
                    "tps_total_avg",
                    f"EXACT identity violated: tps_device_avg ({tps_device_avg!r}) * world_size "
                    f"({world_size}) = {recomputed!r} != tps_total_avg ({tps_total_avg!r}).",
                )

    # --- EXACT: mfu_pct == 100.0 * (per_device * flops_per_token) / device_peak_bf16_flops. ---
    # Source: train_core6_arm.py:943 `model_flops_utilisation`, literal expression
    # `100.0 * (tokens_per_second_per_device * flops_per_token) / device_peak_flops`, called at
    # :1763 with tokens_per_second_per_device=steady_per_device. Reproducing the SAME operator
    # order (parenthesized multiply first, then the outer multiply-by-100, then divide) recovers
    # a bit-identical float64 result, since IEEE-754 arithmetic is deterministic per operation
    # order (it is `(100.0 * (a * b)) / c`, not `100.0 * a * b / c`, and those are not always the
    # same value).
    per_device = obj.get("throughput_tok_s_steady_per_device")
    flops_per_token = obj.get("flops_per_token")
    device_peak = obj.get("device_peak_bf16_flops")
    mfu_pct = obj.get("mfu_pct")
    inputs_present = bool(per_device) and bool(flops_per_token) and bool(device_peak)
    if inputs_present and mfu_pct is None:
        err(
            "mfu_pct",
            f"throughput_tok_s_steady_per_device={per_device!r}, flops_per_token="
            f"{flops_per_token!r} and device_peak_bf16_flops={device_peak!r} are all present, "
            f"so model_flops_utilisation() should have returned a number, not None.",
        )
    elif not inputs_present and mfu_pct is not None:
        err(
            "mfu_pct",
            f"mfu_pct={mfu_pct!r} is present but one of its three inputs is falsy "
            f"(per_device={per_device!r}, flops_per_token={flops_per_token!r}, "
            f"device_peak={device_peak!r}) -- model_flops_utilisation() returns None whenever "
            f"any input is missing/zero.",
        )
    elif inputs_present:
        recomputed = 100.0 * (per_device * flops_per_token) / device_peak
        if recomputed != mfu_pct:
            err(
                "mfu_pct",
                f"EXACT identity violated: 100.0 * ({per_device!r} * {flops_per_token!r}) / "
                f"{device_peak!r} = {recomputed!r} != mfu_pct ({mfu_pct!r}).",
            )

    # --- EXACT: tokens_trained == steps * global_batch_size (an integer, though not itself a
    # printed field). Source: `"tokens_trained": trainer.global_step * opts.global_batch_size`,
    # and `"steps": trainer.global_step` -- literally the same trainer.global_step value in both
    # expressions, so tokens_trained/steps recovers global_batch_size exactly, with ZERO
    # remainder, or the field is inconsistent with itself.
    steps = obj.get("steps")
    tokens_trained = obj.get("tokens_trained")
    if isinstance(steps, int) and steps > 0 and isinstance(tokens_trained, int):
        if tokens_trained % steps != 0:
            err(
                "tokens_trained",
                f"tokens_trained ({tokens_trained}) is not an exact multiple of steps ({steps}); "
                f"tokens_trained = steps * global_batch_size (train_core6_arm.py:1969) so the "
                f"remainder ({tokens_trained % steps}) must be zero.",
            )

    # --- EXACT: val_tokens_present == val_tokens_declared (task-specified cross-check; both are
    # populated together from the same `val` dict or both None -- see summarise():1977-1982). ---
    present = obj.get("val_tokens_present")
    declared = obj.get("val_tokens_declared")
    if (present is None) != (declared is None):
        err(
            "val_tokens_present",
            f"val_tokens_present={present!r} but val_tokens_declared={declared!r} -- both come "
            f"from the same `val` dict and must be None/non-None together.",
        )
    elif present is not None and present != declared:
        err(
            "val_tokens_present",
            f"val_tokens_present ({present}) != val_tokens_declared ({declared}). "
            f"summarise()'s own docstring: 'val_tokens_present is the count that was asserted "
            f"against the manifest -- if it is here at all, that assertion passed.' A mismatch "
            f"means the two numbers disagree about how much held-out data was actually scored.",
        )

    # --- WARN (near-exact, tolerance-based): val_ce vs val_nll_sum / val_tokens_present. This
    # relationship was NOT read out of the source in this repo (the evaluation code that fills
    # `val["ce"]` lives outside .edullm/train_core6_arm.py and was out of scope to chase down),
    # so it is checked as a plausibility sanity, not asserted as a proven bit-exact identity. ---
    val_ce = obj.get("val_ce")
    val_nll_sum = obj.get("val_nll_sum")
    if val_ce is not None and val_nll_sum is not None and present:
        recomputed_ce = val_nll_sum / present
        if val_ce != 0 and abs(recomputed_ce - val_ce) / abs(val_ce) > 1e-9:
            warn(
                "val_ce",
                f"val_nll_sum/val_tokens_present = {recomputed_ce!r} differs from val_ce "
                f"({val_ce!r}) by a relative {abs(recomputed_ce - val_ce) / abs(val_ce):.3e} -- "
                f"NOT asserted as bit-exact (the computation was not verified against source), "
                f"but worth a human look if this is not floating-point noise.",
            )

    # --- EXACT-by-provenance: first_loss / last_loss must round-trip through float32. ----------
    for field in FLOAT32_NATIVE_FIELDS:
        value = obj.get(field)
        if value is None:
            continue
        if not isinstance(value, float):
            continue
        if not float32_roundtrips(value):
            err(
                field,
                f"{field} = {value!r} does not round-trip through float32 "
                f"(struct.unpack('f', struct.pack('f', {value!r}))[0] != {value!r}). "
                f"{field} is read from a loss tensor via `.item()` (LossWatcher), so a genuine "
                f"reading is always exactly float32-representable. This is the exact fingerprint "
                f"that identified run 1's fabricated cells (first_loss == 11.7124 in 16/18 "
                f"files, which fails this same check).",
            )

    # --- WARN: first_loss plausibility band (ln(vocab) +/- 0.5), same band analyse_bakeoff.py
    # uses for FIRST_LOSS_BAND -- a sanity check on magnitude, not a correctness proof. ---------
    first_loss = obj.get("first_loss")
    if isinstance(first_loss, (int, float)) and not (_FIRST_LOSS_BAND[0] <= first_loss <= _FIRST_LOSS_BAND[1]):
        warn(
            "first_loss",
            f"first_loss ({first_loss!r}) is outside the plausible band "
            f"{_FIRST_LOSS_BAND} (ln({_VOCAB_SIZE}) +/- 0.5) for an untrained model's first "
            f"cross-entropy loss.",
        )

    return findings


def validate_cross_cell(cells: Dict[int, Dict[str, Any]]) -> List[Finding]:
    """Checks that only make sense with every cell's object in hand at once."""
    findings: List[Finding] = []
    if not cells:
        return findings

    for key in IDENTICAL_ACROSS_ALL_CELLS_KEYS:
        values = {idx: obj.get(key) for idx, obj in cells.items() if key in obj}
        distinct = set(values.values())
        if len(distinct) > 1:
            findings.append(
                Finding(
                    "ERROR",
                    "cross-cell",
                    key,
                    f"expected one value of {key!r} shared by every cell in this run "
                    f"(it is a run-wide constant), but found {len(distinct)} distinct "
                    f"values: {({v: [i for i, vv in values.items() if vv == v] for v in distinct})}.",
                )
            )

    by_arm: Dict[Any, Dict[int, Any]] = {}
    for idx, obj in cells.items():
        by_arm.setdefault(obj.get("arm"), {})[idx] = obj

    for key in IDENTICAL_WITHIN_ARM_KEYS:
        for arm, arm_cells in by_arm.items():
            values = {idx: obj.get(key) for idx, obj in arm_cells.items()}
            distinct = set(values.values())
            if len(distinct) > 1:
                findings.append(
                    Finding(
                        "ERROR",
                        "cross-cell",
                        key,
                        f"arm {arm!r}: expected one value of {key!r} shared by every seed of "
                        f"this arm, found {len(distinct)} distinct values across cells "
                        f"{sorted(values.keys())}: {values}.",
                    )
                )

    # Data seeds are supposed to repeat IDENTICALLY across arms (run-bakeoff.yaml: "The three
    # data seeds repeat across arms on purpose ... the same data seed gives every arm the
    # identical token stream"). Generalizes to however many seeds run 2 uses: every arm's SET of
    # data_seed values must equal every other arm's set.
    seed_sets = {arm: frozenset(o.get("data_seed") for o in c.values()) for arm, c in by_arm.items()}
    distinct_seed_sets = set(seed_sets.values())
    if len(distinct_seed_sets) > 1:
        findings.append(
            Finding(
                "ERROR",
                "cross-cell",
                "data_seed",
                f"arms do not all share the same set of data seeds, which breaks the paired "
                f"same-token-stream design the bake-off depends on: {seed_sets}.",
            )
        )

    return findings


# ==================== File output + manifest ====================


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_cell_file(out_dir: Path, index: int, raw_text: str, *, force: bool) -> Tuple[Path, str, bool]:
    """Write cell-<index>.json byte-faithfully. Returns (path, sha256, was_written).

    Idempotent: if the file already exists with IDENTICAL content, this is a silent no-op
    (was_written=False) so re-running the fetch is cheap and detectable. If it exists with
    DIFFERENT content, refuse unless --force -- a changed stream (e.g. because Batch retried the
    cell between two fetches) must be a loud, deliberate decision, not an overwrite nobody
    noticed.
    """
    path = out_dir / f"cell-{index}.json"
    new_checksum = sha256_of(raw_text)
    if path.exists():
        existing = path.read_text()
        existing_checksum = sha256_of(existing)
        if existing_checksum == new_checksum:
            return path, new_checksum, False
        if not force:
            raise RuntimeError(
                f"{path} already exists with DIFFERENT content (sha256 {existing_checksum} != "
                f"{new_checksum}). Refusing to overwrite without --force. If the underlying "
                f"stream changed (e.g. a Batch retry happened between two fetches), that is "
                f"itself worth understanding before clobbering the earlier file."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text)
    return path, new_checksum, True


# ==================== CLI orchestration ====================


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-id", help="Platform run id, e.g. run_019f.... Used to resolve the Batch array job id via `batch list-jobs` unless --array-job-id is given directly.")
    p.add_argument("--array-job-id", help="Batch array job UUID. Bypasses --run-id resolution entirely.")
    p.add_argument("--job-queue", default="sbsandbox-intern-edullm-gpu-8xa100", help="Batch job queue name to resolve --run-id against (ignored if --array-job-id is given).")
    p.add_argument("--cell-count", type=int, default=None, help="Number of array children to fetch. Default: read from the array job's own arrayProperties.size.")
    p.add_argument("--log-group", default="/aws/batch/sbsandbox-intern-edullm-gpu", help="CloudWatch log group the workload writes to.")
    p.add_argument("--out-dir", required=True, type=Path, help="Directory to write cell-<i>.json and manifest.json into.")
    p.add_argument("--allow-retries", action="store_true", help="Do not fail cells whose Batch `attempts` count is >1 (default: fail loudly, since a retried cell may have a different token order than its declared seed implies).")
    p.add_argument("--force", action="store_true", help="Overwrite an existing cell-<i>.json whose content differs from what was just fetched.")
    p.add_argument("--dry-run", action="store_true", help="Fetch and validate but do not write any files (manifest is still printed to stdout, not written).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.array_job_id and not args.run_id:
        print("error: one of --run-id or --array-job-id is required", file=sys.stderr)
        return 2

    try:
        if args.array_job_id:
            array_job_id = args.array_job_id
            declared_size = args.cell_count
            if declared_size is None:
                # Ask Batch for the size rather than guessing.
                body = run_aws("batch", "describe-jobs", ["--jobs", array_job_id])
                jobs = (body or {}).get("jobs", [])
                if not jobs:
                    print(f"error: no Batch job found for array-job-id {array_job_id!r}", file=sys.stderr)
                    return 2
                declared_size = (jobs[0].get("arrayProperties") or {}).get("size")
                if not declared_size:
                    print(f"error: {array_job_id!r} has no arrayProperties.size and --cell-count was not given", file=sys.stderr)
                    return 2
        else:
            array_job_id, declared_size = resolve_array_job(args.run_id, args.job_queue)
    except (AwsCallFailed, AwsCallRefused) as exc:
        print(f"error resolving array job: {exc}", file=sys.stderr)
        return 2

    cell_count = args.cell_count or declared_size
    print(f"array job {array_job_id}: fetching {cell_count} cell(s) (declared size {declared_size})")

    manifest_cells: List[Dict[str, Any]] = []
    parsed_by_index: Dict[int, Dict[str, Any]] = {}
    all_findings: List[Finding] = []

    for index in range(cell_count):
        cell_label = f"cell-{index}"
        try:
            job = describe_cell(array_job_id, index)
        except (AwsCallFailed, AwsCallRefused) as exc:
            print(f"error describing {cell_label}: {exc}", file=sys.stderr)
            all_findings.append(Finding("ERROR", cell_label, "<batch>", str(exc)))
            continue

        if not job.log_stream:
            all_findings.append(Finding("ERROR", cell_label, "<batch>", f"job {job.job_id} has no log stream (status={job.status})"))
            continue

        if job.attempts > 1 and not args.allow_retries:
            all_findings.append(
                Finding(
                    "ERROR",
                    cell_label,
                    "<batch attempts>",
                    f"Batch recorded {job.attempts} attempts for {job.job_id} "
                    f"(streams: {job.attempt_log_streams}). A retried cell may have a different "
                    f"token order than its declared seed implies, which would break run 2's "
                    f"seed pairing. Pass --allow-retries to fetch it anyway.",
                )
            )
            continue

        try:
            events = fetch_log_events(args.log_group, job.log_stream)
        except (AwsCallFailed, AwsCallRefused) as exc:
            all_findings.append(Finding("ERROR", cell_label, "<logs>", str(exc)))
            continue

        outcome = extract_summary_blocks(events)
        if outcome.accepted_block is None:
            all_findings.append(Finding("ERROR", cell_label, "<extraction>", outcome.error or "unknown extraction failure"))
            continue
        if outcome.duplicate_blocks:
            all_findings.append(
                Finding(
                    "ERROR",
                    cell_label,
                    "<extraction>",
                    f"found {len(outcome.duplicate_blocks)} additional valid summary block(s) "
                    f"in this stream after the first (at event indices "
                    f"{[b.start_event_index for b in outcome.duplicate_blocks]}, timestamps "
                    f"{[b.start_timestamp_ms for b in outcome.duplicate_blocks]}) -- a retry "
                    f"reprinted the summary into the same stream. Refusing to silently pick one; "
                    f"resolve by hand which block is authoritative.",
                )
            )
            continue

        block = outcome.accepted_block
        parsed_by_index[index] = block.parsed
        cell_findings = validate_cell(cell_label, block.parsed)
        all_findings.extend(cell_findings)

        checksum = sha256_of(block.raw_text)
        written = False
        path_str = str((args.out_dir / f"cell-{index}.json"))
        if not args.dry_run:
            path, checksum, written = write_cell_file(args.out_dir, index, block.raw_text, force=args.force)
            path_str = str(path)

        manifest_cells.append(
            {
                "index": index,
                "arm": block.parsed.get("arm"),
                "data_seed": block.parsed.get("data_seed"),
                "init_seed": block.parsed.get("init_seed"),
                "batch_job_id": job.job_id,
                "batch_status": job.status,
                "batch_exit_code": job.exit_code,
                "batch_attempts": job.attempts,
                "log_group": args.log_group,
                "log_stream": job.log_stream,
                "block_start_timestamp_ms": block.start_timestamp_ms,
                "block_end_timestamp_ms": block.end_timestamp_ms,
                "file_path": path_str,
                "sha256": checksum,
                "newly_written": written,
                "validation": {
                    "status": "ok" if not any(f.severity == "ERROR" for f in cell_findings) else "failed",
                    "findings": [f.line() for f in cell_findings],
                },
            }
        )
        for f in cell_findings:
            print(f.line())

    cross_findings = validate_cross_cell(parsed_by_index)
    all_findings.extend(cross_findings)
    for f in cross_findings:
        print(f.line())

    manifest = {
        "schema_version": 1,
        "generated_at_utc": _utcnow_iso(),
        "run_id": args.run_id,
        "array_job_id": array_job_id,
        "job_queue": args.job_queue if not args.array_job_id else None,
        "log_group": args.log_group,
        "declared_cell_count": cell_count,
        "cells": manifest_cells,
        "cross_cell_validation": {
            "status": "ok" if not any(f.severity == "ERROR" for f in cross_findings) else "failed",
            "findings": [f.line() for f in cross_findings],
        },
    }

    if args.dry_run:
        print("--dry-run: not writing manifest.json or any cell files")
        print(json.dumps(manifest, indent=2))
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {manifest_path}")

    n_errors = sum(1 for f in all_findings if f.severity == "ERROR")
    n_warns = sum(1 for f in all_findings if f.severity == "WARN")
    n_ok_cells = sum(1 for c in manifest_cells if c["validation"]["status"] == "ok")
    print(
        f"\n{n_ok_cells}/{cell_count} cells fetched and validated clean; "
        f"{n_errors} error(s), {n_warns} warning(s) total."
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
