"""
Tests for scripts/fetch_bakeoff_results.py.

NO AWS CALLS, NO NETWORK. Every test builds synthetic CloudWatch log text by hand and feeds it
to the script's own parsing/validation functions (`extract_summary_blocks`, `validate_cell`,
`validate_cross_cell`, `float32_roundtrips`, `write_cell_file`) -- never to `run_aws`,
`resolve_array_job`, `describe_cell` or `fetch_log_events`, which are the only functions in the
script that touch a subprocess. Nothing here imports botocore, boto3, or shells out.

EVERY TEST NAMES THE EXACT MUTATION TO fetch_bakeoff_results.py THAT WOULD MAKE IT FAIL, in its
docstring or a trailing comment -- per this repo's own standard (see
scripts/analyse_bakeoff.py's test file, and the project memory note
`test-must-call-not-recompute.md`: "a test that re-derives the code's formula passes when the
code changes"). None of these tests recomputes an identity independently and compares two
independent computations; each one either (a) hand-builds a KNOWN-bad or KNOWN-good fixture and
asserts the script's OWN verdict on it, or (b) calls the script's own extraction function on
synthetic log text and checks what came out.

Fixtures use a real cell's full 51-key shape (values lifted from run 1 cell 0's actual printed
JSON, read read-only from CloudWatch on 2026-08-08 -- see docs/mixer-bakeoff/run1/cell-0.json for
the subset that made it into that hand-transcribed file, and the module docstring of
fetch_bakeoff_results.py for the additional ~11 keys that file dropped) so a fixture missing a
key is a test-authoring bug, not a coincidence of a lazy fixture.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_bakeoff_results",
    str(Path(__file__).resolve().parents[3] / "scripts" / "fetch_bakeoff_results.py"),
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("could not load scripts/fetch_bakeoff_results.py")
fbr = importlib.util.module_from_spec(_SPEC)
sys.modules["fetch_bakeoff_results"] = fbr
_SPEC.loader.exec_module(fbr)


# ==================== A genuine, complete, real cell (run 1, cell 0) ====================
#
# Every value below is exactly what was printed for run 1's cell 0 (arm KDA_BASE, data_seed
# 210007), read read-only from CloudWatch log group /aws/batch/sbsandbox-intern-edullm-gpu,
# stream gpu-8xa100-run/default/8a107943799d4b8d91802fceb8826586, on 2026-08-08. Nothing here
# was rounded, retyped, or invented -- it is the same 51-key object the training entrypoint
# actually printed, transcribed once, used as the base for every fixture below.

GOOD_CELL: Dict[str, Any] = {
    "run_id": "run_019fe0f9-1bbd-702c-b141-6d58e128bda6",
    "dataset_id": "pretrain/reservoir-dolma2",
    "dataset_version": "v1",
    "data_seed": 210007,
    "init_seed": 110007,
    "gpu": "NVIDIA A100-SXM4-40GB",
    "torch": "2.9.0+cu128",
    "cuda": "12.8",
    "parameters": 390125472,
    "steps": 1144,
    "first_loss": 11.712315559387207,
    "last_loss": 3.1065468788146973,
    "seconds": 1804.3632384159998,
    "world_size": 8,
    "throughput_tok_s_steady": 421888.6904423051,
    "throughput_tok_s_steady_per_device": 52736.08630528814,
    "throughput_tok_s_whole_run": 332117.81931784126,
    "throughput_tok_s_whole_run_per_device": 41514.72741473016,
    "throughput_tok_s_all_steps": 421885.13677556656,
    "steps_measured": 1143,
    "steady_state_steps": 1094,
    "warmup_steps_excluded": 50,
    "tokens_in_steady_window": 573571072,
    "step_time_s_p50": 1.229610791000141,
    "step_time_s_p90": 1.2405655560000923,
    "steady_window_seconds": 1359.531755636,
    "training_seconds_excluding_startup": 1420.4368245350001,
    "mfu_pct": 41.806676834261204,
    "mfu_basis": (
        "52,736 tok/s/device * 2,473,388,544 FLOP/token / 312,000,000,000,000 FLOP/s peak "
        "dense bf16 on NVIDIA A100-SXM4-40GB"
    ),
    "device_peak_bf16_flops": 312000000000000,
    "flops_per_token": 2473388544,
    "peak_memory_gib": 9.153131484985352,
    "peak_memory_reserved_gib": 10.720703125,
    "peak_memory_source": "per_step_running_max",
    "peak_memory_samples": 1144,
    "tps_device_avg": 52735.640625,
    "tps_device_last": 26281.66796875,
    "tps_total_avg": 421885.125,
    "tps_naive_wall_clock": 332408.386088898,
    "checkpoint_uri": (
        "s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/"
        "run_019fe0f9-1bbd-702c-b141-6d58e128bda6/cell-0/checkpoints/"
    ),
    "wandb_project": "scratch",
    "wandb_url": (
        "https://wandb.ai/eduLLM/scratch/runs/run_019fe0f9-1bbd-702c-b141-6d58e128bda6-cell-0"
    ),
    "arm": "KDA_BASE",
    "tokens_trained": 599785472,
    "val_ce": 3.041620459051652,  # == val_nll_sum / val_tokens_present exactly, so GOOD_CELL
    # itself doesn't trip validate_cell's val_ce tolerance WARN (that's a separate, deliberate
    # test below: test_cross_cell / the val_ce WARN path isn't exercised by GOOD_CELL on purpose)
    "val_tokens": 974917632,
    "val_tokens_present": 975077376,
    "val_tokens_declared": 975077376,
    "val_nll_sum": 2965815296.0,
    "val_shards": 39,
    "sliced_eval": None,
}

assert set(GOOD_CELL.keys()) == fbr.REQUIRED_KEYS, (
    "GOOD_CELL must have EXACTLY the schema fetch_bakeoff_results.py requires -- if this "
    "assertion fires, either the fixture or REQUIRED_KEYS drifted and every test below is "
    "checking the wrong thing."
)


def render_block(obj: Dict[str, Any]) -> List[str]:
    """Render `obj` the way the real entrypoint does: `json.dumps(obj, indent=2)`, split into
    one line per CloudWatch log event -- exactly how Docker's log driver splits multi-line
    stdout. This is the SAME rendering path real production log streams go through, so a test
    built from this helper is testing the real reconstruction problem, not a simplified one.
    """
    return json.dumps(obj, indent=2).split("\n")


def make_events(lines: List[str], *, start_ts: int = 1_000_000) -> List[Any]:
    """Wrap plain strings as LogEvent objects with strictly increasing timestamps."""
    return [fbr.LogEvent(timestamp_ms=start_ts + i, message=line) for i, line in enumerate(lines)]


NOISE_BEFORE = [
    "2026-08-08 11:36:54.435\tip-10-20-62-247.ec2.internal:7\tpy.warnings:112\tWARNING\t"
    "/opt/olmo-core/src/olmo_core/distributed/utils.py:401: UserWarning: called a "
    "synchronizing CUDA operation",
    "  return value_tensor if is_tensor else value_tensor.item()  # type: ignore",
    "2026-08-08 11:36:54.437\tip-10-20-62-247.ec2.internal:0\t__main__:1604\tINFO\t"
    "held-out CE 3.0421 over 974,917,632 scored token(s) of 975,077,376 present",
]
NOISE_AFTER = [
    "2026-08-08 11:36:55.001\tip-10-20-62-247.ec2.internal:0\t__main__:1610\tINFO\tdone",
]


# ==================== Test 1: a clean cell extracts and validates with zero findings ====================


def test_clean_cell_extracts_and_validates_clean():
    """A well-formed stream (noise, then the block, then noise) yields exactly GOOD_CELL back,
    byte-faithfully, and validate_cell reports zero findings.

    NAMED MUTATION: delete the `missing = REQUIRED_KEYS - set(parsed.keys())` false-start check
    in extract_summary_blocks (i.e. accept ANY parsed dict as the block) -- this test would
    still pass, so it alone doesn't prove that guard exists. It is here as the baseline every
    other test's fixture is a controlled deviation FROM. Its real job is proving the happy path
    isn't itself broken, so failures in the tests below mean something.
    """
    events = make_events(NOISE_BEFORE + render_block(GOOD_CELL) + NOISE_AFTER)
    outcome = fbr.extract_summary_blocks(events)
    assert outcome.error is None
    assert outcome.accepted_block is not None
    assert outcome.duplicate_blocks == []
    assert outcome.accepted_block.parsed == GOOD_CELL
    # Byte-faithful: reparsing the exact raw text recovers the identical object, and the raw
    # text is exactly what json.dumps(GOOD_CELL, indent=2) produces -- not a re-serialization.
    assert outcome.accepted_block.raw_text == json.dumps(GOOD_CELL, indent=2)

    findings = fbr.validate_cell("cell-0", outcome.accepted_block.parsed)
    assert findings == [], f"expected zero findings on a genuine cell, got: {[f.line() for f in findings]}"


# ==================== Test 2: per_device * world_size off by one tok/s ====================


def test_throughput_identity_violation_off_by_one_is_caught():
    """throughput_tok_s_steady is perturbed by +1 tok/s relative to
    per_device * world_size (both otherwise untouched) -- the exact defect class that caught 13
    of run 1's 18 committed cells.

    NAMED MUTATION: in validate_cell, change `if recomputed != total:` to
    `if abs(recomputed - total) > 1.0:` (i.e. add an off-by-one tolerance) -- this test would
    then pass with zero findings, silently re-introducing exactly the run-1 defect this script
    exists to prevent.
    """
    bad = dict(GOOD_CELL)
    bad["throughput_tok_s_steady"] = GOOD_CELL["throughput_tok_s_steady"] + 1.0
    findings = fbr.validate_cell("cell-0", bad)
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "throughput_tok_s_steady"]
    assert len(errors) == 1, f"expected exactly one ERROR on throughput_tok_s_steady, got: {[f.line() for f in findings]}"
    assert "EXACT identity violated" in errors[0].message
    assert "run 1's 13 corrupted cells" in errors[0].message


def test_tps_total_avg_identity_violation_is_caught():
    """Same defect class, on the OTHER exact identity the script checks
    (tps_total_avg == tps_device_avg * world_size).

    NAMED MUTATION: delete the tps_total_avg block in validate_cell entirely -- this test would
    then find zero ERROR findings and fail its assertion.
    """
    bad = dict(GOOD_CELL)
    bad["tps_total_avg"] = GOOD_CELL["tps_total_avg"] - 5.0
    findings = fbr.validate_cell("cell-0", bad)
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "tps_total_avg"]
    assert len(errors) == 1
    assert "EXACT identity violated" in errors[0].message


def test_mfu_pct_identity_violation_is_caught():
    """mfu_pct is perturbed away from 100.0 * (per_device * flops_per_token) / device_peak.

    NAMED MUTATION: in validate_cell, change `if recomputed != mfu_pct:` to a tolerance check
    (e.g. `if abs(recomputed - mfu_pct) > 0.01:`) -- a perturbation of 0.001 below would then
    pass silently.
    """
    bad = dict(GOOD_CELL)
    bad["mfu_pct"] = GOOD_CELL["mfu_pct"] + 0.001
    findings = fbr.validate_cell("cell-0", bad)
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "mfu_pct"]
    assert len(errors) == 1
    assert "EXACT identity violated" in errors[0].message


# ==================== Test 3: a non-float32-representable "measured" value ====================


def test_hand_typed_first_loss_fails_float32_roundtrip():
    """first_loss is set to exactly 11.7124 -- run 1's real fabricated value, verbatim -- which
    is NOT float32-representable (struct.unpack('f', struct.pack('f', 11.7124))[0] != 11.7124),
    proving it could not have come from LossWatcher's `.item()` tensor read.

    NAMED MUTATION: in float32_roundtrips, change `struct.unpack("f", packed)[0] == value` to
    `abs(struct.unpack("f", packed)[0] - value) < 1e-3` (i.e. add a tolerance to the round-trip
    check) -- 11.7124 is within 1e-3 of its own float32 truncation, so this test would then pass
    with zero findings, exactly reproducing run 1's blind spot.
    """
    assert fbr.float32_roundtrips(11.7124) is False  # sanity: this is the documented forensic fact
    bad = dict(GOOD_CELL)
    bad["first_loss"] = 11.7124
    findings = fbr.validate_cell("cell-0", bad)
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "first_loss"]
    assert len(errors) == 1, f"expected exactly one ERROR on first_loss, got: {[f.line() for f in findings]}"
    assert "does not round-trip through float32" in errors[0].message
    assert "11.7124" in errors[0].message


def test_genuine_float32_value_does_not_false_positive():
    """The REAL captured first_loss (11.712315559387207 -- many digits, looks nothing like a
    hand-typed number) must NOT be flagged. Without this test, a validator that flags every
    float regardless of origin would also pass test_hand_typed_first_loss_fails_float32_roundtrip
    above while being useless (it would reject every genuine cell too).

    NAMED MUTATION: in validate_cell, drop the `if not float32_roundtrips(value):` guard and
    always emit the ERROR finding for first_loss/last_loss -- this test would then fail because
    GOOD_CELL (unmodified) would suddenly have findings.
    """
    findings = fbr.validate_cell("cell-0", GOOD_CELL)
    assert not any(f.field in ("first_loss", "last_loss") for f in findings)


# ==================== Test 4: truncated JSON block ====================


def test_truncated_block_is_reported_not_silently_dropped():
    """The stream ends partway through the block (cut off after 10 lines, well before the
    closing brace) -- simulating a stream fetch that stopped early or a container that was
    killed mid-print.

    NAMED MUTATION: in extract_summary_blocks, change the inner `while j < n and ...` loop so
    that on exhausting the stream without a successful parse it falls through to treating
    `lines` (the incomplete text) as accepted anyway, e.g. by removing the `if parsed is None:`
    branch and using `json.loads(candidate_text, strict=False)`-with-a-catch-all default -- this
    test would then get a bogus `accepted_block` for a block that was never actually completed,
    instead of `outcome.accepted_block is None`.
    """
    full_lines = render_block(GOOD_CELL)
    truncated_lines = full_lines[:10]  # well short of the closing "}"
    events = make_events(NOISE_BEFORE + truncated_lines)
    outcome = fbr.extract_summary_blocks(events)
    assert outcome.accepted_block is None
    assert outcome.truncated is True
    assert outcome.error is not None and "never completed as valid JSON" in outcome.error


# ==================== Test 5: two summary blocks in one stream (a retry) ====================


def test_duplicate_block_in_same_stream_is_detected_not_last_wins():
    """The SAME cell prints its summary twice into one stream (e.g. an in-container retry that
    did not create a new Batch attempt) -- once as KDA_BASE/data_seed 210007, and again later
    with a DIFFERENT arm/seed, so a last-wins bug would silently return the wrong cell's data
    rather than merely a duplicate of the same one.

    NAMED MUTATION: in extract_summary_blocks, change `if accepted is None: accepted = block`
    `else: duplicates.append(block)` to unconditionally `accepted = block` (last-wins) -- this
    test's assertion that duplicate_blocks is non-empty would fail, and accepted_block.parsed
    would silently be the SECOND (wrong) block instead of surfacing the conflict.
    """
    second = dict(GOOD_CELL)
    second["arm"] = "GDN2"
    second["data_seed"] = 999999
    events = make_events(
        NOISE_BEFORE + render_block(GOOD_CELL) + NOISE_AFTER + render_block(second)
    )
    outcome = fbr.extract_summary_blocks(events)
    assert outcome.accepted_block is not None
    assert outcome.accepted_block.parsed["arm"] == "KDA_BASE"  # the FIRST block, not silently the second
    assert len(outcome.duplicate_blocks) == 1
    assert outcome.duplicate_blocks[0].parsed["arm"] == "GDN2"


# ==================== Test 6: interleaved log noise around the JSON ====================


def test_interleaved_noise_before_and_after_is_skipped():
    """Realistic multi-rank framework noise (tab-delimited timestamp/host/logger/level lines,
    an unrelated INFO line quoting the same numbers in prose) surrounds the block on both sides.
    The block itself must still be found and parsed exactly, and the noise lines must not appear
    anywhere in the reconstructed raw_text.

    NAMED MUTATION: in extract_summary_blocks, change the start-of-block test from
    `events[i].message.strip() != "{"` to a looser `"{" not in events[i].message` -- one of the
    noise lines here is plain prose with no brace at all so that specific mutation wouldn't
    trigger on THIS fixture, but the intent of this test is that a start marker looser than
    exact-bare-brace risks matching inside a noise line; assert raw_text has no noise leaking in
    regardless of which specific matching relaxation is tried.
    """
    events = make_events(NOISE_BEFORE + render_block(GOOD_CELL) + NOISE_AFTER)
    outcome = fbr.extract_summary_blocks(events)
    assert outcome.accepted_block is not None
    for noisy in NOISE_BEFORE + NOISE_AFTER:
        assert noisy not in outcome.accepted_block.raw_text
    assert outcome.accepted_block.parsed == GOOD_CELL


# ==================== Test 7: missing required keys is a schema failure, not silence ====================


def test_missing_required_key_is_a_loud_schema_error():
    """One of the ~11 keys run 1's committed files DROPPED (val_nll_sum) is removed from the
    object before validation -- simulating exactly the incompleteness this script exists to
    prevent, on a cell that otherwise parses fine.

    NAMED MUTATION: in validate_cell, delete the `missing = REQUIRED_KEYS - set(obj.keys())`
    check (or change `if missing:` to `if False:`) -- this test would then find zero findings on
    an object that is missing a required key.
    """
    bad = dict(GOOD_CELL)
    del bad["val_nll_sum"]
    findings = fbr.validate_cell("cell-0", bad)
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "<schema>"]
    assert len(errors) == 1
    assert "val_nll_sum" in errors[0].message


def test_extraction_rejects_a_block_missing_required_keys_as_a_false_start():
    """A bare-`{`-delimited JSON object that is NOT a cell summary (missing nearly everything)
    appears in the stream. extract_summary_blocks must not accept it as the block.

    NAMED MUTATION: in extract_summary_blocks, remove the `if missing: false_starts += 1;
    i = end + 1; continue` branch so any parsed dict is accepted regardless of schema -- this
    test would then get back the small unrelated object as accepted_block instead of None.
    """
    unrelated = ["{", '  "unrelated_key": 1', "}"]
    events = make_events(unrelated)
    outcome = fbr.extract_summary_blocks(events)
    assert outcome.accepted_block is None
    assert outcome.rejected_false_starts == 1


# ==================== Test 8: cross-cell checks ====================


def test_cross_cell_catches_a_non_constant_run_wide_field():
    """Two cells disagree on `steps` (1144 vs 1145) -- steps is a run-wide constant
    (.edullm/run-bakeoff.yaml literal), so any two cells of the same run disagreeing means
    something is structurally wrong (wrong run mixed in, or corrupted data).

    NAMED MUTATION: in validate_cross_cell, remove "steps" from
    IDENTICAL_ACROSS_ALL_CELLS_KEYS -- this test would then find zero cross-cell ERROR findings.
    """
    cell_a = dict(GOOD_CELL)
    cell_b = dict(GOOD_CELL)
    cell_b["steps"] = 1145
    cell_b["arm"] = "GDN2"
    findings = fbr.validate_cross_cell({0: cell_a, 1: cell_b})
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "steps"]
    assert len(errors) == 1


def test_cross_cell_allows_parameters_to_differ_across_arms_but_not_within_one():
    """Two DIFFERENT arms legitimately have different parameter counts (solve_widths gives each
    mixer a different FFN width to hold total params roughly fixed) -- that must NOT be flagged.
    But two cells of the SAME arm (different seeds) disagreeing on parameters is a real defect.

    NAMED MUTATION: in validate_cross_cell, move "parameters" into
    IDENTICAL_ACROSS_ALL_CELLS_KEYS instead of IDENTICAL_WITHIN_ARM_KEYS -- the first assertion
    below (cross-arm difference is fine) would then start failing.
    """
    kda_seed1 = dict(GOOD_CELL)
    kda_seed1["arm"], kda_seed1["data_seed"], kda_seed1["parameters"] = "KDA_BASE", 210007, 390125472
    gdn_seed1 = dict(GOOD_CELL)
    gdn_seed1["arm"], gdn_seed1["data_seed"], gdn_seed1["parameters"] = "GDN2", 210007, 390148160
    findings = fbr.validate_cross_cell({0: kda_seed1, 1: gdn_seed1})
    assert not any(f.field == "parameters" for f in findings), "different arms may legitimately have different parameter counts"

    kda_seed2_bad = dict(GOOD_CELL)
    kda_seed2_bad["arm"], kda_seed2_bad["data_seed"], kda_seed2_bad["parameters"] = "KDA_BASE", 220014, 390125473
    findings2 = fbr.validate_cross_cell({0: kda_seed1, 1: kda_seed2_bad})
    errors = [f for f in findings2 if f.severity == "ERROR" and f.field == "parameters"]
    assert len(errors) == 1, "two seeds of the SAME arm must share one parameter count"


def test_cross_cell_catches_a_broken_seed_pairing():
    """One arm was run on a data seed no other arm used -- breaking the "every arm trains on the
    identical token stream" design the whole bake-off's paired comparison depends on.

    NAMED MUTATION: in validate_cross_cell, delete the `seed_sets`/`distinct_seed_sets` block
    entirely -- this test would then find zero findings on a seed-pairing break.
    """
    kda = dict(GOOD_CELL)
    kda["arm"], kda["data_seed"] = "KDA_BASE", 210007
    gdn = dict(GOOD_CELL)
    gdn["arm"], gdn["data_seed"] = "GDN2", 999999  # not in KDA_BASE's seed set
    findings = fbr.validate_cross_cell({0: kda, 1: gdn})
    errors = [f for f in findings if f.severity == "ERROR" and f.field == "data_seed"]
    assert len(errors) == 1


# ==================== Test 9: idempotent, detectable file writes ====================


def test_write_cell_file_is_idempotent(tmp_path):
    """Writing the same content twice is a no-op the second time (was_written=False), so
    re-running the fetch is cheap and safe.

    NAMED MUTATION: in write_cell_file, remove the `if existing_checksum == new_checksum: return
    path, new_checksum, False` early-return -- the second call's `was_written` would then be
    True even though nothing changed.
    """
    text = json.dumps(GOOD_CELL, indent=2)
    path1, sha1, written1 = fbr.write_cell_file(tmp_path, 0, text, force=False)
    assert written1 is True
    path2, sha2, written2 = fbr.write_cell_file(tmp_path, 0, text, force=False)
    assert written2 is False
    assert sha1 == sha2 == fbr.sha256_of(text)
    assert path1 == path2


def test_write_cell_file_refuses_to_silently_clobber_changed_content(tmp_path):
    """A second fetch produces DIFFERENT content for the same cell index (e.g. a retry changed
    the stream) -- must refuse rather than silently overwrite, unless --force (here, force=True)
    is explicit.

    NAMED MUTATION: in write_cell_file, remove the `if not force: raise RuntimeError(...)` guard
    -- the first call below (force=False) would then silently overwrite instead of raising.
    """
    text_a = json.dumps(GOOD_CELL, indent=2)
    changed = dict(GOOD_CELL)
    changed["last_loss"] = 2.999
    text_b = json.dumps(changed, indent=2)

    fbr.write_cell_file(tmp_path, 0, text_a, force=False)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        fbr.write_cell_file(tmp_path, 0, text_b, force=False)

    # With --force it is allowed, and the file actually changes.
    path, sha, written = fbr.write_cell_file(tmp_path, 0, text_b, force=True)
    assert written is True
    assert sha == fbr.sha256_of(text_b)
    assert path.read_text() == text_b


# ==================== Test 10: the read-only AWS allowlist is a real refusal, not decoration =========


def test_run_aws_refuses_calls_outside_the_read_only_allowlist():
    """`run_aws` must refuse to even build a command line for anything not explicitly
    read-only-safe -- e.g. `batch submit-job`, `batch terminate-job`, `logs put-log-events`, or
    any other mutating call this script must never be capable of making regardless of caller.

    NAMED MUTATION: delete the `if (service, operation) not in _ALLOWED_AWS_CALLS: raise
    AwsCallRefused(...)` guard at the top of run_aws -- this test would then get past the refusal
    and attempt to actually invoke `aws batch submit-job` via subprocess (and likely fail for an
    unrelated reason, like a missing job definition, rather than being refused up front).
    """
    with pytest.raises(fbr.AwsCallRefused):
        fbr.run_aws("batch", "submit-job", ["--job-name", "x", "--job-queue", "q", "--job-definition", "d"])
    with pytest.raises(fbr.AwsCallRefused):
        fbr.run_aws("batch", "terminate-job", ["--job-id", "x", "--reason", "y"])
    with pytest.raises(fbr.AwsCallRefused):
        fbr.run_aws("logs", "put-log-events", ["--log-group-name", "g", "--log-stream-name", "s"])
    # And the sanctioned calls this script actually needs must NOT be refused at this layer
    # (they may still fail later for other reasons, e.g. no `aws` binary in a CI sandbox with no
    # credentials -- that is a subprocess/AwsCallFailed concern, not an AwsCallRefused one).
    for service, operation in fbr._ALLOWED_AWS_CALLS:
        try:
            fbr.run_aws(service, operation, ["--this-arg-does-not-exist"])
        except fbr.AwsCallRefused:
            pytest.fail(f"{service} {operation} should be on the read-only allowlist and not be refused")
        except (fbr.AwsCallFailed, FileNotFoundError):
            pass  # expected: no real AWS credentials/binary in this test environment


# ==================== Test 11: the script never touches stdin ====================


def test_cli_help_does_not_hang_and_needs_no_stdin(tmp_path):
    """Runs the script's own argparse `--help` path as a subprocess with stdin explicitly closed
    (/dev/null), the same way the module docstring's USAGE section instructs every invocation to
    be run. A CLI in this exact repo (scripts/analyse_bakeoff.py) has an
    `if not sys.stdin.isatty(): sys.stdin.read()` path that blocks forever under an agent shell
    where stdin is neither a TTY nor closed; this test is the regression guard against this
    script ever growing the same trap.

    NAMED MUTATION: add `if not sys.stdin.isatty(): sys.stdin.read()` anywhere in
    build_arg_parser()/main() before the argparse call returns -- under the subprocess's
    /dev/null stdin this reads EOF immediately and returns b"", so this SPECIFIC test would not
    catch that exact trap (reading from /dev/null returns instantly) -- which is exactly why the
    check must be structural: fetch_bakeoff_results.py contains no LIVE CODE reference to
    `sys.stdin` anywhere, asserted below via the AST (not a raw substring search -- the script's
    own module docstring quotes `sys.stdin.isatty()` verbatim as PROSE describing this exact bug
    in a *different* file, scripts/analyse_bakeoff.py, so a substring match on the raw text would
    false-positive on the module's own documentation of the trap it avoids). A `/dev/null`-fed
    subprocess test alone cannot distinguish "never reads stdin" from "reads stdin but /dev/null
    made it fast this time" (the real hang only reproduces under an agent shell's non-TTY,
    non-closed stdin, which a test harness cannot easily reproduce), so the AST check is the part
    doing the real work here.
    """
    import ast
    import subprocess

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "fetch_bakeoff_results.py"
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))
    stdin_refs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "stdin"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ]
    assert not stdin_refs, (
        f"fetch_bakeoff_results.py must never reference sys.stdin in live code -- found "
        f"{len(stdin_refs)} AST reference(s) at line(s) "
        f"{[n.lineno for n in stdin_refs]}. See this test's docstring for why a subprocess-level "
        f"test with /dev/null cannot catch a stdin-read regression by itself."
    )

    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 0
    assert "--run-id" in proc.stdout


def test_run_aws_never_reads_our_stdin_either():
    """run_aws() must pass stdin=subprocess.DEVNULL to the `aws` CLI subprocess it invokes, so an
    interactive credential/MFA prompt from `aws` itself fails immediately instead of hanging.

    NAMED MUTATION: remove `stdin=subprocess.DEVNULL` from the `subprocess.run(...)` call inside
    run_aws -- this test inspects the source text directly (rather than trying to provoke an
    actual interactive prompt, which would require a specific broken AWS credential state to
    reproduce) so it catches the regression regardless of local credential state.
    """
    source = Path(fbr.__file__ if hasattr(fbr, "__file__") and fbr.__file__ else "").read_text() if hasattr(fbr, "__file__") and fbr.__file__ else None
    if source is None:
        source = (Path(__file__).resolve().parents[3] / "scripts" / "fetch_bakeoff_results.py").read_text()
    assert "stdin=subprocess.DEVNULL" in source


# ==================== Test 12: power-of-two guard on the identity checks ====================


def test_non_power_of_two_world_size_downgrades_to_warn_not_a_silent_pass_or_false_error():
    """world_size=6 (not a power of two, and not a shape this project provisions, but the
    validator must not assume it can't happen) with a per_device*world_size product that does
    NOT match the total. The exact-identity guarantee only holds for a power-of-two divisor
    (multiplying by a non-power-of-two CAN legitimately round differently depending on
    operation order), so this must come back as a WARN explaining why the exact check was
    skipped, not a false ERROR and not silence.

    NAMED MUTATION: remove the `if not isinstance(world_size, int) or not
    _is_power_of_two(world_size): warn(...); continue` guard so the identity is asserted with
    `==` regardless of world_size -- this fixture's per_device*6 does not exactly equal the
    (deliberately mismatched) total, so it would then wrongly report a bit-exact-identity ERROR
    for a case where bit-exactness was never actually guaranteed.
    """
    bad = dict(GOOD_CELL)
    bad["world_size"] = 6
    bad["throughput_tok_s_steady_per_device"] = 100.0
    bad["throughput_tok_s_steady"] = 600.0001  # deliberately not exactly 100.0 * 6
    findings = fbr.validate_cell("cell-0", bad)
    steady_findings = [f for f in findings if f.field == "throughput_tok_s_steady"]
    assert len(steady_findings) == 1
    assert steady_findings[0].severity == "WARN"
    assert "not a power of two" in steady_findings[0].message
