import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from edullm.policy import Policy, load_policy
from edullm.validation import (
    STATUS_MARKER,
    StatusCommentError,
    build_status_comment,
    parse_status_comment,
    validate_request,
    validated_status_for_request,
)

VALIDATED_AT = datetime(2026, 7, 23, 5, 0, 0, tzinfo=timezone.utc)


def _policy_with_option(valid_request, name, rule):
    return Policy(
        wandb_entity="eduLLM",
        allowed_wandb_projects=("test", "pretraining"),
        entrypoints={
            valid_request.entrypoint_profile: {
                "script": valid_request.script_path,
                "launcher": valid_request.launcher,
                "positionals": 3,
                "allowed_positionals": {
                    0: ("dry_run", "train_single", "train"),
                    2: ("local",),
                },
                "allowed_options": {name: rule},
            }
        },
    )


def _request_with_option(valid_request, name, value):
    return replace(
        valid_request,
        argv=("train_single", "skilldag-natural", "local", f"--{name}={value}"),
    )


def _generic_request(valid_request):
    return replace(
        valid_request,
        entrypoint_profile="generic-smoke",
        script_path="src/examples/llm/train.py",
        launcher="torchrun",
        argv=("generic-smoke-run",),
        wandb_project="test",
    )


def _payload(comment):
    marker, encoded = comment.split("\n", 1)
    assert marker == STATUS_MARKER
    return json.loads(encoded)


def _comment(payload):
    return STATUS_MARKER + "\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_valid_request_has_no_validation_errors(valid_request, policy):
    assert validate_request(valid_request, policy) == []


def test_full_lowercase_sha_passes_shape_validation_without_claiming_review_approval(
    valid_request, policy
):
    errors = validate_request(
        replace(valid_request, commit_sha="0123456789abcdef" * 2 + "01234567"), policy
    )

    assert not any("commit SHA" in error for error in errors)


@pytest.mark.parametrize("sha", ["main", "A" * 40, "a" * 39, "a" * 41, "g" * 40])
def test_rejects_invalid_commit_sha_shape(valid_request, policy, sha):
    errors = validate_request(replace(valid_request, commit_sha=sha), policy)

    assert "commit SHA must be 40 lowercase hexadecimal characters" in errors


def test_rejects_unknown_entrypoint_profile(valid_request, policy):
    errors = validate_request(replace(valid_request, entrypoint_profile="arbitrary"), policy)

    assert errors == ["entrypoint profile is not allowed"]


def test_rejects_script_that_does_not_match_profile(valid_request, policy):
    errors = validate_request(replace(valid_request, script_path="src/attacker.py"), policy)

    assert "script and launcher do not match the entrypoint profile" in errors


def test_rejects_script_path_traversal(valid_request, policy):
    errors = validate_request(replace(valid_request, script_path="../attacker.py"), policy)

    assert "script path must be repository-relative without traversal" in errors


def test_rejects_absolute_script_path(valid_request, policy):
    errors = validate_request(replace(valid_request, script_path="/tmp/attacker.py"), policy)

    assert "script path must be repository-relative without traversal" in errors


def test_rejects_unsupported_launcher_and_profile_mismatch(valid_request, policy):
    errors = validate_request(replace(valid_request, launcher="sh"), policy)

    assert "script and launcher do not match the entrypoint profile" in errors
    assert "launcher must be python, torchrun, or bash" in errors


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("train_single", "local"), "positional arguments do not match the entrypoint profile"),
        (
            ("arbitrary", "skilldag-natural", "local", "--seed=0"),
            "positional argument 0 is not allowed",
        ),
        (
            ("train_single", "skilldag-natural", "remote", "--seed=0"),
            "positional argument 2 is not allowed",
        ),
    ],
)
def test_rejects_profile_positional_boundary_violations(valid_request, policy, argv, message):
    errors = validate_request(replace(valid_request, argv=argv), policy)

    assert message in errors


def test_rejects_non_tuple_argv_without_iterating_a_shell_string(valid_request, policy):
    errors = validate_request(
        replace(valid_request, argv="train_single --seed=0"),  # type: ignore[arg-type]
        policy,
    )

    assert errors == ["arguments must be an immutable array of strings"]


def test_rejects_non_string_argv_member(valid_request, policy):
    errors = validate_request(
        replace(valid_request, argv=("train_single", 7)),  # type: ignore[arg-type]
        policy,
    )

    assert errors == ["argument 1 must be a string"]


@pytest.mark.parametrize(
    "argument",
    [
        "--output=ok;curl attacker",
        "--output=ok|cat",
        "--output=`id`",
        "--output=$(id)",
        "--output=$HOME/run",
        "--output=ok&&id",
        "--output=ok>file",
        "--output=line\nnext",
        "--output=line\rnext",
        "--output=null\x00byte",
    ],
)
def test_rejects_shell_control_syntax_in_arguments(valid_request, policy, argument):
    errors = validate_request(replace(valid_request, argv=(argument,)), policy)

    assert f"unsafe argument value: {argument!r}" in errors


def test_rejects_empty_argument(valid_request, policy):
    errors = validate_request(replace(valid_request, argv=("",)), policy)

    assert "argument values must not be empty" in errors


def test_rejects_unknown_options_in_sorted_order(valid_request, policy):
    request = replace(
        valid_request,
        argv=(
            "train_single",
            "skilldag-natural",
            "local",
            "--zeta=1",
            "--alpha=1",
        ),
    )

    errors = validate_request(request, policy)

    assert "options are not allowed for this entrypoint: ['alpha', 'zeta']" in errors


def test_rejects_option_without_an_explicit_value(valid_request, policy):
    errors = validate_request(
        replace(
            valid_request,
            argv=("train_single", "skilldag-natural", "local", "--seed"),
        ),
        policy,
    )

    assert "option must use --name=value form: --seed" in errors


def test_rejects_duplicate_options(valid_request, policy):
    errors = validate_request(
        replace(
            valid_request,
            argv=(
                "train_single",
                "skilldag-natural",
                "local",
                "--seed=0",
                "--seed=1",
            ),
        ),
        policy,
    )

    assert "option may be supplied only once: --seed" in errors


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-an-int", "value for --seed must be an integer"),
        ("-1", "value for --seed is outside its allowed range"),
        ("2147483648", "value for --seed is outside its allowed range"),
    ],
)
def test_validates_integer_option_type_and_range(valid_request, policy, value, message):
    errors = validate_request(_request_with_option(valid_request, "seed", value), policy)

    assert message in errors


def test_validates_enumerated_option_values(valid_request):
    policy = _policy_with_option(
        valid_request,
        "mode",
        {"type": "slug", "values": ("approved",)},
    )

    errors = validate_request(_request_with_option(valid_request, "mode", "other"), policy)

    assert "value for --mode is not allowed" in errors


@pytest.mark.parametrize(
    ("name", "rule", "value", "message"),
    [
        (
            "enabled",
            {"type": "boolean"},
            "yes",
            "value for --enabled must be true or false",
        ),
        (
            "run-name",
            {"type": "slug"},
            "UPPER_CASE",
            "value for --run-name must be a lowercase slug",
        ),
        (
            "duration",
            {"type": "duration", "max_steps": 100},
            "{value: 101, unit: steps}",
            "value for --duration exceeds the allowed smoke duration",
        ),
        (
            "duration",
            {"type": "duration", "max_steps": 100},
            "{value: true, unit: steps}",
            "value for --duration must be a duration mapping",
        ),
        (
            "duration",
            {"type": "duration", "max_steps": 100},
            "{value: 20, unit: epochs}",
            "value for --duration exceeds the allowed smoke duration",
        ),
    ],
)
def test_validates_boolean_slug_and_duration_options(valid_request, name, rule, value, message):
    policy = _policy_with_option(valid_request, name, rule)

    errors = validate_request(_request_with_option(valid_request, name, value), policy)

    assert message in errors


def test_accepts_a_typed_path_under_an_allowed_root(valid_request):
    policy = _policy_with_option(
        valid_request,
        "data-path",
        {"type": "path", "roots": ("/orcd/pool/data",)},
    )

    errors = validate_request(
        _request_with_option(valid_request, "data-path", "/orcd/pool/data/train.npy"),
        policy,
    )

    assert not any("path for --data-path" in error for error in errors)


@pytest.mark.parametrize(
    "value",
    [
        "/orcd/pool/data/../secret",
        "/orcd/pool/data-attacker/file",
        "relative/path",
    ],
)
def test_rejects_typed_paths_outside_component_aligned_roots(valid_request, value):
    policy = _policy_with_option(
        valid_request,
        "data-path",
        {"type": "path", "roots": ("/orcd/pool/data",)},
    )

    errors = validate_request(_request_with_option(valid_request, "data-path", value), policy)

    assert "path for --data-path is outside allowed roots" in errors


def test_fails_closed_for_an_unknown_option_rule_type(valid_request):
    policy = _policy_with_option(valid_request, "value", {"type": "unreviewed"})

    errors = validate_request(_request_with_option(valid_request, "value", "anything"), policy)

    assert "validation rule for --value is invalid" in errors


def test_rejects_researcher_override_of_policy_owned_fixed_options(valid_request):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        _generic_request(valid_request),
        argv=("generic-smoke-run", "--save-folder=$HOME/attacker"),
    )

    errors = validate_request(request, policy)

    assert "option is fixed by policy and cannot be supplied: --save-folder" in errors
    assert "unsafe argument value: '--save-folder=$HOME/attacker'" in errors


def test_rejects_researcher_override_of_fixed_launcher_arguments(valid_request):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    request = replace(
        _generic_request(valid_request),
        argv=("generic-smoke-run", "--standalone"),
    )

    errors = validate_request(request, policy)

    assert "launcher argument is fixed by policy and cannot be supplied: --standalone" in errors


def test_policy_owned_derived_paths_are_not_request_arguments(valid_request):
    policy = load_policy(Path("config/edullm/policy.yaml"))
    profile = policy.entrypoints["generic-smoke"]
    errors = validate_request(_generic_request(valid_request), policy)

    assert profile["fixed_options"]["save-folder"] == {
        "type": "derived_path",
        "root_env": "EDULLM_SCRATCH",
        "relative": "runs/{run_name}",
    }
    assert profile["fixed_options"]["work-dir"] == {
        "type": "derived_path",
        "root_env": "EDULLM_SCRATCH",
        "relative": "runs/{run_name}",
    }
    assert not any(
        "save-folder" in argument or "work-dir" in argument
        for argument in _generic_request(valid_request).argv
    )
    assert errors == []


@pytest.mark.parametrize("digest", ["b" * 63, "B" * 64, "g" * 64])
def test_rejects_invalid_manifest_digest_syntax(valid_request, policy, digest):
    errors = validate_request(replace(valid_request, data_manifest_sha256=digest), policy)

    assert "data manifest SHA-256 must be 64 lowercase hexadecimal characters" in errors


@pytest.mark.parametrize(
    "location",
    [
        "builtin://",
        "builtin://generic-smoke-v1/extra",
        "builtin://generic-smoke-v1?latest=true",
        "builtin://Generic-Smoke",
        "/orcd/pool",
        "/orcd/pool/../secret/manifest.json",
        "/orcd/pool/edullm/./manifest.json",
        "/orcd/pool-attacker/manifest.json",
        "$HOME/manifest.json",
        "s3://bucket/manifest.json",
    ],
)
def test_rejects_manifest_locations_outside_strict_allowed_forms(valid_request, policy, location):
    errors = validate_request(replace(valid_request, data_manifest=location), policy)

    assert "data manifest location is not allowed" in errors


@pytest.mark.parametrize(
    "location",
    [
        "builtin://generic-smoke-v1",
        "/orcd/pool/edullm/manifests/skill-dag-v1.json",
    ],
)
def test_accepts_strict_manifest_location_forms(valid_request, policy, location):
    errors = validate_request(replace(valid_request, data_manifest=location), policy)

    assert "data manifest location is not allowed" not in errors


@pytest.mark.parametrize("gpu_count", [0, 3, True])
def test_enforces_gpu_policy_cap_and_type(valid_request, policy, gpu_count):
    errors = validate_request(replace(valid_request, gpu_count=gpu_count), policy)

    assert f"GPU count must be an integer from 1 to {policy.max_gpu_count}" in errors


@pytest.mark.parametrize("runtime", [0, 361, True])
def test_enforces_runtime_policy_cap_and_type(valid_request, policy, runtime):
    errors = validate_request(replace(valid_request, max_runtime_minutes=runtime), policy)

    assert f"runtime must be an integer from 1 to {policy.max_runtime_minutes} minutes" in errors


def test_rejects_gpu_preference_outside_policy(valid_request, policy):
    errors = validate_request(replace(valid_request, gpu_preference="v100"), policy)

    assert "GPU preference is not allowed" in errors


def test_rejects_wandb_project_outside_policy(valid_request, policy):
    errors = validate_request(replace(valid_request, wandb_project="attacker"), policy)

    assert "W&B project is not allowed" in errors


@pytest.mark.parametrize(
    ("classification", "message"),
    [
        ("private", "data classification is invalid"),
        ("restricted", "restricted data is not accepted by the public pilot queue"),
    ],
)
def test_rejects_invalid_or_restricted_data(valid_request, policy, classification, message):
    errors = validate_request(
        replace(valid_request, data_classification=classification),
        policy,
    )

    assert message in errors


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ((), "at least one emitted success metric is required"),
        (("",), "success metric names must be non-empty and contain no whitespace"),
        (
            ("train/loss", "train/loss"),
            "success metric names must not be duplicated",
        ),
        (
            ("train loss",),
            "success metric names must be non-empty and contain no whitespace",
        ),
    ],
)
def test_validates_success_metric_names(valid_request, policy, metrics, message):
    errors = validate_request(replace(valid_request, success_metrics=metrics), policy)

    assert message in errors


@pytest.mark.parametrize(
    "metrics",
    [
        "train/loss",
        ("train/loss", 7),
    ],
)
def test_rejects_non_string_or_non_tuple_success_metrics(valid_request, policy, metrics):
    errors = validate_request(
        replace(valid_request, success_metrics=metrics),  # type: ignore[arg-type]
        policy,
    )

    assert errors == ["success metrics must be an immutable array of strings"]


@pytest.mark.parametrize("seed", [-1, True, "0"])
def test_requires_nonnegative_integer_seed(valid_request, policy, seed):
    errors = validate_request(replace(valid_request, seed=seed), policy)

    assert "seed must be a non-negative integer" in errors


@pytest.mark.parametrize("field", ["purpose", "study", "condition", "comparison", "success_signal"])
def test_rejects_empty_required_text_for_direct_job_requests(valid_request, policy, field):
    errors = validate_request(replace(valid_request, **{field: "  "}), policy)

    assert f"{field.replace('_', ' ')} must not be empty" in errors


def test_validation_error_order_is_deterministic(valid_request, policy):
    invalid = replace(
        valid_request,
        commit_sha="main",
        entrypoint_profile="arbitrary",
        data_manifest="s3://mutable/latest",
        data_manifest_sha256="BAD",
        gpu_count=3,
        gpu_preference="v100",
        max_runtime_minutes=999,
        wandb_project="attacker",
        data_classification="restricted",
        success_metrics=(),
    )

    expected = [
        "commit SHA must be 40 lowercase hexadecimal characters",
        "entrypoint profile is not allowed",
        "data manifest SHA-256 must be 64 lowercase hexadecimal characters",
        "data manifest location is not allowed",
        f"GPU count must be an integer from 1 to {policy.max_gpu_count}",
        f"runtime must be an integer from 1 to {policy.max_runtime_minutes} minutes",
        "GPU preference is not allowed",
        "W&B project is not allowed",
        "restricted data is not accepted by the public pilot queue",
        "at least one emitted success metric is required",
    ]

    assert validate_request(invalid, policy) == expected
    assert validate_request(invalid, policy) == expected


def test_status_comment_round_trips_canonical_request(valid_request):
    comment = build_status_comment(valid_request, validated_at=VALIDATED_AT)

    status = parse_status_comment(comment)

    assert comment.startswith("<!-- edullm-status:v1 -->\n{")
    assert f'"request":{valid_request.canonical_json()}' in comment
    assert "### Purpose" not in comment
    assert status.request == valid_request
    assert status.request_digest == valid_request.digest
    assert status.validated_at == VALIDATED_AT


def test_status_marker_text_inside_request_is_not_mistaken_for_duplicate_status(valid_request):
    request = replace(valid_request, purpose=f"study of {STATUS_MARKER} parsing")
    comment = build_status_comment(request, validated_at=VALIDATED_AT)

    status = parse_status_comment(comment)

    assert status.request == request


@pytest.mark.parametrize(
    "validated_at",
    [
        datetime(2026, 7, 23, 5, 0, 0),
        datetime(2026, 7, 23, 5, 0, 0, 1, tzinfo=timezone.utc),
    ],
)
def test_status_builder_requires_an_exact_utc_second(valid_request, validated_at):
    with pytest.raises(
        StatusCommentError,
        match="validation timestamp must be UTC with whole-second precision",
    ):
        build_status_comment(valid_request, validated_at=validated_at)


def test_status_parser_rejects_missing_marker():
    with pytest.raises(StatusCommentError, match="validated status marker is missing"):
        parse_status_comment("{}")


def test_status_parser_rejects_duplicate_markers(valid_request):
    comment = build_status_comment(valid_request, validated_at=VALIDATED_AT)

    with pytest.raises(
        StatusCommentError,
        match="validated status marker must appear exactly once",
    ):
        parse_status_comment(comment + "\n" + STATUS_MARKER)


def test_status_parser_rejects_arbitrary_text_around_payload(valid_request):
    comment = build_status_comment(valid_request, validated_at=VALIDATED_AT)

    with pytest.raises(
        StatusCommentError,
        match="validated status comment must contain only marker and payload",
    ):
        parse_status_comment("human text\n" + comment)


def test_status_parser_rejects_malformed_json():
    with pytest.raises(
        StatusCommentError,
        match="validated status payload is not valid JSON",
    ):
        parse_status_comment(STATUS_MARKER + "\n{")


def test_status_parser_rejects_noncanonical_json(valid_request):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    noncanonical = STATUS_MARKER + "\n" + json.dumps(payload, indent=2)

    with pytest.raises(
        StatusCommentError,
        match="validated status payload must use canonical JSON",
    ):
        parse_status_comment(noncanonical)


def test_status_parser_rejects_unknown_payload_fields(valid_request):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    payload["issue_body"] = "arbitrary Issue text"

    with pytest.raises(
        StatusCommentError,
        match="validated status payload fields are invalid",
    ):
        parse_status_comment(_comment(payload))


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-07-23 05:00:00Z",
        "2026-07-23T05:00:00+00:00",
        "2026-07-23T05:00:00.000000Z",
        "2026-02-30T05:00:00Z",
    ],
)
def test_status_parser_rejects_noncanonical_or_invalid_timestamps(valid_request, timestamp):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    payload["validated_at"] = timestamp

    with pytest.raises(
        StatusCommentError,
        match="validation timestamp must use YYYY-MM-DDTHH:MM:SSZ",
    ):
        parse_status_comment(_comment(payload))


def test_status_parser_rejects_malformed_digest(valid_request):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    payload["request_digest"] = "BAD"

    with pytest.raises(
        StatusCommentError,
        match="validated request digest must be 64 lowercase hexadecimal characters",
    ):
        parse_status_comment(_comment(payload))


def test_status_parser_rejects_tampered_request(valid_request):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    payload["request"]["purpose"] = "tampered"

    with pytest.raises(
        StatusCommentError,
        match="validated request digest does not match canonical request",
    ):
        parse_status_comment(_comment(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.pop("purpose"),
        lambda request: request.__setitem__("argv", "train_single --seed=0"),
        lambda request: request.__setitem__("issue_number", True),
        lambda request: request.__setitem__("status", "unknown"),
    ],
)
def test_status_parser_rejects_malformed_canonical_request(valid_request, mutation):
    payload = _payload(build_status_comment(valid_request, validated_at=VALIDATED_AT))
    mutation(payload["request"])
    request_json = json.dumps(payload["request"], sort_keys=True, separators=(",", ":"))
    payload["request_digest"] = hashlib.sha256(request_json.encode()).hexdigest()

    with pytest.raises(StatusCommentError, match="validated request is malformed"):
        parse_status_comment(_comment(payload))


def test_edited_request_is_stale_against_validated_status(valid_request):
    comment = build_status_comment(valid_request, validated_at=VALIDATED_AT)
    edited_request = replace(valid_request, purpose="edited purpose")

    with pytest.raises(
        StatusCommentError,
        match="validated status is stale for the current request",
    ):
        validated_status_for_request(comment, edited_request)


def test_exact_request_matches_validated_status(valid_request):
    comment = build_status_comment(valid_request, validated_at=VALIDATED_AT)

    status = validated_status_for_request(comment, valid_request)

    assert status.request_digest == valid_request.digest
    assert status.validated_at == VALIDATED_AT
