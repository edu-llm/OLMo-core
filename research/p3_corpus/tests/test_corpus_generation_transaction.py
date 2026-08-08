"""Hostile contract tests for versioned atomic corpus generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

import scripts.corpus_generation_transaction as transaction
from scripts.corpus_generation_transaction import (
    MANIFEST_FILENAME,
    ROUTES_FILENAME,
    AccountingError,
    BinaryValidator,
    CommitUncertainError,
    DropRecord,
    GenerationCoordinator,
    GenerationError,
    GenerationPlan,
    InventoryError,
    JsonlValidator,
    JsonObjectValidator,
    OutputRole,
    OutputSpec,
    PublishPhase,
    ValidationError,
)

ROW_SCHEMA = "synthetic-row/v2"
DROP_SCHEMA = "typed-drop/v2"
HELDOUT_SCHEMA = "heldout-family/v2"
SOURCE_GENERATION = "source-snapshot-001"

ROW_VALIDATOR = JsonlValidator(
    schema_version=ROW_SCHEMA,
    required_fields=("id", "proof"),
    allow_empty=False,
)
DROP_VALIDATOR = JsonlValidator(
    schema_version=DROP_SCHEMA,
    required_fields=(
        "details",
        "drop_type",
        "occurrence_id",
        "raw_path",
        "raw_row",
        "raw_sha256",
        "sibling",
    ),
    allow_empty=False,
    require_generation_links=True,
)
HELDOUT_VALIDATOR = JsonObjectValidator(
    schema_version=HELDOUT_SCHEMA,
    required_fields=("family", "marker"),
    require_generation_links=True,
)


def _line(sibling: str, marker: str, row: int) -> bytes:
    payload = {
        "schema_version": ROW_SCHEMA,
        "id": f"{sibling}-{marker}-{row}",
        "proof": f"native  spacing  {marker} \N{GREEK SMALL LETTER ALPHA}",
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _outputs(siblings: tuple[str, ...]) -> tuple[OutputSpec, ...]:
    outputs: list[OutputSpec] = []
    for sibling in siblings:
        outputs.extend(
            (
                OutputSpec(
                    path=f"raw/{sibling}.jsonl",
                    role=OutputRole.RAW,
                    schema=ROW_SCHEMA,
                    sibling=sibling,
                    validator=ROW_VALIDATOR,
                ),
                OutputSpec(
                    path=f"train/{sibling}.jsonl",
                    role=OutputRole.TRAIN,
                    schema=ROW_SCHEMA,
                    sibling=sibling,
                    validator=ROW_VALIDATOR,
                ),
                OutputSpec(
                    path=f"eval/{sibling}.jsonl",
                    role=OutputRole.EVAL,
                    schema=ROW_SCHEMA,
                    sibling=sibling,
                    validator=ROW_VALIDATOR,
                ),
                OutputSpec(
                    path=f"sidecars/{sibling}.drops.jsonl",
                    role=OutputRole.SIDECAR,
                    schema=DROP_SCHEMA,
                    sibling=sibling,
                    drop_types=("duplicate", "overlength", "parse_error"),
                    validator=DROP_VALIDATOR,
                ),
            )
        )
    outputs.append(
        OutputSpec(
            path="heldout/family.json",
            role=OutputRole.HELDOUT,
            schema=HELDOUT_SCHEMA,
            validator=HELDOUT_VALIDATOR,
        )
    )
    return tuple(outputs)


def _plan(
    generation_id: str,
    siblings: tuple[str, ...] = ("alpha", "beta"),
    *,
    outputs: tuple[OutputSpec, ...] | None = None,
) -> GenerationPlan:
    return GenerationPlan(
        generation_id=generation_id,
        source_generation_id=SOURCE_GENERATION,
        requested_siblings=siblings,
        outputs=outputs if outputs is not None else _outputs(siblings),
    )


def _write_siblings(
    writer,
    marker: str,
    *,
    siblings: tuple[str, ...] = ("alpha", "beta"),
    duplicate_first_two: bool = False,
) -> dict[str, tuple]:
    occurrences = {}
    for sibling in siblings:
        rows = [
            _line(sibling, marker, 1),
            _line(sibling, marker, 2),
            _line(sibling, marker, 3),
        ]
        if duplicate_first_two:
            rows[1] = rows[0]
        raw_path = f"raw/{sibling}.jsonl"
        writer.write_bytes(raw_path, b"".join(rows))
        sibling_occurrences = writer.raw_occurrences(raw_path)
        occurrences[sibling] = sibling_occurrences
        writer.write_routed_jsonl(
            f"train/{sibling}.jsonl",
            [sibling_occurrences[0]],
        )
        writer.write_routed_jsonl(
            f"eval/{sibling}.jsonl",
            [sibling_occurrences[1]],
        )
        writer.write_drop_sidecar(
            f"sidecars/{sibling}.drops.jsonl",
            [
                DropRecord(
                    occurrence_id=sibling_occurrences[2].occurrence_id,
                    drop_type="overlength",
                    details={"token_count": 16_385},
                )
            ],
        )
    return occurrences


def _write_complete(
    writer,
    marker: str,
    *,
    siblings: tuple[str, ...] = ("alpha", "beta"),
    duplicate_first_two: bool = False,
) -> dict[str, tuple]:
    occurrences = _write_siblings(
        writer,
        marker,
        siblings=siblings,
        duplicate_first_two=duplicate_first_two,
    )
    writer.write_linked_json(
        "heldout/family.json",
        {"family": list(siblings), "marker": marker},
    )
    return occurrences


def _active_path(coordinator: GenerationCoordinator) -> Path:
    return coordinator.resolve_current().path


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_every_output_requires_a_concrete_versioned_validator():
    with pytest.raises(InventoryError, match="validator"):
        OutputSpec(
            path="raw/alpha.jsonl",
            role=OutputRole.RAW,
            schema=ROW_SCHEMA,
            sibling="alpha",
            validator=None,
        )

    with pytest.raises(InventoryError, match="schema.*validator"):
        OutputSpec(
            path="raw/alpha.jsonl",
            role=OutputRole.RAW,
            schema="different/v9",
            sibling="alpha",
            validator=ROW_VALIDATOR,
        )

    binary = BinaryValidator(
        schema_version="packed-u32le/v1",
        validator_id="u32le-sequence-rows/v1",
        validate=lambda path, _context: path.stat().st_size // 4,
    )
    spec = OutputSpec(
        path="sidecars/tokens.u32le.bin",
        role=OutputRole.SIDECAR,
        schema="packed-u32le/v1",
        validator=binary,
    )
    assert spec.validator is binary


def test_plan_requires_exact_requested_siblings_and_all_roles():
    alpha_only = _outputs(("alpha",))
    with pytest.raises(InventoryError, match=r"beta.*raw.*train.*eval"):
        _plan("missing-beta", outputs=alpha_only)

    without_drop_sidecar = tuple(
        spec
        for spec in _outputs(("alpha",))
        if not (spec.role is OutputRole.SIDECAR and spec.drop_types)
    )
    with pytest.raises(InventoryError, match=r"alpha.*drop sidecar"):
        _plan(
            "missing-drop-sidecar",
            siblings=("alpha",),
            outputs=without_drop_sidecar,
        )

    duplicated = (*_outputs(("alpha",)), _outputs(("alpha",))[0])
    with pytest.raises(InventoryError, match="duplicate output path"):
        _plan("duplicate-path", siblings=("alpha",), outputs=duplicated)


def test_duplicate_native_bytes_have_distinct_physical_occurrences(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    captured = {}

    def producer(writer):
        captured.update(
            _write_complete(
                writer,
                "duplicate",
                siblings=("alpha",),
                duplicate_first_two=True,
            )
        )

    published = coordinator.publish(
        _plan("occurrences", siblings=("alpha",)),
        producer,
    )

    first, second, third = captured["alpha"]
    duplicate_hash = hashlib.sha256(_line("alpha", "duplicate", 1)).hexdigest()
    assert first.raw_sha256 == second.raw_sha256 == duplicate_hash
    assert first.occurrence_id != second.occurrence_id
    assert ":alpha:raw/alpha.jsonl:1:" in first.occurrence_id
    assert ":alpha:raw/alpha.jsonl:2:" in second.occurrence_id
    assert first.occurrence_id.endswith(duplicate_hash)

    routes = [
        json.loads(line)
        for line in (published.path / ROUTES_FILENAME).read_text().splitlines()
    ]
    assert {route["occurrence_id"] for route in routes} == {
        first.occurrence_id,
        second.occurrence_id,
        third.occurrence_id,
    }
    assert {route["disposition"] for route in routes} == {"train", "eval", "drop"}
    assert published.manifest["accounting"]["scheme"] == (
        "physical-occurrence-routes/v2"
    )
    assert published.manifest["accounting"]["siblings"]["alpha"] == {
        "drop_rows": 1,
        "drop_types": {"overlength": 1},
        "eval_rows": 1,
        "raw_rows": 3,
        "train_rows": 1,
    }


@pytest.mark.parametrize("defect", ["assigned_twice", "unassigned"])
def test_occurrence_must_have_exactly_one_disposition(tmp_path, defect):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    plan = _plan(defect, siblings=("alpha",))

    def producer(writer):
        rows = [_line("alpha", defect, index) for index in range(1, 5)]
        writer.write_bytes("raw/alpha.jsonl", b"".join(rows))
        occurrences = writer.raw_occurrences("raw/alpha.jsonl")
        writer.write_routed_jsonl("train/alpha.jsonl", [occurrences[0]])
        writer.write_routed_jsonl(
            "eval/alpha.jsonl",
            [occurrences[0] if defect == "assigned_twice" else occurrences[1]],
        )
        writer.write_drop_sidecar(
            "sidecars/alpha.drops.jsonl",
            [
                DropRecord(
                    occurrence_id=occurrences[2].occurrence_id,
                    drop_type="overlength",
                )
            ],
        )
        writer.write_linked_json(
            "heldout/family.json",
            {"family": ["alpha"], "marker": defect},
        )

    expected = "assigned more than once" if defect == "assigned_twice" else "unassigned"
    with pytest.raises(AccountingError, match=expected):
        coordinator.publish(plan, producer)

    assert not (coordinator.root / "CURRENT").exists()
    assert not (coordinator.generations_directory / defect).exists()


def test_cross_sibling_occurrence_route_is_rejected(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        alpha = [_line("alpha", "cross", index) for index in range(1, 4)]
        beta = [_line("beta", "cross", index) for index in range(1, 4)]
        writer.write_bytes("raw/alpha.jsonl", b"".join(alpha))
        writer.write_bytes("raw/beta.jsonl", b"".join(beta))
        alpha_occurrences = writer.raw_occurrences("raw/alpha.jsonl")
        writer.raw_occurrences("raw/beta.jsonl")
        writer.write_routed_jsonl("train/beta.jsonl", [alpha_occurrences[0]])

    with pytest.raises(AccountingError, match="cross-sibling"):
        coordinator.publish(_plan("cross-sibling"), producer)


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        b'{"schema_version":"synthetic-row/v2","id":"missing-proof"}\n',
        b'{"schema_version":"wrong/v1","id":"x","proof":"y"}\n',
    ],
)
def test_invalid_native_jsonl_schema_fails_before_inventory_acceptance(
    tmp_path,
    payload,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        writer.write_bytes("raw/alpha.jsonl", payload)
        writer.raw_occurrences("raw/alpha.jsonl")

    with pytest.raises(ValidationError, match="raw/alpha.jsonl"):
        coordinator.publish(_plan("invalid-jsonl", siblings=("alpha",)), producer)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_id", "stale-generation", "generation link"),
        ("plan_root_sha256", "0" * 64, "root link"),
    ],
)
def test_stale_or_wrong_root_heldout_is_rejected(tmp_path, field, value, message):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    plan = _plan("linked", siblings=("alpha",))

    def producer(writer):
        _write_siblings(writer, "linked", siblings=("alpha",))
        payload = {
            "schema_version": HELDOUT_SCHEMA,
            "generation_id": plan.generation_id,
            "source_generation_id": plan.source_generation_id,
            "plan_root_sha256": plan.plan_root_sha256,
            "family": ["alpha"],
            "marker": "linked",
        }
        payload[field] = value
        writer.write_bytes(
            "heldout/family.json",
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )

    with pytest.raises(ValidationError, match=message):
        coordinator.publish(plan, producer)


def test_empty_structured_output_and_empty_sibling_are_rejected(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        writer.write_bytes("raw/alpha.jsonl", b"")
        writer.raw_occurrences("raw/alpha.jsonl")

    with pytest.raises(ValidationError, match="empty"):
        coordinator.publish(_plan("empty", siblings=("alpha",)), producer)

    with pytest.raises(InventoryError, match="at least one requested sibling"):
        _plan("no-siblings", siblings=(), outputs=())


@pytest.mark.parametrize("defect", ["missing", "extra", "unregistered"])
def test_native_adapter_requires_exact_inventoried_files(tmp_path, defect):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        _write_complete(writer, "adapter", siblings=("alpha",))
        staging = coordinator.staging_directory / f"{defect}.staging"
        if defect == "missing":
            (staging / "eval/alpha.jsonl").unlink()
        elif defect == "extra":
            (staging / "extra.json").write_text("{}")
        else:
            path = staging / "train/alpha.jsonl"
            path.unlink()
            path.write_bytes(_line("alpha", "adapter", 1))

    expected = "missing" if defect == "missing" else defect
    with pytest.raises(InventoryError, match=expected):
        coordinator.publish(_plan(defect, siblings=("alpha",)), producer)


def test_native_adapter_registers_routes_without_rewriting_bytes(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    plan = _plan("native-adapter", siblings=("alpha",))
    rows = [_line("alpha", "native", index) for index in range(1, 4)]
    trusted_tool_output = tmp_path / "trusted-tool-output.jsonl"
    trusted_tool_output.write_bytes(b"".join(rows))
    trusted_routed_output = tmp_path / "trusted-routed-output.jsonl"
    trusted_routed_output.write_bytes(rows[0])

    def producer(writer):
        writer.copy_file("raw/alpha.jsonl", trusted_tool_output)
        occurrences = writer.raw_occurrences("raw/alpha.jsonl")

        writer.copy_file(
            "train/alpha.jsonl",
            trusted_routed_output,
            occurrence_ids=[occurrences[0].occurrence_id],
        )
        with writer.open_routed_output(
            "eval/alpha.jsonl",
            [occurrences[1].occurrence_id],
        ) as output:
            assert isinstance(output.name, int)
            output.write(occurrences[1].raw_bytes)

        writer.write_drop_sidecar(
            "sidecars/alpha.drops.jsonl",
            [
                DropRecord(
                    occurrence_id=occurrences[2].occurrence_id,
                    drop_type="overlength",
                )
            ],
        )
        writer.write_linked_json(
            "heldout/family.json",
            {"family": ["alpha"], "marker": "native"},
        )

    published = coordinator.publish(plan, producer)
    assert (published.path / "raw/alpha.jsonl").read_bytes() == b"".join(rows)
    assert (published.path / "train/alpha.jsonl").read_bytes() == rows[0]
    assert (published.path / "eval/alpha.jsonl").read_bytes() == rows[1]


PRE_COMMIT_PHASES = (
    PublishPhase.STAGING_CREATED,
    PublishPhase.OUTPUTS_WRITTEN,
    PublishPhase.MANIFEST_WRITTEN,
    PublishPhase.VALIDATED,
    PublishPhase.STAGING_SYNCED,
    PublishPhase.GENERATION_RENAME_BEFORE,
    PublishPhase.GENERATION_RENAME_AFTER,
    PublishPhase.GENERATIONS_FSYNC_BEFORE,
    PublishPhase.GENERATIONS_FSYNC_AFTER,
    PublishPhase.CURRENT_TEMP_WRITE_BEFORE,
    PublishPhase.CURRENT_TEMP_WRITE_AFTER,
    PublishPhase.CURRENT_TEMP_PARENT_FSYNC_BEFORE,
    PublishPhase.CURRENT_TEMP_PARENT_FSYNC_AFTER,
    PublishPhase.CURRENT_REPLACE_BEFORE,
)


@pytest.mark.parametrize("phase", PRE_COMMIT_PHASES)
def test_every_precommit_failure_keeps_old_current_and_allows_same_id_retry(
    tmp_path,
    phase,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    coordinator.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))
    old_tree = _tree_bytes(coordinator.generations_directory / "old")

    class InjectedInterruption(RuntimeError):
        pass

    def interrupt(current_phase, _generation_path):
        if current_phase is phase:
            raise InjectedInterruption(current_phase.value)

    with pytest.raises(InjectedInterruption, match=phase.value):
        coordinator.publish(
            _plan("retryable"),
            lambda writer: _write_complete(writer, "new"),
            fault_injector=interrupt,
        )

    assert coordinator.resolve_current().generation_id == "old"
    assert _tree_bytes(coordinator.generations_directory / "old") == old_tree
    assert not (coordinator.generations_directory / "retryable").exists()
    assert coordinator.quarantine_inventory()

    retried = coordinator.publish(
        _plan("retryable"),
        lambda writer: _write_complete(writer, "new"),
    )
    assert retried.generation_id == "retryable"
    assert coordinator.resolve_current().generation_id == "retryable"


@pytest.mark.parametrize(
    "phase",
    (
        PublishPhase.CURRENT_REPLACE_AFTER,
        PublishPhase.ROOT_FSYNC_BEFORE,
    ),
)
def test_post_replace_failure_is_distinct_commit_uncertain_state(tmp_path, phase):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    coordinator.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))

    def interrupt(current_phase, _generation_path):
        if current_phase is phase:
            raise RuntimeError(current_phase.value)

    with pytest.raises(CommitUncertainError) as raised:
        coordinator.publish(
            _plan("new"),
            lambda writer: _write_complete(writer, "new"),
            fault_injector=interrupt,
        )

    adjudicated = raised.value.resolve(coordinator)
    assert adjudicated.generation_id == "new"
    assert coordinator.resolve_current().generation_id == "new"


def test_failure_after_root_fsync_returns_success_not_ordinary_failure(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def interrupt(phase, _generation_path):
        if phase is PublishPhase.ROOT_FSYNC_AFTER:
            raise RuntimeError("after durable commit")

    published = coordinator.publish(
        _plan("durable"),
        lambda writer: _write_complete(writer, "durable"),
        fault_injector=interrupt,
    )

    assert published.generation_id == "durable"
    assert published.commit_state == "durable"
    assert published.post_commit_warnings
    assert coordinator.resolve_current().generation_id == "durable"


def test_commit_point_is_explicit_and_no_ordinary_failure_changes_current(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    assert coordinator.commit_point == "successful atomic CURRENT replacement"

    coordinator.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))

    def interrupt(phase, _generation_path):
        if phase is PublishPhase.CURRENT_REPLACE_BEFORE:
            raise RuntimeError("before commit")

    with pytest.raises(RuntimeError, match="before commit"):
        coordinator.publish(
            _plan("new"),
            lambda writer: _write_complete(writer, "new"),
            fault_injector=interrupt,
        )
    assert coordinator.resolve_current().generation_id == "old"


def test_sealed_generation_is_read_only_and_writable_modes_are_rejected(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    published = coordinator.publish(
        _plan("sealed"),
        lambda writer: _write_complete(writer, "sealed"),
    )

    for path in [published.path, *published.path.rglob("*")]:
        mode = stat.S_IMODE(path.lstat().st_mode)
        assert mode & 0o222 == 0

    output = published.path / "train/alpha.jsonl"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        descriptor = None
    if descriptor is not None:
        os.close(descriptor)
        if os.geteuid() != 0:
            pytest.fail("sealed output unexpectedly opened writable")

    output.chmod(0o644)
    with pytest.raises(ValidationError, match="writable"):
        coordinator.resolve_current()


def test_digest_mutation_is_detected_even_when_read_only_mode_is_restored(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    published = coordinator.publish(
        _plan("tamper"),
        lambda writer: _write_complete(writer, "tamper"),
    )
    output = published.path / "train/alpha.jsonl"
    output.chmod(0o644)
    output.write_bytes(output.read_bytes() + _line("alpha", "late", 99))
    output.chmod(0o444)

    with pytest.raises(ValidationError, match=r"train/alpha.jsonl.*SHA-256"):
        coordinator.resolve_current()


def test_train_eval_route_swap_mutation_is_rejected(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    published = coordinator.publish(
        _plan("route-swap", siblings=("alpha",)),
        lambda writer: _write_complete(
            writer,
            "route-swap",
            siblings=("alpha",),
        ),
    )
    train = published.path / "train/alpha.jsonl"
    evaluation = published.path / "eval/alpha.jsonl"
    train_bytes = train.read_bytes()
    eval_bytes = evaluation.read_bytes()
    train.chmod(0o644)
    evaluation.chmod(0o644)
    train.write_bytes(eval_bytes)
    evaluation.write_bytes(train_bytes)
    train.chmod(0o444)
    evaluation.chmod(0o444)

    with pytest.raises(ValidationError, match="SHA-256|routed native bytes"):
        coordinator.resolve_current()


def test_logical_root_is_deterministic_across_physical_generation_ids(tmp_path):
    first = GenerationCoordinator(tmp_path / "first")
    second = GenerationCoordinator(tmp_path / "second")

    run_a = first.publish(
        _plan("physical-A"),
        lambda writer: _write_complete(writer, "same-native"),
    )
    run_b = second.publish(
        _plan("physical-B"),
        lambda writer: _write_complete(writer, "same-native"),
    )

    assert run_a.path.name == "physical-A"
    assert run_b.path.name == "physical-B"
    assert run_a.logical_root_sha256 == run_b.logical_root_sha256
    assert run_a.manifest["physical_generation_id_policy"] == (
        "caller-supplied-immutable-id/v1"
    )
    assert (run_a.path / "raw/alpha.jsonl").read_bytes() == (
        run_b.path / "raw/alpha.jsonl"
    ).read_bytes()
    assert (run_a.path / "heldout/family.json").read_bytes() != (
        run_b.path / "heldout/family.json"
    ).read_bytes()


def test_quarantine_is_inventoried_and_ignored_by_current_resolver(tmp_path):
    root = tmp_path / "corpus"
    malformed = root / ".staging/interrupted.staging"
    malformed.mkdir(parents=True)
    (malformed / "partial.jsonl").write_bytes(b'{"partial":true}\n')

    coordinator = GenerationCoordinator(root)
    coordinator.publish(_plan("fresh"), lambda writer: _write_complete(writer, "fresh"))

    inventory = coordinator.quarantine_inventory()
    assert len(inventory) == 1
    assert inventory[0]["original_name"] == "interrupted.staging"
    assert inventory[0]["reason"] == "stale or malformed staging generation"
    payload = root / "quarantine" / inventory[0]["entry"] / "payload"
    assert (payload / "partial.jsonl").read_bytes() == b'{"partial":true}\n'

    (root / "quarantine" / "uninventoried").mkdir()
    assert coordinator.resolve_current().generation_id == "fresh"
    with pytest.raises(ValidationError, match="uninventoried quarantine"):
        coordinator.quarantine_inventory()


def test_current_path_traversal_and_symlink_are_rejected(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    pointer = {
        "schema_version": "corpus-generation-current/v2",
        "generation_id": "../escape",
        "manifest_sha256": "0" * 64,
        "logical_root_sha256": "0" * 64,
    }
    (root / "CURRENT").write_text(json.dumps(pointer))
    with pytest.raises(ValidationError, match="generation ID"):
        GenerationCoordinator(root).resolve_current()

    (root / "CURRENT").unlink()
    target = root / "pointer-target"
    target.write_text(json.dumps(pointer))
    (root / "CURRENT").symlink_to(target)
    with pytest.raises(ValidationError, match="symlink"):
        GenerationCoordinator(root).resolve_current()


def test_invalid_transaction_manifest_is_never_accepted(tmp_path):
    root = tmp_path / "corpus"
    generation = root / "generations/fake"
    generation.mkdir(parents=True)
    manifest = generation / MANIFEST_FILENAME
    manifest.write_text("{}")
    pointer = {
        "schema_version": "corpus-generation-current/v2",
        "generation_id": "fake",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "logical_root_sha256": "0" * 64,
    }
    (root / "CURRENT").write_text(json.dumps(pointer))

    with pytest.raises(ValidationError, match="manifest"):
        GenerationCoordinator(root).resolve_current()


def test_mocked_builder_and_tokenizer_consumers_refuse_unsealed_legacy_paths(tmp_path):
    def mocked_builder_consumer(path):
        return GenerationCoordinator(path).resolve_current(
            required_siblings=("alpha", "beta")
        )

    def mocked_tokenizer_consumer(path):
        return GenerationCoordinator(path).resolve_current(
            required_siblings=("alpha", "beta")
        )

    legacy = tmp_path / "legacy"
    (legacy / "shards").mkdir(parents=True)
    (legacy / "shards/alpha.jsonl").write_bytes(_line("alpha", "legacy", 1))
    for consumer in (mocked_builder_consumer, mocked_tokenizer_consumer):
        with pytest.raises(ValidationError, match="CURRENT"):
            consumer(legacy)

    incomplete = GenerationCoordinator(tmp_path / "incomplete")
    incomplete.publish(
        _plan("alpha-only", siblings=("alpha",)),
        lambda writer: _write_complete(writer, "alpha", siblings=("alpha",)),
    )
    for consumer in (mocked_builder_consumer, mocked_tokenizer_consumer):
        with pytest.raises(ValidationError, match="required sibling"):
            consumer(incomplete.root)

    complete = GenerationCoordinator(tmp_path / "complete")
    complete.publish(
        _plan("all-families"),
        lambda writer: _write_complete(writer, "all"),
    )
    assert mocked_builder_consumer(complete.root).generation_id == "all-families"
    assert mocked_tokenizer_consumer(complete.root).generation_id == "all-families"


def test_hard_crash_orphan_is_reconciled_and_same_physical_id_retries(tmp_path):
    target = GenerationCoordinator(tmp_path / "target")
    target.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))
    source = GenerationCoordinator(tmp_path / "source")
    orphan = source.publish(
        _plan("retry-after-crash"),
        lambda writer: _write_complete(writer, "orphan"),
    )

    copied = target.generations_directory / "retry-after-crash"
    shutil.copytree(orphan.path, copied, copy_function=shutil.copy2)
    pending = target.transactions_directory / "retry-after-crash.pending.json"
    pending.write_text(
        json.dumps(
            {
                "generation_id": "retry-after-crash",
                "logical_root_sha256": orphan.logical_root_sha256,
                "manifest_sha256": orphan.manifest_sha256,
                "schema_version": "generation-transaction-state/v1",
                "state": "pending",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    retried = target.publish(
        _plan("retry-after-crash"),
        lambda writer: _write_complete(writer, "retried"),
    )

    assert retried.generation_id == "retry-after-crash"
    assert target.resolve_current().generation_id == "retry-after-crash"
    originals = {record["original_name"] for record in target.quarantine_inventory()}
    assert "retry-after-crash" in originals
    assert "retry-after-crash.pending.json" in originals


def test_exception_after_current_replace_syscall_is_never_ordinary(
    tmp_path,
    monkeypatch,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    coordinator.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))
    real_rename = transaction.os.rename

    def rename_then_interrupt(source, destination, *args, **kwargs):
        real_rename(source, destination, *args, **kwargs)
        if destination == "CURRENT":
            raise KeyboardInterrupt("after successful replace syscall")

    monkeypatch.setattr(transaction.os, "rename", rename_then_interrupt)
    with pytest.raises(CommitUncertainError) as raised:
        coordinator.publish(
            _plan("new"),
            lambda writer: _write_complete(writer, "new"),
        )

    assert coordinator.resolve_current().generation_id == "new"
    assert (coordinator.generations_directory / "new").is_dir()
    assert raised.value.resolve(coordinator).commit_state == "durable_recovered"


def test_commit_uncertainty_adjudication_retries_root_fsync(tmp_path, monkeypatch):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    coordinator.publish(_plan("old"), lambda writer: _write_complete(writer, "old"))
    calls = []
    real_fsync = transaction._fsync_directory

    def interrupt(phase, _generation_path):
        if phase is PublishPhase.ROOT_FSYNC_BEFORE:
            raise RuntimeError("root fsync not attempted")

    with pytest.raises(CommitUncertainError) as raised:
        coordinator.publish(
            _plan("new"),
            lambda writer: _write_complete(writer, "new"),
            fault_injector=interrupt,
        )

    def record_fsync(path):
        calls.append(Path(path))
        real_fsync(path)

    monkeypatch.setattr(transaction, "_fsync_directory", record_fsync)
    recovered = raised.value.resolve(coordinator)
    assert coordinator.root in calls
    assert recovered.commit_state == "durable_recovered"


def test_resolver_rejects_symlinked_root_and_generations_control_path(tmp_path):
    source = GenerationCoordinator(tmp_path / "source")
    source.publish(_plan("sealed"), lambda writer: _write_complete(writer, "sealed"))

    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(source.root, target_is_directory=True)
    with pytest.raises(ValidationError, match="symlink"):
        GenerationCoordinator(root_alias).resolve_current()

    linked_control = tmp_path / "linked-control"
    linked_control.mkdir()
    (linked_control / "CURRENT").write_bytes((source.root / "CURRENT").read_bytes())
    (linked_control / "generations").symlink_to(
        source.generations_directory,
        target_is_directory=True,
    )
    with pytest.raises(
        ValidationError, match="generations.*symlink|symlink.*generations"
    ):
        GenerationCoordinator(linked_control).resolve_current()


def test_empty_validators_and_linkless_structured_sidecars_are_forbidden():
    empty = JsonlValidator(
        schema_version=ROW_SCHEMA,
        required_fields=("id", "proof"),
        allow_empty=True,
    )
    with pytest.raises(InventoryError, match="empty"):
        OutputSpec(
            path="raw/alpha.jsonl",
            role=OutputRole.RAW,
            schema=ROW_SCHEMA,
            sibling="alpha",
            validator=empty,
        )

    linkless = JsonObjectValidator(
        schema_version="statistics/v1",
        required_fields=("value",),
        require_generation_links=False,
    )
    with pytest.raises(InventoryError, match="sidecar.*generation/root links"):
        OutputSpec(
            path="sidecars/statistics.json",
            role=OutputRole.SIDECAR,
            schema="statistics/v1",
            validator=linkless,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["routes"].__setitem__("rows", 3.0), "routes.*rows"),
        (
            lambda manifest: manifest["outputs"][0]["validator"].__setitem__(
                "allow_empty", 0
            ),
            "validator.*boolean",
        ),
    ],
)
def test_manifest_reconstruction_rejects_loosely_typed_metadata(
    tmp_path,
    mutation,
    message,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def tamper(phase, generation_path):
        if phase is not PublishPhase.MANIFEST_WRITTEN:
            return
        path = generation_path / MANIFEST_FILENAME
        manifest = json.loads(path.read_text())
        mutation(manifest)
        body = dict(manifest)
        body.pop("manifest_root_sha256")
        manifest["manifest_root_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        ).hexdigest()
        path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )

    with pytest.raises(ValidationError, match=message):
        coordinator.publish(
            _plan("typed-manifest", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "typed",
                siblings=("alpha",),
            ),
            fault_injector=tamper,
        )


def test_layout_fsyncs_each_newly_created_ancestor(tmp_path, monkeypatch):
    synced = []
    real_fsync = transaction._fsync_directory

    def record_fsync(path):
        synced.append(Path(path))
        real_fsync(path)

    monkeypatch.setattr(transaction, "_fsync_directory", record_fsync)
    root = tmp_path / "one/two/corpus"
    GenerationCoordinator(root)._ensure_layout()

    assert {tmp_path, tmp_path / "one", tmp_path / "one/two"} <= set(synced)


def test_quarantine_inventory_rejects_extra_material_and_symlink_payload(tmp_path):
    root = tmp_path / "corpus"
    stale = root / ".staging/stale.staging"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("partial")
    coordinator = GenerationCoordinator(root)
    coordinator.publish(_plan("fresh"), lambda writer: _write_complete(writer, "fresh"))
    entry = root / "quarantine" / coordinator.quarantine_inventory()[0]["entry"]
    (entry / "extra").write_text("uninventoried")
    with pytest.raises(ValidationError, match="extra.*quarantine|quarantine.*extra"):
        coordinator.quarantine_inventory()

    (entry / "extra").unlink()
    shutil.rmtree(entry / "payload")
    (entry / "payload").symlink_to(root / "CURRENT")
    with pytest.raises(ValidationError, match="payload.*symlink|symlink.*payload"):
        coordinator.quarantine_inventory()


def test_transaction_state_files_are_published_only_by_atomic_replace(
    tmp_path,
    monkeypatch,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    replaced_destinations = []
    real_rename = transaction.os.rename

    def record_replace(source, destination, *args, **kwargs):
        replaced_destinations.append(destination)
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(transaction.os, "rename", record_replace)
    coordinator.publish(
        _plan("journaled"),
        lambda writer: _write_complete(writer, "journaled"),
    )

    assert (
        coordinator._pending_transaction_path("journaled").name in replaced_destinations
    )
    assert (
        coordinator._committed_transaction_path("journaled").name
        in replaced_destinations
    )
    assert not list(coordinator.transactions_directory.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.__setitem__("api_version", 2.0), "API version"),
        (
            lambda manifest: next(
                item for item in manifest["outputs"] if item["drop_types"]
            ).__setitem__(
                "drop_types",
                {
                    "duplicate": False,
                    "overlength": False,
                    "parse_error": False,
                },
            ),
            "drop_types.*list",
        ),
    ],
)
def test_resolver_strictly_types_top_level_and_output_manifest_fields(
    tmp_path,
    mutation,
    message,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    plan = _plan("strict-manifest", siblings=("alpha",))
    published = coordinator.publish(
        plan,
        lambda writer: _write_complete(
            writer,
            "strict",
            siblings=("alpha",),
        ),
    )
    manifest_path = published.path / MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    mutation(manifest)
    manifest["logical_root_sha256"] = transaction._logical_root(
        plan,
        manifest["outputs"],
        manifest["routes"],
    )
    body = dict(manifest)
    body.pop("manifest_root_sha256")
    manifest["manifest_root_sha256"] = hashlib.sha256(
        (
            json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    manifest_path.chmod(0o444)

    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path = coordinator.root / "CURRENT"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = manifest_sha256
    pointer["logical_root_sha256"] = manifest["logical_root_sha256"]
    pointer_path.write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
    )
    transaction_path = coordinator._committed_transaction_path("strict-manifest")
    transaction_state = json.loads(transaction_path.read_text())
    transaction_state["manifest_sha256"] = manifest_sha256
    transaction_state["logical_root_sha256"] = manifest["logical_root_sha256"]
    transaction_path.write_text(
        json.dumps(
            transaction_state,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with pytest.raises(ValidationError, match=message):
        coordinator.resolve_current()


def test_manifest_and_physical_inventory_include_exact_directories(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    published = coordinator.publish(
        _plan("directory-inventory", siblings=("alpha",)),
        lambda writer: _write_complete(
            writer,
            "directories",
            siblings=("alpha",),
        ),
    )

    assert published.manifest["directories"] == [
        "eval",
        "heldout",
        "raw",
        "sidecars",
        "train",
    ]
    assert coordinator.resolve_current().logical_root_sha256 == (
        published.logical_root_sha256
    )


def test_undeclared_empty_nested_directory_never_publishes(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        _write_complete(writer, "extra-dir", siblings=("alpha",))
        staging = coordinator.staging_directory / "extra-dir.staging"
        (staging / "undeclared/empty/nested").mkdir(parents=True)

    with pytest.raises(InventoryError, match="undeclared director"):
        coordinator.publish(
            _plan("extra-dir", siblings=("alpha",)),
            producer,
        )


def test_resolver_rejects_undeclared_empty_directory_after_seal(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    published = coordinator.publish(
        _plan("late-dir", siblings=("alpha",)),
        lambda writer: _write_complete(
            writer,
            "late-dir",
            siblings=("alpha",),
        ),
    )
    published.path.chmod(0o755)
    extra = published.path / "empty-surprise"
    extra.mkdir()
    extra.chmod(0o555)
    published.path.chmod(0o555)

    with pytest.raises(ValidationError, match="undeclared director"):
        coordinator.resolve_current()


def test_special_node_is_rejected_as_inventory_corruption(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform")
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        _write_complete(writer, "fifo", siblings=("alpha",))
        staging = coordinator.staging_directory / "fifo.staging"
        os.mkfifo(staging / "rogue.fifo")

    with pytest.raises(InventoryError, match="special node"):
        coordinator.publish(
            _plan("fifo", siblings=("alpha",)),
            producer,
        )


def test_symlinked_output_parent_creates_no_outside_file(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    outside = tmp_path / "outside"
    outside.mkdir()

    def producer(writer):
        staging = coordinator.staging_directory / "parent-symlink.staging"
        (staging / "raw").symlink_to(
            outside,
            target_is_directory=True,
        )
        writer.write_bytes("raw/alpha.jsonl", _line("alpha", "escape", 1))

    with pytest.raises((InventoryError, ValidationError), match="symlink"):
        coordinator.publish(
            _plan("parent-symlink", siblings=("alpha",)),
            producer,
        )
    assert not (outside / "alpha.jsonl").exists()


def test_symlinked_output_leaf_is_rejected_without_touching_target(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"sentinel")

    def producer(writer):
        staging = coordinator.staging_directory / "leaf-symlink.staging"
        parent = staging / "raw"
        parent.mkdir()
        (parent / "alpha.jsonl").symlink_to(outside)
        writer.write_bytes("raw/alpha.jsonl", _line("alpha", "leaf", 1))

    with pytest.raises(InventoryError, match="symlink|duplicate physical output"):
        coordinator.publish(
            _plan("leaf-symlink", siblings=("alpha",)),
            producer,
        )
    assert outside.read_bytes() == b"sentinel"


def test_symlinked_staging_control_creates_no_outside_generation(tmp_path):
    root = tmp_path / "corpus"
    outside = tmp_path / "outside-staging"
    outside.mkdir()
    root.mkdir()
    (root / "generations").mkdir()
    (root / "quarantine").mkdir()
    (root / "transactions").mkdir()
    (root / ".staging").symlink_to(outside, target_is_directory=True)

    with pytest.raises((GenerationError, ValidationError), match="symlink"):
        GenerationCoordinator(root).publish(
            _plan("staging-symlink", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "staging-symlink",
                siblings=("alpha",),
            ),
        )
    assert list(outside.iterdir()) == []


def test_symlinked_root_ancestor_creates_no_outside_control_paths(tmp_path):
    outside = tmp_path / "outside-root"
    outside.mkdir()
    alias = tmp_path / "root-alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises((GenerationError, ValidationError), match="symlink"):
        GenerationCoordinator(alias / "corpus").publish(
            _plan("root-symlink", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "root-symlink",
                siblings=("alpha",),
            ),
        )
    assert list(outside.iterdir()) == []


def test_parent_symlink_swap_cannot_redirect_dirfd_write(
    tmp_path,
    monkeypatch,
):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    outside = tmp_path / "race-outside"
    outside.mkdir()
    swapped = []
    real_open = transaction.os.open

    def swap_before_leaf_open(path, flags, *args, dir_fd=None, **kwargs):
        if (
            path == "alpha.jsonl"
            and dir_fd is not None
            and flags & os.O_CREAT
            and not swapped
        ):
            staging = coordinator.staging_directory / "swap-race.staging"
            parent = staging / "raw"
            parent.rename(staging / "raw-before-swap")
            parent.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(transaction.os, "open", swap_before_leaf_open)

    def producer(writer):
        writer.write_bytes("raw/alpha.jsonl", _line("alpha", "race", 1))
        raise RuntimeError("stop after swap")

    with pytest.raises(Exception, match="symlink|stop after swap"):
        coordinator.publish(
            _plan("swap-race", siblings=("alpha",)),
            producer,
        )
    assert swapped == [True]
    assert not (outside / "alpha.jsonl").exists()


def test_registered_output_hardlink_is_rejected(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(_line("alpha", "hardlink", 1))
    linked = tmp_path / "linked.jsonl"
    os.link(outside, linked)

    def producer(writer):
        writer.copy_file("raw/alpha.jsonl", linked)

    with pytest.raises(InventoryError, match="hardlink|hard link"):
        coordinator.publish(
            _plan("hardlink", siblings=("alpha",)),
            producer,
        )
    assert outside.read_bytes() == _line("alpha", "hardlink", 1)


def test_pathname_adapter_apis_are_unconditionally_unsafe(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")

    def producer(writer):
        assert not hasattr(writer, "staging_directory")
        for call in (
            lambda: writer.output_path("raw/alpha.jsonl"),
            lambda: writer.register_existing("raw/alpha.jsonl"),
            lambda: writer.register_routed_existing("train/alpha.jsonl", []),
        ):
            with pytest.raises(
                transaction.UnsafePathAPIError,
                match="secure.*callback|copy_file",
            ):
                call()
        _write_complete(writer, "compatibility", siblings=("alpha",))

    coordinator.publish(
        _plan("compatibility", siblings=("alpha",)),
        producer,
    )


def test_open_output_parent_swap_cannot_redirect_acquired_descriptor(tmp_path):
    coordinator = GenerationCoordinator(tmp_path / "corpus")
    outside = tmp_path / "callback-race-outside"
    outside.mkdir()

    def producer(writer):
        with writer.open_output("raw/alpha.jsonl") as output:
            assert isinstance(output.name, int)
            staging = coordinator.staging_directory / "callback-race.staging"
            parent = staging / "raw"
            parent.rename(staging / "raw-before-swap")
            parent.symlink_to(outside, target_is_directory=True)
            output.write(_line("alpha", "callback-race", 1))

    with pytest.raises(
        (InventoryError, ValidationError),
        match="symlink|replaced",
    ):
        coordinator.publish(
            _plan("callback-race", siblings=("alpha",)),
            producer,
        )
    assert not (outside / "alpha.jsonl").exists()


def test_missing_o_nofollow_fails_before_creating_root(tmp_path, monkeypatch):
    root = tmp_path / "no-nofollow"
    monkeypatch.delattr(transaction.os, "O_NOFOLLOW")

    with pytest.raises(
        transaction.PlatformCapabilityError,
        match="O_NOFOLLOW",
    ):
        GenerationCoordinator(root).publish(
            _plan("no-nofollow", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "no-nofollow",
                siblings=("alpha",),
            ),
        )
    assert not root.exists()


def test_missing_dirfd_support_fails_before_creating_root(tmp_path, monkeypatch):
    root = tmp_path / "no-dirfd"
    monkeypatch.setattr(
        transaction.os,
        "supports_dir_fd",
        transaction.os.supports_dir_fd - {transaction.os.mkdir},
    )

    with pytest.raises(
        transaction.PlatformCapabilityError,
        match="dir_fd.*mkdir",
    ):
        GenerationCoordinator(root).publish(
            _plan("no-dirfd", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "no-dirfd",
                siblings=("alpha",),
            ),
        )
    assert not root.exists()


def test_monkeypatched_open_without_dirfd_fails_before_root(tmp_path, monkeypatch):
    root = tmp_path / "monkeypatched-open"
    real_open = transaction.os.open

    def open_without_dirfd(path, flags, mode=0o777):
        return real_open(path, flags, mode)

    monkeypatch.setattr(transaction.os, "open", open_without_dirfd)
    with pytest.raises(
        transaction.PlatformCapabilityError,
        match="dir_fd.*open",
    ):
        GenerationCoordinator(root).publish(
            _plan("monkeypatched-open", siblings=("alpha",)),
            lambda writer: _write_complete(
                writer,
                "monkeypatched-open",
                siblings=("alpha",),
            ),
        )
    assert not root.exists()
