"""Fail-closed contracts for P3 corpus token production."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

import tokenize_corpus as tokenization  # noqa: E402

FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
SOURCE_SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}
PRODUCER_PATH = (
    Path(__file__).resolve().parents[6]
    / "memorysplit-requery-exact"
    / "scripts"
    / "corpus_generation_transaction.py"
)
BUILD_P3_PATH = PRODUCER_PATH.with_name("build_p3_generation.py")


class _ContractTokenizer:
    is_fast = True
    eos_token_id = 250
    pad_token_id = 250

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [(ord(char) % 200) + 1 for char in text]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        result = {"input_ids": [self._encode(text) for text in texts]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                [(index, index + 1) for index in range(len(text))] for text in texts
            ]
        return result


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_producer():
    if not PRODUCER_PATH.is_file():
        pytest.skip(f"accepted transaction producer unavailable: {PRODUCER_PATH}")
    spec = importlib.util.spec_from_file_location("accepted_corpus_transaction_v2", PRODUCER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_build_p3_generation():
    if not BUILD_P3_PATH.is_file():
        pytest.skip(f"P3 generation builder unavailable: {BUILD_P3_PATH}")
    spec = importlib.util.spec_from_file_location("accepted_build_p3_generation", BUILD_P3_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_row(family: str, ordinal: int) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": SOURCE_SCHEMAS[family],
                "id": f"{family}-{ordinal}",
                "text": f"{family}-{ordinal}---\nGOAL target",
                "mask_end": len(family) + 2,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _real_publication(tmp_path: Path):
    producer = _load_producer()
    drop_validator = producer.JsonlValidator(
        schema_version="p3-drop-ledger/v1",
        required_fields=(
            "details",
            "drop_type",
            "occurrence_id",
            "raw_path",
            "raw_row",
            "raw_sha256",
            "sibling",
        ),
        require_generation_links=True,
    )
    heldout_validator = producer.JsonObjectValidator(
        schema_version="p3-heldout/v1",
        required_fields=("families",),
        require_generation_links=True,
    )
    outputs = []
    for family in FAMILIES:
        row_validator = producer.JsonlValidator(
            schema_version=SOURCE_SCHEMAS[family],
            required_fields=("id", "mask_end", "text"),
        )
        outputs.extend(
            (
                producer.OutputSpec(
                    path=f"raw/{family}.jsonl",
                    role=producer.OutputRole.RAW,
                    schema=row_validator.schema_version,
                    sibling=family,
                    validator=row_validator,
                ),
                producer.OutputSpec(
                    path=f"shards/{family}.jsonl",
                    role=producer.OutputRole.TRAIN,
                    schema=row_validator.schema_version,
                    sibling=family,
                    validator=row_validator,
                ),
                producer.OutputSpec(
                    path=f"eval/{family}.jsonl",
                    role=producer.OutputRole.EVAL,
                    schema=row_validator.schema_version,
                    sibling=family,
                    validator=row_validator,
                ),
                producer.OutputSpec(
                    path=f"sidecars/{family}.drops.jsonl",
                    role=producer.OutputRole.SIDECAR,
                    schema=drop_validator.schema_version,
                    sibling=family,
                    drop_types=("overlength",),
                    validator=drop_validator,
                ),
            )
        )
    outputs.append(
        producer.OutputSpec(
            path="heldout/families.json",
            role=producer.OutputRole.HELDOUT,
            schema=heldout_validator.schema_version,
            validator=heldout_validator,
        )
    )
    plan = producer.GenerationPlan(
        generation_id="p3-corpus-0001",
        source_generation_id="source-snapshot-1",
        requested_siblings=FAMILIES,
        outputs=tuple(outputs),
    )
    coordinator = producer.GenerationCoordinator(tmp_path / "corpus-transaction")

    def write(writer):
        for family in FAMILIES:
            raw = b"".join(_native_row(family, ordinal) for ordinal in range(1, 4))
            writer.write_bytes(f"raw/{family}.jsonl", raw)
            occurrences = writer.raw_occurrences(f"raw/{family}.jsonl")
            writer.write_routed_jsonl(f"shards/{family}.jsonl", (occurrences[0],))
            writer.write_routed_jsonl(f"eval/{family}.jsonl", (occurrences[1],))
            writer.write_drop_sidecar(
                f"sidecars/{family}.drops.jsonl",
                (
                    producer.DropRecord(
                        occurrence_id=occurrences[2].occurrence_id,
                        drop_type="overlength",
                        details={"tokens": 16_385},
                    ),
                ),
            )
        writer.write_linked_json("heldout/families.json", {"families": list(FAMILIES)})

    published = coordinator.publish(plan, write)
    return SimpleNamespace(
        producer=producer,
        coordinator=coordinator,
        plan=plan,
        published=published,
        root=coordinator.root,
        write=write,
    )


def _transaction_fixture(tmp_path: Path) -> Path:
    return _real_publication(tmp_path).root


def test_corpus_contract_resolves_exact_current_family_paths_and_hashes(tmp_path):
    publication = _real_publication(tmp_path)
    root = publication.root

    binding = tokenization.load_corpus_generation_contract(root)

    assert set(publication.published.manifest) == set(
        tokenization.CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["manifest_keys"]
    )
    assert binding["schema_version"] == "p3-tokenizer-corpus-binding-v1"
    assert binding["generation_id"] == "p3-corpus-0001"
    assert binding["logical_root_sha256"] == publication.published.logical_root_sha256
    assert binding["commit_state"] == "durable"
    assert binding["semantic_contract"] == tokenization.CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT
    assert binding["semantic_contract_sha256"] == tokenization.fingerprint_dict(
        binding["semantic_contract"]
    )
    assert (
        binding["producer_source_sha256"] == hashlib.sha256(PRODUCER_PATH.read_bytes()).hexdigest()
    )
    assert set(binding["families"]) == set(FAMILIES)
    for family in FAMILIES:
        for split in ("train", "val"):
            record = binding["families"][family][split]
            path = root / record["path"]
            assert record["sha256"] == _sha256(path)
            assert record["bytes"] == path.stat().st_size


def test_current_build_p3_generation_publication_reaches_tokenization_inputs(tmp_path, monkeypatch):
    builder = _load_build_p3_generation()
    work = tmp_path / "trusted-work"
    work.mkdir()
    published = builder.build_synthetic_generation(
        corpus_root=tmp_path / "corpus-transaction",
        work_root=work,
        generation_id="current-build-p3",
    )

    binding = tokenization.load_corpus_generation_contract(published.published.path.parents[1])

    for family in FAMILIES:
        train = binding["families"][family]["train"]
        val = binding["families"][family]["val"]
        assert train["generation_relative_path"] == f"shards/{family}.jsonl"
        assert val["generation_relative_path"] == f"eval/{family}.jsonl"
        assert train["schema"] == val["schema"] == SOURCE_SCHEMAS[family]
        assert (published.published.path.parents[1] / train["path"]).read_bytes()
        assert (published.published.path.parents[1] / val["path"]).read_bytes()

    tokenizer = _ContractTokenizer()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(identifier, **kwargs):
            assert identifier == "fixture-tokenizer"
            assert kwargs == {"local_files_only": True}
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(
        tokenization,
        "tokenizer_composite_seal",
        lambda *_args, **_kwargs: {
            "schema_version": "p3-tokenizer-four-part-seal-v1",
            "tokenizer_composite_sha256": "a" * 64,
        },
    )
    corpus_generation = {
        key: binding[key]
        for key in (
            "schema_version",
            "generation_id",
            "logical_root_sha256",
            "manifest_root_sha256",
            "manifest_file_sha256",
            "current_sha256",
            "semantic_contract_sha256",
            "producer_source_sha256",
        )
    }
    source_record = binding["families"]["metamath"]["train"]
    result = tokenization.process_corpus(
        {
            "name": "metamath",
            "source": str(
                publication_root := published.published.path.parents[1] / source_record["path"]
            ),
            "source_record": source_record,
            "output_root": str(tmp_path / "token-output"),
            "cache_root": str(tmp_path / "token-cache"),
            "split": "train",
            "tokenizer": "fixture-tokenizer",
            "sequence_length": 1024,
            "shard_tokens": 1024,
            "pack": True,
            "batch_size": 2,
            "suggest": False,
            "test_only": False,
            "defer_done_commit": True,
            "corpus_contract": binding,
            "corpus_generation": corpus_generation,
        }
    )
    assert publication_root.is_file()
    assert result["name"] == "metamath"
    assert result["build"]["source_jsonl"]["schema"] == "metamath-proof-v2"
    assert result["shards"][0]["path"].startswith("tokens/metamath/train-")


def test_data_only_cli_finalizes_train_and_val_with_fixed_qwen_tokenizer(
    tmp_path,
    monkeypatch,
):
    publication = _real_publication(tmp_path)
    tokenizer_root = PRODUCER_PATH.parents[1] / "tokenizers" / "qwen25-vendored"
    assert tokenizer_root.is_dir()
    output = tmp_path / "token-output"
    cache = tmp_path / "token-cache"
    common_args = [
        "tokenize_corpus.py",
        "--corpus-contract-root",
        str(publication.root),
        "--out",
        str(output),
        "--cache-dir",
        str(cache),
        "--tokenizer",
        str(tokenizer_root),
        "--sequence-length",
        "16384",
        "--shard-tokens",
        "16384",
        "--batch-size",
        "2",
        "--jobs",
        "1",
        "--pack",
    ]

    for split in ("train", "val"):
        monkeypatch.setattr(sys, "argv", [*common_args, "--split", split])
        tokenization.main()

    controls = []
    for split in ("train", "val"):
        meta_path = output / f"{split}_meta.json"
        controls.append(meta_path)
        manifest = json.loads(meta_path.read_text())
        assert manifest["corpus_generation"]["generation_id"] == publication.plan.generation_id
        assert set(manifest["groups"]) == set(FAMILIES)
        assert "evaluator_dependency" not in manifest
        for family in FAMILIES:
            done_path = output / "tokens" / family / f"{split}.done.json"
            controls.append(done_path)
            done = json.loads(done_path.read_text())
            assert "evaluator_dependency" not in done
            assert "evaluator_dependency" not in done["build"]
            assert done["build"]["corpus_generation"] == manifest["corpus_generation"]
            assert done["build"]["tokenizer"] == manifest["tokenizer_seal"]
            shard = output / done["shards"][0]["path"]
            assert shard == output / "tokens" / family / f"{split}-00000.u32le.bin"
            assert shard.stat().st_size == 16_384 * 4

    assert all("evaluator" not in path.read_text().lower() for path in controls)
    assert not any("evaluator" in path.as_posix().lower() for path in output.rglob("*"))
    assert len(list(output.rglob("*.done.json"))) == 12

    staged = tmp_path / "staged"
    for meta_path in (output / "train_meta.json", output / "val_meta.json"):
        manifest = json.loads(meta_path.read_text())
        for group in manifest["groups"].values():
            for shard in group["shards"]:
                source = output / shard["path"]
                destination = staged / shard["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    result = tokenization.validate_staged_token_payload(
        staged,
        train_manifest_path=output / "train_meta.json",
        val_manifest_path=output / "val_meta.json",
    )
    assert result["tokenizer"] == {
        "dataset_id": "tokenizer/qwen25-vendored",
        "version": "v1",
    }
    assert len(result["entries"]) == 12
    for split in ("train", "val"):
        manifest = json.loads((output / f"{split}_meta.json").read_text())
        assert manifest["tokenizer_seal"] == tokenization.FIXED_QWEN_TOKENIZER_SEAL


@pytest.mark.parametrize(
    "legacy_argument",
    (
        "--evaluator-release-root",
        "--evaluator-dependency",
        "--expected-evaluator-dataset-id",
        "--expected-evaluator-version",
        "--expected-evaluator-platform-group-manifest-sha256",
        "--expected-evaluator-manifest-root-sha256",
        "--expected-evaluator-seal-sha256",
    ),
)
def test_cli_rejects_legacy_evaluator_arguments(
    tmp_path,
    monkeypatch,
    capsys,
    legacy_argument,
):
    publication = _real_publication(tmp_path)
    tokenizer_root = PRODUCER_PATH.parents[1] / "tokenizers" / "qwen25-vendored"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize_corpus.py",
            "--corpus-contract-root",
            str(publication.root),
            "--tokenizer",
            str(tokenizer_root),
            legacy_argument,
            "legacy-value",
        ],
    )

    with pytest.raises(SystemExit):
        tokenization.main()

    error = capsys.readouterr().err
    assert "unrecognized arguments" in error
    assert legacy_argument in error


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda outputs: outputs.__setitem__(
                0,
                {**outputs[0], "role": "sidecar"},
            ),
            "role|train",
        ),
        (
            lambda outputs: outputs.__setitem__(
                0,
                {
                    **outputs[0],
                    "sibling": "isabelle",
                    "schema": SOURCE_SCHEMAS["isabelle"],
                },
            ),
            "sibling|duplicate",
        ),
        (
            lambda outputs: outputs.__setitem__(
                0,
                {**outputs[0], "schema": "metamath-proof-v1"},
            ),
            "schema",
        ),
        (
            lambda outputs: outputs.append({**outputs[0]}),
            "duplicate|path",
        ),
    ),
)
def test_source_selection_rejects_wrong_role_sibling_schema_and_path_duplication(
    tmp_path, mutation, message
):
    publication = _real_publication(tmp_path)
    outputs = [dict(entry) for entry in publication.published.manifest["outputs"]]
    mutation(outputs)

    with pytest.raises(RuntimeError, match=message):
        tokenization._select_p3_source_records(
            outputs,
            generation_id=publication.published.generation_id,
        )


def test_equivalent_producer_source_is_audit_metadata_not_compatibility_gate(tmp_path):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    binding["producer_source_sha256"] = "0" * 64

    tokenization.require_corpus_generation_current(binding)

    incompatible = dict(binding)
    incompatible["semantic_contract_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="semantic"):
        tokenization.require_corpus_generation_current(incompatible)


def _second_publication(publication, *, generation_id: str, entered=None):
    producer = publication.producer
    plan = producer.GenerationPlan(
        generation_id=generation_id,
        source_generation_id=publication.plan.source_generation_id,
        requested_siblings=publication.plan.requested_siblings,
        outputs=publication.plan.outputs,
    )

    def write(writer):
        if entered is not None:
            entered.set()
        publication.write(writer)

    return publication.coordinator.publish(plan, write)


def _staged_token_manifests(tmp_path: Path):
    output = tmp_path / "token-output"
    group = output / "tokens" / "enigma"
    group.mkdir(parents=True)
    pending_done = group / ".train.done.json.pending"
    pending_meta = output / ".train_meta.json.pending"
    pending_done.write_text('{"kind":"done"}\n')
    pending_meta.write_text('{"kind":"meta"}\n')
    return (
        (pending_done, group / "train.done.json"),
        (pending_meta, output / "train_meta.json"),
    )


@pytest.mark.parametrize("split", ("train", "val"))
@pytest.mark.parametrize("target", ("done", "meta"))
@pytest.mark.parametrize(
    "failure_phase",
    (
        "stage_write_before",
        "stage_write_after",
        "replace_before",
        "replace_after",
        "directory_fsync_before",
        "directory_fsync_after",
    ),
)
def test_manifest_faults_cleanup_pending_preserve_payloads_and_retry(
    tmp_path,
    split,
    target,
    failure_phase,
):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    output = tmp_path / "token-output"
    group = output / "tokens" / "enigma"
    group.mkdir(parents=True)
    done = group / f"{split}.done.json"
    meta = output / f"{split}_meta.json"
    specifications = (
        (
            done.with_name(f".{done.name}.pending"),
            done,
            {"kind": "done", "split": split},
        ),
        (
            meta.with_name(f".{meta.name}.pending"),
            meta,
            {"kind": "meta", "split": split},
        ),
    )
    shard = group / f"{split}-00000.u32le.bin"
    shard.write_bytes(b"sealed-shard")
    cache = tmp_path / "token-cache" / split / "tokens.u32le.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"sealed-cache")
    injected = []

    def fault(phase, _pending, final):
        kind = "done" if final.name.endswith(".done.json") else "meta"
        if phase == failure_phase and kind == target:
            injected.append((phase, kind))
            raise OSError(f"injected {phase} {kind}")

    with pytest.raises(Exception) as captured:
        tokenization.finalize_token_manifests_under_generation_lock(
            binding,
            specifications,
            _fault=fault,
        )

    assert injected == [(failure_phase, target)]
    assert not list(output.rglob("*.pending"))
    assert shard.read_bytes() == b"sealed-shard"
    assert cache.read_bytes() == b"sealed-cache"
    committed = [final for _, final, _ in specifications if final.exists()]
    if committed:
        assert isinstance(
            captured.value,
            tokenization.TokenManifestCommitUncertainError,
        )
        assert set(captured.value.committed_paths) == set(committed)
    else:
        assert not isinstance(
            captured.value,
            tokenization.TokenManifestCommitUncertainError,
        )

    tokenization.finalize_token_manifests_under_generation_lock(
        binding,
        specifications,
    )

    assert not list(output.rglob("*.pending"))
    assert json.loads(done.read_text()) == {"kind": "done", "split": split}
    assert json.loads(meta.read_text()) == {"kind": "meta", "split": split}
    assert shard.read_bytes() == b"sealed-shard"
    assert cache.read_bytes() == b"sealed-cache"


def test_generation_switch_before_shared_commit_refuses_without_final_manifests(tmp_path):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    replacements = _staged_token_manifests(tmp_path)
    shard = replacements[0][0].parent / "train-00000.u32le.bin"
    shard.write_bytes(b"sealed-shard")
    cache = tmp_path / "token-cache" / "tokens.u32le.bin"
    cache.parent.mkdir()
    cache.write_bytes(b"sealed-cache")

    def switch_before_lock():
        _second_publication(publication, generation_id="p3-corpus-0002")

    with pytest.raises(RuntimeError, match="CURRENT.*changed|generation"):
        tokenization.commit_token_manifests_under_generation_lock(
            binding,
            replacements,
            _before_lock=switch_before_lock,
        )

    assert all(not pending.exists() and not final.exists() for pending, final in replacements)
    assert shard.read_bytes() == b"sealed-shard"
    assert cache.read_bytes() == b"sealed-cache"


def test_shared_token_commit_blocks_real_generation_coordinator_exclusive_switch(tmp_path):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    replacements = _staged_token_manifests(tmp_path)
    attempted = threading.Event()
    publisher_entered = threading.Event()
    events = []
    thread = None

    def publish():
        events.append("publisher-attempted")
        attempted.set()
        _second_publication(
            publication,
            generation_id="p3-corpus-0002",
            entered=publisher_entered,
        )
        events.append("publisher-complete")

    def on_locked():
        nonlocal thread
        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        assert attempted.wait(timeout=5)

    def on_committed():
        assert not publisher_entered.is_set()
        events.append("token-committed")

    tokenization.commit_token_manifests_under_generation_lock(
        binding,
        replacements,
        _on_locked=on_locked,
        _on_committed=on_committed,
    )
    assert thread is not None
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert publisher_entered.is_set()
    assert events == [
        "publisher-attempted",
        "token-committed",
        "publisher-complete",
    ]
    assert all(not pending.exists() and final.is_file() for pending, final in replacements)


def test_shared_lock_refuses_intermediate_symlink_component(tmp_path):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(publication.root.parent, target_is_directory=True)
    binding["contract_root"] = str(alias_parent / publication.root.name)
    replacements = _staged_token_manifests(tmp_path)

    with pytest.raises(RuntimeError, match="symlink|unsafe"):
        tokenization.commit_token_manifests_under_generation_lock(
            binding,
            replacements,
        )

    assert all(not pending.exists() and not final.exists() for pending, final in replacements)


def test_corpus_contract_refuses_changed_current_after_resolution(tmp_path):
    root = _transaction_fixture(tmp_path)
    binding = tokenization.load_corpus_generation_contract(root)
    current = json.loads((root / "CURRENT").read_text())
    current["generation_id"] = "p3-corpus-0002"
    _write_json(root / "CURRENT", current)

    with pytest.raises(RuntimeError, match="CURRENT.*changed|generation"):
        tokenization.require_corpus_generation_current(binding)


def _resign_publication(publication, manifest, *, recompute_logical=False):
    producer = publication.producer
    if recompute_logical:
        manifest["logical_root_sha256"] = producer._logical_root(
            publication.plan,
            manifest["outputs"],
            manifest["routes"],
        )
    body = dict(manifest)
    body.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = hashlib.sha256(
        producer._canonical_json_bytes(body)
    ).hexdigest()
    manifest_path = publication.published.path / producer.MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(producer._canonical_json_bytes(manifest))
    manifest_path.chmod(0o444)
    manifest_sha = _sha256(manifest_path)

    current = {
        "generation_id": publication.plan.generation_id,
        "logical_root_sha256": manifest["logical_root_sha256"],
        "manifest_sha256": manifest_sha,
        "schema_version": producer.CURRENT_SCHEMA_VERSION,
    }
    (publication.root / producer.CURRENT_FILENAME).write_bytes(
        producer._canonical_json_bytes(current)
    )
    journal = publication.coordinator._committed_transaction_path(publication.plan.generation_id)
    record = {
        **current,
        "schema_version": producer.TRANSACTION_STATE_SCHEMA_VERSION,
        "state": "committed",
    }
    journal.write_bytes(producer._canonical_json_bytes(record))


@pytest.mark.parametrize(
    "defect",
    (
        "accounting",
        "route",
        "logical_root",
        "writable_mode",
        "directory",
        "validator",
        "plan",
        "journal",
        "symlink",
    ),
)
def test_exact_v2_consumer_rejects_hostile_publications(tmp_path, defect):
    publication = _real_publication(tmp_path)
    producer = publication.producer
    generation = publication.published.path
    manifest = json.loads((generation / producer.MANIFEST_FILENAME).read_text())

    if defect == "accounting":
        manifest["accounting"]["siblings"]["enigma"]["train_rows"] += 1
        _resign_publication(publication, manifest)
    elif defect == "route":
        routes_path = generation / producer.ROUTES_FILENAME
        routes_path.chmod(0o644)
        routes = [json.loads(line) for line in routes_path.read_text().splitlines()]
        routes[0]["destination_row"] = 2
        routes_path.write_bytes(b"".join(producer._canonical_json_bytes(row) for row in routes))
        routes_path.chmod(0o444)
        route_sha = _sha256(routes_path)
        manifest["routes"].update(
            bytes=routes_path.stat().st_size,
            root_sha256=route_sha,
            rows=len(routes),
            sha256=route_sha,
        )
        _resign_publication(publication, manifest, recompute_logical=True)
    elif defect == "logical_root":
        manifest["logical_root_sha256"] = "f" * 64
        _resign_publication(publication, manifest)
    elif defect == "writable_mode":
        (generation / "shards/enigma.jsonl").chmod(0o644)
    elif defect == "directory":
        generation.chmod(0o755)
        (generation / "unknown-empty-directory").mkdir(mode=0o555)
        generation.chmod(0o555)
    elif defect == "validator":
        manifest["outputs"][0]["validator"]["allow_empty"] = 0
        _resign_publication(publication, manifest)
    elif defect == "plan":
        manifest["plan_root_sha256"] = "e" * 64
        _resign_publication(publication, manifest)
    elif defect == "journal":
        publication.coordinator._committed_transaction_path(publication.plan.generation_id).unlink()
    else:
        destination = generation / "shards/enigma.jsonl"
        target = generation / "eval/enigma.jsonl"
        destination.parent.chmod(0o755)
        destination.unlink()
        destination.symlink_to(target)
        destination.parent.chmod(0o555)

    with pytest.raises(RuntimeError):
        tokenization.load_corpus_generation_contract(publication.root)


def test_exact_v2_consumer_rehashes_source_on_current_recheck(tmp_path):
    publication = _real_publication(tmp_path)
    binding = tokenization.load_corpus_generation_contract(publication.root)
    source = publication.published.path / "shards/enigma.jsonl"
    original = source.read_bytes()
    source.chmod(0o644)
    source.write_bytes(original.replace(b"enigma", b"stale!", 1))
    source.chmod(0o444)
    assert source.stat().st_size == len(original)

    with pytest.raises(RuntimeError, match="SHA-256|digest"):
        tokenization.require_corpus_generation_current(binding)


def test_legacy_shards_directory_is_not_a_production_contract(tmp_path):
    legacy = tmp_path / "legacy"
    (legacy / "shards").mkdir(parents=True)
    (legacy / "shards" / "enigma.jsonl").write_text("{}\n")

    with pytest.raises(RuntimeError, match="CURRENT|contract|legacy"):
        tokenization.load_corpus_generation_contract(legacy)


def test_token_producer_exports_no_legacy_evaluator_adapter():
    assert not hasattr(tokenization, "load_evaluator_dependency")
    assert not any(name.startswith("EVALUATOR_") for name in vars(tokenization))
