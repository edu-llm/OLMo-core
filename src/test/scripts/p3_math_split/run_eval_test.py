"""Family coverage and bounded-memory loss checks for the P3 evaluator."""

import hashlib
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from . import load_project_module

run_eval = load_project_module("run_eval")

P3_ROOT = Path("src/scripts/train/p3_math_split")
EVAL_RUN_EVAL = P3_ROOT / "evals" / "run_eval.py"
LEGACY_RUN_EVAL = P3_ROOT / "run_eval.py"


def test_run_eval_entrypoint_lives_under_evals_subfolder():
    assert EVAL_RUN_EVAL.is_file()
    assert not LEGACY_RUN_EVAL.exists()


def test_default_generation_budget_is_large_enough_for_secondary_proof_metrics():
    assert run_eval.DEFAULT_MAX_NEW_TOKENS == 8_192


def test_metamath_verifier_requires_the_corpus_snapshot(tmp_path):
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    files = {}
    for name in ("set.mm", "iset.mm", "nf.mm"):
        payload = name.encode()
        (mm_dir / name).write_bytes(payload)
        files[name] = {"sha256": hashlib.sha256(payload).hexdigest()}
    manifest = tmp_path / "metamath_sources.json"
    manifest.write_text(json.dumps({"commit": "abc", "files": files}))

    assert run_eval.verify_metamath_sources(mm_dir, manifest)["commit"] == "abc"

    (mm_dir / "set.mm").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="does not match corpus snapshot"):
        run_eval.verify_metamath_sources(mm_dir, manifest)


def test_context_filter_matches_training_eos_inclusive_length_rule():
    class CharTokenizer:
        def __call__(self, texts, *, add_special_tokens):
            assert add_special_tokens is False
            if isinstance(texts, str):
                return {"input_ids": list(range(len(texts)))}
            return {"input_ids": [list(range(len(text))) for text in texts]}

    rows = [
        {"id": "fits", "text": "1234"},
        {"id": "too-long", "text": "12345"},
    ]
    kept, excluded = run_eval.partition_context_eligible(
        rows,
        CharTokenizer(),
        context_length=5,
    )

    assert [row["id"] for row in kept] == ["fits"]
    assert excluded == [{"id": "too-long", "tokens_with_eos": 6}]


def test_generation_budget_uses_remaining_context_instead_of_skipping_row():
    assert run_eval.generation_budgets(
        [8, 14, 16],
        context_length=16,
        max_new_tokens=6,
    ) == {0: 6, 1: 2}


def test_rows_for_condition_returns_all_rows_for_every_condition():
    rows = [{"id": f"row-{index}"} for index in range(23)]

    for condition in run_eval.CONDITIONS:
        selected = run_eval.rows_for_condition(
            rows,
            family="mizar",
            condition=condition,
            seed=20260801,
        )
        reordered = run_eval.rows_for_condition(
            list(reversed(rows)),
            family="mizar",
            condition=condition,
            seed=20260801,
        )

        assert selected == rows
        assert selected == list(reversed(reordered))

    with pytest.raises(ValueError, match="unknown condition"):
        run_eval.rows_for_condition(
            rows,
            family="mizar",
            condition="facts_shuffled",
            seed=20260801,
        )


def test_corrupted_condition_never_keeps_the_original_statement():
    row = {"facts": {"f": "A"}, "goal": "G"}
    prompt = run_eval.build_prompt(
        row,
        "facts_corrupted",
        random.Random(1),
        ["A", "B"],
    )
    assert "f : B" in prompt
    assert "f : A" not in prompt


def test_metamath_prompt_keeps_local_assumptions_before_the_separator():
    row = {
        "facts": {"f": "A"},
        "local_assumptions": {"th.1": "|- ph"},
        "goal": "|- ps",
    }

    present = run_eval.build_prompt(
        row,
        "facts_present",
        random.Random(1),
        ["A", "B"],
    )
    absent = run_eval.build_prompt(
        row,
        "facts_absent",
        random.Random(1),
        ["A", "B"],
    )

    expected_local = "Local assumptions:\nth.1 : |- ph"
    assert expected_local in present
    assert present.index(expected_local) < present.index("\n---\nGOAL ")
    assert expected_local in absent, "fact interventions must not remove theorem givens"
    assert "f : A" not in absent


def test_old_metamath_boolean_api_is_quarantined():
    called = False

    class OldVerifier:
        @staticmethod
        def verify_proof(*args, **kwargs):
            nonlocal called
            called = True
            return SimpleNamespace(valid=True)

    status = run_eval.metamath_verifier_availability(OldVerifier)

    assert status["status"] == "unavailable"
    assert "tri-state" in status["reason"]
    assert "valid" not in status
    assert not called


def test_discovers_all_six_eval_families_and_shared_manifests(tmp_path):
    eval_dir = tmp_path / "eval"
    heldout_dir = tmp_path / "heldout"
    eval_dir.mkdir()
    heldout_dir.mkdir()
    for family in run_eval.FAMILIES:
        (eval_dir / f"{family}.jsonl").write_text("{}\n")
    for manifest in set(run_eval.HELDOUT_MANIFEST.values()):
        (heldout_dir / f"{manifest}.json").write_text(json.dumps({"facts": []}))

    assert run_eval.discover_families(tmp_path) == list(run_eval.FAMILIES)
    assert run_eval.HELDOUT_MANIFEST["prf2"] == run_eval.HELDOUT_MANIFEST["enigma"]
    assert run_eval.HELDOUT_MANIFEST["mizar"] == run_eval.HELDOUT_MANIFEST["thproofs"]


def test_target_chunks_cover_each_target_once_with_bounded_context():
    chunks = list(
        run_eval.iter_target_chunks(
            total_tokens=101,
            target_start=19,
            context_length=32,
            chunk_size=7,
        )
    )

    covered = [i for _, start, end in chunks for i in range(start, end)]
    assert covered == list(range(19, 101))
    for context_start, score_start, score_end in chunks:
        assert score_end - context_start <= 32
        assert context_start < score_start
        assert score_end - score_start <= 7


def test_chunked_teacher_forcing_scores_the_same_tokens_as_full_bigram_loss():
    vocab = 16

    class BigramModel:
        def __call__(self, *, input_ids, logits_to_keep, **kwargs):
            del kwargs
            kept = input_ids[:, -logits_to_keep:]
            logits = torch.zeros((*kept.shape, vocab))
            logits.scatter_(-1, ((kept + 1) % vocab).unsqueeze(-1), 4.0)
            return SimpleNamespace(logits=logits)

    ids = torch.arange(13) % vocab
    got_nll, got_tokens, got_correct = run_eval.chunked_sequence_nll(
        BigramModel(),
        ids,
        target_start=4,
        context_length=8,
        chunk_size=3,
        device="cpu",
    )

    logits = torch.zeros((len(ids) - 1, vocab))
    logits.scatter_(-1, ((ids[:-1] + 1) % vocab).unsqueeze(-1), 4.0)
    expected = torch.nn.functional.cross_entropy(
        logits[3:],
        ids[4:],
        reduction="sum",
    )
    assert got_tokens == len(ids) - 4
    assert got_correct == got_tokens
    assert got_nll == pytest.approx(float(expected))


def test_metamath_gold_gate_rejects_unsupplied_assumptions_and_reuse():
    base = {
        "facts": {"ax": "|- ph"},
        "target": "  1  ax       |- ph",
    }
    assert run_eval.gold_trace_uses_only_supplied_labels(base)

    assumption = {**base, "target": "  1  theorem.1  |- ph"}
    reuse = {**base, "target": "  1  (reuse)    |- ph"}
    assert not run_eval.gold_trace_uses_only_supplied_labels(assumption)
    assert not run_eval.gold_trace_uses_only_supplied_labels(reuse)


def test_target_metrics_score_exactly_one_eos_and_store_sufficient_statistics():
    vocab = 256
    eos_token_id = 255

    class CharTokenizer:
        def __init__(self):
            self.eos_token_id = eos_token_id

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    class LookaheadModelWithWrongEos:
        def __call__(self, *, input_ids, **kwargs):
            del kwargs
            logits = torch.full((*input_ids.shape, vocab), -20.0)
            for position in range(input_ids.shape[1] - 1):
                next_token = int(input_ids[0, position + 1])
                prediction = 0 if next_token == eos_token_id else next_token
                logits[0, position, prediction] = 20.0
            logits[0, -1, 0] = 20.0
            return SimpleNamespace(logits=logits)

    stats = run_eval.target_nll(
        LookaheadModelWithWrongEos(),
        CharTokenizer(),
        [{"id": "probe", "facts": {}, "goal": "G", "target": "xy"}],
        "facts_present",
        random.Random(1),
        [],
        128,
        16,
        "cpu",
    )

    item = stats["per_example"]["probe"]
    assert item["target_tokens"] == 3, "two content tokens plus exactly one EOS"
    assert item["target_correct"] == 2, "content is correct but EOS is deliberately wrong"
    assert item["nll_sum"] > 20
    assert item["target_nll_per_token"] == pytest.approx(item["nll_sum"] / 3)
    assert item["target_token_accuracy"] == pytest.approx(2 / 3)
    assert stats["target_tokens"] == 3
    assert stats["target_correct"] == 2
    assert stats["target_token_micro_accuracy"] == pytest.approx(2 / 3)
    assert stats["target_example_macro_accuracy"] == pytest.approx(2 / 3)
    assert stats["target_token_micro_nll_per_token"] == pytest.approx(item["target_nll_per_token"])
    assert stats["target_example_macro_nll_per_token"] == pytest.approx(
        item["target_nll_per_token"]
    )


def test_combined_suffix_plus_eos_drives_budget_and_three_row_denominators():
    class BoundaryTokenizer:
        eos_token_id = 99

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            encodings = {
                "P": [1],
                "T": [8],
                "PT": [1, 2, 3],
            }
            return {"input_ids": encodings[text]}

    prompt_ids, full_ids = run_eval.tokenize_target_with_eos(BoundaryTokenizer(), "P", "T")
    assert prompt_ids == [1]
    assert full_ids == [1, 2, 3, 99]
    assert len(full_ids) - len(prompt_ids) == 3
    assert not run_eval.target_fits_generation_budget(BoundaryTokenizer(), "P", "T", allowance=2)
    assert run_eval.target_fits_generation_budget(BoundaryTokenizer(), "P", "T", allowance=3)

    items = [
        {
            "generation_attempted": False,
            "whole_proof_budget_eligible": False,
            "exact_match": False,
        },
        {
            "generation_attempted": True,
            "whole_proof_budget_eligible": False,
            "exact_match": False,
        },
        {
            "generation_attempted": True,
            "whole_proof_budget_eligible": True,
            "exact_match": True,
        },
    ]
    summary = run_eval.summarize_generation(
        items,
        source_examples=5,
        context_eligible_examples=4,
    )
    assert summary["source_examples"] == 5
    assert summary["context_eligible_examples"] == 4
    assert summary["evaluated_examples"] == 3
    assert summary["generation_attempted_examples"] == 2
    assert summary["whole_proof_budget_eligible_examples"] == 1
    assert summary["exact_match_count_evaluated"] == 1
    assert summary["exact_match_rate_evaluated"] == pytest.approx(1 / 3)
    assert summary["exact_match_count_budget_eligible"] == 1
    assert summary["exact_match_rate_budget_eligible"] == 1
    assert "exact_match_rate_all" not in summary


def test_atp_v2_conditions_preserve_local_inputs_and_present_prefix_exactly():
    target = "  1  step       $false   [resolve local_only]"
    prefix = (
        "I know these mathematical statements:\n"
        "g1 : GLOBAL ONE\n"
        "g2 : GLOBAL TWO\n"
        "Local ATP inputs:\n"
        "local_only : LOCAL PREMISE\n"
        "---\n"
        "GOAL GOAL\n"
    )
    row = {
        "id": "atp",
        "schema_version": "atp-v2",
        "facts": {"g1": "GLOBAL ONE", "g2": "GLOBAL TWO"},
        "local_inputs": {"local_only": "LOCAL PREMISE"},
        "goal": "GOAL",
        "target": target,
        "text": prefix + target,
    }

    present = run_eval.materialize_condition(
        row, "facts_present", random.Random(1), ["GLOBAL ONE", "GLOBAL TWO", "OTHER"]
    )
    absent = run_eval.materialize_condition(
        row, "facts_absent", random.Random(1), ["GLOBAL ONE", "GLOBAL TWO", "OTHER"]
    )
    corrupted = run_eval.materialize_condition(
        row, "facts_corrupted", random.Random(1), ["GLOBAL ONE", "GLOBAL TWO", "OTHER"]
    )

    assert present.prompt == prefix
    for materialized in (present, absent, corrupted):
        assert "Local ATP inputs:\nlocal_only : LOCAL PREMISE" in materialized.prompt
        assert materialized.prompt.index("Local ATP inputs:") < materialized.prompt.index(
            "\n---\nGOAL "
        )
    assert absent.visible_facts == {}
    assert "g1 :" not in absent.prompt and "g2 :" not in absent.prompt
    assert corrupted.visible_facts.keys() == row["facts"].keys()
    assert all(corrupted.visible_facts[name] != row["facts"][name] for name in row["facts"])


def test_mizar_proof_v2_uses_exact_global_fact_prompt_without_derived_local_metadata():
    target = "assume A1: BaseGoal;\n  thus thesis by Base, A1;"
    prefix = (
        "I know these mathematical statements:\n"
        "SAMPLE:1 : BaseGoal\n"
        "---\n"
        "GOAL SharedGoal\n"
    )
    row = {
        "id": "real-shaped-direct-mizar",
        "schema_version": "mizar-proof-v2",
        "family": "mizar",
        "theorem": "SAMPLE:2",
        "facts": {"SAMPLE:1": "BaseGoal"},
        "cited": ["SAMPLE:1"],
        "proof_local_labels": ["A1"],
        "local_assumptions": {"A1": "BaseGoal"},
        "goal": "SharedGoal",
        "target": target,
        "text": prefix + target,
        "mask_start": 0,
        "mask_end": len(prefix.split("\n---\n", 1)[0]),
    }
    pool = ["BaseGoal", "OtherGoal"]

    present = run_eval.materialize_condition(row, "facts_present", random.Random(1), pool)
    absent = run_eval.materialize_condition(row, "facts_absent", random.Random(1), pool)
    corrupted = run_eval.materialize_condition(row, "facts_corrupted", random.Random(1), pool)

    assert present.prompt == prefix
    for materialized in (present, absent, corrupted):
        assert "Local assumptions:" not in materialized.prompt
        assert "A1 : BaseGoal" not in materialized.prompt
    assert "assume A1: BaseGoal;" in row["target"], "local context stays inside the target"


def test_metamath_conditions_do_not_promote_incomplete_api_to_validity():
    row = {
        "theorem": "set:th",
        "facts": {"ax": "|- ph"},
        "local_assumptions": {"th.1": "|- ps"},
        "goal": "|- ph",
        "target": "  1  ax  |- ph",
    }
    absent = run_eval.materialize_condition(
        row, "facts_absent", random.Random(1), ["|- ph", "|- ch"]
    )
    assert absent.visible_facts == {}

    corrupted = run_eval.materialize_condition(
        row, "facts_corrupted", random.Random(1), ["|- ph", "|- ch"]
    )
    assert not corrupted.metamath_validity_supported
    assert "visible corrupted statements" in corrupted.metamath_validity_reason

    status = run_eval.metamath_verifier_availability(run_eval.mm_verify)
    assert status["status"] == "unavailable"
    assert status["detected_schema"] is None


def test_fact_probe_uses_actual_train_visibility_and_reports_eos_budgets(monkeypatch):
    class CharTokenizer:
        eos_token_id = 255

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    rows = [
        {
            "facts": {
                "held": "A",
                "train": "BB",
                "eval": "C",
            }
        }
    ]
    gold_by_prompt = {
        f"{run_eval.HDR}\nheld :": "A",
        f"{run_eval.HDR}\ntrain :": "BB",
        f"{run_eval.HDR}\neval :": "C",
    }

    def fake_generate(model, tok, prompts, *args, **kwargs):
        del model, tok, args, kwargs
        return [gold_by_prompt[prompt] for prompt in prompts]

    monkeypatch.setattr(run_eval, "generate", fake_generate)
    args = SimpleNamespace(
        seed=1,
        probe_n=3,
        probe_max_new_tokens=2,
        batch_size=3,
        context_length=256,
    )
    result = run_eval.run_probe(
        object(),
        CharTokenizer(),
        rows,
        ["held"],
        args,
        "cpu",
        train_fact_names={"train"},
        train_visibility_available=True,
    )

    assert result["pool_counts"] == {
        "heldout": 1,
        "train_visible": 1,
        "eval_only": 1,
    }
    assert result["selected_counts"] == result["pool_counts"]
    assert result["evaluated_names"] == 3
    assert result["generation_attempted_names"] == 3
    assert result["generation_budget_eligible_names"] == 2
    assert result["generation_budget_ineligible_names"] == 1
    assert result["exact_match_count_evaluated"] == 3
    assert result["exact_match_count_budget_eligible"] == 2
    by_name = {item["name"]: item for item in result["items"]}
    assert by_name["held"]["visibility"] == "heldout"
    assert by_name["train"]["visibility"] == "train_visible"
    assert by_name["eval"]["visibility"] == "eval_only"
    assert by_name["train"]["target_tokens"] == 3, "BB plus EOS"
    assert not by_name["train"]["whole_statement_budget_eligible"]


class _SerializedBackend:
    def __init__(self, *, merges):
        self.merges = merges

    def to_str(self):
        return json.dumps(
            {
                "version": "1.0",
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {"type": "ByteLevel"},
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "<eos>": 1},
                    "merges": self.merges,
                },
                "post_processor": None,
                "decoder": {"type": "ByteLevel"},
            }
        )


class _SerializedTokenizer:
    eos_token_id = 1
    pad_token_id = 1
    special_tokens_map = {"eos_token": "<eos>"}

    def __init__(self, *, merges=("a b",)):
        self.backend_tokenizer = _SerializedBackend(merges=list(merges))

    @staticmethod
    def get_vocab():
        return {"a": 0, "<eos>": 1}


def _write_exported_model(
    model,
    *,
    arm="dense",
    architecture="Qwen2ForCausalLM",
    source_commit="c" * 40,
    checkpoint_step=run_eval.FINAL_CHECKPOINT_STEP,
):
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": str(model),
                "cache_dir": str(model / "cache"),
                "model_type": "qwen2",
                "architectures": [architecture],
                "tie_word_embeddings": True,
                "vocab_size": 151_936,
            }
        )
    )
    weight_path = model / "model.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, weight_path)
    trained_weight_files = {
        weight_path.name: {
            "sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
            "bytes": weight_path.stat().st_size,
            "dtype": "BF16",
        }
    }
    trained_weights_root_sha256 = hashlib.sha256(
        json.dumps(
            trained_weight_files,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (model / "model_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "p3-model-export-v1",
                "checkpoint_step": checkpoint_step,
                "arm": arm,
                "base_model_id": "Qwen/Qwen2.5-0.5B",
                "base_model_revision": "revision-abc",
                "initial_weights_sha256": "a" * 64,
                "source_commit": source_commit,
                "platform_run_manifest_id": "manifest-dense-123",
                "platform_run_manifest_sha256": "d" * 64,
                "trained_weight_files": trained_weight_files,
                "trained_weights_root_sha256": trained_weights_root_sha256,
            }
        )
    )


def test_cli_rejects_non_final_checkpoint_before_model_load(tmp_path, monkeypatch):
    model = tmp_path / "hf"
    _write_exported_model(model, arm="dense", checkpoint_step=24_540)

    transformers = ModuleType("transformers")

    class MustNotLoad:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError(
                "non-final checkpoint must fail before model or tokenizer load"
            )

    transformers.AutoModelForCausalLM = MustNotLoad
    transformers.AutoTokenizer = MustNotLoad
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--model",
            str(model),
            "--arm",
            "dense",
            "--families",
            "mizar",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )

    with pytest.raises(SystemExit, match=str(run_eval.FINAL_CHECKPOINT_STEP)):
        run_eval.main()


def test_validate_reportable_checkpoint_step_accepts_final_export():
    run_eval.validate_reportable_checkpoint_step(run_eval.FINAL_CHECKPOINT_STEP)


def test_validate_reportable_checkpoint_step_rejects_other_exports():
    with pytest.raises(ValueError, match=str(run_eval.FINAL_CHECKPOINT_STEP)):
        run_eval.validate_reportable_checkpoint_step(24_540)


def test_cli_rejects_arm_mismatch_before_model_load(tmp_path, monkeypatch):
    model = tmp_path / "hf"
    _write_exported_model(model, arm="dense")

    transformers = ModuleType("transformers")

    class MustNotLoad:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("arm mismatch must fail before model or tokenizer load")

    transformers.AutoModelForCausalLM = MustNotLoad
    transformers.AutoTokenizer = MustNotLoad
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--model",
            str(model),
            "--arm",
            "split",
            "--families",
            "mizar",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )

    with pytest.raises(SystemExit, match="exported arm.*dense.*--arm.*split"):
        run_eval.main()


def test_evaluation_metadata_requires_complete_exporter_provenance(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "eval").mkdir(parents=True)
    (corpus / "heldout").mkdir()
    (corpus / "shards").mkdir()
    (corpus / "eval" / "mizar.jsonl").write_text('{"id":"x"}\n')
    (corpus / "heldout" / "mizar.json").write_text('{"facts":[]}\n')
    mizar_train = corpus / "shards" / "mizar.jsonl"
    mizar_train.write_text('{"facts":{"mizar_fact":"M"}}\n')
    (corpus / "shards" / "isabelle.jsonl").write_text('{"facts":{"sibling_fact":"I"}}\n')

    model = tmp_path / "run" / "step24540" / "hf"
    checkpoint_step = run_eval.FINAL_CHECKPOINT_STEP
    _write_exported_model(model, checkpoint_step=checkpoint_step)

    args = SimpleNamespace(
        seed=20260801,
        conditions=["facts_present", "facts_absent"],
        sample=False,
        temperature=0.7,
        context_length=16_384,
        max_new_tokens=8_192,
        limit=None,
        nll_chunk_size=256,
    )
    first = run_eval.build_evaluation_metadata(
        args=args,
        tokenizer=_SerializedTokenizer(),
        corpus=corpus,
        families=["mizar"],
        model_path=model,
    )
    second = run_eval.build_evaluation_metadata(
        args=args,
        tokenizer=_SerializedTokenizer(),
        corpus=corpus,
        families=["mizar"],
        model_path=model,
    )

    assert first == second
    assert first["schema_version"] == run_eval.RESULT_SCHEMA_VERSION
    assert first["evaluation_controls"]["limit"] is None
    assert first["evaluation_controls"]["conditions"] == [
        "facts_present",
        "facts_absent",
    ]
    assert first["evaluation_controls"]["condition_cohort_policy"] == {
        condition: {"selection": "all-context-eligible-v1"}
        for condition in run_eval.CONDITIONS
    }
    assert first["input_provenance"]["tokenizer_sha256"]
    assert first["input_provenance"]["corpus_sha256"]
    assert first["input_provenance"]["eval_shard_sha256"]["mizar"]
    assert first["input_provenance"]["heldout_manifest_sha256"]["mizar"]
    assert first["input_provenance"]["train_shard_sha256"] == {
        "mizar": hashlib.sha256(mizar_train.read_bytes()).hexdigest(),
    }
    assert first["input_provenance"]["model"]["checkpoint_step"] == checkpoint_step
    assert first["input_provenance"]["model"]["base_model_id"] == "Qwen/Qwen2.5-0.5B"
    assert first["input_provenance"]["model"]["base_model_revision"] == "revision-abc"
    assert first["input_provenance"]["model"]["initial_weights_sha256"] == "a" * 64
    assert first["input_provenance"]["model"]["source_commit"] == "c" * 40
    assert first["input_provenance"]["model"]["platform_run_manifest_id"] == ("manifest-dense-123")
    assert first["input_provenance"]["model"]["platform_run_manifest_sha256"] == "d" * 64
    assert first["input_provenance"]["model"]["arm"] == "dense"
    assert first["input_provenance"]["model"]["semantic_config_sha256"]
    assert first["input_provenance"]["model"]["export_metadata_schema"] == (
        run_eval.MODEL_EXPORT_SCHEMA_VERSION
    )


def test_evaluator_refuses_missing_or_unknown_exporter_provenance(tmp_path):
    model = tmp_path / "hf"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "tie_word_embeddings": True,
                "vocab_size": 151_936,
            }
        )
    )

    with pytest.raises(RuntimeError, match="model export metadata"):
        run_eval.resolve_model_provenance(model)

    (model / "model_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "p3-model-export-v1",
                "checkpoint_step": None,
                "base_model_id": "Qwen/Qwen2.5-0.5B",
                "base_model_revision": "",
                "initial_weights_sha256": "",
            }
        )
    )
    with pytest.raises(RuntimeError, match="checkpoint_step"):
        run_eval.resolve_model_provenance(model)


@pytest.mark.parametrize("arm", [None, "", "both"])
def test_evaluator_requires_exported_dense_or_split_arm(tmp_path, arm):
    model = tmp_path / "hf"
    _write_exported_model(model)
    metadata_path = model / "model_provenance.json"
    metadata = json.loads(metadata_path.read_text())
    if arm is None:
        metadata.pop("arm")
    else:
        metadata["arm"] = arm
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="arm"):
        run_eval.resolve_model_provenance(model)


@pytest.mark.parametrize("source_commit", [None, ""])
def test_evaluator_requires_exported_source_commit(tmp_path, source_commit):
    model = tmp_path / "hf"
    _write_exported_model(model)
    metadata_path = model / "model_provenance.json"
    metadata = json.loads(metadata_path.read_text())
    if source_commit is None:
        metadata.pop("source_commit")
    else:
        metadata["source_commit"] = source_commit
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="source_commit"):
        run_eval.resolve_model_provenance(model)


def test_evaluator_binds_available_platform_manifest_identity(tmp_path):
    model = tmp_path / "hf"
    _write_exported_model(model)

    resolved = run_eval.resolve_model_provenance(model)

    assert resolved["platform_run_manifest_id"] == "manifest-dense-123"
    assert resolved["platform_run_manifest_sha256"] == "d" * 64

    metadata_path = model / "model_provenance.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["platform_run_manifest_sha256"] = "not-a-digest"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="platform run manifest"):
        run_eval.resolve_model_provenance(model)


@pytest.mark.parametrize("tamper", ["mutation", "missing", "replacement", "additional"])
def test_evaluator_rejects_hostile_weight_changes_before_model_load(tmp_path, monkeypatch, tamper):
    model = tmp_path / "hf"
    _write_exported_model(model)
    weight_path = model / "model.safetensors"
    if tamper == "mutation":
        weight_path.write_bytes(weight_path.read_bytes() + b"mutation")
    elif tamper == "missing":
        weight_path.unlink()
    elif tamper == "replacement":
        save_file({"replacement": torch.zeros(2, dtype=torch.bfloat16)}, weight_path)
    else:
        save_file(
            {"additional": torch.ones(1, dtype=torch.bfloat16)},
            model / "additional.safetensors",
        )

    transformers = ModuleType("transformers")

    class MustNotLoad:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("hostile weights must fail before model or tokenizer load")

    transformers.AutoModelForCausalLM = MustNotLoad
    transformers.AutoTokenizer = MustNotLoad
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_eval.py",
            "--model",
            str(model),
            "--arm",
            "dense",
            "--families",
            "mizar",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )

    with pytest.raises(SystemExit, match="trained weight"):
        run_eval.main()


def test_tokenizer_fingerprint_hashes_full_encoding_behavior():
    first = _SerializedTokenizer(merges=("a b",))
    second = _SerializedTokenizer(merges=("b a",))

    assert first.get_vocab() == second.get_vocab()
    assert first.special_tokens_map == second.special_tokens_map
    assert run_eval.tokenizer_sha256(first) != run_eval.tokenizer_sha256(second)


def test_semantic_model_config_hash_ignores_paths_but_retains_architecture():
    first = {
        "_name_or_path": "/exports/dense",
        "cache_dir": "/tmp/dense",
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "tie_word_embeddings": True,
        "vocab_size": 151_936,
    }
    path_only_change = {
        **first,
        "_name_or_path": "/exports/split",
        "cache_dir": "/tmp/split",
    }
    architecture_change = {
        **path_only_change,
        "architectures": ["DifferentArchitecture"],
    }

    assert run_eval.semantic_model_config_sha256(first) == (
        run_eval.semantic_model_config_sha256(path_only_change)
    )
    assert run_eval.semantic_model_config_sha256(first) != (
        run_eval.semantic_model_config_sha256(architecture_change)
    )


def test_load_family_and_target_nll_reject_duplicate_ids_before_scoring(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "heldout").mkdir()
    duplicate = {
        "id": "duplicate",
        "text": "text",
        "facts": {},
        "goal": "G",
        "target": "T",
    }
    (tmp_path / "eval" / "mizar.jsonl").write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n"
    )
    (tmp_path / "heldout" / "mizar.json").write_text('{"facts":[]}')

    with pytest.raises(ValueError, match="duplicate example ID 'duplicate'"):
        run_eval.load_family(tmp_path, "mizar")

    class NeverCalledModel:
        def __call__(self, **kwargs):
            del kwargs
            raise AssertionError("duplicate IDs must fail before model scoring")

    class CharTokenizer:
        eos_token_id = 255

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    with pytest.raises(ValueError, match="duplicate example ID 'duplicate'"):
        run_eval.target_nll(
            NeverCalledModel(),
            CharTokenizer(),
            [duplicate, duplicate],
            "facts_present",
            random.Random(1),
            [],
            128,
            16,
            "cpu",
        )


def test_train_visibility_scans_all_sibling_family_shards(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "isabelle.jsonl").write_text(json.dumps({"facts": {"isabelle_only": "I"}}) + "\n")
    (shards / "mizar.jsonl").write_text(json.dumps({"facts": {"sibling_only": "M"}}) + "\n")

    names, available = run_eval.load_train_fact_names(tmp_path)

    assert available
    assert names == {"isabelle_only", "sibling_only"}


def test_atp_zero_global_prompt_preserves_builder_blank_line():
    target = "  1  step       $false   [resolve local_only]"
    prefix = (
        "I know these mathematical statements:\n"
        "\n"
        "Local ATP inputs:\n"
        "local_only : LOCAL PREMISE\n"
        "---\n"
        "GOAL GOAL\n"
    )
    row = {
        "id": "atp-zero-global",
        "schema_version": "atp-v2",
        "facts": {},
        "local_inputs": {"local_only": "LOCAL PREMISE"},
        "goal": "GOAL",
        "target": target,
        "text": prefix + target,
    }

    present = run_eval.materialize_condition(row, "facts_present", random.Random(1), ["A", "B"])
    absent = run_eval.materialize_condition(row, "facts_absent", random.Random(1), ["A", "B"])

    assert present.prompt == prefix
    assert absent.prompt == prefix
    assert "\n\nLocal ATP inputs:" in absent.prompt


def test_isabelle_transition_v2_materializes_all_conditions_and_eos():
    target = "TACTIC\nby simp\nSTATE_AFTER\nstate two"
    goal = "THEOREM\nshows x\nSTATE_BEFORE\nstate one"
    prefix = (
        "I know these mathematical statements:\n"
        "a [Theory.one] : STATEMENT ONE\n"
        "a2 [Theory.one] : STATEMENT ONE\n"
        "b [Theory.two] : STATEMENT TWO\n"
        "Local assumptions:\n"
        "local [local.hyp] : LOCAL STATEMENT\n"
        "---\n"
        "GOAL\n"
        f"{goal}\n"
    )
    row = {
        "id": "isabelle-v2",
        "schema_version": "isabelle-transition-v2",
        "theorem": "Theory/0",
        "facts": {
            "Theory.one": "STATEMENT ONE",
            "Theory.two": "STATEMENT TWO",
        },
        "premise_aliases": {
            "a": "Theory.one",
            "a2": "Theory.one",
            "b": "Theory.two",
        },
        "local_assumptions": {"local": "LOCAL STATEMENT"},
        "local_names": {"local": "local.hyp"},
        "goal": goal,
        "target": target,
        "text": prefix + target,
    }
    pool = ["STATEMENT ONE", "STATEMENT TWO", "OTHER"]

    present = run_eval.materialize_condition(row, "facts_present", random.Random(1), pool)
    absent = run_eval.materialize_condition(row, "facts_absent", random.Random(1), pool)
    corrupted = run_eval.materialize_condition(row, "facts_corrupted", random.Random(1), pool)

    assert present.prompt == prefix
    assert "\n---\nGOAL\n" in present.prompt
    assert "\n---\nGOAL " not in present.prompt
    assert "a [Theory.one]" not in absent.prompt
    assert "b [Theory.two]" not in absent.prompt
    for materialized in (present, absent, corrupted):
        assert "local [local.hyp] : LOCAL STATEMENT" in materialized.prompt

    corrupted_lines = {
        line.split(" [", 1)[0]: line.rsplit(" : ", 1)[1]
        for line in corrupted.prompt.splitlines()
        if " [Theory." in line
    }
    assert corrupted_lines["a"] == corrupted_lines["a2"]
    assert corrupted_lines["a"] != "STATEMENT ONE"
    assert corrupted_lines["b"] != "STATEMENT TWO"

    class CharTokenizer:
        eos_token_id = 255

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    prompt_ids, full_ids = run_eval.tokenize_target_with_eos(
        CharTokenizer(), present.prompt, target
    )
    assert len(full_ids) - len(prompt_ids) == len(target) + 1
