"""Score a trained arm on all six held-out formal-proof families.

Every family receives target-token NLL, teacher-forced next-token match, and
whitespace-normalized exact whole-output match. Metamath validity is deliberately
unavailable until a versioned sound tri-state verifier API is integrated. The legacy
boolean-compatible API is never reported as proof validity.

The headline condition is ``facts_present``: both arms get the same correct theorem
statements in context, and every example cites at least one family-held-out fact.
That is the question — does the split model match or beat dense when both can read
the facts.

The other three conditions exist because a bare win is ambiguous, and each of them
rules something out:

  facts_present    the real setting. Correct facts, correct order.
  facts_absent     header, no statements. Isolates what each arm stored in weights.
                   Dense should degrade less here if it memorised; split should
                   degrade more. If neither degrades, the facts were never needed and
                   the whole comparison is measuring something else.
  facts_corrupted  names kept, statements swapped between examples. A model that reads
                   its context should collapse. A model that ignores the block and
                   recites from memory will not — which is how you tell the two apart.
  facts_shuffled   block order permuted. Should be a no-op for both arms. If it is not,
                   the model is keying on position rather than content and every other
                   number here needs re-reading.

There is also a direct memorisation probe (--probe): given a fact name alone, state the
fact. It measures the thing the training manipulation is supposed to change, without
routing through proof search.

Generation runs through HuggingFace, so the trained OLMo-core checkpoint is exported
first. Greedy is the default because sampling noise is a comparison confound.
The evaluator applies the same ``text + EOS <= 16,384`` eligibility gate as
training, then teacher-forces bounded logits chunks without losing prompt context.

    python src/scripts/train/p3_math_split/run_eval.py \
      --model runs/split/hf --arm split --corpus corpus \
      --families metamath mizar prf2 --mm-dir /tmp/dscount/mm \
      --out results/split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mm_verify

norm = mm_verify.norm

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local assumptions:"
ATP_LOCAL_HDR = "Local ATP inputs:"
SEP = "---"
CONDITIONS = ("facts_present", "facts_absent", "facts_corrupted", "facts_shuffled")
FAMILIES = ("enigma", "isabelle", "metamath", "mizar", "prf2", "thproofs")
DEFAULT_MAX_NEW_TOKENS = 8_192
RESULT_SCHEMA_VERSION = "p3-eval-v3"
MODEL_EXPORT_SCHEMA_VERSION = "p3-model-export-v1"
SAFETENSORS_SHARD = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
SAFETENSORS_INDEX = "model.safetensors.index.json"
METAMATH_VERIFIER_SCHEMA_VERSION = "p3-metamath-tristate-v1"
NLL_CONTEXT_POLICY = "bounded_sliding_window_preserve_predecessor"
NLL_TARGET_POLICY = "combined_prompt_target_suffix_plus_single_eos"
CORRUPTED_METAMATH_REASON = (
    "validity against visible corrupted statements is unsupported; the current "
    "checker can instantiate only canonical database rules, so using those hidden "
    "canonical statements would not verify the condition the model actually saw"
)
HELDOUT_MANIFEST = {
    "enigma": "atp",
    "prf2": "atp",
    "isabelle": "isabelle",
    "metamath": "metamath",
    "mizar": "mizar",
    "thproofs": "mizar",
}


def discover_families(corpus: str | Path) -> list[str]:
    """Every requested family with a raw held-out JSONL shard."""
    eval_dir = Path(corpus) / "eval"
    return [family for family in FAMILIES if (eval_dir / f"{family}.jsonl").exists()]


def load_family(corpus: str | Path, family: str) -> tuple[list[dict], list[str]]:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    corpus = Path(corpus)
    rows = load_jsonl(corpus / "eval" / f"{family}.jsonl")
    require_unique_ids(rows, f"eval/{family}.jsonl")
    heldout_path = corpus / "heldout" / f"{HELDOUT_MANIFEST[family]}.json"
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))["facts"]
    return rows, heldout


def require_unique_ids(rows: list[dict], context: str) -> None:
    """Reject missing or duplicate IDs before any per-example mapping or scoring."""
    seen = set()
    for row in rows:
        example_id = row.get("id")
        if example_id is None:
            raise ValueError(f"{context}: example ID is missing")
        if example_id in seen:
            raise ValueError(f"{context}: duplicate example ID {example_id!r}")
        seen.add(example_id)


def partition_context_eligible(
    rows: list[dict],
    tok,
    *,
    context_length: int,
    batch_size: int = 256,
) -> tuple[list[dict], list[dict]]:
    """Apply the tokenizer's training-time ``document + EOS`` length gate."""
    if context_length < 1:
        raise ValueError("context_length must be positive")
    kept = []
    excluded = []
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo : lo + batch_size]
        encoded = tok(
            [row["text"] for row in chunk],
            add_special_tokens=False,
        )["input_ids"]
        for row, ids in zip(chunk, encoded):
            tokens_with_eos = len(ids) + 1
            if tokens_with_eos <= context_length:
                kept.append(row)
            else:
                excluded.append({"id": row["id"], "tokens_with_eos": tokens_with_eos})
    return kept, excluded


def generation_budgets(
    prompt_lengths: list[int],
    *,
    context_length: int,
    max_new_tokens: int,
) -> dict[int, int]:
    """Per-example generation allowance without crossing the model window."""
    return {
        i: min(max_new_tokens, context_length - prompt_length)
        for i, prompt_length in enumerate(prompt_lengths)
        if prompt_length < context_length
    }


def tokenize_target_with_eos(tok, prompt: str, target: str) -> tuple[list[int], list[int]]:
    """Tokenize the actual prompt/target boundary and append one evaluator EOS."""
    if tok.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    full_ids = list(tok(prompt + target, add_special_tokens=False)["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("prompt tokenization is not a prefix of prompt+target")
    full_ids.append(int(tok.eos_token_id))
    return prompt_ids, full_ids


def target_fits_generation_budget(
    tok,
    prompt: str,
    target: str,
    *,
    allowance: int,
) -> bool:
    """Whether the actual combined-tokenization target suffix and EOS fit."""
    prompt_ids, full_ids = tokenize_target_with_eos(tok, prompt, target)
    return len(full_ids) - len(prompt_ids) <= allowance


def summarize_generation(
    items: list[dict],
    *,
    source_examples: int,
    context_eligible_examples: int,
) -> dict:
    """Summarize every generation denominator without an ambiguous ``all``."""
    evaluated = len(items)
    attempted = sum(bool(item["generation_attempted"]) for item in items)
    eligible_items = [item for item in items if item["whole_proof_budget_eligible"]]
    exact_evaluated = sum(bool(item["exact_match"]) for item in items)
    exact_eligible = sum(bool(item["exact_match"]) for item in eligible_items)
    budget_eligible = len(eligible_items)
    return {
        "source_examples": source_examples,
        "context_eligible_examples": context_eligible_examples,
        "evaluated_examples": evaluated,
        "generation_attempted_examples": attempted,
        "whole_proof_budget_eligible_examples": budget_eligible,
        "whole_proof_budget_ineligible_examples": evaluated - budget_eligible,
        "whole_proof_budget_coverage_evaluated": (
            budget_eligible / evaluated if evaluated else None
        ),
        "exact_match_count_evaluated": exact_evaluated,
        "exact_match_rate_evaluated": (exact_evaluated / evaluated if evaluated else None),
        "exact_match_count_budget_eligible": exact_eligible,
        "exact_match_rate_budget_eligible": (
            exact_eligible / budget_eligible if budget_eligible else None
        ),
    }


def iter_target_chunks(
    *,
    total_tokens: int,
    target_start: int,
    context_length: int,
    chunk_size: int,
):
    """Yield ``(context_start, score_start, score_end)`` without losing a target.

    Every scored token retains its immediate predecessor, while the model input
    never exceeds the context length used for training.
    """
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if chunk_size < 1 or chunk_size >= context_length:
        raise ValueError("chunk_size must be positive and smaller than context_length")
    if target_start < 1 or target_start > total_tokens:
        raise ValueError("target_start must retain at least one prompt token")
    for score_start in range(target_start, total_tokens, chunk_size):
        score_end = min(score_start + chunk_size, total_tokens)
        context_start = max(0, score_end - context_length)
        if context_start >= score_start:  # defensive for future chunking changes
            context_start = score_start - 1
        yield context_start, score_start, score_end


def chunked_sequence_nll(
    model,
    input_ids,
    *,
    target_start: int,
    context_length: int,
    chunk_size: int,
    device,
) -> tuple[float, int, int]:
    """Teacher-force one sequence with bounded input and logits allocations.

    Qwen's ``logits_to_keep`` computes vocabulary logits only for the new target
    chunk plus its predecessor. Inputs longer than ``context_length`` use a
    sliding context, matching the maximum context the model saw in training.
    """
    import torch

    input_ids = input_ids.to(device=device, dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    for context_start, score_start, score_end in iter_target_chunks(
        total_tokens=len(input_ids),
        target_start=target_start,
        context_length=context_length,
        chunk_size=chunk_size,
    ):
        window = input_ids[context_start:score_end].unsqueeze(0)
        n_score = score_end - score_start
        with torch.no_grad():
            output = model(
                input_ids=window,
                attention_mask=torch.ones_like(window),
                use_cache=False,
                logits_to_keep=n_score + 1,
            )
        prediction_logits = output.logits[:, -(n_score + 1) : -1].float()
        labels = input_ids[score_start:score_end].unsqueeze(0)
        nll = torch.nn.functional.cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll)
        total_tokens += n_score
        total_correct += int((prediction_logits.argmax(dim=-1) == labels).sum().item())
    return total_nll, total_tokens, total_correct


@dataclass
class ConditionMaterialization:
    """The exact context and diagnostics visible under one intervention."""

    prompt: str
    visible_facts: dict[str, str]
    shuffle_eligible: bool
    shuffle_changed: bool
    metamath_validity_supported: bool
    metamath_validity_reason: str | None


def _corrupt_statement(statement: str, corrupt_pool, rng: random.Random) -> str:
    candidates = [value for value in dict.fromkeys(corrupt_pool) if value != statement]
    if not candidates:
        raise ValueError("facts_corrupted requires a distinct replacement statement")
    return rng.choice(candidates)


def _shuffle_nonidentity(items: list, rng: random.Random) -> tuple[list, bool, bool]:
    shuffled = list(items)
    eligible = len(shuffled) >= 2
    if eligible:
        rng.shuffle(shuffled)
        if shuffled == items:
            shuffled = shuffled[1:] + shuffled[:1]
    return shuffled, eligible, shuffled != items


def _expected_prompt_prefix(row: dict) -> str:
    target = row["target"]
    return row["text"][: -len(target)] if target else row["text"]


def _materialize_isabelle_condition(
    row: dict,
    condition: str,
    rng: random.Random,
    corrupt_pool,
) -> tuple[str, dict[str, str], bool, bool]:
    facts = row["facts"]
    aliases = row["premise_aliases"]
    original_lines = []
    for alias in sorted(aliases):
        qualified = aliases[alias]
        if qualified not in facts:
            raise ValueError(
                f"{row.get('id', '<unknown>')}: premise alias {alias!r} references "
                f"missing fact {qualified!r}"
            )
        original_lines.append((alias, qualified, facts[qualified]))

    visible_lines = list(original_lines)
    shuffle_eligible = False
    shuffle_changed = False
    if condition == "facts_absent":
        visible_lines = []
    elif condition == "facts_shuffled":
        visible_lines, shuffle_eligible, shuffle_changed = _shuffle_nonidentity(original_lines, rng)
    elif condition == "facts_corrupted":
        replacements = {
            qualified: _corrupt_statement(statement, corrupt_pool, rng)
            for qualified, statement in facts.items()
        }
        visible_lines = [
            (alias, qualified, replacements[qualified]) for alias, qualified, _ in original_lines
        ]

    lines = [HDR]
    lines.extend(
        f"{alias} [{qualified}] : {statement}" for alias, qualified, statement in visible_lines
    )
    local_assumptions = row.get("local_assumptions") or {}
    local_names = row.get("local_names") or {}
    if local_assumptions:
        lines.append(LOCAL_HDR)
        for alias in sorted(local_assumptions):
            local_name = local_names.get(alias)
            if not local_name:
                raise ValueError(
                    f"{row.get('id', '<unknown>')}: local alias {alias!r} has no local name"
                )
            lines.append(f"{alias} [{local_name}] : {local_assumptions[alias]}")
    block = "\n".join(lines)
    prompt = f"{block}\n{SEP}\nGOAL\n{row['goal']}\n"
    visible_facts = {qualified: statement for _, qualified, statement in visible_lines}
    return prompt, visible_facts, shuffle_eligible, shuffle_changed


def materialize_condition(
    row: dict,
    condition: str,
    rng: random.Random,
    corrupt_pool,
) -> ConditionMaterialization:
    """Render one condition while preserving theorem-local context."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")

    schema_version = row.get("schema_version")
    if schema_version == "isabelle-transition-v2":
        prompt, visible_facts, shuffle_eligible, shuffle_changed = _materialize_isabelle_condition(
            row, condition, rng, corrupt_pool
        )
        validity_supported = condition != "facts_corrupted"
        validity_reason = CORRUPTED_METAMATH_REASON if condition == "facts_corrupted" else None
        if condition == "facts_present" and prompt != _expected_prompt_prefix(row):
            raise RuntimeError(
                f"{row.get('id', '<unknown>')}: facts_present does not reconstruct "
                "the isabelle-transition-v2 prompt prefix"
            )
        return ConditionMaterialization(
            prompt=prompt,
            visible_facts=visible_facts,
            shuffle_eligible=shuffle_eligible,
            shuffle_changed=shuffle_changed,
            metamath_validity_supported=validity_supported,
            metamath_validity_reason=validity_reason,
        )

    original_items = list(row["facts"].items())
    visible_items = list(original_items)
    shuffle_eligible = False
    shuffle_changed = False
    validity_supported = True
    validity_reason = None

    if condition == "facts_absent":
        visible_items = []
    elif condition == "facts_shuffled":
        visible_items, shuffle_eligible, shuffle_changed = _shuffle_nonidentity(original_items, rng)
    elif condition == "facts_corrupted":
        visible_items = [
            (name, _corrupt_statement(statement, corrupt_pool, rng))
            for name, statement in original_items
        ]
        validity_supported = False
        validity_reason = CORRUPTED_METAMATH_REASON

    visible_facts = dict(visible_items)
    if schema_version == "atp-v2":
        block = HDR + "\n" + "\n".join(f"{name} : {stmt}" for name, stmt in visible_items)
        local_inputs = row.get("local_inputs")
        if local_inputs:
            block += (
                "\n"
                + ATP_LOCAL_HDR
                + "\n"
                + "\n".join(f"{name} : {stmt}" for name, stmt in local_inputs.items())
            )
    elif schema_version == "mizar-proof-v2":
        block = HDR + "\n" + "\n".join(f"{name} : {stmt}" for name, stmt in visible_items)
    else:
        block = HDR
        if visible_items:
            block += "\n" + "\n".join(f"{name} : {stmt}" for name, stmt in visible_items)

        if "local_assumptions" in row:
            block += "\n" + LOCAL_HDR
            if row["local_assumptions"]:
                block += "\n" + "\n".join(
                    f"{name} : {stmt}" for name, stmt in row["local_assumptions"].items()
                )

        local_inputs = row.get("local_inputs")
        if local_inputs:
            block += (
                "\n"
                + ATP_LOCAL_HDR
                + "\n"
                + "\n".join(f"{name} : {stmt}" for name, stmt in local_inputs.items())
            )

    prompt = f"{block}\n{SEP}\nGOAL {row['goal']}\n"
    if (
        condition == "facts_present"
        and schema_version in {"atp-v2", "mizar-proof-v2"}
        and "text" in row
        and "target" in row
    ):
        expected = _expected_prompt_prefix(row)
        if prompt != expected:
            raise RuntimeError(
                f"{row.get('id', '<unknown>')}: facts_present does not reconstruct "
                f"the {schema_version} prompt prefix"
            )

    return ConditionMaterialization(
        prompt=prompt,
        visible_facts=visible_facts,
        shuffle_eligible=shuffle_eligible,
        shuffle_changed=shuffle_changed,
        metamath_validity_supported=validity_supported,
        metamath_validity_reason=validity_reason,
    )


def build_prompt(row, condition: str, rng: random.Random, corrupt_pool):
    """Compatibility wrapper returning the condition's exact prompt text."""
    return materialize_condition(row, condition, rng, corrupt_pool).prompt


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(model, tok, prompts, max_new_tokens, batch_size, do_sample, temperature, device):
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(
            chunk,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1] :]
            outputs.append(tok.decode(new, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)
    return outputs


def load_train_fact_names(
    corpus: str | Path,
) -> tuple[set[str], bool]:
    """Load names visible anywhere in the model's complete training corpus."""
    train_paths = sorted((Path(corpus) / "shards").glob("*.jsonl"))
    if not train_paths:
        return set(), False
    names = {
        name
        for train_path in train_paths
        for row in load_jsonl(train_path)
        for name in row.get("facts", {})
    }
    return names, True


def run_probe(
    model,
    tok,
    rows,
    heldout,
    args,
    device,
    *,
    train_fact_names: set[str],
    train_visibility_available: bool,
):
    """Can the model state a fact given only its name?

    This is the mechanism check. The dense arm was trained to predict fact statements
    from their names; the split arm never was. If the probe does not separate the arms,
    the loss mask did not do what it was supposed to do, and any downstream proof
    difference is coming from somewhere else.

    Names are classified from every train shard used by the model, not inferred
    from their presence in the eval shard. Sampling remains pooled and deterministic.
    """
    seen: dict = {}
    for r in rows:
        for name, stmt in r["facts"].items():
            seen.setdefault(name, stmt)

    held = set(heldout)
    if train_visibility_available:
        categories = ("heldout", "train_visible", "eval_only")
    else:
        categories = ("heldout", "train_visibility_unknown")

    def visibility(name: str) -> str:
        if name in held:
            return "heldout"
        if not train_visibility_available:
            return "train_visibility_unknown"
        if name in train_fact_names:
            return "train_visible"
        return "eval_only"

    all_names = sorted(seen)
    pool_counts = Counter(visibility(name) for name in all_names)
    names = list(all_names)
    random.Random(args.seed).shuffle(names)
    names = names[: args.probe_n]
    selected_counts = Counter(visibility(name) for name in names)

    prompts = [f"{HDR}\n{n} :" for n in names]
    prompt_lengths = [len(tok(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    budgets = generation_budgets(
        prompt_lengths,
        context_length=args.context_length,
        max_new_tokens=args.probe_max_new_tokens,
    )
    generated = [""] * len(names)
    buckets: dict[int, list[int]] = {}
    for index, allowance in budgets.items():
        buckets.setdefault(allowance, []).append(index)
    for allowance, indices in buckets.items():
        subset = generate(
            model,
            tok,
            [prompts[index] for index in indices],
            allowance,
            args.batch_size,
            False,
            1.0,
            device,
        )
        for index, text in zip(indices, subset):
            generated[index] = text

    out: dict = {
        "train_visibility_available": train_visibility_available,
        "source_names": len(all_names),
        "pool_counts": {category: pool_counts[category] for category in categories},
        "evaluated_names": len(names),
        "selected_counts": {category: selected_counts[category] for category in categories},
        "items": [],
    }
    for index, (name, prompt, gen) in enumerate(zip(names, prompts, generated)):
        gold = seen[name]
        got = norm(gen.strip().splitlines()[0]) if gen.strip() else ""
        exact = got == norm(gold)
        prompt_ids, full_ids = tokenize_target_with_eos(tok, prompt, gold)
        target_tokens = len(full_ids) - len(prompt_ids)
        allowance = budgets.get(index, 0)
        budget_eligible = target_tokens <= allowance
        out["items"].append(
            {
                "name": name,
                "visibility": visibility(name),
                "gold": gold,
                "generated": got,
                "exact_match": exact,
                "target_tokens": target_tokens,
                "generation_attempted": index in budgets,
                "generation_budget": allowance,
                "whole_statement_budget_eligible": budget_eligible,
            }
        )
    evaluated = len(out["items"])
    attempted = sum(item["generation_attempted"] for item in out["items"])
    eligible_items = [item for item in out["items"] if item["whole_statement_budget_eligible"]]
    exact_evaluated = sum(item["exact_match"] for item in out["items"])
    exact_eligible = sum(item["exact_match"] for item in eligible_items)
    out.update(
        {
            "generation_attempted_names": attempted,
            "generation_budget_eligible_names": len(eligible_items),
            "generation_budget_ineligible_names": evaluated - len(eligible_items),
            "exact_match_count_evaluated": exact_evaluated,
            "exact_match_rate_evaluated": (exact_evaluated / evaluated if evaluated else None),
            "exact_match_count_budget_eligible": exact_eligible,
            "exact_match_rate_budget_eligible": (
                exact_eligible / len(eligible_items) if eligible_items else None
            ),
        }
    )
    return out


def target_nll(
    model,
    tok,
    rows,
    condition,
    rng,
    corrupt_pool,
    context_length,
    chunk_size,
    device,
):
    """Mean per-token NLL over every target span, teacher-forced in chunks.

    Generation metrics have a floor problem: a 0.5B model that cannot yet emit a
    whole valid proof scores zero under both arms, and zero minus zero measures
    nothing. Per-token loss has no floor — it separates the arms from the first
    epoch, long before either can finish a proof.

    Only target tokens are scored. The fact block is masked out for BOTH arms here,
    whatever each was trained on, because the question is whether the arms differ at
    producing the derivation — not whether one of them also modelled the prompt.

    Full-sequence logits are ``sequence x 151,936`` and OOM on long ATP/Metamath
    targets. Each forward therefore retains at most ``context_length`` input tokens
    and materializes logits for only ``chunk_size`` target tokens.
    """
    require_unique_ids(rows, f"{condition} target NLL cohort")

    import torch

    tot_nll = 0.0
    tot_tok = 0
    tot_correct = 0
    sliding = 0
    max_tokens = 0
    per_example = {}
    for i, row in enumerate(rows, 1):
        prompt = build_prompt(row, condition, rng, corrupt_pool)
        try:
            prompt_ids, full_ids = tokenize_target_with_eos(tok, prompt, row["target"])
        except RuntimeError as exc:
            raise RuntimeError(f"{row['id']}: {exc}") from exc
        nll, n_tokens, n_correct = chunked_sequence_nll(
            model,
            torch.tensor(full_ids, dtype=torch.long),
            target_start=len(prompt_ids),
            context_length=context_length,
            chunk_size=chunk_size,
            device=device,
        )
        tot_nll += nll
        tot_tok += n_tokens
        tot_correct += n_correct
        per_example[row["id"]] = {
            "nll_sum": nll,
            "target_tokens": n_tokens,
            "target_correct": n_correct,
            "target_nll_per_token": nll / n_tokens,
            "target_token_accuracy": n_correct / n_tokens,
        }
        sliding += len(full_ids) > context_length
        max_tokens = max(max_tokens, len(full_ids))
        if i % 100 == 0 or i == len(rows):
            print(f"  NLL {i:,}/{len(rows):,}", flush=True)
    n_examples = len(per_example)
    return {
        "target_nll_sum": tot_nll,
        "target_tokens": tot_tok,
        "target_correct": tot_correct,
        "target_token_micro_nll_per_token": (tot_nll / tot_tok if tot_tok else None),
        "target_token_micro_accuracy": (tot_correct / tot_tok if tot_tok else None),
        "target_example_macro_nll_per_token": (
            sum(item["target_nll_per_token"] for item in per_example.values()) / n_examples
            if n_examples
            else None
        ),
        "target_example_macro_accuracy": (
            sum(item["target_token_accuracy"] for item in per_example.values()) / n_examples
            if n_examples
            else None
        ),
        "per_example": per_example,
        "nll_context_length": context_length,
        "nll_chunk_size": chunk_size,
        "nll_context_policy": NLL_CONTEXT_POLICY,
        "nll_target_policy": NLL_TARGET_POLICY,
        "nll_sliding_window_examples": sliding,
        "max_prompt_plus_target_plus_eos_tokens": max_tokens,
    }


def exact_target(generated: str, gold: str) -> bool:
    """Whitespace-insensitive exact proof match, available for every family."""
    return norm(generated.strip()) == norm(gold.strip())


def verify_metamath_sources(
    mm_dir: str | Path,
    source_manifest: str | Path,
) -> dict:
    """Refuse a verifier database different from the corpus-building snapshot."""
    manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    for filename, expected in manifest["files"].items():
        path = Path(mm_dir) / filename
        if not path.exists():
            raise RuntimeError(f"pinned Metamath source is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected["sha256"]:
            raise RuntimeError(
                f"{filename} does not match corpus snapshot {manifest['commit']}: "
                f"expected {expected['sha256']}, got {digest}"
            )
    return manifest


def load_metamath_databases(mm_dir: str | Path | None) -> dict:
    """Load set/iset/nf when available; exact-match evaluation needs none."""
    if mm_dir is None:
        return {}
    from mm_expand import MM

    out = {}
    for name in ("set", "iset", "nf"):
        path = Path(mm_dir) / f"{name}.mm"
        if path.exists():
            print(f"parsing {path} for deterministic Metamath verification")
            out[name] = MM().parse(path)
    return out


def metamath_verifier_availability(verifier_module=mm_verify) -> dict:
    """Describe whether the future sound tri-state verifier contract is present."""
    detected_schema = getattr(verifier_module, "VERIFIER_SCHEMA_VERSION", None)
    tri_state_api = getattr(verifier_module, "verify_proof_tristate", None)
    if detected_schema != METAMATH_VERIFIER_SCHEMA_VERSION or not callable(tri_state_api):
        return {
            "status": "unavailable",
            "required_schema": METAMATH_VERIFIER_SCHEMA_VERSION,
            "detected_schema": detected_schema,
            "reason": (
                "sound Metamath validity requires the versioned tri-state "
                "verify_proof_tristate API; legacy boolean-compatible APIs are ignored"
            ),
        }
    return {
        "status": "integration_pending",
        "required_schema": METAMATH_VERIFIER_SCHEMA_VERSION,
        "detected_schema": detected_schema,
        "reason": (
            "the sound tri-state API is present but this evaluator integration "
            "must be completed before validity is reportable"
        ),
    }


def gold_trace_uses_only_supplied_labels(
    row: dict,
    visible_facts: dict[str, str] | None = None,
) -> bool:
    """Cheaply reject traces the prompt-grounded verifier cannot possibly accept."""
    facts = row["facts"] if visible_facts is None else visible_facts
    return all(label in facts for label, _ in mm_verify.parse_proof(row["target"]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_weight_artifact(filename: str) -> bool:
    return (
        filename.endswith(".safetensors")
        or filename.endswith(".safetensors.index.json")
        or filename == "pytorch_model.bin"
        or filename == "pytorch_model.bin.index.json"
        or re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", filename) is not None
    )


def _validated_weight_layout(filenames: set[str]) -> tuple[list[str], str | None]:
    if "model.safetensors" in filenames:
        expected = {"model.safetensors"}
        extra = filenames - expected
        if extra:
            raise RuntimeError(
                "trained weight file set has additional files: " + ", ".join(sorted(extra))
            )
        return ["model.safetensors"], None
    if SAFETENSORS_INDEX not in filenames:
        raise RuntimeError(
            "trained weight file set must be model.safetensors or exact shards plus "
            f"{SAFETENSORS_INDEX}"
        )
    shard_names = filenames - {SAFETENSORS_INDEX}
    matches = {name: SAFETENSORS_SHARD.fullmatch(name) for name in shard_names}
    extra = sorted(name for name, match in matches.items() if match is None)
    if extra:
        raise RuntimeError("trained weight file set has additional files: " + ", ".join(extra))
    if not matches:
        raise RuntimeError("trained weight shard index has no safetensors shards")
    totals = {int(match.group(2)) for match in matches.values() if match is not None}
    if len(totals) != 1:
        raise RuntimeError("trained weight shard filenames disagree on shard count")
    total = next(iter(totals))
    expected_shards = {
        f"model-{index:05d}-of-{total:05d}.safetensors" for index in range(1, total + 1)
    }
    if shard_names != expected_shards:
        raise RuntimeError(
            "trained weight shard set is incomplete: "
            f"missing={sorted(expected_shards - shard_names)}, "
            f"additional={sorted(shard_names - expected_shards)}"
        )
    return sorted(expected_shards), SAFETENSORS_INDEX


def _validate_trained_weight_inventory_record(
    files, declared_root: str
) -> tuple[list[str], str | None]:
    if not isinstance(files, dict) or not files:
        raise RuntimeError("model export metadata trained_weight_files must be nonempty")
    shard_names, index_name = _validated_weight_layout(set(files))
    for filename in shard_names + ([index_name] if index_name is not None else []):
        entry = files.get(filename)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "dtype"}:
            raise RuntimeError(
                f"trained weight inventory entry {filename!r} must contain "
                "exactly sha256, bytes, and dtype"
            )
        digest = entry["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"trained weight inventory {filename!r} sha256 is invalid")
        size = entry["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(f"trained weight inventory {filename!r} bytes must be positive")
        expected_dtype = "json" if filename == index_name else "BF16"
        if entry["dtype"] != expected_dtype:
            raise RuntimeError(
                f"trained weight inventory {filename!r} dtype must be {expected_dtype!r}"
            )
    if not isinstance(declared_root, str) or re.fullmatch(r"[0-9a-f]{64}", declared_root) is None:
        raise RuntimeError("model export metadata trained_weights_root_sha256 is invalid")
    canonical_root = _stable_json_sha256(files)
    if declared_root != canonical_root:
        raise RuntimeError(
            "model export metadata trained weight root does not match its canonical inventory"
        )
    return shard_names, index_name


def _safetensors_keys_and_dtype(path: Path) -> tuple[set[str], str]:
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            dtypes = {str(handle.get_slice(key).get_dtype()).upper() for key in keys}
    except Exception as error:
        raise RuntimeError(f"trained weight safetensors file is unreadable: {path.name}") from error
    if not keys:
        raise RuntimeError(f"trained weight safetensors file is empty: {path.name}")
    normalized_dtypes = {
        "BF16" if dtype in {"BF16", "BFLOAT16", "TORCH.BFLOAT16"} else dtype for dtype in dtypes
    }
    if normalized_dtypes != {"BF16"}:
        raise RuntimeError(
            f"trained weight file {path.name} must contain only BF16 tensors, "
            f"got {sorted(normalized_dtypes)}"
        )
    return keys, "BF16"


def validate_exported_trained_weights(model_dir: Path, export_metadata: dict) -> None:
    """Rehash and validate the exact trained payload before any model/tokenizer load."""
    expected_files = export_metadata.get("trained_weight_files")
    shard_names, index_name = _validate_trained_weight_inventory_record(
        expected_files,
        export_metadata.get("trained_weights_root_sha256"),
    )
    actual_names = {
        path.name
        for path in model_dir.iterdir()
        if path.is_file() and _is_weight_artifact(path.name)
    }
    if actual_names != set(expected_files):
        raise RuntimeError(
            "trained weight file set differs from export metadata: "
            f"missing={sorted(set(expected_files) - actual_names)}, "
            f"additional={sorted(actual_names - set(expected_files))}"
        )

    expected_keys_by_shard = {}
    if index_name is not None:
        index_path = model_dir / index_name
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("trained weight safetensors index is unreadable") from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("trained weight safetensors index has no nonempty weight_map")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise RuntimeError("trained weight safetensors index weight_map must map strings")
        if set(weight_map.values()) != set(shard_names):
            raise RuntimeError("trained weight safetensors index does not name the exact shard set")
        expected_keys_by_shard = {
            shard: {key for key, filename in weight_map.items() if filename == shard}
            for shard in shard_names
        }

    for filename in shard_names:
        path = model_dir / filename
        keys, dtype = _safetensors_keys_and_dtype(path)
        if index_name is not None and keys != expected_keys_by_shard[filename]:
            raise RuntimeError(f"trained weight shard {filename} tensor keys differ from the index")
        actual = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "dtype": dtype,
        }
        if actual != expected_files[filename]:
            raise RuntimeError(f"trained weight file {filename} differs from export metadata")
    if index_name is not None:
        index_path = model_dir / index_name
        actual_index = {
            "sha256": _sha256_file(index_path),
            "bytes": index_path.stat().st_size,
            "dtype": "json",
        }
        if actual_index != expected_files[index_name]:
            raise RuntimeError("trained weight safetensors index differs from export metadata")


def tokenizer_sha256(tokenizer) -> str:
    """Fingerprint the complete serialized tokenization behavior deterministically."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if not callable(serializer):
        raise RuntimeError(
            "tokenizer must expose complete serialized backend JSON via "
            "backend_tokenizer.to_str()"
        )
    try:
        backend_json = json.loads(serializer())
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("tokenizer backend serialization is not valid JSON") from error
    special_tokens = {
        str(key): str(value) for key, value in sorted(tokenizer.special_tokens_map.items())
    }
    return _stable_json_sha256(
        {
            "backend_tokenizer_json": backend_json,
            "special_tokens_map": special_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )


_LOCATION_ONLY_CONFIG_KEYS = {
    "_commit_hash",
    "_name_or_path",
    "cache_dir",
    "download_dir",
    "name_or_path",
}


def _semantic_model_config(value):
    if isinstance(value, dict):
        return {
            key: _semantic_model_config(child)
            for key, child in value.items()
            if key not in _LOCATION_ONLY_CONFIG_KEYS
            and not key.endswith("_path")
            and not key.endswith("_dir")
        }
    if isinstance(value, list):
        return [_semantic_model_config(child) for child in value]
    return value


def semantic_model_config_sha256(config: dict) -> str:
    """Hash model behavior while excluding export/cache location identity."""
    return _stable_json_sha256(_semantic_model_config(config))


def _step_from_path(path: Path) -> int | None:
    for part in (path, *path.parents):
        match = re.fullmatch(r"step(\d+)", part.name)
        if match:
            return int(match.group(1))
    return None


def resolve_model_provenance(model_path: str | Path) -> dict:
    """Require exporter-supplied model identity and semantic configuration."""
    resolved = Path(model_path).expanduser().resolve()
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"exported model config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"exported model config must be an object: {config_path}")

    metadata_path = resolved / "model_provenance.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"model export metadata is required for reportable evaluation: {metadata_path}"
        )
    export_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(export_metadata, dict):
        raise RuntimeError("model export metadata must be an object")
    if export_metadata.get("schema_version") != MODEL_EXPORT_SCHEMA_VERSION:
        raise RuntimeError(
            "model export metadata schema must be " f"{MODEL_EXPORT_SCHEMA_VERSION!r}"
        )

    checkpoint_step = export_metadata.get("checkpoint_step")
    if (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step <= 0
    ):
        raise RuntimeError("model export metadata checkpoint_step must be a positive integer")
    arm = export_metadata.get("arm")
    if arm not in {"dense", "split"}:
        raise RuntimeError("model export metadata arm must be 'dense' or 'split'")
    required_text = (
        "base_model_id",
        "base_model_revision",
        "initial_weights_sha256",
        "trained_weights_root_sha256",
        "source_commit",
    )
    for key in required_text:
        value = export_metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"model export metadata {key} must be nonempty")
    initial_weights_sha256 = export_metadata["initial_weights_sha256"].strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", initial_weights_sha256) is None:
        raise RuntimeError("model export metadata initial_weights_sha256 must be a SHA-256 digest")
    manifest_id = export_metadata.get("platform_run_manifest_id")
    manifest_sha256 = export_metadata.get("platform_run_manifest_sha256")
    if manifest_id is not None:
        if not isinstance(manifest_id, str) or not manifest_id.strip():
            raise RuntimeError("model export metadata platform run manifest ID must be nonempty")
        manifest_id = manifest_id.strip()
    if manifest_sha256 is not None:
        if (
            not isinstance(manifest_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
        ):
            raise RuntimeError(
                "model export metadata platform run manifest SHA-256 must be lowercase 64-hex"
            )
        if manifest_id is None:
            raise RuntimeError(
                "model export metadata platform run manifest SHA-256 requires its ID"
            )
    validate_exported_trained_weights(resolved, export_metadata)

    provenance = {
        "resolved_path": str(resolved),
        "checkpoint_step": checkpoint_step,
        "arm": arm,
        "base_model_id": export_metadata["base_model_id"].strip(),
        "base_model_revision": export_metadata["base_model_revision"].strip(),
        "initial_weights_sha256": initial_weights_sha256,
        "source_commit": export_metadata["source_commit"].strip(),
        "trained_weight_files": export_metadata["trained_weight_files"],
        "trained_weights_root_sha256": export_metadata["trained_weights_root_sha256"],
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "semantic_config_sha256": semantic_model_config_sha256(config),
        "export_metadata_schema": MODEL_EXPORT_SCHEMA_VERSION,
        "export_metadata": export_metadata,
    }
    if manifest_id is not None:
        provenance["platform_run_manifest_id"] = manifest_id
    if manifest_sha256 is not None:
        provenance["platform_run_manifest_sha256"] = manifest_sha256
    return provenance


def build_evaluation_metadata(
    *,
    args,
    tokenizer,
    corpus: str | Path,
    families: list[str],
    model_path: str | Path,
    model_provenance: dict | None = None,
) -> dict:
    """Build the complete deterministic controls and input fingerprint."""
    corpus = Path(corpus)
    eval_hashes = {}
    heldout_hashes = {}
    train_hashes = {}
    corpus_files = {}
    evaluated_families = set(families)
    for family in families:
        eval_path = corpus / "eval" / f"{family}.jsonl"
        heldout_path = corpus / "heldout" / f"{HELDOUT_MANIFEST[family]}.json"
        eval_hashes[family] = _sha256_file(eval_path)
        heldout_hashes[family] = _sha256_file(heldout_path)
        corpus_files[str(eval_path.relative_to(corpus))] = eval_hashes[family]
        corpus_files[str(heldout_path.relative_to(corpus))] = heldout_hashes[family]

    for train_path in sorted((corpus / "shards").glob("*.jsonl")):
        digest = _sha256_file(train_path)
        corpus_files[str(train_path.relative_to(corpus))] = digest
        if train_path.stem in evaluated_families:
            train_hashes[train_path.stem] = digest
    missing_train_hashes = evaluated_families - set(train_hashes)
    if missing_train_hashes:
        raise RuntimeError(
            "evaluated families are missing train shards: "
            + ", ".join(sorted(missing_train_hashes))
        )

    metamath_sources = corpus / "metamath_sources.json"
    if metamath_sources.exists():
        corpus_files[str(metamath_sources.relative_to(corpus))] = _sha256_file(metamath_sources)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_controls": {
            "evaluator_seed": args.seed,
            "conditions": list(args.conditions),
            "do_sample": bool(args.sample),
            "temperature": args.temperature,
            "context_length": args.context_length,
            "max_new_tokens": args.max_new_tokens,
            "limit": args.limit,
            "nll_chunk_size": args.nll_chunk_size,
            "nll_context_policy": NLL_CONTEXT_POLICY,
            "nll_target_policy": NLL_TARGET_POLICY,
        },
        "input_provenance": {
            "hash_algorithm": "sha256",
            "corpus_hash_policy": (
                "SHA-256 of canonical JSON mapping evaluated eval shards, heldout "
                "manifests, every model-visible train shard, and the optional "
                "Metamath source manifest to their content SHA-256 values"
            ),
            "tokenizer_sha256": tokenizer_sha256(tokenizer),
            "corpus_sha256": _stable_json_sha256(corpus_files),
            "eval_shard_sha256": eval_hashes,
            "heldout_manifest_sha256": heldout_hashes,
            "train_shard_sha256": train_hashes,
            "evaluator_sha256": _sha256_file(Path(__file__)),
            "model": model_provenance or resolve_model_provenance(model_path),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF-format directory for the trained arm")
    ap.add_argument("--arm", required=True, choices=("dense", "split"))
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=None,
        help="defaults to every family present under <corpus>/eval",
    )
    ap.add_argument(
        "--mm-dir",
        default=os.environ.get("P3_MM_DIR"),
        help="directory containing set.mm, iset.mm and nf.mm; without it "
        "Metamath still receives exact-match and NLL metrics",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--context-length", type=int, default=16_384)
    ap.add_argument("--nll-chunk-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--probe", action="store_true", help="also run the fact-recall probe")
    ap.add_argument("--probe-n", type=int, default=500)
    ap.add_argument("--probe-max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    try:
        model_provenance = resolve_model_provenance(args.model)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if model_provenance["arm"] != args.arm:
        raise SystemExit(
            f"exported arm {model_provenance['arm']!r} does not match --arm {args.arm!r}"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    families = args.families or discover_families(args.corpus)
    if not families:
        raise SystemExit(f"no eval families found under {Path(args.corpus) / 'eval'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    metamath_sources = None
    if args.mm_dir:
        manifest_path = Path(args.corpus) / "metamath_sources.json"
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} is required for deterministic Metamath verification")
        metamath_sources = verify_metamath_sources(args.mm_dir, manifest_path)
    metamath_availability = metamath_verifier_availability()
    results = {
        **build_evaluation_metadata(
            args=args,
            tokenizer=tok,
            corpus=args.corpus,
            families=families,
            model_path=args.model,
            model_provenance=model_provenance,
        ),
        "arm": args.arm,
        "families": {},
    }
    if metamath_sources is not None:
        results["metamath_sources"] = metamath_sources
    if args.probe:
        train_fact_names, train_visibility_available = load_train_fact_names(args.corpus)

    for family in families:
        rows, heldout = load_family(args.corpus, family)
        source_examples = len(rows)
        rows, over_context = partition_context_eligible(
            rows,
            tok,
            context_length=args.context_length,
        )
        context_eligible_examples = len(rows)
        if args.limit is not None:
            rows = rows[: args.limit]
        evaluated_examples = len(rows)
        corrupt_pool = sorted({s for row in rows for s in row["facts"].values()})
        family_result = {
            "source_examples": source_examples,
            "context_eligible_examples": context_eligible_examples,
            "evaluated_examples": evaluated_examples,
            "excluded_over_context_examples": len(over_context),
            "excluded_over_context_items": over_context,
            "heldout_manifest": HELDOUT_MANIFEST[family],
            "conditions": {},
        }
        results["families"][family] = family_result
        print(
            f"\n[{family}] context-eligible "
            f"{context_eligible_examples:,}/{source_examples:,}; "
            f"evaluating {evaluated_examples:,}; "
            f"excluded {len(over_context):,} over {args.context_length:,} tokens"
        )

        for condition in args.conditions:
            print(f"\n[{args.arm}/{family}] {condition}: {len(rows):,} examples")

            # Per-token loss exists for every family on the same context-eligible
            # cohort used for whole-proof generation.
            loss_stats = target_nll(
                model,
                tok,
                rows,
                condition,
                random.Random(args.seed),
                corrupt_pool,
                args.context_length,
                args.nll_chunk_size,
                device,
            )
            per_example_target_stats = loss_stats.pop("per_example")
            micro_nll = loss_stats["target_token_micro_nll_per_token"]
            micro_accuracy = loss_stats["target_token_micro_accuracy"]
            if micro_nll is None or micro_accuracy is None:
                print("  no target tokens scored")
            else:
                print(
                    f"  target token-micro NLL {micro_nll:.4f}, "
                    f"token-micro accuracy {micro_accuracy:.1%}; "
                    f"example-macro NLL "
                    f"{loss_stats['target_example_macro_nll_per_token']:.4f}, "
                    f"example-macro accuracy "
                    f"{loss_stats['target_example_macro_accuracy']:.1%}; "
                    f"{loss_stats['target_tokens']:,} target+EOS tokens"
                )

            rng = random.Random(args.seed)
            materialized = [
                materialize_condition(row, condition, rng, corrupt_pool) for row in rows
            ]
            prompts = [item.prompt for item in materialized]
            prompt_lengths = [
                len(tok(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts
            ]
            generated = [""] * len(rows)
            generation_buckets: dict[int, list[int]] = {}
            generation_budget = generation_budgets(
                prompt_lengths,
                context_length=args.context_length,
                max_new_tokens=args.max_new_tokens,
            )
            for i, allowance in generation_budget.items():
                generation_buckets.setdefault(allowance, []).append(i)
            for allowance, indices in generation_buckets.items():
                generated_subset = generate(
                    model,
                    tok,
                    [prompts[i] for i in indices],
                    allowance,
                    args.batch_size,
                    args.sample,
                    args.temperature,
                    device,
                )
                for i, text in zip(indices, generated_subset):
                    generated[i] = text

            per_example = []
            for i, (row, gen, condition_input) in enumerate(zip(rows, generated, materialized)):
                prompt_ids, full_ids = tokenize_target_with_eos(
                    tok, condition_input.prompt, row["target"]
                )
                target_tokens = len(full_ids) - len(prompt_ids)
                target_stats = per_example_target_stats[row["id"]]
                if target_stats["target_tokens"] != target_tokens:
                    raise RuntimeError(
                        f"{row['id']}: NLL and generation target token counts differ"
                    )
                attempted = i in generation_budget
                allowance = generation_budget.get(i, 0)
                budget_eligible = attempted and target_tokens <= allowance
                is_exact = attempted and exact_target(gen, row["target"])
                item = {
                    "id": row["id"],
                    "theorem": row["theorem"],
                    "cited": row["cited"],
                    **target_stats,
                    "exact_match": is_exact,
                    "whole_proof_budget_eligible": budget_eligible,
                    "generation_attempted": attempted,
                    "generation_budget": allowance,
                }

                if condition == "facts_shuffled":
                    item["shuffle_eligible"] = condition_input.shuffle_eligible
                    item["shuffle_changed"] = condition_input.shuffle_changed

                per_example.append(item)

            condition_result = {
                **loss_stats,
                **summarize_generation(
                    per_example,
                    source_examples=source_examples,
                    context_eligible_examples=context_eligible_examples,
                ),
                "per_example": per_example,
            }
            if condition == "facts_shuffled":
                shuffle_eligible = sum(item.shuffle_eligible for item in materialized)
                shuffle_changed = sum(item.shuffle_changed for item in materialized)
                condition_result.update(
                    {
                        "shuffle_eligible_examples": shuffle_eligible,
                        "shuffle_ineligible_examples": (len(materialized) - shuffle_eligible),
                        "shuffle_changed_examples": shuffle_changed,
                        "shuffle_change_rate_eligible": (
                            shuffle_changed / shuffle_eligible if shuffle_eligible else None
                        ),
                    }
                )
            if family == "metamath":
                condition_result["metamath_verification"] = {
                    **metamath_availability,
                    "condition_supported": (
                        condition_input.metamath_validity_supported
                        if materialized
                        else condition != "facts_corrupted"
                    ),
                    "condition_reason": (
                        condition_input.metamath_validity_reason
                        if materialized
                        else (CORRUPTED_METAMATH_REASON if condition == "facts_corrupted" else None)
                    ),
                }
            family_result["conditions"][condition] = condition_result
            exact_rate = condition_result["exact_match_rate_evaluated"]
            budget_coverage = condition_result["whole_proof_budget_coverage_evaluated"]
            exact_display = "n/a" if exact_rate is None else f"{exact_rate:.1%}"
            budget_display = "n/a" if budget_coverage is None else f"{budget_coverage:.1%}"
            print(
                f"  exact/evaluated {exact_display}; " f"whole-proof budget covers {budget_display}"
            )

        if args.probe:
            print(f"\n[{args.arm}/{family}] fact-recall probe")
            family_result["probe"] = run_probe(
                model,
                tok,
                rows,
                heldout,
                args,
                device,
                train_fact_names=train_fact_names,
                train_visibility_available=train_visibility_available,
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
