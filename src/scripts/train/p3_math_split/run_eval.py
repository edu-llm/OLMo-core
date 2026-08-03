"""Score a trained arm on all six held-out formal-proof families.

Every family receives target-token NLL, teacher-forced next-token match, and
whitespace-normalized exact whole-output match. Metamath additionally receives
deterministic proof validity when set.mm,
iset.mm or nf.mm is available *and the gold trace passes the same verifier*. The
gold gate matters: old Metamath rendering emitted local hypotheses and synthetic
``(reuse)`` labels that its checker cannot validate, and evaluator limitations must
not be reported as model failures.

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

    python src/scripts/train/p3_math_split/run_eval.py --model runs/split/hf --arm split --corpus corpus --families metamath mizar prf2 --mm-dir /tmp/dscount/mm --out results/split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm_verify import norm, parse_proof, verify_proof

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local assumptions:"
SEP = "---"
CONDITIONS = ("facts_present", "facts_absent", "facts_corrupted", "facts_shuffled")
FAMILIES = ("enigma", "isabelle", "metamath", "mizar", "prf2", "thproofs")
DEFAULT_MAX_NEW_TOKENS = 8_192
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
    heldout_path = corpus / "heldout" / f"{HELDOUT_MANIFEST[family]}.json"
    heldout = json.loads(heldout_path.read_text(encoding="utf-8"))["facts"]
    return rows, heldout


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
                excluded.append(
                    {"id": row["id"], "tokens_with_eos": tokens_with_eos}
                )
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
        total_correct += int(
            (prediction_logits.argmax(dim=-1) == labels).sum().item()
        )
    return total_nll, total_tokens, total_correct


def build_prompt(row, condition: str, rng: random.Random, corrupt_pool):
    """Render the prompt for one condition. Everything up to and including `GOAL`.

    The prompt is assembled exactly the way build_corpus.py assembles `text`, so
    `facts_present` reproduces the training-time format character for character.
    """
    facts = dict(row["facts"])

    if condition == "facts_absent":
        facts = {}
    elif condition == "facts_shuffled":
        items = list(facts.items())
        rng.shuffle(items)
        facts = dict(items)
    elif condition == "facts_corrupted":
        # Keep the names, replace each statement with a different fact's statement.
        # The block stays well-formed and plausible; it is just wrong.
        if len(corrupt_pool) < 2:
            raise ValueError("facts_corrupted requires at least two distinct statements")
        corrupted = {}
        for name, statement in facts.items():
            replacement = rng.choice(corrupt_pool)
            if replacement == statement and len(corrupt_pool) > 1:
                replacement = corrupt_pool[
                    (corrupt_pool.index(replacement) + 1) % len(corrupt_pool)
                ]
            corrupted[name] = replacement
        facts = corrupted

    block = HDR + ("\n" + "\n".join(f"{n} : {s}" for n, s in facts.items()) if facts else "")
    if "local_assumptions" in row:
        block += "\n" + LOCAL_HDR
        if row["local_assumptions"]:
            block += "\n" + "\n".join(
                f"{n} : {s}" for n, s in row["local_assumptions"].items()
            )
    return f"{block}\n{SEP}\nGOAL {row['goal']}\n"


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


def run_probe(model, tok, rows, heldout, args, device):
    """Can the model state a fact given only its name?

    This is the mechanism check. The dense arm was trained to predict fact statements
    from their names; the split arm never was. If the probe does not separate the arms,
    the loss mask did not do what it was supposed to do, and any downstream proof
    difference is coming from somewhere else.

    Reported separately for held-out facts (neither arm saw them supervised — both
    should fail, and it calibrates the floor) and train facts (only dense saw them).
    """
    seen: dict = {}
    for r in rows:
        for name, stmt in r["facts"].items():
            seen.setdefault(name, stmt)

    held = set(heldout)
    names = sorted(seen)
    random.Random(args.seed).shuffle(names)
    names = names[: args.probe_n]

    prompts = [f"{HDR}\n{n} :" for n in names]
    gens = generate(
        model, tok, prompts, args.probe_max_new_tokens, args.batch_size, False, 1.0, device
    )

    out: dict = {
        "train_facts": {"n": 0, "exact": 0},
        "heldout_facts": {"n": 0, "exact": 0},
        "items": [],
    }
    for name, gen in zip(names, gens):
        gold = seen[name]
        got = norm(gen.strip().splitlines()[0]) if gen.strip() else ""
        exact = got == norm(gold)
        bucket = "heldout_facts" if name in held else "train_facts"
        out[bucket]["n"] += 1
        out[bucket]["exact"] += int(exact)
        out["items"].append(
            {"name": name, "heldout": name in held, "gold": gold, "generated": got, "exact": exact}
        )
    for b in ("train_facts", "heldout_facts"):
        n = out[b]["n"]
        out[b]["exact_rate"] = out[b]["exact"] / n if n else None
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
    import torch

    tot_nll = 0.0
    tot_tok = 0
    tot_correct = 0
    sliding = 0
    max_tokens = 0
    per_example_accuracy = {}
    for i, row in enumerate(rows, 1):
        prompt = build_prompt(row, condition, rng, corrupt_pool)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt + row["target"], add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"{row['id']}: prompt tokenization is not a prefix of prompt+target"
            )
        if len(full_ids) == len(prompt_ids):
            continue
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
        per_example_accuracy[row["id"]] = n_correct / max(n_tokens, 1)
        sliding += len(full_ids) > context_length
        max_tokens = max(max_tokens, len(full_ids))
        if i % 100 == 0 or i == len(rows):
            print(f"  NLL {i:,}/{len(rows):,}", flush=True)
    return {
        "target_nll_per_token": tot_nll / max(tot_tok, 1),
        "target_tokens": tot_tok,
        "target_token_accuracy": tot_correct / max(tot_tok, 1),
        "per_example_target_token_accuracy": per_example_accuracy,
        "nll_context_length": context_length,
        "nll_chunk_size": chunk_size,
        "nll_sliding_window_examples": sliding,
        "max_prompt_plus_target_tokens": max_tokens,
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


def verify_metamath_row(databases: dict, row: dict, generated: str):
    db_name = row["theorem"].partition(":")[0]
    mm = databases.get(db_name)
    if mm is None:
        return None
    return verify_proof(
        mm,
        generated,
        row["goal"],
        row["facts"],
        gold_target=row["target"],
        local_assumptions=row.get("local_assumptions"),
    )


def gold_trace_uses_only_supplied_labels(row: dict) -> bool:
    """Cheaply reject traces the prompt-grounded verifier cannot possibly accept."""
    return all(label in row["facts"] for label, _ in parse_proof(row["target"]))


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
            raise SystemExit(
                f"{manifest_path} is required for deterministic Metamath verification"
            )
        metamath_sources = verify_metamath_sources(args.mm_dir, manifest_path)
    metamath_dbs = load_metamath_databases(args.mm_dir)
    results = {
        "arm": args.arm,
        "model": args.model,
        "greedy": not args.sample,
        "context_length": args.context_length,
        "max_new_tokens": args.max_new_tokens,
        "families": {},
    }
    if metamath_sources is not None:
        results["metamath_sources"] = metamath_sources

    for family in families:
        rows, heldout = load_family(args.corpus, family)
        source_n = len(rows)
        rows, over_context = partition_context_eligible(
            rows,
            tok,
            context_length=args.context_length,
        )
        context_eligible_n = len(rows)
        if args.limit:
            rows = rows[: args.limit]
        corrupt_pool = sorted({s for row in rows for s in row["facts"].values()})
        family_result = {
            "n": len(rows),
            "source_n": source_n,
            "context_eligible_n": context_eligible_n,
            "excluded_over_context": len(over_context),
            "excluded_over_context_items": over_context,
            "heldout_manifest": HELDOUT_MANIFEST[family],
            "conditions": {},
        }
        results["families"][family] = family_result
        print(
            f"\n[{family}] context-eligible {context_eligible_n:,}/{source_n:,}; "
            f"excluded {len(over_context):,} over {args.context_length:,} tokens"
        )

        gold_verifiable = {}
        if family == "metamath" and metamath_dbs:
            for row in rows:
                if gold_trace_uses_only_supplied_labels(row):
                    verification = verify_metamath_row(
                        metamath_dbs, row, row["target"]
                    )
                    gold_verifiable[row["id"]] = bool(
                        verification is not None and verification.valid
                    )
                else:
                    gold_verifiable[row["id"]] = False
            family_result["metamath_gold_verifier_coverage"] = sum(
                gold_verifiable.values()
            ) / max(len(rows), 1)

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
            per_example_token_accuracy = loss_stats.pop(
                "per_example_target_token_accuracy"
            )
            print(
                f"  target NLL/token {loss_stats['target_nll_per_token']:.4f} "
                f"and token match {loss_stats['target_token_accuracy']:.1%} "
                f"over {loss_stats['target_tokens']:,} tokens"
            )

            rng = random.Random(args.seed)
            prompts = [build_prompt(row, condition, rng, corrupt_pool) for row in rows]
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
            exact = exact_on_eligible = exact_eligible = valid = verifier_eligible = 0
            for i, (row, gen) in enumerate(zip(rows, generated)):
                target_tokens = len(
                    tok(row["target"], add_special_tokens=False)["input_ids"]
                )
                budget_eligible = target_tokens <= generation_budget.get(i, 0)
                is_exact = bool(gen) and exact_target(gen, row["target"])
                exact += is_exact
                exact_on_eligible += is_exact and budget_eligible
                exact_eligible += budget_eligible
                item = {
                    "id": row["id"],
                    "theorem": row["theorem"],
                    "cited": row["cited"],
                    "exact_match": is_exact,
                    "target_token_accuracy": per_example_token_accuracy[row["id"]],
                    "target_tokens": target_tokens,
                    "whole_proof_budget_eligible": budget_eligible,
                    "generation_attempted": i in generation_budget,
                    "generation_budget": generation_budget.get(i, 0),
                }

                if family == "metamath" and gold_verifiable.get(row["id"], False):
                    verification = verify_metamath_row(metamath_dbs, row, gen)
                    assert verification is not None
                    verifier_eligible += 1
                    valid += verification.valid
                    item["metamath"] = verification.as_dict()
                per_example.append(item)

            n = max(len(rows), 1)
            condition_result = {
                **loss_stats,
                "exact_match_count": exact,
                "exact_match_rate_all": exact / n,
                "exact_match_rate_budget_eligible": exact_on_eligible
                / max(exact_eligible, 1),
                "whole_proof_budget_eligible": exact_eligible,
                "whole_proof_budget_coverage": exact_eligible / n,
                "generation_attempted": len(generation_budget),
                "per_example": per_example,
            }
            if verifier_eligible:
                condition_result.update(
                    {
                        "metamath_verifier_eligible": verifier_eligible,
                        "metamath_valid_count": valid,
                        "metamath_valid_rate": valid / verifier_eligible,
                    }
                )
            family_result["conditions"][condition] = condition_result
            print(
                f"  exact {condition_result['exact_match_rate_all']:.1%}; "
                f"whole-proof budget covers "
                f"{condition_result['whole_proof_budget_coverage']:.1%}"
                + (
                    f"; Metamath valid {condition_result['metamath_valid_rate']:.1%} "
                    f"on {verifier_eligible:,} gold-verifiable rows"
                    if verifier_eligible
                    else ""
                )
            )

        if args.probe:
            print(f"\n[{args.arm}/{family}] fact-recall probe")
            family_result["probe"] = run_probe(model, tok, rows, heldout, args, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
