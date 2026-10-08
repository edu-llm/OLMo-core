from __future__ import annotations

from .common import digest, require
from .units import training_key
from .acquisition import acquisition_key


def encode_probe(tokenizer, question, answer, context_length):
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    joined = tokenizer.encode(prompt + " " + answer, add_special_tokens=False)
    require(joined[:len(prompt_ids)] == prompt_ids, "Tokenizer merges prompt/answer boundary")
    answer_ids = joined[len(prompt_ids):]
    require(answer_ids and len(joined) + 1 <= context_length,
            "Empty answer or evaluation context overflow; no truncation allowed")
    return {"prompt": prompt, "prompt_ids": prompt_ids, "answer_ids": answer_ids}


def generic_slice(tokens, size, key):
    require(len(tokens) > size, "Generic token source is too short")
    offset = int(digest(key)[:16], 16) % (len(tokens) - size)
    return tokens[offset:offset + size]


def packed_example(tokenizer, text, key, generic_tokens, loss_budget, context_length,
                   answer=None):
    eos = tokenizer.eos_token_id
    require(eos is not None, "An explicit EOS token is required")
    if answer is None:
        body = tokenizer.encode(text, add_special_tokens=False)
        ids, labels = [eos] + body + [eos], [-100] + body + [eos]
        content_tokens = len(body) + 1
    else:
        probe = encode_probe(tokenizer, text, answer, context_length)
        body, target = probe["prompt_ids"], probe["answer_ids"]
        ids, labels = body + target + [eos], [-100] * len(body) + target + [eos]
        content_tokens = len(target) + 1
    require(content_tokens <= loss_budget, f"Complete example exceeds loss budget: {key}")
    filler = generic_slice(generic_tokens, loss_budget - content_tokens, key)
    ids += filler
    labels += filler
    require(len(ids) <= context_length, f"Packed example exceeds context: {key}")
    require(sum(t != -100 for t in labels[1:]) == loss_budget, "Loss mask budget mismatch")
    padding = context_length - len(ids)
    return {"input_ids": ids + [eos] * padding, "labels": labels + [-100] * padding,
            "attention_mask": [1] * len(ids) + [0] * padding,
            "loss_tokens": loss_budget, "content_tokens": content_tokens,
            "generic_filler_tokens": len(filler), "padding_tokens": padding,
            "content_sha256": digest({"ids": ids, "labels": labels})}


def build_examples(facts, tokenizer, generic_tokens, config, qa_teaching_records=None, acquisition_records=None):
    examples = {}
    budget, length = config["loss_tokens_per_example"], config["context_length"]
    eval_length = config.get("eval_context_length", length)
    for index, fact in enumerate(facts):
        fact["statement_ids"] = tokenizer.encode(fact["statement"], add_special_tokens=False)
        fact["probe"] = encode_probe(tokenizer, fact["question"], fact["answer"], eval_length)
        fact["alias_ids"] = [encode_probe(tokenizer, fact["question"], a, eval_length)["answer_ids"]
                              for a in fact["aliases"]]
        if fact.get("paraphrase"):
            fact["paraphrase_probe"] = encode_probe(tokenizer, fact["paraphrase"], fact["answer"], eval_length)
        if fact.get("mcq"):
            mcq = fact["mcq"]
            correct = mcq["choices"].pop(mcq["correct"])
            position = (index + config["eval_seed"]) % (len(mcq["choices"]) + 1)
            mcq["choices"].insert(position, correct)
            mcq["correct"] = position
            question = fact["question"] + "\nChoices:\n" + "\n".join(
                f"{chr(65 + j)}. {a}" for j, a in enumerate(mcq["choices"]))
            question += "\nRespond with the complete answer text."
            mcq["probes"] = [encode_probe(tokenizer, question, a, eval_length) for a in mcq["choices"]]
        role, key = fact["role"], training_key(fact)
        if role in ("old", "new"):
            example = packed_example(
                tokenizer, fact["statement"], role + "/" + key, generic_tokens, budget, length)
            if role + "/" + key in examples:
                require(examples[role + "/" + key] == example, "Probes in one unit have different training content")
            examples[role + "/" + key] = example
        if role == "qa" and qa_teaching_records is None:
            examples["qa/" + key] = packed_example(
                tokenizer, fact["question"], "qa/" + key, generic_tokens, budget, length,
                answer=fact["answer"])
        if role == "old":
            examples["gen/" + key] = packed_example(
                tokenizer, "", "gen/" + key, generic_tokens, budget, length)
    if qa_teaching_records is not None:
        for fact in qa_teaching_records:
            key = "qa/" + fact["id"]
            require(key not in examples, "Duplicate QA-teaching example key")
            examples[key] = packed_example(tokenizer, fact["question"], key, generic_tokens, budget, length,
                                           answer=fact["answer"])
    for record in acquisition_records or []:
        for variant, question in enumerate(record["questions"]):
            key = acquisition_key(record, variant)
            require(key not in examples, "Duplicate acquisition example key")
            examples[key] = packed_example(tokenizer, question, key, generic_tokens, budget, length,
                                           answer=record["answer"])
    examples["filler"] = packed_example(tokenizer, "", "stage1-filler", generic_tokens, budget, length)
    return examples
