from __future__ import annotations

import collections
import math
import time

from .common import require
from .data import normalize


def aggregate(rows):
    result = {}
    metrics = ("loss", "loss_with_eos", "exact_match", "mcq_accuracy", "mcq_accuracy_normalized",
               "mcq_margin", "mcq_margin_normalized", "generation_terminated")
    for role in ("old", "new", "control"):
        subset = [r for r in rows if r["role"] == role]
        result[role] = {"facts": len(subset), "events": len({r["event"] for r in subset})}
        for metric in metrics:
            observed = [r for r in subset if metric in r]
            if not observed:
                continue
            events = collections.defaultdict(list)
            for row in observed:
                events[row["event"]].append(float(row[metric]))
            result[role][metric + "_event_macro"] = sum(sum(v) / len(v) for v in events.values()) / len(events)
            result[role][metric + "_fact_micro"] = sum(float(r[metric]) for r in observed) / len(observed)
    return result


def conditional_scores(model, probes, eos, device, batch_size, autocast):
    """Mean gold answer NLL excludes EOS; return sum/count and EOS NLL separately."""
    import torch
    import torch.nn.functional as F
    results = []
    for offset in range(0, len(probes), batch_size):
        batch = probes[offset:offset + batch_size]
        sequences = [p["prompt_ids"] + p["answer_ids"] + [eos] for p in batch]
        width = max(map(len, sequences))
        ids = torch.tensor([s + [eos] * (width - len(s)) for s in sequences], device=device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences], device=device)
        with torch.no_grad(), autocast():
            logits = model(input_ids=ids, attention_mask=mask).logits
        losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")
        for j, probe in enumerate(batch):
            start, n = len(probe["prompt_ids"]) - 1, len(probe["answer_ids"])
            gold_sum = float(losses[j, start:start + n].sum())
            eos_loss = float(losses[j, start + n])
            results.append({"answer_nll_sum": gold_sum, "answer_tokens": n, "eos_nll": eos_loss,
                            "loss": gold_sum / n, "loss_with_eos": (gold_sum + eos_loss) / (n + 1)})
    return results


def generated_answers(model, tokenizer, probes, device, batch_size, max_tokens, autocast):
    import torch
    eos = tokenizer.eos_token_id
    newline = tokenizer.encode("\n", add_special_tokens=False)
    require(len(newline) == 1, "Frozen generation stop requires a single newline token")
    stop_ids = [eos, newline[0]]
    results = []
    for offset in range(0, len(probes), batch_size):
        batch = probes[offset:offset + batch_size]
        width = max(len(p["prompt_ids"]) for p in batch)
        ids = torch.tensor([[eos] * (width - len(p["prompt_ids"])) + p["prompt_ids"] for p in batch], device=device)
        mask = torch.tensor([[0] * (width - len(p["prompt_ids"])) + [1] * len(p["prompt_ids"]) for p in batch], device=device)
        with torch.no_grad(), autocast():
            tokens = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                    max_new_tokens=max_tokens, eos_token_id=stop_ids,
                                    pad_token_id=eos, use_cache=True)
        for seq in tokens[:, width:].tolist():
            end = next((j for j, token in enumerate(seq) if token in stop_ids), len(seq))
            raw = tokenizer.decode(seq[:end], skip_special_tokens=False)
            results.append({"prediction": raw, "prediction_token_ids": seq[:end],
                            "generation_terminated": int(end < len(seq)),
                            "stop_token": seq[end] if end < len(seq) else None})
    return results


def mcq_result(scores, correct):
    sums = [-s["answer_nll_sum"] for s in scores]
    means = [-s["loss"] for s in scores]
    # Ties are failures for the target unless uniquely highest; no answer-index bias.
    return {"mcq_accuracy": int(sums[correct] > max(s for j, s in enumerate(sums) if j != correct)),
            "mcq_accuracy_normalized": int(means[correct] > max(s for j, s in enumerate(means) if j != correct)),
            "mcq_margin": sums[correct] - max(s for j, s in enumerate(sums) if j != correct),
            "mcq_margin_normalized": means[correct] - max(s for j, s in enumerate(means) if j != correct),
            "candidate_log_likelihoods": sums, "candidate_mean_log_likelihoods": means}


def evaluate(model, tokenizer, facts, config, device, autocast, behavior=True, generic=None):
    import torch
    started = time.monotonic()
    was_training = model.training
    model.eval()
    selected = [f for f in facts if f["role"] in ("old", "new", "control")]
    variants, timings = {}, {}
    for variant in ("canonical", "paraphrase"):
        key = "probe" if variant == "canonical" else "paraphrase_probe"
        pool = [f for f in selected if key in f]
        if not pool:
            continue
        tick = time.monotonic()
        probes = [f[key] for f in pool]
        scores = conditional_scores(model, probes, tokenizer.eos_token_id, device,
                                    config["eval_batch_size"], autocast)
        rows = [dict(id=f["id"], event=f["event"], role=f["role"], **s) for f, s in zip(pool, scores)]
        timings[variant + "_loss_seconds"] = time.monotonic() - tick
        if behavior:
            tick = time.monotonic()
            answers = generated_answers(model, tokenizer, probes, device, config["generation_batch_size"],
                                        config["max_answer_tokens"], autocast)
            for row, fact, answer in zip(rows, pool, answers):
                row.update(answer)
                row["exact_match"] = int(normalize(answer["prediction"]) in {normalize(a) for a in fact["aliases"]})
            timings[variant + "_generation_seconds"] = time.monotonic() - tick
            if variant == "canonical":
                tick = time.monotonic()
                flattened = [p for f in pool for p in f["mcq"]["probes"]]
                mcq_scores = conditional_scores(model, flattened, tokenizer.eos_token_id, device,
                                                config["eval_batch_size"], autocast)
                cursor = 0
                for row, fact in zip(rows, pool):
                    n = len(fact["mcq"]["choices"])
                    row.update(mcq_result(mcq_scores[cursor:cursor + n], fact["mcq"]["correct"]))
                    cursor += n
                timings["mcq_seconds"] = time.monotonic() - tick
        variants[variant] = {"aggregate": aggregate(rows), "facts": rows}
    if generic:
        tick = time.monotonic()
        generic_probes = [{"prompt_ids": ids[:1], "answer_ids": ids[1:]} for ids in generic]
        generic_scores = conditional_scores(model, generic_probes, tokenizer.eos_token_id, device,
                                             config["eval_batch_size"], autocast)
        generic_loss = sum(s["answer_nll_sum"] for s in generic_scores) / sum(s["answer_tokens"] for s in generic_scores)
        timings["generic_seconds"] = time.monotonic() - tick
    else:
        generic_loss = None
    model.train(was_training)
    return {"variants": variants, "generic_loss": generic_loss, "timings": timings,
            "wall_seconds": time.monotonic() - started}
