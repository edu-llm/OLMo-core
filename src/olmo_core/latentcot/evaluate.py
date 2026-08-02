"""
Evaluation harness for the latent-CoT experiment (PRD Phase 6.1/6.2).

Predicts the yes/no reachability answer per arm, aggregates solve-rate by graph depth,
and assembles the two gates:

- **gate A (superposition):** ``acc_continuous(D) - acc_discrete(D)`` and its slope vs depth
  D — the theory predicts this is positive and increasing.
- **gate B (distributional-shift fix):** overall accuracy + latent decodability for the
  CODI arms (A2 vs A3=R1 vs A4=L2 control).

Answers are read at the ``<distill>`` position: CODI runs the continuous thoughts then
decodes; ``no_cot`` decodes from ``question <distill>``; ``explicit_cot`` greedily generates
its CoT up to ``<distill>`` and then decodes.
"""

from collections import defaultdict
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import torch

from .cot import embed_tokens, run_continuous_thoughts
from .tokens import DISTILL, encode

__all__ = [
    "answer_token_ids",
    "node_token_id",
    "greedy_generate",
    "answer_logits",
    "predict_reachable",
    "answer_margin",
    "codi_answer_margin_fn",
    "solve_rate_by_depth",
    "overall_accuracy",
    "gate_a_curve",
    "linear_slope",
    "inference_token_cost",
    "mean_decodability",
    "run_eval",
]

# Which loss/inference mode each arm uses.
ARM_MODES = {
    "A0": "explicit_cot",
    "A1": "no_cot",
    "A2": "codi",
    "A3": "codi",
    "A4": "codi",
}


@lru_cache(maxsize=1)
def answer_token_ids() -> Tuple[int, int]:
    """The (yes, no) first-token ids for the rendered answers ``" yes"`` / ``" no"``."""
    return encode(" yes")[0], encode(" no")[0]


@lru_cache(maxsize=None)
def node_token_id(node: int) -> int:
    """The token id used for a graph node id (the number token in ``f" {node}"``)."""
    return encode(f" {node}")[-1]


@torch.no_grad()
def greedy_generate(model, input_ids: List[int], max_new_tokens: int, stop_token: int) -> List[int]:
    """Greedy-decode from ``input_ids`` until ``stop_token`` is emitted or the cap is hit."""
    device = model.device
    ids = list(input_ids)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        if nxt == stop_token:
            break
    return ids


def _codi_prefix_suffix_embeds(model, ex):
    device = model.device
    k = ex["num_continuous_thoughts"]
    input_ids = ex["input_ids"]
    prefix = torch.tensor([input_ids[: ex["bot_pos"] + 1]], dtype=torch.long, device=device)
    eval_suffix = input_ids[ex["bot_pos"] + 1 + k : ex["distill_pos"] + 1]  # [<eot>, <distill>]
    suffix = torch.tensor([eval_suffix], dtype=torch.long, device=device)
    return embed_tokens(model, prefix), embed_tokens(model, suffix), k


@torch.no_grad()
def answer_logits(model, ex, arm_mode: str, *, max_new_tokens: int = 128) -> torch.Tensor:
    """Return the vocab logits at the answer-predicting (``<distill>``) position for one example."""
    device = model.device
    if arm_mode == "codi":
        prefix_embeds, suffix_embeds, k = _codi_prefix_suffix_embeds(model, ex)
        _, embeds = run_continuous_thoughts(model, prefix_embeds, k)
        full = torch.cat([embeds, suffix_embeds], dim=1)
        dummy = torch.zeros(full.shape[:2], dtype=torch.long, device=device)
        return model(dummy, input_embeddings=full)[0, -1]
    if arm_mode == "no_cot":
        ids = ex["direct_input_ids"][: ex["direct_distill_pos"] + 1]
        return model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1]
    if arm_mode == "explicit_cot":
        prompt = ex["teacher_input_ids"][: ex["teacher_bot_pos"] + 1]  # question <bot>
        gen = greedy_generate(model, prompt, max_new_tokens=max_new_tokens, stop_token=DISTILL)
        return model(torch.tensor([gen], dtype=torch.long, device=device))[0, -1]
    raise ValueError(f"unknown arm mode: {arm_mode!r}")


@torch.no_grad()
def predict_reachable(model, ex, arm_mode: str, **kwargs) -> bool:
    yes_id, no_id = answer_token_ids()
    row = answer_logits(model, ex, arm_mode, **kwargs)
    return bool(row[yes_id] > row[no_id])


@torch.no_grad()
def answer_margin(model, ex, arm_mode: str, **kwargs) -> float:
    """The yes-minus-no logit margin at the answer position (>0 => predicts reachable)."""
    yes_id, no_id = answer_token_ids()
    row = answer_logits(model, ex, arm_mode, **kwargs)
    return float(row[yes_id] - row[no_id])


def codi_answer_margin_fn(model, ex) -> Callable[[torch.Tensor], float]:
    """Return ``f(thoughts) -> yes-no margin`` with the prefix/suffix fixed (for causal probing)."""
    device = model.device
    prefix_embeds, suffix_embeds, _ = _codi_prefix_suffix_embeds(model, ex)
    yes_id, no_id = answer_token_ids()

    @torch.no_grad()
    def margin(thoughts: torch.Tensor) -> float:
        full = torch.cat([prefix_embeds, thoughts, suffix_embeds], dim=1)
        dummy = torch.zeros(full.shape[:2], dtype=torch.long, device=device)
        row = model(dummy, input_embeddings=full)[0, -1]
        return float(row[yes_id] - row[no_id])

    return margin


@torch.no_grad()
def solve_rate_by_depth(model, examples, arm_mode: str) -> Dict[int, float]:
    """Reachability accuracy bucketed by graph depth D."""
    correct: Dict[int, int] = defaultdict(int)
    total: Dict[int, int] = defaultdict(int)
    for ex in examples:
        total[ex["depth"]] += 1
        correct[ex["depth"]] += int(predict_reachable(model, ex, arm_mode) == ex["reachable"])
    return {d: correct[d] / total[d] for d in sorted(total)}


@torch.no_grad()
def overall_accuracy(model, examples, arm_mode: str) -> float:
    correct = sum(int(predict_reachable(model, ex, arm_mode) == ex["reachable"]) for ex in examples)
    return correct / len(examples)


def gate_a_curve(continuous: Dict[int, float], discrete: Dict[int, float]) -> Dict[int, float]:
    """``acc_continuous(D) - acc_discrete(D)`` for depths present in both."""
    return {d: continuous[d] - discrete[d] for d in sorted(continuous) if d in discrete}


def linear_slope(curve: Dict[int, float]) -> float:
    """Least-squares slope of a ``{x: y}`` curve (the gate-A slope vs depth)."""
    xs = list(curve)
    ys = [curve[x] for x in xs]
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def inference_token_cost(ex, arm_mode: str) -> int:
    """
    A simple forward-compute proxy = total tokens processed to produce the answer
    (counts the K sequential continuous-thought passes for CODI). For matched-compute
    accuracy-vs-cost plots (gate 6.2); not a FLOP-exact figure.
    """
    if arm_mode == "no_cot":
        return ex["direct_distill_pos"] + 1
    k = ex["num_continuous_thoughts"]
    prefix_len = ex["bot_pos"] + 1
    if arm_mode == "codi":
        # K sequential passes over growing prefixes, then one pass over prefix+K+suffix.
        thought_passes = sum(prefix_len + i for i in range(k))
        final = prefix_len + k + 2  # + <eot> <distill>
        return thought_passes + final
    if arm_mode == "explicit_cot":
        # rough: generate the CoT autoregressively (length ~ teacher_distill_pos - bot_pos).
        gen_len = ex["teacher_distill_pos"] - ex["teacher_bot_pos"]
        return sum(prefix_len + i for i in range(gen_len)) + ex["teacher_distill_pos"] + 1
    raise ValueError(f"unknown arm mode: {arm_mode!r}")


@torch.no_grad()
def mean_decodability(model, examples) -> float:
    """Average logit-lens top-1 mass of each example's continuous thoughts (CODI arms)."""
    from .probes import decodability

    values = []
    for ex in examples:
        prefix_embeds, _, k = _codi_prefix_suffix_embeds(model, ex)
        thoughts, _ = run_continuous_thoughts(model, prefix_embeds, k)
        values.append(decodability(model, thoughts))
    return sum(values) / len(values)


def run_eval(models_by_arm: Dict[str, object], examples) -> dict:
    """
    Run the full evaluation for the arms whose models are provided.

    :param models_by_arm: e.g. ``{"A0": model, "A2": model, "A3": model, ...}``.
    :returns: A report dict with per-arm accuracy / solve-rate-by-depth / decodability,
        the gate-A curve + slope (continuous A2 minus discrete A0), and the gate-B table
        (A2/A3/A4 accuracy + decodability).
    """
    per_arm: Dict[str, dict] = {}
    for arm, model in models_by_arm.items():
        mode = ARM_MODES[arm]
        model.eval()
        entry = {
            "mode": mode,
            "overall_acc": overall_accuracy(model, examples, mode),
            "solve_rate_by_depth": solve_rate_by_depth(model, examples, mode),
        }
        if mode == "codi":
            entry["decodability"] = mean_decodability(model, examples)
        per_arm[arm] = entry

    report: dict = {"per_arm": per_arm}
    if "A2" in per_arm and "A0" in per_arm:
        curve = gate_a_curve(
            per_arm["A2"]["solve_rate_by_depth"], per_arm["A0"]["solve_rate_by_depth"]
        )
        report["gate_a"] = {"curve": curve, "slope": linear_slope(curve)}
    report["gate_b"] = {
        arm: {"acc": per_arm[arm]["overall_acc"], "decodability": per_arm[arm].get("decodability")}
        for arm in ("A2", "A3", "A4")
        if arm in per_arm
    }
    return report
