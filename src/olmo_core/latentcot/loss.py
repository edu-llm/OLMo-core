"""
CODI loss for latent chain-of-thought (PRD Phase 4).

One training step runs the shared-weight model twice per example and combines four
terms (following CODI, arXiv:2502.21074, adapted to our graph-reachability setup):

1. ``ce_teacher``  — cross-entropy on the explicit-CoT teacher view.
2. ``ce_student``  — cross-entropy on the continuous-thought student view (answer only).
3. ``distill``     — smooth-L1 alignment of the ``<distill>`` token's hidden state across
   all layers, teacher (detached) -> student. This transfers the teacher's reasoning
   into the student's continuous thoughts in a single stage.
4. ``vocab_reg``   — optional regularizer pulling the continuous thoughts toward the
   vocabulary manifold (the novel distributional-shift fix; ``R1`` primary, ``R2`` and
   ``L2`` for ablation/control).

``total = ce_teacher + ce_student + distill_weight*distill + vocab_reg_weight*vocab_reg``.

Examples are processed one at a time (batch dim 1) to sidestep the variable-length
prefix problem of the continuous-thought loop without left-padding/attention masks;
this is correct but not throughput-optimal (batched/bucketed processing is a Phase-5
optimization).
"""

from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F

from olmo_core.data.utils import get_labels

from .cot import embed_tokens, run_continuous_thoughts

__all__ = [
    "codi_loss",
    "vocab_manifold_reg",
    "explicit_cot_loss",
    "no_cot_loss",
    "arm_loss",
    "VocabReg",
]

VocabReg = Literal["none", "R1", "R2", "L2"]


def vocab_manifold_reg(
    model, thoughts: torch.Tensor, kind: VocabReg, entropy_floor: float = 0.0
) -> torch.Tensor:
    """
    Regularize continuous thoughts toward the vocabulary manifold.

    - ``R1`` (primary): pull each thought toward ``E · softmax(logit-lens(thought))`` — a
      soft mixture of real token embeddings — with an optional entropy floor so it stays a
      *mixture* rather than collapsing to a single token.
    - ``R2``: pull toward the single nearest (top-1 logit-lens) token embedding.
    - ``L2``: penalize the thought norm (the matched-strength control that isolates the
      *vocabulary-space direction* of ``R1`` from mere regularization).
    - ``none``: no penalty.

    :param model: A built transformer (uses ``model.lm_head`` and ``model.embeddings``).
    :param thoughts: Continuous thoughts of shape ``(batch, K, d_model)``.
    :returns: A scalar regularization loss.
    """
    if kind == "none":
        return thoughts.new_zeros(())
    if kind == "L2":
        return thoughts.float().pow(2).mean()

    logits = model.lm_head(thoughts)  # (batch, K, vocab) — labels=None returns logits
    embeddings = model.embeddings.weight  # (vocab, d_model)
    if kind == "R2":
        target = embeddings[logits.argmax(dim=-1)]  # (batch, K, d_model)
        return (thoughts.float() - target.float()).pow(2).mean()

    # R1. The softmax runs in fp32: under bf16 autocast a 100k-way softmax loses enough
    # precision in the tail to move the mixture target, and the .float() calls are no-ops
    # in the fp32 path (so both paths stay bit-identical to before).
    probs = torch.softmax(logits.float(), dim=-1)
    target = probs @ embeddings.float()  # (batch, K, d_model)
    reg = (thoughts.float() - target).pow(2).mean()
    if entropy_floor > 0:
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        reg = reg + torch.relu(torch.as_tensor(entropy_floor, device=reg.device) - entropy)
    return reg


def _forward_capture_distill(
    model,
    distill_pos: int,
    *,
    input_ids=None,
    input_embeddings=None,
    labels=None,
    z_loss_multiplier=None,
):
    """Run one forward, capturing each block's hidden state at ``distill_pos`` via hooks."""
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h[:, distill_pos, :]  # (batch, d_model)

        return hook

    handles = [block.register_forward_hook(make_hook(int(k))) for k, block in model.blocks.items()]
    try:
        if input_embeddings is not None:
            batch, seq = input_embeddings.shape[:2]
            dummy = torch.zeros((batch, seq), dtype=torch.long, device=model.device)
            out = model(
                dummy,
                input_embeddings=input_embeddings,
                labels=labels,
                z_loss_multiplier=z_loss_multiplier,
            )
        else:
            out = model(input_ids, labels=labels, z_loss_multiplier=z_loss_multiplier)
    finally:
        for handle in handles:
            handle.remove()
    distill_acts = torch.stack([captured[i] for i in sorted(captured)], dim=0)  # (layers, batch, d)
    return out, distill_acts


def _labels_for(
    model, input_ids: List[int], label_mask: List[bool], ignore_index: int
) -> torch.Tensor:
    ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    mask = torch.tensor([label_mask], dtype=torch.bool, device=model.device)
    return get_labels({"input_ids": ids, "label_mask": mask}, label_ignore_index=ignore_index)


def codi_loss(
    model,
    examples: List[Dict[str, Any]],
    *,
    distill_weight: float,
    vocab_reg: VocabReg = "none",
    vocab_reg_weight: float = 0.0,
    vocab_reg_entropy_floor: float = 0.0,
    label_ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the CODI loss over a list of encoded examples (see :func:`encode_example`).

    :returns: ``(loss, metrics)`` where ``loss`` is the mean total loss (a scalar tensor
        to call ``.backward()`` on) and ``metrics`` maps ``ce_teacher``/``ce_student``/
        ``distill``/``vocab_reg``/``thought_rms`` to floats (for logging). ``thought_rms``
        is diagnostic only — it is not part of the objective.
    """
    device = model.device
    totals = {
        "ce_teacher": 0.0,
        "ce_student": 0.0,
        "distill": 0.0,
        "vocab_reg": 0.0,
        # Scale tripwire: thoughts pass through the final norm (see cot.final_norm), so this
        # should sit near the token-embedding scale and stay flat in K. A climbing value means
        # the latent path is drifting off the manifold the pretrained weights were fit on.
        "thought_rms": 0.0,
    }
    total_loss = torch.zeros((), device=device)

    for ex in examples:
        k = ex["num_continuous_thoughts"]

        # --- Teacher branch (explicit CoT) ---
        teacher_ids = torch.tensor([ex["teacher_input_ids"]], dtype=torch.long, device=device)
        teacher_labels = _labels_for(
            model, ex["teacher_input_ids"], ex["teacher_label_mask"], label_ignore_index
        )
        teacher_out, teacher_acts = _forward_capture_distill(
            model, ex["teacher_distill_pos"], input_ids=teacher_ids, labels=teacher_labels
        )

        # --- Student branch (continuous thoughts) ---
        input_ids = ex["input_ids"]
        prefix_ids = torch.tensor([input_ids[: ex["bot_pos"] + 1]], dtype=torch.long, device=device)
        suffix_ids = torch.tensor(
            [input_ids[ex["bot_pos"] + 1 + k :]], dtype=torch.long, device=device
        )
        prefix_embeds = embed_tokens(model, prefix_ids)
        thoughts, embeds = run_continuous_thoughts(model, prefix_embeds, k)
        full = torch.cat([embeds, embed_tokens(model, suffix_ids)], dim=1)
        student_labels = _labels_for(model, input_ids, ex["label_mask"], label_ignore_index)
        student_out, student_acts = _forward_capture_distill(
            model, ex["distill_pos"], input_embeddings=full, labels=student_labels
        )

        # --- Distillation + vocab regularizer ---
        # fp32: this aligns raw hidden states across all layers, and bf16's ~3 decimal digits
        # would quantize exactly the small teacher-student differences the term exists to close.
        distill = F.smooth_l1_loss(student_acts.float(), teacher_acts.detach().float())
        reg = vocab_manifold_reg(model, thoughts, vocab_reg, vocab_reg_entropy_floor)

        # Optimize `.loss` (the trainable term); `.ce_loss` is detached (logging only).
        ex_loss = (
            teacher_out.loss + student_out.loss + distill_weight * distill + vocab_reg_weight * reg
        )
        total_loss = total_loss + ex_loss
        totals["ce_teacher"] += float(teacher_out.ce_loss.detach())
        totals["ce_student"] += float(student_out.ce_loss.detach())
        totals["distill"] += float(distill.detach())
        totals["vocab_reg"] += float(reg.detach())
        totals["thought_rms"] += float(thoughts.detach().float().pow(2).mean().sqrt())

    n = len(examples)
    metrics = {key: value / n for key, value in totals.items()}
    return total_loss / n, metrics


def _simple_ce(
    model, examples: List[Dict[str, Any]], ids_key: str, mask_key: str, metric: str, ignore: int
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mean cross-entropy over a per-example token view (used by the anchor arms)."""
    device = model.device
    total = torch.zeros((), device=device)
    ce_sum = 0.0
    for ex in examples:
        ids = torch.tensor([ex[ids_key]], dtype=torch.long, device=device)
        labels = _labels_for(model, ex[ids_key], ex[mask_key], ignore)
        out = model(ids, labels=labels)
        total = total + out.loss
        ce_sum += float(out.ce_loss.detach())
    n = len(examples)
    return total / n, {metric: ce_sum / n}


def explicit_cot_loss(
    model, examples: List[Dict[str, Any]], *, label_ignore_index: int = -100
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A0 anchor: standard CE on the explicit-CoT teacher view."""
    return _simple_ce(
        model, examples, "teacher_input_ids", "teacher_label_mask", "ce_teacher", label_ignore_index
    )


def no_cot_loss(
    model, examples: List[Dict[str, Any]], *, label_ignore_index: int = -100
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """A1 anchor: standard CE on the direct (no-reasoning) view ``question <distill> answer``."""
    return _simple_ce(
        model, examples, "direct_input_ids", "direct_label_mask", "ce_answer", label_ignore_index
    )


def arm_loss(
    model,
    examples: List[Dict[str, Any]],
    *,
    mode: str,
    distill_weight: float = 1.0,
    vocab_reg: VocabReg = "none",
    vocab_reg_weight: float = 0.0,
    vocab_reg_entropy_floor: float = 0.0,
    label_ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Dispatch to the loss for an experiment arm.

    :param mode: ``"explicit_cot"`` (A0), ``"no_cot"`` (A1), or ``"codi"`` (A2–A4).
    """
    if mode == "explicit_cot":
        return explicit_cot_loss(model, examples, label_ignore_index=label_ignore_index)
    if mode == "no_cot":
        return no_cot_loss(model, examples, label_ignore_index=label_ignore_index)
    if mode == "codi":
        return codi_loss(
            model,
            examples,
            distill_weight=distill_weight,
            vocab_reg=vocab_reg,
            vocab_reg_weight=vocab_reg_weight,
            vocab_reg_entropy_floor=vocab_reg_entropy_floor,
            label_ignore_index=label_ignore_index,
        )
    raise ValueError(f"unknown arm mode: {mode!r} (expected explicit_cot | no_cot | codi)")
