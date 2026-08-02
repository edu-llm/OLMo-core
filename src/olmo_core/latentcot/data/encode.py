"""
Render graph-reachability instances to token sequences (PRD Phase 2.2).

Two structurally parallel views are produced per instance, so the CODI teacher and
student differ *only* inside the reasoning region and share the ``<distill>`` +
answer suffix (which is what the feature-distillation loss aligns):

- **teacher** = ``question <bot> explicit-CoT <eot> <distill> answer``
- **student** = ``question <bot> [K THOUGHT slots] <eot> <distill> answer``

Labels follow OLMo-core's convention (:func:`olmo_core.data.utils.get_labels`): we
emit a boolean ``label_mask`` that is ``True`` on the *supervised token positions*
(the framework masks then shifts left, so a ``True`` at position ``j`` scores the
prediction of token ``j`` from position ``j-1``). The student supervises only the
answer span; the teacher supervises the CoT tokens and the answer span.
"""

from typing import Any, Dict

from ..tokens import BOT, DISTILL, EOT, THOUGHT
from ..tokens import encode as encode_text
from .graph_gen import Example

__all__ = ["render_question", "render_cot", "render_answer", "encode_example"]


def render_question(ex: Example) -> str:
    """Render the reachability query and edge list to text (nodes as digits)."""
    edges = " ".join(f"{u} > {v}" for u, v in ex.edges)
    return f"source {ex.source} sink {ex.target} edges {edges} reachable"


def render_cot(ex: Example) -> str:
    """Render the BFS frontier expansion as the explicit teacher reasoning trace."""
    steps = " ".join("| " + " ".join(str(n) for n in layer) for layer in ex.frontiers[1:])
    conclusion = "found" if ex.reachable else "none"
    return f"{steps} {conclusion}".strip()


def render_answer(ex: Example) -> str:
    """Render the yes/no answer (leading space so it tokenizes as one clean token)."""
    return " yes" if ex.reachable else " no"


def encode_example(ex: Example, num_continuous_thoughts: int) -> Dict[str, Any]:
    """
    Encode one :class:`Example` into the teacher/student token views.

    :param ex: The reachability instance.
    :param num_continuous_thoughts: ``K`` — number of latent slots in the student view.

    :returns: A dict with ``input_ids``/``label_mask`` (student), ``teacher_input_ids``/
        ``teacher_label_mask``, the ``<bot>`` and ``<distill>`` positions for each view,
        and metadata (``reachable``, ``depth``, ``frontiers``, ``target``, ``num_nodes``,
        ``seed``) used for evaluation and probing.
    """
    if num_continuous_thoughts < 1:
        raise ValueError(f"num_continuous_thoughts must be >= 1, got {num_continuous_thoughts}")

    q = encode_text(render_question(ex))
    cot = encode_text(render_cot(ex))
    ans = encode_text(render_answer(ex))
    k = num_continuous_thoughts

    # Teacher: question <bot> cot <eot> <distill> answer
    teacher_input_ids = q + [BOT] + cot + [EOT, DISTILL] + ans
    teacher_label_mask = (
        [False] * len(q) + [False] + [True] * len(cot) + [False, False] + [True] * len(ans)
    )
    teacher_bot_pos = len(q)
    teacher_distill_pos = len(q) + 1 + len(cot) + 1  # after <bot>, cot, <eot>

    # Student: question <bot> THOUGHT*K <eot> <distill> answer
    student_input_ids = q + [BOT] + [THOUGHT] * k + [EOT, DISTILL] + ans
    student_label_mask = (
        [False] * len(q) + [False] + [False] * k + [False, False] + [True] * len(ans)
    )
    student_bot_pos = len(q)
    student_distill_pos = len(q) + 1 + k + 1  # after <bot>, K thoughts, <eot>

    assert len(teacher_input_ids) == len(teacher_label_mask)
    assert len(student_input_ids) == len(student_label_mask)

    return {
        "input_ids": student_input_ids,
        "label_mask": student_label_mask,
        "teacher_input_ids": teacher_input_ids,
        "teacher_label_mask": teacher_label_mask,
        "bot_pos": student_bot_pos,
        "distill_pos": student_distill_pos,
        "teacher_bot_pos": teacher_bot_pos,
        "teacher_distill_pos": teacher_distill_pos,
        "num_continuous_thoughts": k,
        "answer_len": len(ans),
        # metadata for evaluation / probing (not consumed by the model)
        "reachable": ex.reachable,
        "depth": ex.depth,
        "target": ex.target,
        "num_nodes": ex.num_nodes,
        "frontiers": ex.frontiers,
        "seed": ex.seed,
    }
