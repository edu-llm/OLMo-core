"""Loss-weight sidecars over ONE token stream. Four conditions, two controls.

## Why one stream

Rendering each arm separately makes the streams different lengths, so at a fixed
token budget the arms see different numbers of fact exposures. Measured in the
previous corpus: the split arm got **0.497x** the biography exposures (79.4 vs
160.0 per entity) and **0.713x** the composition exposures, because split
documents were 1.40-2.01x longer. Iso-token and iso-exposure then become mutually
exclusive -- and the previous write-up asserted both, while its README claimed a
"byte-identical token stream" that could not exist under per-arm rendering.

Here the stream is produced once. An arm is a **loss-weight vector over that
stream**, so every arm sees identical tokens in identical order with identical
exposure counts. The fork disappears rather than being chosen.

## The four conditions

* `dense`    -- everything supervised. The memorisation baseline.
* `split`    -- payload spans masked. The treatment.
* `random_contig`  -- payloads supervised; an equal number of *contiguous*
  non-fact spans masked instead, matched on span length and relative position.
* `random_scatter` -- payloads supervised; an equal number of *scattered* tokens
  masked, matched on count and on difficulty.

## Why two controls rather than one

The equal-mass control is what separates "masking *facts* helps" from "masking
*hard tokens* helps". Without it a dense win reads as "any masking hurts" and the
result is uninterpretable. But the two matching criteria cannot be satisfied
together, and this was measured over 21.3B tokens:

    treatment (fact values)                       1.9598 nats
    contiguous, length + position matched         1.2712 nats   35.1% gap
    contiguous, length matched, position free     1.4987 nats   23.5% gap
    best possible contiguous + length matched     1.5667 nats   20.0% gap
    scattered, count + mean matched               1.7677 nats    9.8% gap
    scattered, k hardest available                1.8644 nats    4.9% gap

The blocking criterion is **span-length matching**, not the corpus and not
position: tokens hard enough exist, they are just not arranged in contiguous
runs. So contiguity and difficulty trade off, and the honest move is to ship both
controls -- they have orthogonal confounds -- and bracket the treatment effect
between them. If it survives both, it is not a masking artifact.

Two further biases fixed here:

* **Cue-window contamination.** 24.5% of control-masked tokens previously landed
  inside a value's cue window (the tokens immediately preceding a value), so the
  control was removing fact-relevant supervision across a quarter of its own mass
  and was biased toward the treatment. `cue_window` tokens are excluded from the
  candidate pool.
* **Tab-run controls.** The previous `random_control` spans were literal runs of
  tab characters, which are near-zero-entropy and so cannot be difficulty-matched
  even in principle. Controls must be drawn from real text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from memsplit.records import Span

CONDITIONS: tuple[str, ...] = ("dense", "split", "random_contig", "random_scatter")

# Roles the split condition masks. `restate` is deliberately NOT here: by the
# time the tail restates a value it is already in context, so supervising it
# trains an in-context copy rather than parametric recall. It is reported
# separately by `mask_report` because it is the one place the split arm receives
# gradient on value tokens.
MASKED_ROLES: frozenset[str] = frozenset({"payload"})

CUE_WINDOW = 3  # tokens before a payload treated as fact-relevant


class ControlUndersupply(RuntimeError):
    """Raised when a condition cannot be matched. Never silently degrade."""


@dataclass
class MaskPlan:
    """Per-document loss weights for every condition, over one shared stream."""

    n_tokens: int
    weights: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)

    def check(self) -> None:
        for name, w in self.weights.items():
            if w.shape != (self.n_tokens,):
                raise AssertionError(f"{name}: shape {w.shape} != ({self.n_tokens},)")
            if not np.isin(w, (0, 1)).all():
                raise AssertionError(f"{name}: weights must be binary")
        if self.weights["dense"].sum() != self.n_tokens:
            raise AssertionError("dense must supervise every token")


def _position_bin(start: int, end: int, n: int, n_bins: int = 8) -> int:
    mid = (start + end) / 2.0
    return min(n_bins - 1, int(n_bins * mid / max(n, 1)))


def _cue_tokens(spans: list[Span], n_tokens: int) -> set[int]:
    """Token indices in the cue window immediately before any payload span."""
    out: set[int] = set()
    for s in spans:
        if s.role != "payload":
            continue
        for i in range(max(0, s.start - CUE_WINDOW), s.start):
            out.add(i)
    return out


def derive_weights(
    spans: list[Span],
    n_tokens: int,
    seed: int,
    token_nll: np.ndarray | None = None,
    strict: bool = True,
    mask_restatements: bool = False,
) -> MaskPlan:
    """Build all four loss-weight vectors for one document.

    `token_nll` is an optional per-token difficulty table (nats under a frozen
    reference model). When supplied, `random_scatter` selects the *hardest*
    eligible tokens, which is what closes the difficulty gap from ~35% to ~5-10%.
    Without it the scattered control falls back to uniform sampling and the
    diagnostics record that it is count-matched but not difficulty-matched.

    `strict=True` raises rather than under-supplying a control. A silently
    short control is worse than no control, because it looks matched in the
    manifest.

    `mask_restatements` makes the answer-tail restatement an **ablatable axis**
    rather than a hidden assumption. Default False, which supervises it: by the
    tail the value is already in context, so predicting it trains an in-context
    copy and is what makes a scoreable `Answer:` line possible. But it is not
    free -- measured on the depth-3 corpus, **31% of value-token mass sits in the
    supervised tail**, so the split arm does receive gradient at value positions,
    just never on the (entity, attribute) -> value mapping. Setting this True
    masks those occurrences too, and answers must then be read out of the
    retrieved lookup span. Run it as a robustness check on H1; report the share
    either way.
    """
    weights = {c: np.ones(n_tokens, dtype=np.uint8) for c in CONDITIONS}

    split_roles = set(MASKED_ROLES) | ({"restate"} if mask_restatements else set())
    payloads = [s for s in spans if s.role in split_roles]
    for s in payloads:
        weights["split"][s.start : s.end] = 0
    n_masked = int(sum(s.end - s.start for s in payloads))

    cue = _cue_tokens(spans, n_tokens)
    # Eligible control territory: plain prose only. Never queries (the skill
    # under test), never payloads (that is the treatment), never restatements
    # (they are copies of values), never cue windows.
    eligible_spans = [s for s in spans if s.role == "plain"]
    eligible_tokens = sorted(
        i
        for s in eligible_spans
        for i in range(s.start, s.end)
        if i not in cue
    )

    diag: dict = {
        "n_tokens": n_tokens,
        "n_payload_tokens": n_masked,
        "n_payload_spans": len(payloads),
        "n_eligible_control_tokens": len(eligible_tokens),
        "n_restate_tokens": int(
            sum(s.end - s.start for s in spans if s.role == "restate")
        ),
        "cue_window_excluded": len(cue),
        "difficulty_table": token_nll is not None,
        "mask_restatements": bool(mask_restatements),
    }

    # ---- contiguous control: match span length and relative position bin ----
    rng = random.Random(f"contig:{seed}")
    demands: dict[tuple[int, int], int] = {}
    for s in payloads:
        key = (s.end - s.start, _position_bin(s.start, s.end, n_tokens))
        demands[key] = demands.get(key, 0) + 1

    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for s in eligible_spans:
        length = s.end - s.start
        for L, _ in demands:
            if length < L:
                continue
            for off in range(s.start, s.end - L + 1):
                if any(i in cue for i in range(off, off + L)):
                    continue
                key = (L, _position_bin(off, off + L, n_tokens))
                candidates.setdefault(key, []).append((off, off + L))

    # Placement must avoid collisions. Sampling each key's spans independently
    # lets two chosen spans overlap, and then the masked *union* is smaller than
    # the sum of the span lengths -- a silent under-match that still looks
    # count-correct if you only add up the demands. Reserve as you go.
    placed_contig = 0
    taken: set[int] = set()

    def _place(pool: list[tuple[int, int]], want: int) -> int:
        """Greedily place up to `want` non-overlapping spans from `pool`."""
        nonlocal placed_contig
        order = list(pool)
        rng.shuffle(order)
        placed = 0
        for a, b in order:
            if placed >= want:
                break
            if any(i in taken for i in range(a, b)):
                continue
            for i in range(a, b):
                taken.add(i)
            weights["random_contig"][a:b] = 0
            placed_contig += b - a
            placed += 1
        return placed

    for key in sorted(demands):
        want = demands[key]
        got = _place(candidates.get(key, []), want)
        if got < want:
            # Relax the position bin before giving up. Position matching is the
            # criterion known to cost difficulty, so losing it is cheaper than
            # losing the control entirely -- and it is recorded either way.
            L = key[0]
            wider = [c for k, cs in candidates.items() if k[0] == L for c in cs]
            got += _place(wider, want - got)
            diag.setdefault("position_relaxed_for", []).append(list(key))
        if got < want and strict:
            raise ControlUndersupply(
                f"contiguous control needs {want} non-overlapping spans of "
                f"length {key[0]}, placed {got}"
            )
        if got < want:
            diag.setdefault("undersupplied", []).append([list(key), got, want])

    if placed_contig != n_masked and strict:
        raise ControlUndersupply(
            f"contiguous control masked {placed_contig} tokens against "
            f"{n_masked} payload tokens; equal mass is the point of the control"
        )

    # ---- scattered control: match token COUNT, and difficulty if we can ----
    if len(eligible_tokens) < n_masked:
        if strict:
            raise ControlUndersupply(
                f"scattered control needs {n_masked} tokens, "
                f"only {len(eligible_tokens)} eligible"
            )
        pick = eligible_tokens
    elif token_nll is not None:
        pick = sorted(eligible_tokens, key=lambda i: -float(token_nll[i]))[:n_masked]
    else:
        pick = random.Random(f"scatter:{seed}").sample(eligible_tokens, n_masked)
    for i in pick:
        weights["random_scatter"][i] = 0

    diag["n_contig_masked"] = placed_contig
    diag["n_scatter_masked"] = len(pick)
    diag["count_matched_contig"] = placed_contig == n_masked
    diag["count_matched_scatter"] = len(pick) == n_masked
    if token_nll is not None and n_masked:
        pay_idx = [i for s in payloads for i in range(s.start, s.end)]
        diag["mean_nll_payload"] = float(np.mean(token_nll[pay_idx]))
        if placed_contig:
            ci = np.where(weights["random_contig"] == 0)[0]
            diag["mean_nll_contig"] = float(np.mean(token_nll[ci]))
        if pick:
            diag["mean_nll_scatter"] = float(np.mean(token_nll[pick]))

    plan = MaskPlan(n_tokens=n_tokens, weights=weights, diagnostics=diag)
    plan.check()
    return plan


def difficulty_gap(mean_treatment: float, mean_control: float) -> float:
    """Relative NLL gap between treatment and control masked spans.

    The preregistered tolerance is 20%. Report it per condition; the contiguous
    control is expected to fail it (measured 23.5-35.1%) and the scattered
    control is expected to pass (measured 4.9-9.8%). Reporting both is the point.
    """
    if mean_treatment == 0:
        return float("nan")
    return abs(mean_treatment - mean_control) / mean_treatment


def aggregate_report(diags: list[dict]) -> dict:
    """Corpus-level mask manifest. Every number a referee will ask for."""
    if not diags:
        return {}
    tot = sum(d["n_tokens"] for d in diags)
    pay = sum(d["n_payload_tokens"] for d in diags)
    out = {
        "n_docs": len(diags),
        "n_tokens": tot,
        "masked_token_frac_split": pay / tot if tot else 0.0,
        "n_payload_tokens": pay,
        "n_restate_tokens": sum(d["n_restate_tokens"] for d in diags),
        "count_matched_contig": all(d["count_matched_contig"] for d in diags),
        "count_matched_scatter": all(d["count_matched_scatter"] for d in diags),
        "docs_with_position_relaxed": sum(
            1 for d in diags if d.get("position_relaxed_for")
        ),
        "difficulty_table_used": all(d["difficulty_table"] for d in diags),
    }
    # Difficulty gaps, if a table was supplied.
    for tag in ("contig", "scatter"):
        key = f"mean_nll_{tag}"
        vals = [d[key] for d in diags if key in d]
        pays = [d["mean_nll_payload"] for d in diags if "mean_nll_payload" in d]
        if vals and pays:
            mt, mc = float(np.mean(pays)), float(np.mean(vals))
            out[f"mean_nll_payload"] = mt
            out[key] = mc
            out[f"difficulty_gap_{tag}"] = difficulty_gap(mt, mc)
            out[f"difficulty_gap_{tag}_within_20pct"] = difficulty_gap(mt, mc) <= 0.20
    # The split arm receives gradient on value tokens here and nowhere else.
    out["restate_share_of_value_tokens"] = (
        out["n_restate_tokens"] / (out["n_restate_tokens"] + pay)
        if (out["n_restate_tokens"] + pay)
        else 0.0
    )
    return out
