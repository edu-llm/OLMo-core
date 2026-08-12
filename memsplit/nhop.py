"""n-hop relational composition, depth-parameterised, with integrity gates.

## What this replaces

The previous composition generator hardcoded depth 2 at thirteen distinct sites
and rendered the trace with **exactly one phrasing**. Two consequences the audit
found:

* A 3-hop probe on the fine-tuned split model scored **0.00%**, emitting exactly
  two lookups per item and silently dropping the middle relation -- it had
  pattern-matched a depth-2 template rather than learned a recursion. But the
  3-hop *surface form* had never been trained either, so depth-OOD was confounded
  with format-OOD. That 0% is therefore not evidence that depth generalisation
  fails; it is evidence that the prescribed mixed-depth curriculum was skipped.
* Single-phrasing storage is the failure mode the project measured elsewhere:
  dense answered the same question at **83% under one phrasing and 1.3% under
  another**, i.e. facts were stored as pattern-slot -> value rather than
  (entity, attribute) -> value.

So depth is a first-class parameter here, the surface form is held fixed *across*
depths while varying *within* each depth, and there are >= 10 templates per slot.

## Graph structure

Entities are assigned to **layers** and every bridge edge goes strictly from
layer *i* to layer *i+1*. This makes distinct-node paths and acyclicity
structural rather than something to reject-sample for: a path of length *m*
visits *m+1* distinct entities by construction. The old `assign_bridges` produced
unconstrained functional edges, which is fine at depth 2 but self-intersects and
produces 2-cycles at depth >= 3.

## The p-to-the-n null

Report `expected_chain_accuracy(p, n)` alongside every depth curve. If per-hop
reliability is *p*, end-to-end success is ~p**n, so an arm with p~1.0 beats one
with p~0.93 by a margin that **grows with depth for purely arithmetic reasons**.
The previous two-hop result is quantitatively that: 0.996**2 - 0.956**2 = 7.81pp
predicted against 7.4pp observed. A depth sweep that does not overlay this curve
will manufacture an impressive-looking widening gap that means nothing. The
reasoning quantity is **conditional per-hop accuracy**, not end-to-end.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from memsplit.bios import RELATION_PHRASES, BioRecord
from memsplit.records import Doc, QAItem, lookup_roles, lookup_segments, merge_plain

BRIDGE_RELATIONS: tuple[str, ...] = ("mentor", "advisor", "collaborator")

# Question phrasings. The possessive chain ("X's mentor's advisor") is built
# separately so one template works at every depth -- that is what keeps surface
# form constant across the depth axis while still varying within it.
_QUESTION_TEMPLATES: list[str] = [
    "Question: What is the {attr} of {chain}?",
    "Question: Which {attr} belongs to {chain}?",
    "Question: Give the {attr} of {chain}.",
    "Question: Report the {attr} of {chain}.",
    "Question: What {attr} is recorded for {chain}?",
    "Question: According to the directory, what is the {attr} of {chain}?",
    "Question: State the {attr} of {chain}.",
    "Question: For {chain}, what is the {attr}?",
    "Question: Identify the {attr} of {chain}.",
    "Question: Look up the {attr} of {chain}.",
]

# First step: names the start entity explicitly.
_FIRST_STEP: list[str] = [
    "{subj}'s {rel} is",
    "The {rel} of {subj} is",
    "Records list {subj}'s {rel} as",
    "For {subj}, the {rel} is",
    "{subj} has as {rel}",
    "Listed under {subj}, the {rel} is",
    "Start with {subj}: the {rel} is",
    "Take {subj}. Their {rel} is",
    "Looking up {subj}, the {rel} is",
    "{subj} is linked by {rel} to",
]

# Later steps must NOT re-name the intermediate entity. The model has to carry
# the previously retrieved name into the next query itself; that copy is the
# capability under test, and naming it in prose would give it away.
_NEXT_STEP: list[str] = [
    ". Their {rel} is",
    ". That person's {rel} is",
    ". In turn, their {rel} is",
    ". Their own {rel} is",
    ". Following {rel} from there gives",
    ". From there, the {rel} is",
    ". Their listed {rel} is",
    ". Next, their {rel} is",
    ". Their recorded {rel} is",
    ". Continuing, their {rel} is",
]

_FINAL_STEP: list[str] = [
    ". Their {attr} is",
    ". That person's {attr} is",
    ". Their recorded {attr} is",
    ". Their listed {attr} is",
    ". The {attr} there is",
    ". Their own {attr} is",
    ". Finally, their {attr} is",
    ". Their {attr} on record is",
    ". For them, the {attr} is",
    ". Their {attr} reads",
]

# Tails are `(prefix, middle)` and render as prefix + value + middle + value, so
# the two value occurrences can be role-tagged separately from the connective
# prose around them. Tagging the whole tail `restate` would inflate the reported
# restated-value mass with punctuation and stock phrasing -- and that number is an
# honesty number, since the restatement is the one place the split arm receives
# gradient on value tokens.
_TAIL: list[tuple[str, str]] = [
    (". So the answer is ", ".\nAnswer: "),
    (". The answer is therefore ", ".\nAnswer: "),
    (". Hence the answer is ", ".\nAnswer: "),
    (". That gives ", ".\nAnswer: "),
    (". So it is ", ".\nAnswer: "),
]


def n_templates() -> dict[str, int]:
    return {
        "question": len(_QUESTION_TEMPLATES),
        "first_step": len(_FIRST_STEP),
        "next_step": len(_NEXT_STEP),
        "final_step": len(_FINAL_STEP),
        "tail": len(_TAIL),
    }


def _rng(*parts) -> random.Random:
    return random.Random(":".join(str(p) for p in parts))


# ------------------------------------------------------------------- the graph


@dataclass
class RelGraph:
    """Layered functional bridge graph. Edges go layer i -> layer i+1 only."""

    edges: dict[int, dict[str, int]]
    layer: dict[int, int]
    n_layers: int
    relations: tuple[str, ...]

    def follow(self, start: int, chain: tuple[str, ...]) -> int | None:
        cur = start
        for rel in chain:
            nxt = self.edges.get(cur, {}).get(rel)
            if nxt is None:
                return None
            cur = nxt
        return cur

    def path_nodes(self, start: int, chain: tuple[str, ...]) -> list[int] | None:
        nodes = [start]
        cur = start
        for rel in chain:
            nxt = self.edges.get(cur, {}).get(rel)
            if nxt is None:
                return None
            nodes.append(nxt)
            cur = nxt
        return nodes

    def max_depth_from(self, eid: int) -> int:
        """Hops available before running out of layers."""
        return self.n_layers - 1 - self.layer[eid]


def build_graph(
    records: list[BioRecord],
    n_layers: int,
    seed: int = 0,
    relations: tuple[str, ...] = BRIDGE_RELATIONS,
) -> RelGraph:
    """Assign layers and draw functional edges strictly forward one layer.

    Every entity below the top layer gets one target per relation, so a chain of
    length m from a layer-0 entity always exists and always visits m+1 distinct
    entities.
    """
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2")
    ids = [r.entity_id for r in records]
    if len(ids) < n_layers * 2:
        raise ValueError(f"need >= {n_layers * 2} entities for {n_layers} layers")

    rng = random.Random(f"layers:{seed}")
    shuffled = list(ids)
    rng.shuffle(shuffled)
    layer = {eid: i % n_layers for i, eid in enumerate(shuffled)}
    by_layer: dict[int, list[int]] = {i: [] for i in range(n_layers)}
    for eid, lay in layer.items():
        by_layer[lay].append(eid)
    for lay in by_layer:
        by_layer[lay].sort()

    edges: dict[int, dict[str, int]] = {eid: {} for eid in ids}
    for rel in relations:
        r = random.Random(f"edges:{seed}:{rel}")
        for lay in range(n_layers - 1):
            targets = by_layer[lay + 1]
            for eid in by_layer[lay]:
                edges[eid][rel] = targets[r.randrange(len(targets))]
    return RelGraph(edges=edges, layer=layer, n_layers=n_layers,
                    relations=tuple(relations))


# --------------------------------------------------------------- shortcut gate


def has_shortcut(
    graph: RelGraph,
    by_id: dict[int, BioRecord],
    start: int,
    chain: tuple[str, ...],
    attr: str,
    value: str,
) -> bool:
    """True if some SHORTER relation chain from `start` also yields `value`.

    Without this, a "depth-4" item can be answerable in two hops by coincidence:
    attribute values come from pools of a few hundred, so collisions are not
    rare. An item admitting a shorter derivation is not a depth-4 item, and the
    depth axis stops meaning anything. Also catches the degenerate case where the
    start entity's own attribute already equals the answer.
    """
    if by_id[start].attrs.get(attr) == value:
        return True
    for m in range(1, len(chain)):
        for alt in itertools.product(graph.relations, repeat=m):
            end = graph.follow(start, alt)
            if end is not None and by_id[end].attrs.get(attr) == value:
                return True
    return False


def eligible_starts(graph: RelGraph, max_depth: int) -> list[int]:
    """Entities that can start a chain of *every* depth up to `max_depth`.

    This matters more than it looks. In a layered graph the depth available from
    an entity is fixed by its layer, so if starts are drawn per-depth then deep
    items come from low layers and shallow items from high layers -- and **depth
    becomes correlated with entity identity**. A model could then separate the
    depth strata by recognising which entities appear in them, and the depth axis
    would be partly an entity axis.

    Drawing every depth from this one pool makes depth orthogonal to entity: the
    same start entity appears at depth 1 and at depth 5, so the only thing that
    differs across strata is chain length. Entity novelty is handled separately by
    the population split, which is the axis it belongs on.
    """
    return sorted(e for e in graph.edges if graph.max_depth_from(e) >= max_depth)


def sample_item(
    graph: RelGraph,
    by_id: dict[int, BioRecord],
    start: int,
    depth: int,
    attr: str,
    seed: int,
    max_tries: int = 32,
) -> tuple[tuple[str, ...], int, str] | None:
    """Draw a shortcut-free relation chain of exactly `depth` hops.

    Returns `(chain, final_entity_id, value)` or None if no clean chain exists.
    `depth` counts bridge hops; reading the attribute is not a hop, so a depth-1
    item is the classic two-fact composition and needs 2 lookups.
    """
    if graph.max_depth_from(start) < depth:
        return None
    rng = _rng("chain", seed, start, depth, attr)
    for _ in range(max_tries):
        chain = tuple(rng.choice(graph.relations) for _ in range(depth))
        end = graph.follow(start, chain)
        if end is None:
            continue
        value = by_id[end].attrs.get(attr)
        if not value:
            continue
        if has_shortcut(graph, by_id, start, chain, attr, value):
            continue
        return chain, end, value
    return None


# ------------------------------------------------------------------- rendering


def possessive_chain(name: str, chain: tuple[str, ...]) -> str:
    """`"Ada Vale"` + `("mentor","advisor")` -> `"Ada Vale's mentor's advisor"`."""
    out = f"{name}'s {chain[0]}"
    for rel in chain[1:]:
        out += f"'s {rel}"
    return out


def question_text(name: str, chain: tuple[str, ...], attr: str, variant: int) -> str:
    tmpl = _QUESTION_TEMPLATES[variant % len(_QUESTION_TEMPLATES)]
    return tmpl.format(attr=RELATION_PHRASES[attr],
                       chain=possessive_chain(name, chain))


def prompt_text(name: str, chain: tuple[str, ...], attr: str, variant: int) -> str:
    """The eval prompt: the question plus the trace opener, nothing more.

    Must be a strict prefix of the training document so the model is asked
    exactly what it was trained on. `test_nhop.py` asserts the prefix property.
    """
    return question_text(name, chain, attr, variant) + "\nReasoning:"


def _template_variant(start: int, chain: tuple[str, ...], attr: str) -> int:
    """Question variant, deterministic in the item so doc and eval prompt agree."""
    return _rng("variant", start, chain, attr).randrange(len(_QUESTION_TEMPLATES))


def _hop_keys(
    graph: RelGraph, by_id: dict[int, BioRecord], start: int,
    chain: tuple[str, ...], attr: str,
) -> list[str]:
    """The exact lookup keys a correct trace must emit, in order."""
    keys: list[str] = []
    cur = by_id[start]
    for rel in chain:
        keys.append(f"{cur.name}, {rel}")
        cur = by_id[graph.edges[cur.entity_id][rel]]
    keys.append(f"{cur.name}, {attr}")
    return keys


def render_doc(
    graph: RelGraph,
    by_id: dict[int, BioRecord],
    start: int,
    chain: tuple[str, ...],
    attr: str,
    value: str,
    exposure: int,
) -> Doc:
    """One n-hop document, as role-tagged segments over a single token stream.

    Both arms read this same stream; they differ only in which loss-weight
    sidecar `memsplit.masking` derives from the roles. That is what makes
    iso-token and iso-exposure the same condition rather than a fork you have to
    choose between (and then, as the previous generation did, claim both).
    """
    rng = _rng("doc", start, chain, attr, exposure)
    variant = _template_variant(start, chain, attr)

    subj = by_id[start]
    segs: list[tuple[str, bool]] = []
    roles: list[str] = []

    def plain(text: str) -> None:
        segs.append((text, False))
        roles.append("plain")

    def lookup(subject: str, rel: str, val: str) -> None:
        segs.extend(lookup_segments(subject, rel, val))
        roles.extend(lookup_roles())

    plain(prompt_text(subj.name, chain, attr, variant) + " ")

    cur = subj
    first = _FIRST_STEP[rng.randrange(len(_FIRST_STEP))]
    plain(first.format(subj=cur.name, rel=chain[0]))
    nxt = by_id[graph.edges[cur.entity_id][chain[0]]]
    lookup(cur.name, chain[0], nxt.name)
    cur = nxt

    for rel in chain[1:]:
        step = _NEXT_STEP[rng.randrange(len(_NEXT_STEP))]
        plain(step.format(rel=rel))
        nxt = by_id[graph.edges[cur.entity_id][rel]]
        lookup(cur.name, rel, nxt.name)
        cur = nxt

    final = _FINAL_STEP[rng.randrange(len(_FINAL_STEP))]
    plain(final.format(attr=RELATION_PHRASES[attr]))
    lookup(cur.name, attr, value)

    # The tail restates the value twice. Both occurrences are supervised in EVERY
    # condition, because by then the value is already in context and reproducing
    # it is an in-context copy rather than parametric recall. This is the one
    # place the split arm receives gradient on value tokens, so each occurrence is
    # tagged `restate` while the connective prose stays `plain` -- the previous
    # write-up's blanket "value gradients are masked by construction" was
    # inaccurate here, and the accounting should be exact rather than generous.
    pre, mid = _TAIL[rng.randrange(len(_TAIL))]
    plain(pre)
    segs.append((value, False))
    roles.append("restate")
    plain(mid)
    segs.append((value, False))
    roles.append("restate")

    segs, roles = merge_plain(segs, roles)
    return Doc(
        kind=f"nhop{len(chain)}",
        segments=segs,
        roles=roles,
        meta={
            "start_id": start,
            "start_name": subj.name,
            "chain": list(chain),
            "attr": attr,
            "depth": len(chain),
            "final_id": cur.entity_id,
            "value": value,
            "exposure": exposure,
            "template": variant,
            "hop_keys": _hop_keys(graph, by_id, start, chain, attr),
        },
    )


def make_item(
    graph: RelGraph,
    by_id: dict[int, BioRecord],
    start: int,
    chain: tuple[str, ...],
    attr: str,
    value: str,
    population: str,
) -> QAItem:
    subj = by_id[start]
    variant = _template_variant(start, chain, attr)
    return QAItem(
        task=f"nhop{len(chain)}",
        prompt=prompt_text(subj.name, chain, attr, variant),
        answer=value,
        meta={
            "population": population,
            "depth": len(chain),
            "start_id": start,
            "chain": list(chain),
            "attr": attr,
            "hop_keys": _hop_keys(graph, by_id, start, chain, attr),
            "n_lookups_expected": len(chain) + 1,
        },
    )


# ------------------------------------------------------------ the p**n baseline


def expected_chain_accuracy(per_hop: float, n_lookups: int) -> float:
    """p**n -- the null every depth curve must be read against."""
    return per_hop**n_lookups


def pn_table(p_by_arm: dict[str, float], depths: list[int]) -> dict:
    """Predicted end-to-end accuracy and arm gap at each depth under p**n.

    A widening gap is the *expected* consequence of a per-hop reliability
    difference, not evidence about reasoning. Only a departure from these numbers
    is. Sanity check on the previous generation's two-hop result: with
    p_split=0.996 and p_dense=0.956 at depth 1 (2 lookups) this predicts a
    7.81pp gap against the 7.4pp observed.
    """
    arms = sorted(p_by_arm)
    rows = []
    for d in depths:
        n = d + 1  # bridge hops plus the attribute read
        pred = {a: expected_chain_accuracy(p_by_arm[a], n) for a in arms}
        row = {"depth": d, "n_lookups": n}
        row.update({f"pred_{a}": pred[a] for a in arms})
        if len(arms) == 2:
            row["pred_gap_pp"] = 100.0 * (pred[arms[1]] - pred[arms[0]])
        rows.append(row)
    return {"per_hop": dict(p_by_arm), "rows": rows}


# ------------------------------------------------------- atomic and bridge docs

# Atomic fact documents. These exist so the DENSE arm can actually memorise the
# constituent facts -- without them the contrast is vacuous, because both arms
# would simply read every fact from context and neither would be storing anything.
# >= 10 templates each, for the same pattern-binding reason as the trace templates.
_ATOMIC_TEMPLATES: list[tuple[str, str]] = [
    ("{name}'s {attr} is ", "."),
    ("The {attr} of {name} is ", "."),
    ("Records list {name}'s {attr} as ", "."),
    ("For {name}, the {attr} is ", "."),
    ("{name} has {attr} ", "."),
    ("Listed under {name}, the {attr} is ", "."),
    ("According to the directory, {name}'s {attr} is ", "."),
    ("{name} -- {attr}: ", "."),
    ("On file for {name}, the {attr} reads ", "."),
    ("The directory gives {name}'s {attr} as ", "."),
]

_BRIDGE_TEMPLATES: list[tuple[str, str]] = [
    ("{name}'s {rel} is ", "."),
    ("The {rel} of {name} is ", "."),
    ("Records name ", " as the {rel} of {name}."),
    ("{name} counts ", " as a {rel}."),
    ("Listed beside {name} under \"{rel}\" is ", "."),
    ("The {rel} assigned to {name} is ", "."),
    ("{name} has long regarded ", " as a trusted {rel}."),
    ("For {name}, the {rel} on record is ", "."),
    ("According to the directory, {name}'s {rel} is ", "."),
    ("{name} is linked by {rel} to ", "."),
]


def _fact_doc(prefix: str, value: str, suffix: str, kind: str, meta: dict) -> Doc:
    """One fact, value wrapped as a lookup so the masker can find it."""
    segs = [(prefix.rstrip(" "), False)]
    roles = ["plain"]
    segs.extend(lookup_segments(meta["key_subject"], meta["key_relation"], value))
    roles.extend(lookup_roles())
    segs.append((suffix, False))
    roles.append("plain")
    segs, roles = merge_plain(segs, roles)
    return Doc(kind=kind, segments=segs, roles=roles, meta=meta)


def render_atomic_doc(rec: BioRecord, attr: str, exposure: int) -> Doc:
    rng = _rng("atomic", rec.entity_id, attr, exposure)
    pre, suf = _ATOMIC_TEMPLATES[rng.randrange(len(_ATOMIC_TEMPLATES))]
    phrase = RELATION_PHRASES[attr]
    return _fact_doc(
        pre.format(name=rec.name, attr=phrase),
        rec.attrs[attr],
        suf.format(name=rec.name, attr=phrase),
        "atomic",
        {"entity_id": rec.entity_id, "attr": attr, "exposure": exposure,
         "key_subject": rec.name, "key_relation": attr},
    )


def render_bridge_doc(
    rec: BioRecord, relation: str, target: BioRecord, exposure: int
) -> Doc:
    rng = _rng("bridge", rec.entity_id, relation, exposure)
    pre, suf = _BRIDGE_TEMPLATES[rng.randrange(len(_BRIDGE_TEMPLATES))]
    return _fact_doc(
        pre.format(name=rec.name, rel=relation),
        target.name,
        suf.format(name=rec.name, rel=relation),
        "bridge",
        {"entity_id": rec.entity_id, "relation": relation, "exposure": exposure,
         "target_id": target.entity_id,
         "key_subject": rec.name, "key_relation": relation},
    )


def n_atomic_templates() -> dict[str, int]:
    return {"atomic": len(_ATOMIC_TEMPLATES), "bridge": len(_BRIDGE_TEMPLATES)}
