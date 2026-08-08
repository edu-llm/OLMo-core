"""Reference shard builder for Mizar — the pattern the other three jobs follow.

Produces JSONL where each line is one training example with the fact block, the
goal, the target, the rendered text, and explicit mask offsets. Also emits the
held-out fact set so every machine masks the same 500 facts.

Three decisions are baked in, per the plan:
  * article-local labels are resolved (`Th5` inside POLYRED -> `POLYRED:5`)
  * examples whose block cannot be fully filled are DROPPED, not rendered partial
  * the held-out 500 are drawn from facts cited once or twice, and both the proofs
    citing them and their own proofs are removed from training

Usage:
    python scripts/build_mizar_shard.py --out /tmp/dscount/shards
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

MIZAR = "/tmp/dscount/mizar/html2"
HDR = "I know these mathematical statements:"
SEP = "---"

# The colon belongs to the LABEL, not to the header: a labelled theorem reads
# `theorem Th2: :: AFINSQ_1:2` but an unlabelled one is just
# `theorem :: AFINSQ_1:1`. Requiring the colon unconditionally hid every
# unlabelled theorem — half the MML, and the reason 38% of citations used to
# resolve to nothing.
THM = re.compile(r"^theorem(?:\s+(\w+)\s*:)?\s*::\s*([A-Z_0-9]+:\d+)\s*$")
DEFT = re.compile(r"^::\s*deftheorem(?:\s+(\w+))?\s+defines\s+\S+\s+"
                  r"([A-Z_0-9]+:def_\d+)")
SCHEME = re.compile(
    r"^scheme\s*::\s*([A-Z_0-9]+):sch(?:_|\s+)(\d+)\s*$"
)
LOCAL_LEMMA = re.compile(r"^(Lm\w*):\s*(.*)$")
TOP_LEVEL_BOUNDARY = re.compile(
    r"^(?:definition|registration|notation)\b|^canceled;\s*$"
)
POST_PROOF_BOUNDARY = re.compile(
    r"^(?:begin|consider|deffunc|defpred|reconsider|scheme|set)\b"
)
POST_PROOF_THEN_DECLARATION = re.compile(
    r"""
    then\s+
    (?:
        reconsider\b[^;]*;
        |
        [A-Za-z]\w*\s*:\s*.+?\s+(?:by|from)\s+[^;]+;
    )
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
PROOF = re.compile(r"(?:^|\s)proof(?=\s|$)")
BY = re.compile(r"\b(?:by|from)\s+([^;]*?);")
GREF = re.compile(
    r"\b([A-Z][A-Z_0-9]*):"
    r"(?:(def|sch)(?:_|\s*)?(\d+)|(\d+))"
    r"((?:\s*,\s*(?:(?:def|sch)(?:_|\s*)?)?\d+)*)"
)
QUALIFIED_LABEL = re.compile(r"\b([A-Z][A-Z_0-9]*):([A-Za-z]\w*)\b")
IDENTIFIER = re.compile(r"\b[A-Za-z]\w*\b")
CONVENTIONAL_LOCAL = re.compile(r"^(?:Th|Def|Lm|Sch)\w*$")
BLOCK_OPENERS = frozenset({"now", "hereby", "suppose", "case", "percases"})


def is_canceled(statement):
    """Whether an exported declaration is only a cancellation marker."""
    return statement.strip().lower() == "canceled;"


def _masked_mizar(text):
    """Mask comments and strings while preserving source positions."""
    masked = list(text)
    i = 0
    while i < len(text):
        if text.startswith("::", i):
            while i < len(text) and text[i] != "\n":
                masked[i] = " "
                i += 1
            continue
        if text[i] == '"':
            # In Mizar ``"`` is usually an operator (inverse image, quoted
            # symbols), not a string delimiter. Only treat it as a string when
            # it starts in expression-value position and closes on this line.
            previous = i - 1
            while previous >= 0 and text[previous] in " \t":
                previous -= 1
            line_end = text.find("\n", i)
            if line_end < 0:
                line_end = len(text)
            if previous < 0 or text[previous] in "=([{,:;":
                close = text.find('"', i + 1, line_end)
                if close >= 0:
                    for position in range(i, close + 1):
                        masked[position] = " "
                    i = close + 1
                    continue
        i += 1
    return "".join(masked)


def _is_block_opener(token):
    if token == "proof":
        return True
    lowered = token.lower()
    return (
        lowered in BLOCK_OPENERS
        or lowered.startswith(("now__", "percases"))
        or re.fullmatch(r"(?:suppose|case)[a-z]\w*", lowered) is not None
    )


def _outer_proof_bounds(content):
    """Locate a balanced outer proof without interpreting trailing text."""
    masked = _masked_mizar(content)
    tokens = list(re.finditer(r"\b[A-Za-z_]\w*\b|;", masked))
    proof_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.group(0) == "proof"
        ),
        None,
    )
    if proof_index is None:
        return None

    proof_token = tokens[proof_index]
    stack = ["proof"]
    for index in range(proof_index + 1, len(tokens)):
        raw_token = tokens[index].group(0)
        token = raw_token.lower()
        if _is_block_opener(raw_token):
            stack.append(raw_token)
            continue
        if token != "end":
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].group(0) != ";":
            return proof_token.start(), None, None, None
        stack.pop()
        if stack:
            continue
        return (
            proof_token.start(),
            proof_token.end(),
            tokens[index].start(),
            tokens[index + 1].end(),
        )
    return proof_token.start(), None, None, None


def split_outer_proof(content):
    """Return the outer proof span and body, rejecting unsupported block shapes.

    The returned tuple is ``(proof_start, body)``. ``proof_start`` is ``None``
    when there is no proof keyword. A present keyword with a ``None`` body is
    malformed or incomplete.
    """
    bounds = _outer_proof_bounds(content)
    if bounds is None:
        return None, None
    proof_start, body_start, body_end, proof_end = bounds
    if proof_end is None:
        return proof_start, None
    if _masked_mizar(content[proof_end:]).strip():
        return proof_start, None
    body = content[body_start:body_end].strip()
    return proof_start, body or None


def _post_proof_declaration(content):
    """Find a structural declaration following a completed theorem proof."""
    bounds = _outer_proof_bounds(content)
    if bounds is None or bounds[3] is None:
        return None
    proof_end = bounds[3]
    suffix = content[proof_end:]
    masked_suffix = _masked_mizar(suffix)
    first = re.search(r"\S", masked_suffix)
    if first is None:
        return None
    boundary = first.start()
    if re.search(r"\r?\n[ \t]*\r?\n", suffix[:boundary]) is None:
        return None

    declaration = content[proof_end + boundary:]
    first_line = declaration.splitlines()[0]
    if POST_PROOF_BOUNDARY.match(first_line.lstrip()):
        return proof_end + boundary, "boundary", ()
    compact = re.sub(r"\s+", " ", _masked_mizar(declaration)).strip()
    then_match = POST_PROOF_THEN_DECLARATION.fullmatch(compact)
    if then_match is None:
        first_end = compact.find(";")
        if first_end >= 0:
            then_match = POST_PROOF_THEN_DECLARATION.fullmatch(
                compact[first_end + 1 :].strip()
            )
    if then_match is not None:
        return proof_end + boundary, "boundary", ()

    label_match = re.match(r"\s*([A-Za-z]\w*)\s*:\s*(.*)", first_line)
    if label_match is None:
        return None
    declaration_bounds = _outer_proof_bounds(declaration)
    if declaration_bounds is None or declaration_bounds[3] is None:
        return None
    label, statement_start = label_match.groups()
    return proof_end + boundary, "local_lemma", (label, statement_start)


def _declaration_sections(text):
    """Yield bounded top-level declarations from an html2 article.

    A declaration ends at the next theorem, article-local lemma, definition
    theorem, scheme, definition, registration, notation, or cancellation
    marker. In particular, a later ``proof`` token can never be consumed by an
    earlier theorem.
    """
    lines = text.splitlines(keepends=True)
    starts = []
    offset = 0
    for line in lines:
        plain = line.rstrip("\r\n")
        match = THM.match(plain)
        if match:
            starts.append((offset, offset + len(line), "theorem", match.groups()))
        else:
            match = DEFT.match(plain)
            if match:
                starts.append(
                    (offset, offset + len(line), "definition_theorem", match.groups())
                )
            else:
                match = SCHEME.match(plain)
                if match:
                    article, number = match.groups()
                    starts.append(
                        (
                            offset,
                            offset + len(line),
                            "scheme",
                            (f"{article}:sch_{number}",),
                        )
                    )
                else:
                    match = LOCAL_LEMMA.match(plain)
                    if match:
                        starts.append(
                            (
                                offset,
                                offset + len(line),
                                "local_lemma",
                                match.groups(),
                            )
                        )
                    elif TOP_LEVEL_BOUNDARY.match(plain):
                        starts.append((offset, offset + len(line), "boundary", ()))
        offset += len(line)

    inserted = []
    for i, (_, content_start, kind, _) in enumerate(starts):
        if kind != "theorem":
            continue
        content_end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        declaration = _post_proof_declaration(text[content_start:content_end])
        if declaration is None:
            continue
        relative_start, boundary_kind, metadata = declaration
        start = content_start + relative_start
        line_end = text.find("\n", start)
        if line_end < 0:
            line_end = len(text)
        else:
            line_end += 1
        inserted.append((start, line_end, boundary_kind, metadata))

    starts.extend(inserted)
    starts.sort(key=lambda item: item[0])
    for i, (_, content_start, kind, metadata) in enumerate(starts):
        content_end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        yield kind, metadata, text[content_start:content_end]


def _statement_and_proof(content):
    """Return normalized statement and bounded proof body for one declaration."""
    proof_start, body = split_outer_proof(content)
    statement_text = content[:proof_start] if proof_start is not None else content
    statement = " ".join(statement_text.split())
    if is_canceled(statement):
        return "", None

    # Inline justifications prove a theorem without opening a proof block. They
    # belong to source provenance, not to the statement shown in a fact block.
    statement = re.sub(
        r"\s+(?:by|from)\s+[^;]*;\s*$", "", statement, flags=re.DOTALL
    ).strip()
    return statement, body


def iter_theorem_proofs(text):
    """Yield ``(global_name, statement, proof_body)`` without crossing headers."""
    for kind, metadata, content in _declaration_sections(text):
        if kind != "theorem":
            continue
        _, global_name = metadata
        statement, body = _statement_and_proof(content)
        if statement and body:
            yield global_name, statement, body


def parse_article(path):
    """(global name -> statement, local label -> global name) for one article."""
    with open(path, encoding="utf-8", errors="replace") as article_file:
        text = article_file.read()
    article = os.path.basename(path).split(".", 1)[0].upper()
    stmt, local = {}, {}
    for kind, metadata, content in _declaration_sections(text):
        if kind == "theorem":
            label, global_name = metadata
            statement, _ = _statement_and_proof(content)
        elif kind == "definition_theorem":
            label, global_name = metadata
            prefix = content.split(";", 1)[0]
            statement = " ".join(prefix.split())
        elif kind == "scheme":
            (global_name,) = metadata
            label = None
            statement, _ = _statement_and_proof(content)
        elif kind == "local_lemma":
            label, first_line = metadata
            global_name = f"{article}:{label}"
            statement, _ = _statement_and_proof(f"{first_line}\n{content}")
        else:
            continue

        if not statement or is_canceled(statement):
            continue
        stmt[global_name] = statement
        if label:
            local[label] = global_name
    return stmt, local


def _qualified_references(segment, local_by_article=None):
    found = []
    spans = []
    for match in GREF.finditer(segment):
        article, kind, special_number, theorem_number, tail = match.groups()
        inherited_kind = kind
        number = special_number or theorem_number
        prefix = f"{kind}_" if kind else ""
        found.append((match.start(), f"{article}:{prefix}{number}"))
        for index, item in enumerate(re.findall(r"(?:def|sch)?(?:_|\s*)?\d+", tail)):
            compact = re.sub(r"\s+", "", item)
            tail_match = re.fullmatch(r"(?:(def|sch)_?)?(\d+)", compact)
            tail_kind, tail_number = tail_match.groups()
            effective_kind = tail_kind or inherited_kind
            tail_prefix = f"{effective_kind}_" if effective_kind else ""
            found.append(
                (match.start() + index + 1, f"{article}:{tail_prefix}{tail_number}")
            )
        spans.append(match.span())
    for match in QUALIFIED_LABEL.finditer(segment):
        if any(start <= match.start() < end for start, end in spans):
            continue
        article, label = match.groups()
        # ``TARSKI:def 4`` and ``ORDINAL1:sch 1`` belong to GREF even
        # though the space means its match does not cover the label token.
        suffix = segment[match.end():]
        if label.lower() in {"def", "sch"} and re.match(r"\s+\d+", suffix):
            continue
        article_local = (
            local_by_article.get(article) if local_by_article is not None else None
        )
        name = article_local.get(label) if article_local is not None else None
        found.append((match.start(), name or f"{article}:{label}"))
        spans.append(match.span())
    return found, spans


def scan_references(body, local, local_by_article=None):
    """Return canonical references and unresolved article-local labels."""
    found = []
    unresolved = []
    order = 0
    for justification in BY.finditer(body):
        segment = justification.group(1)
        qualified, qualified_spans = _qualified_references(
            segment, local_by_article=local_by_article
        )
        for position, name in qualified:
            found.append((justification.start() + position, order, name))
            order += 1
        for match in IDENTIFIER.finditer(segment):
            if any(start <= match.start() < end for start, end in qualified_spans):
                continue
            label = match.group(0)
            name = local.get(label)
            if name is not None:
                found.append(
                    (justification.start() + match.start(), order, name)
                )
            elif CONVENTIONAL_LOCAL.fullmatch(label):
                unresolved.append(
                    (justification.start() + match.start(), order, label)
                )
            else:
                continue
            order += 1

    refs = []
    for _, _, name in sorted(found):
        if name not in refs:
            refs.append(name)
    missing_local = []
    for _, _, label in sorted(unresolved):
        if label not in missing_local:
            missing_local.append(label)
    return refs, missing_local


def cited_names(body, local, local_by_article=None):
    """Global names cited in a proof body, with article-local labels resolved."""
    return scan_references(body, local, local_by_article=local_by_article)[0]


def resolve_references(
    body,
    local,
    statements,
    own_name=None,
    local_by_article=None,
):
    """Resolve every external proof reference or report why the row is incomplete."""
    refs, missing = scan_references(
        body, local, local_by_article=local_by_article
    )
    for name in refs:
        statement = statements.get(name)
        if (
            statement is None or is_canceled(statement) or name == own_name
        ) and name not in missing:
            missing.append(name)
    return refs, missing


def statements_match(left, right):
    """Compare Mizar statements modulo labels and insignificant whitespace."""
    def normalize(statement):
        statement = re.sub(
            r"^(?:Th|Lm)\d+\s*:\s*", "", statement.strip()
        )
        statement = statement.rstrip(";")
        return re.sub(r"\s+", "", statement)

    return normalize(left) == normalize(right)


def render(facts, goal, target):
    block = HDR + "\n" + "\n".join(f"{n} : {s}" for n, s in facts.items())
    text = f"{block}\n{SEP}\nGOAL {goal}\n{target}"
    return text, 0, len(block)


def shuffled(refs, key):
    """Order the block independently of citation order.

    Listing premises in the order the derivation uses them hands the model the
    step sequence for free — it can read the proof structure off the prompt
    instead of deriving it. The permutation is seeded per example so the corpus
    is reproducible.
    """
    r = list(refs)
    random.Random(key).shuffle(r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html2", default=MIZAR)
    ap.add_argument("--out", default="/tmp/dscount/shards")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.html2, "*.txt")))
    if not files:
        raise SystemExit(f"no Mizar html2 articles found in {a.html2}")
    stmt, local_of, local_by_article = {}, {}, {}
    for p in files:
        article_statements, article_local = parse_article(p)
        stmt.update(article_statements)
        local_of[p] = article_local
        article = os.path.basename(p).split(".", 1)[0].upper()
        local_by_article[article] = article_local

    # pass 1: citation counts, so the held-out set can come from the low tail
    counts = {}
    proofs = []
    rejected_incomplete = rejected_identity = no_citation = 0
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as article_file:
            t = article_file.read()
        loc = local_of[p]
        for thm, goal, body in iter_theorem_proofs(t):
            source_statement = stmt.get(thm)
            if source_statement is None or not statements_match(goal, source_statement):
                rejected_identity += 1
                continue
            refs, missing = resolve_references(
                body,
                loc,
                stmt,
                own_name=thm,
                local_by_article=local_by_article,
            )
            if missing:
                rejected_incomplete += 1
                continue
            if not refs:
                no_citation += 1
                continue
            proofs.append((thm, goal, body, refs))
            for r in refs:
                counts[r] = counts.get(r, 0) + 1

    tail = sorted(n for n, c in counts.items() if c in (1, 2) and n in stmt)
    rng = random.Random(a.seed)
    held = set(rng.sample(tail, min(a.heldout, len(tail))))
    with open(os.path.join(a.out, "heldout.json"), "w") as heldout_file:
        json.dump(
            {
                "facts": sorted(held),
                "seed": a.seed,
                "policy": "cited 1-2 times; own proof and all citing proofs removed",
            },
            heldout_file,
            indent=1,
        )

    kept = dropped_heldout = dropped_dup = 0
    dropped_incomplete = rejected_incomplete
    tb = 0
    seen_text = set()
    with open(os.path.join(a.out, "mizar.jsonl"), "w") as fh, \
            open(os.path.join(a.out, "mizar_eval.jsonl"), "w") as ev:
        for thm, goal, body, refs in proofs:
            is_eval = thm in held or bool(set(refs) & held)
            if not all(r in stmt for r in refs):
                if not is_eval:
                    dropped_incomplete += 1
                continue
            eid = hashlib.md5(f"{thm}|{goal}".encode()).hexdigest()[:12]
            facts = {r: stmt[r] for r in shuffled(refs, eid)}
            steps = [line.strip() for line in body.split("\n") if line.strip()]
            target = "\n".join(steps)
            text, ms, me = render(facts, goal, target)
            if text in seen_text:
                dropped_dup += 1
                continue
            seen_text.add(text)
            rec = {"id": eid,
                   "theorem": thm, "facts": facts, "cited": refs,
                   "goal": goal, "target": target, "text": text,
                   "mask_start": ms, "mask_end": me}
            if is_eval:
                ev.write(json.dumps(rec) + "\n")
                dropped_heldout += 1
            else:
                fh.write(json.dumps(rec) + "\n")
                kept += 1
                tb += len(text.encode())

    print(f"facts with statements : {len(stmt):,}")
    print(f"held-out facts        : {len(held):,} (from {len(tail):,} cited 1-2x)")
    print(f"train examples        : {kept:,}")
    print(f"eval examples         : {dropped_heldout:,}")
    print(f"dropped, incomplete   : {dropped_incomplete:,}")
    print(f"dropped, source id    : {rejected_identity:,}")
    print(f"dropped, no citation  : {no_citation:,}")
    print(f"dropped, duplicate    : {dropped_dup:,}")
    print(f"train bytes           : {tb/1e6:.1f} MB  ~{tb/2.2/1e6:.0f}M GPT-2 tok")
    print(f"wrote {a.out}/mizar.jsonl, mizar_eval.jsonl, heldout.json")


if __name__ == "__main__":
    main()
