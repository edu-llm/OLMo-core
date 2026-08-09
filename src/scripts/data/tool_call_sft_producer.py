"""
Turn tool-call SFT ``.jsonl`` conversations into the two flat arrays OLMo-core trains from,
building the prompt loss mask **by construction** instead of by re-rendering prefixes.

Why this exists
---------------
``open-instruct``'s ``convert_sft_data_for_olmocore.py`` derives each assistant turn's trainable
span by rendering the conversation twice and requiring the full render to start with the shorter
one. OLMo 3's chat template closes a *non-final* assistant turn with ``<|im_end|>\\n`` but the
*final* one with ``eos_token``, so in the sub-render the turn being measured is ``loop.last`` and
emits a different string. The prefix check then fails on **every conversation with two or more
assistant turns** — see ``docs/tool-call/verify/verify_multiturn_mask.py``. Upstream's own escape
hatch is ``last_turn_only=True``, which works but makes only the final assistant turn trainable
(1 of 10 on a 21-message row).

This module never re-renders a prefix. It emits the conversation as an ordered list of segments
whose character spans are known exactly, tokenizes the concatenation **once**, and marks a token
trainable when its character span overlaps a trainable segment. Every assistant turn stays
trainable, and the multi-turn failure mode cannot arise.

The safety property that makes it trustworthy: the concatenated segments must equal what the real
chat template renders, byte for byte. ``--self-check`` asserts exactly that against the shipped
``chat_template.jinja``, so a template change is a loud failure rather than a silently wrong mask.

Usage
-----
.. code-block:: bash

    # prove the invariants (needs jinja2 + network for the template)
    python src/scripts/data/tool_call_sft_producer.py --self-check

    # build the arrays
    python src/scripts/data/tool_call_sft_producer.py \\
        --input data/tool-call/conversations \\
        --output data/tool-call/tokenized \\
        --tokenizer allenai/olmo-3-tokenizer-instruct-dev

Outputs ``token_ids_part_%04d.npy`` (uint32) and ``labels_mask_part_%04d.npy`` (bool), both
**headerless** despite the extension, which is what ``NumpyPackedFSLDatasetConfig`` memory-maps.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Roles we emit. The profile enforces no role set, and OLMo 3's Think template silently drops
#: ``tool``, so we normalise tool results onto ``environment`` and reject anything else.
VALID_ROLES = frozenset({"system", "user", "assistant", "environment"})

#: Tool results arrive on this role. ``tool`` is accepted on input and rewritten.
RESULT_ROLE = "environment"

TURN_OPEN = "<|im_start|>"
TURN_CLOSE = "<|im_end|>"
DEFAULT_EOS = "<|endoftext|>"
DEFAULT_TOKENIZER = "allenai/olmo-3-tokenizer-instruct-dev"

SHARD_ROWS = 20_000


@dataclass(frozen=True)
class Segment:
    """One contiguous slice of the rendered conversation.

    :param text: The literal characters this segment contributes.
    :param trainable: Whether loss is computed over the tokens covering it.
    """

    text: str
    trainable: bool


def normalise_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Validate and canonicalise a conversation.

    Rewrites ``tool`` to ``environment`` (the Instruct template aliases them, the Think template
    drops ``tool`` outright) and rejects anything the chat template would silently skip — an
    unrecognised role renders as nothing, which downstream shows up as an all-masked row that gets
    dropped rather than as an error.

    :param messages: The raw ``messages`` array from one ``.jsonl`` row.

    :returns: Messages with a valid role and a non-empty string ``content``.

    :raises ValueError: If a role is unrecognised, ``content`` is missing, empty or not a string,
        or the conversation does not end on an ``assistant`` turn.
    """
    if not messages:
        raise ValueError("messages is empty")

    out: list[dict[str, str]] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            role = RESULT_ROLE
        if role not in VALID_ROLES:
            raise ValueError(
                f"messages[{i}] has role {role!r}; expected one of {sorted(VALID_ROLES)}. "
                f"The chat template emits nothing for an unknown role, so this would become an "
                f"all-masked row and be dropped silently."
            )
        content = m.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"messages[{i}] (role={role!r}) needs a non-empty string 'content'; got "
                f"{type(content).__name__}. The profile accepts null, which would teach the model "
                f"to emit nothing."
            )
        out.append({"role": role, "content": content})

    if out[-1]["role"] != "assistant":
        raise ValueError(
            f"conversation ends on {out[-1]['role']!r}; it must end on 'assistant'. The template "
            f"only emits EOS on a final assistant turn, and OLMo-core finds document boundaries "
            f"by EOS — a row ending elsewhere corrupts packing silently."
        )
    return out


def build_segments(messages: Sequence[dict[str, str]], *, eos: str = DEFAULT_EOS) -> list[Segment]:
    """
    Render a conversation as segments, marking the trainable ones.

    Mirrors OLMo 3's ``chat_template.jinja`` exactly: every turn is
    ``<|im_start|>{role}\\n{content}`` followed by ``<|im_end|>\\n``, except the final assistant
    turn which closes with ``eos_token``. For assistant turns the header is split from the body so
    the header can be masked and the body — content plus its closing token — trained.

    The ``message['functions']`` and ``message['function_calls']`` template branches are
    deliberately unreachable here: we inline both into ``content`` so the leakage check can see
    them, and emit no sibling fields at all.

    :param messages: Normalised messages from :func:`normalise_messages`.
    :param eos: The end-of-sequence string the final assistant turn closes with.

    :returns: Segments whose concatenation is the full rendered conversation.
    """
    segs: list[Segment] = []
    last = len(messages) - 1
    for i, m in enumerate(messages):
        role, content = m["role"], m["content"]
        close = eos if (role == "assistant" and i == last) else f"{TURN_CLOSE}\n"
        if role == "assistant":
            # Header masked, content + close trainable.
            segs.append(Segment(f"{TURN_OPEN}{role}\n", trainable=False))
            segs.append(Segment(f"{content}{close}", trainable=True))
        else:
            segs.append(Segment(f"{TURN_OPEN}{role}\n{content}{close}", trainable=False))
    return segs


def trainable_char_spans(segments: Sequence[Segment]) -> tuple[str, list[tuple[int, int]]]:
    """
    Flatten segments into the rendered string plus the character spans loss applies to.

    These spans are the whole point: they are known by *construction* from segment lengths, so
    nothing has to re-render a prefix and no ``loop.last`` divergence can arise.

    :param segments: Output of :func:`build_segments`.

    :returns: ``(rendered, spans)`` where each span is a half-open ``(start, end)``.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for seg in segments:
        end = pos + len(seg.text)
        if seg.trainable:
            spans.append((pos, end))
        parts.append(seg.text)
        pos = end
    return "".join(parts), spans


def mask_from_offsets(
    offsets: Sequence[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> np.ndarray:
    """
    Mark each token trainable when its character span overlaps a trainable span.

    Overlap rather than containment, so a token straddling the header/content boundary stays
    trainable rather than being dropped.

    :param offsets: Per-token ``(start, end)`` character offsets from the tokenizer.
    :param spans: Trainable character spans from :func:`trainable_char_spans`.

    :returns: A boolean array, one entry per token.
    """
    mask = np.zeros(len(offsets), dtype=np.bool_)
    for t, (t0, t1) in enumerate(offsets):
        if t0 == t1:  # zero-width (some special tokens report this)
            continue
        for s0, s1 in spans:
            if t0 < s1 and s0 < t1:
                mask[t] = True
                break
    return mask


def encode_row(
    messages: Sequence[dict[str, Any]], tokenizer: Any, *, eos: str = DEFAULT_EOS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode one conversation into token ids and a matching loss mask.

    :param messages: The raw ``messages`` array from one ``.jsonl`` row.
    :param tokenizer: A ``tokenizers.Tokenizer``.
    :param eos: End-of-sequence string for the final assistant turn.

    :returns: ``(token_ids uint32, label_mask bool)`` of equal length.

    :raises ValueError: If the conversation is malformed, or the rendered text does not contain
        exactly one EOS.
    """
    msgs = normalise_messages(messages)
    rendered, spans = trainable_char_spans(build_segments(msgs, eos=eos))

    n_eos = rendered.count(eos)
    if n_eos != 1:
        raise ValueError(
            f"rendered conversation contains {n_eos} occurrences of {eos!r}; packing needs exactly "
            f"one per document to find the boundary"
        )

    enc = tokenizer.encode(rendered, add_special_tokens=False)
    mask = mask_from_offsets(enc.offsets, spans)
    if not mask.any():
        raise ValueError("no trainable tokens; the row would be dropped downstream")
    return np.asarray(enc.ids, dtype=np.uint32), mask


def iter_rows(input_dir: Path) -> Iterator[dict[str, Any]]:
    """
    Yield rows from every ``.jsonl`` under ``input_dir``, recursively and in sorted path order.

    :param input_dir: Directory holding the conversation shards.

    :returns: An iterator of decoded rows.
    """
    for path in sorted(input_dir.rglob("*.jsonl")):
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e


def _write_headerless(path: Path, arrays: Iterable[np.ndarray], dtype: Any) -> int:
    """Concatenate ``arrays`` and write raw bytes — no ``.npy`` header. Returns the item count."""
    joined = np.concatenate(list(arrays)) if arrays else np.empty(0, dtype=dtype)
    joined = joined.astype(dtype, copy=False)
    with path.open("wb") as fh:
        fh.write(joined.tobytes(order="C"))
    return int(joined.size)


def build(
    input_dir: Path, output_dir: Path, tokenizer: Any, *, eos: str = DEFAULT_EOS
) -> dict[str, int]:
    """
    Convert every conversation under ``input_dir`` into sharded arrays under ``output_dir``.

    :param input_dir: Directory of ``.jsonl`` conversation shards.
    :param output_dir: Destination for the arrays.
    :param tokenizer: A ``tokenizers.Tokenizer``.
    :param eos: End-of-sequence string.

    :returns: Counts for logging: rows, skipped, tokens, trainable tokens, shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_buf: list[np.ndarray] = []
    mask_buf: list[np.ndarray] = []
    stats = {"rows": 0, "skipped": 0, "tokens": 0, "trainable": 0, "shards": 0}

    def flush() -> None:
        if not ids_buf:
            return
        part = stats["shards"]
        n = _write_headerless(output_dir / f"token_ids_part_{part:04d}.npy", ids_buf, np.uint32)
        m = _write_headerless(output_dir / f"labels_mask_part_{part:04d}.npy", mask_buf, np.bool_)
        if n != m:
            raise RuntimeError(f"shard {part}: {n} token ids but {m} mask entries")
        stats["shards"] += 1
        ids_buf.clear()
        mask_buf.clear()

    rows_in_shard = 0
    for row in iter_rows(input_dir):
        try:
            ids, mask = encode_row(row.get("messages", []), tokenizer, eos=eos)
        except ValueError as e:
            stats["skipped"] += 1
            log.warning("skipping row: %s", e)
            continue
        ids_buf.append(ids)
        mask_buf.append(mask)
        stats["rows"] += 1
        stats["tokens"] += int(ids.size)
        stats["trainable"] += int(mask.sum())
        rows_in_shard += 1
        if rows_in_shard >= SHARD_ROWS:
            flush()
            rows_in_shard = 0
    flush()
    return stats


TEMPLATE_URL = "https://huggingface.co/allenai/Olmo-3-7B-Instruct/raw/main/chat_template.jinja"


def render_with_real_template(messages: Sequence[dict[str, str]], *, eos: str) -> str:
    """
    Render a conversation with OLMo 3's shipped ``chat_template.jinja``, for comparison.

    Only used by ``--self-check`` and the tests. Needs ``jinja2`` and network access.

    :param messages: Normalised messages.
    :param eos: End-of-sequence string.

    :returns: The template's rendering.
    """
    import urllib.request  # noqa: PLC0415

    import jinja2  # noqa: PLC0415
    import jinja2.ext  # noqa: PLC0415
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    with urllib.request.urlopen(TEMPLATE_URL, timeout=60) as r:
        source = r.read().decode("utf-8")

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    # transformers' tojson, NOT jinja's builtin: the builtin is HTML-safe and would escape
    # < > & ' inside a JSON Schema.
    env.filters["tojson"] = jinja2.pass_eval_context(
        lambda ctx, v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)
    )
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(
        jinja2.exceptions.TemplateError(m)
    )
    return env.from_string(source).render(
        messages=list(messages), tools=None, eos_token=eos, add_generation_prompt=False
    )


def _self_check(eos: str) -> None:
    """Assert the segment construction matches the real template, and report the mask."""
    sys_content = (
        'You are a helpful function-calling AI assistant. <functions>[{"a":1}]</functions>'
    )
    call = '<function_calls>get_weather(city="Boston")</function_calls>'
    cases = {
        "single-turn": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "weather in Boston?"},
            {"role": "assistant", "content": call},
        ],
        "multi-turn (2 assistants)": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "weather in Boston?"},
            {"role": "assistant", "content": call},
            {"role": "environment", "content": '{"temp_f":54}'},
            {"role": "assistant", "content": "It's 54F in Boston."},
        ],
        "abstention (prose only)": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": "who wrote Hamlet?"},
            {"role": "assistant", "content": "Shakespeare. No tool is needed for that."},
        ],
        "5 assistant turns": [
            {"role": "system", "content": sys_content},
            *[
                m
                for _ in range(4)
                for m in (
                    {"role": "user", "content": "next?"},
                    {"role": "assistant", "content": call},
                )
            ],
            {"role": "user", "content": "and finally?"},
            {"role": "assistant", "content": "Done."},
        ],
    }

    print("=== segment construction vs the real chat_template.jinja ===")
    all_ok = True
    for name, msgs in cases.items():
        norm = normalise_messages(msgs)
        rendered, spans = trainable_char_spans(build_segments(norm, eos=eos))
        theirs = render_with_real_template(norm, eos=eos)
        ok = rendered == theirs
        all_ok &= ok
        n_asst = sum(1 for m in norm if m["role"] == "assistant")
        print(f"  {name:26s} assistants={n_asst}  spans={len(spans)}  IDENTICAL={ok}")
        if not ok:
            k = next(
                (i for i in range(min(len(rendered), len(theirs))) if rendered[i] != theirs[i]),
                min(len(rendered), len(theirs)),
            )
            print(f"      diverge at {k}: ours={rendered[k:k+20]!r} theirs={theirs[k:k+20]!r}")
        if len(spans) != n_asst:
            print(f"      WARNING: {len(spans)} trainable spans for {n_asst} assistant turns")
            all_ok = False

    print()
    print("=== mask, with a real tokenizer ===")
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415

        tok = Tokenizer.from_pretrained(DEFAULT_TOKENIZER)
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  skipped, tokenizer unavailable: {e}")
    else:
        for name, msgs in cases.items():
            ids, mask = encode_row(msgs, tok, eos=eos)
            pct = 100.0 * mask.sum() / mask.size
            print(f"  {name:26s} tokens={ids.size:5d} trainable={int(mask.sum()):5d} ({pct:4.1f}%)")

    print()
    print("ALL INVARIANTS HOLD" if all_ok else "SELF-CHECK FAILED")
    if not all_ok:
        raise SystemExit(1)


def main() -> None:
    """Parse arguments and either run the self-check or build the arrays."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, help="directory of .jsonl conversation shards")
    p.add_argument("--output", type=Path, help="destination for the arrays")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF tokenizer repo or local dir")
    p.add_argument("--eos", default=DEFAULT_EOS)
    p.add_argument(
        "--self-check",
        action="store_true",
        help="assert the segment/template invariants and exit (see the module docstring)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.self_check:
        _self_check(args.eos)
        return

    if not args.input or not args.output:
        p.error("--input and --output are required unless --self-check is given")

    from tokenizers import Tokenizer  # noqa: PLC0415

    tok = Tokenizer.from_pretrained(args.tokenizer)
    stats = build(args.input, args.output, tok, eos=args.eos)
    pct = 100.0 * stats["trainable"] / stats["tokens"] if stats["tokens"] else 0.0
    log.info(
        "%d rows -> %d shards, %d tokens, %d trainable (%.1f%%), %d skipped",
        stats["rows"],
        stats["shards"],
        stats["tokens"],
        stats["trainable"],
        pct,
        stats["skipped"],
    )


if __name__ == "__main__":
    main()
