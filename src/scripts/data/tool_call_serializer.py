"""
Turn structured content into tool-call SFT rows, and read them back.

Everything that generates data goes through here, so no model ever writes the wire format itself.
A generator returns plain structured data — the tool schemas it was given, a user turn, and either a
call or some prose — and this module renders it. That way format correctness is a property of the
code rather than something we hope a model got right, and swapping generators costs nothing.

The format is OLMo 3's, verified against the shipped ``chat_template.jinja``:

* tool schemas ride inside the ``system`` message as `` <functions>[...]</functions>`` — one leading
  space, single-line JSON array;
* a call rides inside the ``assistant`` message as ``<function_calls>name(k="v")</function_calls>``,
  Pythonic, with each argument *value* JSON-encoded;
* parallel calls share **one** block, joined by a bare newline;
* a tool result rides on role ``environment`` with no wrapper;
* abstention is ordinary prose containing no ``<function_calls>`` at all.

Everything is inlined into ``content`` on purpose. ``sft-conversations/v1`` recomputes train/heldout
leakage by hashing ``role`` and ``content`` only, so a call parked in a sibling field would be
invisible to the one integrity check the profile performs on our payload.

One trap this module exists to avoid: argument values are **JSON**, so a boolean serialises as
``true``, not Python's ``True``. ``ast.parse`` reads ``true`` as a *variable name*, and
``ast.literal_eval`` raises on it. :func:`parse_call` is therefore JSON-aware — a plain AST-literal
check would silently reject every row carrying a boolean or null argument.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Take the delimiters from the runtime rather than restating them. `olmo_core.tools.protocol` is
# what actually parses the model's output at inference, so if these ever diverge the dataset trains
# a format the runtime cannot read. Importing is the cheapest way to make that impossible.
from olmo_core.tools.protocol import FUNCTION_CALLS_END as CALL_CLOSE  # noqa: E402
from olmo_core.tools.protocol import FUNCTION_CALLS_START as CALL_OPEN  # noqa: E402
from olmo_core.tools.protocol import FUNCTIONS_END as FUNCS_CLOSE  # noqa: E402
from olmo_core.tools.protocol import FUNCTIONS_START as FUNCS_OPEN  # noqa: E402
from olmo_core.tools.protocol import parse_function_calls  # noqa: E402

#: The template emits ``' <functions>'`` — with a leading space — on the per-message path, which is
#: the path AI2's own training bytes took. Proven byte-identical in
#: ``docs/tool-call/verify/verify_render_identity.py``.
FUNCS_PREFIX = " "

DEFAULT_PREAMBLE = (
    "You are a helpful function-calling AI assistant. You are provided with function signatures "
    "within <functions></functions> XML tags. You may call one or more functions to assist with "
    "the user query. Output any function calls within <function_calls></function_calls> XML tags. "
    "Do not make assumptions about what values to plug into functions."
)

#: JSON spellings of the three values whose Python and JSON forms differ.
_JSON_NAMES = {"true": True, "false": False, "null": None}

#: Flat identifiers only. The runtime parser requires ``ast.Name`` for the callee
#: (`olmo_core/tools/protocol.py`, ``_parse_call``), so a dotted name like
#: ``weather.forecast_weather_api`` raises ``ToolCallParseError`` at inference. Upstream corpora are
#: full of dotted names, so reformatting must flatten them with :func:`flatten_tool_name` rather
#: than pass them through — otherwise we train the model to emit calls our own runtime rejects.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ToolSchema:
    """One tool offered to the model.

    :param name: The function name, as the model must emit it.
    :param description: One line telling the model what the tool does.
    :param parameters: A JSON Schema object describing the arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        """:returns: The OpenAI/Hermes-style wrapper the schema block carries."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_wire(cls, obj: dict[str, Any]) -> "ToolSchema":
        """Rebuild a schema from its wrapper.

        :param obj: One element of a decoded ``<functions>`` array.

        :raises ValueError: If the wrapper is not shaped as expected.
        """
        try:
            fn = obj["function"]
            return cls(
                name=fn["name"], description=fn.get("description", ""), parameters=fn["parameters"]
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"not a function wrapper: {obj!r}") from e


@dataclass(frozen=True)
class Call:
    """One tool call.

    :param name: The function being called.
    :param arguments: Keyword arguments. Values are JSON-encoded on the wire.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def flatten_tool_name(name: str) -> str:
    """
    Turn a dotted upstream tool name into the flat form the runtime can parse.

    Upstream corpora are full of names like ``weather.forecast_weather_api`` and
    ``combinatorics.permutation_count``. The runtime's parser requires a plain ``ast.Name``, so a
    dotted call raises at inference. Reformatting must rename rather than pass through, and the
    rename has to be applied to the *schema* as well as the call so the two still agree.

    :param name: The upstream name, possibly dotted.

    :returns: The same name with dots replaced by underscores.
    """
    return name.replace(".", "_")


def serialize_call(call: Call) -> str:
    """
    Render one call in OLMo 3's Pythonic form.

    Argument *values* are JSON-encoded individually and joined with ``", "``, matching the
    template's ``key ~ '=' ~ (value | tojson)``.

    :param call: The call to render.

    :returns: For example ``get_weather(city="Boston", days=5)``.

    :raises ValueError: If the function name is not a plain identifier (optionally dotted), or an
        argument name is not an identifier, or a value is not JSON-serialisable.
    """
    if not _NAME_RE.match(call.name):
        hint = ""
        if "." in call.name:
            hint = (
                f" The runtime parser requires a flat name; use "
                f"flatten_tool_name({call.name!r}) -> {flatten_tool_name(call.name)!r}."
            )
        raise ValueError(f"function name {call.name!r} is not a flat identifier.{hint}")
    parts = []
    for key, value in call.arguments.items():
        if not key.isidentifier():
            raise ValueError(f"argument name {key!r} is not an identifier")
        try:
            rendered = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"argument {key!r} is not JSON-serialisable: {e}") from e
        parts.append(f"{key}={rendered}")
    return f"{call.name}({', '.join(parts)})"


def _literal(node: ast.AST) -> Any:
    """
    Evaluate one argument value, accepting JSON spellings Python does not have.

    ``ast.literal_eval`` raises on ``true``/``false``/``null`` because they are ``Name`` nodes
    rather than constants. Since the wire format is JSON-encoded, those spellings are exactly what
    we emit, so they must be understood here.

    :param node: The AST node for one argument value.

    :returns: The Python value.

    :raises ValueError: If the node is not a literal we accept.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        # bool subclasses int, so guard it explicitly or `-true` silently becomes -1.
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return -inner
    raise ValueError(f"argument value is not a JSON literal: {ast.dump(node)[:80]}")


def parse_call(text: str) -> Call:
    """
    Read one serialised call back.

    :param text: For example ``get_weather(city="Boston")``.

    :returns: The parsed :class:`Call`.

    :raises ValueError: If the text is not a single keyword-only call with literal values.
    """
    try:
        expr = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not parseable as a call: {text!r} ({e})") from e
    if not isinstance(expr, ast.Call):
        raise ValueError(f"not a call: {text!r}")
    if expr.args:
        raise ValueError(f"positional arguments are not allowed: {text!r}")
    if any(k.arg is None for k in expr.keywords):
        raise ValueError(f"**kwargs unpacking is not allowed: {text!r}")

    func = expr.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        bits: list[str] = []
        cur: ast.AST = func
        while isinstance(cur, ast.Attribute):
            bits.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name):
            raise ValueError(f"unsupported callee in {text!r}")
        bits.append(cur.id)
        name = ".".join(reversed(bits))
    else:
        raise ValueError(f"unsupported callee in {text!r}")

    args = {k.arg: _literal(k.value) for k in expr.keywords if k.arg is not None}
    return Call(name=name, arguments=args)


def serialize_schemas(schemas: Sequence[ToolSchema]) -> str:
    """
    Render the schema block that gets appended to the system message.

    :param schemas: The tools offered on this row.

    :returns: `` <functions>[...]</functions>`` including the leading space.
    """
    arr = [s.to_wire() for s in schemas]
    body = json.dumps(arr, ensure_ascii=False, separators=(",", ":"))
    if "\n" in body:  # pragma: no cover - separators forbid it
        raise ValueError("the schema block must be a single line")
    return f"{FUNCS_PREFIX}{FUNCS_OPEN}{body}{FUNCS_CLOSE}"


def serialize_calls(calls: Sequence[Call]) -> str:
    """
    Render one or more calls as a single block.

    Parallel calls share **one** block joined by a bare newline — not one block each.

    :param calls: The calls to render, in order.

    :returns: ``<function_calls>...</function_calls>``.

    :raises ValueError: If ``calls`` is empty.
    """
    if not calls:
        raise ValueError("no calls to serialize; for abstention pass prose instead")
    return f"{CALL_OPEN}{chr(10).join(serialize_call(c) for c in calls)}{CALL_CLOSE}"


def build_row(
    *,
    schemas: Sequence[ToolSchema],
    user: str,
    calls: Sequence[Call] | None = None,
    prose: str | None = None,
    preamble: str = DEFAULT_PREAMBLE,
    prior_turns: Sequence[dict[str, str]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build one training row.

    Exactly one of ``calls`` or ``prose`` must be given: a row either makes calls or it abstains.
    Mixing prose and a call in the same turn is rejected, because an unambiguous target is what
    makes the verifier exact and what a small model can reliably reproduce.

    :param schemas: Tools offered on this row.
    :param user: The user's turn.
    :param calls: Calls the assistant should emit, if any.
    :param prose: The assistant's prose, for an abstention row.
    :param preamble: System prose the schema block is appended to.
    :param prior_turns: Earlier turns to insert before ``user``, for multi-turn rows. Each needs a
        ``role`` in ``{user, assistant, environment}`` and a ``content``.
    :param extra: Top-level row fields to merge in, such as ``domain`` or ``expected_result``.

    :returns: A row ready to serialise as one ``.jsonl`` line.

    :raises ValueError: If neither or both of ``calls``/``prose`` are given, or a prior turn is
        malformed, or ``prose`` contains a call block.
    """
    if (calls is None) == (prose is None):
        raise ValueError("pass exactly one of calls= or prose=")
    if prose is not None and CALL_OPEN in prose:
        raise ValueError(f"abstention prose must not contain {CALL_OPEN}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"{preamble}{serialize_schemas(schemas)}"}
    ]
    for turn in prior_turns or []:
        role = turn.get("role")
        if role not in {"user", "assistant", "environment"}:
            raise ValueError(f"prior turn role {role!r} must be user, assistant or environment")
        content = turn.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"prior turn ({role}) needs non-empty string content")
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user})
    body = serialize_calls(calls) if calls is not None else prose
    assert body is not None
    messages.append({"role": "assistant", "content": body})

    row: dict[str, Any] = {"messages": messages}
    if extra:
        overlap = set(extra) & {"messages"}
        if overlap:
            raise ValueError(f"extra may not override {sorted(overlap)}")
        row.update(extra)
    return row


def extract_schema_array(system_content: str) -> list[dict[str, Any]]:
    """
    Pull the decoded schema array out of a system message.

    Deliberately not a naive ``find``. The default preamble *itself* contains the literal text
    ``<functions></functions>`` — it tells the model that is where signatures live — so the first
    occurrence in the message is decoration, not data. A tool description could contain the literal
    too. So we anchor on the end of the message and walk candidate opening tags right-to-left,
    taking the first whose body decodes as a JSON array.

    :param system_content: The system message's ``content``.

    :returns: The decoded array of function wrappers.

    :raises ValueError: If no ``<functions>`` block holding a JSON array is present at the end.
    """
    if not system_content.endswith(FUNCS_CLOSE):
        raise ValueError(f"system turn must end with {FUNCS_CLOSE}")
    head = system_content[: -len(FUNCS_CLOSE)]
    cursor = len(head)
    while True:
        start = head.rfind(FUNCS_OPEN, 0, cursor)
        if start == -1:
            raise ValueError("system turn carries no <functions> block holding a JSON array")
        body = head[start + len(FUNCS_OPEN) :]
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            cursor = start
            continue
        if not isinstance(decoded, list):
            cursor = start
            continue
        return decoded


@dataclass(frozen=True)
class ParsedRow:
    """What :func:`parse_row` recovers from a serialised row.

    :param schemas: Tools that were offered.
    :param user: The final user turn.
    :param calls: Calls the assistant made; empty for an abstention row.
    :param prose: The assistant's prose; ``None`` when it made calls.
    """

    schemas: list[ToolSchema]
    user: str
    calls: list[Call]
    prose: str | None


def parse_row(row: dict[str, Any]) -> ParsedRow:
    """
    Read a serialised row back into structured content.

    Used as a round-trip assertion in the build pipeline: if a row we wrote cannot be read back,
    the writer is wrong. Cheaper and stricter than eyeballing samples.

    :param row: A decoded ``.jsonl`` row.

    :returns: The recovered content.

    :raises ValueError: If the row is not shaped as :func:`build_row` produces.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row has no messages")
    if messages[0].get("role") != "system":
        raise ValueError("first message must be the system turn")
    if messages[-1].get("role") != "assistant":
        raise ValueError("row must end on an assistant turn")

    schemas = [
        ToolSchema.from_wire(o) for o in extract_schema_array(messages[0].get("content") or "")
    ]

    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        raise ValueError("row has no user turn")

    content = messages[-1].get("content") or ""
    n_open, n_close = content.count(CALL_OPEN), content.count(CALL_CLOSE)
    if n_open != n_close:
        raise ValueError(f"unbalanced call block: {n_open} open, {n_close} close")
    if n_open == 0:
        return ParsedRow(schemas=schemas, user=users[-1]["content"], calls=[], prose=content)
    if n_open > 1:
        raise ValueError("parallel calls share ONE block; found more than one")
    inner = content[content.index(CALL_OPEN) + len(CALL_OPEN) : content.rindex(CALL_CLOSE)]
    trailing = content[content.rindex(CALL_CLOSE) + len(CALL_CLOSE) :]
    if trailing.strip():
        raise ValueError(f"nothing may follow {CALL_CLOSE}; found {trailing[:40]!r}")
    calls = [parse_call(line) for line in inner.split("\n") if line.strip()]
    return ParsedRow(schemas=schemas, user=users[-1]["content"], calls=calls, prose=None)


def assert_runtime_parseable(row: dict[str, Any]) -> None:
    """
    Assert the runtime can read what we wrote.

    This is the train/serve contract, and it is the one check worth running on every row. The
    dataset teaches the model to emit a string; ``olmo_core.tools.protocol.parse_function_calls``
    is what turns that string back into a call at inference. If a row we ship cannot survive that
    function, we are training a behaviour our own runtime rejects — and the failure would only show
    up after training, as a model that emits something nothing can execute.

    :param row: A row from :func:`build_row`.

    :raises ValueError: If the runtime parser refuses the assistant turn, or recovers a different
        call than we serialised.
    """
    content = row["messages"][-1]["content"]
    ours = parse_row(row).calls
    try:
        theirs = parse_function_calls(content)
    except Exception as e:  # ToolCallParseError, but do not couple to the type here
        raise ValueError(f"the runtime parser rejects this row: {e}") from e

    if len(ours) != len(theirs):
        raise ValueError(f"we serialised {len(ours)} call(s); the runtime recovered {len(theirs)}")
    for mine, runtime in zip(ours, theirs):
        if mine.name != runtime.name or mine.arguments != runtime.arguments:
            raise ValueError(
                f"runtime disagreement: we wrote {mine.name}({mine.arguments!r}), "
                f"the runtime read {runtime.name}({runtime.arguments!r})"
            )


def assert_round_trip(row: dict[str, Any]) -> ParsedRow:
    """
    Parse a row and re-render it, asserting the bytes are unchanged.

    :param row: A row from :func:`build_row`.

    :returns: The parsed content.

    :raises ValueError: If re-rendering does not reproduce the row exactly.
    """
    parsed = parse_row(row)
    if parsed.calls:
        rebuilt = serialize_calls(parsed.calls)
        original = row["messages"][-1]["content"]
        if rebuilt != original:
            raise ValueError(
                f"call block does not round-trip:\n  was {original!r}\n  got {rebuilt!r}"
            )
    rebuilt_schemas = serialize_schemas(parsed.schemas)
    if rebuilt_schemas not in row["messages"][0]["content"]:
        raise ValueError("schema block does not round-trip")
    return parsed


def iter_jsonl(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    """
    Render rows as ``.jsonl`` lines, asserting each round-trips first.

    :param rows: Rows from :func:`build_row`.

    :returns: One JSON line per row, newline-free.
    """
    for row in rows:
        assert_round_trip(row)
        assert_runtime_parseable(row)
        yield json.dumps(row, ensure_ascii=False)
