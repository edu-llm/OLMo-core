"""
Decide and enforce what goes in the test set, before any data is generated.

The point of a test set here is to answer "did the model learn to read a tool description and work
out the call", not "did it memorise which tool goes with which question". A random 10% split cannot
answer that, because the same tools appear on both sides and the model can pass by recall.

So two things are split, and both must be settled *before* generation, because generation consumes
them as inputs:

**Tools.** Some tools appear only in the test set. A test question then offers a function the model
has never seen and asks it to call it correctly from the description alone. The rule is *hold out
the sibling, not the orphan*: holding out a lone tool nobody trained on measures nothing, whereas
holding out ``percent_change`` while training ``compound_interest`` measures whether the model
transfers what it learned about one to the other.

**Question phrasings.** Unseen tools are not enough on their own. If test questions come from the
same phrase templates as training, the model can match the sentence shape and slot the new name in.
That looks like generalisation and is really template recall. So the phrasing bank is split too.

**The three tools that really exist are never held out.** ``calculator``, ``symbolic_math`` and
``web_search`` ship in ``olmo_core.tools`` and the served model can actually invoke them, so they
are what it most needs to be good at — holding one out would trade real capability for a
measurement. Their cells fall back to the substitute axis recorded on the domain (operand magnitude
for arithmetic, an entity bank for web-search), and for those cells "heldout measures schema
generalisation" is simply false and must not be claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REGISTRY_PATH = Path("docs/tool-call/frozen/tool_registry.json")

#: Tools that exist and run in `olmo_core.tools`. These are the ones the served model can actually
#: invoke, so they are the ones it most needs to be good at — and two of them execute, which means
#: rows targeting them can be verified by comparing against a computed result rather than only
#: checking the call's shape. None of them may be held out: holding out a tool the product ships
#: would trade real capability for a measurement.
IMPLEMENTED_TOOLS = ("calculator", "symbolic_math", "web_search")


def implemented_schemas() -> dict[str, dict[str, Any]]:
    """
    Read the live tool schemas out of ``olmo_core.tools``.

    Transcribing them into the frozen registry would let the dataset drift away from the runtime
    silently — the model would train against one description and meet another at inference. So the
    schemas are read from the code that serves them.

    :returns: Name -> the ``function`` half of each tool's JSON schema.

    :raises ImportError: If the tools package is unavailable.
    """
    from olmo_core.tools import CalculatorTool, StaticBackend, SymbolicMathTool, WebSearchTool

    tools = [
        CalculatorTool(),
        SymbolicMathTool(),
        WebSearchTool(backend=StaticBackend(results=[])),
    ]
    return {t.json_schema()["function"]["name"]: t.json_schema()["function"] for t in tools}


#: Fraction of the phrasing bank reserved for the test set.
HELDOUT_TEMPLATE_FRACTION = 0.15

#: Fixed so the split is reproducible. Changing it re-splits everything, which after generation
#: means regenerating; treat it as frozen once the first byte is written.
TEMPLATE_SPLIT_SALT = "tool-call-v1"


@dataclass(frozen=True)
class Tool:
    """One entry from the frozen registry.

    :param name: The function name.
    :param domain: Which of the four domains it belongs to.
    :param exec_kind: ``value`` if a stub computes the true result, else ``bind``.
    :param held_out: Whether it is reserved for the test set.
    :param sibling_of: The trained tool it is held out against.
    :param cannot_hold_out: A dominant tool that must stay in training.
    """

    name: str
    domain: str
    exec_kind: str
    held_out: bool = False
    sibling_of: str | None = None
    cannot_hold_out: bool = False
    implemented: bool = False


@dataclass(frozen=True)
class Registry:
    """The frozen tool registry, loaded.

    :param tools: Every authored tool.
    :param domains: Per-domain notes, including any substitute carve axis.
    """

    tools: tuple[Tool, ...]
    domains: dict[str, dict[str, Any]]

    def by_name(self, name: str) -> Tool | None:
        """:returns: The tool with this name, or ``None``."""
        return next((t for t in self.tools if t.name == name), None)

    def pool(self, *, split: str, domain: str | None = None) -> set[str]:
        """
        The tools a row in this split may offer.

        Training rows may never offer a held-out tool. Test rows may offer both — a realistic test
        row shows the unseen tool alongside familiar distractors — so the constraint that makes the
        test meaningful is on the *gold* tool, checked by :func:`check_corpus`.

        :param split: ``train`` or ``heldout``.
        :param domain: Restrict to one domain, or ``None`` for all.

        :returns: Allowed tool names.
        """
        if split not in {"train", "heldout"}:
            raise ValueError(f"split must be train or heldout, got {split!r}")
        out = set()
        for t in self.tools:
            if domain is not None and t.domain != domain:
                continue
            if split == "train" and t.held_out:
                continue
            out.add(t.name)
        return out

    def heldout_names(self, domain: str | None = None) -> set[str]:
        """:returns: Names reserved for the test set."""
        return {t.name for t in self.tools if t.held_out and (domain is None or t.domain == domain)}


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    """
    Load and validate the frozen registry.

    :param path: Where the registry lives.

    :returns: The parsed registry.

    :raises ValueError: If a name is duplicated, a held-out tool has no trained sibling, a tool is
        both held out and un-holdable, or a sibling reference dangles.
    """
    raw = json.loads(path.read_text())
    tools = tuple(
        Tool(
            name=t["name"],
            domain=t["domain"],
            exec_kind=t.get("exec", "bind"),
            held_out=bool(t.get("held_out", False)),
            sibling_of=t.get("sibling_of"),
            cannot_hold_out=bool(t.get("cannot_hold_out", False)),
            implemented=bool(t.get("implemented", False)),
        )
        for t in raw["tools"]
    )

    seen: set[str] = set()
    for t in tools:
        if t.name in seen:
            raise ValueError(f"duplicate tool name {t.name!r}; gate 5 forbids two schemas per name")
        seen.add(t.name)
        if "." in t.name:
            raise ValueError(
                f"tool name {t.name!r} is dotted. The runtime parser requires a flat name, so a "
                f"dotted tool would train the model to emit calls inference cannot read."
            )
        if t.implemented and t.held_out:
            raise ValueError(
                f"{t.name!r} is implemented and shipped, so holding it out trades real capability "
                f"for a measurement. Hold out an authored sibling instead."
            )

    for t in tools:
        if t.held_out and t.cannot_hold_out:
            raise ValueError(f"{t.name!r} is marked both held_out and cannot_hold_out")
        if t.held_out and not t.sibling_of:
            raise ValueError(
                f"{t.name!r} is held out with no sibling_of. Holding out an orphan measures "
                f"nothing — pair it with a trained tool or do not hold it out."
            )
        if t.sibling_of:
            sib = next((s for s in tools if s.name == t.sibling_of), None)
            if sib is None:
                raise ValueError(
                    f"{t.name!r} names sibling {t.sibling_of!r}, which is not in the registry"
                )
            if sib.held_out:
                raise ValueError(
                    f"{t.name!r} and its sibling {sib.name!r} are BOTH held out, so nothing trains "
                    f"the pattern the test is supposed to measure"
                )
            if sib.domain != t.domain:
                raise ValueError(f"{t.name!r} and sibling {sib.name!r} are in different domains")

    return Registry(tools=tools, domains=raw.get("domains", {}))


def validate_against_runtime(registry: Registry) -> None:
    """
    Check the registry still agrees with the tools ``olmo_core.tools`` actually ships.

    Kept separate from :func:`load_registry` so unit tests can build small fixture registries, and
    so this can run as its own CI gate. If someone adds a tool to the runtime, or renames one, this
    is what notices before the dataset is generated against a stale name.

    :param registry: A loaded registry.

    :raises ValueError: If the implemented set disagrees with the runtime, or a schema differs.
    """
    declared = {t.name for t in registry.tools if t.implemented}
    if declared != set(IMPLEMENTED_TOOLS):
        raise ValueError(
            f"registry marks {sorted(declared)} as implemented but olmo_core.tools ships "
            f"{sorted(IMPLEMENTED_TOOLS)}"
        )
    live = implemented_schemas()
    missing = set(IMPLEMENTED_TOOLS) - set(live)
    if missing:
        raise ValueError(f"olmo_core.tools no longer exposes {sorted(missing)}")
    for name in declared:
        tool = registry.by_name(name)
        assert tool is not None
        if tool.domain not in {"arithmetic", "web-search"}:
            raise ValueError(f"implemented tool {name!r} sits in unexpected domain {tool.domain!r}")


def split_templates(
    templates: Sequence[str],
    *,
    fraction: float = HELDOUT_TEMPLATE_FRACTION,
    salt: str = TEMPLATE_SPLIT_SALT,
) -> tuple[list[str], list[str]]:
    """
    Split the phrasing bank deterministically into train and heldout.

    Hashing rather than shuffling, so the split is stable as the bank grows: adding templates never
    moves an existing one across the boundary, which means an expanded bank does not silently leak
    a phrasing that was previously held out.

    :param templates: The phrasing bank.
    :param fraction: Share reserved for the test set.
    :param salt: Fixed salt; changing it re-splits everything.

    :returns: ``(train, heldout)``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    cutoff = int(fraction * (1 << 32))
    train: list[str] = []
    heldout: list[str] = []
    for t in templates:
        digest = hashlib.sha256(f"{salt}\x1f{t}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big")
        (heldout if bucket < cutoff else train).append(t)
    return train, heldout


@dataclass
class CorpusReport:
    """What :func:`check_corpus` found.

    :param rows: Rows inspected.
    :param violations: One message per problem, empty when the carve holds.
    :param heldout_tools_used: Held-out tools that actually appear as a gold call in the test set.
    """

    rows: int
    violations: list[str]
    heldout_tools_used: set[str]


def check_corpus(
    rows: Iterable[tuple[str, dict[str, Any]]], registry: Registry, *, parse_row: Any
) -> CorpusReport:
    """
    Verify a built corpus actually respects the carve.

    Three things are checked, and each converts an intention into a fact:

    1. no training row offers a held-out tool — otherwise the test set is contaminated;
    2. every test row's *gold* tool is held out, unless its domain has a substitute axis because
       its dominant tool cannot be held out;
    3. every gold tool's domain matches the directory it sits in, which is what keeps the domain
       axis honest.

    :param rows: ``(path, row)`` pairs, where the path carries the split and domain.
    :param registry: The frozen registry.
    :param parse_row: ``tool_call_serializer.parse_row``, injected to avoid a hard import.

    :returns: The report. Empty ``violations`` means the carve holds.
    """
    report = CorpusReport(rows=0, violations=[], heldout_tools_used=set())
    heldout_names = registry.heldout_names()

    for path, row in rows:
        report.rows += 1
        parts = Path(path).parts
        split = "heldout" if Path(path).name.startswith("heldout-") else "train"
        domain = parts[1] if len(parts) > 2 else None

        try:
            parsed = parse_row(row)
        except ValueError as e:
            report.violations.append(f"{path}: unparseable row: {e}")
            continue

        offered = {s.name for s in parsed.schemas}
        if split == "train":
            leaked = offered & heldout_names
            if leaked:
                report.violations.append(
                    f"{path}: training row offers held-out tool(s) {sorted(leaked)}"
                )

        if not parsed.calls:
            continue
        gold = parsed.calls[0].name
        tool = registry.by_name(gold)
        if tool is None:
            continue  # inherited upstream tool; the domain check below does not apply
        if domain and tool.domain != domain:
            report.violations.append(
                f"{path}: gold tool {gold!r} is domain {tool.domain!r} but sits under {domain!r}"
            )
        if split == "heldout":
            if tool.held_out:
                report.heldout_tools_used.add(gold)
            elif domain and not registry.domains.get(domain, {}).get("substitute_carve_axis"):
                report.violations.append(
                    f"{path}: test row's gold tool {gold!r} is not held out, and {domain!r} has no "
                    f"substitute carve axis, so this row measures recall rather than generalisation"
                )

    unused = heldout_names - report.heldout_tools_used
    if unused and report.rows:
        report.violations.append(
            f"held-out tools never used as a gold call in the test set: {sorted(unused)}"
        )
    return report


def summarise(registry: Registry) -> str:
    """:returns: A human-readable summary of the carve."""
    lines = ["Held-out tool carve", "=" * 60]
    for dom in sorted({t.domain for t in registry.tools}):
        dom_tools = [t for t in registry.tools if t.domain == dom]
        held = [t for t in dom_tools if t.held_out]
        blocked = [t for t in dom_tools if t.cannot_hold_out]
        pct = 100.0 * len(held) / len(dom_tools) if dom_tools else 0.0
        lines.append(f"\n{dom}: {len(dom_tools)} tools, {len(held)} held out ({pct:.1f}%)")
        for t in held:
            lines.append(f"    {t.name}  <- sibling of trained {t.sibling_of}")
        for t in blocked:
            axis = registry.domains.get(dom, {}).get("substitute_carve_axis")
            lines.append(f"    {t.name} CANNOT be held out; substitute axis: {axis}")
    total = len(registry.tools)
    held_total = len(registry.heldout_names())
    lines.append(
        f"\nTOTAL {total} tools, {held_total} held out ({100.0 * held_total / total:.1f}%)"
    )
    impl = [t for t in registry.tools if t.implemented]
    lines.append(
        f"\n{len(impl)} are IMPLEMENTED in olmo_core.tools, so the served model can really call "
        f"them:"
    )
    for t in impl:
        runs = "EXECUTES - verify by value" if t.exec_kind == "value" else "no deterministic result"
        lines.append(f"    {t.name:16s} {t.domain:12s} {runs}")
    lines.append(
        f"\nThe other {total - len(impl)} are authored schemas, there so the model learns to read a"
        f"\ntool description it has never seen. A model trained on three tools memorises three."
    )
    return "\n".join(lines)


def main() -> None:
    """Print the carve and validate the registry."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    args = p.parse_args()
    registry = load_registry(args.registry)
    print(summarise(registry))
    print("\nregistry validates: flat names, no duplicates, every held-out tool has a trained")
    print("sibling, and no implemented tool is held out.")
    try:
        validate_against_runtime(registry)
    except ImportError as e:
        print(f"\ncould not check against olmo_core.tools: {e}")
    else:
        print("and it agrees with the tools olmo_core.tools actually ships.")


if __name__ == "__main__":
    main()
