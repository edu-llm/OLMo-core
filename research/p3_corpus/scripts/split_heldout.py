"""Draw the held-out facts once per FAMILY, then split train from eval.

Doing this inside each builder was wrong. `mizar` and `thproofs` both cite the
MML, and `prf2` and `enigma` both cite MPTP, so a fact held out of one shard was
being trained on freely by its sibling — the sweep found 415 such facts. A fact
is only held out if nothing in the corpus trains on it, which is a property of
the union, not of any one shard.

So builders now run with `--heldout 0` and emit every example. This pass:

  1. pools citation counts across all shards in a family
  2. samples the tail (cited once or twice) from that pool
  3. moves every example citing a held-out fact into eval, in EVERY shard of the
     family, plus any example whose own theorem is held out — a fact's statement
     leaks as the goal of its own proof

Usage:
    python scripts/split_heldout.py --corpus corpus \
        --family mizar=mizar,thproofs --family atp=prf2,enigma \
        --family metamath=metamath --family isabelle=isabelle
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

ALTERNATE_SUFFIX = re.compile(r"#\d+$")
ATP_WRAPPERS = {"enigma", "prf2"}
METAMATH_DATABASES = {"set", "iset", "nf"}
METAMATH_TARGET = re.compile(r"^\s*\d+\s+\S+\s+(.+?)\s*$")
CANONICALIZATION_VERSION = 2
CANONICALIZATION_SCHEMES = {
    "atp": "tptp-layout-v2",
    "metamath": "metamath-token-v2",
    "mizar": "quoted-layout-v2",
    "isabelle": "quoted-layout-v2",
}
ATP_DEDUPLICATION_POLICY = "atp-exact-structured"
ATP_DEDUPLICATION_VERSION = 1


@dataclass(frozen=True)
class HeldoutExposure:
    """How one record touches a family-level held-out equivalence class."""

    named_fact: bool
    own_theorem: bool
    statement_alias: bool

    @property
    def should_eval(self):
        """Whether this row must be excluded from training."""
        return self.named_fact or self.own_theorem or self.statement_alias


def _canonical_tptp(statement):
    """Remove TPTP layout outside quotes while preserving quoted atoms."""
    out = []
    quote = None
    escaped = False
    for ch in str(statement).strip():
        if quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif not ch.isspace():
            out.append(ch)
    return "".join(out)


def _canonical_quoted_layout(statement):
    """Collapse layout outside quotes without changing quoted content."""
    out = []
    quote = None
    escaped = False
    pending_space = False
    for ch in str(statement).strip():
        if quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            quote = ch
            out.append(ch)
        elif ch.isspace():
            pending_space = True
        else:
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append(ch)
    return "".join(out)


def canonicalization_metadata(family):
    """Describe the family-scoped statement identity written to manifests."""
    return {
        "family": family,
        "scheme": CANONICALIZATION_SCHEMES.get(family, "quoted-layout-v2"),
        "version": CANONICALIZATION_VERSION,
    }


def canonical_statement(statement, *, family):
    """Canonicalize a statement using its proof language's syntax rules."""
    if family == "atp":
        return _canonical_tptp(statement)
    if family == "metamath":
        return " ".join(str(statement).split())
    return _canonical_quoted_layout(statement)


def statement_hash(statement, *, family):
    """Stable family-scoped exact-equivalence key for one statement."""
    metadata = canonicalization_metadata(family)
    payload = "\0".join(
        (
            "statement",
            str(metadata["version"]),
            metadata["family"],
            metadata["scheme"],
            canonical_statement(statement, family=family),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_theorem_identity(theorem, *, family):
    """Normalize only wrappers that belong to the record's proof family."""
    identity = str(theorem)
    if family == "atp":
        prefix, separator, rest = identity.partition(":")
        if separator and prefix.lower() in ATP_WRAPPERS:
            identity = rest
        return ALTERNATE_SUFFIX.sub("", identity)
    if family == "metamath":
        prefix, separator, rest = identity.partition(":")
        if separator and prefix.lower() in METAMATH_DATABASES:
            return rest
    return identity


def _canonical_atp_mapping(statements):
    return sorted(
        (
            str(name),
            canonical_statement(statement, family="atp"),
        )
        for name, statement in statements.items()
    )


def _canonical_atp_step(step):
    canonical = {}
    for key, value in step.items():
        if key in {"formula", "source"}:
            canonical[key] = canonical_statement(value, family="atp")
        elif key == "parent_sources" and isinstance(value, list):
            canonical[key] = [
                canonical_statement(parent, family="atp") for parent in value
            ]
        else:
            canonical[key] = value
    return canonical


def _numbered_lines(path):
    with open(path, encoding="utf-8") as source_file:
        yield from enumerate(source_file, 1)


def exact_atp_signature(record):
    """Hash one exact ATP proof independent of wrapper, ID, and map ordering."""
    proof_steps = record.get("proof_steps")
    structured = isinstance(proof_steps, list) and bool(proof_steps)
    signature = {
        "theorem": normalize_theorem_identity(
            record.get("theorem", ""), family="atp"
        ),
        "goal": canonical_statement(record.get("goal", ""), family="atp"),
        "goal_name": record.get("goal_name"),
        "facts": _canonical_atp_mapping(record.get("facts", {})),
        "local_inputs": _canonical_atp_mapping(
            record.get("local_inputs", {})
        ),
    }
    if structured:
        signature["schema"] = "atp-v2-structured"
        signature["proof_steps"] = [
            _canonical_atp_step(step) if isinstance(step, dict) else step
            for step in proof_steps
        ]
    else:
        # Legacy targets are not parsed back into guessed structure. Normalize
        # only line endings and surrounding layout to avoid false deduplication.
        signature["schema"] = "atp-legacy-conservative"
        signature["target"] = (
            str(record.get("target", "")).replace("\r\n", "\n").strip()
        )
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    versioned = (
        f"{ATP_DEDUPLICATION_POLICY}-v{ATP_DEDUPLICATION_VERSION}\0{payload}"
    )
    return hashlib.sha256(versioned.encode()).hexdigest()


def _legacy_target_formulas(target):
    """Recover formulas from the old rendered target when structured steps lack."""
    formulas = []
    for line in str(target).splitlines():
        left, marker, _ = line.rpartition("   [")
        if not marker:
            continue
        fields = left.split(None, 2)
        if len(fields) == 3:
            formulas.append(fields[2])
    return formulas


def record_statement_hashes(record, *, family):
    """Hash every statement exposed or supervised by a record."""
    statements = list(record.get("facts", {}).values())
    statements.extend(record.get("local_inputs", {}).values())
    if family == "metamath":
        statements.extend(record.get("local_assumptions", {}).values())
    if record.get("goal"):
        statements.append(record["goal"])
    proof_steps = record.get("proof_steps")
    if isinstance(proof_steps, list):
        statements.extend(
            step.get("formula", "")
            for step in proof_steps
            if isinstance(step, dict) and step.get("formula")
        )
    elif family == "atp":
        statements.extend(_legacy_target_formulas(record.get("target", "")))
    if family == "metamath":
        for line in str(record.get("target", "")).splitlines():
            match = METAMATH_TARGET.match(line)
            if match is not None:
                statements.append(match.group(1))
    return {
        statement_hash(statement, family=family)
        for statement in statements
        if statement
    }


def heldout_exposure(record, held_facts, held_statement_hashes, *, family):
    """Classify all name-, theorem-, and exact-statement holdout paths."""
    supplied_names = set(record.get("facts", {})) | set(
        record.get("local_inputs", {})
    )
    return HeldoutExposure(
        named_fact=bool(supplied_names & held_facts),
        own_theorem=normalize_theorem_identity(
            record.get("theorem", ""), family=family
        )
        in held_facts,
        statement_alias=bool(
            record_statement_hashes(record, family=family)
            & held_statement_hashes
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--family", action="append", required=True,
                    help="NAME=shard1,shard2")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    rd = os.path.join(a.corpus, "raw")
    sd = os.path.join(a.corpus, "shards")
    ed = os.path.join(a.corpus, "eval")
    hd = os.path.join(a.corpus, "heldout")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(ed, exist_ok=True)
    os.makedirs(hd, exist_ok=True)

    requested_families = set()
    requested_shards = set()
    for spec in a.family:
        fam, _, rest = spec.partition("=")
        requested_families.add(fam)
        requested_shards.update(s for s in rest.split(",") if s)
        shards = [s for s in rest.split(",")
                  if os.path.exists(os.path.join(rd, f"{s}.jsonl"))]
        if not shards:
            print(f"{fam}: no shards present, skipping")
            continue

        duplicate_lines = defaultdict(set)
        duplicate_counts = {shard: 0 for shard in shards}
        if fam == "atp":
            seen_signatures = {}
            for shard in shards:
                path = os.path.join(rd, f"{shard}.jsonl")
                for line_number, line in _numbered_lines(path):
                    signature = exact_atp_signature(json.loads(line))
                    if signature in seen_signatures:
                        duplicate_lines[shard].add(line_number)
                        duplicate_counts[shard] += 1
                    else:
                        seen_signatures[signature] = (shard, line_number)

        counts = Counter()
        statement_keys = defaultdict(set)
        aliases_by_statement = defaultdict(set)
        inconsistent = set()
        for s in shards:
            path = os.path.join(rd, f"{s}.jsonl")
            for line_number, line in _numbered_lines(path):
                if line_number in duplicate_lines[s]:
                    continue
                record = json.loads(line)
                for n, st in record["facts"].items():
                    key = statement_hash(st, family=fam)
                    counts[n] += 1
                    statement_keys[n].add(key)
                    aliases_by_statement[key].add(n)
                for n, st in record.get("local_inputs", {}).items():
                    aliases_by_statement[
                        statement_hash(st, family=fam)
                    ].add(n)
                for n, st in record.get("local_assumptions", {}).items():
                    aliases_by_statement[
                        statement_hash(st, family=fam)
                    ].add(n)
                if record.get("goal"):
                    aliases_by_statement[
                        statement_hash(record["goal"], family=fam)
                    ].add(
                        normalize_theorem_identity(
                            record.get("theorem", ""), family=fam
                        )
                    )
        inconsistent = {
            name for name, keys in statement_keys.items() if len(keys) > 1
        }
        if inconsistent:
            # prf2 and ENIGMA were generated against different MML snapshots, so
            # a handful of definitions reference differently-numbered generated
            # Fraenkel terms (a_2_2_waybel_0 vs a_2_7_waybel_0). Pinning one
            # spelling would contradict the other's proof, so drop the examples
            # instead — it is a few hundred out of tens of thousands.
            print(f"  {len(inconsistent)} name(s) disagree across the family: "
                  f"{sorted(inconsistent)[:4]}")
        tail = sorted(n for n, c in counts.items() if c in (1, 2))
        held = set(random.Random(a.seed).sample(
            tail, min(a.heldout, len(tail))))
        held_statement_hashes = {
            key for name in held for key in statement_keys.get(name, ())
        }
        statement_aliases = {
            alias
            for key in held_statement_hashes
            for alias in aliases_by_statement.get(key, ())
        }
        manifest = {
            "facts": sorted(held),
            "seed": a.seed,
            "family": fam,
            "shards": shards,
            "pooled_citations": len(counts),
            "tail_size": len(tail),
            "statement_hashes": sorted(held_statement_hashes),
            "statement_aliases": sorted(statement_aliases - held),
            "canonicalization": canonicalization_metadata(fam),
            "policy": "cited 1-2x across the whole family; every example "
            "citing one, every #N proof of its normalized theorem, "
            "and every exact canonical statement alias in facts, "
            "local inputs, goals, or targets moved to eval",
        }
        if fam == "atp":
            manifest["deduplication"] = {
                "policy": ATP_DEDUPLICATION_POLICY,
                "version": ATP_DEDUPLICATION_VERSION,
                "priority": "ordered family shards then source line",
                "duplicates_total": sum(duplicate_counts.values()),
                "duplicates_by_shard": duplicate_counts,
            }
        manifest_path = os.path.join(hd, f"{fam}.json")
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=1)

        tot_tr = tot_ev = 0
        exposure_counts = Counter()
        for s in shards:
            tr_count = ev_count = 0
            disagreement_drop = 0
            duplicate_drop = 0
            src = os.path.join(rd, f"{s}.jsonl")
            train_path = os.path.join(sd, f"{s}.jsonl")
            eval_path = os.path.join(ed, f"{s}.jsonl")
            train_tmp = train_path + ".tmp"
            eval_tmp = eval_path + ".tmp"
            try:
                with open(src, encoding="utf-8") as source, \
                        open(train_tmp, "w", encoding="utf-8") as train_file, \
                        open(eval_tmp, "w", encoding="utf-8") as eval_file:
                    for line_number, line in enumerate(source, 1):
                        if line_number in duplicate_lines[s]:
                            duplicate_drop += 1
                            continue
                        r = json.loads(line)
                        if set(r["facts"]) & inconsistent:
                            disagreement_drop += 1
                            continue
                        exposure = heldout_exposure(
                            r, held, held_statement_hashes, family=fam
                        )
                        if exposure.should_eval:
                            eval_file.write(line)
                            ev_count += 1
                            exposure_counts["named fact"] += exposure.named_fact
                            exposure_counts["own theorem"] += exposure.own_theorem
                            exposure_counts[
                                "statement alias"
                            ] += exposure.statement_alias
                            if (
                                exposure.statement_alias
                                and not exposure.named_fact
                                and not exposure.own_theorem
                            ):
                                exposure_counts["alias only"] += 1
                        else:
                            train_file.write(line)
                            tr_count += 1
                os.replace(train_tmp, train_path)
                os.replace(eval_tmp, eval_path)
            finally:
                for tmp in (train_tmp, eval_tmp):
                    if os.path.exists(tmp):
                        os.remove(tmp)
            dropped = []
            if duplicate_drop:
                dropped.append(f"{duplicate_drop:,} exact duplicate")
            if disagreement_drop:
                dropped.append(
                    f"{disagreement_drop:,} name disagreement"
                )
            print(
                f"  {s:<12} train {tr_count:>8,}   eval {ev_count:>6,}"
                + (f"   dropped {', '.join(dropped)}" if dropped else "")
            )
            tot_tr += tr_count
            tot_ev += ev_count

        print(f"{fam}: held {len(held):,} of {len(tail):,} tail facts "
              f"(pool {len(counts):,})   train {tot_tr:,}   eval {tot_ev:,}")
        print("  eval exposure paths: "
              + ", ".join(f"{name} {count:,}"
                          for name, count in exposure_counts.items())
              + "\n")

    # Remove only obsolete per-shard manifests covered by this invocation. A
    # partial rebuild must never delete unrelated families' manifests.
    for f in os.listdir(hd):
        stem, ext = os.path.splitext(f)
        if (
            ext == ".json"
            and stem in requested_shards
            and stem not in requested_families
        ):
            os.remove(os.path.join(hd, f))
            print(f"removed stale {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
