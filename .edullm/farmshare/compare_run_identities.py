#!/usr/bin/env python3
"""Make the freeze contract executable across runs, not just within one.

`assert_resume_fingerprint` compares a run to ITSELF on resume. Nothing
compared run 7 to run 1, so a frozen constant could change mid-campaign and
both runs would be internally consistent -- the drift would surface only if
somebody diffed the JSON by hand. Since the entire point of this rebuild is
that every run stays permanently poolable, that check has to exist before
the first campaign run, not after somebody notices a discrepancy.

The contract is expressed as an allowlist of the axes the design is ALLOWED
to vary. Everything else must be byte-identical across every run compared.
Inverting it this way means a newly added identity field is protected by
default: forget to classify it and it is treated as frozen, which fails
loudly instead of silently permitting drift.

Two refinements beyond a flat diff:

  * `parent` and `difficulty_metric_definition` legitimately differ BETWEEN
    metrics but must be identical WITHIN one. Two mtld runs reading
    different corpus manifests means somebody re-materialized, which
    unpools them -- so those fields are compared per metric group.
  * a run built from a dirty tree is reported as unpoolable regardless of
    whether its identity matches, because its code is not recoverable.

Usage:
  compare_run_identities.py RUN_DIR [RUN_DIR ...]
  compare_run_identities.py --replicates RUN_DIR [RUN_DIR ...]

`--replicates` additionally requires that the runs differ ONLY in seed,
which is what a multi-seed condition means.

Exit status: 0 if every run is mutually poolable, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Axes the design varies on purpose. Anything absent from this set is frozen.
DESIGNED_DIMENSIONS = frozenset(
    {
        "arm",  # derived from the three below
        "pacing",
        "difficulty_metric",
        "lr_schedule",
        "seed",
        # Per-metric, so free across metrics but checked within one.
        "parent",
        "difficulty_metric_definition",
    }
)

# Of those, the ones that must still agree among runs sharing a metric.
PER_METRIC_FIELDS = ("parent", "difficulty_metric_definition")

# Replicates of one condition may differ only here.
REPLICATE_FREE = frozenset({"seed"})


class RunRecord:
    """One run's identity plus the provenance needed to judge it poolable."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.identity: dict | None = None
        self.commit: str | None = None
        self.dirty: bool | None = None
        self.errors: list[str] = []
        self._load()

    def _load(self) -> None:
        provenance = self.run_dir / "code_provenance.json"
        if provenance.is_file():
            try:
                payload = json.loads(provenance.read_text(encoding="utf-8"))
                self.commit = payload.get("commit")
                self.dirty = payload.get("dirty")
            except (OSError, json.JSONDecodeError) as exc:
                self.errors.append(f"unreadable code_provenance.json: {exc}")
        else:
            self.errors.append("no code_provenance.json (run predates commit binding)")

        candidates = sorted(self.run_dir.glob("runs/*/progress/run_identity.json"))
        if not candidates:
            candidates = sorted(self.run_dir.glob("runs/*/progress/current_fingerprint/run_fingerprint.json"))
        if not candidates:
            self.errors.append("no run identity found under runs/*/progress/")
            return
        if len(candidates) > 1:
            self.errors.append(f"{len(candidates)} identities found; expected one arm per run dir")
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.errors.append(f"unreadable identity: {exc}")
            return
        # run_fingerprint.json wraps the identity; run_identity.json is bare.
        self.identity = payload.get("identity", payload) if isinstance(payload, dict) else None
        if not isinstance(self.identity, dict):
            self.errors.append("identity is not an object")

    @property
    def label(self) -> str:
        return self.run_dir.name

    @property
    def metric(self):
        return (self.identity or {}).get("difficulty_metric")


def compare_identities(records: list[RunRecord], replicates: bool = False) -> list[str]:
    """Return a list of contract violations. Empty means mutually poolable."""
    violations: list[str] = []
    usable = [r for r in records if isinstance(r.identity, dict)]

    for record in records:
        for error in record.errors:
            violations.append(f"{record.label}: {error}")
        if record.dirty:
            violations.append(
                f"{record.label}: built from a dirty tree (commit {record.commit}), so its "
                f"code is not recoverable and it is not poolable"
            )

    if len(usable) < 2:
        return violations

    # Frozen fields: every key seen anywhere that is not a designed axis.
    all_keys: set[str] = set()
    for record in usable:
        all_keys.update(record.identity or {})
    frozen = sorted(all_keys - DESIGNED_DIMENSIONS)

    reference = usable[0]
    for key in frozen:
        baseline = (reference.identity or {}).get(key)
        for record in usable[1:]:
            value = (record.identity or {}).get(key)
            if value != baseline:
                violations.append(
                    f"frozen field {key!r} differs: {reference.label} has "
                    f"{_brief(baseline)} but {record.label} has {_brief(value)}"
                )

    # Per-metric fields must agree within each metric group.
    groups: dict[object, list[RunRecord]] = {}
    for record in usable:
        groups.setdefault(record.metric, []).append(record)
    for metric, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(group) < 2:
            continue
        head = group[0]
        for key in PER_METRIC_FIELDS:
            baseline = (head.identity or {}).get(key)
            for record in group[1:]:
                value = (record.identity or {}).get(key)
                if value != baseline:
                    violations.append(
                        f"metric {metric!r}: {key!r} differs between {head.label} and "
                        f"{record.label} -- same metric must mean the same corpus and "
                        f"the same metric definition"
                    )

    if replicates:
        for key in sorted(DESIGNED_DIMENSIONS - REPLICATE_FREE - set(PER_METRIC_FIELDS)):
            baseline = (reference.identity or {}).get(key)
            for record in usable[1:]:
                value = (record.identity or {}).get(key)
                if value != baseline:
                    violations.append(
                        f"--replicates: {key!r} differs ({_brief(baseline)} vs "
                        f"{_brief(value)}); replicates may differ only in seed"
                    )
        seeds = [(r.identity or {}).get("seed") for r in usable]
        if len(set(map(repr, seeds))) != len(seeds):
            violations.append(f"--replicates: seeds are not distinct: {seeds}")

    return violations


def _brief(value: object, limit: int = 70) -> str:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--replicates",
        action="store_true",
        help="require the runs to differ only in seed (one multi-seed condition)",
    )
    args = parser.parse_args()

    records = [RunRecord(path) for path in args.run_dirs]

    print(f"{'run':<44} {'commit':<10} {'arm':<34} seed")
    for record in records:
        identity = record.identity or {}
        commit = (record.commit or "?")[:8]
        flag = " DIRTY" if record.dirty else ""
        print(
            f"{record.label:<44} {commit + flag:<10} "
            f"{str(identity.get('arm', '?')):<34} {identity.get('seed', '?')}"
        )

    violations = compare_identities(records, replicates=args.replicates)
    print()
    if violations:
        for violation in violations:
            print(f"  VIOLATION  {violation}")
        print(f"\n{len(violations)} violation(s): these runs are NOT mutually poolable")
        return 1
    print(f"all {len(records)} run(s) are mutually poolable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
