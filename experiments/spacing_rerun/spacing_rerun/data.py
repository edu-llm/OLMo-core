from __future__ import annotations

import collections
import random
import re

from .common import digest, require

DATASET = "jwkirchenbauer/fictionalqa"
REVISION = "131cb74fdc3e601b5e896ed768ad9852ea35a8f9"
MODEL = "allenai/DataDecide-dolma1_7-300M"
MODEL_REVISION_PREFIX = "4b1b42ff"


def normalize(text):
    """Frozen EM rule: Unicode casefold, whitespace and terminal punctuation only."""
    return " ".join(str(text).strip().casefold().split()).rstrip(".!?,;:")


def canonicalize(rows):
    """Resolve complete duplicate clusters before assigning events to roles."""
    by_id = {str(row["question_id"]): row for row in rows}
    require(len(by_id) == len(rows), "question_id is not unique")

    def root_of(key):
        seen = set()
        while True:
            require(key in by_id, f"Missing duplicate root {key}")
            require(key not in seen, f"Duplicate cycle at {key}")
            seen.add(key)
            root = str(by_id[key].get("duplicate_root") or key)
            if root == key:
                return key
            key = root

    clusters = collections.defaultdict(list)
    for key in sorted(by_id):
        clusters[root_of(key)].append(key)
    facts, cross_event_links, exact_text_owner = [], [], {}
    for root, members in sorted(clusters.items()):
        row = by_id[root]
        events = sorted({str(by_id[key]["event_id"]) for key in members})
        if len(events) > 1:
            cross_event_links.append(events)
        style = re.search(r"_style_([a-z]+)_num_", str(row["fiction_id"]))
        require(style is not None, f"No style in {row['fiction_id']}")
        fact = {"id": root, "event": str(row["event_id"]), "events": events,
                "style": style.group(1), "members": members,
                "statement": str(row["fict"]).strip(),
                "question": str(row["question"]).strip(),
                "answer": str(row["natural_answer"]).strip()}
        require(all(fact[k] for k in ("statement", "question", "answer")),
                f"Empty canonical field at {root}; resolve in the source audit")
        fact["aliases"] = [fact["answer"]]
        for member in members:
            other = by_id[member]
            # Exact supporting text / QA duplicates can link distinct root clusters.
            for signature in (normalize(other["fict"]),
                              normalize(other["question"]) + "|" + normalize(other["natural_answer"])):
                event = str(other["event_id"])
                if signature in exact_text_owner and exact_text_owner[signature] != event:
                    cross_event_links.append(sorted([exact_text_owner[signature], event]))
                exact_text_owner[signature] = event
        facts.append(fact)
    return facts, {"rows": len(rows), "facts": len(facts),
                   "events": len({str(row['event_id']) for row in rows}),
                   "cross_event_links": sorted({tuple(x) for x in cross_event_links}),
                   "blind_filter": "none", "source_sha256": digest(rows)}


def outer_partition(facts, links, seed=20261007):
    """Reject linked events rather than silently violate the declared 20/80 geometry."""
    require(not links, "Cross-event duplicate links require an audited grouped/excluded manifest")
    events = sorted({f["event"] for f in facts})
    require(len(events) == 100, f"Expected 100 events; found {len(events)}")
    random.Random(seed).shuffle(events)
    result = {"seed": seed, "development": sorted(events[:20]),
              "confirmation": sorted(events[20:])}
    result["sha256"] = digest(result)
    return result


def assign_roles(facts, partition, mode, seed):
    require(mode in ("development", "confirmation"), "Unknown mode")
    counts = {"old": 5, "new": 7, "control": 5, "qa": 3} if mode == "development" else {
        "old": 20, "new": 30, "control": 20, "qa": 10}
    events = list(partition[mode])
    require(len(events) == sum(counts.values()), "Role geometry does not fit partition")
    # All source events normally contain all five styles. Minimize remaining style
    # imbalance over a fixed 128 random candidates, using metadata only.
    style_counts = {e: collections.Counter(f["style"] for f in facts if f["event"] == e)
                    for e in events}
    styles = sorted({s for c in style_counts.values() for s in c})
    rng, best, best_score = random.Random(seed), None, float("inf")
    for _ in range(128):
        rng.shuffle(events)
        offset, candidate, score = 0, {}, 0.0
        for role, count in counts.items():
            chosen = events[offset:offset + count]
            candidate.update({event: role for event in chosen})
            totals = collections.Counter()
            for event in chosen:
                totals.update(style_counts[event])
            mean = sum(totals.values()) / max(len(styles), 1)
            score += sum((totals[s] - mean) ** 2 for s in styles) / count
            offset += count
        if score < best_score:
            best, best_score = candidate.copy(), score
    selected = [dict(f, role=best[f["event"]]) for f in facts if f["event"] in best]
    manifest = {"mode": mode, "seed": seed, "counts": counts, "roles": best,
                "style_imbalance": best_score, "facts": selected,
                "outer_partition_sha256": partition["sha256"]}
    manifest["sha256"] = digest(manifest)
    return manifest


def assign_development_scale_roles(facts, partition, mode, counts, fixed_qa_events):
    """Stage-1 scale probe: maximize old coverage using metadata-only placeholders."""
    from .units import unit_identity
    require(mode == "development", "Scale-probe role geometry is development-only")
    require(counts == {"old": 15, "new": 1, "control": 1, "qa": 3}, "Unsupported development scale role counts")
    events = set(partition["development"])
    require(len(events) == 20 and len(fixed_qa_events) == 3 and len(set(fixed_qa_events)) == 3 and
            set(fixed_qa_events) <= events, "Scale probe requires three fixed development teaching events")
    units = collections.defaultdict(set)
    for fact in facts:
        if fact["event"] in events:
            units[fact["event"]].add(unit_identity(fact["event"], fact.get("source_statement", fact["statement"]))[0])
    require(all(units[event] for event in events), "Every development scale event requires source units")
    remaining = sorted(events - set(fixed_qa_events), key=lambda event: (len(units[event]), event))
    roles = {event: "old" for event in remaining}
    roles[remaining[0]], roles[remaining[1]] = "new", "control"
    roles.update({event: "qa" for event in fixed_qa_events})
    selected = [dict(fact, role=roles[fact["event"]]) for fact in facts if fact["event"] in roles]
    manifest = {"mode": "development", "counts": counts, "roles": roles, "facts": selected,
                "outer_partition_sha256": partition["sha256"], "development_scale_probe": True,
                "role_selection_policy": "two_smallest_source_unit_events_new_then_control; other_non_teaching_events_old",
                "source_unit_counts_by_event": {event: len(units[event]) for event in sorted(events)}}
    manifest["sha256"] = digest(manifest)
    return manifest


def join_metadata(facts, source_tables):
    """Record every source metadata join; evaluation adapters can consume frozen MCQs."""
    report = {}
    for name, rows in source_tables.items():
        index = collections.defaultdict(list)
        for row in rows:
            if "question_id" in row:
                index[str(row["question_id"])].append(row)
        require(index, f"No question_id join key in {name}")
        missing = [f["id"] for f in facts if f["id"] not in index]
        require(not missing, f"{name}: missing {len(missing)} canonical question IDs")
        for fact in facts:
            fact.setdefault("source_metadata", {})[name] = index[fact["id"]]
        report[name] = {"rows": len(rows), "sha256": digest(rows), "missing": missing}
    return report
