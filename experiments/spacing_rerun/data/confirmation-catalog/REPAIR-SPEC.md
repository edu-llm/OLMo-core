# Round-5 repair spec (Channel A author)

Read /tmp/p4-confirm-catalog/AUDIT-CONFORMANCE-AMENDMENT.md part 3 and /tmp/p4-confirm-catalog/AUTHORING-SPEC.md
(Channel A section) first. Work only from the repair packet for each event: /tmp/p4-confirm-catalog/repair-packets/<event>.json.
Do NOT open paraphrase-packets, authored/B, reviews, compiled* or any other file.

Each packet has `items` (what must be fixed, with instructions), `scope` (the records you may change: source_units,
groups, source_qa), `factsheet_lines_1_indexed`, `other_group_ids_in_event` and `other_question_forms_in_event`.

Write /tmp/p4-confirm-catalog/authored/A-r5/<event>.json:
{"event": "<event>", "round": 5, "repair_packet_sha256": "<packet sha256>",
 "source_units": [every unit in scope.source_units, full record, edited only where an item needs it],
 "groups": [the groups that cover exactly the scope units],
 "source_qa": [the source_qa records for exactly those groups],
 "changes": [{"item_kind": "<kind>", "summary": "<one sentence: what you changed and why>"}]}

Rules:
1. Never change unit_id, event_id, source_statement, canonical_answers or probe_ids.
2. Each scope unit appears once in source_units and in exactly one output group; each unit's group_id names it.
   Output groups contain only scope units.
3. A unit with status assertion_only or conflict_flagged must be alone in its group, and that group's statement must equal
   its source_statement character for character (copy it; do not add a final period). It needs a non-empty "note".
4. Reuse a scope group id for a group that survives. A new group gets id ground-<event>-r5<letter> (r5a, r5b, ...).
5. Every distinct answer label (case-insensitive) of the units in each group has exactly one source_qa record:
   unit_id = the group id, answer = the label verbatim, two closed-book questions, evidence, rationale. Keep an existing
   record's id and text where it is still correct. A new record gets id acq-<event>-r5<letter>.
6. Questions follow AUTHORING-SPEC.md, must not contain the group statement or any member's source_statement, and must differ
   (beyond capitalisation or articles) from every string in other_question_forms_in_event and from each other.
7. Evidence lines are 1-indexed keys of factsheet_lines_1_indexed and quotes are the exact full line text. Evidence may be
   empty only for preserved (assertion_only / conflict_flagged) units, their groups and their source_qa records.
8. Set "revision_round": 5 on every record you change or create.
9. Do not invent facts. Output valid JSON, UTF-8, one file per event.
