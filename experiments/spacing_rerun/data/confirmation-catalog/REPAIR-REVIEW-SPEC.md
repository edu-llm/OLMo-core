# Round-5 repair review spec (Sonnet, Channel A)

You are an independent reviewer. You did not author these records. Read, in this order:
/tmp/p4-confirm-catalog/AUDIT-CONFORMANCE-AMENDMENT.md (part 3), /tmp/p4-confirm-catalog/AUTHORING-SPEC.md (Channel A and Review
sections) and /tmp/p4-confirm-catalog/REPAIR-SPEC.md (the rules the author followed).
For each event read /tmp/p4-confirm-catalog/repair-packets/<event>.json (the items, factsheet lines and the records before repair)
and /tmp/p4-confirm-catalog/authored/A-r5/<event>.json (the repaired records). Do NOT open paraphrase-packets, authored/B,
compiled* or any other file.

Write /tmp/p4-confirm-catalog/reviews/A/<event>.r5.json:
{"event": "<event>", "channel": "A", "round": 5, "reviewer": "claude-sonnet",
 "decisions": [{"id": "<unit_id / group id / source_qa id>", "decision": "APPROVE"|"REVISE"|"REJECT",
                "reason": "<required unless APPROVE>", "suggested_fix": "<optional>"}],
 "missing_records": []}

Give one decision for EVERY source unit, group and source_qa record in the A-r5 file, whether or not it changed. Check:
- each item in the packet is actually fixed per REPAIR-SPEC.md rules 1-9;
- a preserved (assertion_only / conflict_flagged) unit is alone in its group, the statement equals its source_statement exactly,
  and it has a note; a unit changed to grounded really is supported by the exact quoted line(s);
- merged or reworded groups state one fact that is supported for every member (every_owner);
- every source_qa record: closed-book, event-specific context, does not contain its answer, asks for exactly its label, is
  answerable from the group's statement, evidence quotes are exact lines (1-indexed) or empty only for preserved groups;
- nothing outside the items was changed without reason.
Do not edit any authored file.
