# Confirmation catalog authoring spec

Read ELECTIONS-AMENDMENT.md first. Work only from the files named for your task. Write JSON with
UTF-8, valid syntax, one output file per event. Never invent facts not in the source assertion or
the event factsheet.

## Channel A: grounded declarations + source-QA (input: packets/author-packets/chunk-NN.json)

Each packet `units[]` entry has: unit_id, event, source_assertion, canonical_answer_labels,
canonical_probes[] (opaque ids + answer labels only), evidence_candidates[] (factsheet line + quote).
The event factsheet text is in `factsheets[event].text`. Line numbers are 1-indexed lines of
that text split on "\n" (the same numbering the evidence_candidates use).

You must NOT have and must NOT write evaluation questions. Do not read any paraphrase-packets file.

Grounding policy: the source assertion is authoritative. The factsheet may add identifiers and
context. Never override an explicit conflict or add unsupported detail; if the factsheet conflicts,
keep the assertion and set status "conflict_flagged" with a note.

Output `authored/A/<event>.json`:
```
{
 "event": "event_000",
 "source_packet_sha256": "<packet sha256>",
 "source_units": [
  {"unit_id": "...", "event_id": "...", "source_statement": "<source_assertion verbatim>",
   "canonical_answers": [<canonical_answer_labels verbatim>],
   "probe_ids": [<opaque_probe_id of each canonical_probe>],
   "group_id": "ground-<event>-<nn>",
   "proposed_declaration": "<one declarative sentence, <=28 words, that states the fact so the answer label appears verbatim>",
   "evidence": [{"line": <int>, "quote": "<exact full line text from the factsheet>"}],
   "status": "grounded" | "assertion_only" | "conflict_flagged",
   "nonliteral_answer_labels": [<labels that do not appear verbatim in the declaration, normally empty>],
   "note": "<optional>"}
 ],
 "groups": [
  {"id": "ground-<event>-<nn>", "source_unit_ids": [...], "event": "...",
   "statement": "<the shared declaration>", "grouping_rationale": "<why these units state the same fact>"}
 ],
 "related_assertions_not_merged": [{"unit_ids": [...], "reason": "..."}],
 "source_qa": [
  {"id": "acq-<event>-<nn>", "unit_id": "<group id>", "event": "...", "answer": "<answer label verbatim>",
   "questions": ["<closed-book question form 1>", "<closed-book question form 2>"],
   "evidence": [{"line": <int>, "quote": "<exact line>"}],
   "rationale": "<how the evidence supports the answer; quote the source assertion>"}
 ]
}
```
Rules:
- Every unit in the packet for this event appears exactly once in source_units, and in exactly one group.
- Merge units into one group only if they state the same fact with the same answer (every_owner: the
  group statement must be supported for every member). Near-duplicates that differ go in
  related_assertions_not_merged.
- One source_qa record per distinct answer label per group (so each answer target gets 2 questions).
- Each question must be answerable from the declaration alone, must carry event-specific context so it
  cannot be confused with another event, must not contain the answer, and must ask for exactly that label.
- Quotes must be copied exactly (the full line). Use status "assertion_only" when no factsheet line supports it.

## Channel B: evaluation paraphrases (input: paraphrase-packets/chunk-NN.json)

You may see canonical evaluation questions. Do NOT read author-packets or authored/A files.
For every probe write one paraphrase that asks for the same thing, so that the canonical answer is still
exactly correct and complete, uses different wording and sentence structure, does not contain the answer,
and adds no new facts.

Output `authored/B/<event>.json`:
```
{"event": "...", "source_packet_sha256": "...",
 "paraphrases": [{"opaque_probe_id": "...", "canonical_question": "<verbatim>", "canonical_answer": "<verbatim>",
                  "paraphrase": "...", "note": "<optional>"}]}
```
Every probe of the event appears exactly once.

## Review (Sonnet)

Write `reviews/<channel>/<event>.r<round>.json`:
```
{"event": "...", "channel": "A"|"B", "round": 1|2, "reviewer": "claude-sonnet",
 "decisions": [{"id": "<unit_id / group id / acq id / opaque_probe_id>", "decision": "APPROVE"|"REVISE"|"REJECT",
                "reason": "<required unless APPROVE>", "suggested_fix": "<optional>"}],
 "missing_records": [<ids present in the packet but absent from the authored file>]}
```
Channel A: give a decision for every source unit, every group and every source_qa record. Check grounding
against the factsheet text and assertion, exact quotes, that merges are the same fact, that questions are
closed-book, carry event context, do not leak the answer and target exactly the label.
Channel B: one decision per probe. Check the canonical answer stays exactly correct and complete, no answer leak,
real rewording, no new facts.
Review every record. Do not edit the authored files.
