# P4 confirmation catalog: audit-conformance amendment (2026-10-15)

Status: chosen by the study owner on 2026-10-15 ("amend + repair"). The catalog had been approved under
ELECTIONS-AMENDMENT.md, ROUND3-AMENDMENT.md and FINAL-PASS-AMENDMENT.md. No model output or confirmation
outcome has been observed.

Reason: the confirmation code (`spacing_rerun/confirmation.py`, protocol
`spacing-confirmation-mixed-acquisition-agent-audit-v1`) was written for blind review packets and per-record
receipts. The catalog was made under the elections process instead. A read-only check
(`scripts/confirmation_catalog/audit_gap.py`) found three kinds of gap: what the reviewers saw, evidence for
unsupported assertions, and 63 structural records. This amendment changes the validator for the first two,
repairs the third, and lists the mechanical conversions.

## Part 1. Validator changes

These apply only when the audit wrapper binds this file's sha256 as `audit_conformance_amendment_sha256`.
The code pins that hash.

1. Source and evaluation review receipts may declare `author_rationales_seen: true`, because the Sonnet
   reviewers read the authored files, rationales and notes included. `model_outcomes_seen` must still be false.
   Evaluation receipts must still declare `acquisition_content_seen: false`.
2. Evidence may be empty for a source unit with status `preserve_source_ambiguity` or
   `preserve_source_conflict`, for its singleton group, and for the acquisition records that target that group.
   Every other record keeps the exact-line evidence rule.
3. Evaluation records may declare `authoring_inputs: ["canonical_evaluation_questions", "canonical_answers"]`.
   That is what the Channel B authors were given.
4. Review receipts are rebuilt after the fact from the per-event decision files in `reviews/`. Each receipt
   names one reviewer agent for one event and round and approves the exact digest of the final content. Every
   receipt and the wrapper carry `receipts_reconstructed: true`.

The rest of the validator is unchanged. In particular, a preserved unit must still be its own group whose
statement is its source assertion, verbatim.

## Part 2. Mechanical conversions

None of these makes a semantic choice. The converter reports a count for each.

- Evidence line numbers go from 1-indexed to 0-indexed. Each quote becomes the exact raw factsheet line;
  the catalog quotes had their whitespace stripped.
- A preserved declaration that differs from its source assertion only in trailing punctuation gets the source
  assertion verbatim (95 records).
- `nonliteral_answer_labels` is recomputed with the validator rule (3 differ, all case-only).
- Each group's evidence is the union of its member units' evidence. Each evaluation record's evidence is the
  evidence of its probe's source unit.
- Status names change: `assertion_only` becomes `preserve_source_ambiguity` and `conflict_flagged` becomes
  `preserve_source_conflict`. A unit's `note` becomes its `conflict_rationale`.
- Local ids become the hash-derived ids. `source_conflict_status` is derived from the member units' statuses.
- In event_024, two approved source-QA records share one answer target. The labels "The Secret Society of Aten"
  and "the Secret Society of Aten" are identical once normalized. The first record is kept and
  `acq-event_024-10-lc` is dropped; no text is changed.

## Part 3. Repair round (round 5)

Scope is limited to these records, plus the source-QA records attached to their groups:

- 32 preserved units merged into multi-unit groups. All 14 such groups contain only preserved units.
- 27 preserved singleton groups whose declaration rewords the source assertion.
- 2 pairs of groups in one event with identical declarations (event_029, event_054).
- 1 source-QA question form identical to an evaluation question form (event_067, `acq-event_067-12`).
- 2 preserved units without a note.

Rules:

1. A Claude Haiku author fixes each item, working from the event's factsheet and its own Channel A records. It
   never sees evaluation questions. The allowed fixes are:
   - For an `assertion_only` unit, either show that an exact factsheet line supports the assertion as stated
     (status `grounded`, with evidence), or keep it preserved.
   - A preserved unit becomes its own group with the source assertion verbatim. New groups get source-QA
     records under the authoring spec.
   - Duplicate declarations are merged when they state the same fact, or reworded when they do not.
   - The colliding question form is replaced.
2. The FarmShare mechanical checks run, then the full validator.
3. A fresh Claude Sonnet reviewer checks every changed or new record against AUTHORING-SPEC.md. That includes
   every record whose group changed.
4. One revision follows (round 6). Any record that still fails goes to the study owner, as in
   FINAL-PASS-AMENDMENT.md rule 5.
5. No other approved record is reopened.

Disclosure: written after the catalog was approved and the gaps were counted, before any confirmation run.
`human_review_complete` stays false.
