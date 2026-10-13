# P4 confirmation catalog: elections and review-process amendment (2026-10-13)

Status: chosen by the study owner on 2026-10-13, before any confirmation-catalog authoring.
No confirmation outcome or model output was observed. Governs only the 80-event confirmation catalog.
The adopted v1 source-QA protocol (dc5f4728…) still applies wherever this amendment is silent.

## Authoring elections (on-record preferences adopted)

| Election | Value | Meaning |
|---|---|---|
| multi_owner_rule | every_owner | An answer that traces to more than one source unit must be supported for each owning unit. |
| semantic_owner_positive_mix | combine_positive_same_slot | Several owners supporting the same slot count as supported; any caveat is recorded in a note. |
| OBJ_R5_COMPOUND_1 | adopt_N27 | Compound answers ("A, B and C") are judged by content; reordering does not fail. |
| OBJ_SURFACE_1 | nonblocking_L7 | A faithful alternate surface form that exact match cannot credit is disclosed, not blocking. |
| r9_input_variant_applicability | role_conditional | Each record names the packet it was authored from (packet sha256); no further variant receipts. |
| development_operation | retain_all_same_standards | Development records stay under the same standards; no change to the 20 development events. |
| D_and_compound_hazard_scope | population_wide | Hazard checks apply to all 80 events. |
| overlap rule | broad_LEAC | Cross-event answer overlaps are tolerated if the questions carry event-specific context. |

## Review-process amendment (replaces the v2 draft timing/receipt layer: H, RO, CR, LATENESS, DUE, CONVENTION, TERMINATION, RUN_PARTITION_SAFETY, replacement paths)

1. Every record has one author (Claude Haiku) and one independent reviewer (Claude Sonnet). No agent reviews its own records.
2. All records are reviewed; no sampling.
3. Review round 1: the reviewer gives each record APPROVE, REVISE (with a concrete reason) or REJECT.
4. The author revises the REVISE/REJECT records once. Review round 2 checks only those.
5. A record that is not APPROVE after round 2 is marked `UNRESOLVED`. It stays in the catalog with the dispute recorded and is disclosed. It is never dropped silently.
6. Mechanical checks (exact evidence quotes, coverage, no evaluation text in channel A, answer preserved in channel B) run on FarmShare after each round. A mechanical failure counts as REVISE.
7. `human_review_complete` stays false. Agent review is not human review.

Disclosure: this amendment departs from the unadopted v2 draft. It was written because the draft's
receipt-timing layer did not converge over rounds R15–R39 and does not change what is authored.
