# Confirmation catalog (80 events)

Agent-authored, agent-reviewed source catalog for the P4 confirmation partition. Built 2026-10-13.
`human_review_complete` is false throughout. Agent review is not human review.

- Authors: Claude Haiku. Independent reviewers: Claude Sonnet. Every record was reviewed; there was no sampling.
- Rules: `ELECTIONS-AMENDMENT.md` sets the elections and the two-round review. `ROUND3-AMENDMENT.md` adds one extra
  pass, limited to records still UNRESOLVED after round 2. `FINAL-PASS-AMENDMENT.md` gives the 29 records left after
  that to a Claude Opus author and a fresh Sonnet reviewer. The 2 records that still failed were settled by the study
  owner (`reviews/owner-decisions.json`). `AUTHORING-SPEC.md` is the authoring and review spec.
- Sealed artifacts. Each carries a canonical `sha256` over its other fields.
  - `confirmation-grounded-declarations.json`: 1,258 source units and 1,001 merge groups, all APPROVED.
  - `confirmation-acquisition-qa.json`: 1,088 source-QA records with 2 closed-book forms each. All APPROVED.
  - `confirmation-evaluation-paraphrases.json`: 1,449 evaluation paraphrases. All APPROVED.
  - `confirmation-entity-ledger.json`: cross-event entity and fact links over all 100 events.
  - `confirmation-review-record.json`: counts, disclosures (the defective "default" label; the owner-decided paraphrase) and artifact hashes.
- `reviews/`: every per-event reviewer decision file, by channel and round.
- `input-packets-sha256.json`: hashes of the author packets. The packets themselves are not committed because they
  can be rebuilt (see `scripts/confirmation_catalog/`).

Rebuild and check on FarmShare: `scripts/confirmation_catalog/validate.py` runs the mechanical checks and
`scripts/confirmation_catalog/compile.py` merges the review outcomes and seals the artifacts.
