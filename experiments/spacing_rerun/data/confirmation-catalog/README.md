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

## Audit conformance and repair (2026-10-15)

The confirmation code wants its own audit wrapper, and the approved catalog didn't meet three of its rules.
`AUDIT-CONFORMANCE-AMENDMENT.md`, chosen by the study owner, handles them:

- It relaxes the validator in three places. Reviewers may have seen author rationales. Preserved records may lack
  evidence. Channel B may declare the narrower inputs it actually had. The code pins the amendment hash.
- It lists the mechanical conversions.
- It defines a repair round for 63 structural records.

The repair round ran as follows:

- Round 5: Haiku authors fixed the records in 33 events, and fresh Sonnet reviewers then checked every record in
  scope. 18 records did not pass.
- Round 6: one revision by Haiku, reviewed by a fresh Sonnet. 14 of the 18 were approved.
- The last 4 went to the study owner (`reviews/owner-decisions-r6.json`).

Where the files are:

- `REPAIR-SPEC.md` and `REPAIR-REVIEW-SPEC.md` are the author and reviewer instructions.
- `repair/packets/` holds the per-event inputs.
- `repair/authored/` holds the patches.
- `reviews/A/*.r5.json` and `reviews/A/*.r6.json` hold the decisions.

`scripts/confirmation_catalog/build_audit.py` builds the wrapper from these files, and the confirmation validator
checks it against the pinned parquet. `confirmation-audit-record.json` records the hashes of the result and of the
24 prepared replicate manifests. The 15 MB wrapper itself is kept on FarmShare, at the path given in that record.
