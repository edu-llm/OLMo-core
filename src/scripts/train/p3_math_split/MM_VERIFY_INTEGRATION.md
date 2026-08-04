# Metamath evaluator integration

`mm_verify.verify_proof()` now returns a tri-state `ProofResult.status`:
`valid`, `invalid`, or `unknown`. The evaluator must report all three counts and
must use only `valid + invalid` as the decided denominator. `unknown` is not a
failed proof.

Until the evaluator passes the dedicated verifier, pinned-real, and official-oracle
tests, remove or disable the old `metamath_valid` binary metric. Do not map
`ProofResult.valid is None` to `False`; the compatibility property deliberately
returns `None` for `unknown`.

Each call must provide:

- the bare source `target_label` from `row["theorem"]` (for example `pm4.39`
  from `set:pm4.39`);
- the exact visible global `fact_block`;
- only the theorem-local `$e` entries in `local_assumptions`.

Missing target context, bounded substitution/syntax search, and unsupported official
trace conversion are explicit `unknown` results with machine-readable
`reason_code` values. Persist status and reason counts separately. The official
oracle in `mm_official.py` also returns tri-state results; it does not expose a
`metamath_valid` boolean.
