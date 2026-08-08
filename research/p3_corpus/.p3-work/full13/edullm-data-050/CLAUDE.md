# edullm-data — orientation for a fresh session

You are working on **`edullm-data`**: the publisher, validator, and reader for the eduLLM dataset
standard. Public repo: `github.com/edu-llm/edullm-data`. This file auto-loads on every launch; it
orients you, it is not the spec.

## Read these first, in order

1. **`HANDOFF.md`** — the living state of the project. Read it alone and you can continue with no
   other context: what is built, deployed, live, and what's next.
2. **`docs/ONBOARDING.md`** — the 2-minute mental model (the airlock, the bucket layout, the address
   shape `<family>/<name>/<version>/`, what the pipeline forces).
3. **`CONTRIBUTING.md`** — the golden rule and how to add a profile.
4. The full spec lives **outside this repo** at `../docs/dataset-creation/DATASET-STANDARD.md`
   (+ `-DIAGRAMS.md`, + the motivating `s3-dataset-audit-2026-07-28.md`). If launched with this repo
   as the root you won't have it in scope — `--add-dir ../docs` or read it on request. **If the code
   and the standard disagree, the standard wins and this package has a bug.**

## The one rule that governs every check

**Recompute, never trust.** Every validator check must recompute something from the bytes (a
`ContentLength`, a magic-byte sniff, a sampled decode) and compare it to what the manifest claims. A
check that only asserts a field is *present* is decoration — coding agents satisfy schemas
effortlessly and that is exactly how plausible garbage shipped before. See `CONTRIBUTING.md` "the
golden rule."

**Known gap — do not restate this rule as a payload re-hash.** `s3.hash_object` has exactly one
non-definition caller, `publish.py:280`, the PRODUCER. Gate A's per-entry loop
(`validate.py:399-431`) does `s3.head` for SIZE and then set-membership on the *declared* digest; it
never re-reads payload bytes. `fsck.py`'s docstring says so outright ("never a payload byte"). So a
manifest `sha256` is a producer assertion no gate falsifies. What the validator *does* recompute is
listed in `USAGE.md` "What gets rejected"; what `sha256` is actually for (content addressing +
the hash chain) and what defends integrity instead (the airlock's IAM Deny, S3 durability,
CRC64NVME) is in `docs/ONBOARDING.md`. Adding a real re-hash is an open decision, not a fact.

## Invariants you must not break

- **The airlock is an IAM Deny, not a convention.** Producers write ONLY to `s3://edullm-landing`
  (scratch inbox, 14-day expiry). The validator role (`<BATCH_JOB_ROLE>`, assumable only by
  `ecs-tasks` — no human/intern session can assume it) is the *only* principal that can `PutObject`
  to `s3://edullm-data`. After **any** live test that touched permissions, re-verify
  the Deny still fires (intern `PutObject` to `edullm-data` → `AccessDenied`, explicit deny) before
  you consider the task done.
- **Frozen means frozen.** Never edit a published `vN` in place to change data — publish `v2`. The one
  sanctioned in-place write is a **descriptive-keys-only** backfill (see README below), guarded by an
  assertion that `groups`/`manifest_sha256`/`inventory` stay byte-identical.
- **Token shards are `.u32le.bin`, never `.npy`.** The legacy `.npy` files were headerless raw uint32
  (the ".npy lie"); extension must match real bytes. `tokens × dtype_size == file bytes` exactly.
- **Every dataset carries a generated `README.md`.** It is a **derived** artifact (`readme.py`,
  `render_readme(dataset.json) → markdown`) written by `promote()`, and a **control file** (in
  `_CONTROL_BASENAMES`/`CONTROL_BASENAMES`) — never a manifest entry, never in the hash chain. It
  renders only from `dataset.json`, so it can't drift. Feed it via `publish(sources=/about=/notes=/
  limitations=/license=)`; absent sections are omitted, never faked. This applies to
  already-promoted datasets too — backfill the README in place, don't re-copy payload.

## AWS access in this workspace

All AWS goes through the **`sb-aws` MCP broker**, and it is **read-only by default** — you cannot (and
must not try to) write to `edullm-data` from a session; that's the airlock's whole point. Live
mutations happen only via validator-role Batch jobs. `iam:simulate-principal-policy` **lies** for the
intern role (11 known false denials) — smoke-test a permission live, never trust the simulator.

## Running on AWS Batch (hard-won gotchas)

If you ship a wheel-from-S3 Batch job (validate/publish/backfill), all four bite:
1. The Batch image has **no `aws` CLI** — download the wheel/driver/families with **boto3**.
2. ~~`families/` is **not in the wheel**~~ **FIXED** — `pyproject.toml` force-includes it, and
   `_resolve_families_dir()` finds it inside an installed package. Verified in a clean venv on
   `0.2.0`. The `EDULLM_FAMILIES_DIR` override in the publish driver is now redundant (harmless).
   Why it mattered: a missing families dir does not raise, it silently falls back to each
   profile's laxer constant — so it fails **only in production**, which is how the live corpus
   came to be validated at 50% EOS instead of the declared 5%.
3. **pip requires the PEP-427 wheel filename** — keep `edullm_data-<version>-py3-none-any.whl`
   end-to-end; a renamed `w.whl` is rejected. **The live job defs (`edullm-validator:2`,
   `edullm-fsck:2`) bootstrap `0.2.0` by exact filename**, so shipping a new wheel changes
   nothing until those are re-registered. Both EventBridge rules target the job def by
   unversioned name, so a new revision cuts over immediately — for better and worse.
4. Single-threaded publish **times out** on large corpora (the 218-shard/125GB olmo run hit the
   60-min job-def limit). Use `hash_workers`/`copy_workers` and pass
   `--timeout attemptDurationSeconds=7200`.

## Working style

Tests mirror source modules (`test_manifest.py` ↔ `manifest.py`); every profile ships a passing **and**
a failing fixture. Run `python -m pytest -q` (currently **380 passing**). Public repo — scrub internal
AWS account IDs to placeholders in anything committed (bucket names are functional constants, keep
them). Autonomous runs are fine; keep subagent fan-out ≤~16 in sequential waves and persist to disk
continuously (this machine has died mid-run). Ship changes via a branch + PR, not a direct push to
`main`.
