# Contributing

## The golden rule

**Every validator check must recompute something. It must never merely assert that a field is
present.**

Schema-shape validation invites plausible garbage, especially from coding agents, which are excellent
at satisfying schemas but not at telling you whether the bytes behind the schema are real. The only
check in this account's history that ever rejected bad work recomputed a hash — everything that only
read metadata passed straight through.

If your check reads `manifest["count"]["value"]` and stops, it is decoration. If it reads the bytes
(or the object's `ContentLength`, or a magic-byte sniff) and compares against what the manifest claims,
it is a check.

## Adding a profile

Adding a profile is the **expected** path when nothing fits — not an exception to request permission
for. The registry is expected to keep growing. Reach for `experimental/v1` only when you must ship
*today* and the shape isn't known yet; it's capped at 2 live datasets per family, so it isn't meant to
be where a recurring shape lives.

A profile is four small things:

1. **A registry entry** — `profiles/<name>_v1.py`, exporting `REQUIRED_FIELDS` and `CHECKS`.
2. **A JSON Schema fragment** for the fields it adds to a group's metadata.
3. **One or more check functions**, each with signature `(group, manifest, s3) -> list[Violation]`.
   Every check must recompute something (see the golden rule above) — a check that only reads metadata
   adds ceremony without adding safety.
4. **Two fixtures** in `tests/fixtures/` — one tiny passing example, one deliberately broken, so the
   check is proven to actually fire and not just proven to exist.

Add fields only if a check consumes them — a required field no check reads is decoration. Prefer
arithmetic identities over descriptive fields: `count × dtype_size == bytes` is worth more than five
schema fields describing the same shard.

**PR review criteria:** does each check recompute something, and does the broken fixture fail against
it? If either answer is no, the profile isn't done.

### Versioning

Profiles start at `/v1`. Once published, a profile version is never mutated — add `/v2` and leave `/v1`
valid, because artifacts are validated against whichever version they pinned at publish time, never
against "latest." A field added in `v2` must not retroactively invalidate every `v1` dataset.

## Test conventions

- pytest.
- Test modules mirror source modules (`test_manifest.py` for `manifest.py`, etc.).
- Every profile ships with a passing fixture and a failing fixture; a profile PR without both is
  incomplete.

## AWS permission checks

**Do not trust `iam:simulate-principal-policy` for the intern role in this account.** It has reported
`explicitDeny` for actions that actually work — ten of them at last count (`CreateBucket`,
`PutBucketPolicy`, `PutRule`, `SubmitJob`, `DescribeJobs`, `PutTargets`, `PutBucketNotification`, and
both ECR actions). If you need to know whether an action is permitted, smoke-test it live (create a
probe resource, exercise the action, tear it down) rather than asking the simulator.
