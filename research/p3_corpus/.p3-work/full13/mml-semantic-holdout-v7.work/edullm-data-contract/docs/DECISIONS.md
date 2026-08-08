# Decisions

One entry per settled decision. Sourced from
[`../../docs/dataset-creation/DATASET-STANDARD.md`](../../docs/dataset-creation/DATASET-STANDARD.md);
see that document for the full argument. Format: Decision / Why / Rejected alternative.

---

### Airlock over sentinel-gating

**Decision:** Two buckets. Producers can write only to `s3://edullm-landing`; only the validator role
can write `s3://edullm-data`. Unvalidated bytes never enter the published namespace.

**Why:** With sentinel-gating (write to the read bucket, then a CI job writes `_SUCCESS`), bad bytes are
already in the published namespace before validation runs — the sentinel only declines to bless them.
Under create-only writes they can't be overwritten either, so they sit there forever, visible to any
glob.

**Rejected:** CI writes a `_SUCCESS` sentinel after publishing directly to the read bucket.

---

### Reuse `<BATCH_JOB_ROLE>` instead of creating a validator role

**Decision:** The validator runs as the existing `<BATCH_JOB_ROLE>` role.

**Why:** `iam:CreateRole` is explicitly denied by the `<PERMISSION_BOUNDARY>` permissions boundary, so a
purpose-built role can't be created from an intern session. This role's trust policy already allows
only `ecs-tasks.amazonaws.com` — no human or intern session can assume it — and `iam:PutRolePolicy` on
it is allowed, so its S3 grants can be attached without creating anything.

**Rejected:** Request/create a dedicated `DatasetValidator` role.

---

### AWS Batch only, no Lambda tier

**Decision:** One execution route for every dataset size, on AWS Batch. No size-based dispatch to
Lambda in v1.

**Why:** A single path is easier to reason about, and Lambda's 15-minute ceiling can't decode-test a
633 GB corpus anyway. Small datasets pay a queue wait of a minute or two, which nobody will notice.

**Rejected:** Lambda for small datasets, Batch for large, with a size threshold deciding which.

---

### Reuse job definition `<JOB_DEFINITION>` with command overrides

**Decision:** Submit validator jobs against the existing `<JOB_DEFINITION>` job
definition with a container command override, instead of registering a new one.

**Why:** `batch:RegisterJobDefinition` is assume-denied. It isn't needed: the existing definition's
default command is already a placeholder meant to be overridden, and its image is digest-pinned, so a
command override is sufficient to run the validator (fetched at container start for v1).

**Rejected:** `batch:RegisterJobDefinition` a new definition for the validator.

---

### No `-of-N` in shard names

**Decision:** Shards are named `<split>-<NNNNN>.<ext>`, never `<split>-<i>-of-<N>.<ext>`.

**Why:** The total shard count is unknowable at write time — parallel workers tokenize independently,
and the surviving count depends on filtering that hasn't run yet. Path-set equality against the
manifest (reporting `missing=` / `extra=`) proves completeness more strongly than a count baked into
each filename.

**Rejected:** Encode total shard count in each shard's filename.

---

### Profile on the group, not the dataset

**Decision:** `profile` is a property of a payload group (`tokens/`, `sidecars/`, …), not of
`dataset.json` as a whole.

**Why:** Real datasets contain structurally different payload groups in one release (packed tokens plus
a gzipped-CSV sidecar, for example). A dataset-level `profile` would force either lying about half the
files or splitting into two datasets that must never drift apart.

**Rejected:** One `profile` field on the dataset, covering every object underneath it.

---

### `.u32le.bin`, never `.npy`

**Decision:** Packed uint32 token shards use the self-describing extension `.u32le.bin`. The `.npy`
extension is never used for headerless raw arrays.

**Why:** OLMo-core reads token files via `np.memmap(path, mode="r", dtype=dtype)` from byte 0 and
derives the token count from raw file size. A real NumPy header would corrupt both the leading tokens
(read as data) and the size-derived count.

**Rejected:** Use `.npy` because the payload is conceptually a NumPy array.

---

### No Object Lock

**Decision:** Immutability comes from airlock + versioning + deny-delete. S3 Object Lock is not used
anywhere in the design.

**Why:** Object Lock protects a *version*, not a *path* — new versions and delete markers are still
permitted, and delete markers are explicitly not WORM-protected, so `delete → recreate` walks around
it. Lifecycle can't delete a locked version, so long retention means unbounded growth nobody can stop.
GOVERNANCE mode is bypassable, and the S3 console sends the bypass header automatically. It can never
be disabled once enabled on a bucket.

**Rejected:** Enable S3 Object Lock (WORM) on `edullm-data` for tamper-proofing.

---

### PII out of scope

**Decision:** No `pii` field, no PII scanner, no PII-based routing anywhere in the standard.

**Why:** Every dataset under this standard is non-personal by assumption — public corpora, licensed
third-party data, synthetic generations, or model outputs. If that assumption ever stops holding (first-
party student records, parent communications), this standard doesn't cover it and needs a revision
first.

**Rejected:** Build a `pii` field / scanner / routing logic now, ahead of need.

---

### Seeded-random-offset decode smoke test over head-only sampling

**Decision:** The decode smoke test samples ~64 KB per shard at offsets derived from
`sha256(dataset_id|version|shard_path)`, not from the head of the file.

**Why:** A zero-filled or truncated tail is a real failure mode — a crashed writer leaves a
correctly-sized file whose head is perfectly valid. Head-only sampling misses it entirely. Seeded
offsets are still deterministic (any auditor can re-run the identical sample) while covering the whole
shard.

**Rejected:** Sample only the first N bytes of each shard.

---

### One bucket for data, with class as a field — not a bucket per lifecycle class

**Decision:** `edullm-data` is a single bucket. Any lifecycle class is a field, never encoded in the
bucket name or path.

**Why:** Promotion (validator → published copy) is a server-side copy keyed by hash; encoding lifecycle
class in the bucket name or path would mean promotion changes the path and invalidates the hashes that
gate that same promotion.

**Rejected:** Separate buckets per lifecycle/storage class, with promotion moving objects between them.

---

### `experimental` capped by quota, not approval

**Decision:** `experimental/v1` is gated by a hard quota — max 2 live datasets per family — not by an
approval workflow.

**Why:** Quotas don't erode the way approvals do: the third publish mechanically fails and names the
two datasets blocking it, with no meeting to schedule and no discretion to route around under time
pressure.

**Rejected:** Require sign-off from an approver before publishing an `experimental` dataset.
