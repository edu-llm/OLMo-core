# Reading eduLLM datasets from a platform training run

**Audience:** whoever owns [`edu-llm/platform`](https://github.com/edu-llm/platform). This is a
change request against that repository, written from the dataset side. It stands alone — you do
not need any prior conversation, and nothing in `edullm-data` needs to change for any of it.

**Every anchor below was opened and read** at `edu-llm/platform` HEAD, and every live number was
read back from the account rather than from a template. Where a figure in circulation disagrees
with what the repository or the API actually says, the disagreement is named in place.

---

> ## ⚠️ READ THIS FIRST — corrections dated 2026-07-30
>
> This document was written before the current corpus was published and before the reader could
> build mixtures. Re-audited against live AWS and the platform repo; the four requested changes
> still stand, but these specifics have moved:
>
> 1. **The corpus is different.** `pretrain/olmo-mix-1124-31b/v1` was **deleted**. The live one is
>    **`pretrain/olmo-150b-dolma2/v1`** — 6,911 shards, **157,467,202,883 tokens**, 6,851 train +
>    60 val, laid out `tokens/<source>/<domain>/`. Every occurrence of the old id below, including
>    the proposed registry entry, points at objects that no longer exist.
> 2. **Change 3 (mixture) is much smaller than described.** `build_mixture()` now resolves a
>    weighted mixture from `(dataset, version, sources, ratios, total, seed)` — measured at
>    **306 bytes**, comfortably inside the 8192-byte ContainerOverrides cap. Only a *resolved* URI
>    list blows it (53 KB for a 528-shard mix; 753 KB for all 6,911). So the submission form needs
>    a small spec, not a path list. **Include `seed` and `group`** — the field list at "Change 3"
>    omits both, and without `seed` a mixture is not reproducible.
> 3. **Change 4 is two independent things, not one.** The 3600 s timeout is **not** a blocker:
>    `execution.py` sends `AttemptDurationSeconds` on every submit, derived from the form's
>    `maximum_runtime_hours`, which overrides the job-definition default. A submitter can raise it
>    with no repo change (>12 h reclassifies as EXCEPTION, which is approvable — it is not in
>    `denied_outright`). **GPU capacity is the real constraint**: the only provisioned GPU is a
>    single A10G, and `gpu-8xa100` raises `UnprovisionedComputeProfileError` before any queue is
>    chosen — a hard raise, not a policy vote. Good news: the permission boundary no longer blocks
>    `p4d`/`p5`/`g5` (that Deny is gone as of boundary v5), so provisioning is now a budget call.
> 4. **Do not use `SourceMixtureDatasetConfig`.** It routes into `NumpyFSLDatasetMixture`, which
>    derives a `.csv.gz` sidecar name from each shard name — and `.replace(".npy", ".csv.gz")` is a
>    **no-op** on `.u32le.bin`, so it hands raw uint32 to gzip and dies. Verified by execution
>    (`BadGzipFile`). Those sidecars were deliberately not migrated. Use plain
>    `NumpyFSLDatasetConfig` with an explicit dtype; see `docs/CONSUMER-CONTRACT.md` §6.
> 5. **Two blockers this document never listed.** (a) `config/datasets.yaml` has one entry,
>    `dolma-2026-07`, naming nothing in `edullm-data`; an unregistered dataset is
>    **denied outright**, not merely awkward. (b) `config/repositories.yaml` pins
>    `dockerfile_path: .edullm/Dockerfile` for OLMo-core and **that file does not exist**, nor does
>    any image install `edullm_data`/`boto3`. No image, no run.
>
> Change 1 (the IAM grant) is unaffected and remains the cleanest single fix: one statement on
> `batch-gpu-roles.yaml`, no bucket-policy edit — the `edullm-data` Deny covers writes only.
> While you are in that file, note the deployed policy scopes to `teams/*/runs/*` while the
> template says `teams/platform/runs/*`; the committed isolation argument does not match what is
> deployed.

---

## The goal, stated up front

Training runs are launched through `.github/workflows/submit-run.yml` — a `workflow_dispatch`
form → the compile job → an environment approval gate → admission (Step Functions) → AWS Batch.
That path works; four GPU runs have gone down it.

The eduLLM datasets live at `s3://edullm-data/<family>/<name>/<version>/`. Two are published
today, both frozen and validator-sealed:

```
s3://edullm-data/pretrain/olmo-150b-dolma2/v1/     157.467B tokens, 6,911 shards, 629,868,811,532 B
s3://edullm-data/tokenizer/dolma2-bpe/v1/          the tokenizer those tokens were made with
```

**Today a training job on the platform cannot read either of them.** Not "has no convenient
helper" — cannot. The bytes are unreachable from inside the container, and the run's own lineage
record cannot say which dataset it read even if they were.

Four changes close that. They are independent enough to land separately, and they are not equally
hard: two are blockers with no container-side workaround, and two are things you *can* route
around today at the cost of a lineage record that lies.

| # | Change | Where | Workaround? |
|---|---|---|---|
| 1 | `s3:GetObject` + `s3:ListBucket` on `edullm-data` | `infra/iam/batch-gpu-roles.yaml` | **None.** AWS enforces it outside the container. |
| 2 | Bind a `release_id` to a `uri` + `manifest_sha256` | `config/datasets.yaml` + 3 contracts | Possible — pass the URI in `command`. The lineage record then lies. |
| 3 | A real training workload profile + datamix fields | `config/workload-catalog.yaml` | Possible — encode the mix in `command`. Same lie, plus an 8 KB ceiling. |
| 4 | Provision real GPU + raise the 3600 s timeout | `infra/batch-compute-gpu.yaml` | **None.** A 14-hour run cannot fit in a 1-hour attempt. |

Changes 1 and 4 are the ones that stop a run existing at all. 2 and 3 are the ones that decide
whether, six months from now, anybody can tell what a checkpoint was trained on.

---

## Change 1 — the IAM grant (the hard blocker)

### What is there now

`infra/iam/batch-gpu-roles.yaml:150-178` is the whole of `BatchGpuWorkloadRole`'s S3 access, and
every statement in it names one bucket and one prefix:

- `s3:PutObject` / `s3:AbortMultipartUpload` on `sbsandbox-intern-edullm-outputs/teams/platform/runs/*`
- `s3:GetObject` on the **same** ARN — granted, per the comment at lines 156-163, because a
  training job reads back what it wrote (checkpoint success markers, resume)
- `s3:ListBucket` on the outputs bucket, conditioned to `teams/platform/runs/*`

The comment at line 162-163 states the consequence in as many words:

> It cannot reach the dataset bucket, the artifacts bucket, or the lineage store.

That is accurate and it is the blocker. The CPU sibling, `infra/iam/batch-roles.yaml:179-183`,
already anticipates exactly this request:

> No `s3:GetObject` either, so this role cannot read the dataset bucket the dolma fixture's
> command names. That is correct for the workload Phase 3 runs — a CPU smoke against a published
> image that reads nothing — and **it is the first grant a real tokenizer job will need. It
> arrives as an edit here with a reason, not as a prefix somebody widened.**

This document is that reason. The grant belongs on the **GPU** workload role, because that is the
role a training job runs under (`config/execution-targets.yaml` binds `gpu-1xa10g` →
`sbsandbox-intern-edullm-batch-gpu-workload`). The CPU role needs it only when a tokenizer job
becomes real.

### Why there is no workaround

This is worth being precise about, because it is the one item on the list where no amount of
cleverness inside the container helps.

An `s3:GetObject` denial is evaluated by S3, in the AWS control plane, against the task's IAM
role — **before any byte is returned to the container**. The container never sees the request
succeed and fail; it sees an `AccessDenied` that was decided somewhere it cannot reach. So:

- Retrying does not help. The denial is deterministic, not transient.
- A different SDK, a different endpoint, a presigner, `boto3` vs `aws s3 cp` — all the same
  identity, all the same answer.
- The container cannot assume its way out. `BatchGpuWorkloadRole` holds no `sts:AssumeRole`, and
  `InternSandboxBoundary` (read back live, `v5`) would still bind whatever it reached.
- Baking the data into the image is not an alternative at this scale. The published corpus is
  125 GB against a 4.29 GB training image.

The only thing that changes the answer is an edit to the role. Hence "hard blocker".

### Two things that are *not* in the way (both verified live)

Worth stating so you don't budget for work that isn't needed:

**The `edullm-data` bucket policy does not deny reads.** Read back live, `Id:
edullm-data-airlock-v1`. Its one `Deny` statement (`Sid: OnlyValidatorWrites`) enumerates
`s3:PutObject`, `s3:DeleteObject`, `s3:DeleteObjectVersion`, `s3:PutObjectTagging`,
`s3:AbortMultipartUpload` — **write actions only**. `s3:GetObject` and `s3:ListBucket` appear
nowhere in it. The bucket and the role are in the same account (`056956104102`), so a same-account
IAM `Allow` is sufficient on its own; no bucket-policy change is needed and none is being asked
for. **The ask is one-sided.**

**`InternSandboxBoundary` does not block it.** Read back live at `v5`: the boundary opens with
`AdminCeiling` (`Allow` `*` on `*`) and then denies specific escapes — cross-account
`sts:AssumeRole`, policy-version escalation, boundary tampering, reserved-capacity purchases, and
a region lock to `us-east-1`/`us-east-2`. `edullm-data` is in `us-east-1`. No statement in the
boundary mentions S3 buckets or bucket-name prefixes, so the grant is permitted by the boundary
as written.

### The YAML to add

Into `BatchGpuWorkloadRole`'s `write-and-resume-one-teams-outputs` policy — or, if you'd rather
keep the policy name honest, as a second `Policies` entry named something like
`read-the-published-dataset-library`. Both statements are needed; neither is sufficient alone.

```yaml
              # READ-ONLY, AND THE BUCKET IS OUTSIDE THE SANDBOX NAMESPACE ON PURPOSE.
              #
              # s3://edullm-data is the published dataset library. It is not
              # sbsandbox-intern-* because it is not scratch: an airlock bucket policy
              # (Sid OnlyValidatorWrites) denies PutObject/Delete to every principal except
              # the validator role, so nothing in it can be modified by anyone -- including
              # by this role, which is why a read grant here cannot widen into a write one.
              #
              # No s3:PutObject, no s3:DeleteObject, and adding either would be refused by
              # that bucket policy anyway. The grant is one direction by construction.
              - Effect: Allow
                Action: s3:GetObject
                Resource:
                  Fn::Sub: arn:${AWS::Partition}:s3:::edullm-data/*
              # Listing is a bucket-level action and cannot be scoped by an object ARN.
              #
              # DELIBERATELY WITHOUT AN s3:prefix CONDITION, unlike the outputs-bucket
              # ListBucket above. The reader resolves a dataset in two steps: it lists
              # _catalog/<family>/<name>/ to find the latest version, then lists the
              # dataset's own prefix for shards. A condition scoped to one of those breaks
              # the other, and the failure is a dataset that reads as absent rather than as
              # denied. There is nothing to protect by narrowing it: every object in this
              # bucket is published, frozen, and readable by design.
              - Effect: Allow
                Action: s3:ListBucket
                Resource:
                  Fn::Sub: arn:${AWS::Partition}:s3:::edullm-data
```

Note `arn:aws:s3:::edullm-data` has **no account or region segment** — S3 bucket ARNs never do.
That is correct as written, not an omission.

### Why `ListBucket` and not just `GetObject`

Because the reader in `edullm_data.read` cannot resolve a dataset without listing. Two calls
need it:

- `resolve_latest()` lists `_catalog/<dataset_id>/` to find the newest `vN` — otherwise a caller
  has to hardcode a version, which is the thing versioning exists to avoid.
- `dataset_paths()` resolves a split by globbing shard names, and `head`s the `_VALIDATED.json`
  seal. A prefix that carries no seal is **refused** by the reader — belt-and-suspenders behind
  the airlock, and the check a training job wants to keep.

A `GetObject`-only grant produces a job that can read bytes it cannot find.

### After you deploy it

`infra/README.md` is emphatic that IAM is laptop-only and is not redeployed by CI, and the drift
capture exists because of it. Two things worth doing in the same sitting:

1. Re-run whatever refreshes `fixtures/evidence/phase-3/roles/` for the GPU trio, since the
   committed shape of `sbsandbox-intern-edullm-batch-gpu-workload` changes.
2. Smoke-test the grant live rather than trusting `iam:simulate-principal-policy`. On the dataset
   side we have measured the simulator returning **false denials** for the intern role (11 of
   them). It is not evidence. A one-line `GetObject` from a Batch task under the real role is.

---

## Change 2 — bind a dataset id to a location

### What a container is told today

`config/datasets.yaml` is six lines of comment and two of content:

```yaml
schema_version: 1
releases:
  - release_id: dolma-2026-07
```

and its own comment at lines 8-10 says why:

> Entries carry an identifier and nothing else. Binding each identifier to a full
> `DatasetRelease` — checksums, S3 VersionIds, schema, lineage, licence, classification and
> access policy — is the dataset-breadth phase's work, not admission's.

That was the right call for admission, whose only question is "is this release registered". It
stops being sufficient the moment a container has to *open* the dataset. Follow the value through:

| Step | Anchor | What it carries |
|---|---|---|
| The form offers one option | `.github/workflows/submit-run.yml:38-43` | `dolma-2026-07` |
| The submission holds a name | `src/edullm_platform/submission.py:101` | `dataset_release: str = Field(min_length=1)` |
| The manifest holds a name | `src/edullm_platform/contracts/manifest.py:35` | same field, same type |
| The registry validates a name | `src/edullm_platform/contracts/dataset_registry.py:33-34` | `release_id` and nothing else |
| Batch is told a name | `src/edullm_platform/execution.py:145` | `EDULLM_DATASET_RELEASE` |

**So a container receives a name with no address.** `dolma-2026-07` is not an S3 URI, not a
prefix, and not resolvable from anything else in the environment (`execution.py:143-174` sets
`EDULLM_RUN_ID`, `EDULLM_TEAM`, `EDULLM_DATASET_RELEASE`, `EDULLM_COMMIT_SHA`,
`EDULLM_OUTPUT_PREFIX`, `EDULLM_WANDB_PROJECT` — nothing that locates a dataset). The container
must either hardcode a mapping or take the URI from `command`.

### The cost of not fixing it

There *is* a workaround: put `s3://edullm-data/pretrain/olmo-mix-1124-31b/v1/` in the `command`
argv. It works today with Change 1 in place. Here is what it costs.

The approved manifest, the lineage record, and the W&B run all say `dataset_release:
dolma-2026-07`. The job read `pretrain/olmo-mix-1124-31b/v1`. Nothing in the system compares the
two, because nothing in the system knows they are meant to be related. The failure is silent and
permanent:

- The intent and decision records in the lineage store are **immutable** (`infra/lineage-bucket.yaml`,
  write-once, Object Lock). A wrong dataset attribution cannot be corrected later — only appended to.
- A checkpoint in `teams/platform/runs/<run_id>/` is attributable to a dataset only through that
  record. If the record is wrong, the checkpoint's training data is unknown, and an unknown
  training set is the one thing that makes a model result unusable for a paper or a comparison.
- This is precisely the class of defect the platform's own comments are written to prevent —
  `execution.py:157-166` refuses to let a submitter put the W&B project in `command` for the same
  reason: *"a submitter who wrote a different project into their command would be attributing
  their spend somewhere the decision record does not say, and nothing downstream would notice."*
  A dataset URI in `command` is that sentence with one noun changed.

`dolma-2026-07` also names nothing that exists in `edullm-data`. The registered id and the
published library have never been reconciled; whatever gets bound, that reconciliation is part of
the work.

### What to add

Extend `RegisteredDatasetRelease` (`src/edullm_platform/contracts/dataset_registry.py:33-34`)
with the two fields that make a name resolvable, and emit them beside
`EDULLM_DATASET_RELEASE`:

```yaml
schema_version: 1
releases:
  # An identifier, and now the two things that make it resolvable: where the dataset is,
  # and what its content must hash to. Everything else about a release -- schema, licence,
  # classification, per-object version ids -- stays out, per the note above; these two are
  # here because without them a container is told a name it cannot open.
  - release_id: olmo-mix-1124-31b-v1
    uri: "s3://edullm-data/pretrain/olmo-mix-1124-31b/v1/"
    # The dataset's group manifest digest, read from the published dataset.json:
    # groups[0].manifest_sha256. It is a digest over the manifest that lists every shard
    # with its size and checksum, so pinning it pins the whole payload transitively.
    # A dataset is frozen once published -- v1 is never edited, a change means v2 -- so
    # this value is stable for the life of the release_id.
    manifest_sha256: "f05702fae463eccb75c220f905e75564b8501cb2350f12d52f7ca41568059a84"
```

That digest is real: read live from `s3://edullm-data/pretrain/olmo-mix-1124-31b/v1/dataset.json`.
Passing it through to the container is what makes the pin worth having — the reader can verify the
seal it finds against the value the approver signed off on, which turns "the dataset moved" from
an invisible event into a refusal. (The dataset side's rule is *recompute, never trust*; a digest
that is carried but never checked is decoration.)

Then in `src/edullm_platform/execution.py`, beside line 145:

```python
                {"Name": "EDULLM_DATASET_RELEASE", "Value": manifest.dataset_release},
                {"Name": "EDULLM_DATASET_URI", "Value": manifest.dataset_uri},
                {"Name": "EDULLM_DATASET_MANIFEST_SHA256", "Value": manifest.dataset_manifest_sha256},
```

Three env vars, roughly 130 bytes serialized. Well inside the 8192-byte ceiling — see Change 3.

### Two knock-ons you must handle (both verified)

**(a) The URI cannot currently be expressed.** `SandboxS3Prefix`
(`src/edullm_platform/contracts/lifecycle.py:28`) is an alias for
`CHECKPOINT_DESTINATION_PREFIX_PATTERN` (`src/edullm_platform/contracts/workload.py:17-19`):

```python
CHECKPOINT_DESTINATION_PREFIX_PATTERN = (
    rf"^s3://{SANDBOX_BUCKET_PREFIX}[a-z0-9](?:[a-z0-9.-]{{0,44}}[a-z0-9])?/.+/$"
)
```

with `SANDBOX_BUCKET_PREFIX = "sbsandbox-intern-"` at `src/edullm_platform/contracts/base.py:23`.
`DatasetRelease.uri` is typed `SandboxS3Prefix` at `src/edullm_platform/contracts/dataset.py:49`.
So **`s3://edullm-data/...` fails validation** — the bucket name does not start with
`sbsandbox-intern-`.

Do not widen `SandboxS3Prefix`. It is load-bearing for checkpoint destinations and result
manifests (`contracts/results.py:66` and `:102`), where "inside the sandbox namespace" is a real
invariant about where a run may *write*. Introduce a sibling instead — a distinct alias for a
read-only dataset location, so the two constraints stay separable:

```python
# A published dataset location. Separate from SandboxS3Prefix rather than a widening of it:
# that type constrains where a run may WRITE, and every bucket it admits is scratch that
# this account owns. This one constrains where a run may READ, and the library it points at
# is deliberately outside the sandbox namespace because it is not scratch -- an airlock
# bucket policy makes it write-once, so it is the one S3 location a run cannot corrupt.
# Widening SandboxS3Prefix to admit it would also admit it as a checkpoint destination.
PUBLISHED_DATASET_PREFIX_PATTERN = r"^s3://edullm-data/[a-z0-9-]+/[a-z0-9-]+/v[0-9]+/$"
PublishedDatasetPrefix = Annotated[str, Field(pattern=PUBLISHED_DATASET_PREFIX_PATTERN)]
```

That pattern is the eduLLM address shape, and it is enforced on our side too: `<family>/<name>/
<version>/` where family is one of seven fixed values (`pretrain`, `curriculum`, `sft`, `eval`,
`probe`, `vendor`, `tokenizer`), name is kebab-case 2-5 words, version is `v1`, `v2`, … assigned
automatically and immutable once published. If you want the family list closed in the regex rather
than `[a-z0-9-]+`, that is the list.

**(b) A test asserts today's behaviour and will fail.** `tests/test_dataset.py:28-37` defines
`OUTSIDE_SANDBOX_PREFIXES`, whose **first entry is `"s3://edullm-datasets/dolma/2026-07/"`**, and
`tests/test_dataset.py:209-213` asserts every one of them is rejected:

```python
@pytest.mark.parametrize("uri", OUTSIDE_SANDBOX_PREFIXES)
def test_dataset_location_outside_the_sandbox_bucket_namespace_is_rejected(uri: str) -> None:
```

This test is not wrong and should not simply be deleted. It is pinning a real property — that a
`DatasetRelease.uri` cannot point anywhere arbitrary. Keep it, and add a positive case for the
published-library shape, so the suite ends up asserting both "the sandbox namespace is required
for writes" and "the dataset library is the one permitted read location". Deleting the
parametrize entry without replacing the property is how the constraint quietly stops existing.

Also note the six *other* entries in that tuple are shape violations independent of the bucket
name — a missing trailing slash, an empty key, an `https://` form, a mixed-case bucket. A new
pattern needs to keep refusing those. The one above does: it requires the trailing slash and the
three path segments.

---

## Change 3 — a training workload profile, and fields for the datamix

### What the catalog offers

`config/workload-catalog.yaml` registers four workloads and **all four are smoke tests**:

| Workload | Compute profile | Runtime | Attempts | Checkpoint |
|---|---|---|---|---|
| `olmo-core-cpu-smoke` (`:121`) | `cpu-32vcpu` | 1 h | 1 | `null` |
| `dolma-tokenize-smoke` (`:130`) | `cpu-32vcpu` | 2 h | 1 | `null` |
| `olmo-core-gpu-smoke` (`:153`) | `gpu-1xa10g` | 1 h | 1 | `null` |
| `olmo-core-train-smoke` (`:159`) | `gpu-4xa10g` | 1 h | 1 | 30 min |

`olmo-core-train-smoke` is the closest thing to a training profile and it is still a smoke: one
hour, one attempt, and it names `gpu-4xa10g`, which is `provisioned: false` (line 67) and
therefore refused by `resolve_compute_profile_for_execution` with
`UnprovisionedComputeProfileError`. The workflow's `workload_profile` dropdown
(`.github/workflows/submit-run.yml:31-37`) offers only the two `-smoke` entries anyway.

There is also no way to express a data mixture. A submitter can say *which release*, and cannot
say *which parts of it, in what proportion, for how many tokens*. For a mixture corpus that
distinction is the experiment. Our 150B corpus is a **6-source mixture** where the ratios are the
point — the upstream config feeds OLMo-core's `SourceMixtureDatasetConfig` with a per-source
`target_ratio`, and flattening it destroys the dataset's reason to exist:

```
all-dressed-snazzy2  119.3B      finemath-3plus   4.06B
s2pdf-redacted        19.8B      arxiv            1.25B
stack-edu             11.1B      wikipedia        0.064B
```

Published datasets expose that structure as **groups** (one per source) each with `train`/`val`
partitions, so "these sources at these ratios" is expressible against the address shape — but
only if the submission has somewhere to say it.

### The constraint that shapes the whole design

`src/edullm_platform/execution.py:198-200`:

```python
#: What Batch will accept as one job's ``containerOverrides``, serialized. An AWS service
#: limit rather than a choice of ours, and not adjustable.
MAXIMUM_CONTAINER_OVERRIDES_BYTES: Final = 8192
```

`refuse_an_oversized_override` (`:225`) measures the **serialized** block — command *and*
environment, keys, JSON punctuation and all. The docstring at `:206-215` records what it cost to
learn: a 9,230-byte training program was compiled, validated, dispatched, **approved at the
environment gate**, admitted, submitted, and *then* refused by Batch with a message that named
neither the command nor the field the submitter controlled.

So the design rule is not negotiable:

> **A submission must pass source names or globs. Never a resolved shard list.**

The arithmetic, for the 150B corpus (6,921 shards, measured; destination URIs like
`s3://edullm-data/pretrain/olmo-150b-dolma2/v1/all-dressed-snazzy2/train-00000.u32le.bin`, 87
characters):

| | bytes |
|---|---|
| 6,921 URIs, raw | ~602,000 |
| the same, JSON-quoted and comma-separated | ~623,000 |
| the Batch ceiling | 8,192 |
| **over budget by** | **~76x** |

A resolved list is not "tight" or "risky". It is off by two orders of magnitude, and it fails
*after* a human has spent their attention approving it. Even one source's shards
(`all-dressed-snazzy2`, the largest) blows the limit many times over.

The good news: source *names* are tiny. All six names plus ratios is a few hundred bytes. The
container expands names → shard URIs at runtime by listing the prefix, which is exactly what
`ListBucket` in Change 1 is for. This is the same "told, not computed" reasoning
`execution.py:147-156` applies to `EDULLM_OUTPUT_PREFIX`, run the other way: the platform tells
the container *which* sources, and the container resolves *which files*, because the file list is
a property of the frozen dataset rather than of the decision.

### What to add

A real training profile beside the smokes. It depends on Change 4 for the compute profile — with
`gpu-8xa100` still `provisioned: false`, this entry registers and prices correctly but is refused
at resolution, which is the honest "ask for something else" the resolver is designed to give.

```yaml
  # THE FIRST ENTRY THAT IS NOT A SMOKE, AND THE BOUNDS ARE THE DIFFERENCE.
  #
  # maximum_runtime_hours is 20 rather than 1 because a 370M-parameter model on 20B tokens
  # is ~4.44e19 FLOPs (6ND) and does not finish in an hour on any shape in this catalog.
  # See docs/PLATFORM-INTEGRATION.md for the arithmetic and its assumptions. The figure is
  # a bound the approver is shown, not an estimate of the run: it must exceed the expected
  # duration or the attempt timeout kills work that was about to checkpoint.
  #
  # Two consequences the submitter should know before choosing this profile:
  #   - 20 h exceeds policy.yaml's routine_maximum_runtime_hours of 12, so this is an
  #     EXCEPTION-class request and faces the admin gate rather than the team lead. That is
  #     correct: a multi-hour multi-GPU run is not routine.
  #   - maximum_attempts is 2, which is only permitted because checkpoint is non-null.
  #     require_checkpoint_for_retries refuses a retryable workload that cannot resume, and
  #     a second attempt that restarts from step 0 is a second full bill.
  - name: olmo-core-train-370m-20b
    repository: OLMo-core
    compute_profile: gpu-8xa100
    maximum_runtime_hours: "20"
    maximum_attempts: 2
    checkpoint:
      interval_minutes: 30
      destination_prefix: "s3://sbsandbox-intern-edullm-outputs/teams/"
      resume_required: true
```

`destination_prefix` is the root the derived per-run prefix hangs under, matching
`olmo-core-train-smoke`'s repointed value (`config/workload-catalog.yaml:174`) — the actual path
is `output_prefix(team, run_id) + "checkpoints/"`, and a static string cannot carry a run id.

And optional submission fields, on `SubmissionInputs` beside
`src/edullm_platform/submission.py:101` and mirrored into `RunManifest`:

```python
    # SOURCE NAMES AND RATIOS, NEVER A RESOLVED SHARD LIST. The 8192-byte
    # ContainerOverrides ceiling is ~76x under the 6,921 URIs of the 150B corpus, and the
    # refusal arrives after the approval. Names are hundreds of bytes; the container
    # expands them by listing the dataset prefix, which is what its ListBucket grant is for.
    #
    # Optional, and absence means the whole dataset at its published proportions -- not an
    # empty mix. A submission that names no subset is the common case and must stay short.
    dataset_sources: tuple[str, ...] | None = Field(default=None)
    dataset_source_ratios: tuple[Decimal, ...] | None = Field(default=None)
    # A token budget rather than a step count, because steps are a function of batch size
    # and sequence length and are therefore not comparable across two runs. Whole tokens.
    training_tokens: int | None = Field(default=None, ge=1)
```

Three things to enforce, in the spirit of the existing `validate_fanout_is_whole_or_absent`
(`submission.py:114-126`) — declare all of a group or none of it:

1. `dataset_sources` and `dataset_source_ratios` are the same length, or ratios are absent
   (meaning: equal weight, or the dataset's own published proportions — pick one and write it
   down).
2. Ratios sum to 1 within a stated tolerance, using `Decimal` and never binary float. The
   codebase already refuses float for anything an approver reads —
   `_plain`/`serialize_decimal` in `submission.py:66-75` exists because `Decimal("5E+2")`
   rendered as `$5E+2` in front of a reviewer.
3. **Every named source exists in the release.** This is a refusal that belongs in the
   credential-free compile job, for the exact reason `SubmissionRefusedError`
   (`submission.py:78-86`) gives: a submission naming something unresolvable is going to fail
   whatever a reviewer says, and spending a human's attention on it first teaches reviewers that
   approving is a formality. A typo'd source name should cost a red CI run, not an approval and a
   GPU hour.

Check (3) needs the platform to know the release's group names. That is Change 2's `uri` plus one
`GetObject` of `dataset.json` — or, if you'd rather keep the compile job credential-free (which is
a property worth keeping: it holds no `id-token` permission by construction,
`.github/workflows/submit-run.yml:94-99`), list the valid source names in `config/datasets.yaml`
beside the `uri`. The second option keeps the reviewed configuration self-contained and the
compile job offline.

`workflow_dispatch` has room: the form declares 14 inputs against a ceiling of 25, so three more
is 17 with headroom to spare. Note the existing convention for optional `choice` inputs — a choice
cannot offer a blank, so `compute_profile` spells "take the default" as the literal
`inherit`, translated back to absence at `.github/workflows/submit-run.yml:196-201`. Left
untranslated it reaches admission as the name of a profile nothing registers.

---

## Change 4 — provision real GPU, and raise the timeout

### The profile already exists

`config/workload-catalog.yaml:100-107`:

```yaml
  - name: gpu-8xa100
    instance_type: p4d.24xlarge
    accelerator: gpu
    nodes: 1
    hourly_rate_usd: "21.9576"
    pricing_source: "AWS Price List API get-products AmazonEC2 p4d.24xlarge us-east-1 Linux/Shared/NA/Used"
    pricing_observed_at: "2026-07-25"
    provisioned: false
```

It is priced and shaped correctly; it is simply not backed. Promoting it means setting
`provisioned: true` **and** adding a row to `config/execution-targets.yaml`. That file's comment
is explicit that the two are halves of one change, and
`tests/test_phase3_execution.py::test_every_provisioned_profile_is_backed_and_every_target_names_a_provisioned_one`
refuses either alone.

### Deployed limits, read back live

From CloudFormation stack `sbsandbox-intern-edullm-phase4-gpu` (`UPDATE_COMPLETE`, last updated
2026-07-29T02:03Z), confirmed against the live Batch resources rather than the template:

| Limit | Deployed | Template anchor |
|---|---|---|
| Instance types | `["g5.xlarge"]` only | `infra/batch-compute-gpu.yaml:84-85` |
| `MaxvCpus` | `16` | `infra/batch-compute-gpu.yaml:66` |
| `MinvCpus` | `0` | `infra/batch-compute-gpu.yaml:61` |
| `attemptDurationSeconds` | `3600` | `infra/batch-compute-gpu.yaml:170` |
| `ResourceRequirements` | VCPU 4, MEMORY 15360, **GPU 1** | `infra/batch-compute-gpu.yaml:208-214` |
| Job definition | `sbsandbox-intern-edullm-gpu-run` rev 3, ACTIVE | — |

### The arithmetic

**These are 6ND-at-assumed-MFU planning estimates, not quotes.** 6ND is a first-order
approximation of training FLOPs; MFU is an assumption about how much of the hardware's peak a real
training loop actually lands, and it varies with parallelism strategy, batch size, sequence
length, attention implementation and data loading. Treat the shape of the answer as robust and any
single cell as approximate.

A 370M-parameter model on 20B tokens:

```
FLOPs ≈ 6ND = 6 × 3.7e8 × 2.0e10 = 4.44e19
```

Peak bf16 dense throughput: A100-40GB SXM = 312 TFLOPS/GPU, so `p4d.24xlarge` (8 GPUs, confirmed
live via `describe-instance-types`) aggregates ~2,496 TFLOPS. A10G ≈ 62.5 TFLOPS dense, and
`g5.xlarge` has exactly 1 (22,888 MiB), so ~62.5 TFLOPS.

| MFU | p4d.24xlarge | at $21.9576/hr | 1× g5.xlarge | at $1.006/hr |
|---|---|---|---|---|
| 25% | 19.8 h | $434 | 789 h (33 d) | $794 |
| 30% | 16.5 h | $362 | 658 h (27 d) | $662 |
| **35%** | **14.1 h** | **$310** | **564 h (23 d)** | **$567** |
| 40% | 12.4 h | $271 | 493 h (21 d) | $496 |
| 50% | 9.9 h | $217 | 395 h (16 d) | $397 |

**The headline: the small GPU is not cheaper, only slower.** At 35% MFU the eight-A100 node
finishes in about **14 hours**; one `g5.xlarge` takes about **23 days** — roughly **40x the
wall-clock** — and, at the catalog's own rates, costs **more** ($567 vs $310, about 1.8x). There
is no budget saved by waiting three weeks. The cheap instance is cheap per hour and expensive per
result.

**Three corrections to figures currently in circulation**, flagged because a wrong number in a
budget conversation is worse than no number:

1. **`p4d.24xlarge` is $21.9576/hr, not ~$32.77.** The catalog says `21.9576` at line 104, and
   the live Price List API confirms it: *"$21.957642 per On Demand Linux p4d.24xlarge Instance
   Hour"*, SKU `H7NGEAC6UEHNTKSJ`, publication date 2026-07-28. The $32.77 figure matches nothing
   in the repository or the API. Using it inflates the p4d estimate by ~50% ($463 vs $310) and is
   the single number most likely to make this look like a worse trade than it is.
2. **"~16.5 h" corresponds to ~30% MFU, not 35%.** At 35% the figure is ~14.1 h. Minor, but the
   assumption should travel with the number.
3. **"~27 days on g5 at the same total cost" is not the same cost.** At the catalog's rates g5 is
   *more* expensive, not equal. (27 days is the 30%-MFU row; 23 days is 35%.) This strengthens the
   argument rather than weakening it — but state it correctly, because "same cost, just slower" is
   an argument someone can reasonably accept, and "1.8x the cost and 40x slower" is not.

**And the 3600 s timeout kills the run regardless of GPU.** `attemptDurationSeconds: 3600` is
deployed on the job definition, and `execution.py:110-118` sends a per-attempt bound on **every**
submit — the module docstring is explicit that this is unconditional by design, because *"the
specific way that requirement dies quietly is a timeout applied only when the manifest sets a
runtime bound."* A 14-hour run against a 1-hour attempt is killed 14 times over. Raising
`maximum_runtime_hours` on the workload profile raises what admission *sends*; the job
definition's own `Timeout` is the ceiling Batch applies. Both have to move, and only one of them
is in a config file.

### Trap (a) — `p4d.24xlarge` is not offered in `us-east-1f`

Verified live with `describe-instance-type-offerings`. `p4d.24xlarge` is offered in:

```
us-east-1a   us-east-1b   us-east-1c   us-east-1d
```

**`us-east-1f` is absent.** The GPU compute environment currently lists all five subnets
(`infra/batch-compute-gpu.yaml:99-104`) because `g5.xlarge` is offered in all five. The template
already warns about exactly this at lines 91-92 and 48-50:

> It is not a general property of GPU shapes: g6e and p4d are absent from `us-east-1f`, so
> promoting an L40S or an A100 later means dropping a subnet here.

**Promoting the instance type without editing the subnet list produces a job that waits forever
with no error.** Nothing fails. No permission is denied, no template is invalid, no log line is
written. The job sits in `RUNNABLE` and a person eventually notices.

The template also names the trap that makes this hard to diagnose (lines 94-98): **a dry-run is
not evidence and must never be used as one.** `--dry-run` in this account returns
`DryRunOperation` for `g6e.12xlarge` in `us-east-1f` — an AZ that does not offer it — because it
answers "is this principal permitted", not "can this succeed". Use
`describe-instance-type-offerings`, as above.

### Trap (b) — three more edits the instance type does not carry with it

Beyond the subnet list, promoting `gpu-8xa100` needs:

1. **`MaxvCpus: 16` blocks `p4d.24xlarge` outright.** A `p4d.24xlarge` is **96 vCPU** (confirmed
   live). A ceiling of 16 cannot admit a single one. This is the same silent `RUNNABLE` failure as
   the subnet trap, reached by a different route, and it is easy to miss because 16 reads like a
   *cost* guardrail rather than a *placement* one. It is both.
2. **`ResourceRequirements` is sized for `g5.xlarge`** — VCPU 4, MEMORY 15360, **GPU 1** (lines
   208-214, confirmed on rev 3). Left as-is, a container on an 8-GPU node requests **one** GPU:
   seven A100s idle at full price, and, per the template's own comment, *"the count here and the
   nodes/accelerator shape in `config/workload-catalog.yaml` are two statements of the same fact
   with nothing connecting them, which is why a test compares them."*
3. **`hourly_rate_usd` must stay true to the shape that runs.** The comment at lines 67-83 argues
   for one instance type precisely because *"Batch places a job on any listed type that fits its
   `ResourceRequirements`"* — so a job priced at one rate can bill at another, and the estimate in
   an approved record is only true if the shape that runs is the shape that was priced.

### Trap (c) — this is a budget conversation, not a config tweak

Both current limits are **deliberate, documented cost guardrails**, and they should be read as
decisions rather than defaults. `infra/batch-compute-gpu.yaml:62-65`:

> Sixteen, which is four `g5.xlarge` and a ceiling of $4.02/hr. The CPU environment's 128 is sized
> so a fan-out of a few cells fits; this one is sized so a single-node phase cannot accidentally
> become a cluster. **Raising it is a deliberate edit with a number attached.**

And on `MinvCpus: 0` (lines 57-60): *"an idle `g5.xlarge` is $1.006/hr and buys nothing at all
while it waits."*

So the number to attach: one `p4d.24xlarge` is **$21.9576/hr**, which is **21.8x** the ceiling
those 16 vCPUs were sized to hold. `MaxvCpus` must reach at least 96 for one node. At 96 the
environment's worst case is one p4d — about **$527/day** if something is left running. At 192 it
is two, and so on; `MaxvCpus` is the only thing standing between a mistake and a multiple.

Also worth surfacing to whoever approves the budget: a ~14-hour, ~$310 run is an **exception-class**
submission, not routine. `config/policy.yaml` sets `routine_maximum_cost_usd: "500"` and
`routine_maximum_runtime_hours: "12"`, and `classify_request`
(`src/edullm_platform/contracts/policy.py:101-119`) requires **both** to pass for `ROUTINE`. At 14
hours the runtime threshold is exceeded even though the cost is not, so it routes to the
`platform_admin` gate. That is the design working: this run should face a different reviewer than a
smoke test does.

The rollback property is intact and worth stating, because it is what makes the conversation
reversible. The GPU queue and compute environment are their own
(`infra/batch-compute-gpu.yaml:18-21`): disabling them scales GPU spend to zero and does not touch
CPU execution.

---

## What the dataset side guarantees, so you don't re-check it

Context on what is on the far side of the grant, so the platform can rely on it rather than
re-verifying:

- **Immutable.** `v1` is never edited to change data; a change publishes `v2`. So a
  `uri` + `manifest_sha256` pin is stable for the life of the `release_id`. The one sanctioned
  in-place write is a descriptive-keys-only README backfill, guarded by an assertion that the
  manifest hashes stay byte-identical.
- **Nothing unvalidated is in the bucket.** Producers write only to `s3://edullm-landing`; a
  validator role that no human session can assume is the only principal that can `PutObject` to
  `s3://edullm-data`. That is an IAM Deny, not a convention.
- **Every claimed number was recomputed from the bytes.** The validator re-hashes every file
  against the manifest, both directions — no missing files, no stray unlisted ones — and derives
  token counts from file sizes rather than believing a field. This exists because an
  `inventory.json` once claimed 98 files / 172 GB in a folder holding 10 files / 31 GB.
- **`tokens × dtype_size == file bytes`, exactly**, and the extension matches the real bytes.
  Token shards are `.u32le.bin`, never `.npy`. Legacy `.npy` files in the old bucket were
  headerless raw uint32 — the extension lied, and reading them as `.npy` silently corrupted
  training.
- **The dtype is the trap most likely to bite a trainer.** These corpora are **`uint32`**;
  OLMo-core's `NumpyFSLDataset` defaults to **`uint16`**. Inferring the dtype silently halves the
  token count and trains on garbage — no error, plausible-looking loss curve. Use
  `edullm_data.read.dataset_paths()`, which returns the correct dtype from the manifest alongside
  the URIs, rather than pasting a glob:

  ```python
  from edullm_data.read import dataset_paths, resolve_latest
  ver = resolve_latest("pretrain/olmo-150b-dolma2", s3=s3)
  r   = dataset_paths("pretrain/olmo-150b-dolma2", ver, split="train", s3=s3)
  # r.paths        -> 6,851 URIs
  # r.numpy_dtype  -> "<u4"   <-- FEED THIS ONE
  ```

  **Pass `r.numpy_dtype`, not `r.dtype`.** `r.dtype` is the bare name `"uint32"`, and
  `np.dtype("uint32")` resolves to *native* byte order — correct only by accident on a
  little-endian host. `r.numpy_dtype` is byte-order-qualified (`"<u4"`) and correct anywhere.
  An earlier draft of this document said to feed `r.dtype`; that contradicted
  `docs/CONSUMER-CONTRACT.md` on the one field whose silent failure mode is corrupted training
  data.

  The reader also refuses a prefix carrying no `_VALIDATED.json` seal, and can verify the seal
  against a pinned `manifest_sha256` — which is what makes Change 2's digest worth carrying.

More detail in `docs/ONBOARDING.md` (the two-minute model) and `README.md` in this repository.

---

## Suggested landing order

1. **Change 1 alone**, then prove it: a Batch task under the real GPU workload role does one
   `GetObject` and one `ListBucket` against `edullm-data`. Live, not simulated — the simulator
   has known false denials for this account's intern role. Re-capture the role fixtures.
2. **Change 2**, with the new prefix type, the `test_dataset.py` positive case, and the three env
   vars. At this point a smoke-length job can open a real dataset and the lineage record is true.
3. **Change 3**, which is submission plumbing and needs no infrastructure. The profile it adds
   resolves to an unprovisioned compute profile until step 4 — an honest refusal, not a break.
4. **Change 4** last, because it is the one that spends money and the only one that needs a
   budget decision. Bring the arithmetic above, the corrected $21.9576 rate, and the
   `MaxvCpus` → 96 number.

Steps 1-3 are reviewable, testable, and cost nothing to run. Step 4 is a conversation.
