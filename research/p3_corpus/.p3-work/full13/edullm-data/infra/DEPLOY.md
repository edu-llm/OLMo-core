# Deploying the eduLLM airlock

Implements **Step 2** of [`DATASET-STANDARD.md`](../../docs/dataset-creation/DATASET-STANDARD.md) §13,
plus the event wiring from step 9.

## How to run these commands

**There is no local AWS CLI in this environment.** Every command below runs through the `sb-aws` MCP
broker, which mints credentials on the fly. So each step is given as the **argument array** to hand to
the tool, not as a shell line:

```
mcp__sb-aws__aws(account="sbsandbox", command=[ ...array below... ])
```

`account` is **always `sbsandbox`** — account `<ACCOUNT_ID>`, region `us-east-1`. Never `sbproduction`,
never `legacy`. §10: *one region*, `us-east-1`, for everything.

Two consequences of going through the broker:

- **File paths must be readable by the broker's CLI process.** `--template-body` and `--policy-document`
  take `file://` URIs. If the broker runs the CLI somewhere that cannot see this repo, use the inline
  `--cli-input-json`-free fallbacks noted per step, or paste the document as a literal string argument.
  Verify once with the `--generate-cli-skeleton`-free dry check in step 0.
- **The broker's identity is an intern session**, so it is bound by `<PERMISSION_BOUNDARY>`. Everything
  below is on §1's verified-allowed list. Nothing here calls `iam:CreateRole` or
  `batch:RegisterJobDefinition`.

Only actions §1 smoke-tested live are used. The one exception is flagged loudly in step 5.

---

## Order matters

```
0. preflight        who am I, does anything already exist
1. 01-buckets       create both buckets                      (CFN)
2. 03-validator     grant the validator its S3 access        (iam:put-role-policy)
3. 02-bucket-policy lock edullm-data                         (s3api:put-bucket-policy)
4. verify           read back every setting
5. 04-event-wiring  the EventBridge rule, DISABLED           (CFN)
6. smoke test       prove the airlock actually holds
```

**Step 2 must precede step 3.** The bucket policy denies writes from every principal except the two
role ARNs; if the validator has no *identity*-based Allow yet, it is denied by default and the smoke
test in step 6 fails for the wrong reason. Grant first, then lock.

**Do not reorder step 3 after step 6.** The whole point of the smoke test is to exercise the lock.

---

## Step 0 — preflight

Confirm the broker session and the target account:

```
mcp__sb-aws__whoami()
```

```
mcp__sb-aws__aws(account="sbsandbox", command=["sts","get-caller-identity"])
```

Expect account `<ACCOUNT_ID>`.

Confirm the bucket names are free (a global namespace — `edullm-data` may be taken by a stranger):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","head-bucket","--bucket","edullm-data"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","head-bucket","--bucket","edullm-landing"
])
```

**Expect `404 Not Found` for both.** That is success. `403 Forbidden` means the name exists and belongs
to someone else — stop and pick new names, because every ARN in `02-bucket-policy.json`,
`03-validator-policy.json`, and all six `families/*.json` would need to change together.

Confirm the validator role and its existing inline policy, so step 2 cannot clobber it:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","list-role-policies","--role-name","<BATCH_JOB_ROLE>"
])
```

**Expect `write-team-outputs-only` in the list.** §1: *"the role already carries an inline
`write-team-outputs-only` policy — add alongside it, don't replace it."* `put-role-policy` in step 2
uses the *new* name `dataset-validator`, so it adds. If you ever see `dataset-validator` already there,
you are re-running — that is fine and idempotent.

Confirm the Batch queue and job definition exist and the resource nulls are still true:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "batch","describe-job-queues","--job-queues","<JOB_QUEUE>",
  "--query","jobQueues[0].{name:jobQueueName,state:state,status:status,arn:jobQueueArn}"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "batch","describe-job-definitions",
  "--job-definition-name","<JOB_DEFINITION>","--status","ACTIVE",
  "--query","jobDefinitions[].{rev:revision,vcpus:containerProperties.vcpus,mem:containerProperties.memory,jobRole:containerProperties.jobRoleArn,image:containerProperties.image}"
])
```

**Expect `vcpus: null` and `memory: null`.** §1: *"every submission must supply them in the override or
the job fails to place."* Every `submit-job` below carries them. If a later revision has them baked in,
note it — that changes the recommended fix in `04-event-wiring.yaml`.

---

## Step 1 — create the buckets

`s3:CreateBucket`, `PutBucketVersioning`, `PutBucketLifecycleConfiguration`,
`PutBucketNotificationConfiguration` are all ✅ in §1's table.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","deploy",
  "--stack-name","edullm-data-buckets",
  "--template-file","infra/01-buckets.yaml",
  "--no-fail-on-empty-changeset",
  "--tags","Project=edullm","Owner=edullm-data@alphaaiengineering.com",
           "ManagedBy=edullm-data/infra/01-buckets.yaml"
])
```

`--capabilities` is **not** needed: this template creates no IAM resources at all (§1 — `iam:CreateRole`
is denied, so the design reuses an existing role).

`deploy` is a `cloudformation` *custom* command (changeset create + execute + wait). If the broker only
passes through raw API calls, use `create-stack` plus a wait instead:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","create-stack",
  "--stack-name","edullm-data-buckets",
  "--template-body","file://infra/01-buckets.yaml",
  "--tags","Key=Project,Value=edullm","Key=Owner,Value=edullm-data@alphaaiengineering.com"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","wait","stack-create-complete","--stack-name","edullm-data-buckets"
])
```

Then read the outputs:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","describe-stacks","--stack-name","edullm-data-buckets",
  "--query","Stacks[0].Outputs"
])
```

Expect four outputs: `DataBucketName`, `DataBucketArn`, `LandingBucketName`, `LandingBucketArn`.

> **If the stack rolls back on `BucketAlreadyExists`**, step 0's `head-bucket` was skipped. Both buckets
> are `DeletionPolicy: Retain`, so a rollback that already created one bucket will *keep* it and the
> retry will fail again. Delete the orphan by hand (`s3api delete-bucket`, it is empty) or delete the
> failed stack first — see Teardown.

---

## Step 1b — enable EventBridge on landing (out of band)

CloudFormation **rejects** `NotificationConfiguration: {EventBridgeConfiguration: {}}` at
template-validation time — verified: the full-template create rolled back with "Validation failed with
1 error(s)", and a 6-line minimal template reproduced it, while `s3api
put-bucket-notification-configuration` accepts the identical config. So it is applied via the API here,
the same out-of-band pattern as the bucket policy (step 3).

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","put-bucket-notification-configuration",
  "--bucket","edullm-landing",
  "--notification-configuration","{\"EventBridgeConfiguration\":{}}"
])
```

Verify:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-notification-configuration","--bucket","edullm-landing"
])
```

Expect `{"EventBridgeConfiguration": {}}`. This only *enables* the event stream; nothing consumes it
until the rule in step 5 is enabled, which per §"Event wiring" waits for the self-discovering validator.

---

## Step 2 — grant the validator its S3 access

§1: `iam:PutRolePolicy` on this role is **allowed**. Policy name is `dataset-validator`; the existing
`write-team-outputs-only` is untouched.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","put-role-policy",
  "--role-name","<BATCH_JOB_ROLE>",
  "--policy-name","dataset-validator",
  "--policy-document","file://infra/03-validator-policy.json"
])
```

If `file://` is not resolvable through the broker, pass the document inline as a single JSON string
argument in place of the `file://…` value.

Confirm both policies are present and read the new one back:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","list-role-policies","--role-name","<BATCH_JOB_ROLE>"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","get-role-policy",
  "--role-name","<BATCH_JOB_ROLE>",
  "--policy-name","dataset-validator"
])
```

**Expect `["write-team-outputs-only","dataset-validator"]`** (in either order). If
`write-team-outputs-only` is gone, you used `--policy-name write-team-outputs-only` by mistake — restore
it from the audit trail before doing anything else.

---

## Step 3 — lock `edullm-data`

`s3:PutBucketPolicy` is ✅ in §1's table (a conditional-write Deny was applied on a probe bucket).

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","put-bucket-policy",
  "--bucket","edullm-data",
  "--policy","file://infra/02-bucket-policy.json"
])
```

### Deploying the split Delete Deny (policy v2 — PENDING)

`02-bucket-policy.json` in this repo is now **v2** (`Id: edullm-data-airlock-v2`, three
statements). The live bucket still carries v1. Applying it is the ordinary
`put-bucket-policy` call below — but understand what changes before you run it.

**What v1 got wrong.** One Deny covered `PutObject` *and* `DeleteObject`/`DeleteObjectVersion`,
and exempted `<BATCH_JOB_ROLE>` + `<INFRA_DEPLOYER_ROLE>` from **all five actions**. So the
bucket policy permitted the validator to delete published data, and the only thing stopping it
was that `03-validator-policy.json` grants `PutObject` and not `Delete*`.

That is an *identity* policy on a role whose inline policies are editable with
`iam:PutRolePolicy` — which the intern session has (see §"the simulator lies"). One
`put-role-policy` call away from a principal that can erase a published, frozen dataset.
"Frozen means frozen" was resting on the wrong kind of control.

**v2 splits it in two:**

| Sid | Actions | Exempt |
| --- | --- | --- |
| `OnlyValidatorWrites` | `PutObject`, `PutObjectTagging`, `AbortMultipartUpload` | validator + deployer |
| `NobodyDeletesPublishedData` | `DeleteObject`, `DeleteObjectVersion` | **nobody** |

A resource-based Deny with no principal exemption cannot be escaped by editing any identity
policy, because an explicit Deny always wins. This is Object Lock's guarantee without Object
Lock's four failure modes (§"why not Object Lock").

**Consequences to accept before applying:**

- **Deleting a published dataset becomes a two-step, deliberate act**: edit the bucket policy
  first, then delete. That is the point — it cannot happen as a side effect of a buggy job.
- **Lifecycle expiry still works.** Lifecycle runs as the S3 service, and every statement
  carries `BoolIfExists aws:PrincipalIsAWSService=false`. `edullm-data` has no expiry rule
  anyway; landing's is on a different bucket and unaffected.
- **`AbortMultipartUpload` stays on the write side deliberately.** It is cleanup of an
  in-flight upload, not deletion of published data, and the validator needs it to retry a
  failed large copy.
- **Planned deletion of the 31B corpus will require this two-step.** That is a real cost and
  the correct one; note it in the deletion runbook when that happens.

Verify after applying — the Deny must now fire for the validator role too, not just interns:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-policy","--bucket","edullm-data","--output","text"
])
```

Expect three statements and `edullm-data-airlock-v2`. Then re-run the intern `PutObject` check
below; it must still be an explicit deny (v2 must not have widened write access by accident).

### Before you run this, re-read the three footguns

JSON cannot carry comments, so `02-bucket-policy.json` is bare. §1's warnings, restated:

1. **`aws:PrincipalArn` holds the *role* ARN, never the session ARN.** Writing
   `arn:aws:sts::<ACCOUNT_ID>:assumed-role/Role/session` makes the condition *always true* and the Deny
   fires on **everyone**, including you. The file has the `arn:aws:iam::…:role/…` form. Keep it.
2. **Never `s3:*` or `s3:Put*` in the Deny.** That catches `PutBucketPolicy` itself — a hard lockout
   that binds root and needs AWS Support to undo. The file lists five data-plane actions only,
   now split across two Deny statements (three write actions + two delete actions).
3. **`ArnNotEqualsIfExists`, not `ArnNotEquals`.** The `IfExists` suffix matters for requests where
   `aws:PrincipalArn` is absent.

Read it straight back and diff it against the file by eye:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-policy","--bucket","edullm-data","--output","text"
])
```

Confirm you are still able to manage the bucket — this is the "did I lock myself out" check, and it must
be run **now**, while the fix is still one call away:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-policy-status","--bucket","edullm-data"
])
```

Expect `"IsPublic": false`. If `put-bucket-policy` itself starts returning `AccessDenied` on a retry,
the Deny has caught the control plane — use the break-glass role
`arn:aws:iam::<ACCOUNT_ID>:role/<INFRA_DEPLOYER_ROLE>`, which is listed in the
condition exactly so this is recoverable.

`edullm-landing` gets **no** bucket policy. §1: *anything may write here.* Its protection is the 14-day
expiry and the fact that nothing trains from it.

---

## Step 4 — verify every §10 row

Read back, do not assume. §1: *"Do not trust `iam:simulate-principal-policy` for this role"* — it
reported `explicitDeny` for ten actions that all work. The same skepticism applies to CFN success.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-versioning","--bucket","edullm-data"
])
```
Expect `"Status": "Enabled"`.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-versioning","--bucket","edullm-landing"
])
```
Expect an **empty response** — never-versioned. (§10 "off". `Suspended` is only reachable from
`Enabled`, which is why `01-buckets.yaml` omits the property entirely.)

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-lifecycle-configuration","--bucket","edullm-data"
])
```
Expect one rule, `AbortIncompleteMultipartUpload.DaysAfterInitiation: 7`, **no `Expiration`**, **no
`Transitions`**. Also check the response's `TransitionDefaultMinimumObjectSize` field: it should read
`all_storage_classes_128K`, which is the sub-128 KB sidecar protection §10 requires. It is a default AWS
supplies, not something CFN sets, so it must be *observed* here.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-lifecycle-configuration","--bucket","edullm-landing"
])
```
Expect two rules: `ExpirationInDays: 14` and `AbortIncompleteMultipartUpload.DaysAfterInitiation: 1`.
**The 1-day MPU abort is load-bearing** (§10), not hygiene — an in-flight upload is invisible to `LIST`,
so without it a manifest can be sealed while a member is still uploading.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-notification-configuration","--bucket","edullm-landing"
])
```
Expect `{"EventBridgeConfiguration": {}}`. Without this the step-5 rule is inert.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-public-access-block","--bucket","edullm-data"
])
```
```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-public-access-block","--bucket","edullm-landing"
])
```
All four booleans `true` on both.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-bucket-encryption","--bucket","edullm-data"
])
```
Expect `AES256`. **This is the decided encryption mode**, matching §10 — not a placeholder. Both SSE-S3
and SSE-KMS encrypt at rest with AES-256; the difference is key custody. KMS buys revocation and a decrypt
audit trail, neither of which is needed under the standard's No-PII assumption, and a CMK adds a second
authorization system (the key policy) that can make an intact bucket unreadable. Rationale in full in the
comment block in `01-buckets.yaml`.

---

## Step 5 — event wiring (deployed DISABLED)

`events:PutRule` and `events:PutTargets` are both ✅ in §1's table (a real Batch-queue target attached
with `FailedEntryCount: 0`).

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","deploy",
  "--stack-name","edullm-data-event-wiring",
  "--template-file","infra/04-event-wiring.yaml",
  "--no-fail-on-empty-changeset",
  "--parameter-overrides","RuleState=DISABLED",
  "--tags","Project=edullm","ManagedBy=edullm-data/infra/04-event-wiring.yaml"
])
```

### Why DISABLED, and do not flip it yet

Two independent blockers, both documented at length at the top of `04-event-wiring.yaml`:

1. **EventBridge cannot send a container override.** `BatchParameters` has exactly four members
   (`ArrayProperties`, `JobDefinition`, `JobName`, `RetryStrategy`) — no `ContainerOverrides`, and
   `InputTransformer` cannot reach it. So an event-submitted job gets neither `vcpus`/`memory` (both
   `null` on revision 1 → **cannot place**) nor the landing prefix to validate. At best it would run the
   definition's default command, print `no command override was supplied`, and exit 0 — a validator that
   silently blesses nothing while looking healthy.
2. **The target `RoleArn` is very likely unusable.** EventBridge must `sts:AssumeRole` the role to call
   `SubmitJob`, but §1 records this role's trust policy allows **only `ecs-tasks.amazonaws.com`** — and
   that fact is what the entire airlock rests on. `events.amazonaws.com` is not on it, so invocations are
   expected to fail with `AccessDenied`, visible only as the rule's `FailedInvocations` metric. Do not
   widen the trust policy to fix this without thinking hard: it widens the sole writer to `edullm-data`.

The fix is a small dispatcher (EventBridge → `SubmitJob` with a full `--container-overrides`), which
needs an execution role — fold it into the same one-line admin ask as §13's dedicated `DatasetValidator`
role. Until then, run the validator on a **sweep** using the step-6 submit shape, which *can* express
everything. Full option comparison is in the template's KNOWN LIMITATION block.

Confirm the rule exists and is disabled:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "events","describe-rule","--name","edullm-landing-manifest-created"
])
```
Expect `"State": "DISABLED"`.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "events","list-targets-by-rule","--rule","edullm-landing-manifest-created"
])
```
Expect one target whose `Arn` is the job-queue ARN and whose `BatchParameters.JobDefinition` is
`<JOB_DEFINITION>`.

---

## Step 6 — end-to-end smoke test

§13: *"Validate the whole path end-to-end before writing real data."* This is the infrastructure half of
that — it proves the **airlock**, not the validator (the package does not exist yet). Three assertions:

- **6a** the validator role *can* write to `edullm-data`
- **6b** a non-validator principal *cannot*
- **6c** landing accepts writes from anyone

`batch:SubmitJob` and `batch:DescribeJobs` are both ✅ in §1's table.

### 6a — the validator writes one object

`--container-overrides` supplies `vcpus` and `memory` because the definition has both `null` (§1). The
command writes a single trivial object to `edullm-data` using the container's own credentials — i.e. as
`<BATCH_JOB_ROLE>`, the only identity the bucket policy permits.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "batch","submit-job",
  "--job-name","edullm-airlock-smoke-write",
  "--job-queue","<JOB_QUEUE>",
  "--job-definition","<JOB_DEFINITION>",
  "--container-overrides","{\"command\":[\"python\",\"-c\",\"import urllib.request,json,subprocess,sys; subprocess.run([sys.executable,'-m','pip','install','--quiet','awscli'],check=False); open('/tmp/ok.txt','w').write('airlock smoke test\\n'); sys.exit(subprocess.run(['aws','s3api','put-object','--bucket','edullm-data','--key','_smoke/airlock-write-check.txt','--body','/tmp/ok.txt']).returncode)\"],\"vcpus\":2,\"memory\":4096}"
])
```

> The image is digest-pinned (§1) and may or may not already carry an S3 client. If `awscli` is absent
> and cannot be installed, use `boto3` instead — swap the command body for
> `import boto3; boto3.client('s3').put_object(Bucket='edullm-data', Key='_smoke/airlock-write-check.txt', Body=b'ok\n')`.
> This is the same fetch-at-start pattern §1 endorses for the validator itself
> (`uv pip install "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"`).

Poll it (note the returned `jobId`):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "batch","describe-jobs","--jobs","<jobId>",
  "--query","jobs[0].{status:status,reason:statusReason,exit:container.exitCode,log:container.logStreamName}"
])
```

Expect `status: SUCCEEDED`, `exit: 0`. A `RUNNABLE` job stuck forever means the resource override did
not take — re-check `vcpus`/`memory`. `FAILED` with an S3 `AccessDenied` in the log means step 2 did not
land, or the bucket policy's condition is inverted (footgun 1 in step 3).

Confirm the object arrived, and that versioning is genuinely on:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","head-object","--bucket","edullm-data","--key","_smoke/airlock-write-check.txt"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","list-object-versions","--bucket","edullm-data","--prefix","_smoke/"
])
```

Expect a real `VersionId` (not `"null"`). **6a passes.**

### 6b — a non-validator principal is denied

This is the load-bearing assertion. The broker's own intern session is a non-validator principal, and
§1's whole claim is that *no human principal can write to `edullm-data` at all*.

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","put-object","--bucket","edullm-data","--key","_smoke/should-be-denied.txt"
])
```

**Expect `AccessDenied`. A success here is a total failure of the design** — stop, and re-read
`02-bucket-policy.json` against §1 before writing any real data. The most likely cause is footgun 1: a
session-form ARN in the condition, which makes `ArnNotEqualsIfExists` always true and (perversely) can
be misread as working.

Deletion must be denied too (§10 *deny-delete: all but break-glass*):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","delete-object","--bucket","edullm-data","--key","_smoke/airlock-write-check.txt"
])
```

**Expect `AccessDenied`.** Note this is `s3:DeleteObject` — the Deny also covers
`s3:DeleteObjectVersion`, so the intern session cannot remove the version either. Cleanup of the smoke
object therefore requires the break-glass role; see Teardown.

Reads must still work for everyone (§10 *readers: everyone*):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","get-object","--bucket","edullm-data",
  "--key","_smoke/airlock-write-check.txt","/tmp/readback.txt"
])
```

Expect success. The Deny lists data-plane **write** actions only — `s3:GetObject` is not among them.
If this fails, an action crept into the Deny that should not be there.

### 6c — landing accepts writes from anyone

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","put-object","--bucket","edullm-landing","--key","_smoke/landing-write-check.txt"
])
```

Expect success — landing is the only door, and it is open (§1). This also confirms no bucket policy was
applied to landing by mistake.

Then clean up landing (no policy, so this is allowed; and it would expire in 14 days regardless):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","delete-object","--bucket","edullm-landing","--key","_smoke/landing-write-check.txt"
])
```

### What step 6 does **not** prove

- **Not** that Gate A works — the validator package does not exist yet (§13 steps 3–7).
- **Not** that the EventBridge rule fires — it is DISABLED, and step 5 explains why.
- **Not** that promotion works. §7: `CopyObject` caps at 5 GB single-part, so the validator must use
  **multipart copy**; 8 of the 15 largest objects in the audit exceeded that. Untested until the package
  lands, and it is the most likely place for a first-run surprise.
- **Not** the §1 known gap: the validator shares a role with general Batch workloads, so **any** Batch
  job on `<JOB_QUEUE>` inherits write access to `edullm-data`. Step 6a *demonstrates*
  this gap as much as it demonstrates the mechanism — the job that succeeded was an arbitrary one, not a
  validator. Mitigations, in §1's preference order: (a) ask an admin for a dedicated `DatasetValidator`
  role, or (b) add a condition on the job-definition ARN to `03-validator-policy.json`.

---

## Teardown

**Order is the reverse of deployment, and step T2 is mandatory before T3.** Both buckets are
`DeletionPolicy: Retain`, so deleting the stacks does **not** delete the buckets — deliberately (§10:
*"CDK/CFN auto-delete buckets are never a dataset home"*). Buckets must be emptied and deleted by hand.

### T1 — remove the event wiring

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","delete-stack","--stack-name","edullm-data-event-wiring"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","wait","stack-delete-complete","--stack-name","edullm-data-event-wiring"
])
```

`events:RemoveTargets` and `events:DeleteRule` are both ✅ in §1's table, so CFN can do this cleanly.

### T2 — remove the bucket policy **before** trying to empty `edullm-data`

The Deny blocks `s3:DeleteObject` and `s3:DeleteObjectVersion` from the intern session, so an empty
attempt will fail while the policy is in place. Drop the policy first:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","delete-bucket-policy","--bucket","edullm-data"
])
```

If this itself returns `AccessDenied`, the Deny has caught the control plane (footgun 2) — assume the
break-glass role `arn:aws:iam::<ACCOUNT_ID>:role/<INFRA_DEPLOYER_ROLE>` and retry.
**§1: use of the break-glass path should be alarmed.** Say so out loud when you use it.

### T3 — empty and delete the buckets

Versioning is on for `edullm-data`, so `s3 rm --recursive` leaves delete markers and noncurrent versions
behind and `delete-bucket` will still fail with `BucketNotEmpty`. Remove **versions**, not objects:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","list-object-versions","--bucket","edullm-data",
  "--query","{Objects: Versions[].{Key:Key,VersionId:VersionId}}","--output","json"
])
```

Feed that JSON to `delete-objects` (repeat until both `Versions` and `DeleteMarkers` are empty — `LIST`
pages at 1000):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","delete-objects","--bucket","edullm-data","--delete","file:///tmp/versions.json"
])
```

Also drain `DeleteMarkers` the same way, then:

```
mcp__sb-aws__aws(account="sbsandbox", command=["s3","rm","s3://edullm-landing","--recursive"])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=["s3api","delete-bucket","--bucket","edullm-data"])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=["s3api","delete-bucket","--bucket","edullm-landing"])
```

Abort any orphaned multipart uploads first if `delete-bucket` complains — the audit found 116 of them in
one bucket, and they are invisible to `LIST`:

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "s3api","list-multipart-uploads","--bucket","edullm-landing"
])
```

### T4 — remove the (now empty) bucket stack

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "cloudformation","delete-stack","--stack-name","edullm-data-buckets"
])
```

### T5 — remove the validator's inline policy

**Delete only `dataset-validator`.** `write-team-outputs-only` belongs to another workload and must
survive (§1).

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","delete-role-policy",
  "--role-name","<BATCH_JOB_ROLE>",
  "--policy-name","dataset-validator"
])
```

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "iam","list-role-policies","--role-name","<BATCH_JOB_ROLE>"
])
```

**Expect `["write-team-outputs-only"]`.** The role itself is never deleted — it was never created here,
and `iam:CreateRole` is denied, so deleting it would be unrecoverable from an intern session.

---

## What is deliberately not here

| Not done | Why |
|---|---|
| KMS key for `edullm-data` | Not wanted. §10 specifies SSE-S3 (AES256): same at-rest encryption, no second authorization system that could make an intact bucket unreadable. KMS's revocation + decrypt audit trail are unnecessary under the No-PII assumption. |
| Any IAM role | `iam:CreateRole` denied by `<PERMISSION_BOUNDARY>` (§1). Design reuses `<BATCH_JOB_ROLE>`. |
| New Batch job definition | Deferred, not blocked — `batch:RegisterJobDefinition` **is** permitted (smoke-tested since: a probe definition was registered and deregistered). Step 4 of the standard's build order registers a dedicated `edullm-validator` definition with a self-discovering default command; until then revision 1 is reused with overrides. |
| Enabled EventBridge rule | Two reasons, both real: `BatchParameters` cannot express `containerOverrides` *or* pass `detail.object.key`, and `<BATCH_JOB_ROLE>` is not assumable by `events.amazonaws.com` (use `<EVENTBRIDGE_INVOKE_ROLE>` as the rule's `RoleArn` — verified to trust `events.amazonaws.com`). Enable only once the self-discovering job definition exists. See step 5. |
| DLQ on the rule target | `sqs:CreateQueue` not in §1's verified table. Watch `FailedInvocations` until it exists. |
| S3 Inventory on `edullm-data` | §13 step 10, after the package. |
| `wu-fsck` schedule (Gate B) | §13 step 11. Needs the package. Owner: Eric Wu — §7 is explicit that an unowned recurring job gets muted and becomes decoration. **Cadence is WEEKLY** (see below); the rule currently live in EventBridge is still the nightly `cron(6 9 * * ? *)` and is NOT changed by a code deploy. |
| Object Lock | §10 rejects it: protects a version not a path, blocks lifecycle, irreversible. |

---

## wu-fsck cadence — nightly → weekly (MANUAL, not covered by a code deploy)

`fsck.py` now documents itself as a **weekly** sweep. That is a code/doc change only. The live
schedule rule is EventBridge state, so **nothing in a package release moves it** — a released wheel
with the new docstring and an unchanged rule will still fire every night.

The rule currently deployed and ENABLED is:

| | |
|---|---|
| Name | `edullm-wu-fsck-nightly` |
| Schedule | `cron(6 9 * * ? *)` UTC (04:06 local), i.e. daily |
| Target | the `edullm-fsck` Batch job definition |
| Owner | Eric Wu (in the name, deliberately — ownership transfers by renaming, not by editing a field) |

To move it, from a session with `events:PutRule`:

```
events put-rule --name edullm-wu-fsck-nightly \
  --schedule-expression "cron(6 9 ? * MON *)" --state ENABLED
```

`cron(6 9 ? * MON *)` = Mondays 09:06 UTC. Keep the same minute/hour so a comparison against the
old runs is like-for-like.

**The name should be renamed too** (`edullm-wu-fsck-weekly`), but renaming an EventBridge rule means
delete + recreate *with its target re-attached*, and the target `RoleArn` question in step 5 applies —
so changing only the schedule expression is the smaller move. If you do rename it, keep `wu-` in the
name: the owner prefix is the mechanism that stops this becoming an unowned job that gets muted.

Rationale for weekly is in `fsck.py`'s module docstring: every fact the sweep re-checks changes only
when something mutates a frozen prefix or another dataset's lifecycle. Those are rare and no more
urgent at 24-hour granularity than at 7-day, and nightly bought seven times the false-alarm exposure
for the same information.
