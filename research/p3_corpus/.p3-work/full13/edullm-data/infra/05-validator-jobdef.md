# Step: the validator job definition + container image

**Status: BLOCKED on one external step — everything else is ready.**

The validator code is done, committed, and tested (352 tests). The airlock is proven in both
directions. What remains is packaging the validator so a Batch job can *run* it, and
registering a job definition that points at that image.

## The one blocker

The validator must reach the `edullm-data` package inside the container. Two ways, both
needing something this workstation does not have:

1. **Bake it into an image** (`infra/Dockerfile.validator`) and push to ECR. Needs Docker
   (not installed here) and `ecr:PutImage` (untested).
2. **`pip install "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.2.0"`** at
   container start (the repo is public, so no auth needed).

The wheel itself builds cleanly (`python -m pip wheel . --no-deps` →
`edullm_data-0.2.0-py3-none-any.whl`, ~115 KB), so nothing about the package blocks this.

## What is verified and ready

- `batch:RegisterJobDefinition` **is allowed** (smoke-tested: a probe def was registered and
  deregistered).
- The validator role can `pip install` and write `edullm-data` (the live promotion test).
- The image runtime is python 3.12 with pip, no boto3, no uv — hence the bundled image.
- `<EVENTBRIDGE_INVOKE_ROLE>` is the correct EventBridge invocation role (deployed in the
  event-wiring stack, DISABLED).

## To finish (pick one path)

### Path A — bundled image (recommended)

On a machine with Docker + ECR push rights:

```bash
cd edullm-data
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
# A dedicated repo is cleaner than reusing olmo-core; create it if allowed:
aws ecr create-repository --repository-name <VALIDATOR_JOBDEF> || true
docker build -f infra/Dockerfile.validator -t \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<VALIDATOR_JOBDEF>:v0.2.0 .
docker push \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<VALIDATOR_JOBDEF>:v0.2.0
```

Then register the job definition (run through the sb-aws broker; all args verified-allowed):

```
mcp__sb-aws__aws(account="sbsandbox", command=[
  "batch","register-job-definition",
  "--job-definition-name","edullm-validator",
  "--type","container",
  "--container-properties", <the JSON below>
])
```

Container properties (fills the vcpus/memory that revision 1 leaves null, and pins the
image by digest after the push resolves one):

```json
{
  "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<VALIDATOR_JOBDEF>@sha256:<digest>",
  "vcpus": 2,
  "memory": 4096,
  "jobRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/<BATCH_JOB_ROLE>",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/<BATCH_EXEC_ROLE>",
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/aws/batch/<JOB_QUEUE>",
      "awslogs-region": "us-east-1",
      "awslogs-stream-prefix": "validator"
    }
  }
}
```

Because the entrypoint is self-discovering (`python -m edullm_data.validate --promote` with
no args), the EventBridge rule can target this definition with empty `BatchParameters` — the
gap documented in `04-event-wiring.yaml` dissolves. Update the rule's `JobDefinition` to
`edullm-validator`, re-point `RuleTargetRoleArn` is already `<EVENTBRIDGE_INVOKE_ROLE>`, then
flip `RuleState` to `ENABLED`.

### Path B — wheel-from-S3 bootstrap (no Docker needed)

Upload the built wheel to a private prefix (the validator role can read landing), and use
the existing `<JOB_DEFINITION>` job definition with a command override that
installs it first:

```
command: ["sh","-lc",
  "pip install -q boto3 numpy && aws s3 cp s3://edullm-landing/_dist/edullm_data-0.2.0-py3-none-any.whl /tmp/ && pip install -q /tmp/edullm_data-0.2.0-py3-none-any.whl && python -m edullm_data.validate --promote"]
```

> **The live job defs are at revision 2 and bootstrap `0.2.0` (cut over 2026-07-30).** Both
> `edullm-validator:2` and `edullm-fsck:2` also assert the version and the presence of
> `families/` immediately after install, and exit non-zero if either is wrong — a silent
> fallback to an old wheel is what let the live corpus be validated against the wrong bounds
> once already. The image has no `aws` CLI, so the deployed command downloads with boto3
> rather than `aws s3 cp` as sketched above.
>
> Both EventBridge rules target the job definition by **unversioned name**
> (`edullm-validator`, `edullm-fsck`), so registering a new revision cuts traffic over with no
> rule edit. That also means a bad revision takes effect immediately — verify with a manual
> `submit-job` before relying on it.
>
> `0.2.0` packages `families/` into the wheel, so the `EDULLM_FAMILIES_DIR` override the
> publish driver sets is no longer required (harmless to leave).

with `--container-overrides` supplying `vcpus`/`memory`. This works today with zero new
infra, at the cost of a pip install per run. Good enough to run the validator manually or on
a schedule; Path A is better for the event-driven steady state.

## Recommendation

Do Path B now to get the validator running end-to-end on real data (it needs only a wheel
upload, which the current session can do), and move to Path A when a Docker host with ECR
push is available. Either way, the code and the airlock are done; this is packaging, not
design.
