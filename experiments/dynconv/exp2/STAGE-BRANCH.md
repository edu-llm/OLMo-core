# Staging the `edullm/**` branch for the Exp-2 image build

**STATUS: STAGED, NOT PUSHED, NOT DISPATCHED.** Every command below is written out to be run by a
human after the go-ahead. Nothing here has been executed.

## Why this exists

Nothing on the platform runs from source. A run names a **commit**, and that commit must already
have been built and pushed to ECR. **No image → the submission is refused.**

**The blocker:** the Exp-2 code lives in `Capstone_LLM/docs/dynconv-review/build/exp2/`, which is
the *container* directory — **not** the `OLMo-core` git repo the image is built from. Verified:
`git ls-tree -r HEAD` on the LIV branch returns **0** paths matching `dynconv`.

**The trap:** agent worktrees here use branches named `agent/<agent-id>/<task-slug>`. That namespace
matches **neither `edullm/**` nor `main`**, so **pushing an agent branch never builds an image** and
every submission naming that commit is refused. The current worktree is on
`agent/claude-01/liv-short-conv-mixer` — exactly that namespace.

Trigger table, verified per repo:

| repo | `workflow_dispatch` | push branches |
|---|---|---|
| `OLMo-core` | yes | `edullm/**`, **`main`** |
| `edullm-data` | yes | `edullm/**` only — NOT `main` |
| `olmo-eval-full` | yes | `edullm/**` only — NOT `main` |

## Base commit

| | |
|---|---|
| worktree | `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer` |
| local branch | `agent/claude-01/liv-short-conv-mixer` (**builds no image**) |
| HEAD | `016c702fefed06852ad6821abe8e2045cdffd146` |
| already on remote as | `origin/edullm/liv-p1-gate-structure` (identical sha, clean tree) |
| existing image digest | `sha256:f2851084433d941b518fe11a57d507d18d62839fb3662ad71b8a2f0bb072de92` |

The base commit already has a **successful** build (2026-08-05 06:25 UTC) and a **COMPLETE** scan
whose 4 CRITICALs are all in `reviewed_vulnerabilities`. So the only thing missing from the image is
our code.

## The commands — DO NOT RUN WITHOUT THE GO-AHEAD

```bash
WT=/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer
SRC=/Users/ericwu/Developer/Capstone_LLM/docs/dynconv-review/build/exp2

cd "$WT"
git checkout -b edullm/dynconv-exp2 016c702fefed06852ad6821abe8e2045cdffd146

mkdir -p docs/dynconv-review/build/exp2
cp "$SRC"/*.py "$SRC"/*.md "$SRC"/run_farmshare.sbatch docs/dynconv-review/build/exp2/

git add docs/dynconv-review/build/exp2
git commit -m "Add the Exp-2 synthetic mechanism study, so it can run from a built image"

# PUSH -- this is the irreversible step, and it is what triggers the build
git push -u origin edullm/dynconv-exp2
```

Then wait for the build and the scan:

```bash
gh run list --repo edu-llm/OLMo-core --workflow edullm-platform-build.yml --limit 5 \
  --json headSha,headBranch,conclusion,createdAt
```

**Wait ~10 minutes for the ECR scan after the push.** Submitting early is refused as
`image_scan_findings_unreviewed`, which reads like a security block but usually just means the scan
is still running. That code is in `denied_outright`, so **nobody — not even an admin — can wave it
through at approval time.**

If the push trigger does not fire for any reason, dispatch explicitly:

```bash
gh workflow run edullm-platform-build.yml --repo edu-llm/OLMo-core --ref edullm/dynconv-exp2
```

## After the build — re-validate offline before submitting

The digest changes, so the pre-validation must be redone. Get the new digest (read-only), then
re-run the compiler with the new sha and digest:

```bash
NEW_SHA=$(git rev-parse HEAD)
# tag is the first 12 chars of the sha; tags are immutable
aws ecr describe-images --repository-name sbsandbox-intern-edullm-olmo-core \
  --image-ids imageTag=${NEW_SHA:0:12} --region us-east-1
```

Then the same `tools/compile_submission.py` invocation recorded in `EXP2-DESIGN.md` §8.3, with
`commit_sha` and `image_digest` updated. **Expect the same verdict** — AUTOMATIC at $0.76 for the
0.95 h pilot, ROUTINE at $19.32 for the 24 h sweep — because neither the profile nor the machine
changes. If it does not reproduce, something else moved and the difference must be understood before
submitting.

## Caveats

- **`agent/**` builds nothing.** Do not shortcut by pushing the current branch name.
- **Do not pin the platform workflow `uses:` to a sha or tag** — the publisher role's trust policy
  matches `job_workflow_ref` with `StringEquals` against `@refs/heads/main`, so a pinned reference
  mints a claim the role will never accept. This is the one place where pinning makes things worse.
- **A green CI build is not a submittable image.** Confirm the *build* workflow succeeded for the
  new sha, not just that tests passed.
- The 4 inherited CRITICALs (`CVE-2026-5450` glibc; `CVE-2026-13221`, `CVE-2026-57433`,
  `CVE-2026-12087` perl) are already reviewed, so a rebuild inherits them and needs **no admin
  action**. Note `config/image-exceptions.yaml` has **two** lists — `exceptions: []` is empty, and
  stopping there would wrongly suggest an admin is required; the populated list is
  `reviewed_vulnerabilities`.
