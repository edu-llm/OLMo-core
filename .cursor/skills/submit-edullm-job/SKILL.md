---
name: submit-edullm-job
description: Use when a submitting team explicitly invokes /submit-edullm-job to request an eduLLM job on MIT Engaging.
disable-model-invocation: true
---

# Submit an eduLLM job request

Use this workflow only after the user explicitly invokes `/submit-edullm-job`.
The Skill creates one validated request Issue; it never submits ORCD work.
Issue form is not a substitute for this Skill.

Never request or handle credentials. Never read or print tokens, environment
variables, or authentication material. Use only the user's already configured
`git` and `gh` clients. Never call compute, SSH, ORCD, Slurm, W&B APIs, or an
operator command.

Read the [request reference](request-reference.md) before collecting fields.
Execute the [validation adapter](scripts/validate_request.py); do not replace,
copy, or reimplement its parser, policy, or validation.

## 1. Fail closed on repository and PR identity

Run this exact gate from the repository root. A command error is a failed gate,
not evidence of a clean tree or valid PR.

```bash
set -euo pipefail
exec 3>&2
exec 2>/dev/null
STATUS="$(git status --porcelain=v1)" || exit 2
test -z "$STATUS"
BRANCH="$(git branch --show-current)" || exit 2
test -n "$BRANCH"
test "$BRANCH" != main
COMMIT_SHA="$(git rev-parse HEAD)" || exit 2
test "${#COMMIT_SHA}" -eq 40
case "$COMMIT_SHA" in
  (*[!0-9a-f]*|"") exit 2 ;;
  (*) ;;
esac
PR_JSON="$(gh pr view --json number,state,isDraft,headRefOid,url)" || exit 2
test -n "$PR_JSON"
test "$(gh pr view --json headRefOid --jq .headRefOid)" = "$COMMIT_SHA"
test "$(gh pr view --json isDraft --jq .isDraft)" = false
PR_STATE="$(gh pr view --json state --jq .state)"
case "$PR_STATE" in OPEN|MERGED) ;; *) exit 2 ;; esac
exec 2>&3 3>&-
```

Do not push, commit, switch branches, clean the tree, create a PR, or accept a
short SHA for the user. The gate suppresses raw diagnostics. Stop and explain
the failed precondition with a fixed summary.

Read and report the PR URL, approval state, and check state. Checks do not
approve a request: Actions is authoritative for request validation and
assignment.

## 2. Verify observable metrics

Read the selected script and configuration and trace its W&B callback. List the
metric names actually emitted by the reviewed code. Do not call W&B.

If a requested scientific metric is absent, stop and direct the user to
`/weights-and-biases`. Metric wiring must be committed, reviewed in a new or
updated PR, and pass the exact-SHA gate before restarting this Skill.

## 3. Collect team intent

Ask for one missing intent field at a time. The submitting team owns:

- hypothesis, purpose, study, condition, comparison, and seed
- model, code, configuration, script, launcher, and arguments
- data choice, manifest, mixture, and curriculum
- success signal and scientific metrics

Never invent, select, or alter these inputs. For an engineering generic smoke
only, offer exactly the fixture values in the request reference; the team must
choose it. Do not use that fixture to fill a scientific request.

Build one JSON object with all 19 string fields in the `ISSUE_HEADINGS` order
documented in the request reference. The existing parser, policy, and validator
are the sole schema.

## 4. Validate privately and preview

Keep this shell session open through confirmation and Issue creation. Create
private files first:

```bash
REQUEST_DIR="$(mktemp -d)"
chmod 700 "$REQUEST_DIR"
trap 'rm -rf "$REQUEST_DIR"' EXIT
umask 077
: > "$REQUEST_DIR/request.json"
: > "$REQUEST_DIR/issue.md"
chmod 600 "$REQUEST_DIR/request.json" "$REQUEST_DIR/issue.md"
```

Write the JSON directly to `request.json` with a file-writing operation. Never
put the Issue body, Arguments JSON, tokens, or secrets in command arguments,
shell source, or logs. Do not enable shell tracing.

Run the authoritative adapter:

```bash
REQUESTER="$(gh api user --jq .login 2>/dev/null)" || exit 2
test -n "$REQUESTER"
python .cursor/skills/submit-edullm-job/scripts/validate_request.py \
  --input-json "$REQUEST_DIR/request.json" \
  --requester "$REQUESTER" \
  > "$REQUEST_DIR/issue.md"
```

On adapter failure, stop and show only its sanitized validation messages. On
any `git` or `gh` subprocess failure, report the failed step with a fixed
summary; do not replay raw arguments, environment data, stdout, or stderr.

Read `issue.md` using a file-reading operation. Show its exact contents and a
concise request summary, including the full commit SHA, PR URL, script, data
manifest, seed, metrics, GPU request, and runtime. Ask for explicit confirmation
to create the request Issue. Urgency never replaces confirmation.

If the user changes any field, update `request.json`, rerun the adapter, replace
the preview, and ask again. Never continue from a stale preview.

## 5. Create only the confirmed request Issue

After explicit confirmation, use the same shell session and the exact validated
file. `STUDY` and `CONDITION` are the exact validated field values already held
by the Skill. The quoted title remains one argument.

Do not edit, re-render, copy, or transform `issue.md` after validation.

```bash
ISSUE_URL="$(gh issue create \
  --repo edu-llm/OLMo-core \
  --title "[eduLLM job]: ${STUDY}-${CONDITION}" \
  --label edullm-job \
  --label status:requested \
  --body-file "$REQUEST_DIR/issue.md" 2>/dev/null)" || exit 2
gh issue view "$ISSUE_URL" \
  --repo edu-llm/OLMo-core \
  --json number,url,labels,assignees,comments 2>/dev/null || exit 2
```

Never fall back to direct execution when Issue creation or validation is slow
or fails. Delete the private directory after the final read; the trap must also
clean it on every error, refusal, or interruption.

## 6. Report Actions state

Read the created Issue's labels, assignees, and machine-maintained validation
comment. Report only the state Actions recorded:

- `requested` while validation has not completed
- validation errors when Actions rejected the request
- `ready` when Actions recorded readiness without an assignee
- the assigned operator only when Actions recorded that assignee

Never claim acceptance, readiness, or assignment from the initial labels, local
validation, PR checks, elapsed time, or inference. The request Issue is the end
of this Skill's authority.
