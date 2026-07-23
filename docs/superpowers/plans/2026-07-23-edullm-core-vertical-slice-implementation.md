# eduLLM Core Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and prove the real eduLLM path from `/submit-edullm-job` through GitHub Actions, one Slack assignment, `edullm run`, one ORCD L40S generic smoke, W&B metrics, and terminal GitHub Issue reconciliation.

**Architecture:** Continue from the completed Plan 1 and Plan 2 Tasks 1–6, close the existing Task 7 implementation with one range-scoped review, then add the real repository Skill and only the integration glue required by the approved core path. Keep GitHub Actions free of compute and experiment credentials; the enabled operator's local CLI performs fresh exact-SHA validation and the single idempotent Slurm submission, while `edullm jobs` performs initial terminal reconciliation.

**Tech Stack:** Python 3.10+, pytest, mypy, Ruff, Black, isort, PyYAML, GitHub CLI/API/Actions, Cursor project Skills, SSH ControlMaster, Slurm (`sbatch`, `squeue`, `sacct`, `scancel`), and Weights & Biases.

## Global Constraints

The following project-wide requirements are copied verbatim from the approved scope amendment and apply to every task:

- The product must complete one real request-to-running-job path. It is not an Issue-form-only workflow, local-automation endpoint, or other stopgap.
- The `edullm` CLI with `setup`, `run`, `jobs`, `logs`, `stop`, and `logout`.
- The real `/submit-edullm-job` Skill.
- The Issue form and structured request schema.
- Exact-head-SHA review and CI gating.
- GitHub Actions request validation and assignment.
- One enabled operator and one allowlisted team lead.
- Basic Slack assignment notification.
- ORCD generic-smoke submission through the operator's SSH ControlMaster.
- A real W&B run URL and training metrics.
- GitHub Issue lifecycle state.
- Operator-side Slurm-to-Issue reconciliation through `edullm jobs`.
- Documentation and a user-side setup checklist.
- One real one-L40S generic-smoke request completes end-to-end with W&B metrics and Issue terminal state.
The following values are explicitly deferred:

- All Plan 3 Skill-DAG and curriculum work.
- Team or data-mixture design. Submitting research and data teams own those inputs in their PRs.
- S3 rollout.
- Apptainer.
- Operators two and three, and multi-user acceptance.
- A scheduled W&B monitor credential and workflow.
- Advanced Slack reminders, reassignment, and terminal-event threads.
- Strict repository ruleset enforcement.
- Extensive failure drills and rollout polish.
- Contributors use branches and pull requests. Direct-main SHAs are ineligible.
- Software verifies the exact approved PR head SHA and passing CI before assignment and again before submission.
- Current repository writers are trusted during the pilot until strict rulesets are added.
- GitHub Actions receive no ORCD, SSH, Kerberos, W&B, or S3 credentials. The Slack webhook is only for the scoped assignment notification and grants no compute or experiment access.
- Production remains fail-closed until the user supplies one operator identity, one allowlisted team-lead identity, and the Slack webhook, and a separately reviewed enablement change removes the literal workflow disables.
- No current-scope change may enable workflows, add production identities or secrets, or place operator credentials in GitHub.
- During iteration, run only focused tests for touched behavior and changed-file lint/type checks.
- Build remaining core behavior in integrated slices; do not rerun 900-plus tests after each micro-fix.
- Use one independent review per integrated remaining slice. Fix Critical and Important correctness or security findings; defer Minor polish.
- Run one comprehensive local gate after the whole vertical slice is assembled, then one final whole-branch review.
- Perform one user-assisted live generic-smoke acceptance, followed by iterative fixes based on real results.
- No credential leakage.
- Exact approved SHA.
- Shell-safe submission.
- No duplicate `sbatch`.

Those nine deferred bullets are exclusions, not work items. In particular, original Plan 2 Task 8's scheduled W&B monitor (`src/edullm/wandb_status.py`, `.github/workflows/edullm-wandb-reconcile.yml`, and `EDULLM_WANDB_MONITOR_KEY`) and all Plan 3 work are deferred. Initial terminal reconciliation is provided by `edullm jobs`.

Trust `.superpowers/sdd/progress.md` as the durable ledger:

- Plan 1 and Plan 2 Tasks 1–6 are complete and must never be redispatched.
- Plan 2 Task 7 already exists in `c47b7de7ae073721942af0bdfc1626fc1ceff92f`, with verification corrections in `d8eec13`; it is reviewed and repaired, not reimplemented.
- The Task 7 review range is `b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13`.
- The approved scope amendment is `05ed8faeca230e9e92af501207aea5cb18e8665b`.
- Workflows remain literally disabled until Task 4's separate activation change is reviewed and the user completes its external gates.

---

## Exact File Map and Responsibilities

Files created by this continuation:

- `.cursor/skills/submit-edullm-job/SKILL.md` — the actual agent-facing submission workflow, including local Git/PR checks, one-question-at-a-time collection, preview/confirmation, structured Issue creation, and status reporting.
- `.cursor/skills/submit-edullm-job/request-reference.md` — exact field semantics, safe generic-smoke defaults, scientific-input ownership, and expected Issue/assignment states.
- `.cursor/skills/submit-edullm-job/scripts/validate_request.py` — credential-free adapter from Skill field JSON to the existing parser, `JobRequest`, policy, and validator; emits the exact Issue Markdown that is subsequently submitted.
- `src/test/edullm/skill_test.py` — Skill metadata, safety text, adapter behavior, parser/schema reuse, and supported `gh issue create --body-file` contract.
- `docs/source/guides/edullm_engaging.rst` — concise teammate, team-lead, and operator path plus status and failure guidance.
- `docs/source/guides/edullm_activation.rst` — user/admin activation checklist, exact external stop points, credential boundary, labels, and live evidence checklist.
- `docs/superpowers/reports/2026-07-23-edullm-core-acceptance.md` — tracked, redacted evidence for the comprehensive local gate, final review, and live generic-smoke acceptance.

Existing files modified by this continuation:

- `src/edullm/request_parser.py` — add the sole shared renderer from exact Issue headings to Markdown; parsing remains authoritative.
- `src/test/edullm/request_parser_test.py` — prove render/parse equivalence and exact shape rejection.
- `.github/workflows/edullm-assign.yml` — make the reusable Slack secret contract required while retaining the literal disable until activation.
- `.github/workflows/main.yml` — add one named eduLLM/ORCD CPU test matrix entry and avoid duplicate execution in broad CPU entries.
- `config/edullm/policy.yaml` — require the named eduLLM core CI check for exact-SHA approval.
- `config/edullm/main-ruleset.json` — keep the already-tracked static required-check list consistent; this does not apply or automate a ruleset.
- `src/test/edullm/workflow_test.py` — verify handoff, permissions, concurrency, secret scoping, core-only activation, and deferred-workflow disablement.
- `src/test/edullm/task_4_config_test.py` — keep Issue form/parser/policy/static ruleset contracts synchronized through pre-activation and activation states.
- `src/test/edullm/task_5_config_test.py` — verify the required reusable secret contract and assignment workflow boundaries.
- `src/test/edullm/slurm_test.py` — focused W&B identity, exact reviewed SHA, shell-safe argv, and no-secret assertions.
- `src/test/edullm/jobs_test.py` — focused oldest-assigned selection, exact W&B lifecycle identity, terminal `edullm jobs` repair, and one-submit assertions.
- `src/test/edullm/cli_test.py` — public command wiring and command-as-acceptance assertions.
- `docs/source/index.rst` — link the teammate/operator and activation guides.
- `config/edullm/team-leads.yaml` — activation-only one-element allowlist, populated solely from the user's supplied GitHub login.
- `config/edullm/operators.yaml` — activation-only one enabled operator, populated solely from the user's supplied GitHub and Slack identities.
- `.github/workflows/edullm-validate.yml` — activation-only removal of the two core literal disables: validation and reusable assignment handoff.

Ignored local bookkeeping artifacts updated but never staged or committed:

- `.superpowers/sdd/plan-2-task-7-report.md` — append the Task 7 independent review outcome and any task-scoped repair evidence.
- `.superpowers/sdd/progress.md` — mark Task 7 complete and record the remaining-slice commits/reviews.

Task 7's existing implementation files are review inputs, not planned rewrites:

- `.github/workflows/edullm-terminal-notify.yml`
- `config/edullm/entrypoints.yaml`
- `config/edullm/policy.yaml`
- `src/edullm/{cli,data_manifest,github,jobs,models,notifications,policy,slurm,ssh,ssh_helper,validation}.py`
- `src/test/edullm/{cli,conftest,data_manifest,github_issue,jobs,notifications,slurm,validation,workflow}_test.py`
- `src/test/edullm/fixtures/valid_issue.md`

Files intentionally not created or activated:

- `src/edullm/wandb_status.py`
- `src/test/edullm/wandb_status_test.py`
- `.github/workflows/edullm-wandb-reconcile.yml`
- Any Plan 3, S3, Apptainer, additional-operator, reminder/reassignment, or terminal-Slack implementation

---

### Task 1: Close Existing Plan 2 Task 7

**Files:**
- Review: the exact existing range `b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13`
- Read: `docs/superpowers/specs/2026-07-23-edullm-core-vertical-slice-scope.md`
- Read: `.superpowers/sdd/plan-2-task-7-report.md`
- Modify locally, never stage: `.superpowers/sdd/plan-2-task-7-report.md`
- Modify locally, never stage: `.superpowers/sdd/progress.md`
- Conditional repair: only the source/test pair named by a Critical or Important finding inside the Task 7 changed-file set listed above

**Interfaces:**
- Consumes: Task 7 commits `c47b7de` and `d8eec13`, Task 6 head `b222f47`, the existing Task 7 brief/report, and the approved scope.
- Produces: one independent task review with the standard `### Spec Compliance` verdict and `**Task quality:** [Approved | Needs fixes]` assessment, plus focused repair commits if required.
- Preserves: `run_assigned(...) -> LifecycleState`, `jobs(...) -> tuple[LifecycleState, ...]`, `logs(...) -> str`, `stop(...) -> LifecycleState`, `render_sbatch(ResolvedRequest) -> str`, and `submission_transaction(...) -> SubmissionReceipt`.

- [ ] **Step 1: Confirm the range and generate one review artifact**

Run:

```bash
set -euo pipefail
STATUS="$(git status --porcelain=v1)" || exit 2
test -z "$STATUS"
git merge-base --is-ancestor d8eec13a90f336a08f9c5e7f8988be98417d04fb HEAD
git merge-base --is-ancestor 05ed8faeca230e9e92af501207aea5cb18e8665b HEAD
git rev-parse HEAD > /tmp/edullm-task-1-start-head
TASK1_START_HEAD="$(cat /tmp/edullm-task-1-start-head)" || exit 2
git diff --find-renames --stat b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13a90f336a08f9c5e7f8988be98417d04fb
git diff --find-renames --check b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13a90f336a08f9c5e7f8988be98417d04fb
TASK7_REVIEW_PACKAGE=/tmp/edullm-plan-2-task-7-review.txt
{
  printf '%s\n' '# Commit list'
  git log --oneline b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13a90f336a08f9c5e7f8988be98417d04fb
  printf '%s\n' '# Stat'
  git diff --find-renames --stat b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13a90f336a08f9c5e7f8988be98417d04fb
  printf '%s\n' '# Full diff'
  git diff --find-renames -U10 b222f47862ea0549449f1f0e4b694f5437e81324..d8eec13a90f336a08f9c5e7f8988be98417d04fb
} > "$TASK7_REVIEW_PACKAGE"
```

Expected: clean status; both ancestry checks exit 0; `TASK1_START_HEAD` records the plan/correction head before any Task 1 repair; the stat is `24 files changed, 5650 insertions(+), 60 deletions(-)`; no whitespace errors; the review package contains the Task 7 commit list, stat, and full diff only.

- [ ] **Step 2: Run one task-scoped independent review**

Use the subagent-driven-development task-reviewer template once, with the generated Task 1 brief, `.superpowers/sdd/plan-2-task-7-report.md` as the implementer report, base `b222f47862ea0549449f1f0e4b694f5437e81324`, head `d8eec13a90f336a08f9c5e7f8988be98417d04fb`, and `/tmp/edullm-plan-2-task-7-review.txt` as the diff file. The review prompt must retain the template's read-only rule, “Do Not Trust the Report” section, focused-test policy, calibration, and output format.

Use this exact task-specific context in the template:

```text
This is the single task-scoped review for the existing Plan 2 Task 7 range,
not the final integrated review. Review spec compliance first and task quality
second. The binding guarantees are no credential leakage, exact approved SHA
at both submission gates, shell-safe submission, and at-most-one sbatch. The
approved scope defines scheduled W&B monitoring, Plan 3, S3, Apptainer,
multiple operators, advanced Slack behavior, strict ruleset automation, and
broad rollout polish as deferred non-requirements for this task.
```

Require these standard report sections exactly:

```text
### Spec Compliance
- ✅ Spec compliant | ❌ Issues found
- ⚠️ Cannot verify from diff
### Strengths
### Issues
#### Critical (Must Fix)
#### Important (Should Fix)
#### Minor (Nice to Have)
### Assessment
**Task quality:** Approved | Needs fixes
**Reasoning:**
```

Expected: one task-scoped review result, not multiple blanket reviews. Task 7 closes only when the spec-compliance verdict is `✅ Spec compliant` and task quality is `Approved`, with every `⚠️ Cannot verify from diff` item resolved by the controller.

- [ ] **Step 3: Repair only Critical or Important findings test-first**

For each accepted finding, first add one regression to the covering file, set `REPAIR_TEST_FILE` and `REPAIR_TEST_NAME` to that concrete file/function pair, run exactly that test to observe the defect, then make the smallest source change:

```bash
case "$REPAIR_TEST_FILE" in
  src/test/edullm/slurm_test.py|\
  src/test/edullm/jobs_test.py|\
  src/test/edullm/data_manifest_test.py|\
  src/test/edullm/cli_test.py|\
  src/test/edullm/workflow_test.py) ;;
  *) exit 2 ;;
esac
test -n "$REPAIR_TEST_NAME"
pytest -v "${REPAIR_TEST_FILE}::${REPAIR_TEST_NAME}"
```

Expected before repair: the newly added named regression fails for the reviewer-described reason. `REPAIR_TEST_FILE` is one of the five allowlisted covering files and `REPAIR_TEST_NAME` is the exact newly added function, not a broad keyword selection.

Minor findings are recorded under `## Deferred Minor Triage` in the Task 7 report and are not repaired during this gate.

- [ ] **Step 4: Run covering tests and changed-file checks after any repair**

Run only the source/test pair selected in Step 3, followed by:

```bash
TASK1_START_HEAD="$(cat /tmp/edullm-task-1-start-head)" || exit 2
pytest -v "${REPAIR_TEST_FILE}::${REPAIR_TEST_NAME}"
CHANGED_PYTHON="$(git diff --name-only "$TASK1_START_HEAD" -- '*.py')"
test -n "$CHANGED_PYTHON"
python -m ruff check $CHANGED_PYTHON
python -m black --check $CHANGED_PYTHON
python -m isort --check-only $CHANGED_PYTHON
CHANGED_PRODUCTION="$(git diff --name-only "$TASK1_START_HEAD" -- 'src/edullm/*.py')"
if test -n "$CHANGED_PRODUCTION"; then
  python -m mypy $CHANGED_PRODUCTION
fi
git diff --check
```

Expected: the named regression passes; each changed-file command receives only Task 1 repair paths after `TASK1_START_HEAD` and exits 0.

- [ ] **Step 5: Commit and re-review only a required repair**

When a repair exists, commit only the focused source/test change:

```bash
TASK1_START_HEAD="$(cat /tmp/edullm-task-1-start-head)" || exit 2
git add -u src/edullm src/test/edullm .github/workflows config/edullm
git commit -m "fix: close eduLLM Task 7 review findings"
git diff --check "$TASK1_START_HEAD"..HEAD
TASK7_REPAIR_PACKAGE=/tmp/edullm-plan-2-task-7-repair-review.txt
{
  printf '%s\n' '# Commit list'
  git log --oneline "$TASK1_START_HEAD"..HEAD
  printf '%s\n' '# Stat'
  git diff --find-renames --stat "$TASK1_START_HEAD"..HEAD
  printf '%s\n' '# Full diff'
  git diff --find-renames -U10 "$TASK1_START_HEAD"..HEAD
} > "$TASK7_REPAIR_PACKAGE"
```

Pass `/tmp/edullm-plan-2-task-7-repair-review.txt` to the same independent task-review gate. Expected: one focused repair commit and a package containing only its commit list, stat, and diff. The re-review returns `✅ Spec compliant` and `**Task quality:** Approved`. If the first review approved with no Critical/Important findings, skip this step and create no commit.

- [ ] **Step 6: Record closure in the durable ledger**

Append the exact review decisions, reviewer findings, focused commands/results, and repair commit (or `none`) to the ignored local `.superpowers/sdd/plan-2-task-7-report.md`. Replace the current Task 7 line in ignored local `.superpowers/sdd/progress.md` with:

```text
Plan 2 Task 7: complete (commits b222f47..d8eec13 plus focused review repairs if any; one task-scoped independent review approved spec and quality).
```

Expected: both ignored scratch artifacts are updated in the same bookkeeping step, remain unstaged, and preserve Plan 1 and Plan 2 Tasks 1–6 unchanged. No completed task is redispatched.

- [ ] **Step 7: Confirm ignored bookkeeping remains local**

```bash
TASK1_START_HEAD="$(cat /tmp/edullm-task-1-start-head)" || exit 2
git check-ignore -q .superpowers/sdd/progress.md
git check-ignore -q .superpowers/sdd/plan-2-task-7-report.md
git diff --cached --quiet
```

Expected: both bookkeeping files are ignored and the index is empty. Task 1 has one focused repair commit at most; when no repair exists, `HEAD` still equals `TASK1_START_HEAD` and no closure commit is made. `.superpowers/sdd/progress.md` and `.superpowers/sdd/plan-2-task-7-report.md` are never staged or committed.

---

### Task 2: Build the Real `/submit-edullm-job` Skill

**Files:**
- Create: `.cursor/skills/submit-edullm-job/SKILL.md`
- Create: `.cursor/skills/submit-edullm-job/request-reference.md`
- Create: `.cursor/skills/submit-edullm-job/scripts/validate_request.py`
- Create: `src/test/edullm/skill_test.py`
- Modify: `src/edullm/request_parser.py`
- Modify: `src/test/edullm/request_parser_test.py`

**Interfaces:**
- Consumes: `ISSUE_HEADINGS`, `parse_issue(body: str, *, issue_number: int, requester: str) -> JobRequest`, `load_policy(path: Path, entrypoints_path: Path | None = None) -> Policy`, and `validate_request(request: JobRequest, policy: Policy) -> list[str]`.
- Produces: `issue_body_from_fields(fields: Mapping[str, str]) -> str` and `validate_submission(input_path: Path, *, requester: str, policy_path: Path, entrypoints_path: Path) -> str`.
- External boundary: the Skill invokes configured `git` and `gh` commands; the adapter itself never imports a GitHub, SSH, Slurm, W&B, or secrets client.
- Output: the exact validated Markdown file passed unchanged to `gh issue create --body-file`.

- [ ] **Step 1: Write failing shared-renderer tests**

Add:

```python
from edullm.request_parser import (
    IssueParseError,
    fields_from_markdown,
    issue_body_from_fields,
    parse_issue,
)


def test_issue_body_renderer_round_trips_the_authoritative_parser():
    original = FIXTURE.read_text(encoding="utf-8")
    fields = fields_from_markdown(original)
    rendered = issue_body_from_fields(fields)

    assert fields_from_markdown(rendered) == fields
    assert parse_issue(
        rendered, issue_number=42, requester="student"
    ).canonical_json() == parse_issue(
        original, issue_number=42, requester="student"
    ).canonical_json()


def test_issue_body_renderer_rejects_missing_extra_and_non_string_fields():
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(IssueParseError, match="missing heading: Purpose"):
        issue_body_from_fields({key: value for key, value in fields.items() if key != "Purpose"})
    with pytest.raises(IssueParseError, match="unexpected heading at index"):
        issue_body_from_fields({**fields, "Shell command": "sbatch anything"})
    with pytest.raises(IssueParseError, match="GPU count must be text"):
        issue_body_from_fields({**fields, "GPU count": 1})  # type: ignore[dict-item]
```

Run:

```bash
pytest -v src/test/edullm/request_parser_test.py \
  -k 'issue_body_renderer'
```

Expected: collection fails because `issue_body_from_fields` does not exist.

- [ ] **Step 2: Implement the single shared Issue renderer**

Add to `src/edullm/request_parser.py`:

```python
from collections.abc import Mapping


def issue_body_from_fields(fields: Mapping[str, str]) -> str:
    """Render the exact Issue-form Markdown accepted by :func:`parse_issue`."""
    if not isinstance(fields, Mapping):
        raise IssueParseError(("Issue fields must be a mapping",))
    names = list(fields)
    errors = _shape_errors(names)
    expected = set(ISSUE_HEADINGS)
    for heading in ISSUE_HEADINGS:
        if heading in fields and type(fields[heading]) is not str:
            errors.append(f"{heading} must be text")
    if errors:
        raise IssueParseError(tuple(errors))
    typed = cast(Mapping[str, str], fields)
    return "\n\n".join(f"### {heading}\n{typed[heading]}" for heading in ISSUE_HEADINGS)
```

Keep `parse_issue()` authoritative; do not add a second request dataclass or second policy schema.

Run:

```bash
pytest -v src/test/edullm/request_parser_test.py \
  -k 'issue_body_renderer'
python -m ruff check src/edullm/request_parser.py src/test/edullm/request_parser_test.py
python -m mypy src/edullm/request_parser.py
```

Expected: the renderer tests and changed-file checks pass.

- [ ] **Step 3: Write failing Skill and adapter tests**

Create `src/test/edullm/skill_test.py` with these concrete contracts:

```python
import json
import subprocess
import sys
from pathlib import Path

import yaml

from edullm.request_parser import fields_from_markdown

SKILL = Path(".cursor/skills/submit-edullm-job")
FIXTURE = Path("src/test/edullm/fixtures/valid_issue.md")


def test_submission_skill_is_real_agent_facing_workflow():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["name"] == "submit-edullm-job"
    assert "MIT Engaging" in frontmatter["description"]
    assert "/submit-edullm-job" in text
    assert "gh issue create" in text
    assert "--body-file" in text
    assert "explicit confirmation" in text
    assert "Issue form is not a substitute" in text
    assert "Never request or handle credentials" in text
    assert "ssh orcd-login" not in text
    assert "edullm run" not in text
    assert "sbatch " not in text
    assert 'STATUS="$(git status --porcelain=v1)" || exit 2' in text
    assert 'test -z "$STATUS"' in text
    assert 'test -z "$(git status' not in text
    assert len(text.splitlines()) < 500


def test_adapter_reuses_parser_policy_and_emits_exact_validated_body(tmp_path):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts/validate_request.py"),
            "--input-json",
            str(source),
            "--requester",
            "student",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert fields_from_markdown(result.stdout) == fields
    assert result.stderr == ""


def test_adapter_rejects_unsafe_argv_without_echoing_credentials(tmp_path):
    fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    fields["Arguments JSON"] = '["train_single", "x; env"]'
    source = tmp_path / "request.json"
    source.write_text(json.dumps(fields), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL / "scripts/validate_request.py"),
            "--input-json",
            str(source),
            "--requester",
            "student",
        ],
        check=False,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src", "WANDB_API_KEY": "must-not-appear"},
    )
    assert result.returncode == 2
    assert "unsafe argument" in result.stderr
    assert "must-not-appear" not in result.stdout + result.stderr
```

Run:

```bash
pytest -v src/test/edullm/skill_test.py
```

Expected: FAIL because the Skill files and adapter do not exist.

- [ ] **Step 4: Implement the credential-free adapter**

Create `.cursor/skills/submit-edullm-job/scripts/validate_request.py` with:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from edullm.policy import load_policy
from edullm.request_parser import (
    IssueParseError,
    issue_body_from_fields,
    parse_issue,
)
from edullm.validation import validate_request


def validate_submission(
    input_path: Path,
    *,
    requester: str,
    policy_path: Path,
    entrypoints_path: Path,
) -> str:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise ValueError("request JSON must be an object")
    fields = cast(dict[str, object], document)
    if any(type(key) is not str or type(value) is not str for key, value in fields.items()):
        raise ValueError("request JSON fields and values must be strings")
    body = issue_body_from_fields(cast(dict[str, str], fields))
    request = parse_issue(body, issue_number=1, requester=requester)
    errors = validate_request(
        request,
        load_policy(policy_path, entrypoints_path),
    )
    if errors:
        raise ValueError("\n".join(errors))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--requester", required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("config/edullm/policy.yaml"),
    )
    parser.add_argument(
        "--entrypoints",
        type=Path,
        default=Path("config/edullm/entrypoints.yaml"),
    )
    arguments = parser.parse_args()
    try:
        body = validate_submission(
            arguments.input_json,
            requester=arguments.requester,
            policy_path=arguments.policy,
            entrypoints_path=arguments.entrypoints,
        )
    except (IssueParseError, OSError, UnicodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.write(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The script receives a public GitHub login only. It does not read environment variables, tokens, W&B, SSH, Slurm, or webhook values.

- [ ] **Step 5: Write the Skill's exact local and GitHub workflow**

Use this ordered behavior in `.cursor/skills/submit-edullm-job/SKILL.md`:

1. Require a clean non-`main` branch and a 40-character lowercase `git rev-parse HEAD`.
2. Use `gh pr view --json number,state,isDraft,headRefOid,url` to require an open or merged, non-draft PR whose `headRefOid` equals that full SHA. Report approval/check state, but treat Actions as authoritative.
3. Read the selected script/config and W&B callback; list metrics actually emitted. If a requested scientific metric is absent, stop and direct the user to `/weights-and-biases`, then require a new commit/PR approval.
4. Ask one missing intent field at a time. The submitting team owns code, config, hypothesis, comparison, condition, seed, data choice/manifest/mix, and scientific metrics.
5. For an engineering generic smoke only, offer one L40S and 30 minutes; never invent scientific inputs or alter a team's data mix.
6. Build the exact 19-field JSON described by `request-reference.md`.
7. Create mode-`0600` temporary files under a mode-`0700` temporary directory, run the adapter, show the exact body and request summary, and ask for explicit confirmation.
8. After confirmation, pass the validated body unchanged to the supported `gh issue create --body-file` path with `edullm-job` and `status:requested`.
9. Read the created Issue's labels and assignee plus the validation workflow status; report `requested`, validation errors, `ready`, or assigned operator. Never claim assignment before Actions records it.
10. Delete the temporary directory in a trap. Never call `edullm`, SSH, ORCD, Slurm, W&B APIs, or any credential command.

Include these exact command shapes:

```bash
set -euo pipefail
STATUS="$(git status --porcelain=v1)" || exit 2
test -z "$STATUS"
BRANCH="$(git branch --show-current)" || exit 2
test -n "$BRANCH"
test "$BRANCH" != main
COMMIT_SHA="$(git rev-parse HEAD)" || exit 2
test "${#COMMIT_SHA}" -eq 40
test "$(gh pr view --json headRefOid --jq .headRefOid)" = "$COMMIT_SHA"
test "$(gh pr view --json isDraft --jq .isDraft)" = false
PR_STATE="$(gh pr view --json state --jq .state)"
case "$PR_STATE" in OPEN|MERGED) ;; *) exit 2 ;; esac

REQUEST_DIR="$(mktemp -d)"
chmod 700 "$REQUEST_DIR"
trap 'rm -rf "$REQUEST_DIR"' EXIT
umask 077
python .cursor/skills/submit-edullm-job/scripts/validate_request.py \
  --input-json "$REQUEST_DIR/request.json" \
  --requester "$(gh api user --jq .login)" \
  > "$REQUEST_DIR/issue.md"

ISSUE_URL="$(gh issue create \
  --repo edu-llm/OLMo-core \
  --title "[eduLLM job]: ${STUDY}-${CONDITION}" \
  --label edullm-job \
  --label status:requested \
  --body-file "$REQUEST_DIR/issue.md")"
gh issue view "$ISSUE_URL" \
  --repo edu-llm/OLMo-core \
  --json number,url,labels,assignees,comments
```

`STUDY` and `CONDITION` are the exact validated field values held by the Skill, not shell text copied from the Issue. They remain quoted as one title argument.

State explicitly: “Issue form is not a substitute for this Skill.” The Skill creates a request Issue; it never submits ORCD work.

- [ ] **Step 6: Write the exact request reference**

In `request-reference.md`, document all headings in `ISSUE_HEADINGS` order, with:

- `Commit SHA`: exact full PR head SHA; direct-main SHA rejected.
- `Arguments JSON`: ordered JSON array of strings, never a shell command.
- `Data manifest`: `builtin://generic-smoke-v1` or reviewed `/orcd/pool/...`; no S3 URI.
- `Success metrics`: names emitted by reviewed code.
- Generic smoke: profile `generic-smoke`, script `src/examples/llm/train.py`, launcher `torchrun`, arguments JSON `["orcd-bootstrap"]`, one L40S, 30 minutes, W&B project `test`, manifest `builtin://generic-smoke-v1`, and manifest digest `1c82abfc35b17e8a15eae8e0e1afa3dee6696aeb213d46799f204e1c4fc093d7`.
- Scientific ownership: the Skill does not select hypotheses, conditions, comparisons, data mixtures, curricula, or metrics.
- Status reporting: validation and assignment are performed by Actions; `edullm run` is operator-only.

- [ ] **Step 7: Run focused Skill checks**

Run:

```bash
pytest -v \
  src/test/edullm/request_parser_test.py \
  src/test/edullm/skill_test.py
python -m ruff check \
  src/edullm/request_parser.py \
  src/test/edullm/request_parser_test.py \
  src/test/edullm/skill_test.py \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m black --check \
  src/edullm/request_parser.py \
  src/test/edullm/request_parser_test.py \
  src/test/edullm/skill_test.py \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m isort --check-only \
  src/edullm/request_parser.py \
  src/test/edullm/request_parser_test.py \
  src/test/edullm/skill_test.py \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m mypy \
  src/edullm/request_parser.py \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
git diff --check
```

Expected: all focused tests and changed-file checks pass; no service is contacted.

- [ ] **Step 8: Commit in two reviewable units**

```bash
git add src/edullm/request_parser.py src/test/edullm/request_parser_test.py
git commit -m "feat: render canonical eduLLM Issue requests"

git add .cursor/skills/submit-edullm-job src/test/edullm/skill_test.py
git commit -m "feat: add eduLLM submission Skill"
```

Expected: the first commit owns the shared schema adapter; the second owns the real Skill.

- [ ] **Step 9: Run one independent Skill review**

Review from the Task 1 closure head through the second Task 2 commit. Require separate spec and quality decisions and prioritize: unchanged validated body reaches `gh issue create`, no credential access, full PR head SHA, no shell string execution, and no team-owned scientific input invention.

Expected: fix only Critical/Important findings with one new focused failing test per fix; record Minor findings for final triage; do not dispatch another review after approval.

---

### Task 3: Complete Core Integration and Readiness Glue

**Files:**
- Modify: `.github/workflows/edullm-assign.yml`
- Modify: `.github/workflows/main.yml`
- Modify: `config/edullm/policy.yaml`
- Modify: `config/edullm/main-ruleset.json`
- Modify: `src/test/edullm/workflow_test.py`
- Modify: `src/test/edullm/task_4_config_test.py`
- Modify: `src/test/edullm/task_5_config_test.py`
- Modify: `src/test/edullm/slurm_test.py`
- Modify: `src/test/edullm/jobs_test.py`
- Modify: `src/test/edullm/cli_test.py`
- Create: `docs/source/guides/edullm_engaging.rst`
- Modify: `docs/source/index.rst`

**Interfaces:**
- Consumes: the Skill body from Task 2; `automation_validate(...) -> AutomationResult`; `automation_assign(...) -> tuple[AssignmentResult, ...]`; `handle_run() -> int`; `handle_jobs(mine: bool) -> int`; and Task 7's idempotent submission/lifecycle interfaces.
- Produces: one named `Test eduLLM core` CI result, a required reusable Slack secret contract, static proof of Skill → Issue → validate → assign handoff, and concise teammate/operator documentation.
- Preserves: literal workflow disablement in this task. Activation is Task 4 only.
- Reconciliation: terminal state comes from `squeue`/`sacct` through `edullm jobs`; no scheduled W&B monitor is added.

- [ ] **Step 1: Write failing workflow and CI contract tests**

Add assertions:

```python
def test_core_assignment_handoff_requires_only_the_slack_secret():
    assign = _load(WORKFLOWS / "edullm-assign.yml")
    validate = _load(WORKFLOWS / "edullm-validate.yml")
    assert _trigger(assign)["workflow_call"]["secrets"] == {
        "SLACK_WEBHOOK_URL": {"required": True}
    }
    assert validate["jobs"]["assign"]["needs"] == "validate"
    assert validate["jobs"]["assign"]["uses"] == "./.github/workflows/edullm-assign.yml"
    assert validate["jobs"]["assign"]["secrets"] == {
        "SLACK_WEBHOOK_URL": "${{ secrets.SLACK_WEBHOOK_URL }}"
    }


def test_main_ci_has_one_required_edullm_core_check():
    workflow = _load(Path(".github/workflows/main.yml"))
    names = [row["name"] for row in workflow["jobs"]["checks"]["strategy"]["matrix"]["task"]]
    assert names.count("Test eduLLM core") == 1
    policy = load_policy(Path("config/edullm/policy.yaml"))
    assert policy.required_checks.count("Test eduLLM core") == 1
```

Run:

```bash
pytest -v \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  -k 'core_assignment_handoff or main_ci'
```

Expected: FAIL because the reusable secret is optional and the named CI check is absent.

- [ ] **Step 2: Add the minimal named CI hook and exact-SHA requirement**

In `.github/workflows/main.yml`, add this matrix entry:

```yaml
- name: Test eduLLM core
  run: |
    pytest -v --color=yes --durations=3 \
      src/test/edullm/ \
      src/test/scripts/orcd/
```

Exclude `src/test/edullm/**` from the broad `Test` entry and `src/test/scripts/orcd/**` from `Test scripts` so each test runs in exactly one CI job. Add `Test eduLLM core` once to `config/edullm/policy.yaml` and once to the existing static `required_status_checks` list in `config/edullm/main-ruleset.json`.

The two existing matrix commands become:

```yaml
- name: Test
  run: |
    pytest -v --color=yes --durations=6 \
      --ignore-glob='src/test/edullm/**' \
      --ignore-glob='src/test/distributed/checkpoint*' \
      --ignore-glob='src/test/train/checkpoint*' \
      --ignore-glob='src/test/nn/transformer*' \
      --ignore-glob='src/test/nn/attention*' \
      --ignore-glob='src/test/examples/*' \
      --ignore-glob='src/test/scripts/*' \
      src/test/

- name: Test scripts
  run: |
    pytest -v --color=yes --durations=3 \
      --ignore-glob='src/test/scripts/orcd/**' \
      src/test/scripts/
```

Change only this reusable declaration in `.github/workflows/edullm-assign.yml`:

```yaml
on:
  workflow_call:
    secrets:
      SLACK_WEBHOOK_URL:
        required: true
```

Retain all three core literal disables and every deferred workflow disable.

- [ ] **Step 3: Add focused non-negotiable path assertions**

Extend existing tests with:

```python
def test_sbatch_binds_exact_sha_wandb_identity_and_safe_argv(valid_resolved_request):
    text = render_sbatch(valid_resolved_request)
    request = valid_resolved_request.request
    assert f"git fetch --no-tags origin {request.commit_sha}" in text
    assert f'test "$(git rev-parse HEAD)" = {request.commit_sha}' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_PREFIX}-${SLURM_JOB_ID}"' in text
    assert (
        'WANDB_RUN_URL="https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}'
        '/runs/${WANDB_RUN_ID}"'
    ) in text
    assert request.digest in text
    assert "eval " not in text
    assert "WANDB_API_KEY" not in text
```

Add this concrete extension in `jobs_test.py`:

```python
class ReverseOrderGitHub(StatefulGitHub):
    def list_active_queue_issues(self):
        issue_43 = replace(self.issue, number=43)
        return (issue_43, self.issue)


def test_run_selects_oldest_assigned_issue_and_submits_once(valid_request):
    github = ReverseOrderGitHub(valid_request)
    remote = StatefulRemote()
    state = run_assigned(
        operator="operator",
        github=github,
        load_configuration=_config,
        remote=remote,
        now=NOW,
    )
    assert state.issue == 42
    assert len(remote.staged) == 1
    assert len(remote.verified) == 1
    assert len(remote.submissions) == 1
    assert state.attempts[-1].wandb_run_id == "issue-42-attempt-1-12345"
    assert state.attempts[-1].wandb_url == (
        "https://wandb.ai/eduLLM/pretraining/runs/issue-42-attempt-1-12345"
    )
```

Add `jobs` and `SlurmJob` to the existing imports, then add:

```python
class CompletedSlurm:
    def __init__(self):
        self.queries = []

    def query(self, job_ids):
        self.queries.append(tuple(job_ids))
        return {
            "12345": SlurmJob(
                job_id="12345",
                name="issue-42-skill-dag-v1-natural",
                state="COMPLETED",
                user="operator",
                lifecycle_state="completed",
            )
        }

    def reconcile_offline_tracking(self):
        return None


def test_jobs_repairs_terminal_issue_without_submitting(valid_request):
    lifecycle = replace(
        _lifecycle("running"),
        request_digest=valid_request.digest,
        attempts=(replace(_attempt("running"), request_digest=valid_request.digest),),
    )
    github = StatefulGitHub(valid_request, lifecycle=lifecycle)
    slurm = CompletedSlurm()
    repaired = jobs(
        mine=True,
        operator="operator",
        github=github,
        configuration=_config(),
        slurm=slurm,
        now=NOW,
    )
    assert repaired[0].current_state == "completed"
    assert set(github.issue.labels) == {"edullm-job", "research", "status:completed"}
    assert slurm.queries == [("12345",)]
```

`CompletedSlurm` intentionally has no submission method: terminal repair can only query scheduler evidence.

Extend `cli_test.py` to assert the public parser exposes exactly `setup`, `jobs`, `run`, `logs`, `stop`, and `logout`, while automation commands remain hidden from the public metavar. Assert `handle_run()` calls `run_assigned` directly without a manual confirmation/audit prompt.

Expected: these focused tests prove existing Task 7 behavior. If one fails, first retain the failure as a regression and repair only its named function; do not broaden the slice.

- [ ] **Step 4: Run the focused core integration tests**

```bash
pytest -v \
  src/test/edullm/skill_test.py \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  src/test/edullm/slurm_test.py \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py
```

Expected: PASS. This is the integrated core slice, not the final full eduLLM/ORCD gate.

- [ ] **Step 5: Write concise teammate and operator documentation**

Create `docs/source/guides/edullm_engaging.rst` with these exact sections:

- `Prepare and review`: teammate owns branch, PR, code/config/data/metrics; team lead approves exact current SHA after CI.
- `Submit with /submit-edullm-job`: the real Skill validates/previews/confirms and creates the Issue; manual Issue form use does not satisfy acceptance.
- `Assignment`: Actions validate without compute credentials, assign one operator, and send one Slack assignment.
- `Operate`: `edullm setup`, command-as-acceptance `edullm run`, `edullm jobs [--mine]`, `edullm logs ISSUE`, `edullm stop ISSUE`, and `edullm logout`.
- `Identity and safety`: no shared credentials, no direct-main SHA, no Issue-supplied shell, exact SHA rechecked twice, one idempotent `sbatch`.
- `Status and tracking`: deterministic W&B URL/identity and real training metrics; `edullm jobs` maps Slurm terminal state to the Issue.
- `Deferred`: scheduled W&B monitoring (original Plan 2 Task 8), all Plan 3 work, S3, Apptainer, multi-operator rollout, advanced Slack behavior, strict ruleset automation, and broad polish.

Add `guides/edullm_engaging.rst` to the Guides toctree in `docs/source/index.rst`.

- [ ] **Step 6: Run changed-file checks**

```bash
python -m ruff check \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  src/test/edullm/slurm_test.py \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py
python -m black --check \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  src/test/edullm/slurm_test.py \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py
python -m isort --check-only \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  src/test/edullm/slurm_test.py \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py
python - <<'PY'
import json
from pathlib import Path

import yaml

for path in Path(".github/workflows").glob("edullm-*.yml"):
    assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None
yaml.safe_load(Path("config/edullm/policy.yaml").read_text(encoding="utf-8"))
json.loads(Path("config/edullm/main-ruleset.json").read_text(encoding="utf-8"))
PY
git diff --check
```

Expected: all changed-file and parse checks pass.

- [ ] **Step 7: Commit the integrated slice in reviewable units**

```bash
git add \
  .github/workflows/edullm-assign.yml \
  .github/workflows/main.yml \
  config/edullm/policy.yaml \
  config/edullm/main-ruleset.json \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py
git commit -m "ci: add eduLLM core readiness gate"

git add \
  src/test/edullm/slurm_test.py \
  src/test/edullm/jobs_test.py \
  src/test/edullm/cli_test.py
git commit -m "test: lock eduLLM core integration path"

git add docs/source/guides/edullm_engaging.rst docs/source/index.rst
git commit -m "docs: explain eduLLM Engaging workflow"
```

- [ ] **Step 8: Run one independent integrated-slice review**

Review from the Task 2 approval head through Task 3. Require spec and quality decisions. Check Skill-to-workflow body compatibility, exact required CI name, reusable workflow permissions/concurrency/secret scoping, basic assignment Slack behavior, public CLI wiring, W&B identity, lifecycle repair, and all deferrals.

Expected: fix Critical/Important findings only with focused failing tests and changed-file checks. Record Minor findings for final triage. Do not run the comprehensive gate here.

---

### Task 4: Prepare and Apply the Separately Reviewed Activation

**Files:**
- Create: `docs/source/guides/edullm_activation.rst`
- Modify: `docs/source/index.rst`
- Modify after user supplies public identities: `config/edullm/team-leads.yaml`
- Modify after user supplies public identities: `config/edullm/operators.yaml`
- Modify after user approval: `.github/workflows/edullm-validate.yml`
- Modify after user approval: `.github/workflows/edullm-assign.yml`
- Modify: `src/test/edullm/workflow_test.py`
- Modify: `src/test/edullm/task_4_config_test.py`
- Modify: `src/test/edullm/task_5_config_test.py`

**Interfaces:**
- Consumes: exactly one user-supplied team-lead GitHub login, one operator GitHub login, one operator Slack member ID, and a user-configured repository secret named `SLACK_WEBHOOK_URL`.
- Produces: exactly one allowlisted lead, exactly one enabled operator, enabled validation/assignment/assignment-notification core path, required labels, and static proof that enabled eduLLM Actions receive only `github.token` plus the Slack webhook.
- User-only external mutations: configure `SLACK_WEBHOOK_URL`, create labels, review/merge activation, and authorize workflow execution.
- Remains disabled: reminders, reassignment, terminal Slack, scheduled W&B monitor, all Plan 3 work, and strict ruleset automation.

- [ ] **Step 1: Write the activation guide before changing protected controls**

Create `docs/source/guides/edullm_activation.rst` that distinguishes:

```text
Agent can prepare:
- a diff containing one public team-lead login, one public operator GitHub
  login/Slack member ID, the three core guard changes, tests, and docs;
- static config/workflow/credential-boundary evidence;
- exact label and secret commands for the user to inspect.

User must perform:
- choose and provide the three public identity values;
- configure SLACK_WEBHOOK_URL directly in GitHub, never in Git or chat;
- create/confirm labels with repository administration permission;
- review and merge the activation change;
- provide team-lead approval and operator GitHub/SSH/Kerberos/Duo/W&B access
  at the live stop points.
```

Add the guide to `docs/source/index.rst`. Do not add an activation script: the existing `load_team_leads`, `load_operators`, `load_policy`, and workflow tests already provide static validation.

- [ ] **Step 2: Stop for the three public identities**

The executor must not edit protected config until the user supplies:

```text
TEAM_LEAD_GITHUB_LOGIN
OPERATOR_GITHUB_LOGIN
OPERATOR_SLACK_USER_ID
```

These names define runtime inputs, not committed example values. Do not request or accept the Slack webhook, ORCD, SSH, Kerberos, Duo, W&B, or S3 credentials.

Expected: one explicit user response containing only the three public identifiers. Stop if any identifier is absent.

- [ ] **Step 3: Write failing activation-state tests**

Change the static tests to require:

```python
def test_only_core_workflows_are_enabled():
    validate = yaml.load(
        (WORKFLOWS / "edullm-validate.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assign = yaml.load(
        (WORKFLOWS / "edullm-assign.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    reminders = (WORKFLOWS / "edullm-reminders.yml").read_text(encoding="utf-8")
    terminal = (WORKFLOWS / "edullm-terminal-notify.yml").read_text(encoding="utf-8")

    assert validate["jobs"]["validate"]["if"] == (
        "${{ contains(github.event.issue.labels.*.name, 'edullm-job') }}"
    )
    assert validate["jobs"]["assign"]["if"] == "${{ needs.validate.result == 'success' }}"
    assert assign["jobs"]["assign"]["if"] == (
        "${{ github.repository == 'edu-llm/OLMo-core' }}"
    )
    assert "${{ false &&" in reminders
    assert "${{ false &&" in terminal


def test_pilot_has_exactly_one_lead_and_one_enabled_operator():
    leads = load_team_leads(Path("config/edullm/team-leads.yaml"))
    operators = load_operators(Path("config/edullm/operators.yaml"))
    assert len(leads) == 1
    assert len(operators) == 1
    assert sum(operator.enabled for operator in operators) == 1
```

In `src/test/edullm/workflow_test.py`, replace
`test_every_edullm_workflow_remains_literally_disabled_and_sha_pinned` with
`test_only_core_workflows_are_enabled`, while retaining SHA-pin and
`persist-credentials: false` assertions for every eduLLM workflow. In
`src/test/edullm/task_4_config_test.py`, replace
`test_workflow_is_hard_disabled_with_exact_triggers_and_issue_filter` and
`test_production_team_leads_file_is_explicitly_empty` with activated guard and
single-lead/single-operator assertions. In
`src/test/edullm/task_5_config_test.py`, replace
`test_validation_workflow_has_a_literal_false_reusable_assignment_handoff` and
the assignment half of
`test_task_5_workflows_are_hard_disabled_and_globally_serialized`; keep the
reminder workflow's literal-disable and shared-concurrency assertions.

Run:

```bash
pytest -v \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  -k 'only_core_workflows or pilot_has_exactly'
```

Expected: FAIL because rosters are empty and the three core guards are still literal-disabled.

- [ ] **Step 4: Populate only public protected identities**

After exporting the three user-supplied values in the executor's local shell, run this exact one-off command:

```bash
python - "$TEAM_LEAD_GITHUB_LOGIN" "$OPERATOR_GITHUB_LOGIN" "$OPERATOR_SLACK_USER_ID" <<'PY'
import sys
from pathlib import Path

import yaml

team_lead, operator, slack_id = sys.argv[1:]
Path("config/edullm/team-leads.yaml").write_text(
    yaml.safe_dump({"team_leads": [team_lead]}, sort_keys=False),
    encoding="utf-8",
)
Path("config/edullm/operators.yaml").write_text(
    yaml.safe_dump(
        {
            "operators": [
                {
                    "github": operator,
                    "slack_user_id": slack_id,
                    "rotation_order": 0,
                    "enabled": True,
                }
            ]
        },
        sort_keys=False,
    ),
    encoding="utf-8",
)
PY
```

Expected: one lead, one operator, rotation order `0`, enabled `true`; no credential or secret value in either file.

- [ ] **Step 5: Remove only the three core literal disables**

Set these exact guards:

```yaml
# .github/workflows/edullm-validate.yml, jobs.validate
if: ${{ contains(github.event.issue.labels.*.name, 'edullm-job') }}

# .github/workflows/edullm-validate.yml, jobs.assign
if: ${{ needs.validate.result == 'success' }}

# .github/workflows/edullm-assign.yml, jobs.assign
if: ${{ github.repository == 'edu-llm/OLMo-core' }}
```

Do not change `.github/workflows/edullm-reminders.yml` or `.github/workflows/edullm-terminal-notify.yml`. Do not add `.github/workflows/edullm-wandb-reconcile.yml`.

- [ ] **Step 6: Validate config and prove the Actions credential boundary**

Run:

```bash
PYTHONPATH=src python - <<'PY'
import re
from pathlib import Path

from edullm.automation import load_team_leads
from edullm.policy import load_operators, load_policy

root = Path(".")
leads = load_team_leads(root / "config/edullm/team-leads.yaml")
operators = load_operators(root / "config/edullm/operators.yaml")
load_policy(
    root / "config/edullm/policy.yaml",
    root / "config/edullm/entrypoints.yaml",
)
assert len(leads) == 1
assert len(operators) == 1
assert operators[0].enabled is True

core = "\n".join(
    (root / path).read_text(encoding="utf-8")
    for path in (
        ".github/workflows/edullm-validate.yml",
        ".github/workflows/edullm-assign.yml",
    )
)
assert set(re.findall(r"secrets\.([A-Z0-9_]+)", core)) == {"SLACK_WEBHOOK_URL"}
for forbidden in ("ORCD", "SSH", "KERBEROS", "DUO", "WANDB", "AWS", "S3"):
    assert f"secrets.{forbidden}" not in core
for deferred in (
    ".github/workflows/edullm-reminders.yml",
    ".github/workflows/edullm-terminal-notify.yml",
):
    assert "${{ false &&" in (root / deferred).read_text(encoding="utf-8")
PY

pytest -v \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py
git diff --check
```

Expected: PASS; enabled eduLLM workflow text references only `SLACK_WEBHOOK_URL`; deferred workflows remain literal-disabled.

- [ ] **Step 7: Run one independent activation-diff review**

Review only the Task 4 diff. Require:

- exactly one lead and one enabled operator;
- no webhook or compute/experiment credential in Git;
- exactly the three core guard changes;
- unchanged least-privilege permissions, SHA-pinned actions, concurrency, and checkout `persist-credentials: false`;
- reminders, terminal Slack, scheduled W&B monitoring, and Plan 3 remain disabled/deferred.

Expected: separate `Spec: APPROVE` and `Quality: APPROVE`. Repair Critical/Important findings with focused tests; record Minor findings.

- [ ] **Step 8: Commit the reviewed activation package without external mutation**

```bash
git add \
  config/edullm/team-leads.yaml \
  config/edullm/operators.yaml \
  .github/workflows/edullm-validate.yml \
  .github/workflows/edullm-assign.yml \
  src/test/edullm/workflow_test.py \
  src/test/edullm/task_4_config_test.py \
  src/test/edullm/task_5_config_test.py \
  docs/source/guides/edullm_activation.rst \
  docs/source/index.rst
git commit -m "ops: prepare eduLLM core activation"
```

Expected: one separately reviewable commit containing public identities/config/guards/tests/docs and no secret.

- [ ] **Step 9: Stop for user-only GitHub configuration**

Present, but do not execute, these commands. The user with repository permission runs them:

```bash
gh label create edullm-job --repo edu-llm/OLMo-core --color 1D76DB --force
gh label create status:requested --repo edu-llm/OLMo-core --color D4C5F9 --force
gh label create status:validating --repo edu-llm/OLMo-core --color D4C5F9 --force
gh label create status:ready --repo edu-llm/OLMo-core --color 0E8A16 --force
gh label create status:assigned --repo edu-llm/OLMo-core --color 0E8A16 --force
gh label create status:submitted --repo edu-llm/OLMo-core --color FBCA04 --force
gh label create status:running --repo edu-llm/OLMo-core --color FBCA04 --force
gh label create status:completed --repo edu-llm/OLMo-core --color 0E8A16 --force
gh label create status:failed --repo edu-llm/OLMo-core --color D93F0B --force
gh label create status:cancelled --repo edu-llm/OLMo-core --color 6A737D --force
gh label create status:preempted --repo edu-llm/OLMo-core --color D93F0B --force

read -r -s SLACK_WEBHOOK_URL
printf %s "$SLACK_WEBHOOK_URL" | \
  gh secret set SLACK_WEBHOOK_URL --repo edu-llm/OLMo-core
unset SLACK_WEBHOOK_URL
```

The user then reviews and merges the activation commit/PR. The executor captures only:

```bash
gh secret list --repo edu-llm/OLMo-core | awk '$1 == "SLACK_WEBHOOK_URL" {found=1} END {exit !found}'
gh label list --repo edu-llm/OLMo-core --limit 100
```

Expected: secret name present without value; all 11 labels present. No ORCD/SSH/Kerberos/Duo/W&B/S3 credential is requested, stored, printed, or passed to an eduLLM Action.

---

### Task 5: Run the Final Local Gate and User-Assisted Live Acceptance

**Files:**
- Verify: all Plan 2 vertical-slice files changed from `0599e1e544c7621988aef7a08467bbc078d5ec0f..HEAD`
- Modify after real failures only: the smallest source/test/docs pair that reproduces the accepted Critical/Important defect
- Create: `docs/superpowers/reports/2026-07-23-edullm-core-acceptance.md`
- Modify locally, never stage: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: approved Tasks 1–4, one approved generic-smoke PR head SHA, one real Skill-created Issue, one Slack assignment, one configured operator CLI, and user-held GitHub/SSH/Kerberos/Duo/W&B access.
- Produces: one comprehensive local result, one final full-Plan-2 whole-branch review, and one tracked redacted live one-L40S generic-smoke evidence report ending in a terminal Issue state through `edullm jobs`.
- Live stop points: team-lead approval, GitHub activation, operator authentication, `edullm run`, ORCD observation, and W&B inspection all require the user or designated credential holder.

- [ ] **Step 1: Run the comprehensive local gate exactly once after assembly**

Run:

```bash
set -euo pipefail
pytest -v src/test/edullm/ src/test/scripts/orcd/

python -m ruff check \
  src/edullm \
  src/test/edullm \
  src/test/scripts/orcd \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m black --check \
  src/edullm \
  src/test/edullm \
  src/test/scripts/orcd \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m isort --check-only \
  src/edullm \
  src/test/edullm \
  src/test/scripts/orcd \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
python -m mypy src/edullm .cursor/skills/submit-edullm-job/scripts/validate_request.py

go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 \
  .github/workflows/main.yml \
  .github/workflows/edullm-validate.yml \
  .github/workflows/edullm-assign.yml \
  .github/workflows/edullm-reminders.yml \
  .github/workflows/edullm-terminal-notify.yml

python3.10 -m compileall -q \
  src/edullm \
  .cursor/skills/submit-edullm-job/scripts/validate_request.py
PYTHONPATH=src python3.10 - <<'PY'
import edullm.cli
import edullm.data_manifest
import edullm.jobs
import edullm.request_parser
import edullm.slurm
PY

python - <<'PY'
import json
import tomllib
from pathlib import Path

import yaml

for path in Path(".github/workflows").glob("edullm-*.yml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
for path in Path("config/edullm").glob("*.yaml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
json.loads(Path("config/edullm/main-ruleset.json").read_text(encoding="utf-8"))
tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
PY

git diff --check 0599e1e544c7621988aef7a08467bbc078d5ec0f..HEAD
STATUS="$(git status --porcelain=v1)" || exit 2
test -z "$STATUS"
```

Expected: all eduLLM and Plan 1 ORCD tests pass; changed project lint/type/style, actionlint, Python 3.10 compile/import, YAML/JSON/TOML parsing, diff check, and clean-status check exit 0. This gate is not rerun after every later micro-fix.

- [ ] **Step 2: Run one final whole-branch independent review**

Generate:

```bash
set -euo pipefail
FINAL_REVIEW_BASE=0599e1e544c7621988aef7a08467bbc078d5ec0f
FINAL_REVIEW_PACKAGE=/tmp/edullm-plan-2-whole-branch-review.txt
git diff --find-renames --stat \
  "$FINAL_REVIEW_BASE"..HEAD
{
  printf '%s\n' '# Commit list'
  git log --oneline "$FINAL_REVIEW_BASE"..HEAD
  printf '%s\n' '# Stat'
  git diff --find-renames --stat "$FINAL_REVIEW_BASE"..HEAD
  printf '%s\n' '# Full diff'
  git diff --find-renames -U10 "$FINAL_REVIEW_BASE"..HEAD
} > "$FINAL_REVIEW_PACKAGE"
```

Give one fresh final reviewer `/tmp/edullm-plan-2-whole-branch-review.txt`, the approved scope, the ignored local progress ledger, and all task-review outcomes. The package starts at the Plan 2 base and therefore includes completed Plan 2 Tasks 1–6, existing Task 7, and every remaining slice. Tasks 1–6 remain complete and are not redispatched; their code appears only in this one integrated final review.

Require spec-compliance and code-quality verdicts plus explicit checks for credential leakage, exact approved SHA, shell-safe submission, duplicate `sbatch`, Skill/Actions handoff, one-operator activation, W&B identity, terminal reconciliation, and deferred-scope containment.

Expected: fix Critical/Important findings only. Add one focused failing regression, make the smallest repair, run only covering tests and changed-file checks, and obtain approval on that repair. Record Minor findings in final triage. Do not rerun the 900-plus suite unless the reviewer proves the comprehensive gate itself was invalid.

- [ ] **Step 3: Stop for the approved generic-smoke PR**

The teammate and agent prepare generic-smoke code/config on a branch and open a PR. The user-designated team lead approves the exact current PR head SHA after `Test eduLLM core` and all policy-required checks pass.

Capture:

```bash
set -euo pipefail
PR_NUMBER="$(gh pr view --repo edu-llm/OLMo-core --json number --jq .number)" || exit 2
APPROVED_SHA="$(gh pr view "$PR_NUMBER" --repo edu-llm/OLMo-core --json headRefOid --jq .headRefOid)" || exit 2
test "${#APPROVED_SHA}" -eq 40
gh pr view "$PR_NUMBER" --repo edu-llm/OLMo-core \
  --json number,url,state,isDraft,headRefName,headRefOid,reviews,statusCheckRollup
```

Expected evidence: non-`main` head branch; non-draft open or merged PR; exact full SHA; allowlisted lead approval on that SHA; all required checks successful. Stop if the head changes, approval is stale, or CI is not successful.

- [ ] **Step 4: Use the real Skill and capture Actions/Slack assignment**

Invoke `/submit-edullm-job` with the generic-smoke request. The Skill validates, previews, asks for confirmation, and creates the Issue through `gh issue create --body-file`.

Capture:

```bash
gh issue view "$ISSUE_NUMBER" --repo edu-llm/OLMo-core \
  --json number,url,title,author,body,labels,assignees,comments
gh run list --repo edu-llm/OLMo-core \
  --workflow edullm-validate.yml \
  --event issues \
  --limit 10
```

Expected evidence:

- Skill transcript shows clean branch, PR URL, exact approved SHA, local parser/policy success, preview, and explicit confirmation.
- Issue body exactly matches the validated body and contains structured argument JSON, not shell.
- Actions validation records `status:ready`, then assignment records exactly one enabled operator and `status:assigned`.
- One Slack assignment reaches that operator; capture timestamp, channel, Issue number, and mapped operator mention, but never the webhook URL.

Stop if validation/assignment fails; reproduce locally and apply focused test-first fixes before retrying. Do not manually create an Issue as a substitute.

- [ ] **Step 5: Stop for operator-held credentials and command-as-acceptance**

The designated operator, not the agent, confirms local `gh`, SSH/Kerberos/Duo, ORCD, and W&B access and runs:

```bash
edullm setup
edullm run
```

Expected: `edullm run` requires no manual Issue audit, selects the oldest assigned eligible request, repeats the exact-SHA/review/check/policy/data/assignment gates, opens or reuses the SSH ControlMaster, and prints one Slurm ID plus the deterministic W&B URL.

Capture only:

- command exit codes and redacted stdout/stderr;
- Issue number;
- Slurm job ID;
- W&B run ID and URL;
- ControlMaster reuse result;
- no credential, token, key, webhook, or Duo response.

Stop on any pre-submit failure. If submission outcome is ambiguous, rerun the approved recovery path; never issue a manual second `sbatch`.

- [ ] **Step 6: Prove one L40S job, real metrics, and no duplicate submission**

The operator runs:

```bash
ssh orcd-login \
  sacct -X --noheader --parsable2 --jobs "$SLURM_JOB_ID" \
  --format=JobIDRaw,JobName,Partition,AllocTRES,State,ExitCode
edullm logs "$ISSUE_NUMBER"
```

Expected Slurm evidence: one top-level numeric job ID, partition `mit_normal_gpu`, one L40S GPU allocation, and terminal `COMPLETED` with zero exit status. The lifecycle comment contains exactly one attempt and the same Slurm/W&B identities; no second `sbatch` receipt or second attempt exists.

On the operator's authenticated machine, inspect W&B:

```bash
WANDB_RUN_PATH="eduLLM/test/$WANDB_RUN_ID" python - <<'PY'
import math
import os

import wandb

run = wandb.Api(timeout=30).run(os.environ["WANDB_RUN_PATH"])
rows = list(run.scan_history(keys=["_step", "train/CE loss"]))
losses = [row["train/CE loss"] for row in rows if row.get("train/CE loss") is not None]
steps = [row["_step"] for row in rows if row.get("_step") is not None]
assert run.state == "finished"
assert steps and max(steps) >= 19
assert losses and all(math.isfinite(float(value)) for value in losses)
print(run.url)
print(max(steps))
print(losses[-1])
PY
```

Expected W&B evidence: the recorded URL, finished state, at least 20 steps (`_step` 0 through at least 19), and finite real `train/CE loss`. Do not capture the W&B API key or private config.

- [ ] **Step 7: Reconcile terminal Issue state through `edullm jobs`**

The operator runs:

```bash
edullm jobs --mine
gh issue view "$ISSUE_NUMBER" --repo edu-llm/OLMo-core \
  --json number,url,labels,assignees,comments
```

Expected:

- `edullm jobs --mine` queries `squeue`/`sacct`, reports the same Slurm and W&B identities, and exits 0.
- The Issue has exactly one managed terminal label, `status:completed`.
- The canonical lifecycle comment has one attempt, terminal `completed`, the same Slurm ID, W&B run ID, W&B URL, request digest, operator, and timestamps.
- No scheduled W&B workflow or monitor credential participated.

- [ ] **Step 8: Apply only real-result Critical/Important fixes**

For a live defect:

1. Save a redacted reproduction and classify it.
2. Add one focused local failing test.
3. Make the smallest repair.
4. Run that covering test and changed-file lint/type/style only.
5. Obtain one focused review of the repair.
6. Repeat the affected live step with user assistance.

Minor live polish is recorded for final triage. Do not rerun the comprehensive local gate after each fix and do not expand into deferred features.

- [ ] **Step 9: Write tracked redacted acceptance evidence**

Create `docs/superpowers/reports/2026-07-23-edullm-core-acceptance.md` with exactly these sections and fields:

```text
# eduLLM Core Vertical Slice Acceptance

## Summary
- Acceptance date in UTC
- Final status: accepted or failed
- Approved scope commit: 05ed8faeca230e9e92af501207aea5cb18e8665b
- Implementation head: full 40-character SHA

## Redaction Rules
- Evidence contains no GitHub token, Slack webhook URL, SSH private key,
  Kerberos ticket, Duo response, W&B API key, S3/AWS credential, secret
  environment value, or unredacted operator home/ORCD username.
- Public GitHub logins, Slack member ID, Issue/PR/Actions URLs, Slurm job ID,
  W&B run URL, request digest, commit SHA, timestamps, exit codes, and bounded
  sanitized diagnostics are permitted.
- Raw command output is redacted before it enters this tracked report.

## Comprehensive Local Gate
- Execution UTC timestamp
- Full implementation SHA
- Exact command block from Task 5 Step 1
- eduLLM pytest passed/failed count
- Plan 1 ORCD pytest passed/failed count
- Ruff, Black, isort, mypy, actionlint, Python 3.10 compile/import,
  YAML/JSON/TOML parse, git diff, and clean-status exit results

## Final Whole-Branch Review
- Base: 0599e1e544c7621988aef7a08467bbc078d5ec0f
- Head: full 40-character SHA
- Review-package path
- Spec-compliance verdict
- Code-quality verdict
- Critical, Important, and Minor findings
- Repair commit SHAs and focused verification results

## Reviewed Generic-Smoke PR and SHA
- PR number and URL
- Non-main head branch
- Exact full approved PR head SHA
- Allowlisted approving team-lead GitHub login and approval UTC timestamp
- Required CI check names and successful conclusions

## Skill-Created Issue
- `/submit-edullm-job` invocation UTC timestamp
- Skill confirmation result
- Issue number and URL
- Canonical request digest
- Exact requested commit SHA

## GitHub Actions Validation and Assignment
- Validation run ID and URL
- Validation conclusion and resulting status label
- Assignment run ID and URL
- Assigned operator GitHub login and resulting status label

## Slack Assignment Delivery
- Delivery UTC timestamp
- Channel identifier
- Issue number
- Mapped operator Slack member ID
- Delivery confirmation with webhook and message payload secrets omitted

## Operator Command
- Operator public GitHub login
- `edullm setup` exit status
- `edullm run` UTC timestamp and exit status
- Redacted command result
- SSH ControlMaster reuse confirmation

## Slurm and L40S Evidence
- Slurm job ID
- Partition
- Allocated GPU type and count
- Job state and exit code
- Top-level accounting row count
- Canonical lifecycle attempt count

## W&B Metrics Evidence
- Entity, project, run ID, and run URL
- Run state
- Maximum `_step`
- Metric name `train/CE loss`
- Finite final metric value

## Terminal Issue Reconciliation
- `edullm jobs --mine` UTC timestamp and exit status
- Final Issue label
- Canonical lifecycle state
- Slurm job ID, W&B run ID/URL, request digest, operator, and attempt count

## No-Secret and No-Duplicate Proof
- Enabled-workflow secret-reference scan result
- Tracked-file credential scan result
- Redacted-output inspection result
- Submission receipt count
- Canonical lifecycle attempt count
- Top-level Slurm accounting row count

## Deferred Scope
- Original Plan 2 Task 8 scheduled W&B monitoring remains deferred;
  `edullm jobs` supplied terminal reconciliation.
- All Plan 3 work, S3, Apptainer, multiple operators, advanced Slack,
  strict ruleset automation, and broad rollout polish remain deferred.
```

Expected: every field has captured evidence or an explicit failed result; the final status is `accepted` only if all nine approved outcomes pass.

- [ ] **Step 10: Update ignored progress and commit only tracked evidence**

Update ignored local `.superpowers/sdd/progress.md` with the Task 7 closure, remaining-slice commits/reviews, comprehensive local gate, final review, and live acceptance result. Confirm its ignored status:

```bash
git check-ignore -q .superpowers/sdd/progress.md
git add docs/superpowers/reports/2026-07-23-edullm-core-acceptance.md
test "$(git diff --cached --name-only)" = \
  "docs/superpowers/reports/2026-07-23-edullm-core-acceptance.md"
git commit -m "docs: record eduLLM core acceptance"
```

Expected: a documentation-only commit containing the tracked acceptance report. `.superpowers/sdd/progress.md` remains updated local scratch and is not staged. Push only if the user separately requests it.

---

## Completion Definition

This plan is complete only after all five tasks are approved and the live evidence proves the nine approved outcomes: reviewed experiment PR; exact approval and CI; real Skill submission; real Actions routing; Slack delivery; command-as-acceptance; single safe submission; real metrics and `edullm jobs` reconciliation; and one live terminal generic-smoke proof.

Original Plan 2 Task 8 scheduled W&B monitoring and all Plan 3 work remain deferred. `edullm jobs` is the initial terminal reconciliation mechanism.
