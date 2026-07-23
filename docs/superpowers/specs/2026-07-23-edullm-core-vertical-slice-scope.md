# eduLLM Core Vertical Slice Scope Amendment

Date: 2026-07-23
Status: Approved scope; written amendment awaiting user review
Amends: `docs/superpowers/specs/2026-07-22-orcd-job-pool-design.md`

## Purpose and Precedence

This amendment defines the current acceptance boundary for the ORCD job-pool design. Where the
original design describes a broader rollout or staged follow-on work, this amendment controls the
remaining core implementation and acceptance scope. The implementation plan must not be updated
until the user reviews this written amendment.

The product must complete one real request-to-running-job path. It is not an Issue-form-only
workflow, local-automation endpoint, or other stopgap. Existing general code may remain, including
disabled follow-on code, but only the scope below determines current acceptance.

## Required End-to-End Flow

1. A teammate works with an agent to prepare training code/config and opens a PR.
2. A team lead approves the exact PR head SHA and CI passes.
3. The real `/submit-edullm-job` Skill creates the structured GitHub Issue; the existing form is
   not a substitute for the finished Skill.
4. Real GitHub Actions validate the request and assign the enabled operator.
5. Slack assignment notification reaches that operator.
6. The operator does not manually audit the Issue; `edullm run` immediately selects the oldest
   assigned eligible request, performs all automatic revalidation, and submits it. Running the
   command is acceptance.
7. The CLI submits exactly one Slurm job through the operator's SSH ControlMaster, records Slurm
   and W&B identity, and exposes `jobs`, `logs`, `stop`, and `logout`.
8. Training sends real metrics to W&B. `edullm jobs` reconciles terminal Slurm state to the Issue
   for the initial product; dedicated scheduled W&B monitoring may remain deferred.
9. One real one-L40S generic-smoke request completes end-to-end with W&B metrics and Issue terminal
   state.

## Core Scope

The current core scope is:

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

## Explicitly Deferred

The following work is not required for this vertical slice:

- All Plan 3 Skill-DAG and curriculum work.
- Team or data-mixture design. Submitting research and data teams own those inputs in their PRs.
- S3 rollout.
- Apptainer.
- Operators two and three, and multi-user acceptance.
- A scheduled W&B monitor credential and workflow.
- Advanced Slack reminders, reassignment, and terminal-event threads.
- Strict repository ruleset enforcement.
- Extensive failure drills and rollout polish.

Already-built disabled code for deferred capabilities is not deleted. General code may remain, but
its presence does not make it part of current acceptance. In particular, Plan 3 is neither a
dependency nor an acceptance requirement for this vertical slice.

## Team-Owned Experiment Inputs

The submitting teammate and their research or data team own the scientific and experiment inputs
in the reviewed PR:

- Training code and configuration.
- Hypothesis, comparison, condition, seed, and success metrics.
- Model and entrypoint changes.
- Data selection, data manifests, data-mixture design, and any curriculum or ordering logic.
- W&B instrumentation needed to emit the claimed scientific metrics.

The platform team owns the request Skill, schema, gates, assignment, operator CLI, safe Slurm
submission, lifecycle recording, and reconciliation. The platform validates submitted inputs
against policy; it does not design a team's experiment or data mixture. The core generic-smoke
profile remains a platform-owned integration fixture and does not require Plan 3 inputs.

## Temporary Pilot Trust and Fail-Closed Boundary

- Contributors use branches and pull requests. Direct-main SHAs are ineligible.
- Software verifies the exact approved PR head SHA and passing CI before assignment and again
  before submission.
- Current repository writers are trusted during the pilot until strict rulesets are added.
- GitHub Actions receive no ORCD, SSH, Kerberos, W&B, or S3 credentials. The Slack webhook is only
  for the scoped assignment notification and grants no compute or experiment access.
- Production remains fail-closed until the user supplies one operator identity, one allowlisted
  team-lead identity, and the Slack webhook, and a separately reviewed enablement change removes
  the literal workflow disables.

No current-scope change may enable workflows, add production identities or secrets, or place
operator credentials in GitHub.

## Verification and Review Policy

- During iteration, run only focused tests for touched behavior and changed-file lint/type checks.
- Build remaining core behavior in integrated slices; do not rerun 900-plus tests after each
  micro-fix.
- Use one independent review per integrated remaining slice. Fix Critical and Important
  correctness or security findings; defer Minor polish.
- Run one comprehensive local gate after the whole vertical slice is assembled, then one final
  whole-branch review.
- Perform one user-assisted live generic-smoke acceptance, followed by iterative fixes based on
  real results.

The following focused guarantees are non-negotiable:

1. No credential leakage.
2. Exact approved SHA.
3. Shell-safe submission.
4. No duplicate `sbatch`.

## Acceptance Criteria

The core vertical slice is accepted only when all nine outcomes below are demonstrated:

1. **Reviewed experiment PR:** A teammate and agent prepare the generic-smoke training code and
   configuration on a branch and open a PR; no direct-main SHA is accepted.
2. **Exact approval and CI:** The allowlisted team lead approves the exact current PR head SHA,
   required CI passes for that SHA, and a changed head invalidates the earlier approval.
3. **Real Skill submission:** The finished `/submit-edullm-job` Skill validates and previews the
   request, receives teammate confirmation, and creates the structured Issue. Manually opening the
   existing Issue form does not satisfy this criterion.
4. **Real Actions routing:** Enabled GitHub Actions validate the Issue without compute credentials,
   preserve the exact-SHA gate, and assign the single enabled operator.
5. **Slack delivery:** The basic assignment message reaches the mapped operator without exposing a
   secret or experiment credential.
6. **Command-as-acceptance:** Without manually auditing the Issue, the operator runs `edullm run`.
   It selects the oldest assigned eligible request, repeats every automatic gate, and either fails
   closed with an actionable result or proceeds directly to submission.
7. **Single safe submission:** One invocation results in exactly one shell-safe `sbatch` through
   the operator's SSH ControlMaster. The Issue records the Slurm job ID and W&B identity, a retry
   cannot duplicate the submission, and `jobs`, `logs`, `stop`, and `logout` operate on the
   recorded request.
8. **Metrics and reconciliation:** The training process sends real metrics to the recorded W&B run.
   An operator invocation of `edullm jobs` reconciles terminal `squeue`/`sacct` state back to the
   Issue without requiring a scheduled W&B monitor.
9. **Live terminal proof:** One real one-L40S generic-smoke request traverses the complete path,
   produces visible W&B metrics, and reaches the correct terminal Issue state with no credential
   leakage and no duplicate Slurm submission.

Completion of these criteria proves the initial product path. Deferred curriculum, data-mixture,
storage, packaging, multi-operator, monitoring, notification, ruleset, and rollout-polish work
cannot be used to block or substitute for this acceptance.
