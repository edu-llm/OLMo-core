---
name: tdd-loop
description: Test-first control loop for implementing any single scoped change (feature, bug fix, or one-line edit). Use for every implementation trigger with a verifiable definition of done. Author a failing test BEFORE writing code; running the existing suite or an ad-hoc script does not count.
---

# TDD Plan -> Act -> Check Loop

A control structure for implementing a single trigger (a story or ticket) by looping
through **Plan**, **Act**, and **Check** until automated tests confirm the work is done.
Optimizes for speed by default and escalates reasoning power only when the loop stalls.

## Test-first gate (non-negotiable)

Before you edit any implementation file, you MUST have authored (or extended) a test
that asserts the trigger's acceptance criteria and **watched it fail (red)**. State, in
the chat, the test's name/path and the observed failure before writing implementation
code. If there is no red test, do not write code yet.

This applies to **every** trigger, including bug fixes and one-line changes: write a
test that reproduces the bug and fails first, then fix it. Running the existing suite,
eyeballing output, or a throwaway diagnostic script does **NOT** satisfy this gate — a
new or extended test that failed before your change and passes after is the only proof
that counts. The only exemptions are changes with no assertable behavior (docs,
comments, formatting, generated files); name the exemption explicitly when you use it.

## When to use this skill

Use this when you receive a **trigger**: a story, ticket, feature request, bug report,
or any scoped unit of development work that has (or can be given) a verifiable definition
of done. Do **not** use it for open-ended exploration, pure Q&A, or research with no
testable outcome.

**Size ceiling.** `tdd-loop` owns a single scoped change — ideally a single-file change
with one acceptance criterion and no new files. If the work creates several new files,
spans many modules, or bundles 2+ independent deliverables, split it into separate scoped
changes and run the loop on each (or hand it to whatever multi-step orchestration layer
you use). When unsure, prefer the smaller slice.

`tdd-loop` is the implementation control loop. Use a relevant domain skill (e.g.
UI/UX) *inside* the loop to inform acceptance criteria, guidance, and checks — it
never replaces Plan → Act → Check.

## Core inputs (the trigger)

A natural-language description of the desired change, from the user or a parent agent.
Before looping, restate it as:

- **Goal** - one or two sentences describing the intended end state.
- **Acceptance criteria** - concrete, checkable conditions ("given X, when Y, then Z").
- **Scope boundary** - what is explicitly in scope and what is out of scope.
- **Test level** - choose based on the trigger's scope:
  - _Unit_ for a function/module/pure-logic change.
  - _End-to-end / integration_ for a user-facing flow or cross-module behavior.

If the trigger is ambiguous enough that you cannot write a test for "done", ask one
clarifying question before starting. Otherwise, proceed with reasonable assumptions and
note them.

## Loop state to track

Maintain these across iterations (use the todo list to make them visible):

- `model` - current execution model. **Start with a fast, cheap model** (for speed).
- `consecutiveCheckFailures` - count of Check phases that failed in a row. Start at 0.
- `attempt` - total Plan->Act->Check iterations.
- `escalationThreshold` - default **2**. When `consecutiveCheckFailures >= escalationThreshold`,
  escalate the model (see Escalation).

## The loop

### 0. Setup (once)

1. Parse the trigger into Goal / Acceptance criteria / Scope / Test level (above).
2. Create a todo list with at least: "Plan + author tests", "Act (implement)",
   "Check (run tests)", and a placeholder for follow-up iterations.
3. Identify the project's test runner and conventions (e.g. `vitest`, `pytest`, `jest`,
   `go test`) by inspecting config and existing tests. Do not invent a framework.

### 1. Plan

1. **Search for existing tests first.** Look for unit/integration/e2e tests that already
   cover the intended trajectory of development so you extend rather than duplicate them.
   Use code search over the test directories and match on the feature's symbols, routes,
   or behavior.
2. **Do a duplicate-test check before authoring tests.** List the relevant existing
   test files or test cases you found, state whether each will be extended or skipped,
   and explain why. Prefer extending the closest existing test file/case. Create a new
   test file or new top-level `describe` only when no existing location cleanly covers
   the behavior; note that justification in the plan.
3. Write a **comprehensive implementation plan**: the files to touch, the approach, edge
   cases, and the smallest change that satisfies the acceptance criteria.
4. **Author the tests that verify the final output of the trigger** (or extend the
   existing ones found in step 1). Tests must assert the acceptance criteria and should
   **fail initially** (red), proving they exercise the new behavior. Keep tests focused
   on observable outcomes, not implementation details.
5. On a _re-entry_ into Plan after a failed Check, incorporate the captured failure
   evidence (assertion diffs, stack traces, logs) and a root-cause hypothesis into the
   revised plan before acting again.

### 2. Act

0. **Confirm the test-first gate is satisfied** - a red test from the Plan phase exists.
   If not, go back to Plan and write it before editing any implementation file.
1. Implement exactly the plan from step 1.
2. **Stay strictly in scope** - do not add features, refactors, or files beyond what the
   trigger and plan require.
3. **Avoid abstruse logic** - prefer the clearest, most maintainable implementation that
   passes the tests; no clever indirection, premature abstraction, or dead code.
4. Do not weaken, delete, or rewrite the Check tests to force a pass. The tests are the
   contract; if a test is genuinely wrong, fix it deliberately in the Plan phase with a
   stated justification, not silently during Act.

### 3. Check

1. Run the tests authored/identified in the Plan phase (plus the project's existing
   suite and linters/typecheck where cheap) to guard against regressions.
2. Evaluate:
   - **All target tests pass AND the trigger's acceptance criteria are met** ->
     exit the loop. Reset `consecutiveCheckFailures = 0` and proceed to Return.
   - **Any target test fails** -> increment `consecutiveCheckFailures` and `attempt`,
     capture the concrete failure output, and **loop back to Plan** for the next attempt.

## Escalation (model selection)

- **Default:** run Plan and Act with a **fast, cheap model** for speed throughout the loop.
- **Escalate:** when `consecutiveCheckFailures >= escalationThreshold` (default 2),
  switch to a **stronger reasoning model**, return to the **Plan** phase, and add
  context: the full failure history, what was tried, and a fresh root-cause analysis.
  The stronger model is for breaking out of a stall, not the default cruising speed.
- **De-escalate (optional):** once the stronger model produces a passing Check and
  meaningful progress is restored, you may drop back to the fast model for any remaining
  straightforward work.
- Always **reset `consecutiveCheckFailures` to 0** after a passing Check.

**Switching models:** if your harness supports it, delegate Plan/Act to a subagent with
the model set explicitly (pass the full context — plan, tests, prior failures — since
subagents don't share memory); otherwise switch the session to the stronger model before
re-planning. Pick the newest available model in each tier rather than pinning a version.

## Guardrails

- One trigger per loop run. If the trigger decomposes into independent stories, or
  exceeds the size ceiling above (several new files, many modules, or 2+ independent
  deliverables), split it into separate scoped changes (or hand it to your multi-step
  orchestration layer) instead of looping per story.
- **Definition of done.** Never mark the trigger complete without: (a) a specific
  new/extended test that **failed before** your change and **passes after**, named in
  the Return, and (b) a green Check. "The existing suite passes" is not sufficient proof
  on its own — cite the test that exercises *this* change and its red→green transition.
- Keep the diff minimal and reviewable; out-of-scope changes are a defect.
- Put a sane cap on iterations (e.g. stop and report after ~5 attempts or after an
  escalation to the stronger model still fails twice) instead of looping forever; surface
  the blocker.

## Return (loop exit)

When the loop exits, report back to the trigger's source with:

1. Outcome (done / blocked) and the final Check result.
2. The tests that now verify the behavior (new or extended), and confirmation they pass.
3. A concise summary of the changes made (files + intent).
4. Any assumptions, scope decisions, or follow-ups.
5. If blocked: the failure evidence, what was tried (including any model escalation), and
   the most likely next step.
