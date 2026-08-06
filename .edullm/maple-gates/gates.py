"""The Maple gate ladder as data, plus the G0 offline pre-validation driver.

WHY THIS FILE EXISTS RATHER THAN A README WITH COMMANDS IN IT. Every gate below is one
`inputs.json` for `edu-llm/platform`'s `tools/compile_submission.py`, and that tool is the
only thing in this project that reproduces a platform refusal -- and the exact worst-case
dollar figure -- before a human approval is spent on it. Writing the ladder as seven
hand-maintained JSON files would guarantee that the command text drifts from the checkpoint
guard's requirements in at least one of them, silently, because the guard reads raw
characters and a JSON file does not complain. So the ladder is one table, the command
strings are built by one function, and the properties the platform enforces are asserted
here in `selfcheck()` against that table.

RUN IT (offline, credential-free, on a laptop -- this is the one thing in this lane that
may run locally, and the platform skill says so explicitly):

    python .edullm/maple-gates/gates.py selfcheck            # no deps, ~0.1s
    python .edullm/maple-gates/gates.py emit --gate G1 --out /tmp/pv
    python .edullm/maple-gates/gates.py compile --gate G1 \
        --platform /tmp/edullm-platform --digest sha256:... --pushed-at 2026-08-06T01:11:38+00:00

WHAT THE COMPILER CANNOT CATCH, so `selfcheck` catches it here instead:

  1. `no_execution_target`. `compile_submission.py` never reads
     `config/execution-targets.yaml`, so an unprovisioned profile compiles clean, classifies
     routine, SPENDS A LEAD'S SIGNATURE, and only then dies. `PROVISIONED` below is a
     literal copy of the provisioned set and `selfcheck` holds every gate to it. Note there
     is no longer an `inherit` sentinel to hide behind: `compute_profile` became a REQUIRED
     form field, so every gate must name a machine and every gate can get this wrong.
  2. The 8192-byte `ContainerOverrides` cap. A 9,230-byte program once passed compile and a
     human approval before Batch refused it. `selfcheck` bounds every command.
  3. Whether the image really exists. A hand-written `images.json` asserts it. `--digest` is
     required rather than defaulted for that reason: a stale default digest would be an
     assertion nobody re-checked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------------------
# What the account will actually run. Copied from the platform on 2026-08-06 and held by
# selfcheck; re-read both files before trusting it, because the platform config churns.
#
#   config/workload-catalog.yaml  -> rate + provisioned
#   config/execution-targets.yaml -> the queue that backs it
#
# gpu-1xa10g-sagemaker, gpu-1xh100 and gpu-8xh100 are provisioned: false and DELIBERATELY
# ABSENT from this mapping so that naming one is a KeyError here rather than a refusal after
# approval. gpu-8xh100 also came off the submission form's dropdown on 2026-08-04, so it is
# now unexpressible rather than merely refusable -- but gpu-1xl40s is ON the form, IS
# provisioned, and used to be the documented trap, so the trap has moved rather than closed.
PROVISIONED: dict[str, tuple[Decimal, int]] = {
    # name: (usd_per_hour, gpu_count)
    "cpu-32vcpu": (Decimal("1.428"), 0),
    "gpu-1xt4": (Decimal("0.5260"), 1),
    "gpu-4xt4": (Decimal("3.9120"), 4),
    "gpu-8xt4": (Decimal("7.8240"), 8),
    "gpu-1xa10g": (Decimal("1.0060"), 1),
    "gpu-4xa10g": (Decimal("5.672"), 4),
    "gpu-8xa10g": (Decimal("16.2880"), 8),
    "gpu-1xl4": (Decimal("0.8048"), 1),
    "gpu-4xl4": (Decimal("4.6016"), 4),
    "gpu-8xl4": (Decimal("13.3504"), 8),
    "gpu-1xl40s": (Decimal("1.8610"), 1),
    "gpu-4xl40s": (Decimal("10.4926"), 4),
    "gpu-8xl40s": (Decimal("30.1312"), 8),
    "gpu-8xa100": (Decimal("21.9576"), 8),
}

# T4 SHAPES CANNOT RUN THIS RECIPE AND THE REFUSAL IS THE PROGRAM'S, NOT THE PLATFORM'S.
# .edullm/train_on_corpus.py::a_precision_this_hardware_does_not_have reads compute
# capability off the device and raises Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION
# (exit 73) when the config asks for bfloat16 on pre-Ampere silicon. So the $0.53/hr T4 is
# NOT a cheaper A10G for the BF16 track -- it is an exit 73 -- and a gate that picked one to
# save forty cents would fail for a reason no platform check names.
NO_BFLOAT16 = frozenset({"gpu-1xt4", "gpu-4xt4", "gpu-8xt4"})

# ---------------------------------------------------------------------------------------
# POLICY v5, AND EVERY GUARD THIS LANE WAS RELYING ON IS GONE.
#
# THIS BLOCK WAS WRITTEN AGAINST v4 AND WAS WRONG BY 32 MINUTES. v4 had seven thresholds and
# three approval classes; v5 has ONE threshold and TWO classes, and the change landed at
# 2026-08-05 19:51 (platform c33e75c, "Re-cut who releases a run to two tiers") -- half an
# hour before this file's first compile. The first version asserted `check_a100_is_exception`
# and PASSED, because it compared the ladder against a copied constant instead of against the
# platform. A verifier caught it by building a worktree at origin/main; the shared clone at
# /tmp/edullm-platform was checked out on a feature branch, so `git pull` there was a no-op
# against main and reported "Already up to date" while ten commits behind. THE LESSON IS THE
# SKILL'S OWN RULE, WHICH THIS FILE BROKE: never trust a copied vocabulary. `verify_policy()`
# below now reads config/policy.yaml at runtime and refuses to run on a version it was not
# written against, so the next re-cut is a loud failure rather than a silent one.
POLICY_VERSION_WRITTEN_AGAINST = "v5"

# The ONLY threshold left in config/policy.yaml. A single-cell request whose worst case is
# STRICTLY under this is classified AUTOMATIC and released by `run-approval-automatic` -- an
# environment which, verified against the live GitHub API on 2026-08-06, has NO
# required_reviewers while run-approval-lead and run-approval-admin both do. So AUTOMATIC
# means NO HUMAN SEES IT.
AUTOMATIC_BELOW_COST_USD = Decimal("500")

# WHAT WENT, AND WHY EACH ABSENCE MATTERS TO THIS LADDER:
#
#   routine_maximum_cost_usd, _runtime_hours, _attempts, _fanout_size, _parallelism
#       All five removed. There is now NO absolute compile-time cost ceiling. Measured on
#       2026-08-06: 168h x 10 attempts on gpu-8xa100 compiles exit 0 at $36,888.77 and
#       classifies ROUTINE, i.e. releasable by any one team lead. Under v4 that was an
#       EXCEPTION needing a platform admin.
#
#   EXCEPTION_RATE_CEILING_USD_PER_HOUR = 20
#       Removed from the tree entirely. This is the one that mattered most here: it is what
#       CLAUDE.md and every planning document cite as forcing ADMIN approval on every A100
#       submission. THAT IS NO LONGER TRUE. gpu-8xa100 at $21.96/hr now classifies AUTOMATIC
#       for anything under $500, so G5 ($21.96), G6 ($87.83) and G7 ($87.83) would ALL
#       release themselves with nobody asked. The platform's own note says rate was withdrawn
#       because "rate is the wrong instrument" -- worst-case total carries the machine price
#       already. That is defensible policy and it still moves the guard onto US.
#
# THEREFORE THIS LANE CARRIES ITS OWN CEILING, BECAUSE THE PLATFORM NO LONGER DOES.
# A gate at or above this figure must be flagged for an explicit human decision even though
# the platform would let it through unattended. $10 is chosen so that G1-G4 (the ladder that
# is supposed to run first, unattended, and catch the cheap failures) sit below it and every
# A100 gate sits above it. This is not a platform constant and must not be confused with one.
LANE_ATTENTION_CEILING_USD = Decimal("10")

# The largest worst case that still self-releases, measured rather than derived: 22.7h x 1 on
# gpu-8xa100 compiles AUTOMATIC at $498.44, and 23h compiles ROUTINE at $505.02. So the whole
# self-releasing envelope on the most expensive provisioned machine in the account is just
# under $500 per submission, with no limit on how many such submissions are made.
SELF_RELEASING_ENVELOPE_USD = Decimal("500")

CONTAINER_OVERRIDES_CAP_BYTES = 8192

# The workload profiles that carry a checkpoint contract, so the command MUST expand
# $EDULLM_CHECKPOINT_DIR. Collapsed on 2026-08-05: olmo-core-train-1gpu and
# olmo-core-train-4gpu are BOTH now `olmo-core-train`, which is a real behaviour change for
# this ladder -- the contract no longer depends on device count, so G2 on a single GPU gets
# the same 2 attempts and 30-minute interval a 4-GPU run does.
CONTRACTED_WORKLOADS = frozenset({"olmo-core-train", "edullm-alt-cl-train",
                                  "open-instruct-scored-rewards-train"})

# Declared bounds per workload, from config/workload-catalog.yaml. A gate that does not
# override inherits these, and the worst-case cost is computed from them.
WORKLOAD_BOUNDS: dict[str, tuple[Decimal, int]] = {
    # name: (maximum_runtime_hours, maximum_attempts)
    "olmo-core-check": (Decimal("1"), 1),
    "olmo-core-train": (Decimal("24"), 2),
}

CHECKPOINT_TOKEN = '"$EDULLM_CHECKPOINT_DIR"'


@dataclass(frozen=True)
class Gate:
    """One rung of the ladder, as the submission form sees it."""

    gate_id: str
    what_it_catches: str
    workload_profile: str
    compute_profile: str
    experiment: str
    dataset_release: str
    program_args: list[str]
    # Overrides. None means "inherit the workload's declared bound", which is what the form
    # does with an empty override box.
    max_runtime_hours: Decimal | None = None
    max_attempts: int | None = None
    needs_checkpoint_dir: bool = False
    notes: str = ""
    # Set on gates whose failure IS the result. G3 is the only one today.
    provokes_failure: bool = False
    rung_selection: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    # -- derived ------------------------------------------------------------------------

    @property
    def gpu_count(self) -> int:
        return PROVISIONED[self.compute_profile][1]

    @property
    def hourly_rate(self) -> Decimal:
        return PROVISIONED[self.compute_profile][0]

    @property
    def runtime_hours(self) -> Decimal:
        if self.max_runtime_hours is not None:
            return self.max_runtime_hours
        return WORKLOAD_BOUNDS[self.workload_profile][0]

    @property
    def attempts(self) -> int:
        if self.max_attempts is not None:
            return self.max_attempts
        return WORKLOAD_BOUNDS[self.workload_profile][1]

    @property
    def worst_case_usd(self) -> Decimal:
        """rate x nodes x hours x attempts x cells. Nodes and cells are 1 for every gate."""
        return (self.hourly_rate * self.runtime_hours * self.attempts).quantize(Decimal("0.01"))

    @property
    def approval_class(self) -> str:
        """Which gate releases this, by the same test v5's classify_request() uses.

        Two classes, not three. Every gate in this ladder is one cell with a reviewed scan
        and resolving inputs, so the only live test is the cost one -- which means EVERY
        gate here classifies AUTOMATIC and NOBODY IS ASKED. That is the finding, not a
        detail: see the POLICY v5 block above.
        """
        if self.worst_case_usd < AUTOMATIC_BELOW_COST_USD:
            return "automatic"
        return "routine"

    @property
    def needs_human_decision(self) -> bool:
        """Whether THIS LANE requires a human to look, regardless of what the platform does.

        The platform stopped distinguishing a $0.71 dry run from an $87.83 A100 run, so this
        property is what re-introduces the distinction. A gate that is `automatic` on the
        platform but True here must NOT be submitted unattended.
        """
        return self.worst_case_usd >= LANE_ATTENTION_CEILING_USD

    def command(self) -> list[str]:
        """The argv the container is exec'd as.

        ONE `bash -lc` WITH ONE STRING, BECAUSE THE CHECKPOINT GUARD READS RAW CHARACTERS.
        The guard scans the characters of shell command strings rather than argv words, so
        `["python", "--save-folder", "$EDULLM_CHECKPOINT_DIR"]` is refused -- argv is never
        expanded by anything -- while the same text inside a `bash -lc` passes. Single
        quotes suppress expansion and `\\$` escapes it, so both also fail; the token has to
        be double-quoted, which is why CHECKPOINT_TOKEN carries its quotes.
        """
        parts = ["python", ".edullm/train_on_corpus.py", '"$EDULLM_RUN_ID"']
        if self.gpu_count > 1:
            # An absent --nproc-per-node reads as ONE process and is refused on a multi-GPU
            # shape, deliberately: the mere presence of the word torchrun is not enough.
            parts = [
                "python",
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={self.gpu_count}",
                "--standalone",
                ".edullm/train_on_corpus.py",
                '"$EDULLM_RUN_ID"',
            ]
        args = list(self.program_args)
        if self.needs_checkpoint_dir:
            args += ["--save-folder", CHECKPOINT_TOKEN]
        return ["bash", "-lc", " ".join(parts + args)]

    def command_bytes(self) -> int:
        """What Batch measures against its 8192-byte ContainerOverrides cap."""
        return len(json.dumps(self.command()).encode("utf-8"))

    def inputs(self, *, commit_sha: str, wandb_project: str) -> dict:
        doc = {
            "repository": "OLMo-core",
            "commit_sha": commit_sha,
            "workload_profile": self.workload_profile,
            "compute_profile": self.compute_profile,
            "team": "scratch",
            "experiment": self.experiment,
            "dataset_release": self.dataset_release,
            "command": self.command(),
            "wandb_project": wandb_project,
        }
        if self.max_runtime_hours is not None:
            doc["maximum_runtime_hours"] = str(self.max_runtime_hours)
        if self.max_attempts is not None:
            doc["maximum_attempts"] = self.max_attempts
        return doc


# ---------------------------------------------------------------------------------------
# THE LADDER.
#
# Two rung-selection shapes are templated, per contracts/ladder-and-factory.md, because L1
# has not ratified which one it is building. `--model-factory maple_scaled` plus a dotted
# override CANNOT select a rung -- see selfcheck's check_rung_selection for the proof -- so
# the wrapper shape is what every gate below uses and the kwarg shape is recorded as the
# alternative it is not. If L1 ratifies the kwarg shape, `.edullm/train_on_corpus.py` needs
# a `--model-rung` flag before any of this is submittable.
# RATIFIED IN CODE 2026-08-05 21:31. L1 shipped maple_r0..maple_r3 at merged SHA a93e81b, each
# `(cls, vocab_size: int, **kwargs)` delegating to `maple_scaled(vocab_size, rung=...)`. That
# keeps `vocab_size` the sole required positional-or-keyword arg, which is what the platform's
# `factory(vocab_size=corpus.tokenizer.padded_vocab_size())` call at train_on_corpus.py:854
# requires. Verified by reading the merged blob, not by assuming the merge landed.
FACTORY_WRAPPERS = {"R0": "maple_r0", "R1": "maple_r1", "R2": "maple_r2", "R3": "maple_r3"}

# THE PLATFORM'S rank_microbatch_size DEFAULT OOMs R3, SO EVERY R3 GATE MUST PIN IT.
# `.edullm/train_on_corpus.py:1491` defaults `--rank-microbatch-size` to 16*1024 = 16384. L6
# measured the R3 budget against A100-40GB usable (37.99 GiB after CUDA context and NCCL
# buffers) and that default lands at 131% of usable at capacity_factor 1.2 and 142% at 2.0 --
# an OOM, not a tight fit. mb=8192 fits at 78.6% (84.5% including the worst-case math-SDPA path
# on the 9 masked SWA layers).
#
# THIS IS EXACTLY THE CLASS THE LADDER EXISTS TO CATCH, AND IT WOULD HAVE REACHED AN A100
# ANYWAY, because nothing upstream of the GPU can see it: `compile_submission.py` does not read
# a model config, and OOM gets NO RETRY -- Batch retries fire only on `Host EC2*`. An R3 gate
# inheriting the default would have burned $21.96-$87.83 to produce one traceback.
# Source: lanes/L6-memory-ce/evidence/E2-memory-budget-of-record.md §0.
RANK_MICROBATCH = ["--rank-microbatch-size", "8192"]

# L6's chunked CE, verified bitwise-identical to the unchunked path on FarmShare job 1676566
# (73/73 assertions). It takes the logits family from a measured 15.344 GiB to 2.359 GiB at
# N=8192 -- 6.50x -- and makes that line mb-INDEPENDENT, which is what buys the headroom back:
# mb=8192 goes from 78.6% to 40.5% of usable. Used on the R3 gates where the logits family
# dominates. NOT used on R0, whose d=512/L=8 has no memory problem and where the default path
# is the one a production run would take -- G2 should exercise the default, not the mitigation.
CHUNKED_CE = ["--lm-loss-implementation", "chunked_linear"]


def ladder() -> list[Gate]:
    return [
        Gate(
            gate_id="G1",
            what_it_catches=(
                "the factory dispatches, every import resolves, the config round-trips, and "
                "the param ledger prints. NOTHING TRAINS: --dry-run returns before the "
                "process group, so this cannot catch a backward pass."
            ),
            workload_profile="olmo-core-check",
            compute_profile="cpu-32vcpu",
            experiment="maple-g1-config",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R0"], "--dry-run"],
            # HALF AN HOUR RATHER THAN THE DECLARED ONE, AND THE HALF-HOUR IS THE POINT.
            # It drops the worst case from $1.43 to $0.71, which is strictly under
            # automatic_below_cost_usd, and 0.5 is strictly under
            # automatic_below_runtime_hours -- so this gate is released by
            # run-approval-automatic, an environment with no required reviewers. A dry run
            # resolves a manifest and prints a config; it does not need sixty minutes.
            max_runtime_hours=Decimal("0.5"),
            notes=(
                "KNOWN RISK, NOT A KNOWN PASS: cpu-32vcpu runs as "
                "sbsandbox-intern-edullm-batch-workload, whose s3:GetObject on edullm-data "
                "exists in infra/iam/batch-roles.yaml as the read-the-dataset-airlock "
                "policy and is NOT confirmed against the deployed account. The sibling "
                "track measured exit 65 from this exact role. If G1 exits 65 that is a "
                "platform gap, not a Maple defect, and the fallback is gpu-1xa10g "
                "($1.006/hr), whose role IS behaviourally proven to read this bucket."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G2",
            what_it_catches=(
                "R0 forward and backward on a real GPU, step-0 loss inside "
                "[11.5164, 11.8164], no NaN, and the bf16 path. The first gate that "
                "executes a kernel."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-1xa10g",
            experiment="maple-g2-fwdbwd",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R0"], "--steps", "20",
                          "--save-interval", "20"],
            max_runtime_hours=Decimal("1"),
            # ONE ATTEMPT, DELIBERATELY, ON A CONTRACTED WORKLOAD. The contract offers two
            # and a second attempt on a twenty-step check buys nothing except a doubled
            # ceiling: $1.01 against $2.01. Retries fire only on `Host EC2*` anyway, so the
            # failures this gate is looking for -- a NaN, an exit 72, an OOM -- get no
            # second attempt whatever this says.
            max_attempts=1,
            needs_checkpoint_dir=True,
            notes=(
                "olmo-core-train is now the ONLY training workload, so this single-GPU "
                "gate inherits the 30-minute checkpoint contract that used to belong to "
                "the 4-GPU profile. The command must therefore expand the checkpoint dir "
                "even though twenty steps will not reach a 30-minute interval."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G3",
            what_it_catches=(
                "CHECKPOINT TORTURE. Targets the exit-72 class: two unexplained late-stage "
                "failures on the sibling track. save_interval=10 over 120 steps forces 12 "
                "writes, which is the number that matters -- max_checkpoints defaults to 3 "
                "and the prune deletes .metadata.json FIRST, on which the workload role "
                "carries an explicit IAM Deny, so the fourth save is where a run dies "
                "OLMoNetworkError. This gate reaches the fourth save at step 40 for $2 "
                "instead of at hour 11 for $264."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-1xa10g",
            experiment="maple-g3-ckpt-torture",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R0"], "--steps", "120",
                          "--save-interval", "10"],
            max_runtime_hours=Decimal("1"),
            # TWO ATTEMPTS HERE AND ONLY HERE, BECAUSE THE SECOND ATTEMPT IS THE TEST. The
            # torn-directory failure only appears on a resume: attempt 1 loses its host
            # mid-write, attempt 2 resumes from the last good step, trains back to the torn
            # one, and dies FileExistsError in Checkpointer._prepare_dir. One attempt cannot
            # observe that, so it is worth the doubled $2.01 ceiling.
            max_attempts=2,
            needs_checkpoint_dir=True,
            provokes_failure=True,
            notes=(
                "DESIGNED TO PROVOKE, NOT TO PASS. A green G3 that never reached a fourth "
                "save has tested nothing, so the assertion is on the WRITE COUNT and not "
                "on the exit code: 12 step directories must exist, each carrying all of "
                "train/rank0.pt + model_and_optim/.metadata + .metadata.json. A directory "
                "holding only train/rank0.pt is the stub-checkpoint failure that eight real "
                "sibling runs produced and that reads as healthy from a count alone. "
                "remove_torn_checkpoints() at train_on_corpus.py:1369 is the in-tree "
                "mitigation; this gate is what decides whether it works."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G4",
            what_it_catches=(
                "distributed + expert parallel on 4 ranks, FSDP2 sharding, one checkpoint "
                "shard per rank, and THE GLOBAL-BATCH ALL-REDUCE ACTUALLY REDUCING. "
                "OLMo-core's load balancing is rank-local only, so the fix L2 is writing "
                "has no effect at 1 rank and this is the first gate that can see it."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-4xa10g",
            experiment="maple-g4-distributed",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R0"], "--steps", "40",
                          "--save-interval", "20"],
            max_runtime_hours=Decimal("1"),
            max_attempts=1,
            needs_checkpoint_dir=True,
            notes=(
                "THE ALL-REDUCE ASSERTION MUST BE A MAGNITUDE, NOT AN EXISTENCE CHECK. A "
                "rank-local balance metric and a globally-reduced one are the same number "
                "when every rank sees the same tokens, so 'the metric is logged' proves "
                "nothing. Assert that per-rank and global CV DIFFER on identical inputs, or "
                "seed the ranks differently and assert the global value is identical across "
                "ranks while the local values are not. Maple has no bias_gamma, so this "
                "all-reduce is the ONLY globally-balanced mechanism available to us."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G5",
            what_it_catches=(
                "R3 flagship fits on 8xA100-40GB and produces an auditable throughput "
                "number. The binding constraint is fp32 logits (8192 x 100,352 x 4 = "
                "3.06 GiB), not weights (~5.4 GB/GPU under FSDP2)."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-8xa100",
            experiment="maple-g5-r3-throughput",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R3"], "--steps", "200",
                          "--save-interval", "100"] + RANK_MICROBATCH + CHUNKED_CE,
            max_runtime_hours=Decimal("1"),
            max_attempts=1,
            needs_checkpoint_dir=True,
            notes=(
                "$21.96/hr crosses EXCEPTION_RATE_CEILING_USD_PER_HOUR = 20, so this needs "
                "an ADMIN and not a lead, and it needs one however cheap the total is -- the "
                "rate test is on the rate. Queue median ~89 min, worst observed 12.6 h. "
                "200 steps discards 50 warmup and reports median step time over 50-150, per "
                "the throughput discipline; a number from step 1 is torch.compile, not the "
                "model."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G6",
            what_it_catches=(
                "the ternary QAT pair against BF16 at R3, same shape, same data order, so "
                "the only difference is the quantizer."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-8xa100",
            experiment="maple-g6-ternary-pair",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R3"], "--steps", "500",
                          "--save-interval", "250"] + RANK_MICROBATCH + CHUNKED_CE,
            max_runtime_hours=Decimal("4"),
            max_attempts=1,
            needs_checkpoint_dir=True,
            notes=(
                "TWO SUBMISSIONS, NOT ONE, AND NOT A FAN-OUT. A fan-out is never AUTOMATIC "
                "and more importantly its cells share one manifest, so a ternary arm and a "
                "BF16 arm cannot differ in their command. Submit them as two runs with the "
                "same experiment slug. Ternary depends on L4's quant surface; do not "
                "submit before that lands."
            ),
            rung_selection="wrapper",
        ),
        Gate(
            gate_id="G7",
            what_it_catches=(
                "the full E-sweep at fixed d=1024/L=12: R1 (E=64), R2 (E=128), R3 (E=256). "
                "Active params are near-constant 345M-347M, so FLOPs/token are ~constant "
                "and any throughput delta is kernel and routing overhead."
            ),
            workload_profile="olmo-core-train",
            compute_profile="gpu-8xa100",
            experiment="maple-g7-e-sweep",
            dataset_release="olmo-150b-dolma2-v1",
            program_args=["--model-factory", FACTORY_WRAPPERS["R1"], "--steps", "300",
                          "--save-interval", "150"] + RANK_MICROBATCH + CHUNKED_CE,
            max_runtime_hours=Decimal("4"),
            max_attempts=1,
            needs_checkpoint_dir=True,
            notes=(
                "THREE SEPARATE SUBMISSIONS, one per rung, because the rung is in the "
                "command and a fan-out's cells share one. Budget is 3 x this gate's "
                "ceiling. Read moe/audit/logic/W3-literature.md B2 first: arXiv 2508.18672 "
                "already sweeps E in 8..256 at d=1024 and finds granularity negligible at "
                "fixed active params, so fund the THROUGHPUT arm and not a quality arm."
            ),
            rung_selection="wrapper",
        ),
    ]


# ---------------------------------------------------------------------------------------
# selfcheck: the properties the platform enforces, asserted against the table above.

class CheckFailed(Exception):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def check_provisioned(gates: list[Gate]) -> str:
    """no_execution_target is the expensive class: it costs an approval before it dies."""
    for g in gates:
        _require(
            g.compute_profile in PROVISIONED,
            f"{g.gate_id} names {g.compute_profile!r}, which is not in the provisioned set. "
            "It would compile clean, classify routine, spend an approval and then die "
            "no_execution_target.",
        )
    return f"all {len(gates)} gates name a provisioned compute profile"


def check_bfloat16_capable(gates: list[Gate]) -> str:
    for g in gates:
        _require(
            g.compute_profile not in NO_BFLOAT16,
            f"{g.gate_id} names {g.compute_profile!r}, which is pre-Ampere. The recipe asks "
            "for bfloat16, so train_on_corpus.py raises exit 73 before the process group.",
        )
    return "no gate names a pre-Ampere shape"


def check_launch_guard(gates: list[Gate]) -> str:
    """Processes must equal devices exactly, in both directions."""
    for g in gates:
        text = g.command()[-1]
        if g.gpu_count > 1:
            want = f"--nproc-per-node={g.gpu_count}"
            _require(
                want in text,
                f"{g.gate_id} is a {g.gpu_count}-GPU shape and its command does not carry "
                f"{want}. An absent flag reads as one process: {g.gpu_count - 1} cards would "
                "be billed and idle.",
            )
        else:
            _require(
                "--nproc-per-node" not in text and "--nproc_per_node" not in text,
                f"{g.gate_id} is a 1-GPU shape carrying a rank flag; more processes than "
                "devices stacks ranks onto one card.",
            )
        # ONE SPELLING, DELIBERATELY, EVEN THOUGH THE PLATFORM ACCEPTS BOTH. The launch
        # guard reads `--nproc-per-node` or `--nproc_per_node`, but a verifier measured the
        # underscore spelling compiling UNENFORCED on a 4-GPU shape -- i.e. the guard did not
        # read the count from it. Until that is understood, this ladder uses hyphens only, so
        # a gate can never be the one that discovers the gap at $87.83.
        _require(
            "--nproc_per_node" not in text,
            f"{g.gate_id} uses the underscore spelling. The platform's guard was measured "
            "not to enforce a count from it, so a wrong count would pass compile.",
        )
    return "every gate starts exactly one process per device, hyphen spelling only"


def check_checkpoint_guard(gates: list[Gate]) -> str:
    """The guard greps RAW COMMAND TEXT, so the quoting is the check."""
    for g in gates:
        contracted = g.workload_profile in CONTRACTED_WORKLOADS
        _require(
            contracted == g.needs_checkpoint_dir,
            f"{g.gate_id}: workload {g.workload_profile!r} contracted={contracted} but "
            f"needs_checkpoint_dir={g.needs_checkpoint_dir}. A contract with no expansion "
            "exits zero, leaves the prefix empty, and the retry it paid for starts from "
            "nothing.",
        )
        if not contracted:
            continue
        text = g.command()[-1]
        _require(
            CHECKPOINT_TOKEN in text,
            f"{g.gate_id} must contain {CHECKPOINT_TOKEN} verbatim, double-quoted.",
        )
        _require(
            "'$EDULLM_CHECKPOINT_DIR'" not in text,
            f"{g.gate_id} single-quotes the token, which suppresses expansion.",
        )
        _require(
            "\\$EDULLM_CHECKPOINT_DIR" not in text,
            f"{g.gate_id} escapes the token, which suppresses expansion.",
        )
        _require(
            "$EDULLM_CHECKPOINT_DIRECTORY" not in text,
            f"{g.gate_id} names $EDULLM_CHECKPOINT_DIRECTORY. Nothing sets that; the guard "
            "carries a (?![A-Za-z0-9_]) lookahead for exactly this near-miss.",
        )
        _require(
            "#" not in text,
            f"{g.gate_id} carries a '#'; the guard drops everything after a comment word.",
        )
        _require(
            g.command()[0] == "bash" and g.command()[1] == "-lc",
            f"{g.gate_id} must wrap its program in `bash -lc` or nothing expands the "
            "variable. argv is never expanded by anything.",
        )
    return "every contracted gate expands the checkpoint dir in a way the guard accepts"


def check_container_overrides_cap(gates: list[Gate]) -> str:
    """A 9,230-byte program passed compile AND an approval before Batch refused it."""
    for g in gates:
        size = g.command_bytes()
        _require(
            size <= CONTAINER_OVERRIDES_CAP_BYTES,
            f"{g.gate_id} command is {size} bytes against a cap of "
            f"{CONTAINER_OVERRIDES_CAP_BYTES}. Batch refuses this AFTER approval; compile "
            "does not check it.",
        )
    largest = max(g.command_bytes() for g in gates)
    return f"largest command is {largest} bytes, {CONTAINER_OVERRIDES_CAP_BYTES} allowed"


def verify_policy(platform: Path) -> str:
    """READ THE POLICY, DO NOT TRUST THE CONSTANT. This check exists because its absence
    cost this lane a wrong answer.

    The first version of this file hard-coded v4's seven thresholds and asserted the ladder
    against them. Policy v5 had already landed. Every assertion passed and the conclusion --
    that A100 gates need an admin -- was false. So the version is now read at runtime and a
    mismatch is a FAILURE, not a warning: a threshold block that silently describes a policy
    the account no longer runs is worse than no block at all, because it reports health.
    """
    path = platform / "config" / "policy.yaml"
    if not path.exists():
        raise CheckFailed(
            f"cannot read {path}. Pass --platform pointing at a clone checked out on "
            "origin/main. NOTE: a clone sitting on a feature branch answers 'Already up to "
            "date' to git pull while arbitrarily far behind main -- that is exactly how this "
            "lane compiled seven gates against a superseded policy."
        )
    version = None
    threshold = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("policy_version:"):
            version = stripped.split(":", 1)[1].strip()
        if stripped.startswith("automatic_below_cost_usd:"):
            threshold = stripped.split(":", 1)[1].strip().strip('"')
    _require(
        version == POLICY_VERSION_WRITTEN_AGAINST,
        f"config/policy.yaml is {version!r} and this file was written against "
        f"{POLICY_VERSION_WRITTEN_AGAINST!r}. Re-read classify_request before trusting a "
        "single approval class in this file.",
    )
    _require(
        threshold == str(AUTOMATIC_BELOW_COST_USD),
        f"automatic_below_cost_usd is {threshold!r}, this file holds "
        f"{AUTOMATIC_BELOW_COST_USD}. The self-release envelope moved.",
    )
    return f"policy {version}, automatic_below_cost_usd={threshold}, matches this file"


def check_no_platform_ceiling_is_assumed(gates: list[Gate]) -> str:
    """THE CHECK THAT REPLACED check_a100_is_exception, AND IT ASSERTS THE OPPOSITE.

    Under v4 an A100 gate had to classify EXCEPTION, and an admin was the guard. Under v5
    there is no rate ceiling and no cost ceiling, so every gate in this ladder classifies
    AUTOMATIC and is released by an environment with no reviewers. Measured 2026-08-06:
    22.7h x 1 on gpu-8xa100 is AUTOMATIC at $498.44.

    So the assertion worth making is no longer "the platform will stop this". It is that
    THIS FILE KNOWS the platform will not, and has flagged every gate the platform would
    wave through but a person should see.
    """
    for g in gates:
        if g.compute_profile == "gpu-8xa100":
            _require(
                g.approval_class == "automatic",
                f"{g.gate_id} on gpu-8xa100 classifies {g.approval_class}, not automatic. "
                "If the platform started refusing this again, re-read policy.yaml -- the "
                "rate ceiling may be back and the sequencing advice here would be stale.",
            )
            _require(
                g.needs_human_decision,
                f"{g.gate_id} is an A100 gate at ${g.worst_case_usd} and is not flagged "
                "needs_human_decision. The platform will release it with nobody asked.",
            )
        _require(
            g.worst_case_usd < SELF_RELEASING_ENVELOPE_USD,
            f"{g.gate_id} worst case ${g.worst_case_usd} is at or above the self-releasing "
            f"envelope of ${SELF_RELEASING_ENVELOPE_USD}, so it becomes ROUTINE and needs a "
            "lead. That is safer, but it changes the sequencing this ladder assumes.",
        )
    flagged = [g.gate_id for g in gates if g.needs_human_decision]
    unattended = [g.gate_id for g in gates if not g.needs_human_decision]
    _require(
        unattended == ["G1", "G2", "G3", "G4"],
        f"the unattended set is {unattended}; it must be exactly G1-G4. Anything more "
        "expensive than the lane ceiling must be a deliberate human decision.",
    )
    return (
        f"every gate self-releases under v5; {flagged} are flagged for human decision "
        f"anyway, {unattended} may run unattended"
    )


def check_g3_provokes(gates: list[Gate]) -> str:
    """G3 must actually reach the failure it exists to characterize."""
    g3 = next(g for g in gates if g.gate_id == "G3")
    text = g3.command()[-1]
    m_int = re.search(r"--save-interval (\d+)", text)
    m_steps = re.search(r"--steps (\d+)", text)
    _require(bool(m_int and m_steps), "G3 must declare both --steps and --save-interval")
    interval, steps = int(m_int.group(1)), int(m_steps.group(1))
    writes = steps // interval
    # FOUR IS THE NUMBER THAT MATTERS. max_checkpoints defaults to 3, the prune removes
    # .metadata.json first, and the workload role carries an explicit IAM Deny on exactly
    # that key -- so the FOURTH save is where a run dies OLMoNetworkError. A gate that
    # writes three times cannot see it.
    _require(
        writes >= 4,
        f"G3 writes only {writes} checkpoints. The max_checkpoints=3 prune failure appears "
        "on the FOURTH save, so a gate writing fewer cannot provoke it and would pass "
        "vacuously.",
    )
    _require(
        g3.attempts == 2,
        "G3 must request 2 attempts: the torn-directory FileExistsError only appears on a "
        "resume, so one attempt cannot observe the failure this gate exists for.",
    )
    _require(
        g3.provokes_failure,
        "G3 must be flagged provokes_failure so a green result is not read as a pass.",
    )
    return f"G3 forces {writes} checkpoint writes over 2 attempts, past the 4th-save prune"


def check_microbatch_is_pinned(gates: list[Gate]) -> str:
    """THE OOM CLASS NOTHING UPSTREAM OF THE GPU CAN SEE.

    `.edullm/train_on_corpus.py:1491` defaults `--rank-microbatch-size` to 16384, which L6
    measured at 131-142% of A100-40GB usable for R3 -- an OOM. `compile_submission.py` reads
    no model config, so it cannot catch this, and OOM gets NO retry (Batch retries fire only
    on `Host EC2*`). So a gate that inherits the default burns its whole ceiling to produce
    one traceback, and the cheapest place to refuse that is here, for free.

    Asserted on any gate running an R2/R3 rung, since those are the ones L6 measured. R0 and
    R1 are smaller; R1 is included anyway because the E-sweep holds active params constant
    and its dispatch buffers grow with E.
    """
    for g in gates:
        text = g.command()[-1]
        heavy = any(FACTORY_WRAPPERS[r] in text for r in ("R1", "R2", "R3"))
        if not heavy:
            continue
        _require(
            "--rank-microbatch-size" in text,
            f"{g.gate_id} runs a large rung and does not pin --rank-microbatch-size. The "
            "default 16384 was measured at 131-142% of A100-40GB usable for R3. This is an "
            "OOM with no retry, and no platform check can see it.",
        )
        m = re.search(r"--rank-microbatch-size (\d+)", text)
        _require(m is not None, f"{g.gate_id} has an unparseable microbatch size")
        assert m is not None
        size = int(m.group(1))
        _require(
            size <= 8192,
            f"{g.gate_id} pins mb={size}. L6 measured 8192 at 78.6% of usable (84.5% worst "
            f"case) and 16384 as an OOM; anything above 8192 is unmeasured.",
        )
    return "every large-rung gate pins a measured-safe --rank-microbatch-size"


def check_rung_selection(gates: list[Gate]) -> str:
    """A DOTTED OVERRIDE CANNOT SELECT A RUNG, AND THIS IS THE PROOF.

    `.edullm/train_on_corpus.py` calls `factory(vocab_size=...)` at line 854 and merges the
    dotlist at line 1091 -- `return config.merge(overrides)`, the LAST statement of
    build_config. `Config.merge` serializes the already-built config, sets nested keys and
    reconstructs. So `model.rung=R3` would either be rejected as an unknown field or set an
    attribute on a model that was already constructed at R3's default, and either way the
    expert count, layer count and d_model are fixed before the override is read.

    Therefore every gate must name its rung in `--model-factory`, which IS read before the
    factory call. Templating the kwarg shape instead would produce seven submissions that
    all silently train the factory's default rung.
    """
    for g in gates:
        text = g.command()[-1]
        _require(
            "--model-factory" in text,
            f"{g.gate_id} names no --model-factory; the default is olmo2_190M, which is not "
            "a Maple model at all.",
        )
        _require(
            g.rung_selection == "wrapper",
            f"{g.gate_id} uses rung_selection={g.rung_selection!r}. Only the wrapper shape "
            "works: a dotted override is merged AFTER the factory has built the model.",
        )
        _require(
            not re.search(r"\bmodel\.rung=", text),
            f"{g.gate_id} tries to select its rung with a dotted override. build_config "
            "merges the dotlist after factory(vocab_size=...) has already returned, so the "
            "rung would be the factory default and the run would look fine.",
        )
        named = [w for w in FACTORY_WRAPPERS.values() if w in text]
        _require(
            len(named) == 1,
            f"{g.gate_id} must name exactly one rung wrapper, found {named}.",
        )
    return "every gate selects its rung via --model-factory, the only mechanism that works"


def check_dataset_and_slugs(gates: list[Gate]) -> str:
    slug = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    # The live dropdown on 2026-08-06. dolma2 is what CLAUDE.md freezes as the corpus.
    listed = {"none", "regmix-10b-v1", "refhq-regmix-5p5b-v2", "olmo-original-30b-v1",
              "olmo-127b-v1", "olmo-150b-dolma2-v1", "math-frontload-100m-v1",
              "formal-proof-premises-500m-v2", "frontload-cl-10b-v1", "fineweb-edu-1b-v6",
              "fineweb2-phase0-equal-bpe-2b-v1", "fineweb2-phase0-equal-superbpe-2b-v1",
              "fineweb2-unimax-bpe-20b-v1", "fineweb2-unimax-superbpe-20b-v1",
              "reservoir-dolma2-v1"}
    for g in gates:
        _require(slug.match(g.experiment) is not None,
                 f"{g.gate_id} experiment {g.experiment!r} is not a slug.")
        _require(g.dataset_release in listed,
                 f"{g.gate_id} names dataset {g.dataset_release!r}, not on the live dropdown. "
                 "An unlisted value is rejected by GitHub before any code runs, so no run id "
                 "is ever minted.")
        # dataset_release: none leaves EDULLM_DATASET_ID unset and the program refuses with
        # exit 64 before it builds anything.
        _require(g.dataset_release != "none",
                 f"{g.gate_id} names dataset_release=none, which leaves the three dataset "
                 "env vars empty and raises exit 64 THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT.")
    return "every gate names a listed corpus and a well-formed experiment slug"


CHECKS = [
    check_provisioned,
    check_bfloat16_capable,
    check_launch_guard,
    check_checkpoint_guard,
    check_container_overrides_cap,
    check_no_platform_ceiling_is_assumed,
    check_g3_provokes,
    check_microbatch_is_pinned,
    check_rung_selection,
    check_dataset_and_slugs,
]


def selfcheck(platform: Path) -> int:
    gates = ladder()
    failures = []
    # The policy read runs FIRST and its failure invalidates every approval-class claim
    # below it, so it is reported separately rather than folded into the list.
    try:
        print(f"PASS verify_policy: {verify_policy(platform)}")
    except CheckFailed as exc:
        failures.append(("verify_policy", str(exc)))
        print(f"FAIL verify_policy: {exc}")
        print("      -> every approval_class printed below is UNRELIABLE until this passes.")
    for check in CHECKS:
        try:
            print(f"PASS {check.__name__}: {check(gates)}")
        except CheckFailed as exc:
            failures.append((check.__name__, str(exc)))
            print(f"FAIL {check.__name__}: {exc}")
    print()
    print(
        f"{'gate':<5} {'platform':<10} {'human?':<7} {'profile':<12} "
        f"{'h':>5} {'att':>4} {'worst':>9}  bytes"
    )
    total = Decimal(0)
    for g in gates:
        total += g.worst_case_usd
        print(
            f"{g.gate_id:<5} {g.approval_class:<10} "
            f"{('YES' if g.needs_human_decision else 'no'):<7} {g.compute_profile:<12} "
            f"{g.runtime_hours:>5} {g.attempts:>4} {'$' + str(g.worst_case_usd):>9}  "
            f"{g.command_bytes()}"
        )
    unattended = sum(g.worst_case_usd for g in gates if not g.needs_human_decision)
    attended = sum(g.worst_case_usd for g in gates if g.needs_human_decision)
    print(f"\nG1-G4, safe to run unattended: ${unattended}")
    print(f"A100 gates, single submissions: ${attended}")
    print(f"whole ladder, single submissions only: ${total}")
    g5 = next(g for g in gates if g.gate_id == "G5").worst_case_usd
    g6 = next(g for g in gates if g.gate_id == "G6").worst_case_usd
    g7 = next(g for g in gates if g.gate_id == "G7").worst_case_usd
    print(
        f"REAL A100 budget: G5 ${g5} + G6 2x${g6} + G7 3x${g7} = "
        f"${g5 + 2 * g6 + 3 * g7} (G6 is a pair, G7 is three rungs; neither is a fan-out)"
    )
    print(
        "\nPOLICY v5 WARNING: every line above says `automatic`, which means the platform "
        f"releases it with NOBODY ASKED. There is no rate ceiling and no cost ceiling any "
        f"more; the self-releasing envelope is ${SELF_RELEASING_ENVELOPE_USD} per submission "
        "and nothing bounds how many are made. The `human?` column is THIS LANE's judgement, "
        "not a platform guard."
    )
    if failures:
        print(f"\n{len(failures)} CHECK(S) FAILED")
        return 1
    print("\nall checks passed")
    return 0


# ---------------------------------------------------------------------------------------
# emit / compile

def emit(gate_id: str, out: Path, commit_sha: str, wandb_project: str,
         digest: str | None, pushed_at: str | None) -> int:
    gate = next((g for g in ladder() if g.gate_id == gate_id), None)
    if gate is None:
        print(f"no gate {gate_id!r}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    inputs_path = out / f"{gate_id}-inputs.json"
    inputs_path.write_text(
        json.dumps(gate.inputs(commit_sha=commit_sha, wandb_project=wandb_project), indent=2)
        + "\n"
    )
    print(f"wrote {inputs_path}")
    if digest:
        if not pushed_at:
            print("--pushed-at is required with --digest", file=sys.stderr)
            return 2
        images_path = out / "images.json"
        images_path.write_text(
            json.dumps(
                {
                    "published": [{"image_digest": digest, "pushed_at": pushed_at}],
                    "image_scan": {
                        "schema_version": 1,
                        "status": "COMPLETE",
                        "scanned_at": pushed_at,
                        "critical": 0, "high": 0, "medium": 0,
                        "low": 0, "informational": 0, "undefined": 0,
                    },
                    # [] PASSES; absent or null REFUSES. Absent and empty are deliberately
                    # different answers: absent means the findings could not be read.
                    "blocking_findings": [],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {images_path}")
    return 0


def compile_gate(gate_id: str, platform: Path, out: Path, commit_sha: str,
                 wandb_project: str, digest: str, pushed_at: str, submitter: str) -> int:
    rc = emit(gate_id, out, commit_sha, wandb_project, digest, pushed_at)
    if rc:
        return rc
    cmd = [
        "uv", "run", "--python", "3.12", "python", "tools/compile_submission.py",
        "--inputs", str(out / f"{gate_id}-inputs.json"),
        "--config-dir", "config",
        "--published-images", str(out / "images.json"),
        "--submitter", submitter,
        "--repository-url", "https://github.com/edu-llm/OLMo-core",
        "--output", str(out / f"{gate_id}-manifest.json"),
        "--summary", str(out / f"{gate_id}-summary.md"),
    ]
    print("+ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=platform, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    print(f"compile_submission.py exit={proc.returncode} "
          f"({'compiled' if proc.returncode == 0 else 'refused on the merits' if proc.returncode == 1 else 'inputs unreadable, NOT a judgement'})")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("selfcheck")
    # Defaults to a worktree pinned at origin/main rather than to the shared clone, because
    # the shared clone was on a feature branch and answered "Already up to date" to a pull.
    sc.add_argument("--platform", type=Path, default=Path("/tmp/l7-platform-main"))
    for name in ("emit", "compile"):
        p = sub.add_parser(name)
        p.add_argument("--gate", required=True)
        p.add_argument("--out", type=Path, default=Path("/tmp/pv"))
        p.add_argument("--commit-sha", default="d15579237b89887e709e460123a2f8ff99aaacc0")
        p.add_argument("--wandb-project", default="maple-scaledown")
        p.add_argument("--digest", default=None)
        p.add_argument("--pushed-at", default=None)
        if name == "compile":
            p.add_argument("--platform", type=Path, default=Path("/tmp/l7-platform-main"))
            p.add_argument("--submitter", default="ericrcwu001")
    args = ap.parse_args()
    if args.cmd == "selfcheck":
        return selfcheck(args.platform)
    if args.cmd == "emit":
        return emit(args.gate, args.out, args.commit_sha, args.wandb_project,
                    args.digest, args.pushed_at)
    if not args.digest or not args.pushed_at:
        print("compile needs --digest and --pushed-at; a defaulted digest is an assertion "
              "nobody re-checked", file=sys.stderr)
        return 2
    return compile_gate(args.gate, args.platform, args.out, args.commit_sha,
                        args.wandb_project, args.digest, args.pushed_at, args.submitter)


if __name__ == "__main__":
    sys.exit(main())
