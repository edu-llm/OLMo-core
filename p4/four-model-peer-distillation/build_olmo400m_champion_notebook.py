#!/usr/bin/env python3
"""Build the OLMo-400M four-peer peer-learning experiment notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "OLMo400M_four_peer_peer_learning.ipynb"


def source(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        r"""
        # Can four OLMo-400M peers beat distillation from a larger teacher?

        This notebook is written around the actual sell, not merely the peer mechanism:

        > **Can four complementary 400M students produce a better deployable 400M model than
        > four otherwise identical 400M students distilled from a demonstrably stronger,
        > genuinely larger teacher—especially on valid new strategies and sealed
        > compositional generalization?**

        The idea is worth testing, but it is a high-risk, high-information experiment rather
        than a likely result. Peer/on-policy training, smaller students exceeding prompted
        teachers, and multiagent specialization all have precedents. No located primary joins
        the required pieces: four equal ~400M decoder peers, a stronger larger-teacher
        comparator, symmetric four-student selection, equal exposure and all-in cost
        accounting, a one-model deployment endpoint, and item-level teacher-external novelty.
        A positive result would therefore be new; the literature does not let us assume it.

        The notebook **does not submit to FarmShare and does not train by default**. The
        training entry point remains locked unless `ALLOW_OLMO400M_TRAINING` is explicitly
        set to `I_UNDERSTAND_THIS_RUNS_OPTIMIZATION`. With the default environment it only
        builds and audits deterministic manifests.

        ## Validation verdict and scientific correction

        The run is staged so an attractive story cannot outrun the mechanism:

        1. **Stage 0 — prerequisite gates.** The four warmed students must have measurable
           complementary correct answers. The primary teacher should be a checkpoint-stage-
           matched OLMo-compatible model near 1B (roughly the capacity-gap prior for a 400M
           student), must be locally pinned, and must independently beat the best 400M peer
           on two calibration shards. A 7B teacher is a separately gated secondary stress
           test, not a replacement for the 1B primary. If these gates fail, there is no valid
           experiment.
        2. **Stage 1 — mechanism screen.** `peer_frr_onpolicy` must beat matched ordinary
           training and `self_snapshot_op` without damaging specialty, shift, or held-out
           language retention. This asks whether information from another peer matters at all.
        3. **Stage 2 — the championship.** Fork the same four warmed checkpoints into
           `peer_frr_onpolicy`, `large_teacher_single`, `large_teacher_diverse`, and
           `gold_private_equal_cost`. The teacher-diverse arm receives the same attempted and
           accepted verified target counts as the peer arm; every arm uses the same student
           updates, hard data, selection rule, and sealed audit.
        4. **Stage 3 — novelty audit.** Quality comes first. Only correct/test-passing outputs
           can count as novel. Measure teacher-external solves, held-out specialty
           compositions, and valid strategy coverage at fixed `k`; open-ended creativity is a
           blinded, human-calibrated secondary endpoint.

        The headline estimand is the selected peer-trained 400M minus the symmetrically
        selected `large_teacher_diverse` 400M—not the peer ensemble and not the raw teacher.
        Beating the raw larger teacher is a separately reported moonshot. A router or
        best-of-four population win is also useful, but it is a different deployment claim.

        Two fairness views are mandatory because no single run can simultaneously equalize
        supervision exposure and the cost of generating it: (a) an exposure-matched causal
        contrast with equal accepted targets, target tokens, student updates, and selection;
        and (b) a performance-versus-all-in-compute frontier including every peer/teacher
        forward, rejected sample, verifier call, failed run, and selection evaluation.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import contextlib
        import dataclasses
        import fractions
        import hashlib
        import json
        import math
        import os
        import random
        import re
        import shutil
        import statistics
        import time
        from dataclasses import dataclass, field, replace
        from pathlib import Path
        from typing import Any, Iterable, Iterator, Mapping, Sequence

        import numpy as np

        try:
            import torch
            from torch.nn import functional as F
            from torch.utils.data import DataLoader
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            torch = None
            F = None
            DataLoader = None
            AutoModelForCausalLM = None
            AutoTokenizer = None


        TRAINING_UNLOCK = "I_UNDERSTAND_THIS_RUNS_OPTIMIZATION"


        @dataclass(frozen=True)
        class ExperimentConfig:
            protocol_version: str = "olmo400m-four-peer-v4-larger-teacher-championship"
            model_path: str = os.environ.get("OLMO_400M_MODEL", "")
            student_checkpoint_stage: str = os.environ.get(
                "OLMO_400M_STAGE", "unspecified"
            )
            larger_teacher_model_path: str = os.environ.get(
                "OLMO_LARGER_TEACHER_MODEL", ""
            )
            larger_teacher_checkpoint_stage: str = os.environ.get(
                "OLMO_LARGER_TEACHER_STAGE", "matched_to_student"
            )
            output_dir: Path = Path(os.environ.get(
                "OLMO400M_EXPERIMENT_DIR", "./olmo400m_peer_learning_outputs"
            )).resolve()
            local_files_only: bool = True
            allow_training: bool = (
                os.environ.get("ALLOW_OLMO400M_TRAINING", "") == TRAINING_UNLOCK
            )

            n_peers: int = 4
            n_rounds: int = 3
            skills: tuple[str, ...] = (
                "integer_arithmetic",
                "linear_equation",
                "fraction_addition",
                "percent_of",
                "ratio",
                "arithmetic_sequence",
                "modular_arithmetic",
                "manhattan_distance",
            )
            skill_groups: tuple[tuple[str, ...], ...] = (
                ("integer_arithmetic", "linear_equation"),
                ("fraction_addition", "percent_of"),
                ("ratio", "arithmetic_sequence"),
                ("modular_arithmetic", "manhattan_distance"),
            )
            composition_skills: tuple[str, ...] = (
                "percent_then_linear",
                "fraction_then_scale",
                "sequence_then_modulo",
                "distance_then_percent",
            )

            # Calibrate `order_only` against `balanced_specialists`; freeze one mode before
            # confirmatory runs. 40/20/20/20 is a design dose, not a literature constant.
            diversity_mode: str = "balanced_specialists"
            specialist_home_fraction: float = 0.40

            # Confirmatory defaults. Smoke mode below replaces these mechanically.
            warmup_examples_per_peer: int = 32_768
            exchange_examples_per_round: int = 32_768
            private_replay_examples_per_round: int = 8_192
            election_examples_per_round: int = 1_024
            promotion_examples_per_round: int = 2_048
            final_audit_examples: int = 4_096
            shift_audit_examples: int = 4_096
            composition_audit_examples: int = 2_048
            retention_max_tokens: int = 1_000_000

            # Placeholder only: seed-0 must profile all draft+synthesis generation and freeze
            # the fraction of additional gold work that spends the same accelerator budget.
            equal_cost_extra_fraction: float = float(os.environ.get(
                "OLMO400M_EQUAL_COST_EXTRA_FRACTION", "0.25"
            ))

            max_length: int = 256
            generation_max_new_tokens: int = 64
            peer_draft_temperature: float = 0.8
            peer_draft_top_p: float = 0.95
            larger_teacher_temperature: float = 0.8
            larger_teacher_top_p: float = 0.95
            # Two peer rollout banks x four peers. LT-diverse must receive the same number
            # of attempted samples per item, not one greedy trace against eight peer draws.
            larger_teacher_diverse_attempts: int = 8
            synthesis_max_new_tokens: int = 96
            synthesis_fraction: float = 0.20
            minimum_rescue_edge_fraction: float = 0.05
            micro_batch_size: int = 8
            grad_accum_steps: int = 4
            learning_rate: float = 2e-5
            min_learning_rate: float = 2e-6
            weight_decay: float = 0.1
            adam_beta1: float = 0.9
            adam_beta2: float = 0.95
            grad_clip: float = 1.0
            warmup_fraction: float = 0.03

            # Direct same-level LLM evidence favors a mixed loss, not pure imitation.
            kd_alpha: float = 0.20
            kd_temperature: float = 1.0
            pilot_kd_alpha_grid: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
            pilot_kd_temperature_grid: tuple[float, ...] = (1.0, 2.0)
            teacher_topk: int = 32
            peer_kd_scope: str = "answer_only"  # Full rationale KD requires verified continuation prefixes.
            require_itemwise_teacher_correctness: bool = True
            topk_mae_limit: float = 0.01
            topk_p99_limit: float = 0.05

            promotion_margin: float = 0.01
            promotion_confidence: float = 0.95
            promotion_bootstrap_resamples: int = 10_000
            final_noninferiority_margin: float = 0.01
            retention_nll_relative_margin: float = 0.02
            complementarity_headroom_min: float = 0.02
            unique_correct_fraction_min: float = 0.01
            capability_floor: float = 0.20
            capability_ceiling: float = 0.85
            larger_teacher_min_parameter_ratio: float = 2.0
            larger_teacher_max_primary_parameter_ratio: float = 4.0
            allow_extreme_teacher_stress_test: bool = (
                os.environ.get("OLMO_ALLOW_EXTREME_TEACHER", "") == "SECONDARY_ONLY"
            )
            larger_teacher_calibration_margin: float = 0.03
            larger_teacher_superiority_confidence: float = 0.95
            pilot_effect_min: float = 0.01
            diversity_retention_fraction_min: float = 0.75
            max_consecutive_no_promotions: int = 2
            enable_futility_stop: bool = False  # False for fixed-budget quality; True for cost-to-threshold.

            calibration_seed: int = 0
            excluded_pilot_seeds: tuple[int, ...] = (0, 1, 2, 3)
            # Freeze N from excluded paired-pilot variance before looking at these outcomes.
            # Eight pairs are only a large-effect screen; preserve a 16-seed pool.
            confirmatory_seeds: tuple[int, ...] = (
                13, 29, 47, 71, 101, 137, 173, 211,
                251, 293, 337, 379, 421, 463, 509, 557,
            )
            primary_screen_arms: tuple[str, ...] = (
                "gold_private_equal_cost",
                "self_snapshot_op",
                "peer_frr_onpolicy",
            )
            championship_arms: tuple[str, ...] = (
                "gold_private_equal_cost",
                "self_snapshot_op",
                "peer_frr_onpolicy",
                "large_teacher_single",
                "large_teacher_diverse",
            )
            arms: tuple[str, ...] = (
                "gold_private",
                "gold_private_equal_cost",
                "self_snapshot_op",
                "frr_goldprefix",
                "frr_onpolicy",
                "peer_frr_onpolicy",
                "global_champion_goldprefix",
                "unfiltered_mutual_mean",
                "large_teacher_single",
                "large_teacher_diverse",
            )
            exploratory_arms: tuple[str, ...] = ("peer_context_synthesis",)

            expected_parameter_count: int = 400_000_000
            parameter_tolerance_fraction: float = 0.25
            allowed_olmo_model_types: tuple[str, ...] = ("olmo", "olmo2")
            fixed_k_samples: int = 16
            open_ended_creativity_secondary_only: bool = True
            retention_text_jsonl: str = os.environ.get("OLMO400M_RETENTION_JSONL", "")
            require_retention_corpus: bool = True
            gpu_hour_rate_usd: float | None = None


        def b200_10h_config(cfg: ExperimentConfig) -> ExperimentConfig:
            # Compressed championship profile for a hard 10-hour B200 allocation.
            requested_gpus = int(os.environ.get("OLMO400M_B200_GPUS", "4"))
            if requested_gpus not in {2, 3, 4}:
                raise ValueError("OLMO400M_B200_GPUS must be 2, 3, or 4 for b200_10h")
            seeds = (13, 29, 47, 71)[:requested_gpus]
            return replace(
                cfg,
                protocol_version=(
                    cfg.protocol_version + "-b200-10h-compressed-championship"
                ),
                n_rounds=2,
                warmup_examples_per_peer=8_192,
                exchange_examples_per_round=8_192,
                private_replay_examples_per_round=2_048,
                election_examples_per_round=512,
                promotion_examples_per_round=1_024,
                final_audit_examples=2_048,
                shift_audit_examples=1_024,
                composition_audit_examples=1_024,
                retention_max_tokens=250_000,
                promotion_bootstrap_resamples=5_000,
                excluded_pilot_seeds=(0,),
                confirmatory_seeds=seeds,
                fixed_k_samples=8,
            )


        def apply_budget_profile(cfg: ExperimentConfig) -> ExperimentConfig:
            profile = os.environ.get("OLMO400M_BUDGET_PROFILE", "").strip().lower()
            if profile in {"", "full"}:
                return cfg
            if profile == "b200_10h":
                return b200_10h_config(cfg)
            raise ValueError(
                "Unknown OLMO400M_BUDGET_PROFILE. Supported values: full, b200_10h"
            )


        CFG = apply_budget_profile(ExperimentConfig())
        print(json.dumps({
            "protocol": CFG.protocol_version,
            "budget_profile": os.environ.get("OLMO400M_BUDGET_PROFILE", "full") or "full",
            "training_unlocked": CFG.allow_training,
            "model_path_set": bool(CFG.model_path),
            "larger_teacher_path_set": bool(CFG.larger_teacher_model_path),
            "output_dir": str(CFG.output_dir),
            "n_rounds": CFG.n_rounds,
            "confirmatory_seeds": list(CFG.confirmatory_seeds),
        }, indent=2))
        """
    ),
    markdown(
        r"""
        ## Frozen estimands and comparisons

        Let `A_s,r,d` be sealed-final accuracy for meta-seed `s`, arm `r`, and 400M
        checkpoint `d` selected by that arm's frozen policy before the audit. The primary
        exposure-matched estimand is

        \[
        \Delta_{P-LD}=\mathbb{E}_s[A_{s,\mathrm{peer},d_P}-
        A_{s,\mathrm{LT\text{-}diverse},d_L}].
        \]

        Both arms start from the same four warmed checkpoint bytes. Both train four students,
        see the same hard examples, use the same optimizer/update schedule, receive the same
        accepted auxiliary-target and target-token quotas, and select one deployable 400M
        checkpoint by the same independent development policy. `large_teacher_diverse`, not
        a single greedy teacher trace, is the decisive comparator. It gets as many attempted
        verified generations as the two-rollout four-peer bank.

        Three claims are kept separate:

        - **Core:** selected `peer_frr_onpolicy` 400M beats selected
          `large_teacher_diverse` 400M and both ordinary/self controls.
        - **Novelty:** on the full sealed test, the selected peer has more valid solves for
          which matched fixed-`k` banks from *all four pre peers and the raw larger teacher*
          contain no valid solution. This is a secondary interaction unless powered in
          advance; teacher-greedy failure is not enough.
        - **Moonshot:** the selected peer-trained 400M beats the raw larger teacher under the
          identical prompt, decoder, sample/token budget, verifier, and tool policy.

        A population oracle, router, debate, or best-of-four result is reported separately.
        It cannot establish the single-400M claim because it spends more inference or stored
        capacity.

        The all-in resource estimand is a frontier, not a hidden assertion of equality:

        \[
        Q_r(B)=\max\{A_{s,r}:\text{all peer/teacher generation, rejection, verification,
        student training, selection, and evaluation cost}\leq B\}.
        \]

        Exposure matching identifies topology; the frontier answers which method is worth
        buying. Report measured FLOPs where available plus accelerator-hours, processed and
        decoded tokens, latency, and dollars. Shared base pretraining is a stated sunk cost.

        Promotion uses two distinct shards: an election shard nominates a challenger, then a
        promotion shard tests it against the incumbent with paired item outcomes. The final,
        structure-shift, and creativity audits never select teachers, checkpoints,
        hyperparameters, prompts, decoding settings, or stopping times. Confirmatory
        inference is over paired **population-training seeds**. One four-student population is
        one replicate; neither the four members nor thousands of audit items inflate outer
        `n`. The full untouched test is primary; pre-peer-wrong and teacher-wrong subsets are
        secondary and defined with matched fixed-`k` output banks.
        """
    ),
    code(
        r"""
        def canonical_json(value: Any) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


        def sha256_bytes(value: bytes) -> str:
            return hashlib.sha256(value).hexdigest()


        def sha256_json(value: Any) -> str:
            return sha256_bytes(canonical_json(value).encode("utf-8"))


        def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
            h = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(chunk_size):
                    h.update(chunk)
            return h.hexdigest()


        def atomic_json(path: Path, value: Any) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(path)


        def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(canonical_json(dict(row)) + "\n")
            tmp.replace(path)


        def read_jsonl(path: Path) -> list[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]


        def seeded_rng(*parts: Any) -> random.Random:
            digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()
            return random.Random(int.from_bytes(digest[:8], "big"))


        def require_training(cfg: ExperimentConfig) -> None:
            if not cfg.allow_training:
                raise RuntimeError(
                    "Training is locked. Set ALLOW_OLMO400M_TRAINING=" + TRAINING_UNLOCK
                    + " only in the intended compute environment."
                )
            if not cfg.model_path:
                raise RuntimeError("Set OLMO_400M_MODEL to the pinned local checkpoint path.")
            if torch is None or AutoModelForCausalLM is None:
                raise RuntimeError("Install the pinned PyTorch/Transformers environment first.")
            if cfg.peer_kd_scope != "answer_only":
                raise RuntimeError(
                    "Only answer_only peer KD is authorized. Full-rationale KD on student "
                    "prefixes requires separately generated and verified teacher continuations."
                )
            if cfg.n_peers != 4:
                raise RuntimeError("This preregistered championship requires exactly four peers")
            if cfg.larger_teacher_diverse_attempts != 2 * cfg.n_peers:
                raise RuntimeError(
                    "LT-diverse attempts must match two frozen rollout banks across all peers"
                )


        @dataclass
        class CostLedger:
            path: Path
            arm: str
            seed: int

            def emit(self, **event: Any) -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "event_id": sha256_json([self.arm, self.seed, time.time_ns(), event]),
                    "arm": self.arm,
                    "seed": self.seed,
                    "created_ns": time.time_ns(),
                    **event,
                }
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical_json(row) + "\n")
        """
    ),
    markdown(
        r"""
        ## Deterministic, machine-verifiable target capability

        The first run uses one capability—short procedural mathematics—with eight balanced
        subskills. This is deliberate: champion eligibility and final answers can be checked
        without asking another language model to judge them. Training, election, promotion,
        final-audit, and template-shift families have separate renderers and counter-based
        random streams. A publication run should place final/shift generation behind a
        separate evaluation owner; this notebook enforces logical separation and hashes but
        cannot create organizational secrecy by itself.
        """
    ),
    code(
        r"""
        def answer_text(value: int | fractions.Fraction) -> str:
            if isinstance(value, fractions.Fraction):
                return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
            return str(int(value))


        def normalize_answer(text: str) -> str | None:
            matches = re.findall(r"(?i)Answer\s*:\s*([-+]?\d+(?:\s*/\s*[-+]?\d+)?)", text)
            candidate = matches[-1] if matches else None
            if candidate is None:
                loose = re.findall(r"[-+]?\d+(?:\s*/\s*[-+]?\d+)?", text)
                candidate = loose[-1] if loose else None
            if candidate is None:
                return None
            try:
                frac = fractions.Fraction(candidate.replace(" ", ""))
            except (ValueError, ZeroDivisionError):
                return None
            return answer_text(frac)


        def render_problem(skill: str, rng: random.Random, family: str) -> tuple[str, str, str, dict[str, int]]:
            large = family in {"shift", "audit"}
            if skill == "integer_arithmetic":
                a = rng.randint(12 if large else 2, 80 if large else 30)
                b = rng.randint(3, 20)
                c = rng.randint(-40 if large else -12, 40 if large else 12)
                value = a * b + c
                prompts = {
                    "train": f"Compute {a} times {b}, then add {c}.",
                    "election": f"What is ({a} × {b}) + ({c})?",
                    "promotion": f"Evaluate the integer expression {a}*{b}+({c}).",
                    "audit": f"Starting with the product of {a} and {b}, increase it by {c}. Give the result.",
                    "shift": f"A box holds {a} rows of {b} objects and then {abs(c)} objects are {'added' if c >= 0 else 'removed'}. How many remain?",
                }
                rationale = f"{a} × {b} = {a*b}; {a*b} + ({c}) = {value}."
                return prompts[family], rationale, str(value), {"a": a, "b": b, "c": c}

            if skill == "linear_equation":
                a = rng.choice([x for x in range(-12, 13) if x not in {-1, 0, 1}])
                x = rng.randint(-30 if large else -12, 30 if large else 12)
                b = rng.randint(-40 if large else -15, 40 if large else 15)
                c = a * x + b
                prompts = {
                    "train": f"Solve for x: {a}x + ({b}) = {c}.",
                    "election": f"Find the integer x satisfying {a}x + {b} = {c}.",
                    "promotion": f"Which x makes {a}·x + ({b}) equal {c}?",
                    "audit": f"Determine the unknown in the equation {c} = {b} + {a}x.",
                    "shift": f"After multiplying a number by {a} and adding {b}, the result is {c}. What was the number?",
                }
                rationale = f"Subtract {b}: {a}x = {c-b}. Divide by {a}: x = {x}."
                return prompts[family], rationale, str(x), {"a": a, "b": b, "c": c, "x": x}

            if skill == "fraction_addition":
                d1, d2 = rng.randint(2, 18), rng.randint(2, 18)
                n1, n2 = rng.randint(1, d1 - 1), rng.randint(1, d2 - 1)
                value = fractions.Fraction(n1, d1) + fractions.Fraction(n2, d2)
                prompts = {
                    "train": f"Add and simplify: {n1}/{d1} + {n2}/{d2}.",
                    "election": f"What reduced fraction equals {n1}/{d1} plus {n2}/{d2}?",
                    "promotion": f"Combine the fractions {n1}/{d1} and {n2}/{d2}; reduce the result.",
                    "audit": f"Give the simplest rational number for the sum of {n1}/{d1} and {n2}/{d2}.",
                    "shift": f"A container is {n1}/{d1} full and receives another {n2}/{d2} of its capacity. What fraction is filled in total?",
                }
                common_num = n1 * d2 + n2 * d1
                common_den = d1 * d2
                rationale = f"Use denominator {common_den}: the numerator is {common_num}; reducing gives {answer_text(value)}."
                return prompts[family], rationale, answer_text(value), {"n1": n1, "d1": d1, "n2": n2, "d2": d2}

            if skill == "percent_of":
                p = rng.choice([5, 10, 20, 25, 40, 50, 60, 75, 80])
                unit = 20 if p == 5 else 10 if p % 10 == 0 else 4
                n = unit * rng.randint(3, 30 if large else 15)
                value = n * p // 100
                prompts = {
                    "train": f"Find {p}% of {n}.",
                    "election": f"What quantity is {p} percent of {n}?",
                    "promotion": f"Calculate ({p}/100) × {n}.",
                    "audit": f"Take {p} parts per hundred of {n}. What is the result?",
                    "shift": f"A class completes {p}% of {n} exercises. How many exercises is that?",
                }
                rationale = f"{p}% = {p}/100, and ({p}/100) × {n} = {value}."
                return prompts[family], rationale, str(value), {"p": p, "n": n}

            if skill == "ratio":
                a, b = rng.randint(2, 12), rng.randint(2, 12)
                k = rng.randint(2, 30 if large else 12)
                left, right = a * k, b * k
                prompts = {
                    "train": f"The ratio of red to blue is {a}:{b}. If there are {left} red items, how many blue items are there?",
                    "election": f"A quantity pair is in ratio {a} to {b}; the first is {left}. Find the second.",
                    "promotion": f"Complete the equivalent ratio {a}:{b} = {left}:?.",
                    "audit": f"Two counts scale as {a}/{b}. When the first count equals {left}, what is the other count?",
                    "shift": f"A recipe uses {a} cups of ingredient A per {b} cups of B. With {left} cups of A, how many cups of B are needed?",
                }
                rationale = f"{left} ÷ {a} = {k}, so scale {b} by {k}: {right}."
                return prompts[family], rationale, str(right), {"a": a, "b": b, "k": k}

            if skill == "arithmetic_sequence":
                start = rng.randint(-30 if large else -12, 30 if large else 12)
                step = rng.choice([x for x in range(-12, 13) if x != 0])
                seq = [start + step * i for i in range(5)]
                value = start + step * 5
                shown = ", ".join(map(str, seq))
                prompts = {
                    "train": f"Give the next term: {shown}.",
                    "election": f"Continue this constant-difference sequence by one term: {shown}.",
                    "promotion": f"What follows {shown} if the step stays fixed?",
                    "audit": f"The list {shown} is arithmetic. State term six.",
                    "shift": f"A counter starts at {start} and changes by {step} each turn. After the displayed five values ({shown}), what is next?",
                }
                rationale = f"Consecutive terms differ by {step}; {seq[-1]} + ({step}) = {value}."
                return prompts[family], rationale, str(value), {"start": start, "step": step}

            if skill == "modular_arithmetic":
                m = rng.randint(3, 19)
                a = rng.randint(10, 120 if large else 60)
                b = rng.randint(2, 25)
                c = rng.randint(0, 30)
                value = (a * b + c) % m
                prompts = {
                    "train": f"Compute ({a}×{b}+{c}) mod {m}.",
                    "election": f"What remainder results when {a*b+c} is divided by {m}?",
                    "promotion": f"Reduce {a}*{b}+{c} modulo {m}.",
                    "audit": f"Find the least nonnegative residue of {a*b+c} under modulus {m}.",
                    "shift": f"A clock has {m} positions. Starting at 0 and moving {a*b+c} steps, at which position does it stop?",
                }
                rationale = f"{a}×{b}+{c} = {a*b+c}; dividing by {m} leaves remainder {value}."
                return prompts[family], rationale, str(value), {"a": a, "b": b, "c": c, "m": m}

            if skill == "manhattan_distance":
                bound = 40 if large else 15
                x1, y1, x2, y2 = [rng.randint(-bound, bound) for _ in range(4)]
                value = abs(x1 - x2) + abs(y1 - y2)
                prompts = {
                    "train": f"Find the Manhattan distance from ({x1},{y1}) to ({x2},{y2}).",
                    "election": f"Using |Δx|+|Δy|, compute the distance between ({x1},{y1}) and ({x2},{y2}).",
                    "promotion": f"How far apart are ({x1},{y1}) and ({x2},{y2}) in taxicab distance?",
                    "audit": f"Evaluate |{x1}-{x2}| + |{y1}-{y2}|.",
                    "shift": f"On a square street grid, how many blocks are in a shortest route from ({x1},{y1}) to ({x2},{y2})?",
                }
                rationale = f"|{x1}-{x2}| + |{y1}-{y2}| = {abs(x1-x2)} + {abs(y1-y2)} = {value}."
                return prompts[family], rationale, str(value), {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

            raise KeyError(skill)


        def render_composition_problem(
            skill: str, rng: random.Random
        ) -> tuple[str, str, str, dict[str, int]]:
            if skill == "percent_then_linear":
                p = rng.choice([25, 50])
                a, x, b = rng.randint(2, 9), rng.randint(2, 20), rng.randint(-10, 20)
                rhs = a * x + b
                total = rhs * 100 // p
                prompt = (
                    f"First find {p}% of {total}. Then solve {a}x + ({b}) equal to "
                    "that result. What is x?"
                )
                rationale = (
                    f"{p}% of {total} is {rhs}; solving {a}x + ({b}) = {rhs} gives x = {x}."
                )
                return prompt, rationale, str(x), {"p": p, "total": total, "a": a, "b": b, "x": x}
            if skill == "fraction_then_scale":
                d1, d2 = rng.randint(2, 12), rng.randint(2, 12)
                n1, n2 = rng.randint(1, d1 - 1), rng.randint(1, d2 - 1)
                k = rng.randint(2, 9)
                summed = fractions.Fraction(n1, d1) + fractions.Fraction(n2, d2)
                value = summed * k
                prompt = f"Add {n1}/{d1} and {n2}/{d2}, reduce it, then scale the result by {k}."
                rationale = (
                    f"The reduced sum is {answer_text(summed)}; multiplying by {k} gives "
                    f"{answer_text(value)}."
                )
                return prompt, rationale, answer_text(value), {
                    "n1": n1, "d1": d1, "n2": n2, "d2": d2, "k": k,
                }
            if skill == "sequence_then_modulo":
                start = rng.randint(-20, 20)
                step = rng.choice([value for value in range(-9, 10) if value != 0])
                shown = [start + step * index for index in range(5)]
                next_value = start + 5 * step
                modulus = rng.randint(3, 17)
                answer = next_value % modulus
                prompt = (
                    f"Continue the arithmetic sequence {', '.join(map(str, shown))} by one "
                    f"term, then give that term modulo {modulus}."
                )
                rationale = (
                    f"The step is {step}, so the next term is {next_value}; modulo {modulus} "
                    f"the result is {answer}."
                )
                return prompt, rationale, str(answer), {
                    "start": start, "step": step, "modulus": modulus,
                }
            if skill == "distance_then_percent":
                dx, dy = 4 * rng.randint(1, 10), 4 * rng.randint(0, 10)
                x1, y1 = rng.randint(-20, 20), rng.randint(-20, 20)
                x2, y2 = x1 + rng.choice([-dx, dx]), y1 + rng.choice([-dy, dy])
                p = rng.choice([25, 50, 75])
                distance = dx + dy
                answer = distance * p // 100
                prompt = (
                    f"Find the Manhattan distance from ({x1},{y1}) to ({x2},{y2}), then "
                    f"report {p}% of that distance."
                )
                rationale = (
                    f"The distance is |{x1}-{x2}|+|{y1}-{y2}|={distance}; {p}% is {answer}."
                )
                return prompt, rationale, str(answer), {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2, "p": p,
                }
            raise KeyError(skill)


        def generate_composition_partition(
            cfg: ExperimentConfig,
            split: str,
            size: int,
            used_signatures: set[str],
        ) -> list[dict[str, Any]]:
            rows = []
            candidate_index = 0
            while len(rows) < size:
                skill = cfg.composition_skills[len(rows) % len(cfg.composition_skills)]
                rng = seeded_rng(cfg.protocol_version, split, "composition", skill, candidate_index)
                candidate_index += 1
                prompt, rationale, answer, params = render_composition_problem(skill, rng)
                signature = sha256_json([skill, params])
                if signature in used_signatures:
                    continue
                used_signatures.add(signature)
                target = f"Reasoning: {rationale}\nAnswer: {answer}"
                row = {
                    "protocol_version": cfg.protocol_version,
                    "split": split,
                    "family": "composition",
                    "skill": skill,
                    "prompt": prompt,
                    "target": target,
                    "answer": answer,
                    "params": params,
                    "signature": signature,
                    "item_index": len(rows),
                }
                row["artifact_id"] = sha256_json(row)
                rows.append(row)
            return rows


        def generate_partition(
            cfg: ExperimentConfig,
            split: str,
            family: str,
            size: int,
            used_signatures: set[str],
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            candidate_index = 0
            while len(rows) < size:
                skill = cfg.skills[len(rows) % len(cfg.skills)]
                rng = seeded_rng(cfg.protocol_version, split, family, skill, candidate_index)
                prompt, rationale, answer, params = render_problem(skill, rng, family)
                signature = sha256_json([skill, params])
                candidate_index += 1
                if signature in used_signatures:
                    continue
                used_signatures.add(signature)
                target = f"Reasoning: {rationale}\nAnswer: {answer}"
                row = {
                    "protocol_version": cfg.protocol_version,
                    "split": split,
                    "family": family,
                    "skill": skill,
                    "prompt": prompt,
                    "target": target,
                    "answer": answer,
                    "params": params,
                    "signature": signature,
                    "item_index": len(rows),
                }
                row["artifact_id"] = sha256_json(row)
                rows.append(row)
            return rows


        def generate_weighted_skill_partition(
            cfg: ExperimentConfig,
            split: str,
            family: str,
            size: int,
            skill_weights: Mapping[str, float],
            used_signatures: set[str],
        ) -> list[dict[str, Any]]:
            if set(skill_weights) != set(cfg.skills):
                raise ValueError("skill_weights must name every configured skill exactly once")
            if not math.isclose(sum(skill_weights.values()), 1.0, abs_tol=1e-9):
                raise ValueError("skill_weights must sum to one")
            raw = {skill: size * skill_weights[skill] for skill in cfg.skills}
            counts = {skill: math.floor(raw[skill]) for skill in cfg.skills}
            remainder = size - sum(counts.values())
            order = sorted(cfg.skills, key=lambda s: (-(raw[s] - counts[s]), cfg.skills.index(s)))
            for skill in order[:remainder]:
                counts[skill] += 1

            schedule: list[str] = []
            remaining = dict(counts)
            while sum(remaining.values()):
                for skill in cfg.skills:
                    if remaining[skill] > 0:
                        schedule.append(skill)
                        remaining[skill] -= 1

            rows: list[dict[str, Any]] = []
            candidate_by_skill = {skill: 0 for skill in cfg.skills}
            for item_index, skill in enumerate(schedule):
                while True:
                    candidate_index = candidate_by_skill[skill]
                    candidate_by_skill[skill] += 1
                    rng = seeded_rng(cfg.protocol_version, split, family, skill, candidate_index)
                    prompt, rationale, answer, params = render_problem(skill, rng, family)
                    signature = sha256_json([skill, params])
                    if signature not in used_signatures:
                        break
                used_signatures.add(signature)
                target = f"Reasoning: {rationale}\nAnswer: {answer}"
                row = {
                    "protocol_version": cfg.protocol_version,
                    "split": split,
                    "family": family,
                    "skill": skill,
                    "prompt": prompt,
                    "target": target,
                    "answer": answer,
                    "params": params,
                    "signature": signature,
                    "item_index": item_index,
                }
                row["artifact_id"] = sha256_json(row)
                rows.append(row)
            return rows


        def specialist_weights(cfg: ExperimentConfig, peer: int) -> dict[str, float]:
            if len(cfg.skill_groups) != cfg.n_peers:
                raise ValueError("balanced_specialists requires one home skill group per peer")
            flattened = [skill for group in cfg.skill_groups for skill in group]
            if sorted(flattened) != sorted(cfg.skills) or len(flattened) != len(set(flattened)):
                raise ValueError("skill_groups must partition configured skills")
            home = set(cfg.skill_groups[peer])
            other = set(cfg.skills) - home
            return {
                skill: (
                    cfg.specialist_home_fraction / len(home)
                    if skill in home
                    else (1 - cfg.specialist_home_fraction) / len(other)
                )
                for skill in cfg.skills
            }


        def materialize_data(cfg: ExperimentConfig) -> dict[str, Any]:
            data_dir = cfg.output_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            used: set[str] = set()
            partitions: dict[str, list[dict[str, Any]]] = {}

            if cfg.diversity_mode in {"exact_clone", "order_only"}:
                partitions["warmup_shared"] = generate_partition(
                    cfg, "warmup_shared", "train", cfg.warmup_examples_per_peer, used
                )
            elif cfg.diversity_mode == "balanced_specialists":
                for peer in range(cfg.n_peers):
                    name = f"warmup_peer_{peer}"
                    partitions[name] = generate_weighted_skill_partition(
                        cfg,
                        name,
                        "train",
                        cfg.warmup_examples_per_peer,
                        specialist_weights(cfg, peer),
                        used,
                    )
            else:
                raise ValueError(f"Unknown diversity_mode={cfg.diversity_mode!r}")
            for round_index in range(cfg.n_rounds):
                name = f"exchange_r{round_index}"
                partitions[name] = generate_partition(
                    cfg, name, "train", cfg.exchange_examples_per_round, used
                )
                extra = f"equal_cost_r{round_index}"
                extra_size = max(1, math.ceil(
                    cfg.exchange_examples_per_round * cfg.equal_cost_extra_fraction
                ))
                partitions[extra] = generate_partition(
                    cfg, extra, "train", extra_size, used
                )
                for peer in range(cfg.n_peers):
                    private = f"private_peer_{peer}_r{round_index}"
                    weights = (
                        specialist_weights(cfg, peer)
                        if cfg.diversity_mode == "balanced_specialists"
                        else {skill: 1 / len(cfg.skills) for skill in cfg.skills}
                    )
                    partitions[private] = generate_weighted_skill_partition(
                        cfg,
                        private,
                        "train",
                        cfg.private_replay_examples_per_round,
                        weights,
                        used,
                    )
            for boundary in range(cfg.n_rounds + 1):
                election = f"election_r{boundary}"
                promotion = f"promotion_r{boundary}"
                partitions[election] = generate_partition(
                    cfg, election, "election", cfg.election_examples_per_round, used
                )
                partitions[promotion] = generate_partition(
                    cfg, promotion, "promotion", cfg.promotion_examples_per_round, used
                )
            partitions["final_audit"] = generate_partition(
                cfg, "final_audit", "audit", cfg.final_audit_examples, used
            )
            partitions["shift_audit"] = generate_partition(
                cfg, "shift_audit", "shift", cfg.shift_audit_examples, used
            )
            partitions["composition_audit"] = generate_composition_partition(
                cfg, "composition_audit", cfg.composition_audit_examples, used
            )

            manifest: dict[str, Any] = {
                "protocol_version": cfg.protocol_version,
                "skills": list(cfg.skills),
                "composition_skills": list(cfg.composition_skills),
                "skill_groups": [list(group) for group in cfg.skill_groups],
                "diversity_mode": cfg.diversity_mode,
                "specialist_home_fraction": cfg.specialist_home_fraction,
                "equal_cost_extra_fraction": cfg.equal_cost_extra_fraction,
                "partitions": {},
            }
            all_ids: set[str] = set()
            all_signatures: set[str] = set()
            for name, rows in partitions.items():
                path = data_dir / f"{name}.jsonl"
                write_jsonl(path, rows)
                ids = {row["artifact_id"] for row in rows}
                signatures = {row["signature"] for row in rows}
                if all_ids & ids or all_signatures & signatures:
                    raise AssertionError(f"Cross-partition collision in {name}")
                all_ids |= ids
                all_signatures |= signatures
                known_skills = cfg.skills + cfg.composition_skills
                per_skill = {skill: sum(r["skill"] == skill for r in rows) for skill in known_skills}
                manifest["partitions"][name] = {
                    "path": str(path),
                    "rows": len(rows),
                    "sha256": sha256_file(path),
                    "per_skill": per_skill,
                }
            manifest["manifest_sha256"] = sha256_json(manifest)
            atomic_json(data_dir / "manifest.json", manifest)
            return manifest
        """
    ),
    code(
        r"""
        def audit_manifest(cfg: ExperimentConfig, manifest: Mapping[str, Any]) -> dict[str, Any]:
            warmup_partitions = 1 if cfg.diversity_mode in {"exact_clone", "order_only"} else cfg.n_peers
            expected = (
                warmup_partitions
                + (2 + cfg.n_peers) * cfg.n_rounds
                + 2 * (cfg.n_rounds + 1)
                + 3
            )
            if len(manifest["partitions"]) != expected:
                raise AssertionError((len(manifest["partitions"]), expected))
            if manifest.get("diversity_mode") != cfg.diversity_mode:
                raise AssertionError("Manifest diversity mode does not match config")
            if not math.isclose(
                float(manifest.get("equal_cost_extra_fraction", -1)),
                cfg.equal_cost_extra_fraction,
                abs_tol=1e-12,
            ):
                raise AssertionError("Equal-cost fraction changed after manifest creation")
            seen_ids: set[str] = set()
            seen_signatures: set[str] = set()
            checked = 0
            for name, info in manifest["partitions"].items():
                path = Path(info["path"])
                if sha256_file(path) != info["sha256"]:
                    raise AssertionError(f"Hash mismatch: {name}")
                rows = read_jsonl(path)
                if len(rows) != info["rows"]:
                    raise AssertionError(f"Row mismatch: {name}")
                for row in rows:
                    if row["artifact_id"] in seen_ids or row["signature"] in seen_signatures:
                        raise AssertionError(f"Leakage/collision at {name}:{row['artifact_id']}")
                    seen_ids.add(row["artifact_id"])
                    seen_signatures.add(row["signature"])
                    if normalize_answer(row["target"]) != row["answer"]:
                        raise AssertionError(f"Verifier mismatch: {row['artifact_id']}")
                    checked += 1
            return {
                "partitions": len(manifest["partitions"]),
                "records": checked,
                "unique_ids": len(seen_ids),
                "unique_signatures": len(seen_signatures),
                "manifest_sha256": manifest["manifest_sha256"],
            }


        def smoke_config(cfg: ExperimentConfig) -> ExperimentConfig:
            return replace(
                cfg,
                output_dir=(cfg.output_dir / "dry_run"),
                warmup_examples_per_peer=32,
                exchange_examples_per_round=32,
                private_replay_examples_per_round=16,
                election_examples_per_round=32,
                promotion_examples_per_round=32,
                final_audit_examples=64,
                shift_audit_examples=64,
                composition_audit_examples=64,
            )
        """
    ),
    markdown(
        r"""
        ## Model and tokenizer preflight

        `OLMO_400M_MODEL` must point to the exact team checkpoint; the notebook never silently
        substitutes a public model. `OLMO_LARGER_TEACHER_MODEL` must independently point to a
        pinned local artifact. “Larger” is enforced by parameter ratio, then “stronger” is
        enforced behaviorally on two calibration shards before Stage 2.

        Token-level KL is permitted only when the two tokenizers have identical token-to-id
        maps, vocabulary sizes, special-token IDs, and probe encodings. A mere shared OLMo
        name is not enough. A mismatch automatically restricts the larger-teacher arms to
        verified sequence/rationale distillation through the 400M tokenizer. The peer arm is
        unaffected because all four students share one exact tokenizer.
        """
    ),
    code(
        r"""
        def package_versions() -> dict[str, str | None]:
            versions: dict[str, str | None] = {
                "python": os.sys.version,
                "numpy": np.__version__,
                "torch": getattr(torch, "__version__", None),
            }
            try:
                import transformers
                versions["transformers"] = transformers.__version__
            except ImportError:
                versions["transformers"] = None
            return versions


        def local_artifact_manifest(model_path: Path) -> dict[str, Any]:
            wanted = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
                "vocab.json",
                "merges.txt",
                "generation_config.json",
                "model.safetensors.index.json",
                "model.safetensors",
            ]
            files: dict[str, dict[str, Any]] = {}
            for name in wanted:
                path = model_path / name
                if path.exists() and path.is_file():
                    files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(model_path.glob("model-*.safetensors")):
                files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if "config.json" not in files:
                raise FileNotFoundError(f"Missing config.json in {model_path}")
            return {"path": str(model_path.resolve()), "files": files}


        def load_and_validate_model(cfg: ExperimentConfig, device: str = "cpu") -> tuple[Any, Any, dict[str, Any]]:
            require_training(cfg)
            model_path = Path(cfg.model_path).expanduser().resolve()
            artifact = local_artifact_manifest(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=cfg.local_files_only, use_fast=True
            )
            if cfg.peer_kd_scope == "answer_only" and not getattr(tokenizer, "is_fast", False):
                raise ValueError(
                    "answer_only peer KD requires the pinned fast tokenizer for exact offsets"
                )
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer needs a pad or EOS token.")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"  # batched decoder-only generation
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=cfg.local_files_only,
                torch_dtype=dtype,
            ).to(device)
            count = sum(p.numel() for p in model.parameters())
            model_type = str(getattr(model.config, "model_type", ""))
            if model_type not in cfg.allowed_olmo_model_types:
                raise ValueError(
                    f"Expected an OLMo-family 400M checkpoint; model_type={model_type!r}"
                )
            lower = cfg.expected_parameter_count * (1 - cfg.parameter_tolerance_fraction)
            upper = cfg.expected_parameter_count * (1 + cfg.parameter_tolerance_fraction)
            if not (lower <= count <= upper):
                raise ValueError(f"Expected ~400M parameters; loaded {count:,} outside [{lower:,.0f}, {upper:,.0f}].")
            if model.get_input_embeddings().num_embeddings != len(tokenizer):
                raise ValueError("Tokenizer/model vocabulary mismatch; ordinary tokenwise KL is unsafe.")
            manifest = {
                "artifact": artifact,
                "parameter_count": count,
                "model_type": model_type,
                "tokenizer_length": len(tokenizer),
                "packages": package_versions(),
            }
            manifest["preflight_sha256"] = sha256_json(manifest)
            return model, tokenizer, manifest


        def tokenizer_compatibility(
            student_tokenizer: Any,
            teacher_tokenizer: Any,
        ) -> dict[str, Any]:
            student_vocab = student_tokenizer.get_vocab()
            teacher_vocab = teacher_tokenizer.get_vocab()
            student_vocab_hash = sha256_json(sorted(student_vocab.items()))
            teacher_vocab_hash = sha256_json(sorted(teacher_vocab.items()))
            special_names = (
                "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
            )
            special_ids_equal = all(
                getattr(student_tokenizer, name, None)
                == getattr(teacher_tokenizer, name, None)
                for name in special_names
            )
            probes = (
                "Answer: -17/23",
                "A café costs $4.50.",
                "x_1 + x_2 = 11\nResponse:",
                " leading and  trailing ",
            )
            probe_encodings_equal = all(
                student_tokenizer(text, add_special_tokens=False)["input_ids"]
                == teacher_tokenizer(text, add_special_tokens=False)["input_ids"]
                for text in probes
            )
            exact_vocab_ids = bool(student_vocab == teacher_vocab)
            token_level_kd_allowed = bool(
                len(student_tokenizer) == len(teacher_tokenizer)
                and exact_vocab_ids
                and special_ids_equal
                and probe_encodings_equal
            )
            return {
                "student_vocab_size": len(student_tokenizer),
                "teacher_vocab_size": len(teacher_tokenizer),
                "student_vocab_sha256": student_vocab_hash,
                "teacher_vocab_sha256": teacher_vocab_hash,
                "exact_token_to_id_map": exact_vocab_ids,
                "special_token_ids_equal": special_ids_equal,
                "probe_encodings_equal": probe_encodings_equal,
                "token_level_kd_allowed": token_level_kd_allowed,
                "authorized_distillation": (
                    "token_or_sequence" if token_level_kd_allowed else "sequence_only"
                ),
            }


        def load_and_validate_larger_teacher(
            cfg: ExperimentConfig,
            student_tokenizer: Any,
            student_parameter_count: int,
            device: str = "cpu",
        ) -> tuple[Any, Any, dict[str, Any]]:
            require_training(cfg)
            if not cfg.larger_teacher_model_path:
                raise RuntimeError(
                    "Set OLMO_LARGER_TEACHER_MODEL to a pinned local checkpoint before Stage 2."
                )
            teacher_path = Path(cfg.larger_teacher_model_path).expanduser().resolve()
            artifact = local_artifact_manifest(teacher_path)
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                str(teacher_path), local_files_only=cfg.local_files_only, use_fast=True
            )
            if teacher_tokenizer.pad_token_id is None:
                if teacher_tokenizer.eos_token_id is None:
                    raise ValueError("Larger-teacher tokenizer needs a pad or EOS token")
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
            teacher_tokenizer.padding_side = "left"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            teacher_model = AutoModelForCausalLM.from_pretrained(
                str(teacher_path),
                local_files_only=cfg.local_files_only,
                torch_dtype=dtype,
            ).to(device)
            teacher_count = sum(parameter.numel() for parameter in teacher_model.parameters())
            teacher_model_type = str(getattr(teacher_model.config, "model_type", ""))
            if teacher_model_type not in cfg.allowed_olmo_model_types:
                raise ValueError(
                    "The larger teacher must be OLMo-family compatible; "
                    f"model_type={teacher_model_type!r}"
                )
            minimum = student_parameter_count * cfg.larger_teacher_min_parameter_ratio
            if teacher_count < minimum:
                raise ValueError(
                    f"The larger teacher has {teacher_count:,} parameters; Stage 2 requires "
                    f"at least {minimum:,.0f} ({cfg.larger_teacher_min_parameter_ratio:.2f}x)."
                )
            ratio = teacher_count / student_parameter_count
            if (
                ratio > cfg.larger_teacher_max_primary_parameter_ratio
                and not cfg.allow_extreme_teacher_stress_test
            ):
                raise ValueError(
                    f"Teacher/student ratio {ratio:.2f}x is above the frozen primary range. "
                    "Use an approximately 1B teacher for the 400M primary comparator. A 7B "
                    "teacher is a separately gated secondary stress test; enable it only with "
                    "OLMO_ALLOW_EXTREME_TEACHER=SECONDARY_ONLY."
                )
            if teacher_model.get_input_embeddings().num_embeddings != len(teacher_tokenizer):
                raise ValueError("Larger-teacher tokenizer/model vocabulary mismatch")
            compatibility = tokenizer_compatibility(
                student_tokenizer, teacher_tokenizer
            )
            stage_match = bool(
                cfg.larger_teacher_checkpoint_stage == "matched_to_student"
                or cfg.larger_teacher_checkpoint_stage == cfg.student_checkpoint_stage
            )
            compatibility["checkpoint_stage_match"] = stage_match
            compatibility["student_checkpoint_stage"] = cfg.student_checkpoint_stage
            compatibility["teacher_checkpoint_stage"] = (
                cfg.larger_teacher_checkpoint_stage
            )
            compatibility["token_level_kd_allowed"] = bool(
                compatibility["token_level_kd_allowed"] and stage_match
            )
            compatibility["authorized_distillation"] = (
                "token_or_sequence"
                if compatibility["token_level_kd_allowed"] else "sequence_only"
            )
            manifest = {
                "artifact": artifact,
                "parameter_count": teacher_count,
                "model_type": teacher_model_type,
                "student_parameter_count": int(student_parameter_count),
                "parameter_ratio": teacher_count / student_parameter_count,
                "checkpoint_stage": cfg.larger_teacher_checkpoint_stage,
                "teacher_role": (
                    "secondary_extreme_capacity_stress_test"
                    if ratio > cfg.larger_teacher_max_primary_parameter_ratio
                    else "primary_capacity_matched_larger_teacher"
                ),
                "tokenizer_compatibility": compatibility,
                "packages": package_versions(),
            }
            manifest["preflight_sha256"] = sha256_json(manifest)
            return teacher_model, teacher_tokenizer, manifest
        """
    ),
    code(
        r"""
        def prompt_text(record: Mapping[str, Any]) -> str:
            return f"Question: {record['prompt']}\nResponse:"


        def response_ids_and_kd_labels(
            tokenizer: Any,
            response: str,
            cfg: ExperimentConfig,
        ) -> tuple[list[int], list[int]]:
            if cfg.peer_kd_scope != "answer_only":
                raise RuntimeError(
                    "Full-rationale peer KD is blocked until teacher continuations from "
                    "student prefixes are separately generated and verified."
                )
            marker = "Answer:"
            marker_start = response.rfind(marker)
            if marker_start < 0:
                raise ValueError("Peer-KD response has no final Answer: marker")
            marker_end = marker_start + len(marker)
            answer_start = marker_end
            while answer_start < len(response) and response[answer_start].isspace():
                answer_start += 1
            if answer_start >= len(response):
                raise ValueError("Peer-KD response has an empty final answer span")
            try:
                encoded = tokenizer(
                    response,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
            except (TypeError, NotImplementedError) as exc:
                raise RuntimeError(
                    "answer_only peer KD requires a fast tokenizer with offset mappings"
                ) from exc
            response_ids = list(encoded["input_ids"])
            offsets = encoded.get("offset_mapping")
            if offsets is None or len(offsets) != len(response_ids):
                raise RuntimeError("Tokenizer did not return one offset per response token")
            kd_labels = []
            for token_id, offset in zip(response_ids, offsets):
                start, end = map(int, offset)
                # Include answer content (and any following EOS text), but never the marker.
                in_answer = start >= marker_end and end > answer_start
                kd_labels.append(int(token_id) if in_answer else -100)
            if not any(label != -100 for label in kd_labels):
                raise ValueError("Tokenizer produced no answer-only KD tokens")
            return response_ids, kd_labels


        def tokenize_record(tokenizer: Any, record: Mapping[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
            prompt_ids = tokenizer(prompt_text(record), add_special_tokens=False)["input_ids"]
            response = " " + record["target"]
            if tokenizer.eos_token:
                response += tokenizer.eos_token
            response_ids, response_kd_labels = response_ids_and_kd_labels(
                tokenizer, response, cfg
            )
            bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            input_ids = bos + prompt_ids + response_ids
            labels = [-100] * (len(bos) + len(prompt_ids)) + response_ids
            kd_labels = [-100] * (len(bos) + len(prompt_ids)) + response_kd_labels
            if len(input_ids) > cfg.max_length:
                raise ValueError(f"Record {record['artifact_id']} has {len(input_ids)} tokens > {cfg.max_length}; never truncate.")
            return {
                "artifact_id": record["artifact_id"],
                "input_ids": input_ids,
                "labels": labels,
                "kd_labels": kd_labels,
                "answer": record["answer"],
                "skill": record["skill"],
            }


        class TokenizedRecords:
            def __init__(self, tokenizer: Any, records: Sequence[Mapping[str, Any]], cfg: ExperimentConfig):
                self.rows = [tokenize_record(tokenizer, row, cfg) for row in records]

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, Any]:
                return self.rows[index]


        def collate_tokenized(rows: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
            if torch is None:
                raise RuntimeError("PyTorch is required for model operations.")
            width = max(len(row["input_ids"]) for row in rows)
            ids, labels, kd_labels, attention = [], [], [], []
            for row in rows:
                pad = width - len(row["input_ids"])
                ids.append(row["input_ids"] + [pad_token_id] * pad)
                labels.append(row["labels"] + [-100] * pad)
                kd_labels.append(row["kd_labels"] + [-100] * pad)
                attention.append([1] * len(row["input_ids"]) + [0] * pad)
            return {
                "artifact_ids": [row["artifact_id"] for row in rows],
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "kd_labels": torch.tensor(kd_labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "answers": [row["answer"] for row in rows],
                "skills": [row["skill"] for row in rows],
            }
        """
    ),
    markdown(
        r"""
        ## Greedy evaluation and verifier

        “Evaluate on one thing” means a capability suite containing many independent items,
        not a single question. Every peer sees the same items. Generation is greedy, the
        canonical answer is parsed deterministically, and per-skill as well as macro scores
        are retained. Election and promotion content is never added to training.
        """
    ),
    code(
        r"""
        def evaluate_records(
            model: Any,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            batch_size: int | None = None,
        ) -> dict[str, Any]:
            if torch is None:
                raise RuntimeError("PyTorch is required for evaluation.")
            batch_size = batch_size or cfg.micro_batch_size
            model.eval()
            item_rows: list[dict[str, Any]] = []
            start = time.perf_counter()
            for start_index in range(0, len(records), batch_size):
                batch_records = records[start_index:start_index + batch_size]
                encoded = tokenizer(
                    [prompt_text(row) for row in batch_records],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=cfg.generation_max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_width = encoded["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
                for record, output in zip(batch_records, decoded):
                    prediction = normalize_answer(output)
                    item_rows.append({
                        "artifact_id": record["artifact_id"],
                        "skill": record["skill"],
                        "gold": record["answer"],
                        "prediction": prediction,
                        "correct": int(prediction == record["answer"]),
                        "output": output,
                        "decoded_tokens": len(tokenizer(
                            output, add_special_tokens=False
                        )["input_ids"]),
                    })
            elapsed = time.perf_counter() - start
            per_skill = {}
            observed_skills = sorted({row["skill"] for row in item_rows})
            for skill in observed_skills:
                values = [row["correct"] for row in item_rows if row["skill"] == skill]
                per_skill[skill] = float(np.mean(values)) if values else float("nan")
            return {
                "accuracy": float(np.mean([row["correct"] for row in item_rows])),
                "macro_skill_accuracy": float(np.mean(list(per_skill.values()))),
                "worst_skill_accuracy": float(np.min(list(per_skill.values()))),
                "per_skill": per_skill,
                "items": item_rows,
                "elapsed_seconds": elapsed,
            }


        def item_vector(evaluation: Mapping[str, Any]) -> np.ndarray:
            return np.asarray([row["correct"] for row in evaluation["items"]], dtype=np.float64)


        def population_complementarity(
            evaluations: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
        ) -> dict[str, Any]:
            if len(evaluations) != cfg.n_peers:
                raise ValueError("Expected one evaluation per peer")
            ids = [[row["artifact_id"] for row in result["items"]] for result in evaluations]
            if any(row_ids != ids[0] for row_ids in ids[1:]):
                raise ValueError("Complementarity requires aligned item order")
            skills = [row["skill"] for row in evaluations[0]["items"]]
            correct = np.stack([item_vector(result) for result in evaluations])
            peer_accuracy = correct.mean(axis=1)
            oracle = correct.max(axis=0)
            unique = []
            directed_rescue = np.zeros((cfg.n_peers, cfg.n_peers), dtype=np.float64)
            for peer in range(cfg.n_peers):
                others = np.delete(correct, peer, axis=0)
                unique.append(float(np.mean((correct[peer] == 1) & (others.max(axis=0) == 0))))
                for student in range(cfg.n_peers):
                    directed_rescue[peer, student] = float(np.mean(
                        (correct[peer] == 1) & (correct[student] == 0)
                    ))
            pairwise_disagreement = []
            for a in range(cfg.n_peers):
                for b in range(a + 1, cfg.n_peers):
                    pairwise_disagreement.append(float(np.mean(correct[a] != correct[b])))

            per_skill = {}
            for skill in cfg.skills:
                mask = np.asarray([value == skill for value in skills])
                skill_correct = correct[:, mask]
                skill_peer = skill_correct.mean(axis=1)
                skill_oracle = float(skill_correct.max(axis=0).mean())
                per_skill[skill] = {
                    "peer_accuracy": skill_peer.tolist(),
                    "oracle_accuracy": skill_oracle,
                    "oracle_headroom": skill_oracle - float(skill_peer.max()),
                    "all_wrong_fraction": 1 - skill_oracle,
                }

            best = float(peer_accuracy.max())
            oracle_accuracy = float(oracle.mean())
            headroom = oracle_accuracy - best
            failures = []
            if best < cfg.capability_floor:
                failures.append(f"best peer {best:.4f} is below floor {cfg.capability_floor:.4f}")
            if best > cfg.capability_ceiling:
                failures.append(f"best peer {best:.4f} is above ceiling {cfg.capability_ceiling:.4f}")
            if headroom < cfg.complementarity_headroom_min:
                failures.append(
                    f"oracle headroom {headroom:.4f} < {cfg.complementarity_headroom_min:.4f}"
                )
            if min(unique) < cfg.unique_correct_fraction_min:
                failures.append(
                    f"minimum unique-correct fraction {min(unique):.4f} < "
                    f"{cfg.unique_correct_fraction_min:.4f}"
                )
            return {
                "peer_accuracy": peer_accuracy.tolist(),
                "best_peer_accuracy": best,
                "oracle_accuracy": oracle_accuracy,
                "oracle_headroom": headroom,
                "all_wrong_fraction": float(np.mean(oracle == 0)),
                "unique_correct_fraction": unique,
                "pairwise_disagreement": pairwise_disagreement,
                "directed_rescue": directed_rescue.tolist(),
                "per_skill": per_skill,
                "gate_passed": not failures,
                "gate_failures": failures,
            }


        def require_population_gate(report: Mapping[str, Any]) -> None:
            if not report["gate_passed"]:
                raise RuntimeError(
                    "Peer population failed the preregistered complementarity gate: "
                    + "; ".join(report["gate_failures"])
                )


        def generate_rollouts_for_model(
            model: Any,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            seed: int,
        ) -> dict[str, dict[str, Any]]:
            model.eval()
            rows: dict[str, dict[str, Any]] = {}
            for batch_start in range(0, len(records), cfg.micro_batch_size):
                batch = records[batch_start:batch_start + cfg.micro_batch_size]
                encoded = tokenizer(
                    [prompt_text(row) for row in batch],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(device)
                torch.manual_seed(seed + batch_start)
                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=cfg.peer_draft_temperature,
                        top_p=cfg.peer_draft_top_p,
                        max_new_tokens=cfg.generation_max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_width = encoded["input_ids"].shape[1]
                decoded = tokenizer.batch_decode(
                    generated[:, prompt_width:], skip_special_tokens=True
                )
                for record, output in zip(batch, decoded):
                    prediction = normalize_answer(output)
                    rows[record["artifact_id"]] = {
                        "artifact_id": record["artifact_id"],
                        "skill": record["skill"],
                        "output": output,
                        "prediction": prediction,
                        "correct": bool(prediction == record["answer"]),
                        "output_sha256": sha256_bytes(output.encode("utf-8")),
                        "decoded_tokens": len(tokenizer(
                            output, add_special_tokens=False
                        )["input_ids"]),
                    }
            return rows


        def generate_population_rollouts(
            models: Sequence[Any],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            seed: int,
        ) -> list[dict[str, dict[str, Any]]]:
            if len(models) != cfg.n_peers:
                raise ValueError("Need one frozen model per peer")
            return [
                generate_rollouts_for_model(
                    model, tokenizer, records, cfg, device,
                    seed=seed * 10_000 + peer * 1_000_003,
                )
                for peer, model in enumerate(models)
            ]


        def route_rescue_edges(
            records: Sequence[Mapping[str, Any]],
            rollouts: Sequence[Mapping[str, Mapping[str, Any]]],
            confirmation_rollouts: Sequence[Mapping[str, Mapping[str, Any]]] | None,
            routing_skill_scores: Sequence[Mapping[str, float]],
            cfg: ExperimentConfig,
            seed: int,
        ) -> tuple[list[dict[str, int]], dict[str, Any]]:
            # Route one independently correct peer only where the target student failed.
            # With the default strict gate, both facts must reproduce on a second frozen
            # rollout so a lucky sample cannot nominate an unreliable tutor.
            if cfg.require_itemwise_teacher_correctness and confirmation_rollouts is None:
                raise ValueError("Strict teacher correctness requires confirmation rollouts")
            edges: list[dict[str, int]] = [dict() for _ in range(cfg.n_peers)]
            edge_counts = np.zeros((cfg.n_peers, cfg.n_peers), dtype=np.int64)
            rejected_unconfirmed_teachers = 0
            rejected_unconfirmed_targets = 0
            for record in records:
                artifact_id = record["artifact_id"]
                skill = record["skill"]
                for student in range(cfg.n_peers):
                    if rollouts[student][artifact_id]["correct"]:
                        continue
                    if (
                        confirmation_rollouts is not None
                        and confirmation_rollouts[student][artifact_id]["correct"]
                    ):
                        rejected_unconfirmed_targets += 1
                        continue
                    candidates = [
                        teacher for teacher in range(cfg.n_peers)
                        if teacher != student and rollouts[teacher][artifact_id]["correct"]
                        and (
                            confirmation_rollouts is None
                            or confirmation_rollouts[teacher][artifact_id]["correct"]
                        )
                    ]
                    if confirmation_rollouts is not None:
                        rejected_unconfirmed_teachers += sum(
                            1 for teacher in range(cfg.n_peers)
                            if teacher != student
                            and rollouts[teacher][artifact_id]["correct"]
                            and not confirmation_rollouts[teacher][artifact_id]["correct"]
                        )
                    if not candidates:
                        continue
                    # Competence is estimated only on the prior one-use routing/election shard.
                    # A seeded hash provides a stable least-used-style tiebreak without looking
                    # at any final result.
                    teacher = max(
                        candidates,
                        key=lambda value: (
                            float(routing_skill_scores[value][skill]),
                            -int(sha256_json([seed, artifact_id, student, value])[:12], 16),
                        ),
                    )
                    edges[student][artifact_id] = teacher
                    edge_counts[teacher, student] += 1
            eligible = [len(row) for row in edges]
            return edges, {
                "eligible_by_student": eligible,
                "eligible_fraction_by_student": [value / len(records) for value in eligible],
                "teacher_to_student_counts": edge_counts.tolist(),
                "total_edges": int(edge_counts.sum()),
                "rejected_unconfirmed_teacher_candidates": rejected_unconfirmed_teachers,
                "rejected_unconfirmed_targets": rejected_unconfirmed_targets,
            }
        """
    ),
    markdown(
        r"""
        ## Go/no-go gate before peer teaching

        Different seeds are not evidence of useful peer knowledge. Before any KD arm runs,
        the warmed-up quartet is evaluated once on the calibration/election population. The
        notebook records best-peer accuracy, oracle best-of-four accuracy, all-wrong mass,
        each peer's uniquely correct mass, directed rescue, pairwise disagreement, and
        per-skill headroom. The default project gates require the task to sit between 20% and
        85% best-peer accuracy, at least two points of oracle headroom, and at least one
        percent uniquely correct mass from every peer. These are preregistered project choices,
        not literature constants. Failing the gate stops the expensive peer-teaching stage.
        """
    ),
    markdown(
        r"""
        ## Detached teacher cache

        A round teacher is immutable. Its logits are computed under teacher forcing on the
        verified canonical response and cached once for reuse by all three challengers. Peer
        KL is masked to tokens after the final `Answer:` marker; verified CE still covers the
        complete canonical rationale and answer.
        Unless disabled for an ablation, a record receives KD only when the teacher also
        solves that item greedily. The cache stores the teacher's top-k token probabilities
        plus one aggregate “other” bucket; hard-target CE remains available on every record.
        Validate the approximation against full-vocabulary KL on a stratified sample before a
        confirmatory run.
        """
    ),
    code(
        r"""
        def build_teacher_cache(
            model: Any,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            cache_path: Path,
            eligibility: Mapping[str, bool] | None = None,
        ) -> dict[str, Any]:
            if torch is None:
                raise RuntimeError("PyTorch is required for teacher scoring.")
            dataset = TokenizedRecords(tokenizer, records, cfg)
            loader = DataLoader(
                dataset,
                batch_size=cfg.micro_batch_size,
                shuffle=False,
                collate_fn=lambda rows: collate_tokenized(rows, tokenizer.pad_token_id),
            )
            cache: dict[str, Any] = {}
            model.eval()
            start = time.perf_counter()
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                kd_labels = batch["kd_labels"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1]
                    log_probs = F.log_softmax(logits.float() / cfg.kd_temperature, dim=-1)
                shifted_kd_labels = kd_labels[:, 1:]
                for row_index, artifact_id in enumerate(batch["artifact_ids"]):
                    if eligibility is not None and not eligibility.get(artifact_id, False):
                        continue
                    mask = shifted_kd_labels[row_index] != -100
                    token_log_probs = log_probs[row_index][mask]
                    values, indices = torch.topk(token_log_probs, k=cfg.teacher_topk, dim=-1)
                    mass = values.exp().sum(dim=-1).clamp(max=1 - 1e-7)
                    cache[artifact_id] = {
                        "indices": indices.to(torch.int32).cpu(),
                        "log_probs": values.to(torch.float16).cpu(),
                        "log_other": torch.log1p(-mass).to(torch.float32).cpu(),
                        "response_tokens": int(mask.sum().item()),
                    }
            payload = {
                "protocol_version": cfg.protocol_version,
                "temperature": cfg.kd_temperature,
                "topk": cfg.teacher_topk,
                "kd_scope": cfg.peer_kd_scope,
                "records": cache,
                "eligible_records": len(cache),
                "total_records": len(records),
                "elapsed_seconds": time.perf_counter() - start,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
            return payload


        def build_leave_one_out_teacher_caches(
            models: Sequence[Any],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            cache_paths: Sequence[Path],
            eligibility_by_teacher: Sequence[Mapping[str, bool]] | None = None,
        ) -> list[dict[str, Any]]:
            # Build one verifier-gated peer-mixture cache for each prospective student.
            # Every frozen teacher is forwarded once per batch. For student i and item x, the
            # target is the arithmetic mean of temperature-scaled distributions from peers
            # j!=i that independently solved x. If no peer qualifies, x is absent from i's
            # cache and its update falls back to hard-target CE.
            if torch is None:
                raise RuntimeError("PyTorch is required for teacher scoring.")
            if len(models) != cfg.n_peers or len(cache_paths) != cfg.n_peers:
                raise ValueError("Need one frozen model and cache path per peer")
            dataset = TokenizedRecords(tokenizer, records, cfg)
            loader = DataLoader(
                dataset,
                batch_size=cfg.micro_batch_size,
                shuffle=False,
                collate_fn=lambda rows: collate_tokenized(rows, tokenizer.pad_token_id),
            )
            for model in models:
                model.eval()
            caches: list[dict[str, Any]] = [dict() for _ in range(cfg.n_peers)]
            started = time.perf_counter()
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                kd_labels = batch["kd_labels"].to(device)
                teacher_probs = []
                with torch.no_grad():
                    for model in models:
                        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1]
                        teacher_probs.append(F.softmax(
                            logits.float() / cfg.kd_temperature, dim=-1
                        ))
                shifted_kd_labels = kd_labels[:, 1:]
                for row_index, artifact_id in enumerate(batch["artifact_ids"]):
                    mask = shifted_kd_labels[row_index] != -100
                    for student in range(cfg.n_peers):
                        eligible = [
                            teacher
                            for teacher in range(cfg.n_peers)
                            if teacher != student
                            and (
                                eligibility_by_teacher is None
                                or eligibility_by_teacher[teacher].get(artifact_id, False)
                            )
                        ]
                        if not eligible:
                            continue
                        mixture = torch.stack([
                            teacher_probs[teacher][row_index][mask] for teacher in eligible
                        ]).mean(dim=0)
                        mixture_logp = mixture.clamp_min(1e-30).log()
                        k = min(cfg.teacher_topk, mixture_logp.shape[-1] - 1)
                        values, indices = torch.topk(mixture_logp, k=k, dim=-1)
                        mass = values.exp().sum(dim=-1).clamp(max=1 - 1e-7)
                        caches[student][artifact_id] = {
                            "indices": indices.to(torch.int32).cpu(),
                            "log_probs": values.to(torch.float16).cpu(),
                            "log_other": torch.log1p(-mass).to(torch.float32).cpu(),
                            "response_tokens": int(mask.sum().item()),
                            "eligible_teachers": tuple(eligible),
                        }
                del teacher_probs
            elapsed = time.perf_counter() - started
            payloads = []
            for student, cache_path in enumerate(cache_paths):
                payload = {
                    "protocol_version": cfg.protocol_version,
                    "policy": "verified_leave_one_out_probability_mean",
                    "student": student,
                    "temperature": cfg.kd_temperature,
                    "topk": cfg.teacher_topk,
                    "kd_scope": cfg.peer_kd_scope,
                    "records": caches[student],
                    "eligible_records": len(caches[student]),
                    "total_records": len(records),
                    "elapsed_seconds_shared": elapsed,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                payloads.append(payload)
            return payloads


        def topk_entry_from_log_probs(
            token_log_probs: Any,
            cfg: ExperimentConfig,
            **metadata: Any,
        ) -> dict[str, Any]:
            k = min(cfg.teacher_topk, token_log_probs.shape[-1] - 1)
            values, indices = torch.topk(token_log_probs, k=k, dim=-1)
            mass = values.exp().sum(dim=-1).clamp(max=1 - 1e-7)
            return {
                "indices": indices.to(torch.int32).cpu(),
                "log_probs": values.to(torch.float16).cpu(),
                "log_other": torch.log1p(-mass).to(torch.float32).cpu(),
                "response_tokens": int(token_log_probs.shape[0]),
                **metadata,
            }


        def build_routed_goldprefix_caches(
            models: Sequence[Any],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            edges: Sequence[Mapping[str, int]],
            cfg: ExperimentConfig,
            device: str,
            cache_paths: Sequence[Path],
        ) -> list[dict[str, Any]]:
            # Cache one selected peer distribution on canonical response prefixes. Store
            # the canonical sequence too so GP uses the same auxiliary student-forward
            # schedule as OP rather than receiving a hidden compute advantage.
            dataset = TokenizedRecords(tokenizer, records, cfg)
            tokenized_by_id = {row["artifact_id"]: row for row in dataset.rows}
            loader = DataLoader(
                dataset,
                batch_size=cfg.micro_batch_size,
                shuffle=False,
                collate_fn=lambda rows: collate_tokenized(rows, tokenizer.pad_token_id),
            )
            for model in models:
                model.eval()
            caches: list[dict[str, Any]] = [dict() for _ in range(cfg.n_peers)]
            started = time.perf_counter()
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                kd_labels = batch["kd_labels"].to(device)
                needed = {
                    edges[student].get(artifact_id)
                    for student in range(cfg.n_peers)
                    for artifact_id in batch["artifact_ids"]
                    if artifact_id in edges[student]
                }
                teacher_logp = {}
                with torch.no_grad():
                    for teacher in sorted(needed):
                        logits = models[teacher](
                            input_ids=input_ids, attention_mask=attention
                        ).logits[:, :-1]
                        teacher_logp[teacher] = F.log_softmax(
                            logits.float() / cfg.kd_temperature, dim=-1
                        )
                shifted_kd_labels = kd_labels[:, 1:]
                for row_index, artifact_id in enumerate(batch["artifact_ids"]):
                    mask = shifted_kd_labels[row_index] != -100
                    for student in range(cfg.n_peers):
                        teacher = edges[student].get(artifact_id)
                        if teacher is None:
                            continue
                        canonical = tokenized_by_id[artifact_id]
                        caches[student][artifact_id] = topk_entry_from_log_probs(
                            teacher_logp[teacher][row_index][mask],
                            cfg,
                            selected_teacher=int(teacher),
                            prefix_policy="gold",
                            input_ids=torch.tensor(
                                canonical["input_ids"], dtype=torch.int32
                            ),
                            kd_labels=torch.tensor(
                                canonical["kd_labels"], dtype=torch.int32
                            ),
                        )
                del teacher_logp
            elapsed = time.perf_counter() - started
            payloads = []
            for student, cache_path in enumerate(cache_paths):
                payload = {
                    "protocol_version": cfg.protocol_version,
                    "policy": "frozen_routed_rescue_goldprefix",
                    "student": student,
                    "temperature": cfg.kd_temperature,
                    "topk": cfg.teacher_topk,
                    "kd_scope": cfg.peer_kd_scope,
                    "records": caches[student],
                    "eligible_records": len(caches[student]),
                    "total_records": len(records),
                    "elapsed_seconds_shared": elapsed,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                payloads.append(payload)
            return payloads


        def tokenize_free_response(
            tokenizer: Any,
            record: Mapping[str, Any],
            response_text: str,
            cfg: ExperimentConfig,
        ) -> dict[str, Any] | None:
            prompt_ids = tokenizer(prompt_text(record), add_special_tokens=False)["input_ids"]
            response = " " + response_text
            marker = "Answer:"
            marker_start = response.rfind(marker)
            if marker_start < 0 or not response[marker_start + len(marker):].strip():
                return None
            if tokenizer.eos_token:
                response += tokenizer.eos_token
            try:
                response_ids, response_kd_labels = response_ids_and_kd_labels(
                    tokenizer, response, cfg
                )
            except ValueError:
                return None
            bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            input_ids = bos + prompt_ids + response_ids
            if len(input_ids) > cfg.max_length or not response_ids:
                return None
            return {
                "artifact_id": record["artifact_id"],
                "input_ids": input_ids,
                "labels": [-100] * (len(bos) + len(prompt_ids)) + response_ids,
                "kd_labels": [-100] * (len(bos) + len(prompt_ids)) + response_kd_labels,
                "answer": record["answer"],
                "skill": record["skill"],
            }


        def build_routed_onpolicy_caches(
            models: Sequence[Any],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            rollouts: Sequence[Mapping[str, Mapping[str, Any]]],
            edges: Sequence[Mapping[str, int]],
            cfg: ExperimentConfig,
            device: str,
            cache_paths: Sequence[Path],
            self_teacher: bool = False,
        ) -> list[dict[str, Any]]:
            # Score each student's sampled prefixes with its routed frozen peer (or self).
            record_by_id = {row["artifact_id"]: row for row in records}
            payloads = []
            global_started = time.perf_counter()
            for student in range(cfg.n_peers):
                grouped: dict[int, list[dict[str, Any]]] = {}
                rejected_answer_spans = 0
                for artifact_id, routed_teacher in edges[student].items():
                    teacher = student if self_teacher else routed_teacher
                    tokenized = tokenize_free_response(
                        tokenizer,
                        record_by_id[artifact_id],
                        str(rollouts[student][artifact_id]["output"]),
                        cfg,
                    )
                    if tokenized is None:
                        rejected_answer_spans += 1
                        continue
                    tokenized["selected_teacher"] = int(teacher)
                    grouped.setdefault(int(teacher), []).append(tokenized)

                cache: dict[str, Any] = {}
                for teacher, rows in grouped.items():
                    models[teacher].eval()
                    for start in range(0, len(rows), cfg.micro_batch_size):
                        batch_rows = rows[start:start + cfg.micro_batch_size]
                        batch = collate_tokenized(batch_rows, tokenizer.pad_token_id)
                        input_ids = batch["input_ids"].to(device)
                        attention = batch["attention_mask"].to(device)
                        kd_labels = batch["kd_labels"].to(device)
                        with torch.no_grad():
                            logits = models[teacher](
                                input_ids=input_ids, attention_mask=attention
                            ).logits[:, :-1]
                            logp = F.log_softmax(
                                logits.float() / cfg.kd_temperature, dim=-1
                            )
                        shifted_kd_labels = kd_labels[:, 1:]
                        for row_index, row in enumerate(batch_rows):
                            mask = shifted_kd_labels[row_index] != -100
                            entry = topk_entry_from_log_probs(
                                logp[row_index][mask],
                                cfg,
                                selected_teacher=int(teacher),
                                prefix_policy="student_onpolicy",
                                input_ids=torch.tensor(row["input_ids"], dtype=torch.int32),
                                labels=torch.tensor(row["labels"], dtype=torch.int32),
                                kd_labels=torch.tensor(row["kd_labels"], dtype=torch.int32),
                            )
                            cache[row["artifact_id"]] = entry
                payload = {
                    "protocol_version": cfg.protocol_version,
                    "policy": "self_snapshot_onpolicy" if self_teacher else "frozen_routed_rescue_onpolicy",
                    "student": student,
                    "temperature": cfg.kd_temperature,
                    "topk": cfg.teacher_topk,
                    "kd_scope": cfg.peer_kd_scope,
                    "records": cache,
                    "eligible_records": len(cache),
                    "routed_records": len(edges[student]),
                    "rejected_missing_or_empty_answer_span": rejected_answer_spans,
                    "total_records": len(records),
                    "elapsed_seconds_shared": time.perf_counter() - global_started,
                }
                cache_paths[student].parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_paths[student])
                payloads.append(payload)
            return payloads


        def approximate_forward_kl(
            student_log_probs: Any,
            teacher_entry: Mapping[str, Any],
        ) -> Any:
            # student_log_probs has shape [response_tokens, vocab].
            indices = teacher_entry["indices"].to(student_log_probs.device, dtype=torch.long)
            teacher_logp = teacher_entry["log_probs"].to(student_log_probs.device, dtype=torch.float32)
            teacher_log_other = teacher_entry["log_other"].to(student_log_probs.device, dtype=torch.float32)
            student_selected = student_log_probs.gather(-1, indices)
            student_mass = student_selected.exp().sum(dim=-1).clamp(max=1 - 1e-7)
            student_log_other = torch.log1p(-student_mass)
            teacher_p = teacher_logp.exp()
            selected_kl = (teacher_p * (teacher_logp - student_selected)).sum(dim=-1)
            other_p = teacher_log_other.exp()
            other_kl = other_p * (teacher_log_other - student_log_other)
            return (selected_kl + other_kl).mean()


        def validate_topk_cache(
            teacher_model: Any,
            student_model: Any,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            sample_positions: int = 10_000,
        ) -> dict[str, Any]:
            if torch is None:
                raise RuntimeError("PyTorch is required for KL calibration.")
            dataset = TokenizedRecords(tokenizer, records, cfg)
            loader = DataLoader(
                dataset,
                batch_size=cfg.micro_batch_size,
                shuffle=False,
                collate_fn=lambda rows: collate_tokenized(rows, tokenizer.pad_token_id),
            )
            teacher_model.eval()
            student_model.eval()
            full_values: list[float] = []
            coarse_values: list[float] = []
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                kd_labels = batch["kd_labels"].to(device)
                with torch.no_grad():
                    teacher_logits = teacher_model(
                        input_ids=input_ids, attention_mask=attention
                    ).logits[:, :-1].float() / cfg.kd_temperature
                    student_logits = student_model(
                        input_ids=input_ids, attention_mask=attention
                    ).logits[:, :-1].float() / cfg.kd_temperature
                    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
                    student_logp = F.log_softmax(student_logits, dim=-1)
                mask = kd_labels[:, 1:] != -100
                for row_index in range(mask.shape[0]):
                    t_logp = teacher_logp[row_index][mask[row_index]]
                    s_logp = student_logp[row_index][mask[row_index]]
                    full = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
                    k = min(cfg.teacher_topk, t_logp.shape[-1] - 1)
                    top_logp, top_idx = torch.topk(t_logp, k=k, dim=-1)
                    teacher_mass = top_logp.exp().sum(dim=-1).clamp(max=1 - 1e-7)
                    student_selected = s_logp.gather(-1, top_idx)
                    student_mass = student_selected.exp().sum(dim=-1).clamp(max=1 - 1e-7)
                    teacher_other = torch.log1p(-teacher_mass)
                    student_other = torch.log1p(-student_mass)
                    coarse = (
                        (top_logp.exp() * (top_logp - student_selected)).sum(dim=-1)
                        + teacher_other.exp() * (teacher_other - student_other)
                    )
                    full_values.extend(full.detach().cpu().tolist())
                    coarse_values.extend(coarse.detach().cpu().tolist())
                if len(full_values) >= sample_positions:
                    break
            if not full_values:
                raise RuntimeError("KL calibration sampled no response positions")
            full_array = np.asarray(full_values[:sample_positions], dtype=np.float64)
            coarse_array = np.asarray(coarse_values[:sample_positions], dtype=np.float64)
            absolute = np.abs(full_array - coarse_array)
            report = {
                "positions": int(len(absolute)),
                "topk": int(cfg.teacher_topk),
                "temperature": float(cfg.kd_temperature),
                "mean_full_kl": float(full_array.mean()),
                "mean_coarse_kl": float(coarse_array.mean()),
                "mean_abs_error": float(absolute.mean()),
                "p99_abs_error": float(np.quantile(absolute, 0.99)),
                "max_abs_error": float(absolute.max()),
            }
            report["accepted"] = bool(
                report["mean_abs_error"] <= cfg.topk_mae_limit
                and report["p99_abs_error"] <= cfg.topk_p99_limit
            )
            return report
        """
    ),
    code(
        r"""
        class CachedOnPolicyRecords:
            def __init__(self, cache_records: Mapping[str, Mapping[str, Any]]):
                self.rows = []
                for artifact_id, entry in cache_records.items():
                    self.rows.append({
                        "artifact_id": artifact_id,
                        "input_ids": entry["input_ids"].to(torch.long).tolist(),
                        "kd_labels": entry["kd_labels"].to(torch.long).tolist(),
                        "teacher_entry": entry,
                    })

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, index: int) -> dict[str, Any]:
                return self.rows[index]


        def collate_onpolicy(rows: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, Any]:
            width = max(len(row["input_ids"]) for row in rows)
            ids, kd_labels, attention = [], [], []
            for row in rows:
                pad = width - len(row["input_ids"])
                ids.append(row["input_ids"] + [pad_token_id] * pad)
                kd_labels.append(row["kd_labels"] + [-100] * pad)
                attention.append([1] * len(row["input_ids"]) + [0] * pad)
            return {
                "artifact_ids": [row["artifact_id"] for row in rows],
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "kd_labels": torch.tensor(kd_labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "teacher_entries": [row["teacher_entry"] for row in rows],
            }


        def cosine_lr(step: int, total_steps: int, cfg: ExperimentConfig) -> float:
            warmup = max(1, round(total_steps * cfg.warmup_fraction))
            if step < warmup:
                return cfg.learning_rate * (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            return cfg.min_learning_rate + (cfg.learning_rate - cfg.min_learning_rate) * cosine


        def train_checkpoint(
            source_path: Path,
            destination_path: Path,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            seed: int,
            peer: int,
            arm: str,
            device: str,
            ledger: CostLedger,
            teacher_cache_path: Path | None = None,
            onpolicy_cache_path: Path | None = None,
            sequence_cache_path: Path | None = None,
            optimizer_state_path: Path | None = None,
        ) -> dict[str, Any]:
            require_training(cfg)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(source_path), local_files_only=True, torch_dtype=dtype
            ).to(device)
            model.train()
            dataset = TokenizedRecords(tokenizer, records, cfg)
            order_generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                dataset,
                batch_size=cfg.micro_batch_size,
                shuffle=True,
                generator=order_generator,
                collate_fn=lambda rows: collate_tokenized(rows, tokenizer.pad_token_id),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.learning_rate,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.weight_decay,
            )
            if optimizer_state_path and optimizer_state_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
            teacher_cache = None
            if teacher_cache_path is not None:
                teacher_cache = torch.load(teacher_cache_path, map_location="cpu")["records"]
            if sum(path is not None for path in (
                teacher_cache_path, onpolicy_cache_path, sequence_cache_path
            )) > 1:
                raise ValueError(
                    "Use exactly one auxiliary target mode: gold-prefix KL, on-policy KL, "
                    "or verified sequence distillation"
                )
            onpolicy_loader = None
            onpolicy_iterator = None
            if onpolicy_cache_path is not None:
                onpolicy_records = torch.load(
                    onpolicy_cache_path, map_location="cpu"
                )["records"]
                if onpolicy_records:
                    onpolicy_dataset = CachedOnPolicyRecords(onpolicy_records)
                    onpolicy_loader = DataLoader(
                        onpolicy_dataset,
                        batch_size=cfg.micro_batch_size,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(seed + 17),
                        collate_fn=lambda rows: collate_onpolicy(rows, tokenizer.pad_token_id),
                    )
                    onpolicy_iterator = iter(onpolicy_loader)
            sequence_loader = None
            sequence_iterator = None
            if sequence_cache_path is not None:
                sequence_rows = torch.load(
                    sequence_cache_path, map_location="cpu"
                )["rows"]
                if sequence_rows:
                    sequence_loader = DataLoader(
                        sequence_rows,
                        batch_size=cfg.micro_batch_size,
                        shuffle=True,
                        generator=torch.Generator().manual_seed(seed + 29),
                        collate_fn=lambda rows: collate_tokenized(
                            rows, tokenizer.pad_token_id
                        ),
                    )
                    sequence_iterator = iter(sequence_loader)

            total_steps = math.ceil(len(loader) / cfg.grad_accum_steps)
            optimizer.zero_grad(set_to_none=True)
            loss_rows: list[dict[str, float]] = []
            processed_tokens = 0
            loss_tokens = 0
            kd_tokens = 0
            sequence_tokens = 0
            started = time.perf_counter()
            update = 0
            for micro_step, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                kd_labels = batch["kd_labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention)
                logits = outputs.logits[:, :-1]
                shifted_labels = labels[:, 1:]
                ce_mask = shifted_labels != -100
                kd_mask = kd_labels[:, 1:] != -100
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    shifted_labels.reshape(-1),
                    ignore_index=-100,
                )
                kd_terms = []
                if teacher_cache is not None:
                    student_log_probs = F.log_softmax(
                        logits.float() / cfg.kd_temperature, dim=-1
                    )
                    for row_index, artifact_id in enumerate(batch["artifact_ids"]):
                        entry = teacher_cache.get(artifact_id)
                        if entry is None:
                            continue
                        row_log_probs = student_log_probs[row_index][kd_mask[row_index]]
                        if row_log_probs.shape[0] != entry["response_tokens"]:
                            raise AssertionError(f"Teacher/student token misalignment: {artifact_id}")
                        kd_terms.append(approximate_forward_kl(row_log_probs, entry))
                        kd_tokens += int(entry["response_tokens"])
                if onpolicy_loader is not None:
                    try:
                        op_batch = next(onpolicy_iterator)
                    except StopIteration:
                        onpolicy_iterator = iter(onpolicy_loader)
                        op_batch = next(onpolicy_iterator)
                    op_ids = op_batch["input_ids"].to(device)
                    op_attention = op_batch["attention_mask"].to(device)
                    op_kd_labels = op_batch["kd_labels"].to(device)
                    op_logits = model(input_ids=op_ids, attention_mask=op_attention).logits[:, :-1]
                    op_log_probs = F.log_softmax(
                        op_logits.float() / cfg.kd_temperature, dim=-1
                    )
                    op_mask = op_kd_labels[:, 1:] != -100
                    for row_index, entry in enumerate(op_batch["teacher_entries"]):
                        row_log_probs = op_log_probs[row_index][op_mask[row_index]]
                        if row_log_probs.shape[0] != entry["response_tokens"]:
                            raise AssertionError("On-policy teacher/student token misalignment")
                        kd_terms.append(approximate_forward_kl(row_log_probs, entry))
                        kd_tokens += int(entry["response_tokens"])
                    processed_tokens += int(op_attention.sum().item())
                sequence_ce_terms = []
                if sequence_loader is not None:
                    try:
                        seq_batch = next(sequence_iterator)
                    except StopIteration:
                        sequence_iterator = iter(sequence_loader)
                        seq_batch = next(sequence_iterator)
                    seq_ids = seq_batch["input_ids"].to(device)
                    seq_attention = seq_batch["attention_mask"].to(device)
                    seq_labels = seq_batch["labels"].to(device)
                    seq_logits = model(
                        input_ids=seq_ids, attention_mask=seq_attention
                    ).logits[:, :-1]
                    seq_shifted = seq_labels[:, 1:]
                    sequence_ce_terms.append(F.cross_entropy(
                        seq_logits.reshape(-1, seq_logits.shape[-1]).float(),
                        seq_shifted.reshape(-1),
                        ignore_index=-100,
                    ))
                    sequence_tokens += int((seq_shifted != -100).sum().item())
                    processed_tokens += int(seq_attention.sum().item())
                kd = torch.stack(kd_terms).mean() if kd_terms else torch.zeros((), device=device)
                sequence_ce = (
                    torch.stack(sequence_ce_terms).mean()
                    if sequence_ce_terms else torch.zeros((), device=device)
                )
                if not kd_terms and not sequence_ce_terms:
                    loss = ce
                else:
                    # Keep verified task CE at full weight. Both peer KL and teacher-sequence
                    # CE are low-weight auxiliary signals on an identical extra-forward
                    # schedule; their processed tokens remain explicit in the cost ledger.
                    loss = ce + cfg.kd_alpha * (
                        cfg.kd_temperature**2 * kd + sequence_ce
                    )
                (loss / cfg.grad_accum_steps).backward()
                processed_tokens += int(attention.sum().item())
                loss_tokens += int(ce_mask.sum().item())
                if (micro_step + 1) % cfg.grad_accum_steps == 0 or micro_step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    lr = cosine_lr(update, total_steps, cfg)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_rows.append({
                        "update": float(update),
                        "loss": float(loss.detach().cpu()),
                        "ce": float(ce.detach().cpu()),
                        "kd": float(kd.detach().cpu()),
                        "sequence_ce": float(sequence_ce.detach().cpu()),
                        "lr": float(lr),
                    })
                    update += 1

            destination_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(destination_path, safe_serialization=True)
            tokenizer.save_pretrained(destination_path)
            next_optimizer = destination_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), next_optimizer)
            elapsed = time.perf_counter() - started
            ledger.emit(
                model_role=f"peer_{peer}",
                operation="train",
                source=str(source_path),
                destination=str(destination_path),
                elapsed_seconds=elapsed,
                processed_tokens=processed_tokens,
                loss_tokens=loss_tokens,
                updates=update,
                teacher_cache=str(teacher_cache_path) if teacher_cache_path else None,
                onpolicy_cache=str(onpolicy_cache_path) if onpolicy_cache_path else None,
                sequence_cache=str(sequence_cache_path) if sequence_cache_path else None,
                kd_tokens=kd_tokens,
                sequence_tokens=sequence_tokens,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "checkpoint": str(destination_path),
                "optimizer_state": str(next_optimizer),
                "elapsed_seconds": elapsed,
                "processed_tokens": processed_tokens,
                "loss_tokens": loss_tokens,
                "kd_tokens": kd_tokens,
                "sequence_tokens": sequence_tokens,
                "updates": update,
                "loss_log": loss_rows,
            }
        """
    ),
    markdown(
        r"""
        ## Larger-teacher target banks

        The teacher controls are deliberately strong. `large_teacher_single` uses one frozen
        greedy, verifier-passing eligibility trace per item and—when token IDs match—scores
        the same student-authored states as the peer arm with 1B logits. On a tokenizer
        mismatch it falls back explicitly to a single verified sequence target.
        `large_teacher_diverse` draws exactly as
        many attempts per item as the full two-rollout four-peer bank, keeps every
        verifier-passing attempt, and exposes the 400M students to a deterministically sampled
        matched quota. Both controls see each target student's two frozen rollout failures.

        Diverse larger-teacher targets are converted into the 400M tokenizer and learned
        through an auxiliary sequence-CE forward. This remains valid when tokenizers differ
        and gives the large teacher its reasoning trace, not merely its final answer. The
        notebook never aligns incompatible logit arrays or silently changes the treatment.
        """
    ),
    code(
        r"""
        def generate_larger_teacher_bank(
            teacher_model: Any,
            teacher_tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            seed: int,
            diverse: bool,
        ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
            bank: dict[str, list[dict[str, Any]]] = {
                row["artifact_id"]: [] for row in records
            }
            attempted_decoded_tokens = 0
            started = time.perf_counter()
            if not diverse:
                evaluation = evaluate_records(
                    teacher_model, teacher_tokenizer, records, cfg, device
                )
                attempted_decoded_tokens += sum(
                    int(item["decoded_tokens"]) for item in evaluation["items"]
                )
                for item in evaluation["items"]:
                    if item["correct"]:
                        bank[item["artifact_id"]].append({
                            "attempt": 0,
                            "output": item["output"],
                            "output_sha256": sha256_bytes(
                                item["output"].encode("utf-8")
                            ),
                        })
                attempts = 1
            else:
                teacher_cfg = replace(
                    cfg,
                    peer_draft_temperature=cfg.larger_teacher_temperature,
                    peer_draft_top_p=cfg.larger_teacher_top_p,
                )
                attempts = cfg.larger_teacher_diverse_attempts
                for attempt in range(attempts):
                    rollout = generate_rollouts_for_model(
                        teacher_model,
                        teacher_tokenizer,
                        records,
                        teacher_cfg,
                        device,
                        seed=seed + attempt * 1_000_003,
                    )
                    attempted_decoded_tokens += sum(
                        int(item["decoded_tokens"]) for item in rollout.values()
                    )
                    for artifact_id, item in rollout.items():
                        if item["correct"]:
                            bank[artifact_id].append({
                                "attempt": attempt,
                                "output": item["output"],
                                "output_sha256": item["output_sha256"],
                            })
            valid = sum(len(rows) for rows in bank.values())
            unique = sum(
                len({row["output_sha256"] for row in rows})
                for rows in bank.values()
            )
            report = {
                "policy": "diverse_sampling" if diverse else "single_greedy_trace",
                "attempts_per_item": attempts,
                "attempted_outputs": attempts * len(records),
                "verified_valid_outputs": valid,
                "unique_valid_outputs_within_item": unique,
                "items_with_any_valid_output": sum(bool(rows) for rows in bank.values()),
                "valid_decoded_tokens": int(sum(
                    len(teacher_tokenizer(
                        str(row["output"]), add_special_tokens=False
                    )["input_ids"])
                    for outputs in bank.values() for row in outputs
                )),
                "attempted_decoded_tokens": attempted_decoded_tokens,
                "elapsed_seconds": time.perf_counter() - started,
            }
            return bank, report


        def build_larger_teacher_sequence_caches(
            records: Sequence[Mapping[str, Any]],
            student_tokenizer: Any,
            student_rollouts: Sequence[Mapping[str, Mapping[str, Any]]],
            student_confirmations: Sequence[Mapping[str, Mapping[str, Any]]],
            teacher_bank: Mapping[str, Sequence[Mapping[str, Any]]],
            matched_targets_by_student: Sequence[int],
            cfg: ExperimentConfig,
            seed: int,
            cache_paths: Sequence[Path],
        ) -> list[dict[str, Any]]:
            if len(matched_targets_by_student) != cfg.n_peers:
                raise ValueError("Need one frozen peer-target quota per 400M student")
            record_by_id = {row["artifact_id"]: row for row in records}
            payloads = []
            for student in range(cfg.n_peers):
                candidates: list[tuple[str, int, str, dict[str, Any]]] = []
                for artifact_id, outputs in teacher_bank.items():
                    if student_rollouts[student][artifact_id]["correct"]:
                        continue
                    if student_confirmations[student][artifact_id]["correct"]:
                        continue
                    item_candidates = []
                    for output in outputs:
                        tokenized = tokenize_free_response(
                            student_tokenizer,
                            record_by_id[artifact_id],
                            str(output["output"]),
                            cfg,
                        )
                        if tokenized is None:
                            continue
                        tokenized = dict(tokenized)
                        tokenized["artifact_id"] = (
                            f"{artifact_id}:lt:{int(output['attempt'])}:s{student}"
                        )
                        item_candidates.append((
                            sha256_json([
                                seed, student, artifact_id, output["attempt"],
                                output["output_sha256"],
                            ]),
                            int(output["attempt"]),
                            str(output["output_sha256"]),
                            tokenized,
                        ))
                    # Match the peer arm's one accepted tutor target per student/item.
                    # Diverse sampling improves the trace choice and success coverage; it
                    # does not buy repeated copies of the same input.
                    if item_candidates:
                        candidates.append(min(item_candidates, key=lambda row: row[0]))
                candidates.sort(key=lambda row: row[0])
                quota = int(matched_targets_by_student[student])
                if len(candidates) < quota:
                    raise RuntimeError(
                        f"Larger teacher produced only {len(candidates)} usable verified "
                        f"targets for student {student}; the peer arm exposed {quota}. "
                        "Do not upsample or weaken the verifier—this calibration is invalid."
                    )
                selected = candidates[:quota]
                rows = [row[3] for row in selected]
                payload = {
                    "protocol_version": cfg.protocol_version,
                    "student": student,
                    "matched_peer_target_quota": quota,
                    "candidate_verified_sequences": len(candidates),
                    "selected_unique_output_hashes": len({row[2] for row in selected}),
                    "selected_response_tokens": int(sum(
                        sum(label != -100 for label in row["labels"]) for row in rows
                    )),
                    "rows": rows,
                }
                cache_paths[student].parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_paths[student])
                payloads.append(payload)
            return payloads


        def build_external_teacher_onpolicy_caches(
            teacher_model: Any,
            records: Sequence[Mapping[str, Any]],
            student_tokenizer: Any,
            student_rollouts: Sequence[Mapping[str, Mapping[str, Any]]],
            student_confirmations: Sequence[Mapping[str, Mapping[str, Any]]],
            teacher_bank: Mapping[str, Sequence[Mapping[str, Any]]],
            matched_targets_by_student: Sequence[int],
            cfg: ExperimentConfig,
            seed: int,
            device: str,
            cache_paths: Sequence[Path],
        ) -> list[dict[str, Any]]:
            # Exact topology-matched external-teacher control: the frozen 1B teacher scores
            # the same 400M student-authored states used by FRR. This is forbidden unless the
            # tokenizer preflight has already established exact token-ID compatibility.
            record_by_id = {row["artifact_id"]: row for row in records}
            teacher_model.eval()
            payloads = []
            for student in range(cfg.n_peers):
                candidate_ids = [
                    artifact_id for artifact_id, outputs in teacher_bank.items()
                    if outputs
                    and not student_rollouts[student][artifact_id]["correct"]
                    and not student_confirmations[student][artifact_id]["correct"]
                ]
                candidate_ids.sort(key=lambda artifact_id: sha256_json([
                    seed, student, artifact_id, "external_teacher_onpolicy"
                ]))
                quota = int(matched_targets_by_student[student])
                tokenized_rows = []
                for artifact_id in candidate_ids:
                    row = tokenize_free_response(
                        student_tokenizer,
                        record_by_id[artifact_id],
                        str(student_rollouts[student][artifact_id]["output"]),
                        cfg,
                    )
                    if row is not None:
                        tokenized_rows.append(row)
                    if len(tokenized_rows) == quota:
                        break
                if len(tokenized_rows) != quota:
                    raise RuntimeError(
                        f"External teacher has {len(tokenized_rows)} usable on-policy targets "
                        f"for student {student}; the peer quota is {quota}."
                    )
                cache: dict[str, Any] = {}
                for start in range(0, len(tokenized_rows), cfg.micro_batch_size):
                    batch_rows = tokenized_rows[start:start + cfg.micro_batch_size]
                    batch = collate_tokenized(batch_rows, student_tokenizer.pad_token_id)
                    input_ids = batch["input_ids"].to(device)
                    attention = batch["attention_mask"].to(device)
                    kd_labels = batch["kd_labels"].to(device)
                    with torch.no_grad():
                        logits = teacher_model(
                            input_ids=input_ids, attention_mask=attention
                        ).logits[:, :-1]
                        logp = F.log_softmax(
                            logits.float() / cfg.kd_temperature, dim=-1
                        )
                    mask = kd_labels[:, 1:] != -100
                    for row_index, row in enumerate(batch_rows):
                        cache[row["artifact_id"]] = topk_entry_from_log_probs(
                            logp[row_index][mask[row_index]],
                            cfg,
                            selected_teacher="external_larger_teacher",
                            prefix_policy="student_onpolicy",
                            input_ids=torch.tensor(
                                row["input_ids"], dtype=torch.int32
                            ),
                            labels=torch.tensor(row["labels"], dtype=torch.int32),
                            kd_labels=torch.tensor(
                                row["kd_labels"], dtype=torch.int32
                            ),
                        )
                payload = {
                    "protocol_version": cfg.protocol_version,
                    "policy": "external_larger_teacher_onpolicy",
                    "student": student,
                    "temperature": cfg.kd_temperature,
                    "topk": cfg.teacher_topk,
                    "records": cache,
                    "eligible_records": len(cache),
                    "total_response_tokens": int(sum(
                        int(entry["response_tokens"]) for entry in cache.values()
                    )),
                }
                cache_paths[student].parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_paths[student])
                payloads.append(payload)
            return payloads
        """
    ),
    markdown(
        r"""
        ## Election and promotion

        Initial champion selection is descriptive; all arms receive the same selected model.
        Later promotion is deliberately harder: an election shard nominates one challenger,
        while an independent promotion shard must show at least the practical margin and a
        paired bootstrap lower bound above zero. Otherwise the incumbent stays frozen. This
        hysteresis prevents noisy one-point wins from repeatedly moving the teacher.
        """
    ),
    code(
        r"""
        def paired_bootstrap_interval(
            candidate: np.ndarray,
            incumbent: np.ndarray,
            confidence: float,
            resamples: int,
            seed: int,
        ) -> tuple[float, float, float]:
            if candidate.shape != incumbent.shape:
                raise ValueError("Paired vectors must have the same shape.")
            delta = candidate - incumbent
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
            values = delta[indices].mean(axis=1)
            alpha = 1 - confidence
            return float(delta.mean()), float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


        def consider_promotion(
            incumbent: int,
            election_evaluations: Sequence[Mapping[str, Any]],
            promotion_evaluations: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            seed: int,
            boundary: int,
            retention_nll: Sequence[float] | None = None,
            baseline_retention_nll: Sequence[float] | None = None,
        ) -> dict[str, Any]:
            challengers = [p for p in range(cfg.n_peers) if p != incumbent]
            candidate = max(
                challengers,
                key=lambda p: (election_evaluations[p]["macro_skill_accuracy"], -p),
            )
            point, lower, upper = paired_bootstrap_interval(
                item_vector(promotion_evaluations[candidate]),
                item_vector(promotion_evaluations[incumbent]),
                cfg.promotion_confidence,
                cfg.promotion_bootstrap_resamples,
                seed=seed * 1009 + boundary,
            )
            retention_safe = True
            retention = None
            if cfg.require_retention_corpus:
                if retention_nll is None or baseline_retention_nll is None:
                    raise RuntimeError("Promotion requires current and baseline retention NLL")
                limit = baseline_retention_nll[candidate] * (
                    1 + cfg.retention_nll_relative_margin
                )
                retention_safe = bool(retention_nll[candidate] <= limit)
                retention = {
                    "candidate_nll": float(retention_nll[candidate]),
                    "baseline_nll": float(baseline_retention_nll[candidate]),
                    "maximum_nll": float(limit),
                    "safe": retention_safe,
                }
            # The confidence bound itself—not only the point estimate—must clear the
            # practical margin. The two-sided interval is conservative for promotion.
            promoted = bool(lower > cfg.promotion_margin and retention_safe)
            return {
                "incumbent": incumbent,
                "candidate": candidate,
                "paired_difference": point,
                "ci": [lower, upper],
                "margin": cfg.promotion_margin,
                "retention": retention,
                "promoted": promoted,
                "next_champion": candidate if promoted else incumbent,
            }
        """
    ),
    code(
        r"""
        def checkpoint_eval(
            checkpoint: Path,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
        ) -> dict[str, Any]:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), local_files_only=True, torch_dtype=dtype
            ).to(device)
            result = evaluate_records(model, tokenizer, records, cfg, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result


        def evaluate_population(
            checkpoints: Sequence[Path],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
        ) -> list[dict[str, Any]]:
            return [checkpoint_eval(path, tokenizer, records, cfg, device) for path in checkpoints]


        def load_retention_texts(cfg: ExperimentConfig) -> list[str]:
            if not cfg.retention_text_jsonl:
                if cfg.require_retention_corpus:
                    raise RuntimeError(
                        "OLMO400M_RETENTION_JSONL is required for promotion and final safety checks"
                    )
                return []
            path = Path(cfg.retention_text_jsonl).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            texts = [str(row["text"]) for row in rows if str(row.get("text", "")).strip()]
            if not texts:
                raise ValueError("Retention JSONL must contain nonempty {'text': ...} rows")
            return texts


        def evaluate_retention_nll(
            model: Any,
            tokenizer: Any,
            texts: Sequence[str],
            cfg: ExperimentConfig,
            device: str,
        ) -> dict[str, Any]:
            if torch is None:
                raise RuntimeError("PyTorch is required for retention evaluation")
            model.eval()
            total_nll = 0.0
            total_tokens = 0
            started = time.perf_counter()
            for text_value in texts:
                token_ids = tokenizer(
                    text_value, add_special_tokens=True, return_attention_mask=False
                )["input_ids"]
                for offset in range(0, len(token_ids) - 1, cfg.max_length - 1):
                    chunk = token_ids[offset:offset + cfg.max_length]
                    if len(chunk) < 2:
                        continue
                    input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(input_ids=input_ids).logits[:, :-1].float()
                    targets = input_ids[:, 1:]
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    count = int(targets.numel())
                    total_nll += float(loss_sum.cpu())
                    total_tokens += count
                    if total_tokens >= cfg.retention_max_tokens:
                        break
                if total_tokens >= cfg.retention_max_tokens:
                    break
            if total_tokens == 0:
                raise RuntimeError("Retention evaluation produced zero scored tokens")
            nll = total_nll / total_tokens
            return {
                "nll": nll,
                "perplexity": float(math.exp(min(nll, 50))),
                "tokens": total_tokens,
                "elapsed_seconds": time.perf_counter() - started,
            }


        def checkpoint_retention_nll(
            checkpoint: Path,
            tokenizer: Any,
            texts: Sequence[str],
            cfg: ExperimentConfig,
            device: str,
        ) -> dict[str, Any]:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), local_files_only=True, torch_dtype=dtype
            ).to(device)
            result = evaluate_retention_nll(model, tokenizer, texts, cfg, device)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result


        def prepare_warmup_quartet(cfg: ExperimentConfig, manifest: Mapping[str, Any], seed: int, device: str) -> list[Path]:
            require_training(cfg)
            root = cfg.output_dir / f"seed_{seed}" / "warmup"
            complete = root / "COMPLETE.json"
            if complete.exists():
                payload = json.loads(complete.read_text())
                return [Path(path) for path in payload["checkpoints"]]
            base = Path(cfg.model_path).expanduser().resolve()
            _, tokenizer, preflight = load_and_validate_model(cfg, device="cpu")
            del _
            checkpoints: list[Path] = []
            for peer in range(cfg.n_peers):
                warmup_name = (
                    "warmup_shared"
                    if cfg.diversity_mode in {"exact_clone", "order_only"}
                    else f"warmup_peer_{peer}"
                )
                records = read_jsonl(Path(manifest["partitions"][warmup_name]["path"]))
                destination = root / f"peer_{peer}"
                ledger = CostLedger(root / "cost_events.jsonl", "warmup", seed)
                peer_seed = seed * 10_000 if cfg.diversity_mode == "exact_clone" else seed * 10_000 + peer
                if cfg.diversity_mode == "exact_clone" and peer > 0:
                    # A true implementation null must be byte-identical, not merely trained
                    # with the same nominal seed under potentially nondeterministic kernels.
                    shutil.copytree(checkpoints[0], destination)
                    ledger.emit(
                        model_role=f"peer_{peer}",
                        operation="copy_exact_clone",
                        source=str(checkpoints[0]),
                        destination=str(destination),
                    )
                else:
                    train_checkpoint(
                        base,
                        destination,
                        tokenizer,
                        records,
                        cfg,
                        seed=peer_seed,
                        peer=peer,
                        arm="warmup",
                        device=device,
                        ledger=ledger,
                    )
                checkpoints.append(destination)
            atomic_json(complete, {
                "seed": seed,
                "diversity_mode": cfg.diversity_mode,
                "checkpoints": [str(path) for path in checkpoints],
                "preflight": preflight,
            })
            return checkpoints


        def prepare_population_state(
            cfg: ExperimentConfig,
            manifest: Mapping[str, Any],
            checkpoints: Sequence[Path],
            seed: int,
            device: str,
        ) -> dict[str, Any]:
            # Evaluate/gate the warmed population once, before forking experimental arms.
            root = cfg.output_dir / f"seed_{seed}" / "population_state.json"
            if root.exists():
                payload = json.loads(root.read_text())
                if payload["manifest_sha256"] != manifest["manifest_sha256"]:
                    raise RuntimeError("Cached population state belongs to a different manifest")
                return payload
            _, tokenizer, _ = load_and_validate_model(cfg, device="cpu")
            del _
            election_records = read_jsonl(Path(manifest["partitions"]["election_r0"]["path"]))
            confirmation_records = read_jsonl(Path(manifest["partitions"]["promotion_r0"]["path"]))
            election_eval = evaluate_population(checkpoints, tokenizer, election_records, cfg, device)
            confirmation_eval = evaluate_population(
                checkpoints, tokenizer, confirmation_records, cfg, device
            )
            complementarity = population_complementarity(election_eval, cfg)
            require_population_gate(complementarity)
            retention_texts = load_retention_texts(cfg)
            retention = [
                checkpoint_retention_nll(path, tokenizer, retention_texts, cfg, device)
                for path in checkpoints
            ] if retention_texts else []
            baseline_nll = [row["nll"] for row in retention] if retention else None
            initial_decision = consider_promotion(
                incumbent=0,
                election_evaluations=election_eval,
                promotion_evaluations=confirmation_eval,
                cfg=cfg,
                seed=seed,
                boundary=0,
                retention_nll=baseline_nll,
                baseline_retention_nll=baseline_nll,
            )
            payload = {
                "seed": seed,
                "manifest_sha256": manifest["manifest_sha256"],
                "diversity_mode": cfg.diversity_mode,
                "complementarity": complementarity,
                "routing_skill_scores": [row["per_skill"] for row in election_eval],
                "baseline_retention": retention,
                "initial_decision": {"boundary": 0, **initial_decision},
                "initial_champion": initial_decision["next_champion"],
            }
            atomic_json(root, payload)
            return payload
        """
    ),
    markdown(
        r"""
        ## Stage 0b: prove that the larger teacher is actually stronger

        Parameter count is not the behavioral gate. On the first calibration shard, select
        the best warmed 400M peer. On the untouched confirmation shard, compare that frozen
        peer with the larger teacher item by item. Stage 2 is blocked unless both shard point
        differences clear the practical margin and the confirmation one-sided bootstrap lower
        bound is above zero. This gate is decided before any championship arm trains and never
        uses the sealed final test.
        """
    ),
    code(
        r"""
        def one_sided_paired_bootstrap_lower(
            candidate: np.ndarray,
            control: np.ndarray,
            confidence: float,
            resamples: int,
            seed: int,
        ) -> float:
            if candidate.shape != control.shape or candidate.ndim != 1:
                raise ValueError("Aligned one-dimensional item vectors are required")
            delta = candidate - control
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, len(delta), size=(resamples, len(delta)))
            return float(np.quantile(delta[picks].mean(axis=1), 1 - confidence))


        def prepare_championship_teacher_state(
            cfg: ExperimentConfig,
            manifest: Mapping[str, Any],
            warmup_checkpoints: Sequence[Path],
            seed: int,
            device: str,
        ) -> dict[str, Any]:
            root = cfg.output_dir / f"seed_{seed}" / "larger_teacher_state.json"
            if root.exists():
                cached = json.loads(root.read_text())
                if cached["manifest_sha256"] != manifest["manifest_sha256"]:
                    raise RuntimeError("Cached larger-teacher gate uses a different data manifest")
                configured_teacher = str(
                    Path(cfg.larger_teacher_model_path).expanduser().resolve()
                ) if cfg.larger_teacher_model_path else ""
                cached_teacher = cached["larger_teacher_manifest"]["artifact"]["path"]
                if configured_teacher != cached_teacher:
                    raise RuntimeError(
                        "Cached larger-teacher gate belongs to a different checkpoint path"
                    )
                if not cached["superiority_gate_passed"]:
                    raise RuntimeError("Cached larger teacher failed the superiority gate")
                return cached

            student_model, student_tokenizer, student_manifest = load_and_validate_model(
                cfg, device="cpu"
            )
            del student_model
            election_records = read_jsonl(Path(
                manifest["partitions"]["election_r0"]["path"]
            ))
            confirmation_records = read_jsonl(Path(
                manifest["partitions"]["promotion_r0"]["path"]
            ))
            peer_election = evaluate_population(
                warmup_checkpoints, student_tokenizer, election_records, cfg, device
            )
            best_peer = max(
                range(cfg.n_peers),
                key=lambda index: (peer_election[index]["macro_skill_accuracy"], -index),
            )
            peer_confirmation = checkpoint_eval(
                warmup_checkpoints[best_peer], student_tokenizer,
                confirmation_records, cfg, device,
            )
            teacher_model, teacher_tokenizer, teacher_manifest = (
                load_and_validate_larger_teacher(
                    cfg,
                    student_tokenizer,
                    int(student_manifest["parameter_count"]),
                    device=device,
                )
            )
            teacher_election = evaluate_records(
                teacher_model, teacher_tokenizer, election_records, cfg, device
            )
            teacher_confirmation = evaluate_records(
                teacher_model, teacher_tokenizer, confirmation_records, cfg, device
            )
            election_delta = float(
                teacher_election["macro_skill_accuracy"]
                - peer_election[best_peer]["macro_skill_accuracy"]
            )
            confirmation_delta = float(
                teacher_confirmation["macro_skill_accuracy"]
                - peer_confirmation["macro_skill_accuracy"]
            )
            lower = one_sided_paired_bootstrap_lower(
                item_vector(teacher_confirmation),
                item_vector(peer_confirmation),
                cfg.larger_teacher_superiority_confidence,
                cfg.promotion_bootstrap_resamples,
                seed=seed * 1009 + 911,
            )
            passed = bool(
                election_delta >= cfg.larger_teacher_calibration_margin
                and confirmation_delta >= cfg.larger_teacher_calibration_margin
                and lower > 0
            )
            payload = {
                "seed": seed,
                "manifest_sha256": manifest["manifest_sha256"],
                "best_peer_selected_on_election": best_peer,
                "election_delta": election_delta,
                "confirmation_delta": confirmation_delta,
                "confirmation_one_sided_lower": lower,
                "required_point_margin": cfg.larger_teacher_calibration_margin,
                "confidence": cfg.larger_teacher_superiority_confidence,
                "superiority_gate_passed": passed,
                "student_model_manifest": student_manifest,
                "larger_teacher_manifest": teacher_manifest,
            }
            payload["sha256"] = sha256_json(payload)
            atomic_json(root, payload)
            del teacher_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not passed:
                raise RuntimeError(
                    "The proposed larger teacher was not demonstrably stronger on both "
                    f"calibration shards: election={election_delta:.4f}, "
                    f"confirmation={confirmation_delta:.4f}, lower={lower:.4f}."
                )
            return payload
        """
    ),
    code(
        r"""
        def load_frozen_population(
            checkpoints: Sequence[Path], cfg: ExperimentConfig, device: str
        ) -> list[Any]:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            return [
                AutoModelForCausalLM.from_pretrained(
                    str(path), local_files_only=True, torch_dtype=dtype
                ).to(device).eval()
                for path in checkpoints
            ]


        def summarize_cost_events(path: Path) -> dict[str, Any]:
            rows = read_jsonl(path) if path.exists() else []
            return {
                "events": len(rows),
                "device_seconds": float(sum(float(row.get("elapsed_seconds", 0)) for row in rows)),
                "student_processed_tokens": int(sum(int(row.get("processed_tokens", 0)) for row in rows)),
                "student_loss_tokens": int(sum(int(row.get("loss_tokens", 0)) for row in rows)),
                "kd_tokens": int(sum(int(row.get("kd_tokens", 0)) for row in rows)),
                "sequence_tokens": int(sum(int(row.get("sequence_tokens", 0)) for row in rows)),
                "optimizer_updates": int(sum(int(row.get("updates", 0)) for row in rows)),
                "attempted_outputs": int(sum(int(row.get("attempted_outputs", 0)) for row in rows)),
                "decoded_tokens": int(sum(int(row.get("decoded_tokens", 0)) for row in rows)),
                "accepted_targets": int(sum(int(row.get("accepted_targets", 0)) for row in rows)),
                "scored_tokens": int(sum(int(row.get("scored_tokens", 0)) for row in rows)),
                "model_evaluations": int(sum(int(row.get("model_evaluations", 0)) for row in rows)),
            }


        def logged_population_eval(
            checkpoints: Sequence[Path],
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            ledger: CostLedger,
            operation: str,
            boundary: int,
        ) -> list[dict[str, Any]]:
            result = evaluate_population(checkpoints, tokenizer, records, cfg, device)
            ledger.emit(
                model_role="population",
                operation=operation,
                boundary=boundary,
                model_evaluations=len(result),
                records=len(records),
                elapsed_seconds=sum(row["elapsed_seconds"] for row in result),
            )
            return result


        def logged_retention_eval(
            checkpoints: Sequence[Path],
            tokenizer: Any,
            texts: Sequence[str],
            cfg: ExperimentConfig,
            device: str,
            ledger: CostLedger,
            boundary: int,
        ) -> list[dict[str, Any]]:
            result = [
                checkpoint_retention_nll(path, tokenizer, texts, cfg, device)
                for path in checkpoints
            ]
            ledger.emit(
                model_role="population",
                operation="retention_eval",
                boundary=boundary,
                model_evaluations=len(result),
                scored_tokens=sum(row["tokens"] for row in result),
                elapsed_seconds=sum(row["elapsed_seconds"] for row in result),
            )
            return result


        def run_arm(
            cfg: ExperimentConfig,
            manifest: Mapping[str, Any],
            warmup_checkpoints: Sequence[Path],
            population_state: Mapping[str, Any],
            seed: int,
            arm: str,
            device: str,
        ) -> dict[str, Any]:
            require_training(cfg)
            if arm not in cfg.arms:
                raise ValueError(arm)
            if arm in {"large_teacher_single", "large_teacher_diverse"}:
                raise ValueError(
                    "Larger-teacher arms require the superiority gate and matched peer "
                    "quotas; call run_championship_seed."
                )
            arm_root = cfg.output_dir / f"seed_{seed}" / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            ledger = CostLedger(arm_root / "cost_events.jsonl", arm, seed)
            _, tokenizer, _ = load_and_validate_model(cfg, device="cpu")
            del _

            current = []
            for peer, warmup in enumerate(warmup_checkpoints):
                target = arm_root / "r0" / f"peer_{peer}"
                if not target.exists():
                    shutil.copytree(warmup, target)
                current.append(target)

            initial_champion = int(population_state["initial_champion"])
            champion = initial_champion  # governance/deployment identity, not a universal teacher
            history: list[dict[str, Any]] = [dict(population_state["initial_decision"])]
            routing_skill_scores = [dict(row) for row in population_state["routing_skill_scores"]]
            baseline_retention = [
                float(row["nll"]) for row in population_state["baseline_retention"]
            ] if population_state["baseline_retention"] else None
            retention_texts = load_retention_texts(cfg)
            current_retention = list(population_state["baseline_retention"])
            initial_diversity = dict(population_state["complementarity"])
            no_promotion_streak = 0
            rounds_completed = 0
            round_reports: list[dict[str, Any]] = []

            for round_index in range(cfg.n_rounds):
                exchange_records = read_jsonl(Path(
                    manifest["partitions"][f"exchange_r{round_index}"]["path"]
                ))
                extra_records = read_jsonl(Path(
                    manifest["partitions"][f"equal_cost_r{round_index}"]["path"]
                )) if arm == "gold_private_equal_cost" else []
                teacher_cache_by_peer: dict[int, Path] = {}
                onpolicy_cache_by_peer: dict[int, Path] = {}
                route_report = None
                target_attempted_outputs = 0
                target_decoded_tokens = 0
                target_accepted = 0

                cache_arms = {
                    "self_snapshot_op", "frr_goldprefix", "frr_onpolicy",
                    "peer_frr_onpolicy", "global_champion_goldprefix",
                    "unfiltered_mutual_mean",
                }
                if arm in cache_arms:
                    cache_started = time.perf_counter()
                    frozen_models = load_frozen_population(current, cfg, device)
                    cache_paths = [
                        arm_root / f"r{round_index}" / f"peer_{peer}_targets.pt"
                        for peer in range(cfg.n_peers)
                    ]
                    if arm == "unfiltered_mutual_mean":
                        payloads = build_leave_one_out_teacher_caches(
                            frozen_models, tokenizer, exchange_records, cfg, device,
                            cache_paths, eligibility_by_teacher=None,
                        )
                        teacher_cache_by_peer = {
                            peer: cache_paths[peer] for peer in range(cfg.n_peers)
                        }
                        route_report = {
                            "policy": "unfiltered_leave_one_out_mean",
                            "eligible_by_student": [payload["eligible_records"] for payload in payloads],
                        }
                    else:
                        rollouts = generate_population_rollouts(
                            frozen_models, tokenizer, exchange_records, cfg, device,
                            seed=seed * 100 + round_index,
                        )
                        confirmation_rollouts = (
                            generate_population_rollouts(
                                frozen_models, tokenizer, exchange_records, cfg, device,
                                seed=seed * 100 + round_index + 50_000_003,
                            )
                            if cfg.require_itemwise_teacher_correctness else None
                        )
                        rollout_banks = [rollouts]
                        if confirmation_rollouts is not None:
                            rollout_banks.append(confirmation_rollouts)
                        target_attempted_outputs = sum(
                            len(bank) for population in rollout_banks for bank in population
                        )
                        target_decoded_tokens = sum(
                            int(item["decoded_tokens"])
                            for population in rollout_banks
                            for bank in population
                            for item in bank.values()
                        )
                        edges, route_report = route_rescue_edges(
                            exchange_records, rollouts, confirmation_rollouts,
                            routing_skill_scores, cfg,
                            seed=seed * 100 + round_index,
                        )
                        route_report["policy"] = arm
                        edge_fraction = route_report["total_edges"] / (
                            cfg.n_peers * len(exchange_records)
                        )
                        route_report["edge_fraction"] = edge_fraction
                        if arm in {
                            "self_snapshot_op", "frr_onpolicy", "peer_frr_onpolicy",
                            "frr_goldprefix",
                        } and (
                            edge_fraction < cfg.minimum_rescue_edge_fraction
                        ):
                            raise RuntimeError(
                                f"Rescue signal is degenerate: {edge_fraction:.4f} < "
                                f"{cfg.minimum_rescue_edge_fraction:.4f}"
                            )
                        if arm == "global_champion_goldprefix":
                            global_edges = [dict() for _ in range(cfg.n_peers)]
                            for student in range(cfg.n_peers):
                                for record in exchange_records:
                                    artifact_id = record["artifact_id"]
                                    if (
                                        student != champion
                                        and not rollouts[student][artifact_id]["correct"]
                                        and rollouts[champion][artifact_id]["correct"]
                                        and (
                                            confirmation_rollouts is None
                                            or (
                                                not confirmation_rollouts[student][artifact_id]["correct"]
                                                and confirmation_rollouts[champion][artifact_id]["correct"]
                                            )
                                        )
                                    ):
                                        global_edges[student][artifact_id] = champion
                            edges = global_edges
                            route_report["global_champion"] = champion
                        if arm in {"frr_goldprefix", "global_champion_goldprefix"}:
                            payloads = build_routed_goldprefix_caches(
                                frozen_models, tokenizer, exchange_records, edges, cfg,
                                device, cache_paths,
                            )
                            route_report["cached_kd_records_by_student"] = [
                                payload["eligible_records"] for payload in payloads
                            ]
                            route_report["cached_kd_tokens_by_student"] = [
                                int(sum(
                                    int(entry["response_tokens"])
                                    for entry in payload["records"].values()
                                ))
                                for payload in payloads
                            ]
                            if arm == "frr_goldprefix":
                                # Match FRR-OP's auxiliary student-forward schedule. The only
                                # intended mechanism difference is canonical vs student prefix.
                                onpolicy_cache_by_peer = {
                                    peer: cache_paths[peer] for peer in range(cfg.n_peers)
                                }
                            else:
                                teacher_cache_by_peer = {
                                    peer: cache_paths[peer] for peer in range(cfg.n_peers)
                                }
                        elif arm in {"frr_onpolicy", "peer_frr_onpolicy", "self_snapshot_op"}:
                            payloads = build_routed_onpolicy_caches(
                                frozen_models, tokenizer, exchange_records, rollouts, edges,
                                cfg, device, cache_paths,
                                self_teacher=(arm == "self_snapshot_op"),
                            )
                            cached_records = sum(
                                payload["eligible_records"] for payload in payloads
                            )
                            cached_fraction = cached_records / (
                                cfg.n_peers * len(exchange_records)
                            )
                            route_report["cached_kd_records_by_student"] = [
                                payload["eligible_records"] for payload in payloads
                            ]
                            route_report["cached_kd_tokens_by_student"] = [
                                int(sum(
                                    int(entry["response_tokens"])
                                    for entry in payload["records"].values()
                                ))
                                for payload in payloads
                            ]
                            route_report["rejected_answer_spans_by_student"] = [
                                payload["rejected_missing_or_empty_answer_span"]
                                for payload in payloads
                            ]
                            route_report["cached_kd_fraction"] = cached_fraction
                            if cached_fraction < cfg.minimum_rescue_edge_fraction:
                                raise RuntimeError(
                                    f"Answer-only on-policy signal is degenerate: "
                                    f"{cached_fraction:.4f} < "
                                    f"{cfg.minimum_rescue_edge_fraction:.4f}"
                                )
                            onpolicy_cache_by_peer = {
                                peer: cache_paths[peer] for peer in range(cfg.n_peers)
                            }
                    if route_report is not None:
                        target_accepted = int(sum(
                            route_report.get("cached_kd_records_by_student", [])
                        ))
                    ledger.emit(
                        model_role="frozen_population",
                        operation="build_peer_targets",
                        round=round_index,
                        elapsed_seconds=time.perf_counter() - cache_started,
                        attempted_outputs=target_attempted_outputs,
                        decoded_tokens=target_decoded_tokens,
                        accepted_targets=target_accepted,
                        route_report=route_report,
                    )
                    del frozen_models
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                next_paths = list(current)
                for peer in range(cfg.n_peers):
                    private_records = read_jsonl(Path(
                        manifest["partitions"][f"private_peer_{peer}_r{round_index}"]["path"]
                    ))
                    train_records = exchange_records + private_records + extra_records
                    source_path = current[peer]
                    destination = arm_root / f"r{round_index + 1}" / f"peer_{peer}"
                    train_checkpoint(
                        source_path,
                        destination,
                        tokenizer,
                        train_records,
                        cfg,
                        seed=seed * 1_000_000 + round_index * 10_000 + peer,
                        peer=peer,
                        arm=arm,
                        device=device,
                        ledger=ledger,
                        teacher_cache_path=teacher_cache_by_peer.get(peer),
                        onpolicy_cache_path=onpolicy_cache_by_peer.get(peer),
                        optimizer_state_path=source_path / "optimizer.pt",
                    )
                    next_paths[peer] = destination
                current = next_paths
                rounds_completed += 1

                boundary = round_index + 1
                election_records = read_jsonl(Path(
                    manifest["partitions"][f"election_r{boundary}"]["path"]
                ))
                promotion_records = read_jsonl(Path(
                    manifest["partitions"][f"promotion_r{boundary}"]["path"]
                ))
                election_eval = logged_population_eval(
                    current, tokenizer, election_records, cfg, device, ledger,
                    operation="routing_election_eval", boundary=boundary,
                )
                promotion_eval = logged_population_eval(
                    current, tokenizer, promotion_records, cfg, device, ledger,
                    operation="promotion_confirmation_eval", boundary=boundary,
                )
                current_retention = logged_retention_eval(
                    current, tokenizer, retention_texts, cfg, device, ledger, boundary
                ) if retention_texts else []
                current_nll = [row["nll"] for row in current_retention] if current_retention else None
                decision = consider_promotion(
                    champion, election_eval, promotion_eval, cfg, seed, boundary,
                    retention_nll=current_nll,
                    baseline_retention_nll=baseline_retention,
                )
                champion = decision["next_champion"]
                history.append({"boundary": boundary, **decision})
                routing_skill_scores = [row["per_skill"] for row in election_eval]
                boundary_diversity = population_complementarity(election_eval, cfg)
                diversity_warning = bool(
                    boundary_diversity["oracle_headroom"]
                    < initial_diversity["oracle_headroom"] * cfg.diversity_retention_fraction_min
                    or min(boundary_diversity["unique_correct_fraction"])
                    < min(initial_diversity["unique_correct_fraction"])
                    * cfg.diversity_retention_fraction_min
                )
                round_reports.append({
                    "round": round_index,
                    "route": route_report,
                    "promotion": decision,
                    "diversity": boundary_diversity,
                    "diversity_warning": diversity_warning,
                })
                no_promotion_streak = 0 if decision["promoted"] else no_promotion_streak + 1
                if cfg.enable_futility_stop and no_promotion_streak >= cfg.max_consecutive_no_promotions:
                    break

            preaudit = {
                "protocol_version": cfg.protocol_version,
                "seed": seed,
                "arm": arm,
                "initial_champion": initial_champion,
                "selected_peer": champion,
                "history": history,
                "rounds_completed": rounds_completed,
                "round_reports": round_reports,
                "final_routing_skill_scores": routing_skill_scores,
                "checkpoints": [str(path) for path in current],
                "retention": current_retention,
                "baseline_retention_nll": baseline_retention,
                "data_manifest_sha256": manifest["manifest_sha256"],
            }
            preaudit["cost"] = summarize_cost_events(ledger.path)
            preaudit["preaudit_sha256"] = sha256_json(preaudit)
            atomic_json(arm_root / "preaudit_result.json", preaudit)
            return preaudit


        def run_larger_teacher_arm(
            cfg: ExperimentConfig,
            manifest: Mapping[str, Any],
            warmup_checkpoints: Sequence[Path],
            population_state: Mapping[str, Any],
            teacher_state: Mapping[str, Any],
            peer_reference_preaudit: Mapping[str, Any],
            seed: int,
            arm: str,
            device: str,
        ) -> dict[str, Any]:
            require_training(cfg)
            if arm not in {"large_teacher_single", "large_teacher_diverse"}:
                raise ValueError(arm)
            if not teacher_state.get("superiority_gate_passed"):
                raise RuntimeError("Stage 2 requires a demonstrably stronger larger teacher")
            reference_rounds = peer_reference_preaudit.get("round_reports", [])
            if len(reference_rounds) != cfg.n_rounds:
                raise RuntimeError(
                    "The frozen peer reference must complete every round before teacher arms "
                    "can receive matched target quotas"
                )

            arm_root = cfg.output_dir / f"seed_{seed}" / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            ledger = CostLedger(arm_root / "cost_events.jsonl", arm, seed)
            student_model, student_tokenizer, student_manifest = load_and_validate_model(
                cfg, device="cpu"
            )
            del student_model

            current = []
            for peer, warmup in enumerate(warmup_checkpoints):
                target = arm_root / "r0" / f"peer_{peer}"
                if not target.exists():
                    shutil.copytree(warmup, target)
                current.append(target)

            initial_champion = int(population_state["initial_champion"])
            champion = initial_champion
            history: list[dict[str, Any]] = [dict(population_state["initial_decision"])]
            routing_skill_scores = [
                dict(row) for row in population_state["routing_skill_scores"]
            ]
            baseline_retention = [
                float(row["nll"]) for row in population_state["baseline_retention"]
            ] if population_state["baseline_retention"] else None
            retention_texts = load_retention_texts(cfg)
            current_retention = list(population_state["baseline_retention"])
            initial_diversity = dict(population_state["complementarity"])
            no_promotion_streak = 0
            rounds_completed = 0
            round_reports: list[dict[str, Any]] = []

            for round_index in range(cfg.n_rounds):
                exchange_records = read_jsonl(Path(
                    manifest["partitions"][f"exchange_r{round_index}"]["path"]
                ))
                reference_route = reference_rounds[round_index].get("route") or {}
                matched_targets = reference_route.get("cached_kd_records_by_student")
                if matched_targets is None:
                    raise RuntimeError(
                        f"Peer reference round {round_index} lacks accepted-target quotas"
                    )
                matched_kd_tokens = reference_route.get("cached_kd_tokens_by_student")

                target_started = time.perf_counter()
                frozen_students = load_frozen_population(current, cfg, device)
                student_rollouts = generate_population_rollouts(
                    frozen_students,
                    student_tokenizer,
                    exchange_records,
                    cfg,
                    device,
                    seed=seed * 100 + round_index,
                )
                student_confirmations = generate_population_rollouts(
                    frozen_students,
                    student_tokenizer,
                    exchange_records,
                    cfg,
                    device,
                    seed=seed * 100 + round_index + 50_000_003,
                )
                del frozen_students
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                teacher_model, teacher_tokenizer, teacher_manifest = (
                    load_and_validate_larger_teacher(
                        cfg,
                        student_tokenizer,
                        int(student_manifest["parameter_count"]),
                        device=device,
                    )
                )
                diverse = arm == "large_teacher_diverse"
                teacher_bank, bank_report = generate_larger_teacher_bank(
                    teacher_model,
                    teacher_tokenizer,
                    exchange_records,
                    cfg,
                    device,
                    seed=seed * 100 + round_index + 80_000_009,
                    diverse=diverse,
                )
                if diverse:
                    expected_attempts = cfg.n_peers * 2
                    if bank_report["attempts_per_item"] != expected_attempts:
                        raise RuntimeError(
                            "LT-diverse must match the full peer attempted-output bank: "
                            f"{bank_report['attempts_per_item']} != {expected_attempts}"
                        )
                token_level_allowed = bool(
                    teacher_manifest["tokenizer_compatibility"][
                        "token_level_kd_allowed"
                    ]
                )
                use_onpolicy_kl = bool(
                    arm == "large_teacher_single" and token_level_allowed
                )
                cache_paths = [
                    arm_root / f"r{round_index}" / f"peer_{peer}_targets.pt"
                    for peer in range(cfg.n_peers)
                ]
                if use_onpolicy_kl:
                    target_payloads = build_external_teacher_onpolicy_caches(
                        teacher_model,
                        exchange_records,
                        student_tokenizer,
                        student_rollouts,
                        student_confirmations,
                        teacher_bank,
                        matched_targets,
                        cfg,
                        seed=seed * 100 + round_index,
                        device=device,
                        cache_paths=cache_paths,
                    )
                    auxiliary_records = [
                        payload["eligible_records"] for payload in target_payloads
                    ]
                    auxiliary_tokens = [
                        payload["total_response_tokens"] for payload in target_payloads
                    ]
                    auxiliary_mode = "tokenizer_matched_external_teacher_onpolicy_kl"
                else:
                    target_payloads = build_larger_teacher_sequence_caches(
                        exchange_records,
                        student_tokenizer,
                        student_rollouts,
                        student_confirmations,
                        teacher_bank,
                        matched_targets,
                        cfg,
                        seed=seed * 100 + round_index,
                        cache_paths=cache_paths,
                    )
                    auxiliary_records = [
                        len(payload["rows"]) for payload in target_payloads
                    ]
                    auxiliary_tokens = [
                        payload["selected_response_tokens"]
                        for payload in target_payloads
                    ]
                    auxiliary_mode = (
                        "diverse_verified_sequence_distillation"
                        if diverse else "tokenizer_mismatch_single_sequence_fallback"
                    )
                del teacher_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                route_report = {
                    "policy": arm,
                    "auxiliary_mode": auxiliary_mode,
                    "teacher_bank": bank_report,
                    "matched_peer_target_records_by_student": list(matched_targets),
                    "matched_peer_kd_tokens_by_student": matched_kd_tokens,
                    "accepted_teacher_targets_by_student": auxiliary_records,
                    "accepted_teacher_target_tokens_by_student": auxiliary_tokens,
                    "student_update_schedule_identical": True,
                    "target_record_count_identical": bool(
                        auxiliary_records == list(matched_targets)
                    ),
                    "target_token_counts_reported_not_assumed_equal": True,
                    "larger_teacher_preflight_sha256": teacher_manifest[
                        "preflight_sha256"
                    ],
                }
                ledger.emit(
                    model_role="frozen_students_and_larger_teacher",
                    operation="build_larger_teacher_targets",
                    round=round_index,
                    elapsed_seconds=time.perf_counter() - target_started,
                    attempted_outputs=bank_report["attempted_outputs"],
                    decoded_tokens=bank_report["attempted_decoded_tokens"],
                    accepted_targets=sum(matched_targets),
                    route_report=route_report,
                )

                next_paths = list(current)
                for peer in range(cfg.n_peers):
                    private_records = read_jsonl(Path(
                        manifest["partitions"][
                            f"private_peer_{peer}_r{round_index}"
                        ]["path"]
                    ))
                    train_records = exchange_records + private_records
                    source_path = current[peer]
                    destination = arm_root / f"r{round_index + 1}" / f"peer_{peer}"
                    train_checkpoint(
                        source_path,
                        destination,
                        student_tokenizer,
                        train_records,
                        cfg,
                        seed=seed * 1_000_000 + round_index * 10_000 + peer,
                        peer=peer,
                        arm=arm,
                        device=device,
                        ledger=ledger,
                        onpolicy_cache_path=(
                            cache_paths[peer] if use_onpolicy_kl else None
                        ),
                        sequence_cache_path=(
                            None if use_onpolicy_kl else cache_paths[peer]
                        ),
                        optimizer_state_path=source_path / "optimizer.pt",
                    )
                    next_paths[peer] = destination
                current = next_paths
                rounds_completed += 1

                boundary = round_index + 1
                election_records = read_jsonl(Path(
                    manifest["partitions"][f"election_r{boundary}"]["path"]
                ))
                promotion_records = read_jsonl(Path(
                    manifest["partitions"][f"promotion_r{boundary}"]["path"]
                ))
                election_eval = logged_population_eval(
                    current, student_tokenizer, election_records, cfg, device, ledger,
                    operation="routing_election_eval", boundary=boundary,
                )
                promotion_eval = logged_population_eval(
                    current, student_tokenizer, promotion_records, cfg, device, ledger,
                    operation="promotion_confirmation_eval", boundary=boundary,
                )
                current_retention = logged_retention_eval(
                    current, student_tokenizer, retention_texts, cfg, device, ledger,
                    boundary,
                ) if retention_texts else []
                current_nll = [
                    row["nll"] for row in current_retention
                ] if current_retention else None
                decision = consider_promotion(
                    champion,
                    election_eval,
                    promotion_eval,
                    cfg,
                    seed,
                    boundary,
                    retention_nll=current_nll,
                    baseline_retention_nll=baseline_retention,
                )
                champion = decision["next_champion"]
                history.append({"boundary": boundary, **decision})
                routing_skill_scores = [row["per_skill"] for row in election_eval]
                boundary_diversity = population_complementarity(election_eval, cfg)
                diversity_warning = bool(
                    boundary_diversity["oracle_headroom"]
                    < initial_diversity["oracle_headroom"]
                    * cfg.diversity_retention_fraction_min
                    or min(boundary_diversity["unique_correct_fraction"])
                    < min(initial_diversity["unique_correct_fraction"])
                    * cfg.diversity_retention_fraction_min
                )
                round_reports.append({
                    "round": round_index,
                    "route": route_report,
                    "promotion": decision,
                    "diversity": boundary_diversity,
                    "diversity_warning": diversity_warning,
                })
                no_promotion_streak = (
                    0 if decision["promoted"] else no_promotion_streak + 1
                )
                if (
                    cfg.enable_futility_stop
                    and no_promotion_streak >= cfg.max_consecutive_no_promotions
                ):
                    break

            preaudit = {
                "protocol_version": cfg.protocol_version,
                "seed": seed,
                "arm": arm,
                "initial_champion": initial_champion,
                "selected_peer": champion,
                "history": history,
                "rounds_completed": rounds_completed,
                "round_reports": round_reports,
                "final_routing_skill_scores": routing_skill_scores,
                "checkpoints": [str(path) for path in current],
                "retention": current_retention,
                "baseline_retention_nll": baseline_retention,
                "data_manifest_sha256": manifest["manifest_sha256"],
                "larger_teacher_state_sha256": teacher_state["sha256"],
                "matched_peer_preaudit_sha256": peer_reference_preaudit[
                    "preaudit_sha256"
                ],
                "comparison_scope": "exposure_matched_target_counts_and_student_schedule",
            }
            preaudit["cost"] = summarize_cost_events(ledger.path)
            preaudit["preaudit_sha256"] = sha256_json(preaudit)
            atomic_json(arm_root / "preaudit_result.json", preaudit)
            return preaudit


        def synergy_decomposition(
            pre_evaluations: Sequence[Mapping[str, Any]],
            post_evaluation: Mapping[str, Any],
        ) -> dict[str, Any]:
            pre = np.stack([item_vector(row) for row in pre_evaluations])
            post = item_vector(post_evaluation)
            oracle = pre.max(axis=0)
            novel = float(np.mean((oracle == 0) & (post == 1)))
            lost = float(np.mean((oracle == 1) & (post == 0)))
            excess = float(np.mean(post - oracle))
            if not math.isclose(excess, novel - lost, abs_tol=1e-12):
                raise AssertionError("Rescue-minus-loss identity failed")
            all_wrong = oracle == 0
            return {
                "post_single_minus_pre_oracle": excess,
                "novel_rescue_mass": novel,
                "precoverage_loss_mass": lost,
                "pre_all_wrong_items": int(all_wrong.sum()),
                "conditional_rescue_rate": (
                    float(post[all_wrong].mean()) if all_wrong.any() else None
                ),
                "pre_oracle_accuracy": float(oracle.mean()),
                "post_single_accuracy": float(post.mean()),
            }


        def teacher_external_decomposition(
            pre_evaluations: Sequence[Mapping[str, Any]],
            teacher_evaluation: Mapping[str, Any],
            post_evaluation: Mapping[str, Any],
        ) -> dict[str, Any]:
            pre = np.stack([item_vector(row) for row in pre_evaluations])
            teacher = item_vector(teacher_evaluation)
            post = item_vector(post_evaluation)
            if pre.shape[1] != len(teacher) or len(teacher) != len(post):
                raise ValueError("Pre-peer, teacher, and post item banks must align")
            external_frontier = (pre.max(axis=0) == 0) & (teacher == 0)
            post_external = external_frontier & (post == 1)
            teacher_only = (teacher == 1) & (pre.max(axis=0) == 0)
            peer_only = (teacher == 0) & (pre.max(axis=0) == 1)
            return {
                "attempt_budget": "greedy_pass_at_1",
                "teacher_and_all_pre_wrong_items": int(external_frontier.sum()),
                "teacher_external_valid_rescues": int(post_external.sum()),
                "teacher_external_valid_rescue_mass": float(post_external.mean()),
                "conditional_external_rescue_rate": (
                    float(post[external_frontier].mean())
                    if external_frontier.any() else None
                ),
                "teacher_only_prefrontier_items": int(teacher_only.sum()),
                "prepeer_only_teacher_wrong_items": int(peer_only.sum()),
                "warning": (
                    "Confirmatory novelty requires the separate matched fixed-k bank; "
                    "this pass@1 decomposition is descriptive."
                ),
            }


        def expertise_retention(
            pre_evaluations: Sequence[Mapping[str, Any]],
            post_evaluations: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
        ) -> list[dict[str, Any]]:
            rows = []
            for peer, home_skills in enumerate(cfg.skill_groups):
                before = float(np.mean([
                    pre_evaluations[peer]["per_skill"][skill] for skill in home_skills
                ]))
                after = float(np.mean([
                    post_evaluations[peer]["per_skill"][skill] for skill in home_skills
                ]))
                rows.append({
                    "peer": peer,
                    "home_skills": list(home_skills),
                    "pre": before,
                    "post": after,
                    "change": after - before,
                    "noninferior": after - before >= -cfg.final_noninferiority_margin,
                })
            return rows


        def directed_transfer_audit(
            pre_evaluations: Sequence[Mapping[str, Any]],
            post_evaluations: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
            pre = np.stack([item_vector(row) for row in pre_evaluations])
            post = np.stack([item_vector(row) for row in post_evaluations])
            if pre.shape != post.shape:
                raise ValueError("Pre/post population item tensors must align")
            directed = []
            for donor in range(pre.shape[0]):
                donor_exclusive = (pre[donor] == 1) & (pre.sum(axis=0) == 1)
                for student in range(pre.shape[0]):
                    if student == donor:
                        continue
                    rescue_set = (pre[donor] == 1) & (pre[student] == 0)
                    directed.append({
                        "donor": donor,
                        "student": student,
                        "donor_correct_student_wrong_items": int(rescue_set.sum()),
                        "post_student_accuracy_on_rescue_set": (
                            float(post[student][rescue_set].mean())
                            if rescue_set.any() else None
                        ),
                        "donor_exclusive_items": int(donor_exclusive.sum()),
                        "post_student_accuracy_on_donor_exclusive": (
                            float(post[student][donor_exclusive].mean())
                            if donor_exclusive.any() else None
                        ),
                    })
            donor_retention = []
            for donor in range(pre.shape[0]):
                exclusive = (pre[donor] == 1) & (pre.sum(axis=0) == 1)
                donor_retention.append({
                    "donor": donor,
                    "exclusive_items": int(exclusive.sum()),
                    "post_donor_retention": (
                        float(post[donor][exclusive].mean()) if exclusive.any() else None
                    ),
                })
            return {
                "directed_pairs": directed,
                "donor_exclusive_retention": donor_retention,
            }


        def static_skill_router(
            evaluations: Sequence[Mapping[str, Any]],
            routing_skill_scores: Sequence[Mapping[str, float]],
        ) -> dict[str, Any]:
            mapping = {
                skill: max(
                    range(len(routing_skill_scores)),
                    key=lambda peer: (float(routing_skill_scores[peer][skill]), -peer),
                )
                for skill in routing_skill_scores[0]
            }
            item_rows = evaluations[0]["items"]
            routed = []
            selected_counts = {str(peer): 0 for peer in range(len(evaluations))}
            for index, item in enumerate(item_rows):
                peer = mapping[item["skill"]]
                selected_counts[str(peer)] += 1
                routed.append(evaluations[peer]["items"][index]["correct"])
            return {
                "skill_to_peer": mapping,
                "accuracy": float(np.mean(routed)),
                "selected_counts": selected_counts,
                "model_forwards_per_query": 1,
            }


        def finalize_seed(
            cfg: ExperimentConfig,
            manifest: Mapping[str, Any],
            warmup_checkpoints: Sequence[Path],
            preaudit_results: Mapping[str, Mapping[str, Any]],
            seed: int,
            device: str,
            include_raw_larger_teacher: bool = False,
        ) -> dict[str, Any]:
            # Open the sealed audit only after every requested arm is frozen.
            _, tokenizer, _ = load_and_validate_model(cfg, device="cpu")
            del _
            final_records = read_jsonl(Path(manifest["partitions"]["final_audit"]["path"]))
            shift_records = read_jsonl(Path(manifest["partitions"]["shift_audit"]["path"]))
            composition_records = read_jsonl(Path(
                manifest["partitions"]["composition_audit"]["path"]
            ))
            pre_final = evaluate_population(
                warmup_checkpoints, tokenizer, final_records, cfg, device
            )
            pre_shift = evaluate_population(
                warmup_checkpoints, tokenizer, shift_records, cfg, device
            )
            pre_composition = evaluate_population(
                warmup_checkpoints, tokenizer, composition_records, cfg, device
            )
            pre_final_complementarity = population_complementarity(pre_final, cfg)
            raw_teacher_bundle = None
            if include_raw_larger_teacher:
                student_model, _, student_manifest = load_and_validate_model(
                    cfg, device="cpu"
                )
                del student_model
                student_tokenizer = tokenizer
                teacher_model, teacher_tokenizer, teacher_manifest = (
                    load_and_validate_larger_teacher(
                        cfg,
                        student_tokenizer,
                        int(student_manifest["parameter_count"]),
                        device=device,
                    )
                )
                raw_teacher_bundle = {
                    "manifest": teacher_manifest,
                    "final": evaluate_records(
                        teacher_model, teacher_tokenizer, final_records, cfg, device
                    ),
                    "shift": evaluate_records(
                        teacher_model, teacher_tokenizer, shift_records, cfg, device
                    ),
                    "composition": evaluate_records(
                        teacher_model, teacher_tokenizer, composition_records, cfg, device
                    ),
                    "evaluated_only_after_all_arm_policies_froze": True,
                }
                del teacher_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results = {}
            for arm, preaudit in preaudit_results.items():
                checkpoints = [Path(path) for path in preaudit["checkpoints"]]
                final_eval = evaluate_population(checkpoints, tokenizer, final_records, cfg, device)
                shift_eval = evaluate_population(checkpoints, tokenizer, shift_records, cfg, device)
                composition_eval = evaluate_population(
                    checkpoints, tokenizer, composition_records, cfg, device
                )
                selected = int(preaudit["selected_peer"])
                final_complementarity = population_complementarity(final_eval, cfg)
                relative_retention_changes = []
                baseline_nll = preaudit.get("baseline_retention_nll")
                if baseline_nll and preaudit.get("retention"):
                    relative_retention_changes = [
                        (float(post["nll"]) - float(before)) / float(before)
                        for before, post in zip(baseline_nll, preaudit["retention"])
                    ]
                diversity_retained = bool(
                    final_complementarity["oracle_headroom"]
                    >= pre_final_complementarity["oracle_headroom"]
                    * cfg.diversity_retention_fraction_min
                    and min(final_complementarity["unique_correct_fraction"])
                    >= min(pre_final_complementarity["unique_correct_fraction"])
                    * cfg.diversity_retention_fraction_min
                )
                result = dict(preaudit)
                result.update({
                    "final": final_eval,
                    "shift": shift_eval,
                    "composition": composition_eval,
                    "selected_final_score": float(final_eval[selected]["macro_skill_accuracy"]),
                    "selected_shift_score": float(shift_eval[selected]["macro_skill_accuracy"]),
                    "selected_shift_change": float(
                        shift_eval[selected]["macro_skill_accuracy"]
                        - pre_shift[selected]["macro_skill_accuracy"]
                    ),
                    "selected_composition_score": float(
                        composition_eval[selected]["macro_skill_accuracy"]
                    ),
                    "population_mean_final": float(np.mean([
                        row["macro_skill_accuracy"] for row in final_eval
                    ])),
                    "population_worst_final": float(np.min([
                        row["macro_skill_accuracy"] for row in final_eval
                    ])),
                    "population_mean_shift": float(np.mean([
                        row["macro_skill_accuracy"] for row in shift_eval
                    ])),
                    "pre_final_complementarity": pre_final_complementarity,
                    "final_complementarity": final_complementarity,
                    "diversity_retained": diversity_retained,
                    "general_retention_relative_changes": relative_retention_changes,
                    "general_retention_noninferior": bool(
                        relative_retention_changes
                        and max(relative_retention_changes)
                        <= cfg.retention_nll_relative_margin
                    ),
                    "synergy": synergy_decomposition(pre_final, final_eval[selected]),
                    "composition_synergy": synergy_decomposition(
                        pre_composition, composition_eval[selected]
                    ),
                    "expertise_retention": expertise_retention(pre_final, final_eval, cfg),
                    "directed_transfer": directed_transfer_audit(pre_final, final_eval),
                    "static_skill_router": static_skill_router(
                        final_eval, preaudit["final_routing_skill_scores"]
                    ),
                    "audit_opened_after_all_requested_arms": True,
                })
                if raw_teacher_bundle is not None:
                    result.update({
                        "raw_larger_teacher_final_score": float(
                            raw_teacher_bundle["final"]["macro_skill_accuracy"]
                        ),
                        "selected_minus_raw_larger_teacher": float(
                            final_eval[selected]["macro_skill_accuracy"]
                            - raw_teacher_bundle["final"]["macro_skill_accuracy"]
                        ),
                        "teacher_external_novelty": teacher_external_decomposition(
                            pre_final,
                            raw_teacher_bundle["final"],
                            final_eval[selected],
                        ),
                        "teacher_external_composition_novelty": (
                            teacher_external_decomposition(
                                pre_composition,
                                raw_teacher_bundle["composition"],
                                composition_eval[selected],
                            )
                        ),
                    })
                result["result_sha256"] = sha256_json(result)
                atomic_json(
                    cfg.output_dir / f"seed_{seed}" / arm / "result.json", result
                )
                results[arm] = result
            audit_bundle = {
                "seed": seed,
                "manifest_sha256": manifest["manifest_sha256"],
                "pre_final": pre_final,
                "pre_shift": pre_shift,
                "pre_composition": pre_composition,
                "raw_larger_teacher": raw_teacher_bundle,
                "arms": sorted(results),
            }
            audit_bundle["sha256"] = sha256_json(audit_bundle)
            atomic_json(cfg.output_dir / f"seed_{seed}" / "sealed_audit.json", audit_bundle)
            atomic_json(cfg.output_dir / f"seed_{seed}" / "seed_results.json", results)
            return results


        def run_seed(
            cfg: ExperimentConfig,
            seed: int,
            device: str = "cuda:0",
            arms: Sequence[str] | None = None,
        ) -> dict[str, Any]:
            require_training(cfg)
            manifest_path = cfg.output_dir / "data" / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else materialize_data(cfg)
            audit_manifest(cfg, manifest)
            warmup = prepare_warmup_quartet(cfg, manifest, seed, device)
            population_state = prepare_population_state(cfg, manifest, warmup, seed, device)
            selected_arms = tuple(arms or cfg.primary_screen_arms)
            unknown = set(selected_arms) - set(cfg.arms)
            if unknown:
                raise ValueError(f"Unknown arms: {sorted(unknown)}")
            preaudit_results = {
                arm: run_arm(
                    cfg, manifest, warmup, population_state, seed, arm, device
                )
                for arm in selected_arms
            }
            return finalize_seed(
                cfg, manifest, warmup, preaudit_results, seed, device
            )


        def run_championship_seed(
            cfg: ExperimentConfig,
            seed: int,
            device: str = "cuda:0",
        ) -> dict[str, Any]:
            # Run the frozen Stage-2 matrix; never call on a calibration/pilot seed.
            require_training(cfg)
            if seed in cfg.excluded_pilot_seeds:
                raise ValueError(
                    "Championship outcomes require a fresh confirmatory seed; pilot seeds "
                    "may tune the frozen protocol but cannot enter the final estimate."
                )
            manifest_path = cfg.output_dir / "data" / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text())
                if manifest_path.exists() else materialize_data(cfg)
            )
            audit_manifest(cfg, manifest)
            warmup = prepare_warmup_quartet(cfg, manifest, seed, device)
            population_state = prepare_population_state(
                cfg, manifest, warmup, seed, device
            )
            teacher_state = prepare_championship_teacher_state(
                cfg, manifest, warmup, seed, device
            )
            if (
                teacher_state["larger_teacher_manifest"]["teacher_role"]
                != "primary_capacity_matched_larger_teacher"
            ):
                raise RuntimeError(
                    "An extreme-capacity teacher is secondary-only and cannot replace the "
                    "approximately 1B primary comparator in run_championship_seed."
                )

            # The peer arm runs first only to freeze its per-round exposure quotas. Its
            # endpoint remains sealed; the teacher arms cannot inspect any final outcome.
            preaudit_results: dict[str, dict[str, Any]] = {}
            for arm in (
                "gold_private_equal_cost", "self_snapshot_op", "peer_frr_onpolicy"
            ):
                preaudit_results[arm] = run_arm(
                    cfg, manifest, warmup, population_state, seed, arm, device
                )
            peer_reference = preaudit_results["peer_frr_onpolicy"]
            for arm in ("large_teacher_single", "large_teacher_diverse"):
                preaudit_results[arm] = run_larger_teacher_arm(
                    cfg,
                    manifest,
                    warmup,
                    population_state,
                    teacher_state,
                    peer_reference,
                    seed,
                    arm,
                    device,
                )
            if set(preaudit_results) != set(cfg.championship_arms):
                raise AssertionError("The frozen championship arm matrix is incomplete")
            return finalize_seed(
                cfg,
                manifest,
                warmup,
                preaudit_results,
                seed,
                device,
                include_raw_larger_teacher=True,
            )
        """
    ),
    markdown(
        r"""
        ## Confirmatory aggregation

        The population-training seed is the replicate. The notebook reports paired seed
        differences and does not manufacture narrow confidence intervals from four members or
        thousands of audit items. `peer_frr_onpolicy - large_teacher_diverse` is the one
        confirmatory superiority contrast. Ordinary/self, single-trace teacher, valid novelty,
        and raw-teacher crossing form an explicit success hierarchy rather than a menu from
        which to select the best p-value. Excluded pilot seeds may set `N`; they never enter the
        estimate.
        """
    ),
    code(
        r"""
        def bootstrap_seed_difference(
            treatment: Sequence[float],
            control: Sequence[float],
            resamples: int = 100_000,
            seed: int = 20260725,
        ) -> dict[str, Any]:
            a = np.asarray(treatment, dtype=np.float64)
            b = np.asarray(control, dtype=np.float64)
            if a.shape != b.shape or len(a) < 2:
                raise ValueError("Need at least two paired seed results.")
            delta = a - b
            rng = np.random.default_rng(seed)
            picks = rng.integers(0, len(delta), size=(resamples, len(delta)))
            means = delta[picks].mean(axis=1)
            return {
                "n_pairs": len(delta),
                "mean_difference": float(delta.mean()),
                "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
                "per_seed": delta.tolist(),
            }


        def paired_seed_analysis(
            treatment: Sequence[float],
            control: Sequence[float],
        ) -> dict[str, Any]:
            a = np.asarray(treatment, dtype=np.float64)
            b = np.asarray(control, dtype=np.float64)
            if a.shape != b.shape or len(a) < 2:
                raise ValueError("Need at least two paired population results")
            delta = a - b
            n = len(delta)
            sd = float(delta.std(ddof=1))
            se = sd / math.sqrt(n)
            try:
                from scipy.stats import t as student_t
                critical_two = float(student_t.ppf(0.975, n - 1))
                critical_one = float(student_t.ppf(0.95, n - 1))
                critical_source = "scipy_student_t"
            except ImportError:
                critical_two, critical_one = 1.959963984540054, 1.6448536269514722
                critical_source = "normal_fallback_install_scipy_for_final"
            observed = abs(float(delta.mean()))
            if n <= 20:
                signed_means = []
                for mask in range(1 << n):
                    signs = np.asarray([
                        1.0 if (mask >> index) & 1 else -1.0 for index in range(n)
                    ])
                    signed_means.append(abs(float(np.mean(delta * signs))))
                sign_p = float(np.mean(np.asarray(signed_means) >= observed - 1e-15))
                sign_mode = "exact"
            else:
                rng = np.random.default_rng(20260725)
                signs = rng.choice([-1.0, 1.0], size=(1_000_000, n))
                sign_p = float(np.mean(np.abs((signs * delta).mean(axis=1)) >= observed))
                sign_mode = "monte_carlo"
            mean = float(delta.mean())
            return {
                "n_pairs": n,
                "mean_difference": mean,
                "median_difference": float(np.median(delta)),
                "sd_difference": sd,
                "ci95": [mean - critical_two * se, mean + critical_two * se],
                "one_sided_lower95": mean - critical_one * se,
                "exact_sign_flip_two_sided_p": sign_p,
                "sign_flip_mode": sign_mode,
                "critical_source": critical_source,
                "per_seed": delta.tolist(),
                "percentile_bootstrap_descriptive": bootstrap_seed_difference(a, b),
            }


        def required_paired_seeds(pilot_sd: float, meaningful_difference: float = 0.02) -> int:
            # Normal-approximation planning only; freeze N before confirmatory results.
            if pilot_sd <= 0:
                return 4
            z_alpha, z_power = 1.959963984540054, 0.8416212335729143
            n = math.ceil(((z_alpha + z_power) * pilot_sd / meaningful_difference) ** 2)
            return max(4, min(16, n))


        def load_completed_results(cfg: ExperimentConfig) -> dict[int, dict[str, Any]]:
            collected = {}
            for seed in cfg.confirmatory_seeds:
                path = cfg.output_dir / f"seed_{seed}" / "seed_results.json"
                if path.exists():
                    collected[seed] = json.loads(path.read_text())
            return collected


        def summarize_confirmatory(cfg: ExperimentConfig) -> dict[str, Any]:
            results = load_completed_results(cfg)
            expected_seeds = set(cfg.confirmatory_seeds)
            completed_seeds = set(results)
            missing_seeds = expected_seeds - completed_seeds
            if missing_seeds:
                raise RuntimeError(
                    "Missing configured confirmatory seeds "
                    f"{sorted(missing_seeds)}. Do not summarize only successful "
                    "seed jobs; set OLMO400M_B200_GPUS to the actual planned seed "
                    "count before launch, or rerun failed seeds."
                )
            if len(results) < 2:
                raise RuntimeError(
                    "At least two completed paired seeds are required for the final claim."
                )
            seeds = sorted(results)
            required = set(cfg.championship_arms)
            for seed in seeds:
                missing = required - set(results[seed])
                if missing:
                    raise RuntimeError(f"Seed {seed} is missing primary arms {sorted(missing)}")
            proposed = [
                results[s]["peer_frr_onpolicy"]["selected_final_score"] for s in seeds
            ]
            comparisons = {}
            for baseline in (
                "gold_private_equal_cost",
                "self_snapshot_op",
                "large_teacher_single",
                "large_teacher_diverse",
                "frr_goldprefix",
                "global_champion_goldprefix",
                "unfiltered_mutual_mean",
            ):
                if all(baseline in results[s] for s in seeds):
                    control = [results[s][baseline]["selected_final_score"] for s in seeds]
                    comparisons[baseline] = paired_seed_analysis(proposed, control)

            synergy_absolute = [
                results[s]["peer_frr_onpolicy"]["synergy"]["post_single_minus_pre_oracle"]
                for s in seeds
            ]
            synergy_absolute_analysis = paired_seed_analysis(
                synergy_absolute, [0.0] * len(synergy_absolute)
            )
            synergy_comparisons = {}
            for baseline in (
                "gold_private_equal_cost", "self_snapshot_op",
                "large_teacher_single", "large_teacher_diverse",
            ):
                control = [
                    results[s][baseline]["synergy"]["post_single_minus_pre_oracle"]
                    for s in seeds
                ]
                synergy_comparisons[baseline] = paired_seed_analysis(
                    synergy_absolute, control
                )

            expertise = {}
            for peer in range(cfg.n_peers):
                changes = [
                    results[s]["peer_frr_onpolicy"]["expertise_retention"][peer]["change"]
                    for s in seeds
                ]
                zeros = [0.0] * len(changes)
                expertise[str(peer)] = paired_seed_analysis(changes, zeros)

            shift_changes = [
                results[s]["peer_frr_onpolicy"]["selected_shift_change"] for s in seeds
            ]
            shift_noninferiority = paired_seed_analysis(
                shift_changes, [0.0] * len(shift_changes)
            )
            retention_buffers = []
            diversity_flags = []
            boundary_diversity_flags = []
            for seed in seeds:
                treatment = results[seed]["peer_frr_onpolicy"]
                changes = treatment["general_retention_relative_changes"]
                if not changes:
                    raise RuntimeError(
                        f"Seed {seed} lacks held-out general-retention measurements"
                    )
                retention_buffers.append(
                    cfg.retention_nll_relative_margin - max(changes)
                )
                diversity_flags.append(bool(treatment["diversity_retained"]))
                boundary_diversity_flags.append(not any(
                    row["diversity_warning"] for row in treatment["round_reports"]
                ))
            general_retention_safety = paired_seed_analysis(
                retention_buffers, [0.0] * len(retention_buffers)
            )
            pass1_external_novelty = [
                results[s]["peer_frr_onpolicy"]["teacher_external_novelty"][
                    "teacher_external_valid_rescue_mass"
                ]
                for s in seeds
            ]
            pass1_external_novelty_absolute = paired_seed_analysis(
                pass1_external_novelty, [0.0] * len(pass1_external_novelty)
            )
            pass1_external_novelty_comparisons = {}
            for baseline in (
                "gold_private_equal_cost", "large_teacher_single",
                "large_teacher_diverse",
            ):
                control = [
                    results[s][baseline]["teacher_external_novelty"][
                        "teacher_external_valid_rescue_mass"
                    ]
                    for s in seeds
                ]
                pass1_external_novelty_comparisons[baseline] = paired_seed_analysis(
                    pass1_external_novelty, control
                )

            # A greedy miss is not evidence of absence. Level 4 is gated only by the
            # separate fixed-k bank written after identical sampling/audit policies run.
            fixed_k_paths = {
                seed: cfg.output_dir / f"seed_{seed}" / "fixed_k_novelty.json"
                for seed in seeds
            }
            fixed_k_complete = all(path.exists() for path in fixed_k_paths.values())
            fixed_k_external_novelty_absolute = None
            fixed_k_external_novelty_comparisons = None
            valid_external_novelty_pass = False
            if fixed_k_complete:
                fixed_k_rows = {
                    seed: json.loads(path.read_text())
                    for seed, path in fixed_k_paths.items()
                }
                fixed_peer = [
                    fixed_k_rows[seed]["arms"]["peer_frr_onpolicy"][
                        "teacher_external_valid_novelty_rate"
                    ]
                    for seed in seeds
                ]
                fixed_k_external_novelty_absolute = paired_seed_analysis(
                    fixed_peer, [0.0] * len(fixed_peer)
                )
                fixed_k_external_novelty_comparisons = {}
                for baseline in (
                    "gold_private_equal_cost", "large_teacher_single",
                    "large_teacher_diverse",
                ):
                    control = [
                        fixed_k_rows[seed]["arms"][baseline][
                            "teacher_external_valid_novelty_rate"
                        ]
                        for seed in seeds
                    ]
                    fixed_k_external_novelty_comparisons[baseline] = (
                        paired_seed_analysis(fixed_peer, control)
                    )
                valid_external_novelty_pass = bool(
                    fixed_k_external_novelty_absolute["one_sided_lower95"] > 0
                    and fixed_k_external_novelty_comparisons[
                        "large_teacher_diverse"
                    ]["one_sided_lower95"] > 0
                    and fixed_k_external_novelty_comparisons[
                        "gold_private_equal_cost"
                    ]["one_sided_lower95"] > 0
                )

            raw_teacher_scores = [
                results[s]["peer_frr_onpolicy"]["raw_larger_teacher_final_score"]
                for s in seeds
            ]
            raw_teacher_moonshot = paired_seed_analysis(
                proposed, raw_teacher_scores
            )

            safety_pass = bool(
                all(
                    row["one_sided_lower95"] > -cfg.final_noninferiority_margin
                    for row in expertise.values()
                )
                and shift_noninferiority["one_sided_lower95"]
                > -cfg.final_noninferiority_margin
                and general_retention_safety["one_sided_lower95"] > 0
                and min(retention_buffers) >= 0
                and all(diversity_flags)
                and all(boundary_diversity_flags)
            )
            mechanism_pass = bool(
                comparisons["gold_private_equal_cost"]["one_sided_lower95"] > 0
                and comparisons["self_snapshot_op"]["one_sided_lower95"] > 0
                and safety_pass
            )
            beats_large_teacher_single = bool(
                comparisons["large_teacher_single"]["one_sided_lower95"]
                > cfg.pilot_effect_min
                and safety_pass
            )
            beats_large_teacher_diverse = bool(
                comparisons["large_teacher_diverse"]["one_sided_lower95"]
                > cfg.pilot_effect_min
                and safety_pass
            )
            moonshot_pass = bool(
                raw_teacher_moonshot["one_sided_lower95"] > 0
            )
            success_level = 0
            for level, passed in enumerate((
                mechanism_pass,
                beats_large_teacher_single,
                beats_large_teacher_diverse,
                valid_external_novelty_pass,
                moonshot_pass,
            ), start=1):
                if passed and level == success_level + 1:
                    success_level = level
                else:
                    break
            cost_by_seed_arm = {
                str(seed): {
                    arm: results[seed][arm]["cost"] for arm in sorted(required)
                }
                for seed in seeds
            }
            summary = {
                "seeds": seeds,
                "primary_estimand": (
                    "selected peer_frr_onpolicy 400M minus symmetrically selected "
                    "large_teacher_diverse 400M"
                ),
                "practical_superiority_margin": cfg.pilot_effect_min,
                "comparisons": comparisons,
                "synergy_absolute_per_seed": synergy_absolute,
                "synergy_absolute_analysis": synergy_absolute_analysis,
                "synergy_comparisons": synergy_comparisons,
                "expertise_retention": expertise,
                "shift_noninferiority": shift_noninferiority,
                "general_retention_margin_buffers": retention_buffers,
                "general_retention_safety": general_retention_safety,
                "final_diversity_retained_by_seed": diversity_flags,
                "no_boundary_diversity_warnings_by_seed": boundary_diversity_flags,
                "pass1_external_novelty_descriptive": {
                    "absolute": pass1_external_novelty_absolute,
                    "comparisons": pass1_external_novelty_comparisons,
                    "cannot_satisfy_success_level_4": True,
                },
                "fixed_k_novelty_audit_complete": fixed_k_complete,
                "fixed_k_external_novelty_absolute": (
                    fixed_k_external_novelty_absolute
                ),
                "fixed_k_external_novelty_comparisons": (
                    fixed_k_external_novelty_comparisons
                ),
                "raw_larger_teacher_moonshot": raw_teacher_moonshot,
                "safety_gate_passed": safety_pass,
                "success_hierarchy": {
                    "1_beats_ordinary_and_self": mechanism_pass,
                    "2_beats_large_teacher_single": beats_large_teacher_single,
                    "3_beats_large_teacher_diverse": beats_large_teacher_diverse,
                    "4_teacher_external_valid_novelty": valid_external_novelty_pass,
                    "5_beats_raw_larger_teacher": moonshot_pass,
                    "highest_contiguous_level": success_level,
                },
                "peer_vs_larger_teacher_primary_gate_passed": bool(
                    mechanism_pass
                    and beats_large_teacher_single
                    and beats_large_teacher_diverse
                ),
                "all_in_cost_ledger_by_seed_and_arm": cost_by_seed_arm,
                "all_in_compute_note": (
                    "Do not infer equal cost from exposure matching. Plot quality against "
                    "the complete measured ledger and any preregistered productive top-ups."
                ),
            }
            atomic_json(cfg.output_dir / "confirmatory_summary.json", summary)
            return summary
        """
    ),
    markdown(
        r"""
        ## Fixed-budget valid novelty and strategy audit

        “Creative” is not a synonym for lexically different. The confirmatory order is:

        1. objective correctness, executable tests, or preregistered usefulness;
        2. absence from equal-budget pre-peer and larger-teacher output banks;
        3. structural strategy identity rather than embedding distance;
        4. diversity across the remaining valid outputs.

        Every selected post-training 400M receives `k` samples per item. Each pre peer also
        receives `k`; for the stringent external-novelty bank, the raw larger teacher receives
        `4k` so a single teacher is not compared with a four-model population lottery. Report
        selected-model pass@1/pass@k separately from population pass@4k. Decoding temperature,
        top-p, seeds, token cap, prompt history, tool access, verifier, and rejection policy are
        frozen. Strategy labels come from executable/AST features or blinded adjudication;
        semantic distance alone never proves a new method.

        Open-ended ideation/writing is secondary. Strip system identity, randomize and swap
        order, length-match, use a judge family absent from training, calibrate it against
        preregistered blinded humans, and report quality, originality, and diversity separately.
        """
    ),
    code(
        r"""
        def sample_fixed_k_bank(
            model: Any,
            tokenizer: Any,
            records: Sequence[Mapping[str, Any]],
            cfg: ExperimentConfig,
            device: str,
            role: str,
            samples_per_item: int,
            seed: int,
        ) -> list[dict[str, Any]]:
            rows = []
            for sample_index in range(samples_per_item):
                rollout = generate_rollouts_for_model(
                    model,
                    tokenizer,
                    records,
                    cfg,
                    device,
                    seed=seed + sample_index * 1_000_003,
                )
                for record in records:
                    item = rollout[record["artifact_id"]]
                    rows.append({
                        "item_id": record["artifact_id"],
                        "skill": record["skill"],
                        "source_role": role,
                        "sample_index": sample_index,
                        "output": item["output"],
                        "output_sha256": item["output_sha256"],
                        "valid": bool(item["correct"]),
                        # Fill only through a frozen structural extractor or blinded audit.
                        "strategy_id": None,
                    })
            return rows


        def strategy_entropy(strategy_ids: Sequence[str]) -> float:
            if not strategy_ids:
                return 0.0
            counts: dict[str, int] = {}
            for strategy_id in strategy_ids:
                counts[strategy_id] = counts.get(strategy_id, 0) + 1
            probabilities = np.asarray(list(counts.values()), dtype=np.float64)
            probabilities /= probabilities.sum()
            return float(-(probabilities * np.log(probabilities)).sum())


        def fixed_k_valid_novelty_summary(
            rows: Sequence[Mapping[str, Any]],
            post_role: str,
            k: int,
            n_peers: int = 4,
        ) -> dict[str, Any]:
            expected = {
                **{f"pre_peer_{index}": k for index in range(n_peers)},
                "raw_larger_teacher": n_peers * k,
                post_role: k,
            }
            grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(
                    (str(row["item_id"]), str(row["source_role"])), []
                ).append(row)
            item_ids = sorted({str(row["item_id"]) for row in rows})
            for item_id in item_ids:
                for role, count in expected.items():
                    observed = grouped.get((item_id, role), [])
                    if len(observed) != count:
                        raise ValueError(
                            f"{item_id}/{role}: expected {count} frozen samples, "
                            f"observed {len(observed)}"
                        )
                    indices = sorted(int(row["sample_index"]) for row in observed)
                    if indices != list(range(count)):
                        raise ValueError(f"{item_id}/{role}: sample indices are not frozen")

            external_items = []
            post_valid_items = []
            post_strategy_ids: list[str] = []
            novel_strategy_ids: list[str] = []
            for item_id in item_ids:
                pre_rows = [
                    row
                    for peer in range(n_peers)
                    for row in grouped[(item_id, f"pre_peer_{peer}")]
                ]
                teacher_rows = grouped[(item_id, "raw_larger_teacher")]
                post_rows = grouped[(item_id, post_role)]
                post_valid = [row for row in post_rows if bool(row["valid"])]
                if post_valid:
                    post_valid_items.append(item_id)
                if (
                    post_valid
                    and not any(bool(row["valid"]) for row in pre_rows)
                    and not any(bool(row["valid"]) for row in teacher_rows)
                ):
                    external_items.append(item_id)
                prior_strategies = {
                    str(row["strategy_id"])
                    for row in pre_rows + teacher_rows
                    if bool(row["valid"]) and row.get("strategy_id") not in (None, "")
                }
                for row in post_valid:
                    if row.get("strategy_id") in (None, ""):
                        continue
                    strategy_id = str(row["strategy_id"])
                    post_strategy_ids.append(strategy_id)
                    if strategy_id not in prior_strategies:
                        novel_strategy_ids.append(strategy_id)
            return {
                "post_role": post_role,
                "items": len(item_ids),
                "k_selected_model": k,
                "k_raw_teacher_external_bank": n_peers * k,
                "pass_at_k_items": len(post_valid_items),
                "pass_at_k_rate": len(post_valid_items) / len(item_ids),
                "teacher_and_prepeer_external_valid_items": external_items,
                "teacher_external_valid_novelty_rate": len(external_items) / len(item_ids),
                "valid_strategy_count": len(set(post_strategy_ids)),
                "valid_strategy_entropy": strategy_entropy(post_strategy_ids),
                "valid_strategies_absent_from_all_pre_and_teacher_banks": len(
                    set(novel_strategy_ids)
                ),
                "quality_novelty_diversity_kept_separate": True,
            }


        def save_fixed_k_seed_audit(
            cfg: ExperimentConfig,
            seed: int,
            rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
        ) -> dict[str, Any]:
            required = set(cfg.championship_arms)
            missing = required - set(rows_by_arm)
            if missing:
                raise ValueError(
                    f"Fixed-k audit is missing championship arms: {sorted(missing)}"
                )
            summaries = {
                arm: fixed_k_valid_novelty_summary(
                    rows_by_arm[arm],
                    post_role=f"post_{arm}",
                    k=cfg.fixed_k_samples,
                    n_peers=cfg.n_peers,
                )
                for arm in sorted(required)
            }
            payload = {
                "seed": seed,
                "k": cfg.fixed_k_samples,
                "raw_teacher_external_bank_k": cfg.n_peers * cfg.fixed_k_samples,
                "arms": summaries,
                "identical_sampling_and_verifier_policy_attested": True,
            }
            payload["sha256"] = sha256_json(payload)
            atomic_json(
                cfg.output_dir / f"seed_{seed}" / "fixed_k_novelty.json",
                payload,
            )
            return payload
        """
    ),
    markdown(
        r"""
        ## Safe dry-run preflight

        This cell is the only cell intended to run in the current workspace. It creates a tiny
        deterministic version of every partition, verifies exact answers, checks split/signature
        separation, and writes a manifest. It does not import model weights or optimize anything.
        """
    ),
    code(
        r"""
        if CFG.allow_training:
            print("SKIP_DRY_RUN: training is unlocked; operator cells control the real run.")
        else:
            DRY_CFG = smoke_config(CFG)
            dry_manifest = materialize_data(DRY_CFG)
            dry_audit = audit_manifest(DRY_CFG, dry_manifest)
            print(json.dumps(dry_audit, indent=2))
        """
    ),
    markdown(
        r"""
        ## 10-hour B200 compressed championship profile

        Set `OLMO400M_BUDGET_PROFILE=b200_10h` when the allocation is capped at two to four
        B200 GPUs and roughly ten wall-clock hours. This profile preserves the core
        championship question and all five required arms, but shrinks the run to two peer
        rounds, smaller training/election/audit partitions, a 250k-token retention check,
        optional fixed-`k=8` novelty scaffolding, and one fresh confirmatory seed per requested B200 GPU:
        two B200s run seeds `(13, 29)`, three run `(13, 29, 47)`, and four run
        `(13, 29, 47, 71)`.

        This is the strongest budgeted version, not the full confirmatory study. It can
        support a practical go/no-go or directional claim if the peer arm beats
        `large_teacher_diverse` consistently across seeds, but it should report the seed
        count plainly and avoid presenting a four-seed run as the original 16-seed
        confirmatory protocol.

        Minimal AWS environment additions:

        ```bash
        export OLMO400M_BUDGET_PROFILE=b200_10h
        export OLMO400M_B200_GPUS=4  # set to 2, 3, or 4
        ```

        Under a hard wall-clock cap, do not run all seeds serially in one notebook kernel.
        Materialize the full data manifest once, then launch one controlled notebook/process per
        B200 with a single confirmatory seed:

        ```bash
        export OLMO400M_RUN_MODE=manifest_only
        # execute the notebook once to create and hash the shared data manifest

        export OLMO400M_RUN_MODE=championship_seed
        export OLMO400M_SEED=13
        export CUDA_VISIBLE_DEVICES=0
        export OLMO400M_DEVICE=cuda:0
        # execute one notebook/process for this seed; repeat on other B200s with seeds 29/47/71

        export OLMO400M_RUN_MODE=summarize
        # execute once after all seed directories are complete
        ```

        `OLMO400M_RUN_MODE=championship_all` exists for convenience, but it runs the seeds
        serially in the current kernel and is not the intended path for the 10-hour B200 cap.
        The fixed-`k` novelty audit is not part of the guaranteed 10-hour core; if it is not
        run separately, the summary will correctly leave success level 4 incomplete and treat
        the primary claim as the level-3 peer-vs-larger-teacher result.
        """
    ),
    markdown(
        r"""
        ## Operator-only execution cells

        Do not unlock these cells until the exact OLMo-compatible 400M checkpoint, the pinned
        larger teacher, both tokenizers, local retention corpus, software lockfile, and final
        protocol manifest are frozen.
        In the full profile, seeds 0–3 are excluded calibration/pilot populations; in the
        compressed B200 profile, seed 0 is the excluded pilot and fresh seeds 13/29/47/71 are
        reserved for the directional championship. After pilots fix routing,
        on-policy masking, replay quota, cache width, teacher sampling, matched-target quotas,
        equal-cost productive top-ups, round length, and confirmatory `N`, cryptographically
        lock the protocol before running fresh seeds.
        This notebook contains no scheduler or FarmShare submission code.
        """
    ),
    code(
        r"""
        # Model preflight only; still requires the explicit training unlock because it loads
        # a large local artifact. No optimization happens in this cell.
        if CFG.allow_training:
            preflight_model, preflight_tokenizer, model_manifest = load_and_validate_model(CFG, device="cpu")
            atomic_json(CFG.output_dir / "model_manifest.json", model_manifest)
            del preflight_model
            print(json.dumps(model_manifest, indent=2))
            if CFG.larger_teacher_model_path:
                larger_model, larger_tokenizer, larger_manifest = load_and_validate_larger_teacher(
                    CFG,
                    preflight_tokenizer,
                    int(model_manifest["parameter_count"]),
                    device="cpu",
                )
                atomic_json(CFG.output_dir / "larger_teacher_manifest.json", larger_manifest)
                del larger_model
                print(json.dumps(larger_manifest, indent=2))
        else:
            print("LOCKED: set the explicit environment token in the intended compute environment.")
        """
    ),
    code(
        r"""
        # Explicit optimization entry point. This notebook never calls it while locked.
        RUN_MODE_ENV = os.environ.get("OLMO400M_RUN_MODE", "").strip().lower()
        BUDGET_PROFILE = os.environ.get("OLMO400M_BUDGET_PROFILE", "").strip().lower()
        RUN_MODE = RUN_MODE_ENV or "calibration"
        RUN_DEVICE = os.environ.get("OLMO400M_DEVICE", "cuda:0")


        def requested_championship_seed(cfg: ExperimentConfig) -> int:
            if os.environ.get("OLMO400M_SEED", "").strip():
                return int(os.environ["OLMO400M_SEED"])
            if os.environ.get("OLMO400M_GPU_SLOT", "").strip():
                slot = int(os.environ["OLMO400M_GPU_SLOT"])
                seeds = list(cfg.confirmatory_seeds)
                if slot < 0 or slot >= len(seeds):
                    raise ValueError(
                        f"OLMO400M_GPU_SLOT={slot} is outside available seeds {seeds}"
                    )
                return int(seeds[slot])
            raise RuntimeError(
                "Set OLMO400M_SEED, or set OLMO400M_GPU_SLOT to select from "
                f"{list(cfg.confirmatory_seeds)}."
            )


        if CFG.allow_training:
            if BUDGET_PROFILE == "b200_10h" and not RUN_MODE_ENV:
                raise RuntimeError(
                    "b200_10h requires an explicit OLMO400M_RUN_MODE so the 10-hour "
                    "allocation is not accidentally spent on calibration. Use "
                    "manifest_only, championship_seed, championship_all, or summarize."
                )
            manifest_path = CFG.output_dir / "data" / "manifest.json"
            if RUN_MODE == "manifest_only":
                full_manifest = materialize_data(CFG)
            elif manifest_path.exists():
                full_manifest = json.loads(manifest_path.read_text())
            elif BUDGET_PROFILE == "b200_10h" and RUN_MODE == "championship_seed":
                raise RuntimeError(
                    "Run OLMO400M_RUN_MODE=manifest_only once before launching parallel "
                    "b200_10h championship_seed jobs. This avoids concurrent writes to "
                    "the shared data manifest."
                )
            else:
                full_manifest = materialize_data(CFG)
            print(json.dumps(audit_manifest(CFG, full_manifest), indent=2))
            print(json.dumps({
                "run_mode": RUN_MODE,
                "device": RUN_DEVICE,
                "confirmatory_seeds": list(CFG.confirmatory_seeds),
            }, indent=2))
            if RUN_MODE == "manifest_only":
                print("MANIFEST_ONLY complete. Launch championship_seed jobs next.")
            elif RUN_MODE == "calibration":
                # Calibration only. This runs the three primary screen arms by default.
                # Validate dense-vs-top-k KL, profile all-in cost, freeze N and the manifest,
                # then run fresh confirmatory seeds in separate invocations.
                calibration_results = run_seed(
                    CFG, CFG.calibration_seed, device=RUN_DEVICE,
                    arms=CFG.primary_screen_arms,
                )
                print(json.dumps({
                    "calibration_seed": CFG.calibration_seed,
                    "arms": list(CFG.primary_screen_arms),
                }, indent=2))
            elif RUN_MODE == "championship_seed":
                seed = requested_championship_seed(CFG)
                championship_results = run_championship_seed(
                    CFG, seed, device=RUN_DEVICE
                )
                print(json.dumps({
                    "championship_seed_complete": seed,
                    "arms": sorted(championship_results),
                    "seed_results": str(CFG.output_dir / f"seed_{seed}" / "seed_results.json"),
                }, indent=2))
            elif RUN_MODE == "championship_all":
                championship_results = {}
                for seed in CFG.confirmatory_seeds:
                    championship_results[int(seed)] = run_championship_seed(
                        CFG, int(seed), device=RUN_DEVICE
                    )
                print(json.dumps({
                    "championship_all_complete": list(map(int, CFG.confirmatory_seeds)),
                    "warning": (
                        "This ran serially in one kernel; use championship_seed per GPU "
                        "for the 10-hour B200 profile."
                    ),
                }, indent=2))
            elif RUN_MODE == "summarize":
                print(json.dumps(summarize_confirmatory(CFG), indent=2))
            else:
                raise ValueError(
                    "Unknown OLMO400M_RUN_MODE. Use manifest_only, calibration, "
                    "championship_seed, championship_all, or summarize."
                )
        else:
            print("TRAINING DISABLED. No model update or FarmShare submission was performed.")
        """
    ),
    markdown(
        r"""
        ## Required pre-confirmatory checklist

        - Pin the internal OLMo-400M config, tokenizer, and checkpoint hashes. Pin a
          checkpoint-stage-matched approximately 1B OLMo-compatible primary teacher; record
          its hashes and parameter ratio. Treat 7B as secondary-only after a strength and
          400M-absorption pilot.
        - Require the larger teacher to beat the election-selected best 400M peer by the
          practical margin on both calibration shards, with a positive confirmation lower
          bound. Do not inspect the final audit to find a favorable teacher.
        - Freeze the tokenizer decision. Exact token-to-ID and probe equality permits
          token-level KL; any mismatch restricts the cross-tokenizer comparison to verified
          sequence distillation.
        - Create a locked Python/CUDA environment and record package/device versions.
        - Supply a local, held-out general-language retention corpus; absence blocks a strong
          “improved without narrowing” claim.
        - Run `validate_topk_cache` against full-vocabulary KL and require its configured mean
          and p99 error limits before any confirmatory cache is built.
        - In the full protocol, use excluded seeds 0–3 to detect floor/ceiling, require
          nondegenerate routed rescue, profile every peer/teacher/gold cost, estimate
          paired-population variance, and freeze the confirmatory replication count. In the
          `b200_10h` profile, report that this pilot screen is compressed to seed 0 or omitted
          only by explicit team-lead decision. Never adapt after confirmatory outcomes.
        - Tune loss mixing/temperature only on excluded data with the same trial budget and
          stopping access for peer and teacher policies. Pure teacher imitation is not assumed
          optimal; freeze one common or symmetrically selected setting before confirmation.
        - For every confirmatory seed, fork the same four checkpoint bytes into all five
          championship arms. Use identical student seeds, updates, hard examples, tuning
          opportunities, independent selection split, and untouched final test.
        - Match `large_teacher_diverse` to the full peer attempted-output count and the peer
          arm's accepted target count. Publish auxiliary token counts rather than pretending
          that peer answer-only KL and teacher rationale CE have identical exposure; provide
          both the exposure-matched analysis and all-in-compute frontier.
        - Build a sealed final suite containing ordinary specialties, predeclared all-wrong
          difficulty, and at least one held-out two-specialty composition family. Do not expose
          pre-checkpoint errors either: pre and post checkpoints are evaluated only after every
          requested arm is frozen.
        - Keep private specialty replay out of peer KD. Require every specialist's absolute
          retention bound, not only the population average.
        - Build fixed-`k` output banks with identical decoding/tool/verifier policy. Give the
          raw teacher `4k` attempts when defining failure against four pre peers. Report
          pass@1, pass@k, valid strategy coverage, and teacher-external novelty; do not infer
          creativity from embedding distance or one greedy teacher miss.
        - Report every arm and every student, failed route, attempted/accepted/rejected sample,
          processed/KD/sequence token, device-second, selection decision, rescue-minus-loss
          decomposition, raw-teacher score, and oracle/unique-correct collapse.

        The notebook is prepared, but a scientifically complete run remains blocked on the
        exact internal student and larger-teacher artifacts plus external retention and sealed
        audit handoffs. Those are inputs, not implementation choices the notebook should guess.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUT)
