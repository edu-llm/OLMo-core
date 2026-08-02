# General instruction-following — base Instruct vs SFT (forgetting probe)

Does the SFT model still follow **general, non-pedagogy instructions** as well as the base
Instruct model? (Co-training was meant to prevent regression here.) This is a *prior-task*
forgetting axis.

**Primary metric: IFEval (deterministic).** We follow RL's Razor (Shenfeld et al. 2025), which
measures instruction-following with **IFEval** (Zhou et al. 2023, arXiv:2311.07911) via
rule-based verification — no LLM judge, no subagents. Because it's deterministic it can be run
and logged to W&B at every checkpoint. This replaces the earlier MT-Bench-style LLM-judge flow.

## Files

| File | What it is |
|------|------------|
| `ifeval_registry.py` | Deterministic checkers for 24 verifiable instruction types + strict/loose scoring. |
| `build_ifeval_set.py` | Builds `ifeval_prompts.jsonl` (asserts every instruction has a checker). |
| `ifeval_prompts.jsonl` | 34 prompts / 37 verifiable instructions, each with `instruction_ids` + `kwargs`. |
| `grade_ifeval.py` | Scores `generate_eval.py` results → prompt/inst-level strict & loose acc, base vs sft. |
| `general_prompts.jsonl`, `judge_build.py`, `judge_aggregate.py` | **Deprecated** MT-Bench LLM-judge flow (needs subagents). Kept for reference only. |

## Workflow (deterministic, per checkpoint)

```bash
# 1. generate base-vs-checkpoint outputs on the IFEval prompts
python ../generate_eval.py --prompts ifeval_prompts.jsonl \
    --adapter ../../out/<run>/checkpoint-16 --out results_c16.jsonl
# 2. score — prints the four IFEval numbers, writes ifeval_graded_results_c16.json
python grade_ifeval.py results_c16.jsonl
```

`grade_ifeval.py` reports, for base and sft:

- `prompt_level_strict` / `prompt_level_loose` — every instruction in the prompt satisfied.
- `inst_level_strict` / `inst_level_loose` — fraction of individual instructions satisfied.

Loose accepts a response if any standard transform (strip markdown `*`, drop first/last line)
passes, exactly as in the reference implementation. Use `sft.prompt_level_loose` as the
retention number in `master_summary.json` (or `--acc_metric inst_level_loose` for finer grain).

## What "good" looks like

SFT should be **at parity** with base — `sft` IFEval accuracy ≈ `base`. A materially lower `sft`
score means the fine-tune degraded general instruction-following (forgetting); at/above base
means co-training preserved it.

## Rebuild / extend the set

Add rows to `SET` in `build_ifeval_set.py` (only using `instruction_id`s present in
`ifeval_registry.REGISTRY`) and re-run `python build_ifeval_set.py`. To instead score the
official 541-prompt `google/IFEval` set, the same `ifeval_registry` checkers apply to any row
carrying `instruction_ids` + `kwargs` — our registry covers 24 of the 25 official types (only
`language:response_language` is omitted, as it needs a language-ID dependency).
