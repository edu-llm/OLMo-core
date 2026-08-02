# Math retention probe

Measures what the tutor SFT **lost**: `base` (OLMo-2-1B-Instruct) vs a pedagogy-tuned adapter on
verifiable final-answer accuracy. This is the old-task axis of the KL–forgetting curve.

## Test set — `math_logic_prompts.jsonl` (250 items, GSM8K only)

Rebuild deterministically with `python build_math_logic_set.py` (seed=7). Every item is an
integer exact-match, so grading needs no symbolic solver and no LLM judge.

It got here by elimination, and the reasons are worth keeping:

- **45 items was underpowered.** The original mixed set could not resolve 5–15 point accuracy
  gaps, which is exactly the size of the effects being compared. 250 items was chosen from a
  power analysis, and the original GSM8K ids are retained as a subset so old numbers stay
  comparable.
- **MATH-500 was dropped** because its `expr` answers need symbolic or LLM verification, which
  puts a subagent inside an otherwise deterministic loop.
- **AIME was dropped** as far too hard for a 1B model to produce signal.
- **BBH logical-deduction was dropped** after measuring it: 6.7% against a 14.3% chance floor.
  A probe sitting below chance cannot show forgetting, because there is nothing left to lose.

## Two prompt conditions, and why both are required

| condition | prompt | what it measures |
|---|---|---|
| **bare** | the question alone | arithmetic skill |
| **hinted** | question + "Put your final answer inside `\boxed{}`" | skill *and* willingness to answer |

The hint collides with what the tutor was trained to do — never state the final answer — so a
tutor-tuned model deflects into a Socratic question instead of answering. The same SFT
checkpoint scores **0.212 hinted and 0.456 bare**, while the base model is unaffected (0.664 vs
0.656). The hinted number is not "wrong": refusing to answer *is* a loss of prior-task ability
from the user's point of view. But it mixes refusal with genuine skill loss, and only the pair
separates them. `score_results.py` reports the commit rate alongside accuracy for this reason.

Neither condition uses a pedagogy system instruction — the probe measures the model as a
general assistant, which is the condition prior-task ability actually matters in.

## Files

- `build_math_logic_set.py` — rebuilds the 250-item set from the Hub.
- `math_scoring.py` — answer extraction and equivalence. Imported by both scorers below, so the
  per-run table and the per-checkpoint sweep can never disagree on what counts as correct.
- `score_results.py` — scores `generate_eval.py` output files and prints a comparison table.
- For a whole sweep, use `../sweep_ckpt_eval.py`, which scores every checkpoint in one pass.
