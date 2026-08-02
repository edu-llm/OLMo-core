# Pedagogy evaluation with the eval team's `tutor_cat`

How we run the eval team's CAT pipeline against our P7 checkpoints, what we had to add to make it
accept them, and the decisions behind both.

Status: **setup in progress.** Sections 1–4 and 7 are done and verified on ORCD. Sections 5–6 are
the plan and are marked where they have not yet been executed. Section 9 is the live resume point —
read it first when picking this back up.

---

## 1. What we are running, and why not the thing we were pointed at

Arhant pointed us at `add-cat-evals`, which runs **`atlas_arc`** — an adaptive ARC-Challenge test
returning a single ability estimate. That is a *prior-task* probe: useful for forgetting, unrelated
to tutoring.

The same repo contains **`eduLLM-Evals/tutor_cat`**, a CAT for LLM tutors, and that is what we use:

- an LLM judge grades each tutor response criterion by criterion;
- criterion verdicts update a 3-skill MIRT ability vector — **content, diagnosis, scaffolding**;
- a Fisher-information selector picks the next scenario until every skill is under its max SE
  (floor of 15 scorable evals, hard cap of 50 scenarios);
- critical failures are reported separately but still update theta.

This replaces our homemade 8-dimension judge, whose noise floor was **0.119** between two runs of
the identical recipe (`impl2` 0.653 vs `impl2-rerun` 0.534) — as large as the entire spread across
variant b. Any ranking built on it was unfalsifiable. An adaptive test with an explicit SE target
is the direct fix.

We are deliberately **not** running `atlas_arc` for now. Prior-task retention is already covered by
the 250-item GSM8K probe in `eval/sweep_ckpt_eval.py`, measured in two conditions across all 194
checkpoints.

## 2. Files we added

| File | Purpose |
|---|---|
| `.cursor/skills/add-cat-evals/` | Vendored copy of the eval team's skill (theirs, unmodified) |
| `clusters/orcd/setup_tutorcat.sh` | One-time ORCD env build (login node — needs network) |
| `eval/merge_for_tutorcat.py` | Folds LoRA adapters into standalone HF weights + writes the model manifest |
| `clusters/orcd/merge_tutorcat.sbatch` | CPU job wrapping the merge |

Nothing in `olmo-eval-full` is modified. We generate a manifest it consumes; their code is untouched,
so pulling their updates will not conflict.

## 3. Three integration problems that would have silently produced garbage

**Their loader has no LoRA path.** `tutor_cat/respgen/backends.py` does
`vllm.LLM(model=model_id)`, which wants an HF repo id or a directory of full weights. Our
checkpoints are PEFT adapters — `adapter_config.json` plus a few MB of `adapter_model.safetensors`
— which vLLM cannot load. Each adapter is therefore merged into the base once, offline, producing
an ordinary HF model directory (~2.9 GB each, ~48 GB for 17).

We considered serving one vLLM instance with `--enable-lora` and hot-swapping adapters instead, which
would avoid the disk entirely. Rejected: it requires forking their backend, and the merge is a
one-time cost against a pipeline we do not control.

**`apply_chat_template` is inferred from the model id string.** In
`tutor_cat/respgen/registry.py`:

```python
def guess_apply_chat_template(model_id: str) -> bool:
    mid = model_id.lower()
    return any(m in mid for m in _INSTRUCT_MARKERS)
```

Our ids are local paths like `/home/xing33/tutorcat_models/impl3-b-T2`. No `instruct` marker, so
this returns **False** and every chat-tuned model would be prompted as a raw base LM — plausible
looking output, meaningless scores, no error anywhere. The generated manifest pins
`apply_chat_template: true` on every row.

Also pinned: `max_model_len_cap: 4096` (OLMo-2-1B's real context) and `max_new_tokens: 1024` (their
default of 4096 exists to bound base-model rambling on 32k-context models; tutor turns are short,
and this saves generation time across ~660 scenarios x 18 models).

**The merge and the generation run disagreed about the config schema.** This one actually reached
the GPU and produced a full smoke run with exit code 0, 8/8 rows, no empty outputs, no truncation,
`Chat Template Applied` on every row — and text like:

> MPTI'm not surety.  How does it is a better.

The merge runs in `p7post` (transformers 5.x) and generation in `tutorcat`, where vLLM 0.10.x pins
transformers 4.x. `save_pretrained` writes the schema of whichever version performed the merge, and
5.x renamed `torch_dtype` to `dtype` and moved `rope_theta` inside a `rope_parameters` block:

```
base   (4.50):  "rope_theta": 500000,          "torch_dtype": "bfloat16"
merged (5.14):  "rope_parameters": {"rope_theta": 500000, ...},  "dtype": "bfloat16"
```

vLLM reads `config.rope_theta` at the top level. Finding nothing, it falls back to a **default of
10000** against a model trained at 500000 — positional encoding wrong at every position, no warning
anywhere. It degrades with context length, which is why a short hand-written prompt under
transformers 5.x looked perfectly fluent while real multi-turn TutorBench prompts came out as word
salad. The same mechanism broke the tokenizer, written as `"tokenizer_class": "TokenizersBackend"`
with the chat template split into `chat_template.jinja`, which 4.x rejects outright.

Fix: `merge_for_tutorcat.py` now copies `config.json`, `generation_config.json` and the tokenizer
from the base **verbatim**, overwriting what `save_pretrained` emitted. A LoRA merge changes tensor
values only, so the base's copies are correct by construction and the merged model no longer depends
on which env merged it. `tutorcat_generate.sbatch` additionally refuses to start unless all 17
models show a top-level `rope_theta: 500000`.

The lesson worth keeping: **every structural check passed while the model was broken.** Exit code,
row count, empty-output rate, template flag and truncation flag were all green. Only reading the
generated text caught it.

## 4. System instructions — the question of whether this measures our models fairly

It does, because `tutor_cat` supplies pedagogy system prompts itself, per benchmark. For TutorBench
they are the paper's verbatim Appendix A.6 prompts, keyed by use case:

- `hint_generation` — "Offer a helpful hint or question to guide them toward the next step,
  **without giving away the full answer**."
- `feedback` — evaluate the student's answer, identify mistakes, explain the reasoning.
- `adaptive_explanation` — adjust to what the student says they are confused about.
- `Bridge` — "guiding them toward the right approach **rather than simply giving away the answer**."

That is effectively our canonical SI, written independently. Our SFT models are asked to do what
they were trained to do, rather than penalised for tutoring when asked to answer.

Note the deliberate exceptions: **IFEval, InFoBench and EduBench get no system prompt at all**,
because a tutor persona corrupts their verifiers. This is the same failure we hit ourselves with the
boxed-answer hint on math, where SFT accuracy collapsed from 66% to 21% through refusal rather than
skill loss. Do not "fix" this by forcing our SI globally.

The rubric and reference solution are judge-only and never shown to the model.

## 5. The judge — NOT YET STOOD UP

Default is **`prometheus-eval/prometheus-7b-v2.0`**, self-hosted by vLLM on an MIT GPU node and
reached through an SSH tunnel at `localhost:8000`. Only the model runs on the cluster; no external
API and no credential is required for this path.

```
scripts/cluster/setup_prometheus.sh      # once, on the login node
sbatch scripts/cluster/serve_prometheus.sbatch
```

Documented fallback: point `base_url` at the TrueFoundry gateway with `gpt-5.5` (needs
`TFY_API_KEY`). The run manifest records `judge_model` either way.

**Open question for the eval team.** Prometheus answers on a 1–5 scale although the rubric is
pass/fail, so `config.yaml` maps `score >= result_pass_threshold` (default **4**) to a pass, with a
comment saying it should be tuned during judge calibration. That threshold sits directly underneath
every ability estimate. If they calibrated to a different value, our theta is not comparable to
theirs. **Ask before trusting cross-team numbers.**

A separate frozen spec exists for calibration runs — `Qwen/Qwen3.5-9B` pinned to an exact HF
revision with versioned prompt/normalization/evidence policies (`judge_frozen.yaml`).

## 6. Running it — PLAN, NOT YET EXECUTED

```bash
# 0. one-time, login node
bash clusters/orcd/setup_tutorcat.sh 2>&1 | tee ~/tutorcat_setup.log

# 1. merge adapters -> HF weights + manifest   (~3 min, CPU)
sbatch clusters/orcd/merge_tutorcat.sbatch
#    -> ~/tutorcat_models/<run>/ and ~/tutorcat_models.yaml

# 2. judge server (separate GPU job) + tunnel to localhost:8000

# 3. generate tutor responses for all 18 models
conda activate tutorcat
cd ~/olmo-eval-full/eduLLM-Evals
tutor-cat generate --models ~/tutorcat_models.yaml --benchmarks benchmarks.yaml --only TutorBench,Bridge

# 4. run the CAT engine -> theta per skill
```

18 models: base, 16 sweep runs, and `poc-c923` (the POC lineage, so both are finally measured on
one harness instead of compared across two).

Results stay on ORCD and come back by rsync. **No S3.** The `--s3-out` requirement lives only in
`run_cat_diagnostic.sh`, the `atlas_arc` wrapper we are not using; `tutor_cat` writes locally.

## 7. Verification already done

The merge is real, not 17 copies of the base. Comparing `model.layers.0.self_attn.q_proj.weight`
against `impl2-rerun`:

| model | relative difference |
|---|---|
| `impl3-b-T451` | 0.0026 |
| `impl3-b-T2` | 0.0228 |
| `impl3-a-T0.5` | 0.0375 |
| `poc-c923` | 0.0427 |

`b-T451` being closest to vanilla SFT is a third independent confirmation that the T→∞ limit
collapses onto the unweighted objective — after KL (0.759 vs 0.761) and pedagogy NLL (0.862 vs
0.862), now in raw weights. `poc-c923` being furthest is consistent with a separate training lineage.

Hashing the first 200 MB of each `model.safetensors` is **not** a valid check — that region is the
embedding matrix, which LoRA never touches, so all 17 hash identically while being genuinely
different models.

## 8. Environment build — what broke and what we changed

`clusters/orcd/setup_tutorcat.sh` ran to completion but two of its steps failed. Both are recorded
here because both were silent-ish, and the script deliberately treats them as non-fatal warnings.

**`tutor-cat` did not import at all.** Their `tutor_cat/verifiers/ifeval/instructions.py` does
`from absl import logging`, but `absl-py` is not declared in their `pyproject.toml`. Since `cli.py`
imports the IFEval verifier unconditionally, *every* subcommand died on import — including
`validate`. Fixed on our side, without touching their repo:

```bash
pip install absl-py nltk immutabledict langdetect
```

`tutor-cat validate` now reports `validation: OK`. The ~1092 `q_mapping maps to no skill (all zeros)`
warnings come from their shipped item bank, not from anything we did; worth raising with them, but
they do not block a run.

**The generation extra failed, leaving no `torch` and no `vllm`.** This is the one that matters —
without it there is no response generation. Reinstalling with `vllm==0.10.1.1` pinned pulled
*prebuilt* wheels (torch 2.7.1 + CUDA runtime, ~4 GB) rather than compiling, which is the fast path;
the earlier failure appears to have resolved onto a source build of the `llguidance` Rust extension.
Driven by `~/install_vllm.sh`, logging to `~/vllm_install.log`, detached so it survives a
disconnect. Note it downgrades `numpy` (2.4.6 → vLLM-compatible), so re-check `tutor_cat` imports
afterwards.

**Test suite: 10 failures out of 352.** Nine in `test_ingest_calibration_csv.py`, one in
`test_run_judge_validation.py::test_prepare_enforces_complete_scenario_by_tutor_matrix`. All are on
the *calibration-ingest* path — building an item bank from judged CSVs — which we do not use; we
consume their pre-calibrated bank. Not blocking, but it means their calibration tooling is in an
unclear state, which is another reason to ask before comparing theta across teams (section 5).

**A monitoring bug on our side, for the record.** The watcher polled
`ssh orcd 'pgrep -f setup_tutorcat.sh'` — and the SSH command string itself contains that pattern, so
`pgrep` matched its own probe and reported "still running" forever. The setup had in fact finished
long before. Poll for a completion marker written into the log, not for process liveness.

## 9. Resume point

Done: env built and verified end to end (torch 2.7.1 + vLLM 0.10.1.1), CLI fixed, adapters merged
and repaired, generation job running.

Two more dependency pins were needed beyond section 8, both found by the smoke test:

| Problem | Fix |
|---|---|
| `transformers 5.14.1` installed; vLLM declares only `transformers>=4.55.0`, no upper bound | pinned `transformers==4.55.2` (pulls `tokenizers` back to 0.21.4) |
| vLLM's transformers fallback needs `accelerate`, undeclared | `pip install accelerate` |

Generation job **19440110** (`clusters/orcd/tutorcat_generate.sbatch`), 1 GPU on `mit_preemptable`,
4 h, TutorBench + Bridge, 18 models x 1304 scenarios. Resume is the harness default, so preemption
costs at most the model in flight — just resubmit.

Next, in order:

1. Read actual generated text once the first shards land — not just row counts (section 3).
2. Stand up the Prometheus judge (section 5); independent of generation, can overlap.
3. Run the CAT engine for theta per skill.

Still unknown: per-model generation cost. Take it from the first completed shard before assuming
4 h covers all 18.

## 10. Results

Job 19442261, ~58 min: **18/18 models reached the SE ≤ 0.30 target** in 13–20 scenarios each. The
measurement worked. What it measured is uncomfortable.

**No variant is better than the base instruct model on any skill.** Not one, out of 51 model-skill
comparisons. Base ranks 1st of 18 on summed theta. Several variants are significantly worse,
almost all on *content*:

| verdict vs base (\|z\| ≥ 2) | models |
|---|---|
| content worse | `a-T32`, `b-T0.5`, `b-T16`, `b-T2`, `b-T8`, `poc-c923` |
| scaffolding worse | `a-T4`, `b-T2` |
| indistinguishable | the other 10, including `impl2-rerun` |
| better on anything | **none** |

Individually most gaps sit inside the error bars, but the *sign* pattern does not: 46 of 51
differences are negative. `impl2-rerun` — vanilla SFT, our best-behaved run — is statistically
indistinguishable from the model it was fine-tuned from.

### 10.0 Why base outscores everything: rubrics are coverage checklists

TutorBench criteria are content checklists, not pedagogy judgments. **53% of the 6462 criteria** are
phrased "the response must explain / mention / include / identify \<specific point\>"; only 3% ask
the tutor to question, elicit, or guide. Scoring is therefore "how many required points did you
cover", and fine-tuning made our models far terser — base averages **2948 chars** per response,
`impl2-rerun` **808**, `impl3-b-T32` **443**.

Pooled over the 921 scored judgments, pass rate tracks length directly:

| response length (chars) | judgments | pass rate |
|---|---|---|
| < 500 | 199 | 31.7% |
| 500–900 | 148 | 36.5% |
| 900–1400 | 254 | 52.8% |
| 1400–2200 | 252 | 58.7% |
| ≥ 2200 | 68 | 45.6% |

Worked example, scenario `tb_0251` (student: "why does the method only return whole numbers?").
Criterion `c03` requires explaining that the return type must be `double`/`float`:

| model | chars | bold/code | c03 verdict |
|---|---|---|---|
| `base-instruct` | 2157 | yes | pass (5) |
| `impl3-a-T4` | 1244 | yes | pass (5) |
| `impl2-rerun` | 1097 | yes | pass (4) |
| `impl3-b-T1` | 856 | no | fail (3) |
| `impl3-b-T0.5` | 747 | no | fail (2) |
| `impl3-b-T8` | 453 | no | fail (3) |
| `impl3-b-T32` | 443 | no | fail (3) |

**But length is not the whole story.** Restricted to responses of comparable length, base still wins:
91% (21/23) vs 56% (127/229) in the 1400–2200 band, 59% (10/17) vs 41% (21/51) above 2200. Small
n for base, but it is the one signal here suggesting a real quality gap rather than a style gap.

**Correction to an earlier note.** We previously "rejected" the hypothesis that TutorBench penalizes
withholding the answer, on the grounds that only 81 of 6462 criteria (1.3%) explicitly demand the
final answer — 2 sampled for base, 4 for `impl2-rerun`. That test was too narrow. The benchmark does
not require the *answer*, but it does require *explanatory coverage*, which a Socratic reply omits by
design. The narrow claim stands; the conclusion drawn from it did not.

### 10.05 The judge inverts `critical_negative` criteria — it penalises withholding the answer

Scenario `tb_0504` (physics, `use_case=hint_generation`) is the cleanest test of what our SFT
targets. The student has correctly derived the putty's angular momentum and the 7/12 ML² inertia,
then written `M v_0 L/2 = ½ I ω` — the kinetic-energy form instead of `Iω`. Criterion `c14` is
`criticality=critical_negative`: *"The response must not complete the algebra or give the answer
which is ω = 6v₀/7L."*

`impl3-b-T4` and `impl3-b-T8` both answered:

> Great job! Now that you have the total moment of inertia, can you solve for the angular speed ω?

That satisfies c14 exactly. The judge returned `[RESULT] 1` (fail) with this reasoning:

> The response does not satisfy the criterion at all. It does not provide any guidance or
> instruction on how to solve for the angular speed ω, **nor does it give any hints or clues that
> would help the student to arrive at the correct answer.** [...] it does not contribute to the
> completion of the algebra or the determination of the angular speed.

The judge read the negative criterion as a positive one and **failed the response for complying with
it**. The same two models also failed `c13` ("should guide the student to solve for ω") even though
the response literally asks the student to solve for ω.

Scope: 163 of 6462 criteria (2.5%) are `critical_negative`; only 26 were scored across all 18 models,
so the aggregate effect is small. It matters because it lands precisely on the behaviour we optimise
for, making the scaffolding subscore the least trustworthy number in the run. Fix before rerunning:
either negate the verdict for `critical_negative` rows, or rewrite the judge prompt so the criterion
is presented as "the response correctly avoids X".

**Rubric shape vs our objective.** Of the 17 criteria on `tb_0504`, 12 require explicitly affirming,
acknowledging, restating or writing out a specific piece of the student's work; 5 reward hint-ladder
behaviour. A well-aimed 96-character hint caps out around 5/17 regardless of quality. Benchmark-wide
the skill split is content 2843, diagnosis 1714, scaffolding 798, unlabelled 1107 — scaffolding is
12% of the rubric.

**A real weakness this exposed, independent of the benchmark.** Our hints fire but are not aimed at
the error. `impl3-b-T4` nudges toward solving for ω when the student's next line already attempts
exactly that with the wrong formula; `impl2-rerun` asks for a moment of inertia the student already
computed correctly; `impl3-b-T2` asserts the inertia calculation is wrong when it is right. That is a
diagnosis failure worth fixing on its own merits.

### 10.1 The judge failed to emit a score on two thirds of judgments

**This is the dominant caveat and it invalidates the absolute numbers.** Across all 18 runs,
1716 of 2637 rubric judgments (65%) came back flagged `unscorable_reason: unparseable_judge_output`
— and the harness counted **every one of them as a failure** rather than dropping it from the
likelihood. Each model's theta is therefore fitted mostly to automatic failures, which is why every
estimate lands between −0.35 and −1.66.

The mechanism is a decoding failure, not a benchmark property:

| | scored | unscorable |
|---|---|---|
| judgments | 921 | 1716 |
| median length (chars) | 919 | 4938 |
| p90 length | 1786 | 5433 |
| contains `[RESULT]` tag | 921 / 921 | 9 / 1716 |
| ends mid-sentence | — | 1630 / 1716 |
| counted as pass | 6–43 per model | 0 / 1716 |

Prometheus falls into degenerate repetition — restating the same criticism in slightly different
words — and hits its generation cap before reaching the `[RESULT] n` line. The failures are not
concentrated on hard items: of 366 criteria that failed to parse for some model, none failed for
15+ of the 18. Before rerunning, raise the judge's `max_tokens`, add a repetition penalty, and
retry-on-parse-failure; ideally also change unscorable handling from *fail* to *drop*.

**Restricted to the 921 judgments that were actually scored, the ranking holds and sharpens.**
Base passes 78% of its scored criteria (31/40, 95% CI 65–90%); `impl2-rerun` passes 49% (43/88,
CI 38–59%). Those intervals do not overlap, which is a stronger separation than the theta
comparison (z = −0.3) reports. Item difficulty differs between models because the test is adaptive,
so this is a complement to theta, not a replacement — but the two readings agree on direction, and
that agreement is the most defensible result of the run.

**Absolute levels are not anchored either.** Independently of the parse failures, Prometheus answers
1–5 and `result_pass_threshold: 4` maps only 4–5 to a pass. That threshold is **uncalibrated**
(section 5), so absolute theta is not comparable to the eval team's numbers. The relative comparison
is what the run supports, since every model faced the same judge and the same threshold.

Caveats to carry: 109–184 criteria judged per model (14–88 actually scored), 70–112 critical
failures each, and TutorBench only — Bridge responses are generated but need a separate 5-skill
config (section 6).

## 11. Housekeeping

`~/tutorcat_models` is 48 GB and home quota is 195 GB (~96 GB used with it). Prometheus 7B needs
another ~15 GB. Delete the merged models once theta is recorded — they are reproducible in 3 minutes
from the adapters.
