# Impl-3 comparability bundle

Everything needed to land Impl-4 numbers on the same axes as our Impl-3 results. Start with
`IMPL3_HANDOFF.md`; this file answers your three questions and lists what's in the tarball.

---

## Your three questions

### 1. Hub as-is, or locally regenerated SIs?

**Hub as-is — do not regenerate.** This is the important one, and the answer is the opposite of
what the handoff implied.

The per-dialogue system instructions are **already baked into the dataset rows**. A train row looks
like:

```
keys:  ['answer', 'dialogue_id', 'kind', 'messages', 'problem_id', 'source']
roles: ['system', 'user', 'assistant', 'user', ...]
```

Our training path is `build_sft_datasets(hf_dataset=...)` → `load_hf_sft_datasets()` →
`datasets.load_dataset()`, and it never calls `assemble_pedagogy_example()` or
`build_system_instruction()`. Those live in `common/system_instructions.py` and were used
*upstream, in the POC, to construct the dataset* — not at training time. If you regenerate SIs
locally you will get **different** system prompts from the ones we trained on and your pedagogy NLL
will not be comparable.

So: pull the Hub dataset and pin the revision.

```
meric533/socrateach-sft
revision sha: 1fd0b54ab8a0d96d07471f1f7d7173666d4071b8
```

`common/system_instructions.py` is included anyway, because `CANONICAL_SI` (loaded from
`common/prompts/canonical_si.txt`) *is* used at eval time — it is the exact string the `+SI` KL
condition prepends. That file is in the bundle; use it byte-for-byte.

Split sizes you should see: train 30,000 (22,500 pedagogy + 7,500 general, tagged by `kind`),
validation 1,724 (all pedagogy). After tokenization/filtering, 29,509 usable train rows and
923 steps at effective batch 32.

### 2. The 64 / 128 selection rule

**First-N in validation-split file order. No shuffle, no seeded sample.**

Both draw from `data/socrateach_sft_val.jsonl`, which is the Hub validation split written out in
order by `snapshot_hf_dataset.py` (included). I've put the actual file in the bundle so you don't
have to trust the reconstruction.

- **Pedagogy NLL (128):** literally `val[:128]`. All 128 `dialogue_id`s are unique.
- **KL (64):** `pedagogy_contexts(val, 64)` walks rows in file order and takes the first 64 usable
  ones. On this file **no rows are skipped — the 64 come from exactly `val[:64]`.** (The skip rule
  exists for rows with no tutor turn to stop before; none occur in this range.)

Two details in `pedagogy_contexts` that matter for matching the number:

- The dialogue is truncated **before the first assistant turn**, so every KL context is a *single
  user turn* — the student's opening problem. Verified: all 64 contexts have length 1 and end on
  `user`.
- **The row's own baked-in system message is stripped**, and then the canonical SI is prepended for
  the `+SI` condition (or nothing, for the no-SI condition). So the training SI never leaks into the
  KL measurement; `CANONICAL_SI` is the only system string involved.

### 3. Is the SFT baseline still around?

Yes. The adapter is on ORCD at `out/impl2-rerun/checkpoint-923/` (LoRA, 48 MB) and I can send it —
say the word. All 12 of its log-spaced checkpoints are in the results jsonl under run name
`impl2-rerun` (rendered as **SFT** in figures).

Its final checkpoint, which is what your A1 arm should reproduce:

| metric | value |
|---|---|
| kl_new_SI | 0.7607 |
| kl_ped_noSI | 0.1500 |
| ped_nll | 0.862 |
| math_hint | 0.212 |
| math_bare | 0.456 |
| math_hint_commit | 0.904 |
| math_hint_deflect | 0.476 |

Good sanity check, with one caveat: two vanilla SFT runs of ours that *should* have been identical
landed 0.11 apart on the LLM judge, so treat the judge as noisy. The deterministic columns above are
far more reproducible — a POC-lineage adapter trained on the same recipe matched `impl2-rerun` to
within 1% of the axis range on KL and math, despite different seeds and fp16-vs-bf16.

---

## Contents

| path | why |
|---|---|
| `IMPL3_HANDOFF.md` | the full writeup: objective, hyperparameters, eval protocols, results, pitfalls |
| `common/prompts/canonical_si.txt` | **the exact `+SI` string** — `kl_new_SI` is meaningless if this differs |
| `common/kl.py` | KL, cached base continuations, `pedagogy_contexts` (the truncation rule) |
| `common/system_instructions.py` | `CANONICAL_SI` loader; `build_system_instruction` is upstream-only (see Q1) |
| `eval/sweep_ckpt_eval.py` | the per-checkpoint driver: all three axes, protocol stamping, resume |
| `eval/math_eval/build_math_logic_set.py` | deterministic 250-item GSM8K builder |
| `eval/math_eval/math_logic_prompts.jsonl` | **the built 250-item set** — use this rather than rebuilding |
| `eval/math_eval/math_scoring.py` | answer extraction + equivalence (importable, no side effects) |
| `eval/math_eval/{grade_math_logic,score_results}.py` | standalone scorers |
| `eval/plot_figure3.py`, `eval/make_figures.sh` | figures in the same encoding |
| `out/ckpt_sweep_bare_hint250.jsonl` | our 194 rows — the schema to emit, and our points to plot beside yours |
| `data/socrateach_sft_val.jsonl` | the 1,724-row val split, in order — pins your 64 KL / 128 NLL items exactly |
| `snapshot_hf_dataset.py` | regenerates that val file from the Hub |
| `requirements.txt` | pinned versions |

On the math set: I've included **both** the builder and the built jsonl. Prefer the built jsonl.
You asked how the 45-item set expanded to 250 while staying a strict superset — the builder reads
the *existing* `math_logic_prompts.jsonl` and re-keeps its GSM8K ids (`previously_used()`) before
sampling the remainder at `random.seed(7)`. That means rebuilding from scratch without the old file
present will **not** reproduce our ids. Just use the jsonl.

---

## Emitting rows we can merge

Match these field names and `eval/plot_figure3.py` will read your rows unchanged:

```
run, step, variant, temperature,
kl_new_SI, kl_ped_noSI,
ped_nll,
math_bare, math_bare_commit, math_bare_deflect, math_bare_acc_given_commit,
math_hint, math_hint_commit, math_hint_deflect, math_hint_acc_given_commit,
prior_score,          # == math_hint
epoch, protocol
```

The `protocol` string stamps *how* a row was measured — KL context rule, math conditions, and a
hash of the math item ids:

```
kl=ctx-first-turn;math=bare+hint@250/<sha1[:8]>;ifeval=off
```

If yours differs from ours the merge should be refused rather than silently mixed, which is the
whole point of the stamp — we corrupted a results file twice before adding it. Using the shipped
`math_logic_prompts.jsonl` will make the hash match.

**Verify you're aligned before running anything expensive.** Every row in the shipped results file
carries this exact protocol string, and the shipped math set hashes to the same value:

```
kl=ctx-first-turn;math=bare+hint@250/995cd590;ifeval=off
```

```bash
# should print 995cd590
python -c "import json,hashlib;rows=[json.loads(l) for l in open('eval/math_eval/math_logic_prompts.jsonl')];\
print(hashlib.sha1(';'.join(sorted(r['id'] for r in rows)).encode()).hexdigest()[:8])"

# should print e2bde3bb... (614 bytes) — the canonical SI, byte-for-byte
shasum -a 256 common/prompts/canonical_si.txt
```

`canonical_si.txt` is 614 bytes, sha256 `e2bde3bbfdb8d6a56856b73f393b55606b8c54af7d413412b65fdf1c6f469e12`.
If that hash differs, every `kl_new_SI` you produce is on a different axis from ours.

Set `variant`/`temperature` to `null` for a run with no reweighting (that's how `impl2-rerun` is
encoded, and it's why it renders as the black reference line).

---

## Things that will silently break comparability

Restating the ones most likely to bite an Impl-4 arm specifically:

1. **Regenerating the SIs** instead of pinning the Hub revision (Q1). Changes the pedagogy training
   data; NLL becomes incomparable and nothing downstream will warn you.
2. **`kind` tagging.** Your replay stream is π₀-on-SuperNI instead of Tülu gold, which is fine — but
   those rows must still be tagged `kind: "general"` and carry **no** SI, matching ours. The 75/25
   pedagogy:general ratio should hold at 22,500 / 7,500.
3. **Reporting one math number.** Bare and hinted differ by 24 points on an SFT checkpoint and by
   ~0 on base. Always say which.
4. **Config drift between the YAML and the sbatch.** `impl3_kl_reweighted_sft/config.yaml` says
   per-device 8 / accum 4 with gradient checkpointing on; we actually ran 32 / 1 / off. Same
   effective batch 32, but quote the sbatch, not the YAML.
5. **`--resume auto` on torch < 2.6** silently restarts from scratch rather than resuming — two
   separate guards, both described in §7 of the handoff. If you're on preemptable nodes, verify
   before trusting a resumed run's step count.
