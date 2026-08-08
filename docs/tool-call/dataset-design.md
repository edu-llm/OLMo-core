# Dataset design: tool-call SFT for OLMo MoE (~4B active)

Companion to [`prd.md`](prd.md). This is the *shape* document — what the bytes look like, where
they sit, and what the validator does to them. Log changes in [`progress.md`](progress.md).

> **Evidence policy.** Claims here are either (a) executed against the installed
> `edullm-data==0.2.0`, (b) cited to a repo `file:line`, or (c) cited to a paper/dataset card.
> Anything else is marked **UNVERIFIED**. Two claims in the first draft of this document were
> wrong and are corrected in §7 and §4 — do not restore them.

---

## 1. Verified pipeline contract

Executed, not read. Reproduce with [`verify/verify_record_shape.py`](verify/verify_record_shape.py)
and [`verify/verify_naming.py`](verify/verify_naming.py), using a venv that has `edullm-data`
installed (it is **not** importable from the default python — pin the venv in any CI gate or the
pre-publish checks silently do not run):

```bash
<venv>/bin/python3 docs/tool-call/verify/verify_record_shape.py
<venv>/bin/python3 docs/tool-call/verify/verify_naming.py
```

### Registered profiles (exhaustive, as installed)

```
eval-results/v1   pretrain-tokens/v1   sft-conversations/v1   token-order/v1   tokenizer/v1
```

`sft-conversations/v1` is registered. **`eval-items/v1` is not** — see §9.

### What `sft-conversations/v1` enforces

Source: `edullm_data/profiles/sft_conversations_v1.py`.

| Rule | Detail |
| --- | --- |
| Row container | `.jsonl` / `.jsonl.gz`, one JSON object per line |
| `messages` | non-empty list of objects |
| `role` | non-empty **string** on every message |
| `content` | the **key** must be present. `null` is **accepted**; a missing key is rejected |
| roles enforced? | **No.** `_ROLES_HINT = {system,user,assistant,tool}` is defined and never referenced |
| Leakage | **recomputed** from rows, not trusted. `max_leakage` defaults to **0** |
| Heldout detection | partition whose **`name`** contains `heldout`/`held-out`/`holdout`/`test`/`val`/`eval` |

Required `group_meta`: `record_schema`, `partitions` (≥2), `dedup`, `leakage`.

Per partition (`validate.py:418-439`): `name`, a **`rows`** key, `by ∈ {path,field,range,indices}`,
and for `by:"path"` a `glob` that **must match ≥1 manifest path**. Globs are `fnmatch` against
basename **or** full path, so a basename glob reaches through arbitrary nesting.

### Two things the skill docs get wrong for this version

1. **No path labels in 0.2.0.** `ManifestEntry.from_dict` accepts *only*
   `{path, sha256, bytes, count, format}` and **raises on unknown keys** — no `labels`, no `split`.
   `PATH_LABEL_KEYS` and `labels_from_path` do not exist. Slicing is by **object path only**, which
   is why read-side mixing (§11) is glob-based.
2. **`coverage` is required and neither skill mentions it.** `validate.py:_validate_partitions`
   emits `bad-coverage` unless the group declares `coverage ∈ {partition, overlapping,
   incomplete}` whenever partitions exist. This profile requires partitions, so it is mandatory.

---

## 2. IRREVERSIBLE: the tool call goes in `content`

Not in a sibling `tool_calls` field. Forced by the leakage check, and measured:

```
A vs A2  same text, DIFFERENT call in SIBLING field:  COLLIDE=True
B vs B2  same text, DIFFERENT call INSIDE content:    COLLIDE=False
```

`_dedup_key` hashes `role \x1f content` per message. An OpenAI-style row parking the call in
`tool_calls` with `content: null` is **invisible to the only integrity check this profile performs
on our payload**. That breaks both ways: a train/heldout pair differing only in the call collides
and (with `max_leakage: 0`) refuses the whole publish; and genuine near-duplicates get graded on
the user turn alone.

**`dedup_key` is not an escape hatch.** Its docstring says "message fields" but the code is
`str(row.get(f, ""))` — top-level row keys only, it cannot reach inside `messages`.

**Consequence for `tools`:** tool definitions may ride as a top-level row field, but that field is
likewise outside the dedup key. Definitions therefore live **inside the `system` message
`content`** (§3), which is what the model conditions on anyway.

**Consequence for `content: null`:** the profile accepts it. An abstention generator bug emitting
`null` instead of prose would publish cleanly and teach the model to emit nothing. Our own
non-empty-string check (§10) is load-bearing.

---

## 3. Record format

Roles `system` · `user` · `assistant` · `tool` — we enforce this set ourselves, since the profile
does not. **Pin the whitespace**: `<tool_call>` and `<tool_call>\n` tokenize differently, so the
spec is the bytes.

- **Schema block** — inside `system` `content`, once, at index 0:
  `<tools>\n` + one compact JSON object per line + `</tools>`, each
  `{"type":"function","function":{"name":…,"description":…,"parameters":{JSON Schema 2020-12}}}`.
  The OpenAI/Hermes wrapper, chosen so the one clean upstream source needs no schema conversion.
  **Pick one wrapper and never mix.**
- **Call** — inside `assistant` `content`:
  `<tool_call>\n{"name":"…","arguments":{…}}\n</tool_call>`, **name-first always**. Parallel calls
  are ≥2 blocks joined by a single `\n`.
- **Result** — inside a `tool`-role message: `<tool_response>\n{…}\n</tool_response>` plus a
  per-message `name`. Fixed now so the deferred multi-turn set agrees.
- **Abstention** — ordinary prose with **zero** occurrences of `<tool_call>`. No positive marker;
  that would be a second thing to get wrong.

**(a) single call**

```json
{"messages":[
 {"role":"system","content":"<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current conditions for a city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}\n</tools>"},
 {"role":"user","content":"weather in Boston?"},
 {"role":"assistant","content":"<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Boston\"}}\n</tool_call>"}
]}
```

**(b) multi-turn with the result fed back** (deferred to v2, format fixed now)

```json
{"messages":[
 {"role":"system","content":"<tools>…</tools>"},
 {"role":"user","content":"weather in Boston?"},
 {"role":"assistant","content":"<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Boston\"}}\n</tool_call>"},
 {"role":"tool","name":"get_weather","content":"<tool_response>\n{\"temp_f\":54}\n</tool_response>"},
 {"role":"assistant","content":"It's 54°F in Boston."}
]}
```

**(c) abstention** — tools offered, the applicable one deliberately absent, assistant answers or
asks; no `<tool_call>` substring anywhere.

All shapes were run through `_messages_wellformed` and pass.

Heldout rows additionally carry a **top-level `answer_key`** with per-slot *sets* of acceptable
values, mirroring BFCL's `possible_answer/`. Without it AST exact-match under-reports, because
BFCL explicitly allows multiple correct values per slot. Extra top-level keys are legal, and
being outside the dedup key is correct for a grading key.

---

## 4. Path layout

```
conversations/<domain>/<category>/<split>-<NNNNN>.jsonl
```

- `<domain>` ∈ `general` | `edu`.
- **`<category>` is a CAPABILITY band, not a tool family.** The first draft used
  `edu/gradebook`, `edu/curriculum` — that was a mistake and is corrected here. Two levels is the
  entire budget; spending level 2 on tool family makes it impossible to ask "is parallel calling
  worse on edu tools", which is the one cross-domain question the domain split exists to answer.
  **Tool family rides as a top-level row field.** Category vocabulary is identical in both domains
  so the subtrees are comparable.
- **Exactly two levels below the group.** 0.2.0 has no `labels_from_path` so a third level would
  not raise *today*, but the newer pipeline derives exactly two label levels and **raises on a
  third**. Two levels keeps us forward-compatible with a feature we would otherwise have to
  republish every byte to adopt.
- **`<split>-<NNNNN>`, no `-of-NNNNN`.** Verified: `train-00000.jsonl` → `('train', 0)`. Both
  `conversations/train.jsonl` (no index) and `train-00000-of-00004.jsonl` are **rejected**.
- Format `container: "jsonl", codec: "none"`. Declaring `gzip` on a `.jsonl` name is rejected.

---

## 5. Partitions, coverage, heldout

```python
"coverage": "partition",          # REQUIRED whenever partitions exist (§1)
"partitions": [
    {"name": "train",   "by": "path", "glob": "train-*.jsonl",   "rows": 36000},
    {"name": "heldout", "by": "path", "glob": "heldout-*.jsonl", "rows": 4000},
]
```

Verified: those globs match through all nesting levels and are disjoint, as
`coverage: "partition"` requires. `rows` is emitted by the writer, never hand-typed.

**Heldout is carved by held-out tool schema, before generation** (~12% of schemas per domain →
~4,000 rows / 10%). The row fraction is a *consequence* of the schema carve, not a knob. A random
row split leaves the same functions on both sides and measures memorization.

**Schema carving alone is not sufficient.** If heldout user turns come from the same template bank
as train, heldout measures template recall with a new function name pasted in. **Carve the query
template bank alongside the schema pool**, or generate heldout queries with a different
generator/prompt. Nothing in the pipeline can see this.

---

## 6. Composition — 40,000 rows

`<category>` = capability. Domain split **general 27,900 (69.75%) / edu 12,100 (30.25%)**.

| path (`conversations/…`) | capability / BFCL analogue | rows | % | source |
| --- | --- | --- | --- | --- |
| `general/single-call` | 1 tool offered, 1 call — `simple` | 6,300 | 15.75 | reformat 2,400 / synth 3,900 |
| `general/multi-tool-select` | 3–20 offered, 1 call — `multiple`, `live_multiple` | 7,000 | 17.50 | reformat 2,300 / synth 4,700 |
| `general/parallel-call` | ≥2 calls in one turn — `parallel`, `parallel_multiple` | 4,500 | 11.25 | reformat 1,200 / synth 3,300 |
| `general/nested-args` | argument fidelity: nested objects, arrays, enums, unit/date coercion | 5,200 | 13.00 | synth, schema-first + executable stub |
| `general/relevance-hard` | a tool *does* apply, behind near-miss distractors — `live_relevance` | 2,100 | 5.25 | derive from verified positives |
| `general/no-suitable-tool` | **abstention**: gold function deleted from inventory — `irrelevance` | 1,960 | 4.90 | derive (Hammer deletion) |
| `general/missing-args` | **abstention**: required arg absent → clarifying question | 840 | 2.10 | derive (arg elision) |
| `edu/single-call` | ” | 2,700 | 6.75 | synth |
| `edu/multi-tool-select` | ” | 3,000 | 7.50 | synth |
| `edu/parallel-call` | ” | 1,500 | 3.75 | synth |
| `edu/nested-args` | ” (edu args are richest: rubrics, date ranges, roster filters) | 2,800 | 7.00 | synth |
| `edu/relevance-hard` | ” | 900 | 2.25 | derive |
| `edu/no-suitable-tool` | ” | 840 | 2.10 | derive |
| `edu/missing-args` | ” | 360 | 0.90 | derive |
| **TOTAL** | | **40,000** | **100** | reformat 5,900 (14.75%) / derived 7,000 (17.5%) / **fresh synthesis 27,100 (67.75%)** |

**Abstention = 10.0%** (`no-suitable-tool` 2,800 + `missing-args` 1,200 = 4,000). Best-evidenced
number here: Hammer swept the ratio on 10,000 sampled instances fine-tuning **Qwen2-1.5B-Instruct**,
found the optimum ≈10%, shipped 7,500/67,500 = 11.1%. The per-ratio curve is **figure-only,
UNVERIFIED**, so what 5% vs 20% costs is not knowable from the literature. Split 70/30 between the
two flavours because "no suitable tool" is what BFCL `irrelevance` actually scores.

**Why 40,000 — stated honestly: there is no published data-quantity sweep for function-calling SFT
at any scale.**

- **Floor 25,000** — the only *controlled* datapoint: ToolACE's matched 25k SFT on
  Llama-3.1-8B-Instruct → BFCL-v3 overall 58.19 / non-live AST 86.96 / irrelevance 86.42, beating
  xLAM-25k (40.51 / 81.94 / 11.87) and ToolLLM-25k (24.90 / 42.46 / 4.41) at identical size.
- **40,000 recommended** — the general subtree alone (27,900) already sits at that controlled
  point, so the domain we want comparability for is independently at a demonstrated size; edu adds
  on top. Inside the 25k–67.5k band every leaderboard-competitive small-model recipe occupies.
- **Ceiling 67,500** — Hammer (60k xLAM + 7.5k irrelevance); 60k is the only size with a
  demonstrated sub-2B success. Reachable only if xLAM licensing resolves permissively (§8).
- **Not a target: ~100 rows.** "Awakening the Sleeping Agent" moves a model 0% → 83.8% BFCL with
  ~100 traces, but that is *format reactivation* on a model with latent tool ability. It bounds the
  cheapness of the syntax component, not capability acquisition.

Category weighting is **directional only**. The one size-matched signal is Hammer-4b's per-slot
table: AST simple **62.58** vs multiple 77.72 / parallel 69.12 / parallel-multiple 68.92, and Exec
simple **67.79**. At ~4B the weak slot is *argument-value fidelity*, not function selection —
which is why `nested-args` gets 8,000 rows (20%) despite having no BFCL category, and why
`single-call` is not the largest cell.

---

## 7. Special tokens — plain text now, reserve the option

**CORRECTION.** The first draft of this document claimed "a control-token block already occupies
slots at the top of that range." **That is false for this branch.** `grep -rn
"control_tokens\|assert_control_tokens_fit\|reserved_special\|CONTROL_TOKEN" src/` returns **zero
hits**. (The registry that claim came from lives on the unmerged `latent-superposition-module`
branch, not on `main`.)

Verified headroom, `src/olmo_core/data/tokenizer.py:84-94`:

| Quantity | Value |
| --- | --- |
| `vocab_size` | **100278** (max real id 100277) |
| `eos_token_id` | **100257** |
| `pad_token_id` | **100277** |
| `bos_token_id` | **None** |
| `padded_vocab_size()` | **100352** (`pad_multiple` 128) |
| **Free ids** | **74 — `100278`–`100351` inclusive, 0 claimed by any registry** |

Those 74 embedding and `lm_head` rows are **already allocated** by every training script and are
currently garbage-initialised.

**Recommendation: plain-text delimiters for the dataset. Spend zero ids now.**

1. **The JSONL is byte-identical either way** — only tokenization is affected. Deciding late costs
   nothing; deciding wrong costs a republish.
2. **No study isolates the variable** — nobody holds tool-call semantics fixed and varies
   reserved-id vs plain text, at any scale. Any claimed win is UNVERIFIED.
3. **Cold embedding rows are a documented failure mode.** Llama's untrained reserved slots behave
   badly — models effectively refuse to emit them. Our 74 rows are untrained. Claiming an id at
   *SFT* time on a model whose pretraining never emitted it is exactly that failure. Reserving ≠
   training.
4. **Both sides of the efficiency ledger are rounding errors** — ~16–20 ids saved of a 150–400-id
   exchange; 8 tokens × d≈2560 × 2 ≈ 41k params ≈ 1e-5 of a 4B model.
5. **Revealed preference:** every family that spent real compute on tool use converged on
   *envelope as token, schema and args as plain-text JSON*. Nobody tokenizes `<tools>` or a schema.

**Reserved-id option, for pretraining/mid-training only, never SFT:** `100344 <tools>`,
`100345 </tools>`, `100346 <tool_call>`, `100347 </tool_call>`, `100348 <tool_response>`,
`100349 </tool_response>`, `100350`–`100351` spare — 8 of 74, no embedding resize. **This needs a
control-token registry that does not exist in this repo.** Claiming ids without one is how two
workstreams collide on the same id — and `latent-superposition-module` already claims ids at the
top of this block. Write the registry before any run.

**Measure, do not guess:** the contents of dolma2 ids **100258–100276** (19 unnamed ids between
`eos` and `pad`; needs HF `allenai/dolma2-tokenizer` `added_tokens`, not resolvable from this repo
— UNVERIFIED). In cl100k-derived vocabs these are typically FIM / `<|endofprompt|>` specials. If
one is a usable control token, reuse beats append.

---

## 8. Reusable upstream sources

The test is not "does the card say Apache-2.0" but "could the uploader grant model-training rights
at all."

| dataset | license | rows we can take | verdict |
| --- | --- | --- | --- |
| **Team-ACE/ToolACE** | `apache-2.0`, cleanest in the survey; generator models undisclosed (UNVERIFIED whether OpenAI output is in the release) | ≤11,300 total; single-turn fraction **UNVERIFIED**; est. **~4,000** after scope + quality filters | **TAKE** — the only source whose bytes can go into an open release, and the only negatives source with published transfer evidence |
| **NousResearch/hermes-function-calling-v1** | `apache-2.0` | **~1,890** (`func_calling_singleturn` only — exactly system+user+assistant) | **TAKE-SMALL**, and use as the **format reference**: the only upstream already in our target shape |
| **Salesforce/xlam-function-calling-60k** | `cc-by-4.0` **tag** vs gated "research purposes only … in support of an academic paper" **prose** | **0 today**; 60,000 if a human resolves it | **BLOCKED** — needs a decision, not more research |
| **ToolBench / ToolLLM** | see card | — | **EXCLUDE** — dead endpoints, hallucinated APIs |
| **glaiveai/glaive-function-calling-v2** | see card | — | **EXCLUDE** — quality; also the source Hermes' `glaive_*` subset re-derives |

Fixes required on ToolACE rows: drop empty `{"from":"assistant","value":""}` rows (documented, and
trains the model to emit `""`); repair find-replace corruption in API **names and param names**
(`valistring`←`valid`, `start_string`←`date`); convert bracket calls
`[Quotes by Keywords(word="inspiration")]` → our JSON-in-`content`; parse tool specs out of the
free-text `system` string; classify the four modes ourselves (unannotated); fix temporal
incoherence ("last financial year"→2025, "past month"→2022).

Fixes on Hermes: **shuffle before splitting** (rows are grouped by category, ~15 consecutive "IoT
and Home Automation"); undo doubled JSON escaping; pick one key order — the card shows
`{"arguments":…,"name":…}` inside `<tool_call>` but `{"name":…,"arguments":…}` in history. We are
name-first everywhere.

**Not yet investigated, and possibly the most valuable source of all:** AI2's own OLMo-3 SFT mix
already contains tool-use data — `src/scripts/train/sft/README.md:52-56` lists
`allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed` plus four
`allenai/olmo-toolu-s2-sft-*` mixes, and the tokenizer
`allenai/dolma-2-tokenizer-olmo-3-instruct-final` **carries its own chat template** (line 64: "the
chat template is loaded from the tokenizer"). If that template already defines tool-call
delimiters, §3 and §7 should **match OLMo's own convention rather than invent one**. See §12 Q1.

---

## 9. Eval set — the heldout partition *is* the eval set in v1

**Not a separate `dataset_id`.** `sft-conversations/v1` already requires a held-out partition, so
this is zero extra pipeline work. A separate `eval/<name>` of benchmark **items** is **blocked**:
`eval-items/v1` is unregistered and publishing into it raises `ProfileError`. Getting it costs a
registry entry, a schema fragment, recomputing checks, two fixtures, and a conversation with Eric.
The `eval/` dataset that *does* have a home is `eval-results/v1` — harness **outputs**, which is a
better thing to publish anyway.

Heldout rows keep the full conversation including the gold assistant turn (the leakage check must
see the call — that is why it is in `content`); the harness truncates at the last `user` message.

| our heldout shards | BFCL category | BFCL n | ours (gen) | ours (edu) |
| --- | --- | --- | --- | --- |
| `*/single-call/heldout-*` | `simple_python` | 400 | 630 | 270 |
| `*/multi-tool-select/heldout-*` | `multiple` / `live_multiple` | 200 / 1037 | 700 | 300 |
| `*/parallel-call/heldout-*` | `parallel` / `parallel_multiple` | 200 / 200 | 450 | 150 |
| `*/nested-args/heldout-*` | none — grades *into* AST arg checking | — | 520 | 280 |
| `*/relevance-hard/heldout-*` | `live_relevance` | ~41 (UNVERIFIED) | 210 | 90 |
| `*/no-suitable-tool/heldout-*` | `irrelevance` / `live_irrelevance` | 240 / 875 | 196 | 84 |
| `*/missing-args/heldout-*` | `multi_turn_miss_param` (multi-turn only) | 200 | 84 | 36 |
| **total** | | | **2,790** | **1,210** |

Sizing rule: **each category ≥ its BFCL counterpart's n** where one exists, so our confidence
interval is never worse than the number we compare against. Deliberately not replicated:
`live_parallel` at 16 cases, where one flip moves the column 6.25 pts.

Reporting rules that make the numbers comparable rather than merely similar:

- **Report `irrelevance` and `relevance` as a pair, never blended.** Irrelevance alone is trivially
  gamed: Deepseek-Coder-1.3B-Instruct scores 100.00 / 0.00 by never calling; ToolLLM-SFT scores
  4.41 / 100.00 by always calling. Target shape is a pair like ToolACE-8B (83.81 / 85.37).
- **Follow the post-CHANGELOG convention** — BFCL now excludes irrelevance/relevance from Non-Live
  and Live and reports them separately as Hallucination.
- **Grade AST-only and say so.** BFCL v4 comments out `exec_*`/`rest`/`sql` from default scoring,
  so AST-only is now the *same shape* as v4 default — we do not need live APIs to be comparable.
- **Pool any cell under 200 rows** before reporting (`edu/no-suitable-tool` 84,
  `edu/missing-args` 36, `edu/relevance-hard` 90) → report domain-pooled at 280 / 120 / 300.
- **Do not claim a BFCL overall.** v4's stated weighting (Agentic 40% + Multi-Turn 30% + Live 10% +
  Non-Live 10% + Hallucination 10%) conflicts with the leaderboard's "unweighted average" text, and
  this dataset touches **0%** of Agentic and Multi-Turn.

---

## 10. Generation + verification

No LLM judge in the write path; a judge's verdict is advisory and never decides admissibility.
Cheap gates first. **Per row, dropped if any fails:**

1. **Container** — line parses; `messages` non-empty; every message has non-empty string `role` and
   a present `content`. We additionally require `content` to be a **non-empty string** (§2) and
   `role ∈ {system,user,assistant,tool}` (the profile enforces neither).
2. **Schema block** — exactly one `system` at index 0, exactly one `<tools>…</tools>`, parses as a
   JSON array, each `parameters` validates against the JSON Schema 2020-12 metaschema.
3. **Assistant dichotomy** — assistant `content` is *either* only `<tool_call>` blocks separated by
   single newlines, *or* contains zero `<tool_call>`/`</tool_call>`. Mixed prose-plus-call is
   rejected: unambiguous for a 4B model, exact for the verifier.
4. **Call payload** — body is a JSON object with exactly `{name, arguments}`, `arguments` an object,
   **name-first** checked by compact re-serialisation and byte comparison.
5. **Resolve** — `name` is declared in *that row's* `<tools>`. Plus globally: no function name may
   appear anywhere in the corpus with two different schemas (Glaive's documented defect).
6. **Schema validation** — `jsonschema` validate `arguments` with `additionalProperties: false`
   forced (catches invented params); all `required` present; types/`enum`/`format` correct.
7. **Value plausibility, partial** — enum membership, declared numeric bounds, ISO-8601 parse where
   `format: date-time`, unit consistency where a unit enum exists. Not general semantics.
8. **Abstention rows invert 3–6** — zero `<tool_call>`; the intended function is **absent** from
   `<tools>` (the Hammer deletion, so correctness is *constructive*, not judged); assistant content
   contains no function name from the global inventory.
9. **`missing-args` rows** — no call; content contains `?` and names ≥1 `required` parameter of the
   intended tool that is absent from the user turn. A mechanical proxy, and the strongest available.
10. **Executable stub, `nested-args` only** — every synthesised tool ships a type-annotated Python
    stub; `inspect.signature(stub).bind(**arguments)` then call it; admissible only if it returns.
    The one place execution earns its cost: it catches coercion and range errors JSON Schema
    cannot, and argument fidelity is the measured weak slot at 4B.
11. **Token budget** — tokenize with dolma2, record `n_tokens_dolma2` as a top-level field, reject
    over the SFT sequence length. That length is UNVERIFIED until post-training fixes it; recording
    the field lets the trainer filter later without re-tokenizing. A 20-schema `system` message can
    blow the window on its own.

**Pre-publish, over the whole build directory:**

12. **Schema-pool disjointness** — every tool name in a row's `<tools>` belongs to the pool matching
    that file's split. This converts "heldout carved by schema" from an intention into a fact.
    Nothing in the pipeline checks it for us.
13. **Recompute the profile's leakage key locally** — sha256 over `role \x1f content` per message;
    assert `train ∩ heldout = ∅` **before** uploading. With `max_leakage: 0`, 40,000 rows and a
    finite schema pool, one accidental duplicate refuses the entire publish. Do not discover this in
    landing. Also report the within-partition duplicate rate.
14. **Naming and depth** — `parse_shard_name` on every basename; reject `-of-NNNNN` and index-less
    names; assert the path is exactly `conversations/<domain>/<category>/<split>-NNNNN.jsonl` so a
    third nesting level can never slip in.
15. **Diversity caps** — no single function >**2%** of its category's rows; no single user-turn
    5-gram >**0.5%**. These are Glaive's exact failure modes (`generate_password`/`calculate_tip`
    dominance; "Can you order a pizza for me?" as the sole refusal trigger, which teaches
    pizza-refusal rather than relevance).
16. **Temporal coherence** — each row carries a top-level `as_of`; every date-typed argument falls
    in a declared window around it.

**Not machine-checkable — do not claim it:** whether the chosen tool is the one a competent human
would choose (mitigated *constructively*: generate the query **from** the chosen schema so the label
is true by construction, and reject rows where ≥2 tools validate the same `arguments`); whether a
user turn is natural; whether a tool *result* payload is realistic; whether a clarifying question
is the *best* one.

---

## 11. Mixing with general SFT

**tool-call : general = 1 : 1** for a full finetune (40,000 vs 40,000). **Floor: general ≥ 20%.**

- **Alopex (arXiv 2411.05209)** is on-point and in our size class: function-calling-only finetuning
  caused **significant decreases on MMLU, GSM8K, ARC, HellaSwag, Winogrande, TruthfulQA** on
  Gemma-2B, Qwen1.5-1.8B, StableLM-2-1.6B, Fox-1-1.6B. Mixing at **1:1** mitigated it, some metrics
  ending better than before.
- A math-reasoning sweep (1:1 → 15:1) found **~6.2% general data already regularises**;
  practitioner guidance is 5–20% rehearsal. Those license the 20% **floor**, not lower — different
  task.
- **ToolACE's "negligible degradation" is LoRA-confounded** (rank 16, alpha 32) and structurally
  limits drift. A full finetune must not inherit that optimism.
- Mechanism, directly observed: tool-only SFT made **invocation rates rise sharply** while
  TriviaQA/GSM8K/NQ-Open fell; mixed data beat tool-only on every dataset.

Three things beyond the number: **rehearse the real corpus** (Alopex used tiny-textbooks only
because pretraining data was not public — ours is, and `src/scripts/train/sft/` is already wired to
it); **separate "tool-call-happy" from "dumber"** by running general benchmarks with **no tools in
context at all**; and **do not bake the ratio into the bytes** — publish 40,000 rows and let the
trainer's mixture weights set it. Since 0.2.0 has no labels, read-side mixing is by **object-path
glob**, which is the concrete reason §4's two-level path matters and why abstention has its own
path cells.

**MoE caveat:** there is **no MoE-vs-dense comparison at matched active parameters** for tool
calling and **no MoE-specific tool-call SFT guidance at all**. Both xLAM MoEs are far from ~4B
active. Log **per-expert routing entropy on tool-call vs general batches** and treat it as
something to measure, not inherit.

---

## 12. Consumption — nothing in this repo can read our `.jsonl`

This blocks *use*, not publishing, and it is worth knowing before anyone plans a run.

- **There is no `.jsonl` reader under `src/olmo_core/`.** Verified: the only matches are path
  strings inside data-mix files that point at `.npy`. Everything starts from pre-tokenized arrays.
- **The consumption side is fully wired.** `label_mask_paths` on `numpy_dataset.py:406`; the masked
  fill lives in `src/olmo_core/data/utils.py` `get_labels()`; `src/scripts/train/sft/Olmo-2-7B-SFT.py`
  shows the working configuration.
- **The producer is external.** `src/scripts/train/sft/README.md:19-61` points at open-instruct's
  `scripts/data/convert_sft_data_for_olmocore.py`, run with `--chat_template_name` and a tokenizer
  that carries the template.
- **Cheapest path: extend nothing.** Emit the two flat arrays offline and point
  `NumpyPackedFSLDatasetConfig` at them, exactly as the 7B SFT script does.
- **One trap:** packing finds document boundaries by **EOS**, so emit exactly **one EOS per
  conversation**. A converter emitting EOS per turn corrupts packing silently. Readers also memmap
  from byte 0 and derive counts from file size, so those arrays must be **headerless** despite a
  `.npy` extension.

---

## 13. Naming

Machine-validated with `validate_dataset_id`:

| Candidate | Verdict |
| --- | --- |
| **`sft/tool-call-single-turn`** | PASS — **recommended**: the sibling it must be distinguished from is the deferred multi-turn set, so "single-turn" is the load-bearing axis |
| `sft/tool-call-general-edu` | PASS |
| `sft/function-call-abstention-mix` | PASS |
| `eval/tool-call-heldout-apis` | PASS (but see §9 — no registered profile for eval *items*) |

Rejected for reference: `sft/toolcall-v2` → *"word 2 ('v2') … is a version token"*.

---

## 14. The `publish()` call

```python
import datetime
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

publish(
    "data/tool-call/",                                  # gitignored via /data/
    dataset_id="sft/tool-call-single-turn",
    purpose="Single-turn tool-call SFT conversations, general + edu tool inventories with a "
            "held-out API split, to teach function calling to the ~4B-active OLMo MoE",
    profile="sft-conversations/v1",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    group_meta={
        "conversations": {
            "record_schema": {"type": "object", "required": ["messages"]},
            "coverage": "partition",
            "partitions": [
                {"name": "train",   "by": "path", "glob": "train-*.jsonl",   "rows": 36000},
                {"name": "heldout", "by": "path", "glob": "heldout-*.jsonl", "rows": 4000},
            ],
            "dedup":   {"method": "sha256-of-role-and-content", "scope": "within-group"},
            "leakage": {"method": "train-heldout-key-intersection", "max_leakage": 0,
                        "carved_by": "held-out tool schemas + query templates, before generation"},
        }
    },
    sources=[...],   # ToolACE, Hermes — record scope honestly
    license={...},
    about="...",     # backfillable; never block a publish on it
)
```

No `tokenizer=`: this is an `sft` group of raw `.jsonl`, not packed shards, so the vocab check that
requires a tokenizer never runs on this profile.

We write to `s3://edullm-landing` (14-day expiry); the validator promotes into `s3://edullm-data`,
which nobody here can write. On refusal, read the `_REJECTED.json` beside the upload.

---

## 15. Open questions

| # | Question | Blocks |
| --- | --- | --- |
| 1 | **Does `allenai/dolma-2-tokenizer-olmo-3-instruct-final` already define tool-call delimiters, and are the `allenai/olmo-toolu-*` mixes usable?** (§8) | §3 and §7 conventions. Highest-value unknown: matching OLMo's own format beats inventing one |
| 2 | Held-out schema list + fraction, **and** the held-out query-template bank (§5) | **Generation start** |
| 3 | **xLAM licensing** — `cc-by-4.0` tag vs research-only prose. Needs a human with authority | Whether the total can be 67,500 instead of 40,000; ~1 week of reformat work |
| 4 | ToolACE single-turn (2-message) fraction — UNVERIFIED, cheap to count | The reformat vs synthesis budget in §6 |
| 5 | Post-training **sequence length** | §10 gate 11, and max tools per row |
| 6 | Who writes the JSONL → `(tokens, label_mask)` producer, in-repo or offline (§12) | **Any use** of the dataset; not the publish |
| 7 | Does a dolma2 control-token registry get written, and what are ids 100258–100276? | The reserved-token option only |
| 8 | Does the ~10% abstention optimum move at 4B active? | Nothing now — re-weightable by path glob. Best ablation candidate |
