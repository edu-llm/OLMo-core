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

## 3. Record format — OLMo 3's convention, inlined into `content`

**Superseded 2026-08-08.** The first draft invented `<tools>`/`<tool_call>`/`<tool_response>` with
name-first JSON calls. OLMo 3 already has a convention, it ships as **single token ids**, and we
adopt it verbatim. Retrieved and independently confirmed from
`allenai/olmo-3-tokenizer-instruct-dev` (see §7).

Roles `system` · `user` · `assistant` · **`environment`** — we enforce this set ourselves, since
the profile does not. **Tool results come back on `environment`, not `tool`.** The Instruct chat
template aliases `tool` → `environment`; the Think template has no `tool` branch and **silently
drops** such messages. Emit `environment`.

| Element | Exact bytes | Token id |
| --- | --- | --- |
| Schema block, in `system` `content` | `<functions>` + **single-line** JSON array + `</functions>` | 100266 / 100267 |
| Call block, in `assistant` `content` | `<function_calls>` + Pythonic call(s) + `</function_calls>` | 100268 / 100269 |
| Result, in `environment` `content` | **no wrapper at all** — raw content | — |

- **Calls are Pythonic, not JSON**: `name(k="v", k2=3)`, with argument *values* individually
  JSON-encoded. Verbatim from a real row: `weather.forecast_weather_api(q="Paris", days=5)`.
- **Parallel calls are ONE block**, calls joined by a bare `\n` inside it — not two blocks.
- **No newlines** immediately inside either tag, and nothing after `</function_calls>`.
- **One leading space before `<functions>`** in system content. AI2's own training bytes went
  through the legacy `message['functions']` path, which emits `' <functions>'`; the row-level
  `tools` path emits no separator. INFERRED from template + schema — settle it with §15 Q1.
- **Abstention** = ordinary prose with **zero** occurrences of `<function_calls>`. No positive
  marker; that would be a second thing to get wrong.
- **No call ids.** Results correlate to calls **positionally** only. This is a real expressiveness
  loss we accept — it also means the deferred multi-turn set cannot express interleaved or
  out-of-order results.

### We adopt OLMo's rendered bytes, NOT its row layout

This is the crux. AI2's rows park the call in a **sibling `function_calls` field** and the schema in
a sibling `functions` field, with **`content: null`** on the calling assistant turn. Confirmed
against a real row:

```
--- message 2 role= assistant
    content: <None>
    function_calls: weather.forecast_weather_api(q="Paris", days=5)
                    weather.forecast_weather_api(q="Madrid", days=5)
```

**That layout is exactly what our validator cannot see** (§2): `_dedup_key` hashes
`role \x1f content`, so two rows differing only in the sibling call collide, and `max_leakage: 0`
would refuse the whole publish. It is also the `content: null` trap that passes well-formedness
while teaching the model to emit nothing.

So we **inline**: the literal `<functions>…</functions>` goes inside `system` `content`, the literal
`<function_calls>…</function_calls>` inside `assistant` `content`, the raw result inside
`environment` `content`, and we emit **no `functions`, no `function_calls`, and no row-level
`tools`** field at all.

The chat template emits `content` verbatim and skips all three conditional branches, so **the
rendered token stream is byte-identical to AI2's** — the added-token trie still resolves the four
delimiters to single ids — while every payload sits inside `content` where the leakage key can
reach it. Reformatting a Dolci row is therefore a *lift*, not a translation.

Two consequences worth naming: `content` is never `null` on any row, which independently satisfies
§2's non-null requirement; and because the call is inside assistant `content`, **the call tokens are
trainable and the schema tokens are masked** (§12) — exactly what we want.

**(a) single call**

```json
{"messages":[
 {"role":"system","content":"You are a helpful function-calling AI assistant. … <functions>[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current conditions for a city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]</functions>"},
 {"role":"user","content":"weather in Boston?"},
 {"role":"assistant","content":"<function_calls>get_weather(city=\"Boston\")</function_calls>"}
]}
```

**(b) parallel calls** — one block, bare `\n` between calls:

```json
{"role":"assistant","content":"<function_calls>get_weather(city=\"Paris\")\nget_weather(city=\"Madrid\")</function_calls>"}
```

**(c) multi-turn with the result fed back** (deferred to v2 — and see §12, the converter currently
**cannot tokenize** ≥2 assistant turns; format fixed now so v2 agrees)

```json
{"messages":[
 {"role":"system","content":"… <functions>[…]</functions>"},
 {"role":"user","content":"weather in Boston?"},
 {"role":"assistant","content":"<function_calls>get_weather(city=\"Boston\")</function_calls>"},
 {"role":"environment","content":"{\"temp_f\":54}"},
 {"role":"assistant","content":"It's 54°F in Boston."}
]}
```

**(d) abstention** — tools offered, the applicable one deliberately absent; no `<function_calls>`
substring anywhere.

Every row must **end on an assistant turn** — the template only emits EOS there, and OLMo-core finds
document boundaries by EOS (§12).

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
| **TOTAL** | | **40,000** | **100** | see revised provenance below |

**Provenance, revised 2026-08-08** once `allenai/Dolci-Instruct-SFT-Tool-Use` was found public
(§8). Fresh synthesis drops from 67.75% → **51.0%**; roughly 8,000 rows of generate-and-verify work
becomes a filter-and-lift pass over data already in the exact target shape.

| Provenance | Rows | % | Where |
| --- | --- | --- | --- |
| Reformat — AI2 `Dolci-Instruct-SFT-Tool-Use` | **10,600** | 26.5 | `general/single-call` 4,000 · `general/multi-tool-select` 4,200 · `general/parallel-call` 2,400 |
| Reformat — `Team-ACE/ToolACE` (provenance hedge) | **2,000** | 5.0 | 800 / 800 / 400 across the same three |
| Derived (schema deletion, arg elision, distractor injection — now off *reformatted* positives) | **7,000** | 17.5 | `*/relevance-hard` 3,000 · `*/no-suitable-tool` 2,800 · `*/missing-args` 1,200 |
| Fresh synthesis | **20,400** | 51.0 | all 12,100 edu less its 2,100 derived → 10,000 · `general/nested-args` 5,200 · general top-up 5,200 |
| **TOTAL** | **40,000** | 100 | **31.5% reformat / 17.5% derived / 51.0% fresh** |

Deriving negatives from 227K human-curated AI2 positives is also strictly better than deriving them
from our own synthesis — it breaks the circularity of grading a synth generator against negatives
derived from that same generator.

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

## 7. Special tokens — nothing to reserve, nothing to resize

**Two corrections to earlier drafts of this section. Do not restore either.**

1. Draft 1 claimed "a control-token block already occupies slots at the top of that range." False on
   this branch — `grep -rn "control_tokens\|assert_control_tokens_fit\|reserved_special\|
   CONTROL_TOKEN" src/` returns **zero hits**. (That registry is on the unmerged
   `latent-superposition-module`.)
2. Draft 2 proposed reserving ids `100344`–`100351` for our own delimiters. **Obsolete and
   inferior.** OLMo already carved its four tool delimiters *inside the real vocab*, over previously
   unused `<|extra_id_*|>` slots. There is nothing to reserve.

Independently verified against the live tokenizers:

```
allenai/dolma-2-tokenizer-olmo-3-instruct-final  ->  HTTP 307 (rename alias)
    canonical: allenai/olmo-3-tokenizer-instruct-dev   sha 55f211dfda3974963b869e490617447045069a64

olmo-3-tokenizer-instruct-dev            dolma2-tokenizer
  100264 '<|im_start|>'      special=True
  100265 '<|im_end|>'        special=True
  100266 '<functions>'       special=False      100266 '<|extra_id_1|>'
  100267 '</functions>'      special=False      100267 '<|extra_id_2|>'
  100268 '<function_calls>'  special=False      100268 '<|extra_id_3|>'
  100269 '</function_calls>' special=False      100269 '<|extra_id_4|>'
  max added id: 100277
```

**The delimiters cost nothing.** Same `vocab_size` **100278** as plain dolma2 — the extras were
renumbered in place, so vocab did **not** grow. Any checkpoint pretrained with `dolma2-tokenizer` is
embedding-compatible byte-for-byte; `TokenizerConfig.dolma2()`
(`src/olmo_core/data/tokenizer.py:84-94`: `vocab_size=100278, eos=100257, pad=100277`,
`padded_vocab_size()` = **100352**) is already numerically correct for the instruct tokenizer. Only
the `identifier` string differs. **No resize, no new rows, no registry needed.**

For the record on the 74 embedding rows at `100278`–`100351`: they do exist (padding to a multiple
of 128) and remain unclaimed, but the tokenizer's max id is **100277**, so nothing can emit them
without adding tokens to the tokenizer itself. That option is now moot — carving inside the real
vocab, as OLMo did, is strictly better.

**`special: false` on the four tool tags is deliberate and load-bearing.** They are atomic
added-tokens (one id, never split) that **survive `skip_special_tokens=True`** on decode and are
absent from `all_special_tokens`. One consequence to inherit deliberately rather than by accident:
**`</function_calls>` (100269) is not a stop token.** The eval harness must register it explicitly,
or generated tool calls will not terminate.

`<think>` / `</think>` are **not tokens** in any repo checked — plain multi-token BPE text, absent
from every `added_tokens_decoder`. See §16.

---

## 8. Reusable upstream sources

The test is not "does the card say Apache-2.0" but "could the uploader grant model-training rights
at all."

| dataset | license | rows we can take | verdict |
| --- | --- | --- | --- |
| **`allenai/olmo-toolu-*`** (all five named in `src/scripts/train/sft/README.md:52-56`) | UNVERIFIED — unobtainable | **0** | **FORECLOSED — all five return HTTP 401.** Control: `allenai/tulu-3-sft-mixture` returns 200 from the same client, so the 401 is real. Private and nonexistent are indistinguishable from outside; neither is obtainable |
| **`allenai/Dolci-Instruct-SFT-Tool-Use`** | **ODC-BY stated in the description prose only — no `license:` key in frontmatter** | **227,579 rows** available (2.54 GB, public, ungated, `private:false gated:false`); we take **10,600** filtered | **TAKE — and it becomes the format reference.** The public, non-thinking equivalent of the private mixes: already AI2-native, already in OLMo's convention, needs zero call-syntax and zero schema conversion |
| **`allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA`** | **CONFLICT**: frontmatter `cc-by-sa-4.0` vs description "ODC-By" | **0** | **EXCLUDE for bytes.** 1,597 rows of deep-research browse trajectories, and a self-contradictory licence is the worst provenance case for an open model. See §16 |
| **Team-ACE/ToolACE** | `apache-2.0` | **2,000** (down from ~4,000) | **TAKE — as a provenance hedge**, so 26.5% of the corpus does not come from one source and the ≤2%-per-function cap has something to bite on |
| **NousResearch/hermes-function-calling-v1** | `apache-2.0` | **0** | **DROPPED.** Its only unique value was being "the only upstream already in our target shape." Once the target shape is OLMo's, Dolci holds that role with 120× the rows and none of Hermes' doubled-escaping or key-order repair |
| **Salesforce/xlam-function-calling-60k** | `cc-by-4.0` **tag** vs gated "research purposes only" **prose** | **0 today** | **BLOCKED** — needs a human decision, not more research |
| **ToolBench / ToolLLM** · **glaiveai/glaive-function-calling-v2** | see cards | — | **EXCLUDE** — dead endpoints and hallucinated APIs; quality |

**Reformatting Dolci is a lift, not a translation.** Rows are already `messages` with `functions` as
a JSON string and Pythonic calls; we move those two sibling fields into `content` per §3 and the
semantics are unchanged. Compare the repair list ToolACE still needs (below) — Dolci needs none of
it.

**Filters that must be applied to the Dolci slice** (in addition to §10):

- **Single-turn only.** The corpus is multi-turn (a real row runs system → user → assistant →
  environment → …). Our v1 scope is `system + user + assistant`, and §12's converter blocker makes
  that a hard requirement, not a preference.
- **Our own BFCL decontamination.** Only the *private* 200K mix is named `bfclv3-decontaminated`.
  The public cut's status is **UNVERIFIED** — run n-gram decontamination ourselves before claiming a
  BFCL number.
- **The ≤2% per-function cap** as a *filter on the slice*, not an assumption about it.
- **Role-set assertion.** An unrecognised role emits nothing from the template and the row is then
  silently deleted downstream (§12). Assert `{system,user,assistant,environment}` ourselves.
- **No `is_refusal` column exists** on the Instruct cut (only the 1,597-row thinking set has one),
  so its abstention content is UNVERIFIED and cannot be assumed. All 4,000 abstention rows stay
  derived by our own deletion/elision.

ToolACE repairs still required on its 2,000: drop empty `{"from":"assistant","value":""}` rows;
repair find-replace corruption in API **names and param names** (`valistring`←`valid`,
`start_string`←`date`); convert bracket calls `[Quotes by Keywords(word="inspiration")]`; parse tool
specs out of the free-text `system` string; classify the four modes ourselves; fix temporal
incoherence.

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

- **There is no `.jsonl` reader under `src/olmo_core/`.** The only matches are path strings inside
  data-mix files that point at `.npy`. Everything starts from pre-tokenized arrays.
- **The consumption side is fully wired.** `label_mask_paths` on `numpy_dataset.py:406`; the masked
  fill in `src/olmo_core/data/utils.py` `get_labels()`; `src/scripts/train/sft/Olmo-3-7B-SFT.py`
  shows the working configuration (`token_ids_part_*.npy` + `labels_mask_*.npy`,
  `NumpyPackedFSLDatasetConfig`, `generate_doc_lengths=True`).
- **The producer is external** — open-instruct's `scripts/data/convert_sft_data_for_olmocore.py`
  (`src/scripts/train/sft/README.md:19-61`). Cheapest path: extend nothing, emit the arrays offline.

### Field deltas our `.jsonl` must satisfy

1. **The field must literally be `messages`.** `TokenizerConfig.sft_messages_key` exists and
   defaults to `"messages"`, but the tokenize path **hardcodes `row["messages"]`** —
   `--sft_messages_key conversations` is silently ignored. Our profile already requires `messages`;
   just never rename it.
2. **`role` and `content` must be plain strings.** Templates concatenate `message['content']`
   directly, so no OpenAI content-block lists, and `null` content on a `system`/`user` turn raises
   `TypeError`. Our §3 inlining makes `content` non-null everywhere, which covers this.
3. **Roles must be in `{system, user, assistant, environment}`.** The templates have **no `else`
   branch — an unrecognised role emits nothing, silently.** A row whose only assistant turn had a
   typo'd role becomes all-`-100` labels and is then **deleted** by the row filter. Silent row loss;
   assert the role set ourselves.
4. **Do not supply `dataset_source`** (injected) and **do not supply a row-level `tools` column** —
   with an `olmo*` template it is normalised, passed, and then ignored. §3's inlining makes this moot.

### The EOS rule

Tokenization passes `add_special_tokens=False`, so every special token comes from the template.
A non-final assistant turn closes with `<|im_end|>\n`; the **final** assistant turn closes with
`eos_token` = `<|endoftext|>` (**100257**), *not* `<|im_end|>`. If a conversation ends on `user` or
`environment`, **no EOS is emitted at all** — and OLMo-core finds document boundaries by EOS, so
packing would corrupt silently. **Every row must end on an assistant turn.** Ours do.

### The multi-turn blocker — and why v1 is safe

Label masking is offset-mapping based: labels start fully `-100`, and for each assistant message `i`
a trainable span is derived from `apply_chat_template(messages[:i], add_generation_prompt=True)`
versus `apply_chat_template(messages[:i+1])`. The converter **raises** if the render is not
prefix-stable. For a **non-final** assistant turn, the sub-render makes that turn `loop.last` → it
emits `eos_token`, while the full render has `<|im_end|>\n` at the same position. So prefix-stability
holds **only if `eos_token == "<|im_end|>"`** — and here eos is `<|endoftext|>`.

**INFERRED, not executed:** any conversation with ≥2 assistant turns should fail with *"the rendered
conversation is not prefix-stable"*. Our v1 is `system + user + assistant` — exactly one assistant
turn, and it is final, so v1 is safe. **The deferred multi-turn set is blocked on this, and it is a
converter bug our data format cannot dodge.** Confirm empirically before planning v2 (§15 Q6).

### What the mask means for us

System, user, `environment`, the inlined tool schemas, and the assistant *header* are all `-100`.
Only assistant content plus its closing token is trainable. Because §3 inlines the call into
assistant `content`, **the call tokens are trainable and the schema tokens are not** — exactly right,
and a second dividend of the inlining decision.

**Outputs** are `token_ids_part_%04d.npy` (`uint32`), `labels_mask_part_%04d.npy` (`bool`), plus
doc-boundary CSVs and a tokenizer dir. **Those `.npy` files are not real `.npy`** — written via
`np.memmap(mode="w+")`, headerless flat binary. `np.load` fails on them; read with
`np.memmap`/`np.fromfile`, which is what `NumpyDatasetBase` does.

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
    sources=[...],   # AI2 Dolci (ODC-BY attribution) + ToolACE — record scope honestly
    license={...},
    about="...",     # backfillable; never block a publish on it
)
```

`sources[]` must name AI2 Dolci with its ODC-BY attribution, pin the tokenizer sha
(`55f211dfda3974963b869e490617447045069a64`), and state that our record layout is a **reformat** of
AI2's row schema into `sft-conversations/v1` (call inlined into `content`), **not** a verbatim copy.
There is no `vendored/v1` profile in the installed registry, which is precisely why a reformat rather
than a mirror is the only publishable form.

No `tokenizer=`: this is an `sft` group of raw `.jsonl`, not packed shards, so the vocab check that
requires a tokenizer never runs on this profile.

We write to `s3://edullm-landing` (14-day expiry); the validator promotes into `s3://edullm-data`,
which nobody here can write. On refusal, read the `_REJECTED.json` beside the upload.

---

## 15. Open questions

**Answered 2026-08-08** (was Q1): the OLMo-3 instruct tokenizer **does** define tool-call
delimiters — four single non-`special` ids `100266`–`100269`, carved in place over `<|extra_id_1..4|>`
at unchanged vocab 100278. `allenai/dolma-2-tokenizer-olmo-3-instruct-final` is a **307 alias** for
`allenai/olmo-3-tokenizer-instruct-dev`, sha `55f211dfda3974963b869e490617447045069a64` (pin it).
The `olmo-toolu-*` mixes are **all 401**, but the public non-thinking equivalent
`allenai/Dolci-Instruct-SFT-Tool-Use` (227,579 rows) replaces them.

| # | Question | Blocks |
| --- | --- | --- |
| 1 | **Render one Dolci row through `Olmo-3-7B-Instruct/chat_template.jinja` and diff against the same row inlined per §3.** Settles the leading-space question and proves byte-identity in one command | **The reformat pass. Do not start it before this passes** |
| 2 | Held-out schema list + fraction, **and** the held-out query-template bank (§5) | **Generation start** |
| 3 | **Dolci licence sign-off** — ODC-BY is stated in the description prose with **no `license:` key** in frontmatter. Weaker than a tagged licence; needs a human, same decision class as xLAM | All 10,600 reformat rows, i.e. 26.5% of the corpus |
| 4 | **xLAM licensing** — `cc-by-4.0` tag vs research-only prose | Whether the total can reach 67,500 |
| 5 | Post-training **sequence length** | §10 gate 11, and max tools per row |
| 6 | **Confirm the multi-turn prefix-stability failure** — one row with two assistant turns through the converter. Expected `ValueError: … not prefix-stable` (§12) | The deferred multi-turn set; nothing in v1 |
| 7 | Which chat template the tokenization run uses. The README's `olmo123` is a deliberate placeholder that falls back to the tokenizer's own template; the *registry's* `olmo` template reportedly appends `" You do not currently have access to any functions. <functions></functions>"` when a system message carries no `functions` field — which would corrupt every one of our inlined rows. **UNVERIFIED, one-command check** | The tokenization run, not the publish |
| 8 | Who writes the JSONL → `(tokens, label_mask)` producer, in-repo or offline (§12) | **Any use** of the dataset; not the publish |
| 9 | Does the ~10% abstention optimum move at 4B active? | Nothing now — re-weightable by path glob. Best ablation candidate |

---

## 16. The thinking-trace question

Every private `olmo-toolu-*` mix is named `thinking`, and the one public thinking tool-use set is
`allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA`. We take **none** of it, deliberately.

- **`<think>`/`</think>` are not tokens** — plain multi-token BPE text in every repo checked, absent
  from every `added_tokens_decoder`. So traces live inside `content` as ordinary text and stripping
  them is a content-level span deletion: the delimiters, the `environment` role, the EOS rule and the
  label-mask spans are all untouched. Format-wise, adopting their data would **not** force us into
  reasoning mode.
- **But the evidence does not support paying for it.** The entire published BFCL evidence for OLMo 3
  is a single number: **7B Instruct = 49.8**, against Qwen 3 8B at 60.2, and RLVR moved it **+0.9**
  over SFT. **There is no BFCL number for any Think model at any size** — the Think tables have no
  Tool Use row. Paying a multiplier on target tokens to teach an inference-time behaviour we do not
  intend to serve on a non-reasoning ~4B-active model is not defensible.
- **Stripping has a real quality cost.** In a multi-step trajectory the trace often carries the
  derivation justifying specific argument values; delete it and the row is underdetermined — a bare
  call that does not follow from the visible prompt. Stripped rows need a re-verification pass (does
  every argument value appear in, or follow deterministically from, the surviving turns?), not a
  regex.
- **It would not serve the latent-reasoning / CODI work either**, and should not be planned as
  dual-purpose: 1,597 rows is the whole public supply; they are deep-research browse trajectories
  (`serper_google_webpage_search`, `\boxed{}` answers), a retrieval-agent shape rather than the
  deductive shape CODI distillation is evaluated on; and the licence is **self-contradictory**
  (`cc-by-sa-4.0` frontmatter vs ODC-BY prose) on that exact set.
- **Keep CODI and tool-call as separate `dataset_id`s.** A with-trace and without-trace version of
  the same row do *not* collide under `_dedup_key` (different `content`), so the validator would
  accept both in one dataset. That is a hash artifact, not a licence to do it.
