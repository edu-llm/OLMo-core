# Dataset design: tool-call SFT for OLMo MoE (~4B active)

Companion to [`prd.md`](prd.md). This is the *shape* document — what the bytes look like, where
they sit, and what the validator does to them. Log changes in [`progress.md`](progress.md).

Split out for length: **[`tool-inventories.md`](tool-inventories.md)** (the 64 tool schemas, per
domain, with held-out siblings and the arithmetic/web-search design decisions) and
**[`pedagogy.md`](pedagogy.md)** (how "educational" shows up in the bytes — pedagogy tools vs prose
vs learning-science knowledge, the principle taxonomy, and the myth negatives).

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
- **One leading space before `<functions>`** in system content. Read from the template source, not
  inferred: the per-message `message['functions']` path emits `' <functions>'`, the row-level `tools`
  path emits `'<functions>'` with no separator. AI2's own training bytes went through the former, so
  we match the former. Proven byte-identical by
  [`verify/verify_render_identity.py`](verify/verify_render_identity.py).
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

**Revised 2026-08-08** for the three mandated must-haves — arithmetic tools, web search, pedagogy.
Only the two *vocabularies* changed; **depth is untouched at exactly two levels.**

- **`<domain>` ∈ `general` | `arithmetic` | `web-search` | `pedagogy`.** `edu` is **renamed** to
  `pedagogy`, not added alongside it. Each must-have sits at level 1, so each is independently
  trainable, ablatable and measurable by glob.
- **`<category>` is a CAPABILITY band, not a tool family** — now **8 values, identical across all
  four domains**, so the 32-cell grid is a real cross-tab with no holes: `single-call` ·
  `multi-tool-select` · `parallel-call` · `nested-args` · `relevance-hard` · `no-suitable-tool` ·
  `missing-args` · **`answer-directly`**. (The first draft used `edu/gradebook` — tool family in the
  path. That was a mistake; tool family rides as a top-level row field and is **not** sliceable.)

### Why the eighth category is forced, not cosmetic

`no-suitable-tool` is BFCL `irrelevance` — the gold function was **deleted**, so correctness is
constructive. **`answer-directly` is the opposite construction:** the tool is *present and
applicable-looking*, and calling it is still wrong because the answer is settled parametric
knowledge. That is the whole fresh-vs-parametric boundary, and it is precisely what the mandate
asks for — the model should *know* learning science, not search for it. Folded into
`no-suitable-tool` it becomes unmeasurable and unreweightable.

### Why the inventory axis belongs at level 1

Globs work either way (`fnmatch` matches basename *or* full path), so the argument is operational:
**the tool inventory is a property of the domain, not the capability.** Held-out schema carving,
gate 5 (no function name may carry two schemas corpus-wide), gate 12 (schema-pool disjointness) and
new gate 34 (domain ↔ gold-tool agreement) are all inventory-scoped. With inventory at level 1 a
schema-pool audit is a per-directory operation; with capability at level 1 every directory mixes all
four inventories and the audit can only go through a row field.

*Rejected:* a five-value set adding `learning-science` as its own domain — it would be an
almost-pure-prose domain whose `parallel-call` and `nested-args` cells are empty or nonsensical,
destroying the identical-vocabulary property that makes the cross-tab interpretable.

### IRREVERSIBLE labelling rule

**A row's `<domain>` is the domain of the GOLD tool** — for abstention rows, of the deleted or
tempting tool. Not the topic of the user turn, not the union of the offered inventory. A
pedagogy-topic question answered by `web_search` is `web-search/single-call`. Quote this whenever a
domain total is quoted: **domain totals measure gold-tool domain, not inventory composition.**

### What spending the domain axis this way costs

- **Domain is now confounded with provenance.** All 9,600 reformat rows can only land in `general`;
  the three new domains are **0% reformat**. So any general-vs-pedagogy delta is partly a
  human-curated-vs-synthetic delta. Unfixable by path — mitigate with a `provenance` row field and
  **state the confound wherever a cross-domain number appears.** Biggest single cost.
- **The "general subtree alone sits at ToolACE's controlled 25k" claim dies.** `general` is 15,000.
  Honest replacement: **general + arithmetic + web-search = 29,000 ≥ 25,000**, because a calculator
  and a search tool *are* general-purpose tools. Say it that way or not at all.
- **The axis is dimensionally inhomogeneous** — `arithmetic` and `web-search` are roughly single
  families promoted to level 1; `pedagogy` holds six; `general` is a catch-all.
- **Pedagogy tools and pedagogy prose are not separately sliceable.** Partially recovered: since the
  dichotomy gate forbids prose-plus-call in one turn, *prose-only* pedagogy rows are definitionally
  abstention rows and land in `pedagogy/{no-suitable-tool,missing-args,answer-directly}`, which are.
- **Exactly two levels below the group.** 0.2.0 has no `labels_from_path` so a third level would
  not raise *today*, but the newer pipeline derives exactly two label levels and **raises on a
  third**. Two levels keeps us forward-compatible with a feature we would otherwise have to
  republish every byte to adopt.

**Renaming `edu` → `pedagogy` is free right now and never again** — no bytes are published, and path
is hashed into `manifest_sha256`.
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

**Heldout is carved by held-out tool schema, before generation.** A random row split leaves the same
functions on both sides and measures memorization. Carve rates: **≈9.4% on non-abstention cells,
15% on abstention cells** — the uplift exists because abstention cells are the smallest *and* the
ones we must report; without it `no-suitable-tool` heldout falls below BFCL `irrelevance`'s n=240.
**64 authored schemas, 10 held out (15.6%)**, plus ~12% of the inherited `general` pool
(data-dependent, UNVERIFIED until the filter runs) — see
[`tool-inventories.md`](tool-inventories.md).

**Carve rule: hold out the sibling, not the orphan.** A held-out orphan measures nothing; a held-out
sibling of a trained tool measures schema generalization. Worked examples: `differentiate_expression`
held out against a trained `integrate_expression`; `openai_web_search` (filters nested) against a
trained `web_search_configured` (filters flat); `bulk_update_grades` against a trained
`grade_submission_with_rubric`.

**Schema carving alone is not sufficient.** If heldout user turns come from the same template bank
as train, heldout measures template recall with a new function name pasted in. **Carve the query
template bank alongside the schema pool**, or generate heldout queries with a different
generator/prompt. Nothing in the pipeline can see this.

**Two domains cannot carve by schema at all.** `calculator` and `web_search` are single dominant
tools that must be in train, so for those cells **"heldout measures schema generalization" is false
and must not be written.** Substitute axes, which measure a *different* claim and must be named as
such: arithmetic uses an **operand-magnitude band** (train ≤4-digit, heldout 5–7-digit) and an
operator-depth band; web-search uses a held-out **entity bank** plus held-out **parameter
combinations**.

So `carved_by` in the publish call names **four** axes, not one: held-out tool schemas, the query
template bank, the operand-magnitude band, and the entity bank.

---

## 6. Composition — 40,000 rows across a 32-cell grid

**Rules:** total 40,000 · train 36,000 / heldout 4,000 · abstention
(`no-suitable-tool` + `missing-args` + `answer-directly`) exactly **4,000 = 10.00%** · heldout carve
**≈9.4% on non-abstention cells, 15% on abstention cells** (the uplift exists because abstention
cells are the smallest *and* the ones we must report — without it `no-suitable-tool` heldout falls
below BFCL `irrelevance`'s n=240).

| path (`conversations/…`) | rows | % | train | heldout | provenance |
| --- | --- | --- | --- | --- | --- |
| `general/single-call` | 3,600 | 9.00 | 3,260 | 340 | reformat 2,600 (Dolci 2,200 / ToolACE 400) + fresh 1,000 |
| `general/multi-tool-select` | 4,000 | 10.00 | 3,620 | 380 | reformat 3,400 (Dolci 2,800 / ToolACE 600) + fresh 600 |
| `general/parallel-call` | 2,200 | 5.50 | 1,990 | 210 | reformat 1,700 (Dolci 1,400 / ToolACE 300) + fresh 500 |
| `general/nested-args` | 2,800 | 7.00 | 2,540 | 260 | reformat 1,900 (Dolci 1,700 / ToolACE 200) + fresh 900 |
| `general/relevance-hard` | 1,200 | 3.00 | 1,090 | 110 | derived (distractor injection off reformatted positives) |
| `general/no-suitable-tool` | 700 | 1.75 | 595 | 105 | derived (Hammer schema deletion) |
| `general/missing-args` | 300 | 0.75 | 255 | 45 | derived (arg elision) |
| `general/answer-directly` | 200 | 0.50 | 170 | 30 | fresh-curated (conversational / creative only) |
| **general** | **15,000** | **37.50** | **13,520** | **1,480** | reformat 9,600 / derived 2,200 / fresh 3,200 |
| `arithmetic/single-call` | 2,400 | 6.00 | 2,175 | 225 | fresh, schema-first, **value-executed** |
| `arithmetic/multi-tool-select` | 1,300 | 3.25 | 1,180 | 120 | fresh |
| `arithmetic/parallel-call` | 700 | 1.75 | 635 | 65 | fresh |
| `arithmetic/nested-args` | 1,600 | 4.00 | 1,450 | 150 | fresh, value-executed (arrays / precision / units / large operands) |
| `arithmetic/relevance-hard` | 450 | 1.13 | 405 | 45 | derived |
| `arithmetic/no-suitable-tool` | 250 | 0.63 | 210 | 40 | derived |
| `arithmetic/missing-args` | 100 | 0.25 | 85 | 15 | derived |
| `arithmetic/answer-directly` | 200 | 0.50 | 170 | 30 | fresh-curated (mental-arithmetic band) |
| **arithmetic** | **7,000** | **17.50** | **6,310** | **690** | derived 800 / fresh 6,200 |
| `web-search/single-call` | 1,900 | 4.75 | 1,720 | 180 | fresh (query built *from* a target document) |
| `web-search/multi-tool-select` | 1,300 | 3.25 | 1,180 | 120 | fresh |
| `web-search/parallel-call` | 800 | 2.00 | 725 | 75 | fresh (multi-entity comparison) |
| `web-search/nested-args` | 1,500 | 3.75 | 1,360 | 140 | fresh (`filters`, `user_location`, date filters) |
| `web-search/relevance-hard` | 500 | 1.25 | 455 | 45 | derived (provider-spelling near-misses) |
| `web-search/no-suitable-tool` | 350 | 0.88 | 300 | 50 | derived (all retrieval tools removed) |
| `web-search/missing-args` | 150 | 0.38 | 125 | 25 | derived |
| `web-search/answer-directly` | 500 | 1.25 | 420 | 80 | fresh-curated (parametric fact bank, k=8 consistency) |
| **web-search** | **7,000** | **17.50** | **6,285** | **715** | derived 1,000 / fresh 6,000 |
| `pedagogy/single-call` | 2,200 | 5.50 | 1,995 | 205 | fresh |
| `pedagogy/multi-tool-select` | 2,400 | 6.00 | 2,175 | 225 | fresh |
| `pedagogy/parallel-call` | 900 | 2.25 | 815 | 85 | fresh |
| `pedagogy/nested-args` | 3,200 | 8.00 | 2,890 | 310 | fresh (`rubric_assessment`, `grade_data`, Caliper envelopes) |
| `pedagogy/relevance-hard` | 1,050 | 2.63 | 950 | 100 | derived |
| `pedagogy/no-suitable-tool` | 400 | 1.00 | 340 | 60 | derived |
| `pedagogy/missing-args` | 250 | 0.63 | 210 | 40 | derived (learner state unspecified) |
| `pedagogy/answer-directly` | 600 | 1.50 | 510 | 90 | fresh-curated (myth / low-utility / contested bank) |
| **pedagogy** | **11,000** | **27.50** | **9,885** | **1,115** | derived 1,700 / fresh 9,300 |
| **TOTAL** | **40,000** | **100.00** | **36,000** | **4,000** | |

**Category totals (rows / heldout):** single-call 10,100 / 950 · multi-tool-select 9,000 / 845 ·
parallel-call 4,600 / 435 · nested-args 9,100 / 860 · relevance-hard 3,200 / 300 ·
no-suitable-tool 1,700 / 255 · missing-args 800 / 125 · answer-directly 1,500 / 230.

### Provenance roll-up — revised 2026-08-08

| Provenance | Rows | % |
| --- | --- | --- |
| **Reformat** from licence-verified upstreams (§8) | **11,500** | **28.75** |
| **Derived** (schema deletion, arg elision, distractor injection, DSL interpretation) | **12,000** | **30.00** |
| **Fresh** (of which ~3,800 is programmatic, not model-generated) | **16,500** | **41.25** |

Two things changed at once. The incumbent 8,100-row Dolci slice is **dropped** — it has no
frontmatter licence tag at all (§8) — and the hunt found a **verified-single-turn, tagged pool of
56,902 rows** for general alone (`Synth-APIGen` 49,402 + `xlam-irrelevance` 7,500), which is 5.4× what
we allocate. So reformat *rises* from 24% → 28.75% while dropping the one source that needed a human
sign-off. Fresh falls from 61.75% → **41.25%**.

**Reformat is capped near 29%, and the cap is structural, not a dial we left low:**

1. **Wire-format eligibility.** v1 takes `system + user + assistant`, exactly one assistant turn, last.
   Toucan, ToolACE, and every tutoring-dialogue set are multi-turn by construction. Anything salvaged
   by lifting a first exchange is **derived**, not reformat, because we chose where to cut.
2. **Licence.** In three of four domains the best content is unusable — see §8's reject list. The
   single most painful: `Eedi/…-Tutoring-Dialogues-2k` is the best pedagogy content in existence and
   is `cc-by-nc-4.0`.
3. **Domain coverage.** Reformat needs the upstream to *already contain a tool call*. True for
   general, true in a foreign syntax for arithmetic, **false by construction for web-search and
   pedagogy**. That alone pins 18,000 of 40,000 rows at 0% reformat.
4. **Held-out hygiene.** Every reformatted row imports an upstream schema, so pushing reformat higher
   shrinks the pool we can credibly hold out — the generalization claim degrades exactly as the reuse
   number improves.

**The general plan no longer depends on any UNVERIFIED yield.** `Synth-APIGen` and
`xlam-irrelevance` are both 100% single-turn and tagged. Toucan is load-bearing for one thing only —
`nested-args`, where Synth-APIGen's flatter schemas will not stretch — and if its single-turn yield
disappoints we derive those rows from first exchanges instead, moving rows from reformat to derived
without changing any domain total.

### Per-domain totals, justified

`general` 15,000 keeps the whole reformat slice plus enough fresh top-up that no single upstream
exceeds ~54% of the domain. `arithmetic` and `web-search` at 7,000 each have the narrowest tool
surface, so rows buy less new schema coverage and go instead into argument fidelity and the
abstention boundary — **7,000 is a judgement, not evidence: there is no published data-quantity
sweep for a multi-domain function-calling split. UNVERIFIED.** Because domain is level 1, reweighting
is a glob change with no regeneration, which makes this the cheapest thing in the design to be wrong
about. `pedagogy` 11,000 is the largest non-general domain because it carries 20 schemas with the
deepest nesting *and* the learning-science surface.

**`nested-args` = 9,100 (22.75%)**, up from 8,000, because the two new domains contribute the deepest
*real* nesting in the corpus (Canvas `rubric_assessment` criterion maps, Perplexity's 13 constrained
params, Anthropic `user_location`) and because argument-value fidelity is the measured weak slot at
~4B (Hammer-4b AST simple **62.58** vs multiple 77.72; Exec simple **67.79**).

**Abstention stays exactly 10.00%**, now split three ways: `no-suitable-tool` 1,700 +
`missing-args` 800 + `answer-directly` 1,500. Note train abstention is 9.42% while *total* is
10.00%, a consequence of the 15% carve uplift. The Hammer evidence for ~10% was a **two-way**
call/no-call sweep at 1.5B; a three-way split at 4B active is a different question — UNVERIFIED, and
still the best ablation candidate, re-weightable by three globs.

**Smallest cell is `arithmetic/missing-args` at 100 rows / 15 heldout** — below the 200-row
reporting floor. Pool it into a domain-pooled abstention column; never report it alone.

Deriving negatives from human-curated AI2 positives rather than from our own synthesis also breaks
the circularity of grading a synth generator against negatives derived from that same generator.

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

**Rewritten 2026-08-08 after a wider hunt.** The standard: a **frontmatter `license:` tag** is
acceptable; prose-only needs a human read; **no tag at all is a reject**; NC and share-alike are
rejects because copyleft propagates onto the whole mix; gated research-only is a reject.

All rows below verified live — [`verify/verify_sources.py`](verify/verify_sources.py).

### The incumbent is dropped

```
allenai/Dolci-Instruct-SFT-Tool-Use    public   <NO TAG>   227,579   REJECT (no frontmatter tag)
```

Earlier drafts had this at **8,100 rows — 20.25% of the whole dataset** — pending a human licence
sign-off.

**The full picture, since this is a judgment call and should stay reversible.** The card's frontmatter
carries `dataset_info` but **no `license:` key**. The licence appears only in the body:

> This dataset is licensed under ODC-BY. It is intended for research and educational use in
> accordance with Ai2's Responsible Use Guidelines.

Both halves of that matter. **ODC-BY is permissive** — attribution, essentially — and "intended for
research and **educational** use" arguably *describes* an educational organisation rather than
excluding one. So there is a real argument that Dolci is usable here.

**Why we dropped it anyway, and it is not the licence on its own:**

1. Nothing machine-readable. A prose licence plus a narrowing phrase of unclear legal weight is the
   weakest form of the claim, and this would have been our single largest source.
2. **A fully-tagged alternative covers the same ground better.** `argilla/Synth-APIGen-v0.1` is
   `apache-2.0` as a real tag, 49,402 rows, **100% single-turn**.
3. **Measured: only ~10% of Dolci rows are 3-message**, so its 227,579 headline is really ~22,750
   usable rows for v1 — *less* than Synth-APIGen, before any licence question.

So the sign-off went from "worth asking" to "asking for permission we no longer need." **If someone
decides to use Dolci regardless** — it is AI2's own tool-use data, `dataset_source` names the
BFCL-v3-decontaminated mix, and 227k rows is real — that is a defensible position; it just puts the
sign-off back on the list. Its `-SA` sibling stays out either way: `cc-by-sa-4.0` share-alike would
propagate copyleft onto the whole mix.

### Verified usable

| id | tag | rows | what we take | cost | domain |
| --- | --- | --- | --- | --- | --- |
| `argilla/Synth-APIGen-v0.1` | **apache-2.0** | 49,402 | `query`+`tools`+`answers` → single-call, multi-tool-select, parallel-call. **100% single-turn**, clean-room APIGen, `hash_id` for dedup | low | general |
| `MadeAgents/xlam-irrelevance-7.5k` | **cc-by-4.0** | 7,500 | all of it → relevance-hard, no-suitable-tool. **100% single-turn**, and exactly our two hardest categories | low | general |
| `nvidia/When2Call` | **cc-by-4.0** | 27,952 | **train split only** → answer-directly, no-suitable-tool | med | general |
| `Agent-Ark/Toucan-1.5M` | **apache-2.0** (tag *and* prose) | 1,646,546 | rich nested MCP schemas → nested-args. Single-turn fraction **UNVERIFIED** | med | general |
| `Team-ACE/ToolACE` | **apache-2.0** | 11,300 | first exchange of multi-turn rows. Keep at 1,500; do not grow | med | general |
| `MU-NLPC/Calc-gsm8k` | **mit** | 7,273 train | `<gadget id="calculator">expr</gadget>` → our `<function_calls>`. **The only real reformat source in arithmetic** | low | arithmetic |
| `openai/gsm8k` | **mit** | 7,473 train | `<<a op b=c>>` spans → ordered calls with genuine operand *dependency chains*, plus natural phrasing | low | arithmetic |
| `allenai/math_qa` | **apache-2.0** | 29,837 train | `linear_formula` operator DSL → multi-call programs. Cost is a ~60-op interpreter | med | arithmetic |
| `ChilleD/SVAMP` | **mit** | 700 train | `Equation` → single-op wrap; phrasing diversity | low | arithmetic |
| `aialt/RetrievalQA` | **mit** | 2,785 | `param_knowledge_answerable ∈ {0,1}` — the **only** clean-licensed explicit search-vs-memory label that exists | med | web-search |
| `xanhho/2WikiMultihopQA` | **apache-2.0** | UNVERIFIED | gold supporting paragraphs → per-hop sub-question search calls | med | web-search |
| `ChilleD/StrategyQA` | **mit** | 1,603 train | implicit decomposition → query | med | web-search |
| `spacemanidol/orcas` | **mit** | 10,405,341 | real Bing query **phrasing distribution** — taken as a prior, not as rows, so it contributes 0 to the provenance ratio | low | web-search |
| `eth-nlped/mathdial` | **cc-by-4.0** | 2,262 train | real tutor–student dialogue → learner-state phrasing and misconception surface forms | med | pedagogy |
| `allenai/mathfish` | **odc-by** | UNVERIFIED | **CCSS codes + grade/unit/lesson metadata ONLY** — never `problem_activity_html` or `text` (those bodies are compiled from Illustrative Mathematics / Fishtank, and are live curriculum content) | med | pedagogy |
| `allenai/tutormoments-preview` | **cc-by-4.0** | 10,053 | `transcripts` + `annotations` configs only. **Exclude `moments`, `ground_truth`, `benchmark_520`** — a derived benchmark | med | pedagogy |

### Rejected, with the reason visible

- **No frontmatter tag:** `allenai/Dolci-Instruct-SFT-Tool-Use` (227,579) · `MU-NLPC/Calc-X`
  (319,169) · `Calc-X-big-numbers` (224,409 — conceptually the closest thing to what we want) ·
  MuSiQue · TutorChat · `VityaVitalich/adaptive_rag_*` (which has literally our ideal schema) ·
  `THUDM/AgentInstruct` · `driaforall/pythonic-function-calling` · `interstellarninja/tool-calls-multiturn`.
- **NC:** `Eedi/Question-Anchored-Tutoring-Dialogues-2k` (`cc-by-nc-4.0`, 79,574 — **the best pedagogy
  content in existence**, 10,857 diagnostic questions with all four real distractors, and unusable in
  an open release. Do not revisit) · `Salesforce/APIGen-MT-5k` · `EleutherAI/asdiv` · DrawEduMath.
- **Share-alike:** `allenai/Dolci-…-SA` · HotpotQA (90,447) · Natural Questions · BeIR-msmarco.
- **Gated:** `Salesforce/xlam-function-calling-60k` (`gated: auto` + research-only prose that
  contradicts its CC-BY tag) · `Trelis/*` · `alucent/mirror-*`.
- **Relabels that do not cure the parent:** `argilla/apigen-function-calling` (CC-BY over an
  xlam-60k merge) · `Locutusque/function-calling-chatml` (glaive-v2) · `internlm/Agent-FLAN`.
- **Reasoning traces** (we are not training a reasoning model):
  `interstellarninja/hermes_reasoning_tool_use` (51,004) and `tool-use-relevance-reasoning` (15,218).
- **Needs a human read:** `nvidia/OpenMathInstruct-1` (`license: other` = NVIDIA License; a grep
  found no NC clause, but `other` is not a tag we accept on faith; also ~114 GB and `<llm-code>` is
  code-as-action, not a named-function call). v2 upside only.
- **Verified to contain no tool calls:** `nvidia/OpenMathInstruct-2` (13.97M, pure LaTeX CoT).
- **Do not exist** as public HF datasets: ToRA, MathCoder, ToolQA.

### Eval sets we must never ingest

BFCL is **eval-only**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, `BFCL_v3_*`, `BFCL_v4_*`
(including `BFCL_v4_web_search`), `MadeAgents/HammerBench`, `gorilla-llm/APIBench`,
`Nexusflow/NexusRaven_API_evaluation`, `google/frames-benchmark`, `allenai/mathfish-tasks`.

**Two sources we consume are themselves evals**, and taking them burns them for this model:
`aialt/RetrievalQA` (no train split at all) and `nvidia/When2Call` (train split only). Accept both —
RetrievalQA is the only clean-licensed search-vs-memory label there is — but **record it in the
dataset card.** Deciding this quietly is how a benchmark number becomes meaningless a year later.

### Decontamination must key on tool identity, not text

For a function-calling dataset the leak that matters is a repeated **API**, not a repeated question.
Build one canonical key per row — normalized `(function_name, sorted(param_name_set))` plus a MinHash
of the user turn — and run it against every BFCL v3/v4 category **and** our own held-out schema pool.
Text-only decontamination passes rows that teach the exact APIs BFCL scores.

Highest collision risk: `MadeAgents/xlam-irrelevance-7.5k` shares lineage with xlam-60k, which shares
provenance with BFCL simple/multiple, and Hammer trained on it. Also: **the arithmetic held-out set
must be programmatically generated, never drawn from GSM8K/MathQA/SVAMP** — those problems sit in
every base model's pretraining data, so a held-out slice from them measures memorization, not tool use.
Allocate each GSM8K problem **once**: `Calc-gsm8k` and `openai/gsm8k` are the same problems.

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

Heldout rows also carry a top-level **`answer_key`** with per-slot *sets* of acceptable values,
mirroring BFCL's `possible_answer/` — without it AST exact-match under-reports, because BFCL
explicitly allows multiple correct values per slot.

| our heldout shards (all 4 domains) | BFCL category | BFCL n | ours |
| --- | --- | --- | --- |
| `*/single-call/heldout-*` | `simple_python` | 400 | **950** |
| `*/multi-tool-select/heldout-*` | `multiple` / `live_multiple` | 200 / 1037 | **845** |
| `*/parallel-call/heldout-*` | `parallel` / `parallel_multiple` | 200 / 200 | **435** |
| `*/nested-args/heldout-*` | none — grades *into* AST arg checking | — | **860** |
| `*/relevance-hard/heldout-*` | `live_relevance` | ~41 (UNVERIFIED) | **300** |
| `*/no-suitable-tool/heldout-*` | `irrelevance` / `live_irrelevance` | 240 / 875 | **255** |
| `*/missing-args/heldout-*` | `multi_turn_miss_param` (multi-turn only) | 200 | **125** |
| `*/answer-directly/heldout-*` | **none** | — | **230** |
| **total** | | | **4,000** |

Sizing rule: **each category ≥ its BFCL counterpart's n** where one exists, so our confidence
interval is never worse than the number we compare against. Deliberately not replicated:
`live_parallel` at 16 cases, where one flip moves the column 6.25 pts.

**Two comparability regressions to state, not bury.** `multi-tool-select` at **845 < live_multiple's
1,037** — the "never worse CI" rule now fails for that one category; either accept the wider interval
or move ~200 rows in from `general/single-call`. `missing-args` at **125 < 200**, though
`multi_turn_miss_param` is multi-turn-only so the comparison was always notional. The win bought by
the 15% abstention carve: `no-suitable-tool` **255 ≥ 240** ✓.

**`answer-directly` has no BFCL counterpart.** Report it as its own column and **never fold it into
Hallucination** — it is a different construction from `irrelevance` (§4). Also:
`arithmetic/heldout-*` is the only slice with non-AST ground truth, so it is the only one that can
carry an Exec-style number.

Reporting rules that make the numbers comparable rather than merely similar:

- **Report `irrelevance` and `relevance` as a pair, never blended.** Irrelevance alone is trivially
  gamed: Deepseek-Coder-1.3B-Instruct scores 100.00 / 0.00 by never calling; ToolLLM-SFT scores
  4.41 / 100.00 by always calling. Target shape is a pair like ToolACE-8B (83.81 / 85.37).
- **Follow the post-CHANGELOG convention** — BFCL now excludes irrelevance/relevance from Non-Live
  and Live and reports them separately as Hallucination.
- **Grade AST-only and say so.** BFCL v4 comments out `exec_*`/`rest`/`sql` from default scoring,
  so AST-only is now the *same shape* as v4 default — we do not need live APIs to be comparable.
- **Pool any cell under 200 rows** before reporting. With 32 cells this now bites hard: 6 of the 12
  abstention cells are under the floor (smallest is `arithmetic/missing-args` at 100 rows / 15
  heldout). Report those domain-pooled, never alone.
- **Do not claim a BFCL overall.** v4's stated weighting (Agentic 40% + Multi-Turn 30% + Live 10% +
  Non-Live 10% + Hallucination 10%) conflicts with the leaderboard's "unweighted average" text, and
  this dataset touches **0%** of Agentic and Multi-Turn.

---

## 10. Generation + verification

No LLM judge in the write path; a judge's verdict is advisory and never decides admissibility.
Cheap gates first.

> **Gates 1–8 were corrected 2026-08-08.** They still named the first draft's invented
> `<tools>`/`<tool_call>` delimiters and its name-first JSON payload, all of which §3 superseded.
> Left as they were, every one of the 40,000 rows would have been mis-gated. Do not reintroduce the
> old strings.

**Per row, dropped if any fails:**

1. **Container** — line parses; `messages` non-empty; every message has a non-empty string `role` and
   a present `content`. We additionally require `content` to be a **non-empty string** (§2, since the
   profile accepts `null`) and `role ∈ {system, user, assistant, `**`environment`**`}` — **not
   `tool`** — because the profile enforces no role set and the Think template silently drops `tool`.
2. **Schema block** — exactly one `system` at index 0, exactly one `<functions>…</functions>`, whose
   body parses as a **single-line JSON array**; each `parameters` validates against the JSON Schema
   2020-12 metaschema.
3. **Assistant shape** — see gate 35: `prose? <function_calls>…</function_calls>` with nothing after,
   *or* zero occurrences of `<function_calls>`. Note OLMo uses **one** block with calls joined by a
   bare `\n` *inside* it, never multiple blocks.
4. **Call payload** — calls are **Pythonic, not JSON**. `ast.parse(body, mode="eval")` must yield a
   single `ast.Call` whose `func` is an `ast.Name`/`ast.Attribute`, with **keyword arguments only**
   (no positional), each value a literal.
   **Corrected 2026-08-09 — the obvious implementation is wrong.** Argument *values* are JSON, so a
   boolean serialises as `true`, not Python's `True`. `ast.parse` reads `true` as a **`Name` node**
   (a variable reference), and `ast.literal_eval` raises on it outright. A naive
   "`ast.literal_eval` each value" gate would therefore **reject every row carrying a boolean or
   null argument** — and those are everywhere in our schemas (`excuse`, `inclusive`,
   `peer_reviewed`, `open_now`, `keep_integer`, `grades_released`, `external_web_access`…).
   The parser must map `Name` nodes `true`/`false`/`null` to their values and recurse through lists
   and dicts. Implemented and tested as `parse_call` in
   [`src/scripts/data/tool_call_serializer.py`](../../src/scripts/data/tool_call_serializer.py).
5. **Resolve** — the called name is declared in *that row's* `<functions>`. Plus globally: no function
   name may appear anywhere in the corpus with two different schemas (Glaive's documented defect).
6. **Schema validation** — `jsonschema` validate the arguments with `additionalProperties: false`
   forced (catches invented params); all `required` present; types/`enum`/`format` correct.
7. **Value plausibility, partial** — enum membership, declared numeric bounds, ISO-8601 parse where
   `format: date-time`, unit consistency where a unit enum exists. Not general semantics.
8. **Abstention rows invert 3–6** — zero `<function_calls>`; for `no-suitable-tool` the intended
   function is **absent** from `<functions>` (the Hammer deletion, so correctness is *constructive*,
   not judged); assistant content contains no function name from the global inventory.
9. **`missing-args` rows** — no call; content contains `?` and names ≥1 `required` parameter of the
   intended tool that is absent from the user turn. A mechanical proxy, and the strongest available.
10. **Executable stub** — `nested-args` **plus every row whose gold tool is value-executable**;
    `inspect.signature(stub).bind(**arguments)` then call it. Compared **by value** where possible
    (gate 17), not merely "it returned".
11. **Token budget** — tokenize with dolma2, record `n_tokens_dolma2` as a top-level field, reject
    over the SFT sequence length. That length is UNVERIFIED until post-training fixes it; recording
    the field lets the trainer filter later without re-tokenizing. A 20-schema `system` message plus
    Perplexity's 13-parameter schema plus a reasoning prefix can blow the window on its own.

**Pre-publish, over the whole build directory:**

12. **Schema-pool disjointness** — every tool name in a row's `<functions>` belongs to the pool
    matching that file's split. Converts "heldout carved by schema" from intention into fact.
13. **Recompute the profile's leakage key locally** — sha256 over `role \x1f content` per message;
    assert `train ∩ heldout = ∅` **before** uploading. With `max_leakage: 0`, 40,000 rows and a finite
    schema pool, one accidental duplicate refuses the entire publish. Do not discover this in landing.
14. **Naming and depth** — `parse_shard_name` on every basename; reject `-of-NNNNN` and index-less
    names; assert the path is exactly `conversations/<domain>/<category>/<split>-NNNNN.jsonl` so a
    third nesting level can never slip in.
15. **Diversity caps** — no single function >**2%** of its category's rows; no single user-turn
    5-gram >**0.5%**. Glaive's exact failure modes (`calculate_tip` dominance; "Can you order a pizza
    for me?" as the sole refusal trigger, which teaches pizza-refusal rather than relevance).
16. **Temporal coherence** — each row carries a top-level `as_of`; every date-typed argument falls in
    a declared window around it.

### Domain-specific gates (17–35)

**Arithmetic**

17. **Value-execution** — for every row whose gold tool is value-executable (15 of 18 arithmetic
    tools, 2 pedagogy schedulers): bind, run the stub, compare against top-level `expected_result`.
    Strictly stronger than gate 10. `expected_result` is stub-written, never hand-typed.
18. **Operand-magnitude consistency** — in `arithmetic/answer-directly`: ≤2 operands, both |x| ≤ 12,
    operator ∈ {+, −, ×}. In `arithmetic/{single-call,nested-args,parallel-call}`: ≥1 operand >12
    **or** ≥3 operators. Stops the two mirror cells teaching contradictory thresholds.
19. **Numeric formatting** — an integer-valued result must not render as `…​.0`; an irrational result
    must carry `precision`/`digits` or route through `round_and_format`. Targets the documented
    `551368 / 82 → 6724.0` class of failure.
20. **Expression safety** — every `expression` string must evaluate under
    `numexpr.evaluate(expr, global_dict={}, local_dict={"pi":…, "e":…})`; any identifier outside that
    set rejects the row. This makes the LangChain calculator-exfiltration shape
    (`os.environ["OPENAI_API_KEY"]` through the expression) **unrepresentable in the corpus**, so we
    never train it. `sympify` is assumed **not** hardened for this threat model — UNVERIFIED.

**Web search**

21. **Domain-filter shape** — `allowed_domains` and `blocked_domains` together → reject (providers
    return 400). No scheme in domain strings; ≤100 entries (OpenAI); ≤20 entries and ≤253 chars each
    (Perplexity `search_domain_filter`).
22. **`user_location`** — `type` exactly `approximate`; ≥1 of city/region/country/timezone; `country`
    a valid ISO 3166-1 alpha-2; `timezone` ∈ `zoneinfo.available_timezones()`.
23. **Date format** — Perplexity `search_*_date_filter` must match `MM/DD/YYYY`; no filter may name a
    date after the row's `as_of`.
24. **Freshness agreement** — every web-search row carries `freshness ∈ {static, slow, fast}`.
    Assert `*/answer-directly ⟹ static` and `web-search/{single-call,parallel-call} ⟹ {slow, fast}`.
25. **Parametric-knowledge consistency** — every `*/answer-directly` assertion is either in the
    curated fact bank or survives **k=8 samples at 8/8 agreement**; failures are demoted to
    `web-search/single-call`. This is a *consistency measurement*, not an LLM judge, so it does not
    violate the no-judge-in-the-write-path rule. **k=8 is a choice; the optimal k is UNVERIFIED.**

**Pedagogy**

26. **Taxonomy closure** — every id in `principles_present` / `principles_violated` /
    `myths_asserted` / `myth_corrected` must appear in the frozen v1 id list (§17). Unknown → reject.
    Never renumber; deprecate.
27. **Tool→principle entailment** — `principles_present ⊆ map[gold_tool] ∪ arg_conditioned(...)`
    using §17's frozen map. A row claiming spaced repetition without a scheduling tool and without an
    explicit named interval in prose is rejected.
28. **Expertise-reversal matched pair** — `pair_id` appears exactly twice; the arms differ in
    `learner_level`; their assistant `content` differs; computed support density is strictly greater
    in the novice arm. The only pairwise gate, and the only way scaffolding/fading is reachable in v1.
29. **Myth gate** — frozen regex bank over **assistant content only**. A hit rejects, *unless* the row
    is in `*/answer-directly`, names the id in `myth_corrected`, **and** matches ≥1 token from that
    myth's frozen refutation bank. Without the co-occurrence requirement, "learning styles" in a
    correction is indistinguishable from an assertion.
30. **Overclaim policy** — reject: `(immediate|delayed) feedback is (better|superior)` without a
    conditioning token from {procedural, conceptual, novice, expert, acquisition, retention,
    transfer}; "2 sigma"/"2σ" without a confound token; the 0.4–0.7 formative figure presented as an
    effect size; germane load asserted as a third additive load; "10,000 hours" outside a correction.
31. **Unlabelable-tag guard** — assert the four structurally undecidable tags are absent from
    `principles_present` unless their precondition is met in-row. If the format has no session
    boundary, do not fake spacing.
32. **Citation closure** — citations must be ids in §17's frozen table, and a row **may not emit
    volume or page numbers for an entry flagged unverified**.

**Cross-domain**

33. **`abstain_reason` ↔ path** — `abstain_reason ∈ {no_tool_matches, required_arg_absent,
    answer_is_parametric}` must match the category segment (`no-suitable-tool` / `missing-args` /
    `answer-directly`). One shared negative-example schema across all four domains, auditable.
34. **Domain ↔ gold tool** — the gold tool's domain, from §17's frozen tool→domain registry, must
    equal the `<domain>` path segment; for abstention rows, the deleted or tempting tool's must.
    **This is the gate that keeps the domain axis honest**; without it §4's labelling rule is an
    intention rather than a fact.
35. **Reasoning-prefix shape** — assistant content containing a call must match exactly
    `prose? <function_calls>…</function_calls>` with nothing after; prose ≤2 sentences and ≤40 dolma2
    tokens, zero `<function_calls>` inside it, `reasoning_prefix_tokens` recorded. Replaces the strict
    dichotomy formerly in gate 3. The prefix is mechanically strippable, so the
    with-prefix/without-prefix ablation stays free — but the claim that it *helps* is **UNVERIFIED**
    (the supporting evidence is inference-time, on models not trained for the format).

### Not machine-checkable — do not claim it

- Whether the chosen tool is the one a competent human would choose. Mitigated **constructively**:
  generate the user turn *from* the chosen schema so the label is true by construction, and reject
  rows where ≥2 tools validate the same arguments.
- Whether a pedagogical move is right **for this learner**; whether prose is level-appropriate;
  whether a hedge on a contested claim is *proportionate* (we check a conditioning token is present —
  presence is not sufficiency).
- Whether a citation is **correct** — only that its id is in the frozen table.
- Whether a user turn is natural, a tool result realistic, or a clarifying question the *best* one.
- **Search-query quality has no established scorer.** BFCL grades the `answer` field and deliberately
  ignores free-text justification; we have no environment turn, so the emitted query *is* the graded
  object. Our `query_required_terms` / `query_forbidden_terms` scorer is **ours alone** — never
  present it as a BFCL number.

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

### Multi-turn: NOT blocked. Corrected 2026-08-08.

An earlier draft called this "a converter bug our data format cannot dodge." **That was wrong** — it
is a transform-function choice. Measured and source-read:
[`verify/verify_multiturn_mask.py`](verify/verify_multiturn_mask.py).

**The mechanism is real.** Label masking is offset-mapping based: for each assistant message `i` a
trainable span comes from `apply_chat_template(messages[:i], add_generation_prompt=True)` versus
`apply_chat_template(messages[:i+1])`, and `dataset_transformation.py:1284` **raises** unless
`rendered.startswith(through)`. For a **non-final** assistant turn the sub-render makes that turn
`loop.last`, so it emits `eos_token`; the full render has `<|im_end|>\n` at the same position:

```
multi-turn (2 assistants)  eos='<|endoftext|>'  ->  RAISES
  assistant[2] (final=False) UNSTABLE at byte 233:
      full has 'im_end|>\n<|i', sub-render has 'endoftext|>'
  assistant[4] (final=True) stable
```

**8 of 8 sampled real Dolci rows fail this way** under the default transform — which is precisely why
the claim "AI2 cannot train these" was suspicious, and it is false.

**The fix already exists upstream.** `dataset_transformation.py:1212,1248`:

```python
last_turn_only: bool = False,
...
if last_turn_only and message_idx != last_assistant_idx:
    continue
```

exposed as two transforms — `sft_tulu_tokenize_and_truncate_v1` (`last_turn_only=False`, the default)
and **`last_turn_tulu_tokenize_and_truncate_v1`** (`last_turn_only=True`). With the latter, non-final
assistant turns are skipped, so the only stability check is on the final turn, which is stable. Their
own error text names the cause: *"the template appends eos_token only on the final turn."*

**Ranked fixes for the deferred multi-turn set:**

1. **`last_turn_tulu_tokenize_and_truncate_v1`** — zero code. Cost: only the **final** assistant turn
   is trainable; earlier tool calls become masked context. On a 21-message row with 10 assistant
   turns that trains 1 of 10, which for a tool-calling dataset throws away most of the signal.
2. **Our own producer, mask built by construction** — render turn-by-turn, tokenize each segment,
   concatenate. No offset mapping, therefore no prefix requirement, and **every** assistant turn stays
   trainable. We already own the producer (it is offline and external either way), so this is the
   recommended route. It is safe here for a specific reason: every segment boundary is an **atomic
   added-token** (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`), so no BPE merge can straddle a
   boundary and change the tokenization.
3. **Patch the template** so non-final and final assistant turns close identically, then append the
   real EOS once. Upstream-visible; least attractive.

**Rejected on evidence:** splitting a conversation into prefix rows at assistant boundaries. The
longer prefix rows still contain a non-final assistant turn, so they still raise — the split only
helps the shortest row.

Forcing `eos_token = "<|im_end|>"` also makes the render stable, but the conversation then no longer
ends in **100257**, and OLMo-core finds document boundaries by EOS — so it trades one silent
corruption for another unless the producer appends the real EOS itself.

**v1 is unaffected either way:** `system + user + assistant` has exactly one assistant turn and it is
final, so it is stable under the default transform.

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
    purpose="Single-turn tool-call SFT conversations across general, arithmetic, web-search and "
            "pedagogy tool inventories with a held-out schema split, to teach function calling "
            "to the ~4B-active OLMo MoE",
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
                        "carved_by": "held-out tool schemas; query-template bank; "
                                     "operand-magnitude band (arithmetic); entity bank "
                                     "(web-search) — all fixed before generation"},
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

**Answered 2026-08-08** (was Q1): byte-identity is **proven**, not inferred —
[`verify/verify_render_identity.py`](verify/verify_render_identity.py) renders 5 real Dolci rows both
as AI2 publishes them and as we inline them, `IDENTICAL=True` on all 5 including a 27-turn row. The
leading space is read from the template source: per-message `functions` emits `' <functions>'`, the
row-level `tools` path emits `'<functions>'`.

| # | Question | Blocks |
| --- | --- | --- |
| 1 | ~~Does the registry chat template corrupt our inlined rows?~~ **ANSWERED — yes, and we are unaffected.** `dataset_transformation.py:203-209` does append `" You do not currently have access to any functions. <functions></functions>"` whenever a system message has no `functions` **field**, which ours never do. Demonstrated in [`verify/verify_template_choice.py`](verify/verify_template_choice.py): the row lists its tools and then denies having any. **Our producer never calls that template** — it reproduces the shipped `chat_template.jinja`, proven byte-identical. The check is kept so nobody reintroduces the hazard via open-instruct's converter | Nothing now |
| 2 | The **curated fact bank** for the 1,500 `answer-directly` rows. k=8 consistency is a proxy and k is UNVERIFIED | Publishing `answer-directly`. This is the cell most likely to install a confident falsehood |
| 3 | **Dolci licence sign-off** — ODC-BY is stated in the description prose with **no `license:` key** in frontmatter. Weaker than a tagged licence; needs a human, same decision class as xLAM | All 10,600 reformat rows, i.e. 26.5% of the corpus |
| 4 | **xLAM licensing** — `cc-by-4.0` tag vs research-only prose | Whether the total can reach 67,500 |
| 5 | Post-training **sequence length** | §10 gate 11, and max tools per row |
| 6 | **Which multi-turn masking route** — `last_turn_only=True` (free, trains only the final turn) or our own by-construction producer (all turns trainable). §12 settles that both work; this is a cost decision | The deferred multi-turn set; nothing in v1 |
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

---

## 17. Frozen artifacts

Gates 26–32 and 34 read from these. They are **code inputs, not documentation** — versioned,
diffable, and frozen at first publish. Living under `docs/tool-call/frozen/` when written.

| Artifact | Consumed by | Note |
| --- | --- | --- |
| Learning-science principle id list (`LS.*`) | gates 26, 27, 31 | 20 ids, **11 decidable on a v1 row** — publish 11, never 20 |
| Myth + low-utility id list (`MYTH.*`, `LOWU.*`) | gates 26, 29 | 10 myths + 5 low-utility strategies |
| Myth → required-refutation-token bank | gate 29 | without the co-occurrence requirement a correction is indistinguishable from an assertion |
| Bloom verb → level table | gate 26 | the labeller may not improvise verbs |
| Citation table, **with a per-entry `verified` flag** | gate 32 | a row may not emit volume/page numbers for an unverified entry |
| Tool → domain registry | **gate 34** | this is what keeps the domain axis honest; without it §4's labelling rule is an intention, not a fact |
| Tool → allowed-principle map | gate 27 | plus the argument-conditioned extensions |

**Never renumber an id — deprecate it.** Ids are baked into published rows, and a renumber silently
re-labels history.

Pin a **retrieval date per schema** in the tool registry: third-party schemas drift (Anthropic ships
three `web_search_*` versions with different defaults; several CASE and Eedi field lists are already
UNVERIFIED). Nothing about this blocks a publish, but it blocks any claim the schemas are *current*.

---

## 18. The generator model

### Read this first: two thirds of the corpus needs no model at all

| provenance class | rows | needs an LLM? |
| --- | --- | --- |
| Reformat, all domains | 11,500 | **No** — field mapping + serializer |
| Derived where the language comes from upstream | 11,500 | **No** — upstream supplies the prose; we compute or author the call |
| Derived needing new phrasing | 500 | Yes |
| Fresh but programmatic (operand generators, CCSS recombination, entity/date pools) | 3,800 | **No** — template + Python stub |
| Fresh needing natural-language generation | 12,700 | Yes |
| **Needs an LLM** | **13,200 (33%)** | |
| **Needs no LLM** | **26,800 (67%)** | |

Where the 13,200 sits: pedagogy ~6,000 (student utterances, misconception surface forms), web-search
~4,000 (questions with a genuine search-necessity boundary), general ~2,000, arithmetic ~1,200
(**surface phrasing only** — every operand and every `expected_result` comes from the generator script
and the Python stub, never from a model).

**So the generator governs 33% of the corpus and 0% of its correctness.** Schemas are authored,
held-out pools are fixed, arithmetic truth is computed, and the wire format is emitted by a
deterministic serializer. It is a *phrasing and scenario* engine. Calibrate the effort accordingly.

### Primary: `Qwen/Qwen3-235B-A22B-Instruct-2507`. Fallback: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

**Licence — verified at the file level, not the tag.** Both are `apache-2.0`, and the defence is the
**absence** of any output clause: Apache 2.0 has no provision covering model outputs, no naming
condition, no MAU gate. **We checked the LICENSE file itself**, because the tag is not sufficient —
[`verify/verify_sources.py`](verify/verify_sources.py):

```
Qwen/Qwen3-235B-A22B-Instruct-2507   tag: apache-2.0   file: Apache License Version 2.0, January 2004
Qwen/Qwen2.5-72B-Instruct            tag: other        file: Qwen LICENSE AGREEMENT ...
```

That second line is why the check exists. Qwen2.5-72B is widely assumed Apache and is not; its
agreement adds a "Built with Qwen" display requirement, a 100M-MAU gate and Hangzhou jurisdiction.

Disqualified, with the governing clause:

| Model | Clause | Verdict |
| --- | --- | --- |
| Anthropic API | Commercial ToS §D.4: may not *"train competing AI models"* — even though §B grants that the customer *owns its Outputs* | Out. Output ownership does not license this use |
| Google Gemini API | *"You may not use the Services to develop models that compete with the Services"* | Out |
| OpenAI API | *"use output from the Services to develop models that compete with OpenAI"* — **UNVERIFIED by primary fetch** (openai.com returns 403 here); sourced secondarily | Out |
| **Gemma** | ToU "Model Derivatives" expressly reaches *"the generation of synthetic data Outputs by Gemma for training that model"* — so our model would be a Model Derivative and must propagate Google's use restrictions into our licence | Out — **viral** |
| Qwen2.5-72B | §5.b "Built with Qwen" display requirement + MAU gate | Out |
| Llama 3.3 | Outputs are permitted (the old prohibition is gone), but §1.b.i requires *"Llama" at the beginning of any such AI model name* | Declined — forces `Llama-…` on a model whose weights are not Llama-derived |
| `deepseek-ai/DeepSeek-V3` | `cardData.license` is **null** | Out |
| `deepseek-ai/DeepSeek-R1` | `mit`, zero obligations — licence-clean | Reserve only: it is a **reasoning** model, so every output arrives wrapped in a trace to strip, and its register is the opposite of a terse tool-caller |

**Tool-calling ability: UNVERIFIED, and we should stop trying to source it.** No authoritative
per-model BFCL number could be obtained — the leaderboard renders client-side and the aggregators
serve bot walls. Circulating figures are vendor self-reported, do not name the variant, and are v3
while the board is on v4. **Do not cite one in the dataset card.**

Replace it with a measurement in our own units: **a 200-prompt bake-off** across Qwen3-235B,
Mistral-Small-3.2 and OLMo-3-Instruct, scored by our own schema validator plus the arithmetic stub —
schema-valid rate, correct-abstention rate, argument-name fidelity against the given schema, and
distinct-n over user turns. One afternoon, and it answers the only question that matters.

**Machine shape, not token cost, is the real discriminator.** 13,200 rows at ~2× overgeneration ≈
26,400 generations ≈ 90M tokens — a rounding error that does not distinguish the candidates.
But Qwen3-235B-A22B is 235B total / 22B active ≈ 470 GB in bf16, so it wants **8×H100-80 minimum**
under vLLM with expert parallelism, whereas Mistral-Small-3.2-24B is ≈48 GB and **fits one 80 GB
card**. That is the actual argument for the fallback: if the 8-GPU shape is refused or queued, a 24B
Apache model on one card unblocks the same 13,200 rows the same day.

**Platform rules that apply to the generation run:** price and approval class come from
`edullm check --json` and nowhere else — quote nothing from a doc. **Write the dtype into the command
text** (`--param-dtype bfloat16`), because the precision guard reads the command and cannot see a
dtype set in code. Pass the literal `--dataset none`, since the generation run reads no corpus release.

### Self-distillation with OLMo-3-Instruct: use it as a discriminator, not a generator

The format-fidelity argument for self-distillation **dissolves on inspection.** Our wire format is
deterministic string assembly from three inputs (schema list, call name, kwargs). **No LLM should be
emitting those literals at all** — the generator returns structured JSON
(`{"user": …, "call": {"name": …, "args": {…}}}`) and *our serializer* renders it. A serializer is
100% format-correct by construction; a model is not, and every format error OLMo-3 did make would
arrive pre-blessed as correct because it came from the reference model. The strongest argument for
self-distillation turns out to be an argument for writing a serializer — which we need anyway.

The licence argument is real but non-differentiating: OLMo-3-Instruct and Qwen3-235B are both Apache
with no output obligations.

**The case against is decisive.** A 49.8-BFCL teacher caps any capability for which it is the sole
label source, and the student is **4B-active** — we would be capping a smaller model below an already
mediocre teacher. Worse, self-distillation amplifies *systematic* error rather than averaging random
error, and OLMo-3-Instruct's characteristic failure is **over-calling**: invoking a tool when none
fits, inventing parameters, drifting argument names off the schema. Those are exactly our
`relevance-hard`, `no-suitable-tool` and `missing-args` categories — the three that justify the
dataset existing. Training a model to reproduce its predecessor's over-calling bias, in the categories
designed to cure over-calling, is the one outcome that makes the exercise worse than not doing it.

**What it is genuinely good for:**

1. **Difficulty filter.** Run it over every candidate row from any source and **prefer rows it gets
   wrong.** A row the current model already answers carries almost no gradient. This turns its 49.8
   ceiling from a liability into a free, exactly-calibrated hardness signal at one forward pass per row.
2. **Serializer round-trip check.** If OLMo-3-Instruct cannot parse a row we emitted, the serializer
   is wrong.
3. **Third arm in the bake-off.** If Qwen3-235B does not beat it on schema-valid rate and
   correct-abstention, that is worth knowing *before* we spend the 8-GPU shape.
