# PRD: tool calling for the OLMo MoE

**Branch:** `tool-call-amy` (off `origin/main` @ `08df5aa0`) · **Owner:** Amy ·
**Started:** 2026-08-08

Read this first if you are an agent picking up this work. Then
[`dataset-design.md`](dataset-design.md) for the byte-level contract, and log what you change in
[`progress.md`](progress.md).

---

## 1. Goal

Teach the final OLMo MoE (**~4B active parameters**) to call tools: given a set of tool schemas and
a user request, emit a **well-formed, schema-valid call with correct arguments** — or decline to
call anything when no tool fits.

**Three mandated must-haves** (2026-08-08), each at path level 1 so each is independently
measurable: **arithmetic tools**, **web search**, and a **pedagogy focus**. The third is three
separable things — pedagogy *tools*, pedagogical *prose*, and learning-science *knowledge* — treated
separately in [`pedagogy.md`](pedagogy.md), because only the first two are tool calls. The knowledge
requirement ("know learning science if it isn't web-searchable") is what the new `answer-directly`
category exists for.

The deliverable of *this* workstream is **the dataset**, not the trained model. A post-training
pipeline is expected shortly; this work makes sure that when it lands, the data is already the
right shape and does not need regenerating.

## 2. Non-goals

- **Not** training or serving a model. No GPU run is part of this.
- **Not** building the tool *runtime* — no executor, sandbox, or agent loop in the product sense.
  Execution appears only as a *verification* step on generated rows (§6).
- **Not** reinforcement learning or preference data. SFT conversations only.
- **Not** multi-turn agentic trajectories in the first cut — see §3.

## 3. Scope (decided 2026-08-08)

| Decision | Choice | Why |
| --- | --- | --- |
| Tool domain | **Four, at path level 1: `general` · `arithmetic` · `web-search` · `pedagogy`** | Revised 2026-08-08 for the three mandated must-haves. Each sits at level 1 so each is independently trainable, ablatable and measurable by glob. `edu` was **renamed** to `pedagogy`, not added alongside. **Irreversible labelling rule:** a row's domain is the domain of the **gold tool**, not the topic of the user turn — so a pedagogy-topic question answered by `web_search` is `web-search/*`. Domain totals therefore measure gold-tool domain, not inventory composition |
| Capability axis | **8 categories, identical in all four domains** (32-cell grid) | The eighth, **`answer-directly`**, is forced not cosmetic: `no-suitable-tool` deletes the gold function (BFCL `irrelevance`), whereas `answer-directly` keeps a plausible-looking tool present and calling it is *still* wrong because the answer is settled knowledge. That is the whole "know learning science, don't search for it" requirement, and folding it into `no-suitable-tool` makes it unmeasurable |
| Capability scope | **Single-turn + irrelevance first.** Multi-turn deferred to a second dataset | Covers the BFCL simple/multiple/parallel/relevance axes. Multi-turn roughly triples generation and verification cost and is where small models are weakest — not the place to discover pipeline bugs |
| Provenance | **Hybrid: 31.5% reformat / 17.5% derived / 51% fresh synthesis** | Revised 2026-08-08. The five `allenai/olmo-toolu-*` mixes named in `src/scripts/train/sft/README.md` are **all HTTP 401** — foreclosed. But their **public non-thinking equivalent** `allenai/Dolci-Instruct-SFT-Tool-Use` (**227,579 rows**, ungated) is not, and it is already in OLMo's convention, so 10,600 filtered rows come in as a *lift* rather than a translation. ToolACE keeps 2,000 as a provenance hedge; **Hermes is dropped**; xLAM still blocked |
| **Wire format** | **Adopt OLMo 3's convention verbatim** — `<functions>` / `<function_calls>` / role `environment` | Not invented: those delimiters are **single token ids 100266–100269** in the OLMo-3 instruct tokenizer, carved in place over `<|extra_id_1..4|>` at **unchanged vocab 100278**. Swapping the tokenizer costs no resize and no new embedding rows, and any dolma2-pretrained checkpoint stays byte-compatible |
| Record layout | **Tool call serialized into the assistant message `content`** — we adopt OLMo's *rendered bytes*, not its *row layout* | Forced by the validator, measured against a real Dolci row: AI2 parks the call in a sibling `function_calls` field with `content: null`, which makes two rows with different calls hash identically (`COLLIDE=True`). Inlining renders byte-identical output while keeping the call where the leakage key can see it. See `dataset-design.md` §2–§3 |
| Category axis | `<category>` in the path is a **capability band**, not a tool family | Two path levels is the entire budget. Tool family in the path would make "is parallel calling worse on edu tools" unanswerable — the one cross-domain question the domain split exists to answer |

## 4. Deliverables

1. **`sft/tool-call-single-turn`** — profile `sft-conversations/v1`, **40,000 rows** (train 36,000 /
   heldout 4,000). Heldout carved by **held-out tool schema *and* query-template bank**, not by row.
2. **The eval set is the `heldout` partition of (1) — not a separate dataset id.** Corrected from
   the first draft: `eval-items/v1` is **not registered** in the installed pipeline, so publishing
   benchmark *items* raises `ProfileError`. Heldout categories are sized ≥ their BFCL counterparts
   so the numbers are comparable. A separate `eval/` dataset is for `eval-results/v1` — harness
   **outputs** — which is a better thing to publish anyway.
3. **A generator + verifier** that emits the JSONL and refuses to write a row failing any gate in
   `dataset-design.md` §10, plus the two whole-directory pre-publish checks (schema-pool
   disjointness, and a **local recompute of the leakage key before upload**).
4. **These three docs**, kept current.

**Explicitly not delivered, and it matters:** nothing in this repo can consume the `.jsonl` we
publish. There is no `.jsonl` reader under `src/olmo_core/`; the conversation → `(tokens,
label_mask)` converter lives in AI2's external `open-instruct`. The consumption side *is* wired
(`label_mask_paths`, `get_labels()`, `src/scripts/train/sft/`). See `dataset-design.md` §12 —
this blocks *use*, not publishing.

Generated bytes go to `data/` (gitignored). Docs are tracked. Run outputs go to `runs/`.

## 5. Success criteria

The dataset is done when all of these hold:

1. `edullm-data`'s validator **accepts** it — meaning well-formed `messages` on every row and
   **zero** recomputed train/heldout leakage.
2. Every row's call is **schema-valid** against its declared tool: function exists, required
   params present, no invented params, types correct. Machine-checked, not sampled by eye.
3. The **abstention** category is present and non-trivial, so the model has evidence that *not*
   calling a tool is sometimes correct.
4. Heldout contains **tool schemas absent from train**, so heldout score measures generalization
   from a schema rather than recall of a name.
5. Heldout categories map onto BFCL's taxonomy and are each **≥ the BFCL counterpart's n**, so our
   confidence interval is never worse than the number we compare against. Irrelevance and relevance
   are reported **as a pair, never blended** — either one alone is trivially gamed (a model that
   never calls scores 100/0; one that always calls scores 4/100). We grade **AST-only** and say so,
   and we do **not** claim a BFCL overall, because this dataset touches 0% of Agentic and Multi-Turn.

## 6. Verification (what makes a row admissible)

A row is written only if it passes, in order:

1. **Parse** — the Pythonic call parses to a single **keyword-only** `ast.Call` (calls are not JSON).
2. **Resolve** — the named function is one of the tools offered in that row's `<functions>` block.
3. **Schema** — arguments validate against the tool's JSON Schema: required present, no extras,
   types right.
4. **Abstention rows invert 1–3** — assert that **no** call was emitted.
5. **Leakage** — the row's dedup key is absent from the opposite partition.
6. **Value execution** where the gold tool is value-executable — 15 of 18 arithmetic tools and 2
   pedagogy schedulers. `expected_result` is written by running the stub, never typed, so correctness
   is a value comparison rather than a shape check. **0 of 14 web-search tools are executable** —
   that is a stated limit, not a gap.
7. **Pedagogy tagging** against the frozen taxonomy, and the **myth negatives** — a myth asserted in
   assistant content is a hard reject unless the row is a correction naming the id and carrying a
   required refutation token.

Anything not mechanically checkable is not claimed as a quality property. In particular: **11 of 20
learning-science principles are decidable on a single-turn row** — publish 11, never 20.

## 7. Hard constraints

- **Pipeline:** profile `sft-conversations/v1`; `.jsonl`; every message needs a `role` and a
  present `content` key; `coverage` + ≥2 partitions with one heldout; `max_leakage` defaults to 0.
  Object **path is hashed into `manifest_sha256`** — layout is irreversible without re-copying
  every byte. Full detail and the verified evidence in [`dataset-design.md`](dataset-design.md).
- **Buckets:** write to `s3://edullm-landing`; the validator promotes into `s3://edullm-data`,
  which nobody here can write directly.
- **Platform:** any GPU work goes through `edullm`, never Beaker, never a direct AWS call. Read
  price and approval class from `edullm check --json`, never from a doc.
- **Model:** ~4B **active** params in an MoE. Format choices should favour what a small model can
  reliably emit; complexity that a 70B tolerates is not automatically affordable here.

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| Model learns to **always** call a tool | Abstention is **10%** of rows (Hammer's swept optimum at 1.5B) in its own path cells, so the ratio is re-weightable at train time by glob without regenerating |
| Heldout measures memorization, not generalization | Hold out whole tool schemas **and** the query-template bank, both carved *before* generation. Schema carving alone is insufficient — same templates + a new function name measures template recall |
| **General ability regresses** from narrow SFT | Mix **1:1** with general instruction data, floor 20% (Alopex arXiv 2411.05209, measured on 1.5–2B models). Publish 40,000 rows and let mixture weights set the ratio — do not bake it into the bytes. Probe with general benchmarks run with **no tools in context** |
| `max_leakage: 0` refuses the whole publish on **one** duplicate | 40,000 templated rows over a finite schema pool makes collision likely, not hypothetical. **Recompute the exact leakage key locally before upload** (`dataset-design.md` §10 gate 13) |
| Degenerate diversity teaches the wrong lesson | Caps: no function >2% of its category, no user-turn 5-gram >0.5%. These are Glaive's documented failures (`calculate_tip` dominance; pizza-refusal standing in for relevance) |
| Tool call parked outside `content` → integrity check blind | **Closed.** Call is serialized into `content`; measured in `dataset-design.md` §2 |
| Path layout wrong → republish every byte | **Closed.** Two nesting levels, capability as level 2, forward-compatible with the newer label feature |
| Upstream data licensing unfit for an open model | ToolACE + Hermes only (both `apache-2.0`); xLAM blocked pending a human decision; ToolBench and Glaive excluded. Record provenance in `sources[]` |
| Argument-value hallucination | `additionalProperties: false` forced during schema validation, plus an **executable Python stub** for `nested-args`. Argument fidelity is the measured weak slot at ~4B (Hammer-4b: AST simple 62.58, Exec simple 67.79) |
| **Domain is confounded with provenance** | All 9,600 reformat rows land in `general`; the three new domains are 0% reformat. So any general-vs-pedagogy delta is partly human-curated-vs-synthetic. Unfixable by path — carry a `provenance` row field and **state the confound wherever a cross-domain number appears** |
| `answer-directly` is the only category where the assistant **asserts facts** | 1,500 rows. k=8 self-consistency is a proxy and k is UNVERIFIED. Needs a **curated fact bank** before publish — the cell most likely to install a confident falsehood |
| Repeating the org's own unreviewed claims as fact | Any question about Alpha School / 2 Hour Learning results goes in **`web-search/*`, never `pedagogy/answer-directly`**. The 2.6× MAP figure is company-sourced and not independently reviewed. `MYTH.10` *is* the Bloom 2-sigma claim, so the canon and the caution are the same rows |
| Two domains **cannot** carve heldout by schema | `calculator` and `web_search` must be in train. Substitute axes: operand-magnitude band (arithmetic), entity + parameter bank (web-search). For those cells "heldout measures schema generalization" is **false** and must not be written |
| Deferring multi-turn defers the **largest** headroom | Accepted deliberately, and stated plainly: multi-turn gaps run −50 to −87 pts below non-live AST, and it is the axis most responsive to data (ToolACE-8B 87.54/**7.75** vs xLAM-2-8b 84.35/**69.25** at identical size — a +61.5 pt swing from data alone). This dataset **cannot move** the categories dominating BFCL v3/v4 scoring |
| JSON call syntax may be worse than code for small models | Every controlled *inference-time* comparison favours code (programmatic ≥ JSON on 11/14 models, BFCL v4). But all of those test models **not SFT'd for the format**, and JSON is proven at 1.3–1.5B (xLAM, Hammer). No matched small-model JSON-vs-Python SFT comparison exists — **UNVERIFIED**. Hedge: keep the verifier's tool representation format-agnostic so a Python-syntax arm can be emitted from the same verified rows |

## 9. Open questions

Full list with what each blocks: [`dataset-design.md`](dataset-design.md) §15. The two that need a
**human with authority**, not another research pass:

1. **Dolci licence sign-off.** `allenai/Dolci-Instruct-SFT-Tool-Use` states ODC-BY **in its
   description prose with no `license:` key in the frontmatter**. That is weaker than a tagged
   licence, and it gates 26.5% of the corpus.
2. **xLAM licensing** — tagged `cc-by-4.0`, but the gated prose says "research purposes only in
   support of an academic paper." Decides whether the total can reach 67,500.

The load-bearing *technical* unknown is cheap and must be done first: **render one Dolci row through
`Olmo-3-7B-Instruct/chat_template.jinja` and diff it against the same row inlined per design §3.**
That proves byte-identity and settles the leading-space question in one command. Do not start the
reformat pass before it passes.

**Answered since the first draft:** the OLMo-3 instruct tokenizer **does** define tool-call
delimiters (single ids `100266`–`100269`, vocab unchanged at 100278;
`allenai/dolma-2-tokenizer-olmo-3-instruct-final` is a 307 alias for
`allenai/olmo-3-tokenizer-instruct-dev`, sha `55f211df…` — pin it); the `olmo-toolu-*` mixes are all
401 but a public equivalent exists; size 40,000; abstention 10%; mixing 1:1; and the SFT path
(consumption wired, producer external). **Multi-turn is NOT blocked** — an earlier draft said it was,
which was wrong. It needs either `last_turn_tulu_tokenize_and_truncate_v1` (free, but trains only the
final assistant turn) or our own by-construction mask (all turns trainable). See design §12.

Two things to inherit deliberately rather than by accident: **register `100269`
(`</function_calls>`) as an explicit stop token** — it is `special: false`, so tool calls otherwise
do not terminate — and **run our own BFCL decontamination**, because only the *private* 200K mix is
named `decontaminated` and the public cut's status is unverified.
