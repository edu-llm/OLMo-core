# Tool-call dataset — progress log

Append one line per change, newest last. Keep entries to a single line where possible.
Anything longer belongs in [`prd.md`](prd.md) or [`dataset-design.md`](dataset-design.md).

**Convention:** `YYYY-MM-DD — what changed — why / consequence.`

---

## 2026-08-08

- Branch `tool-call-amy` created off `origin/main` @ `08df5aa0`, in a **separate worktree** at
  `../OLMo-core-tool-call` — a second chat is working `latent-superposition-module` in the original
  directory and git branch state is per-directory, not per-chat.
- Copied untracked `.claude/skills/edullm-{dataset-design,datasets}`, `sb-aws-readonly` and `.mcp.json`
  into the worktree; they are untracked in the source tree so a fresh checkout does not carry them.
- Added `/data/` to `.gitignore` — generated dataset bytes are reproducible build artifacts and must
  not be committed (`/runs/` and `/dataset-cache/` were already ignored on `main`).
- Verified the pipeline contract against the **installed `edullm-data==0.2.0`**, not the skill docs.
  Recorded two places the skill text is stale — see `dataset-design.md` §"Verified pipeline contract".
- Created `docs/tool-call/` with this log, `prd.md`, and `dataset-design.md`.
- **Scope decided:** both general + edu tool domains (split in the path); single-turn + irrelevance
  first with multi-turn deferred; hybrid provenance (reformat permissive open sets, synthesize gaps).
- **Record format decided, with evidence:** the tool call is serialized into the assistant message
  `content`, **not** an OpenAI-style sibling `tool_calls` field. Measured: a sibling field makes two
  rows with different calls hash to the *same* leakage key (`COLLIDE=True`), so the validator's only
  payload integrity check would be blind to the exact thing the dataset teaches. Probe:
  `scratch/verify_record_shape.py`.
- **Path layout decided:** `conversations/<domain>/<category>/<split>-<NNNNN>.jsonl` — exactly two
  levels below the group, which is forward-compatible with the two-level path-label feature the
  newer pipeline derives (and raises on a third). Verified shard naming and disjoint train/heldout
  globs: `scratch/verify_naming.py`.
- Found a required field **neither skill doc mentions**: `coverage` ∈ `{partition, overlapping,
  incomplete}` is mandatory on the group whenever `partitions` exist (`validate.py` `bad-coverage`).
- Heldout will be carved by **held-out tool schema before generation**, not by random row split.
- Research pass landed (6 parallel investigations: xLAM/ToolACE/Hermes formats, BFCL taxonomy +
  small-model scores, chat-template conventions, small-model recipes, local SFT path, dolma2 vocab).
  It **corrected three things** in the first draft:
  - **`<category>` must be a capability band, not a tool family.** Two path levels is the whole
    budget; tool family in the path makes per-capability edu numbers unsliceable. Tool family moves
    to a top-level row field.
  - **§7's control-token claim was wrong.** `grep control_tokens` returns **zero hits** on this
    branch — that registry lives on the unmerged `latent-superposition-module`. Verified headroom:
    **74 free ids `100278`–`100351`, zero claimed** (`src/olmo_core/data/tokenizer.py:84-94`).
  - **A separate `eval/` dataset for benchmark items is blocked** — `eval-items/v1` is unregistered
    and raises `ProfileError`. v1 eval = the `heldout` partition. PRD Deliverable 2 reworded.
- **Filled in:** 40,000 rows (floor 25k = ToolACE's controlled datapoint, ceiling 67.5k = Hammer);
  general 27,900 / edu 12,100; abstention **10%** (Hammer's swept optimum at 1.5B); 14-cell
  composition table; plain-text delimiters with the reserved-id option costed but unspent; heldout
  category sizes each ≥ their BFCL counterpart's n; **1:1** general-SFT mixing ratio (Alopex
  arXiv 2411.05209, measured on 1.5–2B); 16 machine-checkable verification gates.
- **Licensing narrows provenance sharply:** only ToolACE (~4,000 usable) and Hermes single-turn
  (~1,890) are cleanly reusable, so the real split is ~15% reformat / ~17% derived / **~68% fresh
  synthesis** — not "fill the gaps". xLAM blocked on a `cc-by-4.0`-tag-vs-research-only-prose
  conflict; ToolBench and Glaive excluded on quality.
- **Nothing in this repo can read our `.jsonl`.** No `.jsonl` reader under `src/olmo_core/`;
  consumption *is* wired (`label_mask_paths` `numpy_dataset.py:406`, `get_labels()`,
  `src/scripts/train/sft/`) but the converter lives in AI2's external `open-instruct`. Blocks use,
  not publishing. Trap recorded: exactly **one EOS per conversation** or packing corrupts silently.
- **New highest-value open question:** `src/scripts/train/sft/README.md:52-64` shows AI2's OLMo-3 SFT
  mix already contains tool-use data (`allenai/olmo-toolu-*`) and loads its chat template *from the
  tokenizer*. If `dolma-2-tokenizer-olmo-3-instruct-final` already defines tool-call delimiters we
  should match it, not invent one — could change the record format, tokens and sources.
- Note: the first research workflow was killed after stalling 15 min — a subagent called
  `EnterWorktree` inside this worktree-isolated session and wedged it. Relaunched read-only.
- Moved the two verification probes from gitignored `scratch/` into tracked `docs/tool-call/verify/`
  — the design doc cites them as reproducible evidence, so they cannot be gitignored.
- Pushed `593fa970` to `origin/tool-call-amy`. **Fixed a footgun:** `git worktree add -b … origin/main`
  left the branch tracking `origin/main`, so a bare `git push` would have targeted **main**. Re-pointed
  upstream with `git push -u origin tool-call-amy`. (Note for later: a GPU run needs a branch named
  `edullm/<something>` for the platform to build an image — `tool-call-amy` will not do for that.)

- Chased the `allenai/olmo-toolu-*` lead. **All five are HTTP 401** (control: `tulu-3-sft-mixture`
  returns 200 from the same client, so the 401 is real). "Use their dataset" is foreclosed for those.
- **But their public non-thinking equivalent exists:** `allenai/Dolci-Instruct-SFT-Tool-Use` —
  **227,579 rows**, 2.54 GB, `private:false gated:false`, features `messages`/`dataset_source`/`id`.
  ODC-BY is stated **only in the description prose; no `license:` key in frontmatter**.
- **Adopted OLMo 3's wire format, verbatim.** Verified against the live tokenizer: `<functions>` /
  `</functions>` / `<function_calls>` / `</function_calls>` are **single non-special token ids
  100266–100269**, carved in place over `<|extra_id_1..4|>` at **unchanged vocab 100278**. So
  `dolma-2-tokenizer-olmo-3-instruct-final` (a **307 alias** → `olmo-3-tokenizer-instruct-dev`,
  sha `55f211df…`) is embedding-compatible with any dolma2-pretrained checkpoint — no resize.
  Calls are **Pythonic** (`get_weather(city="Boston")`), parallel calls share **one** block joined by
  a bare `\n`, and tool results come back on role **`environment`** with **no wrapper**.
- **Third correction to §7:** the draft-2 proposal to reserve ids `100344`–`100351` is deleted. There
  is nothing to reserve — OLMo already carved the delimiters inside the real vocab. (Max tokenizer id
  is 100277; the 74 padding rows above it exist but nothing can emit them without tokenizer surgery.)
- **We adopt OLMo's rendered bytes, not its row layout.** Verified against a real Dolci row: AI2 puts
  the call in a sibling `function_calls` field with **`content: null`** — measured `COLLIDE=True`, so
  our validator would be blind to it and `max_leakage: 0` could refuse the publish. We inline all
  three payloads into `content`; the template emits `content` verbatim so the token stream is
  byte-identical, and the call tokens end up **trainable** while schema tokens stay masked.
- **Provenance revised:** 31.5% reformat / 17.5% derived / **51% fresh** (was 15/17/68). Dolci 10,600
  + ToolACE 2,000; **Hermes dropped** (its only value was being the format reference, which Dolci now
  is, with 120× the rows).
- **Rejected the thinking traces.** `<think>` is not a token (plain text, strippable), but OLMo 3's
  only published tool-use BFCL number is 7B Instruct **49.8** vs Qwen3-8B 60.2, with **no Think
  number at any size** — no evidence traces buy tool accuracy here. The one public thinking tool set
  has a **self-contradictory licence** (`cc-by-sa-4.0` frontmatter vs ODC-BY prose) and is 1,597
  browse trajectories, so it will not serve the latent-reasoning work either. CODI keeps its own id.
- **New blocker found for v2:** the open-instruct converter's prefix-stability check should **fail on
  any conversation with ≥2 assistant turns**, because a non-final assistant turn renders `<|im_end|>`
  in the full pass but `eos_token` (`<|endoftext|>`) in the sub-pass. v1 is single-turn so it is safe.
  INFERRED — confirm empirically before planning multi-turn.
- Rewrote `dataset-design.md` §3, §6, §7, §8, §12, §14, §15 and added §16; updated `prd.md` and
  `docs/tool-call/verify/verify_record_shape.py` (now diffs our layout against AI2's real one).

- **§15 Q1 ANSWERED — byte-identity proven.** New tracked check
  `docs/tool-call/verify/verify_render_identity.py` fetches the real
  `Olmo-3-7B-Instruct/chat_template.jinja` and 5 real Dolci rows, renders each **as AI2 publishes it**
  (sibling `functions`/`function_calls`, `content: null`) and **as we inline it**, then byte-compares:
  `IDENTICAL=True` on all 5, including a 27-turn row. The inlining is lossless.
  - The leading-space question is settled **from the template source**, not inferred: the
    per-message path emits `' <functions>'` **with** a space, the row-level `tools` path emits
    `'<functions>'` **without** one. We use the per-message spacing, which is the path AI2's own
    training bytes took.
  - Also read out of the template: `tool` role **is** aliased to `<|im_start|>environment`; a
    `tool_calls` branch exists that builds Pythonic calls as `name(k=v, …)` joined by `', '` with
    parallel calls joined by `\n` — confirming our serialization exactly.
  - Note for the harness: the jinja env must use **transformers' `tojson`**, not jinja's builtin —
    the builtin is HTML-safe and escapes `< > & '`, which would corrupt schemas.

- **Three must-haves mandated** (arithmetic tools, web search, pedagogy focus). Respent the domain
  axis rather than adding a path level: `<domain>` ∈ `general` | `arithmetic` | `web-search` |
  `pedagogy` (`edu` **renamed**, not added), so each must-have is independently glob-sliceable.
  **Still exactly two path levels** — only the vocabularies changed.
- **Eighth category added: `answer-directly`.** Forced, not cosmetic: `no-suitable-tool` *deletes* the
  gold function (BFCL `irrelevance`), whereas `answer-directly` keeps a plausible tool **present** and
  calling it is still wrong because the answer is settled knowledge. That is exactly the "know
  learning science, don't search for it" requirement. 32-cell grid, no holes.
- **Irreversible labelling rule:** a row's domain is the **gold tool's** domain, not the user turn's
  topic. Gate 34 enforces it. Domain totals therefore measure gold-tool domain, not inventory mix.
- Composition rebuilt as a 32-cell grid at 40,000: general 15,000 / arithmetic 7,000 /
  web-search 7,000 / pedagogy 11,000. Abstention held at **exactly 10%**, now split three ways.
  Provenance moves to **24% reformat / 14.25% derived / 61.75% fresh** — the honest price of the
  mandate, since the three new domains have **no reusable upstream at all** (+4,300 rows of work).
- **Two new tracked docs:** `tool-inventories.md` (64 schemas across 4 domains, 10 held out, with the
  raw-expression-vs-operands and single-turn-search decisions) and `pedagogy.md` (pedagogy tools vs
  prose vs knowledge; the principle taxonomy; the myth negatives).
- **Fixed six stale gates in §10** — they still named `<tools>`/`<tool_call>`, the `tool` role, and a
  name-first JSON payload, all superseded by §3. Left alone they would have mis-gated all 40,000 rows.
  Added gates 17–35 (value execution, expression safety, freshness agreement, taxonomy closure, myth
  bank, domain↔gold-tool, reasoning-prefix shape).
- **Honesty items recorded rather than smoothed over:** only **11 of 20** learning-science principles
  are decidable on a single-turn row (publish 11); **0 of 14** web-search tools are value-executable;
  `multi-tool-select` heldout 845 **< `live_multiple` 1,037**, so the "never worse CI" rule now fails
  for that one category; domain is **confounded with provenance**; and `calculator`/`web_search`
  cannot be held out, so "heldout measures schema generalization" is false for those cells.
- **Alpha School / 2 Hour Learning claims go in `web-search/*`, never `answer-directly`** — the 2.6×
  MAP figure is company-sourced and not independently reviewed, and `MYTH.10` is the Bloom 2-sigma
  claim itself, so the canon and the caution are the same rows. An educational model repeating an
  unreviewed effect size about its own vendor is the failure most worth designing out.
- Dolci sampling (100 rows): **only 10% are 3-turn**, so the single-turn filter is severe (~22,750
  usable, enough for 8,100). Every sampled row's `dataset_source` is
  `allenai/olmo-toolu-sft-mix-…-bfclv3-decontaminated` — **Dolci is the public release of the private
  401 mix**, and it already contains arithmetic-shaped tools (`cosine_similarity`,
  `combinatorics.permutation_count`, `physics.final_velocity`) and search-shaped ones
  (`top_headlines`). Pedagogy: essentially nothing, hence 100% fresh there.

- **CORRECTION: multi-turn is not blocked.** An earlier entry called it "a converter bug our data
  format cannot dodge" — wrong. `verify/verify_multiturn_mask.py` reproduces the mechanism (a non-final
  assistant turn renders `<|im_end|>\n` in the full pass but `eos_token` in the sub-pass, so
  `rendered.startswith(through)` fails at byte 233; **8/8 sampled real Dolci rows fail**) — and then
  reading `open_instruct/dataset_transformation.py:1212,1248` shows a **`last_turn_only`** flag and a
  second exported transform `last_turn_tulu_tokenize_and_truncate_v1` that skips non-final assistant
  turns. The suspicious part was always that AI2 trains 21-turn rows; they do, via that transform.
  - Cost of that route: only the **final** assistant turn is trainable — 1 of 10 on a 21-message row,
    which throws away most of the signal for a *tool-calling* set.
  - **Recommended instead:** our own producer builds the mask **by construction** (render turn-by-turn,
    tokenize each segment, concatenate) — no offset mapping, no prefix requirement, all turns
    trainable. Safe because every segment boundary is an atomic added-token, so no BPE merge can
    straddle it.
  - **Rejected on evidence:** prefix-splitting at assistant boundaries — the longer rows still contain
    a non-final assistant turn and still raise. Forcing `eos = <|im_end|>` also fixes stability but
    drops the 100257 document boundary that packing depends on.

- **Dolci DROPPED — it has no licence tag at all.** `cardData.license` is **absent** (not "ODC-BY in
  prose", which is what the earlier entry assumed), so it fails our own standard and the pending human
  sign-off is **withdrawn, not deferred**. That removed 8,100 rows = 20.25% of the dataset.
- **A wider hunt more than replaced it**, all frontmatter-tagged and verified via
  `verify/verify_sources.py`: `argilla/Synth-APIGen-v0.1` (apache-2.0, 49,402, **100% single-turn**),
  `MadeAgents/xlam-irrelevance-7.5k` (cc-by-4.0, 7,500 — exactly our two hardest categories),
  `nvidia/When2Call` (cc-by-4.0, 27,952), `Agent-Ark/Toucan-1.5M` (apache-2.0),
  `MU-NLPC/Calc-gsm8k` + `openai/gsm8k` (mit — the only real arithmetic reformat source),
  `allenai/math_qa` (apache-2.0), `aialt/RetrievalQA` (mit — the **only** clean-licensed
  search-vs-memory label in existence), `eth-nlped/mathdial` (cc-by-4.0), `allenai/mathfish` (odc-by,
  CCSS metadata only), `allenai/tutormoments-preview` (cc-by-4.0).
- **Provenance: 28.75% reformat / 30% derived / 41.25% fresh** (was 24 / 14.25 / 61.75). Reformat
  *rose* while dropping the source that needed sign-off. ~59% of rows now trace to a licensed record.
  **Reformat is capped near 29% structurally** — web-search and pedagogy have no upstream containing a
  tool call at all, which pins 18,000 of 40,000 rows at 0% reformat.
- **Corrected the framing:** this is not "half synthetic, half scraped". We **scrape nothing** — we
  reformat published datasets under licence. And **only 33% of rows need an LLM at all**; the other 67%
  is field mapping, DSL interpretation, or programmatic generation.
- **Generator: `Qwen/Qwen3-235B-A22B-Instruct-2507`** (primary), `Mistral-Small-3.2-24B` (fallback,
  fits one 80 GB card if the 8×H100 shape is refused or queued). Verified **at the LICENSE file, not
  the tag** — and that check earned its keep: `Qwen2.5-72B-Instruct` has tag `other` and a file reading
  "Qwen LICENSE AGREEMENT" with a "Built with Qwen" display requirement, despite being widely assumed
  Apache. Gemma is **viral** for this use (its ToU counts synthetic-data training as a Model
  Derivative); Anthropic/Google/OpenAI API terms all bar training a competing model; Llama 3.3 permits
  outputs but would force `Llama-` into our model's name.
- **BFCL numbers for the candidates are UNVERIFIED and we stop chasing them** (leaderboard renders
  client-side, aggregators bot-wall). Replaced with a 200-prompt bake-off in our own units — schema-valid
  rate, correct-abstention, argument-name fidelity, distinct-n — including OLMo-3-Instruct as an arm.
- **OLMo-3-Instruct: discriminator, not generator.** The format argument for self-distillation
  dissolves once you notice **no model should emit the delimiters at all** — the generator returns
  structured JSON and *our serializer* renders the wire format, which is correct by construction. And
  its characteristic failure is **over-calling**, which is precisely what `relevance-hard`,
  `no-suitable-tool` and `missing-args` exist to cure. Use it instead as a **difficulty filter**
  (prefer rows it gets wrong) and as the serializer's round-trip check.
- **Two sources we consume are themselves evals** and taking them burns them for this model:
  `aialt/RetrievalQA` (no train split at all) and `nvidia/When2Call` (train only). Recorded here and
  destined for the dataset card. Also noted: `Eedi/…-Tutoring-Dialogues-2k` is the best pedagogy
  content in existence, is `cc-by-nc-4.0`, and is therefore unusable — do not revisit.

- **Multi-turn fix BUILT** — `src/scripts/data/tool_call_sft_producer.py`, with 24 tests in
  `src/test/scripts/tool_call_sft_producer_test.py`. Mask is built **by construction**: segments with
  known character spans, tokenize the concatenation **once**, mark a token trainable when its span
  overlaps a trainable segment. No prefix re-render, so the `loop.last` divergence cannot arise and
  **every assistant turn stays trainable** (verified: 5 assistant turns → 5 trainable spans, where
  `last_turn_only` would give 1).
  - Safety property, asserted by `--self-check`: the concatenated segments equal what the real
    `chat_template.jinja` renders, **byte for byte**, on single-turn, multi-turn, abstention and
    5-assistant-turn cases. A template change becomes a loud failure, not a silent mis-mask.
  - End-to-end with the real tokenizer: exactly **one EOS per conversation** (packing boundary
    intact), arrays **headerless**, equal lengths, delimiters resolving to 100266/100268/100269, max
    id 100269 ≤ 100277, terminator trainable, and **no `<|im_start|>` ever trainable**.
  - Guardrails: rewrites `tool` → `environment`; rejects unknown roles, null/empty `content`, and any
    conversation not ending on an assistant turn — each with the reason it corrupts something.
- **Token budget MEASURED** (`verify/measure_token_budget.py`, 100 real rows + real tokenizer):
  single-turn rows are **median 485 / mean 513 tokens**, and the masked schema block alone is a
  **median 49% of the row**. 40,000 rows ⇒ **~19.4–20.5M tokens**, of which only the assistant turns
  train. Treat as a **floor** — our pedagogy schemas nest deeper than Dolci's general-purpose ones.
  `seq_len=4096` fits every sampled row; 2048 would clip 9% of multi-turn.

<!-- next entry goes below -->
