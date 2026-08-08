# Pedagogy: how "educational" shows up in the bytes

Companion to [`dataset-design.md`](dataset-design.md). This is the subtle part of the mandate. The
brief was *arithmetic tools, web search, and a pedagogy-focused model* — the first two are tool
inventories ([`tool-inventories.md`](tool-inventories.md)), but "pedagogy-focused" is three different
things and only two of them are tool calls.

---

## 1. Three distinct things, deliberately separated

| # | What | Where it lives | Checkable? |
| --- | --- | --- | --- |
| a | **Pedagogy TOOLS** — standards lookup, gradebook, spaced-repetition scheduling, misconception diagnosis | `pedagogy/{single-call,multi-tool-select,parallel-call,nested-args}` — 8,700 rows | Yes, like any tool call |
| b | **Pedagogical PROSE** — the assistant teaches rather than tells | `pedagogy/*` assistant content, and the ≤2-sentence reasoning prefix everywhere | **Partly** — 11 of 20 principles are decidable on a single row |
| c | **Learning-science KNOWLEDGE** — the model knows the canon, and does not repeat the myths | `pedagogy/answer-directly` — 600 rows | Yes for myths (negative regex + required refutation); no for "is this good pedagogy" |

The mandate said the model should know learning science *if it is not web-searchable*. That maps
exactly onto (c): **settled science is answered from parametric knowledge with no tool call**, and
that is what `answer-directly` exists for. Current or specific facts still route to `web_search`.

---

## 2. Example rows

**(a) pedagogy tool call** — spaced repetition, which is a real computation:

```json
{"messages":[
 {"role":"system","content":"You are a helpful function-calling AI assistant. <functions>[{\"type\":\"function\",\"function\":{\"name\":\"schedule_review_sm2\",\"description\":\"Next review interval under SM-2.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"quality\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":5},\"repetitions\":{\"type\":\"integer\"},\"easiness_factor\":{\"type\":\"number\",\"minimum\":1.3},\"interval_days\":{\"type\":\"number\"}},\"required\":[\"quality\",\"repetitions\",\"easiness_factor\",\"interval_days\"]}}}]</functions>"},
 {"role":"user","content":"Maya recalled the card with some hesitation — call it a 3. It's her 4th review, EF 2.36, last interval 12 days. When should she see it again?"},
 {"role":"assistant","content":"<function_calls>schedule_review_sm2(quality=3, repetitions=4, easiness_factor=2.36, interval_days=12)</function_calls>"}
]}
```

**(b) settled science → answer directly, no call** (`pedagogy/answer-directly`). The inventory
*offers* `web_search`; calling it is the wrong answer:

```json
{"messages":[
 {"role":"system","content":"You are a helpful function-calling AI assistant. <functions>[{\"type\":\"function\",\"function\":{\"name\":\"web_search\",\"description\":\"Search the live web.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}}}]</functions>"},
 {"role":"user","content":"Should I have my students reread the chapter or quiz themselves on it?"},
 {"role":"assistant","content":"Quiz themselves. Retrieving something from memory strengthens later recall more than re-exposing yourself to it — the testing effect. Rereading feels productive because the text gets easier, but that fluency is recognition, not retrieval. Dunlosky et al. rate practice testing high-utility and rereading low-utility. Make the quiz low-stakes so a wrong answer costs nothing but the retrieval attempt."}
]}
```

**(c) needs current facts → search** (`web-search/single-call`, *not* `pedagogy`, per §4's gold-tool
labelling rule):

```json
{"messages":[
 {"role":"system","content":"You are a helpful function-calling AI assistant. <functions>[…web_search…]</functions>"},
 {"role":"user","content":"What did the second edition of Deans for Impact's Science of Learning change?"},
 {"role":"assistant","content":"<function_calls>web_search(query=\"Deans for Impact Science of Learning second edition changes\")</function_calls>"}
]}
```

That pairing is deliberate: **~20% of myth/canon targets get a genuine web-search sibling**, so we do
not teach "never search about learning science."

---

## 3. The tagging scheme, and its honest limit

Every pedagogy row carries a nested `pedagogy` object with ids drawn from the **frozen taxonomy**
([`frozen/`](frozen/), §17 of the design doc). Ids are never renumbered — deprecate instead.

**Of 20 canonical principles, 11 are decidable on a single v1 row.** Publish that number, never 20.

| Decidable | The predicate that makes it checkable |
| --- | --- |
| retrieval practice | prose contains an interrogative whose answer (in `answer_key.expected_learner_response`) does **not** appear anywhere in the row — a violation fires if it does, because "What's the derivative? It's 2x" is re-exposure, not retrieval |
| generation effect | prose contains a cloze marker (`___`, "fill in") **and** a non-empty expected response |
| interleaving *(structural half only)* | the call's arguments carry ≥3 items spanning ≥2 distinct skill ids, no two adjacent sharing one |
| dual coding | a non-text representation (table, LaTeX, ASCII figure) **and** ≥1 referent named in prose within 200 chars |
| worked example | ≥2 enumerated steps, ≥1 rationale connective per step, `learner_level == "novice"` |
| extraneous load | *negative only* — a non-text representation with zero referential tie is decoration, and that is a violation, not an absence |
| mastery learning | an explicit numeric `mastery_threshold` plus a progression-gate verb |
| Bloom level | exactly one `bloom_level` + one `knowledge_type`, verbs resolved against a **frozen verb→level table** so the labeller cannot improvise |
| expertise reversal | **pairwise** — `pair_id` twice, arms differ in `learner_level`, computed support density strictly greater in the novice arm |
| scaffolding | reachable in v1 *only* through that matched pair |
| concreteness fading | likewise, only via the pair |

**Four are policy-negatives only** (we reject a violation, we cannot certify presence). **Four are
structurally unlabelable in single-turn v1** — spacing across sessions, the doer effect's
production/exposition ratio, and two others need turn structure we do not have. Where a real tag is
unreachable we use a **named proxy** and mark the real tag unlabelable. We do not fake it.

### Not checkable — do not claim it

- Whether the pedagogical move is right **for this learner**. Mitigated constructively: generate the
  user turn *from* the chosen principle so the label is true by construction.
- Whether prose is level-appropriate, whether the described learner state is realistic, whether a
  clarifying question is the best one.
- Whether a hedge on a contested claim is **proportionate**. We check a conditioning token is
  present; presence is not sufficiency.
- Whether a citation is **correct** — only that its id is in the frozen table, and rows may not emit
  volume/page numbers for entries flagged unverified.

---

## 4. Myths: the negative set the model must never assert

All in `pedagogy/answer-directly` (600 rows), because a correction is settled science and needs no
tool. Allocation: **10 myths × 25 = 250**, 5 low-utility strategies × 15 = **75**, 10 contested
targets × 20 = **200** hedged rows, **75** plain direct answers.

| id | The claim | Refutation anchor |
| --- | --- | --- |
| `MYTH.01` | matching instruction to a learner's modality helps (meshing hypothesis) | Pashler et al. 2008; Cuevas & Dawson 2018 — no style × condition interaction |
| `MYTH.02` | VAK/VARK validly measures a style | Coffield et al. 2004; Kirschner 2017 |
| `MYTH.03` | people are left- or right-brained as a learning type | Nielsen et al. 2013. **Careful: hemispheric specialization is real; the personality extrapolation is the myth** |
| `MYTH.04` | we use 10% of our brains | Beyerstein 1999 |
| `MYTH.05` | Mozart raises intelligence | Pietschnig et al. 2010 — KV448 d=0.37 vs any other music d=0.38 |
| `MYTH.06` | digital natives are inherently information-skilled and can multitask | Kirschner & De Bruyckere 2017; Sana et al. 2013 |
| `MYTH.07` | learners can pick their own optimal path unaided | Kirschner & van Merriënboer 2013 |
| `MYTH.08` | content must be withheld as "developmentally inappropriate" | Deans for Impact 2015 |
| `MYTH.09` | 10,000 hours; practice explains most of expertise variance | Macnamara et al. 2014 — 26% games, 21% music, 18% sports, **4% education** |
| `MYTH.10` | 1:1 AI tutoring delivers Bloom's 2 sigma | VanLehn 2011 threshold confound; replication d=0.52 |
| `LOWU.01–05` | summarizing, highlighting, keyword mnemonics, imagery-for-text, rereading as *primary* strategies | Dunlosky et al. 2013 low-utility five |

### Three rules a careless reading would get wrong

1. **The myth goes in the USER turn, never the assistant's.** A paired
   `{correct, confident_overclaim}` design taken literally would publish an overclaiming assistant
   turn *as a training target*. It must not. The overclaim arm exists in exactly two places: inside
   the **user turn** of a correction row, and as a **fixture in the verifier's own test suite**
   proving gate 29 fires.
2. **Gate 29 computes over assistant content only**, and a hit rejects *unless* the row is in
   `*/answer-directly`, names the id in `myth_corrected`, **and** matches ≥1 token from that myth's
   frozen refutation bank. Without the co-occurrence requirement, "learning styles" appearing in a
   correction is indistinguishable from an assertion.
3. **`MYTH.10` is the one this model is most likely to emit about itself**, and it needs weighting
   accordingly. Which leads to the next section.

---

## 5. The organization's own claims are `web-search`, never parametric

Alpha School / 2 Hour Learning figures circulating publicly — the 2.6× MAP growth multiple most
prominently — are **company-sourced and not independently reviewed**. Training them as parametric
knowledge would install marketing copy as fact, in a model whose whole purpose is to be trustworthy
about learning.

**Rule:** any question about the organization's own product, results, or claims belongs in
`web-search/*`, never `pedagogy/answer-directly`. The model should know *how to find out*, and should
know the learning-science canon those claims invoke (mastery learning, Bloom's 2-sigma, adaptive
practice, spaced repetition) **including the confounds** — `MYTH.10` is exactly the 2-sigma claim, so
the canon and the caution are the same row set.

This is the respectful version of "fits him": fluent in the vocabulary, accurate about the evidence,
and not a press release. An educational model that repeats an unreviewed effect size about its own
vendor is the specific failure most likely to embarrass everyone.

---

## 6. What is still open

| # | Question | Blocks |
| --- | --- | --- |
| 1 | `pedagogy/answer-directly` is the only category where the assistant **asserts facts**. k=8 consistency is a proxy and k is UNVERIFIED. | Publishing those 600 rows without a **curated fact bank**. This is the cell most likely to install a confident falsehood. |
| 2 | Pedagogical explanation *after* a tool result needs `assistant → environment → assistant`, which the converter cannot tokenize (design §12). | The doer effect, feedback timing, and the tool-result-to-teaching move — arguably the product surface. All v2. |
| 3 | Several standards-API field lists (CASE association types, Eedi column spellings) are UNVERIFIED. | Claiming the schemas are *current*. Pin a retrieval date per schema in the frozen registry. |
| 4 | Only 11 of 20 principles are decidable. | The claim "grounded in the 20-principle taxonomy". The publishable number is **11**. |
