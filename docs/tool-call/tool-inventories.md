# Tool inventories

Companion to [`dataset-design.md`](dataset-design.md) (see §4 for the domain axis) and
[`pedagogy.md`](pedagogy.md). **64 authored schemas, 10 held out (15.6%)**, plus ~12% of the
inherited `general` pool.

**Provenance tags:** `REAL` = parameter names lifted verbatim from documented API docs ·
`HYBRID` = real names plus invented composites · `INVENTED` = no upstream.

**`EXEC` column:** **value** = a local Python stub computes the true result, so the row is verifiable
by *comparison* (gate 17) · **bind** = signature-bind plus enum/format/constraint checks only. Only a
value-executable tool catches the coercion and precision errors JSON Schema cannot.

**Carve rule: hold out the sibling, not the orphan.** A held-out orphan measures nothing; a held-out
sibling of a trained tool measures schema generalization.

**Corpus-wide invariant (gate 5): no function name may carry two schemas.** So weather lives only in
`general`; `wolfram_alpha_query` (answer) and `compute_step_by_step` (steps) are two *names* over one
upstream endpoint with different descriptions — a deliberate near-miss pair; `convert_units`
(arithmetic) and `currency_convert` (general) are distinct because currency needs a rate lookup.

---

## `arithmetic` — 18 schemas, 3 held out (16.7%)

| # | name | one-line | key params | src | EXEC |
| --- | --- | --- | --- | --- | --- |
| 1 | `calculator` | Evaluate a single-line arithmetic/algebraic expression. | `expression: string` (req, only param) | REAL (LangChain, numexpr) | **value** |
| 2 | `wolfram_alpha_query` | NL math/science query, Full Results v2. | `input`(req), `output: enum[xml,json]`, `includepodid: array`, `excludepodid`, `podstate`, `scanner`, `units: enum[metric,nonmetric]`, `podtimeout`, `reinterpret: bool` | REAL | bind |
| 3 | `evaluate_wolfram_language` | Evaluate Wolfram Language code. | `input: string`(req) | REAL | bind |
| 4 | `introduce_expression` | Register a SymPy expression, return a handle. | `expr_str: string`(req) | REAL (sympy-mcp) | **value** |
| 5 | `simplify_expression` | Simplify a registered expression. | `expr_key: string`(req, only param) | REAL | **value** |
| 6 | `solve_algebraically` | Solve for a variable. | `expr_key`(req), `variable`(req), `domain: enum[real,complex]` | REAL | **value** |
| 7 | `integrate_expression` | Definite or indefinite integral. | `expr_key`(req), `var`(req), `lower: number`, `upper: number` | REAL | **value** |
| 8 | `differentiate_expression` | Derivative. **HELD OUT** (sibling of #7). | `expr_key`(req), `var`(req), `order: integer` | REAL | **value** |
| 9 | `create_matrix` | Register a matrix. | `rows: array[array[number]]`(req) | REAL | **value** |
| 10 | `matrix_determinant` | Determinant. | `matrix_key: string`(req) | REAL | **value** |
| 11 | `matrix_eigenvalues` | Eigenvalues. **HELD OUT** (sibling of #10). | `matrix_key: string`(req) | REAL | **value** |
| 12 | `convert_units` | Convert a quantity between units. | `value: number`(req), `from_unit`(req), `to_unit`(req), `precision: integer` | INVENTED | **value** |
| 13 | `statistics_summary` | Summary stats over a numeric array. | `data: array[number]`(req, minItems 2), `metrics: array[enum[mean,median,stdev,variance,min,max,sum,count]]`, `ddof: integer` | INVENTED (`ddof` numpy) | **value** |
| 14 | `percent_change` | Percent change. **HELD OUT** (sibling of #15). | `old_value`(req), `new_value`(req), `precision` | INVENTED | **value** |
| 15 | `compound_interest` | Future value with periodic compounding. | `principal`(req), `annual_rate`(req), `times_compounded_per_year: integer`(req), `years`(req), `contribution_per_period` | INVENTED | **value** |
| 16 | `date_difference` | Elapsed time between dates. | `start_date: date`(req), `end_date: date`(req), `unit: enum[days,weeks,months,years]`, `inclusive: bool` | INVENTED | **value** |
| 17 | `round_and_format` | Round/format under an explicit mode. Exists to teach the `6724.0`-vs-`6724` failure. | `value`(req), `mode: enum[half_up,half_even,floor,ceil,truncate]`(req), `digits`, `keep_integer: bool` | INVENTED | **value** |
| 18 | `code_interpreter` | Run Python in a sandbox. Deliberate relevance-hard distractor against #1. | `code: string`(req), `container: object{type: enum[auto], memory_limit, file_ids: array}` | REAL (OpenAI) | bind |

**15 of 18 are value-executable.** That density exists in no other domain, and it is the whole reason
arithmetic is the only slice that can carry an Exec-style number.

### Arithmetic design decisions

**Raw expression string, not structured operands** — for `calculator`, one required `expression`
param. A 4B model emits one string more reliably than a nested operand tree, it is what LangChain and
Wolfram actually expose, and gate 20 (`numexpr` with an empty global dict) makes it safe. The
structured-operand alternative multiplies the argument surface for no measured gain.

**"Compute mentally vs call the tool" is taught in exactly two mirror cells**, and gate 18 keeps them
from contradicting each other:

- `arithmetic/answer-directly` (200 rows) — ≤2 operands, both |x| ≤ 12, operator ∈ {+, −, ×}. The
  model answers `7 × 8` itself. Calling a calculator for this is the wrong answer.
- `arithmetic/{single-call,nested-args,parallel-call}` — ≥1 operand >12 **or** ≥3 operators. Call it.

**The executable-verification win:** `expected_result` is written by running the stub, never typed. So
for 15 of 18 tools, "is this row correct" is a value comparison rather than a shape check — the only
place in the corpus where that is true.

---

## `web-search` — 14 schemas, 2 held out (14.3%)

| # | name | one-line | key params | src | EXEC |
| --- | --- | --- | --- | --- | --- |
| 1 | `web_search` | Search the live web. The only field emitted is the query. | `query: string`(req, only param) | REAL (Anthropic) | bind |
| 2 | `web_search_configured` | Search with caller-policy domain and location constraints. | `query`(req), `allowed_domains: array` / `blocked_domains: array` (**XOR**), `user_location: object{type: enum[approximate](req), city, region, country(ISO-3166-1 a2), timezone(IANA)}`, `max_uses` | REAL | bind |
| 3 | `openai_web_search` | Same intent, filters **nested** under `filters`. **HELD OUT** (sibling of #2 — tests transfer of one concept across two provider spellings). | `query`(req), `filters: object{allowed_domains(max 100), blocked_domains}`, `search_context_size: enum[low,medium,high]`, `user_location`, `external_web_access: bool` | REAL | bind |
| 4 | `perplexity_search` | Ranked results with language/date/recency filters. | `query: string\|array`(req), `max_results: 1-20`, `search_context_size`, `max_tokens`, `country`, `search_language_filter: array(≤20 ISO-639-1)`, `search_domain_filter: array(≤20, ≤253 chars)`, `search_after_date_filter: MM/DD/YYYY`, `search_before_date_filter`, `last_updated_after_filter`, `search_recency_filter: enum[hour,day,week,month,year]` | REAL | bind |
| 5 | `duckduckgo_search` | Keyword search → title/URL pairs. | `keywords: string`(req), `max_results` (default 10), `region` (default `wt-wt`) | REAL (BFCL v4) | bind |
| 6 | `fetch_url_content` | Retrieve one URL. | `url: uri`(req), `mode: enum[raw,markdown,truncate]` | REAL (BFCL v4) | bind |
| 7 | `news_search` | Search news in a publication window. **HELD OUT** (sibling of #4's date filters). | `query`(req), `published_after: date`, `published_before`, `sources: array`, `language` | INVENTED | bind |
| 8 | `academic_search` | Search scholarly sources. | `query`(req), `filters: object{allowed_domains}`, `published_after`, `peer_reviewed: bool`, `max_results` | HYBRID | bind |
| 9 | `local_search` | Location-scoped search. | `query`(req), `user_location`(req), `radius_km`, `open_now: bool` | HYBRID | bind |
| 10 | `image_search` | Search images. | `query`(req), `max_results`, `safe_search: enum[off,moderate,strict]`, `license: enum` | INVENTED | bind |
| 11 | `wikipedia_lookup` | Fetch a stable encyclopedia article. Primary relevance-hard distractor for static knowledge. | `title: string`(req), `language`, `section` | INVENTED | bind |
| 12 | `stock_quote` | Current market data. | `symbol`(req), `fields: array[enum[price,open,high,low,volume,market_cap]]` | HYBRID | bind |
| 13 | `sports_scores` | Scores/fixtures. | `league: enum`(req), `team`, `date: date` | HYBRID | bind |
| 14 | `get_web_search_usage` | Report search-budget consumption. Makes "how many searches have I used" a legitimate non-search call. | `period: enum[day,month]`(req), `group_by: enum[tool,day]` | INVENTED | bind |

**0 of 14 are value-executable.** State that as a limit, not a gap: correctness claims for this
domain stop at call shape and parameter-constraint conformance. There is no Exec-style number for
`web-search` and there never will be in v1.

### Web-search design decisions

Single-turn means we can only teach **"emit the right search call"** — there is no result to read.
The fresh-vs-parametric boundary is therefore carried by three cells:

- `web-search/{single-call,parallel-call}` — `freshness ∈ {slow, fast}`. Needs the live web.
- `*/answer-directly` — `freshness == static`. Settled; answer from knowledge.
- `web-search/relevance-hard` (500 rows) — **the provider-spelling cell**, and the sharpest near-miss
  set in the corpus. One intent ("restrict to these domains") is spelled `allowed_domains` flat
  (Anthropic), nested under `filters` (OpenAI), and `search_domain_filter` with ≤20-entry /
  ≤253-char limits (Perplexity). Choosing the right spelling *is* schema fidelity. Second axis:
  `wikipedia_lookup` vs `web_search` vs `news_search` vs `academic_search` for the same question,
  where recency and authority decide.

**Query quality is made checkable constructively, not judged:** build the query *from* a target
document so the gold query is true by construction, then score against `answer_key`'s
`query_required_terms` / `query_forbidden_terms` plus a no-unresolved-pronoun check.

**Do not train BFCL's literal abstention strings.** BFCL imposes "I cannot answer this question" via
system prompt so its exact-match scorer has a target; training that as the only abstention form
teaches a canned string. Train natural prose and grade heldout abstention **structurally**
(`answer_key.abstain: true` + zero `<function_calls>`), which is strictly stronger. The BFCL strings
can be imposed at eval time by system prompt if a directly comparable number is ever wanted.

---

## `pedagogy` — 20 schemas, 3 held out (15%)

| # | name | one-line | key params | src | EXEC |
| --- | --- | --- | --- | --- | --- |
| 1 | `search_academic_standards` | Search CASE standards frameworks. | `filter: string` (CASE grammar), `limit`, `offset`, `sort`, `orderBy: enum[asc,desc]`, `fields: array` | REAL (CASE 1.1) | bind |
| 2 | `get_standards_framework_package` | Bulk-fetch a framework. | `sourcedId: uuid`(req), `fields: array` | REAL | bind |
| 3 | `get_standard_item` | One standard statement. | `sourcedId: uuid`(req), `fields: array[enum[fullStatement,humanCodingScheme,abbreviatedStatement,educationLevel,CFItemType,subject,conceptKeywords]]` (enum partly UNVERIFIED) | REAL | bind |
| 4 | `get_standard_associations` | Prerequisite/crosswalk edges. **HELD OUT** (sibling of #3). | `sourcedId`(req), `associationType: array[enum[isChildOf,isPeerOf,isPartOf,exactMatchOf,precedes,isRelatedTo,replacedBy,hasSkillLevel,exemplar]]` (UNVERIFIED) | REAL | bind |
| 5 | `resolve_standard_uri` | Resolve an ASN URI to RDF/JSON-LD. | `uri: uri`(req), `accept: enum[text/html,application/rdf+xml,text/turtle,application/ld+json]` | REAL (ASN) | bind |
| 6 | `get_user_grade_items` | One learner's grade-item breakdown. | `courseid: integer`(req), `userid: integer`, `moodlewsrestformat: enum[json,xml]` | REAL (Moodle) | bind |
| 7 | `list_competency_progress` | Competencies + proficiency evidence. | `filters: array[object{column,value}]`, `sort`, `order: enum[ASC,DESC]`, `skip`, `limit`, `context: object{contextid,contextlevel: enum[system,user,coursecat,course,module,block],instanceid}`, `includes: enum[children,parents,self]` | REAL (Moodle) | bind |
| 8 | `list_line_item_results` | Read results for one gradebook column. | `lineitem_url: uri`(req), `user_id`, `limit` | REAL (LTI AGS 2.0) | bind |
| 9 | `create_line_item` | Create a gradebook column. | `lineitems_url: uri`(req), `scoreMaximum: number exclusiveMinimum 0`(req), `label`(req), `resourceId`, `tag`, `startDateTime: date-time`, `endDateTime`, `gradesReleased: bool` | REAL | bind |
| 10 | `post_score` | Post one learner's score. **5 required params — the richest `missing-args` source in the corpus.** | `lineitem_url`(req), `userId`(req), `timestamp: date-time`(req), `activityProgress: enum[Initialized,Started,InProgress,Submitted,Completed]`(req), `gradingProgress: enum[NotReady,Failed,Pending,PendingManual,FullyGraded]`(req), `scoreGiven`, `scoreMaximum`, `comment` | REAL | bind |
| 11 | `get_submission_summary` | Class-level graded/ungraded counts. | `course_id`(req), `assignment_id`(req), `grouped: bool`, `include_deactivated: bool` | REAL (Canvas) | bind |
| 12 | `list_multi_assignment_submissions` | Cross-assignment gradebook view. Widest flat surface. | `course_id`(req), `student_ids: array`, `assignment_ids: array`, `workflow_state: enum[submitted,unsubmitted,graded,pending_review]`, `enrollment_state: enum[active,concluded]`, `submitted_since: date-time`, `graded_since`, `grading_period_id`, `order: enum[id,graded_at]`, `order_direction: enum[ascending,descending]`, `include: array` | REAL (Canvas) | bind |
| 13 | `grade_submission_with_rubric` | Criterion-level grading + feedback. **Deepest nesting in the corpus.** | `course_id`,`assignment_id`,`user_id`(all req), `submission: object{posted_grade, excuse: bool, late_policy_status: enum[late,missing,extended,none], seconds_late_override}`, `rubric_assessment: map[criterion_id → object{points,rating_id,comments}]`, `comment: object{text_comment, attempt, group_comment, file_ids: array}` | REAL (Canvas) | bind |
| 14 | `bulk_update_grades` | Grade many learners asynchronously. **HELD OUT** (sibling of #13). | `course_id`(req), `assignment_id`, `grade_data: map[student_id → object{posted_grade,excuse,text_comment,rubric_assessment}]`(req) | REAL (Canvas) | bind |
| 15 | `list_student_submissions` | Classroom submissions, scope-filtered. | `courseId`(req), `courseWorkId`(req, `"-"` = all), `userId`, `states: array[enum[NEW,CREATED,TURNED_IN,RETURNED,RECLAIMED_BY_STUDENT,STUDENT_EDITED_AFTER_TURN_IN]]`, `late: enum[LATE_ONLY,NOT_LATE_ONLY]`, `pageSize`, `pageToken` | REAL (Google Classroom) | bind |
| 16 | `list_course_roster` | Enrolled students. | `courseId`(req), `pageSize: integer max 100`, `pageToken` | REAL (Google Classroom) | bind |
| 17 | `schedule_review_sm2` | Next review interval under SM-2. | `quality: integer 0-5`(req), `repetitions: integer`(req), `easiness_factor: number min 1.3`(req), `interval_days: number`(req), `review_datetime: date-time` | REAL (SM-2) | **value** |
| 18 | `schedule_review_fsrs` | Next interval under FSRS. **HELD OUT** (sibling of #17). | `rating: integer 1-4`(req), `elapsed_days`(req), `stability`, `difficulty: 1-10`, `desired_retention: 0.7-0.99`, `maximum_interval`, `w: array[number]` | REAL (FSRS) | **value** |
| 19 | `diagnose_misconception_from_distractor` | Name the misconception a distractor encodes. | `QuestionText`(req), `CorrectAnswer: enum[A,B,C,D]`(req), `SelectedAnswer: enum[A,B,C,D]`(req), `AnswerAText`…`AnswerDText`(req), `ConstructName`, `SubjectName`, `top_k: 1-25` | REAL (Eedi; spellings UNVERIFIED) | bind |
| 20 | `compute_step_by_step` | Worked intermediate steps. Deliberate relevance-hard partner for `calculator`. | `input: string`(req), `includepodid: array`, `podstate: array`, `format`, `podtimeout` | REAL (Wolfram v2) | bind |

**Reserve (v2 or top-up):** `lookup_misconception`, `generate_diagnostic_distractors`,
`explain_error_pattern`, `generate_rubric`, `summarize_mastery_by_standard`,
`search_open_educational_resources`, `sync_roster_from_ed_fi`, `emit_caliper_grade_event`.

---

## `general` — inherited pool + 12 authored, 2 authored held out

The general inventory is **inherited, not authored**: Dolci's and ToolACE's own function pool after
the single-turn filter, the ≤2%-per-function cap and our BFCL decontamination. Hold out ~12% of the
*observed* pool by name — the count is data-dependent, so **UNVERIFIED until the filter runs**.

Twelve tools are authored for the 3,200 fresh general rows:
`weather.forecast_weather_api(q, days)` (REAL, verbatim from a Dolci row) ·
`create_calendar_event(title, date, start_time, duration, attendee, topic)` (REAL) ·
`create_github_issue(owner, repo, title, body, labels, assignees)` (REAL) ·
`translate_text(text, source_language, target_language, formality: enum)` (HYBRID) ·
`get_directions(origin, destination, mode: enum, departure_time)` (HYBRID) ·
`currency_convert(amount, from_currency, to_currency, date)` ·
`send_email(to: array, subject, body, cc, bcc, attachments: array[object])` · `create_reminder` ·
`book_flight(…, passengers: object{adults,children,infants}, cabin: enum)` ·
`query_sql(database, query, timeout_s)` · **`set_thermostat` (HELD OUT)** ·
**`place_order` (HELD OUT)**.

`place_order` must also appear as a frequent **positive**, so pizza-ordering does not become the
refusal cue — Glaive's documented failure.

---

## Held-out carving where schema carving is impossible

Two domains have a single dominant tool that **must** be in train: `calculator` and `web_search`.
For those cells, "heldout measures schema generalization" is **false** and must not be written.
Substitute axes, which measure a different generalization claim and must be named as such:

- **arithmetic** — operand-magnitude band (train ≤4-digit, heldout 5–7-digit) and operator-depth band.
- **web-search** — a held-out **entity bank** plus held-out **parameter combinations**.
