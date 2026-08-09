"""Estimate the corpus token count per cell, using real schemas and the real tokenizer.

    python3 docs/tool-call/verify/estimate_corpus_tokens.py

Why not just median-times-rows: the earlier figure extrapolated a median from only 10 single-turn
Dolci rows, and our 32-cell composition is nothing like Dolci's shape. `multi-tool-select` offers
3-20 schemas, `nested-args` uses the deepest schemas we have, and `arithmetic/single-call` offers a
one-parameter calculator. Those differ by an order of magnitude in system-message cost, which is
~half of every row.

So: tokenize real schemas from docs/tool-call/tool-inventories.md, model how many each cell offers,
and sum over the actual row counts from dataset-design.md section 6.
"""

import json

from tokenizers import Tokenizer

TOKENIZER = "allenai/olmo-3-tokenizer-instruct-dev"

PREAMBLE = (
    "You are a helpful function-calling AI assistant. You are provided with function signatures "
    "within <functions></functions> XML tags. You may call one or more functions to assist with "
    "the user query. Output any function calls within <function_calls></function_calls> XML tags. "
    "Do not make assumptions about what values to plug into functions."
)


def fn(name, desc, params, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": params, "required": required},
        },
    }


S = {}  # representative real schemas, one per size class

S["calculator"] = fn(
    "calculator",
    "Evaluate a single-line arithmetic or algebraic expression.",
    {"expression": {"type": "string", "description": "The expression to evaluate."}},
    ["expression"],
)
S["web_search"] = fn(
    "web_search",
    "Search the live web for current information.",
    {"query": {"type": "string", "description": "The search query."}},
    ["query"],
)
S["get_weather"] = fn(
    "weather.forecast_weather_api",
    "Fetches weather forecast and alerts from a weather API.",
    {
        "q": {"type": "string", "description": "Query parameter to specify the location."},
        "days": {"type": "integer", "description": "Number of forecast days.", "default": 3},
    },
    ["q"],
)
S["web_search_configured"] = fn(
    "web_search_configured",
    "Search the web with caller-policy domain and location constraints.",
    {
        "query": {"type": "string"},
        "allowed_domains": {"type": "array", "items": {"type": "string"}},
        "blocked_domains": {"type": "array", "items": {"type": "string"}},
        "user_location": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["approximate"]},
                "city": {"type": "string"},
                "region": {"type": "string"},
                "country": {"type": "string", "description": "ISO 3166-1 alpha-2."},
                "timezone": {"type": "string", "description": "IANA timezone."},
            },
            "required": ["type"],
        },
        "max_uses": {"type": "integer"},
    },
    ["query"],
)
S["perplexity_search"] = fn(
    "perplexity_search",
    "Ranked structured web results with language, date and recency filters.",
    {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        "search_context_size": {"type": "string", "enum": ["low", "medium", "high"]},
        "max_tokens": {"type": "integer"},
        "max_tokens_per_page": {"type": "integer"},
        "country": {"type": "string"},
        "search_language_filter": {"type": "array", "items": {"type": "string"}},
        "search_domain_filter": {"type": "array", "items": {"type": "string"}},
        "search_after_date_filter": {"type": "string", "description": "MM/DD/YYYY"},
        "search_before_date_filter": {"type": "string", "description": "MM/DD/YYYY"},
        "last_updated_after_filter": {"type": "string"},
        "last_updated_before_filter": {"type": "string"},
        "search_recency_filter": {
            "type": "string",
            "enum": ["hour", "day", "week", "month", "year"],
        },
    },
    ["query"],
)
S["post_score"] = fn(
    "post_score",
    "Post one learner's score to a gradebook line item.",
    {
        "lineitem_url": {"type": "string", "format": "uri"},
        "userId": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "activityProgress": {
            "type": "string",
            "enum": ["Initialized", "Started", "InProgress", "Submitted", "Completed"],
        },
        "gradingProgress": {
            "type": "string",
            "enum": ["NotReady", "Failed", "Pending", "PendingManual", "FullyGraded"],
        },
        "scoreGiven": {"type": "number"},
        "scoreMaximum": {"type": "number"},
        "comment": {"type": "string"},
    },
    ["lineitem_url", "userId", "timestamp", "activityProgress", "gradingProgress"],
)
S["grade_with_rubric"] = fn(
    "grade_submission_with_rubric",
    "Grade a submission at criterion level with feedback.",
    {
        "course_id": {"type": "string"},
        "assignment_id": {"type": "string"},
        "user_id": {"type": "string"},
        "submission": {
            "type": "object",
            "properties": {
                "posted_grade": {"type": "string"},
                "excuse": {"type": "boolean"},
                "late_policy_status": {
                    "type": "string",
                    "enum": ["late", "missing", "extended", "none"],
                },
                "seconds_late_override": {"type": "integer"},
            },
        },
        "rubric_assessment": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "points": {"type": "number"},
                    "rating_id": {"type": "string"},
                    "comments": {"type": "string"},
                },
            },
        },
        "comment": {
            "type": "object",
            "properties": {
                "text_comment": {"type": "string"},
                "attempt": {"type": "integer"},
                "group_comment": {"type": "boolean"},
                "file_ids": {"type": "array", "items": {"type": "integer"}},
            },
        },
    },
    ["course_id", "assignment_id", "user_id"],
)

# rows per cell, from dataset-design.md section 6
PLAN = {
    "general": {
        "single-call": 3600,
        "multi-tool-select": 4000,
        "parallel-call": 2200,
        "nested-args": 2800,
        "relevance-hard": 1200,
        "no-suitable-tool": 700,
        "missing-args": 300,
        "answer-directly": 200,
    },
    "arithmetic": {
        "single-call": 2400,
        "multi-tool-select": 1300,
        "parallel-call": 700,
        "nested-args": 1600,
        "relevance-hard": 450,
        "no-suitable-tool": 250,
        "missing-args": 100,
        "answer-directly": 200,
    },
    "web-search": {
        "single-call": 1900,
        "multi-tool-select": 1300,
        "parallel-call": 800,
        "nested-args": 1500,
        "relevance-hard": 500,
        "no-suitable-tool": 350,
        "missing-args": 150,
        "answer-directly": 500,
    },
    "pedagogy": {
        "single-call": 2200,
        "multi-tool-select": 2400,
        "parallel-call": 900,
        "nested-args": 3200,
        "relevance-hard": 1050,
        "no-suitable-tool": 400,
        "missing-args": 250,
        "answer-directly": 600,
    },
}

# Which schema size class each domain's tools resemble, and how many a cell offers.
# (n_schemas is the modelled count; "typical" and "deep" name the representative schema.)
DOMAIN_SCHEMA = {
    "general": ("get_weather", "web_search_configured"),
    "arithmetic": ("calculator", "perplexity_search"),
    "web-search": ("web_search", "perplexity_search"),
    "pedagogy": ("post_score", "grade_with_rubric"),
}

# n_schemas offered, and whether the cell uses the deep schema, per category.
CATEGORY_SHAPE = {
    "single-call": (1, False),
    "multi-tool-select": (8, False),  # design says 3-20; 8 is the modelled middle
    "parallel-call": (3, False),
    "nested-args": (2, True),  # deepest schemas we have
    "relevance-hard": (6, False),  # near-miss distractors
    "no-suitable-tool": (5, False),
    "missing-args": (2, False),
    "answer-directly": (2, False),
}

# assistant content, by category: a call, several calls, or prose
ASSISTANT = {
    "single-call": '<function_calls>weather.forecast_weather_api(q="Boston", days=5)</function_calls>',
    "multi-tool-select": '<function_calls>weather.forecast_weather_api(q="Boston", days=5)</function_calls>',
    "parallel-call": (
        '<function_calls>weather.forecast_weather_api(q="Paris", days=5)\n'
        'weather.forecast_weather_api(q="Madrid", days=5)</function_calls>'
    ),
    "nested-args": (
        '<function_calls>grade_submission_with_rubric(course_id="C1", assignment_id="A7", '
        'user_id="U42", rubric_assessment={"crit_1": {"points": 8, "comments": "Clear reasoning."}})'
        "</function_calls>"
    ),
    "relevance-hard": '<function_calls>web_search(query="2026 tax bracket changes")</function_calls>',
    "no-suitable-tool": (
        "None of the available tools can do that. They cover weather, scheduling and search, but "
        "not currency conversion. If you can tell me the rate you want to use, I can compute it."
    ),
    "missing-args": (
        "I can post that score, but I need the line item URL and the learner's user id before I "
        "can call the gradebook. Which assignment and which student?"
    ),
    "answer-directly": (
        "Quiz them rather than having them reread. Retrieving something from memory strengthens "
        "later recall more than re-exposure does; that is the testing effect. Rereading feels "
        "productive because the text gets easier, but that fluency is recognition, not retrieval."
    ),
}

USER_TURN = "Can you compare the 5-day forecasts for Boston and tell me which day is warmest?"

tok = Tokenizer.from_pretrained(TOKENIZER)


def n(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def schema_block(names):
    arr = [S[x] for x in names]
    return f" <functions>{json.dumps(arr, ensure_ascii=False, separators=(',', ':'))}</functions>"


print("=== per-schema cost (real tokenizer, compact JSON) ===")
for k in S:
    print(f"  {k:24s} {n(json.dumps(S[k], ensure_ascii=False, separators=(',', ':'))):5d} tokens")
print(f"  {'system preamble':24s} {n(PREAMBLE):5d} tokens")
print(f"  {'turn overhead (4 turns)':24s} {'~12':>5s} tokens")

rows_total = 0
tok_total = 0
print()
print("=== per cell ===")
print(f"  {'cell':34s} {'rows':>6s} {'schemas':>8s} {'tok/row':>8s} {'tokens':>12s}")
per_domain: dict[str, int] = {}
for domain, cells in PLAN.items():
    typical, deep = DOMAIN_SCHEMA[domain]
    for cat, rows in cells.items():
        k, use_deep = CATEGORY_SHAPE[cat]
        names = [deep] * min(k, 2) + [typical] * max(0, k - 2) if use_deep else [typical] * k
        sys_txt = PREAMBLE + schema_block(names)
        per_row = (
            n(f"<|im_start|>system\n{sys_txt}<|im_end|>\n")
            + n(f"<|im_start|>user\n{USER_TURN}<|im_end|>\n")
            + n(f"<|im_start|>assistant\n{ASSISTANT[cat]}<|endoftext|>")
        )
        total = per_row * rows
        rows_total += rows
        tok_total += total
        per_domain[domain] = per_domain.get(domain, 0) + total
        print(f"  {domain + '/' + cat:34s} {rows:6,} {k:8d} {per_row:8,} {total:12,}")

print()
print("=== by domain ===")
for d, t in per_domain.items():
    print(f"  {d:14s} {t:12,} tokens  ({100.0 * t / tok_total:4.1f}%)")

print()
print("=== TOTAL ===")
print(f"  rows            {rows_total:12,}")
print(f"  tokens          {tok_total:12,}   ({tok_total / 1e6:.1f}M)")
print(f"  mean tokens/row {tok_total // rows_total:12,}")
print()
print("  Trainable share: only the assistant turn. Everything else is masked.")
train = 0
for domain, cells in PLAN.items():
    for cat, rows in cells.items():
        train += rows * n(f"{ASSISTANT[cat]}<|endoftext|>")
print(f"  trainable       {train:12,}   ({100.0 * train / tok_total:.1f}% of the corpus)")
print()
print("  Sensitivity: multi-tool-select is modelled at 8 schemas of a 3-20 range.")
for k in (3, 8, 20):
    CATEGORY_SHAPE["multi-tool-select"] = (k, False)
    t = 0
    for domain, cells in PLAN.items():
        typical, _ = DOMAIN_SCHEMA[domain]
        rows = cells["multi-tool-select"]
        sys_txt = PREAMBLE + schema_block([typical] * k)
        t += rows * (
            n(f"<|im_start|>system\n{sys_txt}<|im_end|>\n")
            + n(f"<|im_start|>user\n{USER_TURN}<|im_end|>\n")
            + n(f"<|im_start|>assistant\n{ASSISTANT['multi-tool-select']}<|endoftext|>")
        )
    print(f"    at {k:2d} schemas offered: multi-tool-select alone = {t:,} tokens")
