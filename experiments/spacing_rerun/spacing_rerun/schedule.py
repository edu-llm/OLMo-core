from __future__ import annotations

import collections
import math
import random

from .common import digest, require

ARMS = ("NONE", "UNI", "EXP", "MASS", "GEN")


def offsets(arm, span):
    require(span >= 21 and span % 21 == 0, "span must be a positive multiple of 21")
    return {"UNI": [0, span // 3, 2 * span // 3, span],
            "EXP": [0, span // 7, 3 * span // 7, span],
            "MASS": [span - 3, span - 2, span - 1, span]}[arm]


def ordered_pool(facts, role, seed):
    pool = sorted(f["id"] for f in facts if f["role"] == role)
    require(pool, f"No {role} facts")
    random.Random(seed).shuffle(pool)
    return pool


def compile_schedule(facts, config):
    batch, cap, cohort = config["batch_size"], config["review_cap"], config["cohort_size"]
    qa_per_step, span = config["qa_per_step"], config["span"]
    require(0 < cohort and 4 * cohort <= cap <= batch - qa_per_step,
            "Capacity requires 4*cohort <= review_cap <= batch - QA")
    require(0 <= qa_per_step < batch, "Invalid QA count")
    require(config["buffer_steps"] >= 1, "A positive common buffer is required")
    old = ordered_pool(facts, "old", config["order_seed"])
    # Spread events over starts; shuffle equal-length bins independently of outcomes.
    by_id = {f["id"]: f for f in facts}
    old.sort(key=lambda k: (len(by_id[k].get("statement_ids", [])) // 8, by_id[k]["event"]))
    # Round-robin events avoids putting all facts of one event at one end.
    groups = collections.defaultdict(list)
    for key in old:
        groups[by_id[key]["event"]].append(key)
    rng = random.Random(config["order_seed"])
    events = sorted(groups)
    rng.shuffle(events)
    old = []
    while any(groups.values()):
        for event in events:
            if groups[event]:
                old.append(groups[event].pop(0))
    starts = {key: i // cohort for i, key in enumerate(old)}
    stagger = max(starts.values())
    require(stagger / span <= config["max_stagger_span_ratio"],
            f"Start width {stagger} dominates span {span}; increase span/cohort capacity")
    steps = stagger + span + 1
    due, times = {}, {}
    for arm in ("UNI", "EXP", "MASS"):
        due[arm] = collections.defaultdict(list)
        times[arm] = {}
        for key in old:
            schedule = [starts[key] + x for x in offsets(arm, span)]
            times[arm][key] = schedule
            for step in schedule:
                due[arm][step].append("old/" + key)
        require(max(map(len, due[arm].values())) <= cap, "Review capacity exceeded")
    new_pool = ordered_pool(facts, "new", config["order_seed"] + 1)
    qa_pool = ordered_pool(facts, "qa", config["order_seed"] + 2)
    plans = {}
    for arm in ARMS:
        rows, ni, qi, extra = [], 0, 0, 0
        for step in range(steps):
            review = list(due[arm if arm in due else "UNI"].get(step, []))
            if arm == "GEN":
                review = ["gen/" + key.split("/", 1)[1] for key in review]
            elif arm == "NONE":
                review = ["new/" + new_pool[(extra + j) % len(new_pool)]
                          for j in range(len(review))]
                extra += len(review)
            row = review
            for _ in range(qa_per_step):
                row.append("qa/" + qa_pool[qi % len(qa_pool)])
                qi += 1
            for _ in range(batch - len(row)):
                row.append("new/" + new_pool[ni % len(new_pool)])
                ni += 1
            rows.append(row)
        # Buffer resets to the same deterministic stream cursor in every arm.
        for step in range(config["buffer_steps"]):
            row = ["qa/" + qa_pool[(steps * qa_per_step + step * qa_per_step + j) % len(qa_pool)]
                   for j in range(qa_per_step)]
            row += ["new/" + new_pool[(step * (batch - qa_per_step) + j) % len(new_pool)]
                    for j in range(batch - qa_per_step)]
            rows.append(row)
        plans[arm] = rows
    result = {"stage2_steps": steps, "buffer_steps": config["buffer_steps"],
              "batch_size": batch, "span": span, "stagger_width": stagger,
              "stagger_span_ratio": stagger / span, "starts": starts,
              "review_times": times, "arms": plans, "config": config}
    validate_schedule(result, facts)
    result["sha256"] = digest(result)
    return result


def validate_schedule(plan, facts, examples=None):
    by_id = {f["id"]: f for f in facts}
    arms, stage2 = plan["arms"], plan["stage2_steps"]
    require(set(arms) == set(ARMS), "Five arms required")
    old = {k for k, f in by_id.items() if f["role"] == "old"}
    expected = collections.Counter({"old/" + key: 4 for key in old})
    multisets = {}
    reference_new = [k for row in arms["UNI"] for k in row if k.startswith("new/")]
    reference_qa = [k for row in arms["UNI"] for k in row if k.startswith("qa/")]
    for arm, rows in arms.items():
        require(len(rows) == stage2 + plan["buffer_steps"], "Unequal step counts")
        realized, all_ids, new_ids, qa_ids = collections.defaultdict(list), [], [], []
        for step, row in enumerate(rows):
            require(len(row) == plan["batch_size"], "Unequal batch shapes")
            for key in row:
                role, fact_id = key.split("/", 1)
                require(fact_id in by_id, f"Unknown fact {fact_id}")
                require(role == "gen" and by_id[fact_id]["role"] == "old" or
                        role == by_id[fact_id]["role"], "Training role leakage")
                require(role != "control", "Never-trained control entered training")
                if role == "old":
                    require(step < stage2, "Old fact entered no-review buffer")
                    realized[fact_id].append(step)
                all_ids.append(key)
                if role == "new":
                    new_ids.append(key)
                if role == "qa":
                    qa_ids.append(key)
        counts = collections.Counter(k for k in all_ids if k.startswith("old/"))
        require(counts == (expected if arm in ("UNI", "EXP", "MASS") else {}),
                f"Incorrect old-fact dose in {arm}")
        if arm in ("UNI", "EXP", "MASS"):
            require(dict(realized) == plan["review_times"][arm], "Realized reviews differ")
            for key in old:
                expected_times = [plan["starts"][key] + x for x in offsets(arm, plan["span"])]
                require(realized[key] == expected_times, "Invalid review shape/endpoints")
        multisets[arm] = collections.Counter(all_ids)
        if arm in ("EXP", "MASS", "GEN"):
            require(new_ids == reference_new, "New stream ordering differs")
            require(qa_ids == reference_qa, "QA stream ordering differs")
        require(rows[stage2:] == arms["UNI"][stage2:], "Buffer differs among arms")
    require(multisets["UNI"] == multisets["EXP"] == multisets["MASS"],
            "Training-example multisets differ")
    for uni, gen in zip(arms["UNI"], arms["GEN"]):
        require([x for x in uni if not x.startswith("old/")] ==
                [x for x in gen if not x.startswith("gen/")], "GEN displaced new/QA")
    if examples is not None:
        totals = []
        for arm, rows in arms.items():
            token_counts = [examples[k]["loss_tokens"] for row in rows for k in row]
            require(len(set(token_counts)) == 1, "Per-example token budget differs")
            totals.append(sum(token_counts))
        require(len(set(totals)) == 1, "Arm loss-bearing token budgets differ")


def stage1_epoch(facts, epoch, config):
    old = ordered_pool(facts, "old", config["order_seed"] + epoch * 1009)
    qa = ordered_pool(facts, "qa", config["order_seed"] + 2)
    capacity = config["batch_size"] - config["qa_per_step"]
    rows = []
    for offset in range(0, len(old), capacity):
        row = ["old/" + key for key in old[offset:offset + capacity]]
        row += ["qa/" + qa[(epoch * len(old) + offset + j) % len(qa)]
                for j in range(config["qa_per_step"])]
        row += ["filler"] * (config["batch_size"] - len(row))
        rows.append(row)
    return rows
