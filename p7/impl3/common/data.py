"""Data plumbing for SI-conditioned SFT (PRD §2.1–2.5).

The training data is the published POC mix ``meric533/socrateach-sft`` (30k train:
75% Socratic pedagogy + 25% SI-free general replay; plus validation/test). Point any
entrypoint at it with ``hf_dataset`` (default in the impl configs) or snapshot it to
local JSONL with ``snapshot_hf_dataset.py`` and use ``--data_dir``.

This module also keeps the *recipe* helpers (method, not data) used to build a mix
from raw sources via ``prepare_data.py``:

  - ``assemble_pedagogy_example``  : prefix a per-dialogue System Instruction onto a
                                     pedagogy dialogue (uses common.system_instructions).
  - ``normalize_general``          : strip the system message from a general/replay
                                     conversation (general data is SI-free — PRD §2.3).
  - ``is_english``                 : language filter for general replay data.
  - ``make_group_splits``          : split grouped BY PROBLEM (no leakage; PRD §2.5).
  - ``co_train_mix``               : mix ~75% pedagogy / 25% SI-free general (PRD §2.3).
  - ``load_hf_sft_datasets`` / ``build_sft_datasets`` / ``load_jsonl`` : train-time IO.
"""
from __future__ import annotations

import json
import os
import random
import re
import unicodedata

from .system_instructions import build_system_instruction

# Candidate local filenames per split, tried in order. The first keeps the POC's
# SocraTeach names so old data drops in unchanged; the second is the generic name
# written by the HF snapshot script (snapshot_hf_dataset.py).
SPLIT_FILE_CANDIDATES = {
    k: [f"socrateach_sft_{k}.jsonl", f"sft_{k}.jsonl"] for k in ("train", "val", "test")
}
# Back-compat alias (first candidate) for any external caller.
SPLIT_FILES = {k: v[0] for k, v in SPLIT_FILE_CANDIDATES.items()}


def _resolve_split_path(data_dir, split):
    for name in SPLIT_FILE_CANDIDATES[split]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    # Return the primary candidate so the missing-file error message is sensible.
    return os.path.join(data_dir, SPLIT_FILE_CANDIDATES[split][0])


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_hf_sft_datasets(hf_dataset, *, train_cap=0, seed=13, require_test=False):
    """Load SFT splits straight from a HuggingFace Hub dataset id.

    Used to reuse a published SFT dataset (e.g. ``meric533/socrateach-sft``, which ships
    ``train`` / ``validation`` / ``test`` splits) without staging local files. Rows must
    carry a ``messages`` list; a leading ``system`` turn is optional and any extra columns
    (e.g. ``kind``, ``problem_id``) are kept — the tokenizer only reads ``messages`` (and
    Impl 3 reads ``kind``).

    NOTE: this pulls from the Hub, so run it where there is network (ORCD login node or
    a compute node with internet). For offline compute nodes, snapshot to local JSONL
    first with ``snapshot_hf_dataset.py`` and point ``--data_dir`` at that folder.
    """
    from datasets import load_dataset

    raw = load_dataset(hf_dataset)
    train_key = next((k for k in ("train", "sft_train") if k in raw), None)
    val_key = next((k for k in ("validation", "val", "valid", "dev", "sft_val", "test") if k in raw), None)
    if train_key is None:
        raise KeyError(f"No train split found in {hf_dataset!r}; got splits {list(raw)}")
    if val_key is None:
        raise KeyError(f"No validation/test split found in {hf_dataset!r}; got splits {list(raw)}")

    train_ds = raw[train_key]
    if train_cap and len(train_ds) > train_cap:
        train_ds = train_ds.shuffle(seed=seed).select(range(train_cap))

    out = {"train": train_ds, "val": raw[val_key]}
    if require_test:
        test_key = next((k for k in ("test", "sft_test") if k in raw and k != val_key), None)
        if test_key is None:
            raise KeyError(f"require_test but no test split in {hf_dataset!r}; got {list(raw)}")
        out["test"] = raw[test_key]
    print(f"Loaded HF dataset '{hf_dataset}': "
          + " | ".join(f"{k}={len(v)} ({train_key if k=='train' else val_key})" for k, v in out.items()))
    return out


def build_sft_datasets(data_dir, *, train_cap=0, seed=13, require_test=False, hf_dataset=None):
    """Load prepared SFT splits into HuggingFace ``Dataset`` objects for training.

    If ``hf_dataset`` is set, splits are pulled from the Hub (see
    ``load_hf_sft_datasets``). Otherwise local JSONL files under ``data_dir`` are used.
    Each row must have a ``messages`` list. Optional ``kind`` ("pedagogy"/"general")
    and per-token ``weights`` fields are preserved if present.
    """
    from datasets import Dataset

    if hf_dataset:
        return load_hf_sft_datasets(hf_dataset, train_cap=train_cap, seed=seed,
                                    require_test=require_test)

    needed = ["train", "val"] + (["test"] if require_test else [])
    paths = {k: _resolve_split_path(data_dir, k) for k in needed}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing prepared data files: " + ", ".join(missing) +
            "\n\nEither pass a Hub id via hf_dataset=/--hf_dataset (e.g. "
            "meric533/socrateach-sft), or snapshot it locally with snapshot_hf_dataset.py "
            "(see common/data.py)."
        )

    train_recs = load_jsonl(paths["train"])
    if train_cap and len(train_recs) > train_cap:
        random.Random(seed).shuffle(train_recs)
        train_recs = train_recs[:train_cap]

    kinds = {}
    for e in train_recs:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1

    out = {"train": Dataset.from_list(train_recs), "val": Dataset.from_list(load_jsonl(paths["val"]))}
    if require_test:
        out["test"] = Dataset.from_list(load_jsonl(paths["test"]))
    print(f"Loaded splits from '{data_dir}/': "
          + " | ".join(f"{k}={len(v)}" for k, v in out.items()) + f" | train kinds={kinds}")
    return out


# ---------------------------------------------------------------------------
# Pedagogy: SI-prefixing
# ---------------------------------------------------------------------------
def assemble_pedagogy_example(messages, dialogue_id, *, problem_id=None, answer=None, source=None):
    """Prefix a per-dialogue System Instruction onto a pedagogy dialogue.

    ``messages`` is the SI-free alternating [user(problem), assistant, user, ...]
    conversation (ending on an assistant turn). Returns a training row.
    """
    si = build_system_instruction(messages, dialogue_id)
    return {
        "messages": [{"role": "system", "content": si}] + messages,
        "problem_id": problem_id,
        "dialogue_id": dialogue_id,
        "answer": answer,
        "source": source,
        "kind": "pedagogy",
    }


# ---------------------------------------------------------------------------
# Conversation normalization / validation
# ---------------------------------------------------------------------------
def _clean(text):
    return (text or "").strip()


def is_valid(turns):
    """Alternating user/assistant, starts on user, ends on assistant."""
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return False
    expected = "user"
    for msg in turns:
        if msg["role"] != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def _merge_and_trim(turns):
    merged = []
    for m in turns:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    while len(merged) > 1 and merged[-1]["role"] == "user":
        merged.pop()  # never end on a student turn
    return merged


def normalize_general(messages):
    """Format a general conversation as SI-free alternating user/assistant.

    DROPS any system message so general/replay data carries no System Instruction
    (PRD §2.3; also makes the "SFT-no-SI" eval cell in-distribution).
    """
    turns = [{"role": m.get("role"), "content": _clean(m.get("content"))}
             for m in messages if m.get("role") in ("user", "assistant") and _clean(m.get("content"))]
    merged = _merge_and_trim(turns)
    return merged if is_valid(merged) else None


# ---------------------------------------------------------------------------
# Language filter for general replay data
# ---------------------------------------------------------------------------
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_LATIN_WORD = re.compile(r"[A-Za-z\u00C0-\u024F]{2,}")

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
    _HAVE_LANGDETECT = True
except ImportError:
    _HAVE_LANGDETECT = False


def _nonlatin_alpha_ratio(t):
    latin = nonlatin = 0
    for c in t:
        if not c.isalpha():
            continue
        if ord(c) <= 0x24F:
            latin += 1
        else:
            try:
                latin += 1 if unicodedata.name(c).startswith("LATIN") else 0
                nonlatin += 0 if unicodedata.name(c).startswith("LATIN") else 1
            except ValueError:
                pass
    total = latin + nonlatin
    return (nonlatin / total) if total else 0.0


def is_english(text):
    """Keep English (incl. math/code/reasoning); drop genuine foreign-language content."""
    t = (text or "").strip()
    if not t:
        return False
    if _nonlatin_alpha_ratio(t) > 0.10:
        return False
    prose = _INLINE_CODE.sub(" ", _CODE_FENCE.sub(" ", t))
    if len(_LATIN_WORD.findall(prose)) < 3:
        return True  # essentially no prose (pure code/math/numbers) -> keep
    if not _HAVE_LANGDETECT:
        return True
    try:
        return detect_langs(prose[:2000])[0].lang == "en"
    except LangDetectException:
        return True


# ---------------------------------------------------------------------------
# Splitting (grouped by problem) + co-training mix
# ---------------------------------------------------------------------------
def make_group_splits(per_problem_groups, *, val_frac=0.05, test_frac=0.05, seed=13):
    """Split a list of per-problem example groups so no problem leaks across splits.

    ``per_problem_groups``: list of lists; each inner list is all pedagogy examples
    for one problem. Val/test are pedagogy-only (PRD §2.5).
    """
    groups = list(per_problem_groups)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    def flatten(gs):
        return [ex for g in gs for ex in g]

    return {
        "test": flatten(groups[:n_test]),
        "val": flatten(groups[n_test:n_test + n_val]),
        "train": flatten(groups[n_test + n_val:]),
    }


def co_train_mix(train_pedagogy, general, *, max_total=30000, general_frac=0.25, seed=13):
    """Mix ~(1-general_frac) pedagogy / general_frac SI-free general into TRAIN only.

    Trims pedagogy to the cap, appends ``general`` (already SI-free), shuffles.
    Pass ``general=[]`` and ``general_frac=0`` for a pedagogy-only train split.
    """
    n_general_target = int(round(max_total * general_frac))
    n_ped_target = max_total - n_general_target
    ped = train_pedagogy[:n_ped_target] if len(train_pedagogy) > n_ped_target else list(train_pedagogy)
    mixed = ped + list(general[:n_general_target])
    random.Random(seed + 1).shuffle(mixed)
    if mixed:
        pct = 100 * min(len(general), n_general_target) / len(mixed)
        print(f"train mix: {len(ped)} pedagogy + {min(len(general), n_general_target)} general "
              f"= {len(mixed)} ({pct:.0f}% general)")
    return mixed


# ---------------------------------------------------------------------------
# Reference loaders (OPTIONAL) — plug in your own dataset here.
#
# These mirror the POC (SocraTeach + Tulu). They are NOT called by default because
# we have no data yet. Adapt them, or write your own that yields the same shapes:
#   pedagogy: list of per-problem groups, each a list of assemble_pedagogy_example(...)
#   general : list of rows with SI-free ``messages`` and kind="general".
# ---------------------------------------------------------------------------
def load_pedagogy_groups(source=None):  # pragma: no cover - data hook
    raise NotImplementedError(
        "No pedagogy dataset wired up yet. Provide multi-turn Socratic dialogues and "
        "map each into assemble_pedagogy_example(messages, dialogue_id, ...), grouped by "
        "problem_id. See the POC's prepare_socrateach_sft.py for a worked example "
        "(ulises-c/SocraTeach_Multi)."
    )


def load_general_examples(n, seed, source=None):  # pragma: no cover - data hook
    raise NotImplementedError(
        "No general/replay dataset wired up yet. Provide the base model's own SI-free "
        "SFT/instruction mixture (POC used allenai/tulu-3-sft-olmo-2-mixture-0225), "
        "filtered with is_english(), normalized with normalize_general()."
    )
