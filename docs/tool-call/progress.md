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

<!-- next entry goes below -->
