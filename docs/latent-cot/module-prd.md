# PRD: the latent-CoT module as an isolated component

Scope of *this* document: how the latent-reasoning / superposition work lands in this repository
without colliding with the twelve other workstreams. The **science** is specified in
[`latent-cot-superposition-prd.md`](latent-cot-superposition-prd.md); the **procedure** in
[`phase8-runbook.md`](phase8-runbook.md). Changelog: [`progress.md`](progress.md).

## 1. Goal and non-goals

Build **latent space reasoning and superposition** only. Concretely: CODI continuous thoughts
(K=10) on a pretrained OLMo-370M, five arms A0–A4, two pre-registered gates.

Explicit non-goals — nothing here reads, writes, imports or depends on: KDA · xLSTM · GDN2 ·
Mamba-3 · hyperparameter optimization · residuals / hyper-connections · lngram · engram · LIV
convolution layers · curriculum learning · diffusion · MuonH. Several of those workstreams edit
`nn/transformer/` and `nn/attention/` heavily, which is why §3 is a hard constraint rather than a
preference.

## 2. Starting point: the code exists

`origin/edullm/latent-cot-superposition-amy` carries phases 1–8: 44 files, ~6,100 lines, 126 CPU
tests passing, style-clean. So this is a **port with the isolation hardened**, not a build from
zero. Rewriting under the current deadline would be strictly worse.

Bring it over with `git merge --squash` + one commit. Two reasons: a squash commit's only parent is
this branch, so the 1.6 MB of `local/` accidentally pushed in `e0dff82a` never becomes an ancestor
here; and one commit is one revert. Cost: granular history and philote-dev's authorship collapse,
so both are credited in the commit message.

## 3. The isolation contract

Everything the module owns lives in four directories that no other workstream has any reason to
open:

| Path | Contents |
| --- | --- |
| `src/olmo_core/latentcot/` | the module (tokens, cot, loss, train_module, arms, evaluate, probes, preflight, train_driver, data/) |
| `src/scripts/latentcot/` | CLI entry points |
| `src/test/latentcot/` | tests |
| `docs/latent-cot/` | these documents |

Verified: nothing outside `src/olmo_core/latentcot/` imports it, and no core module imports *from*
it. Generated data → `data/` and run outputs → `runs/`, both gitignored.

### Shared files: the whole list, and why each one is unavoidable

**`src/olmo_core/nn/transformer/model.py` — the one core edit, and we are dropping it.**
The amy branch adds a `return_hidden_states: bool = False` parameter to `Transformer.forward` (+9
lines) so a thought can be read out of the residual stream. `Transformer.forward` is the single
most contended function in the repo. It is also unnecessary: register a forward hook on the last
block and read its output. That is byte-identical to what the parameter returned (the parameter's
early `return h` sits right after the block loop), so **checkpoints from the live run stay valid**
and no measured number changes.

The one thing the parameter bought was skipping the LM head on the K thought forwards. Recover that
with `logits_to_keep=1`, already on `main` — it slices `h` to one position *before* the projection
([`lm_head.py:235`](../../src/olmo_core/nn/lm_head.py#L235)), so the waste is a single
`d_model × vocab` matmul per thought step. Negligible against a 370M forward. Net: **zero lines of
shared code changed.** Confined to `_forward_hidden` in `cot.py` — 4 references, one test, one
comment.

**`.edullm/Dockerfile` — append-only, required, and everyone else's problem too.** Adds a prebuilt
flash-attn wheel and `tokenizers` + a baked dolma2 tokenizer cache. Not optional: every `olmo3_*`
config hardcodes `attn_backend=flash_2` and `Attention.__init__` asserts support at construction,
so without the wheel the model cannot be *built* — `run_019fde30` died 11 s in. `main` still has
neither. The two blocks append at the end of the file, so a conflict here is an adjacent-append
conflict, not a semantic one. **This belongs in its own PR to `main`** — any workstream touching an
`olmo3_*` rung needs it.

**`.edullm/run.yaml` — per-branch by construction.** One file, one run spec, every branch
overwrites it. Nothing to coordinate.

**`.gitignore` — one line, `/data/`.** `/local/` is already ignored on this branch.

**`.mcp.json` — dropped.** Personal `sb-aws-creds` tooling; committing it imposes it on everyone.
Stays untracked.

## 4. Steps

1. `git merge --squash origin/edullm/latent-cot-superposition-amy`.
2. Revert `src/olmo_core/nn/transformer/model.py` to `main`; unstage `.mcp.json`.
3. Rewrite `cot.py::_forward_hidden` as a hook + `logits_to_keep=1`; update its docstring, the
   `test_cot.py` reference, and the `verify_checkpoint.py` comment.
4. `pytest -v src/test/latentcot/` and `make checks` both green.
5. One commit, crediting philote-dev. **No push until asked.**
6. Update `progress.md`.

## 5. Done when

- `git diff origin/main --stat` names only the four owned directories plus `.edullm/{Dockerfile,run.yaml}`
  and `.gitignore`.
- `git diff origin/main -- src/olmo_core/nn src/olmo_core/train src/olmo_core/data` is empty.
- 126 tests pass; `make checks` clean (the image build lints the whole repo).

## 6. Out of scope here, tracked in the science PRD

The live pilot `run_019fdf83-5216-70fd-8d7d-3b52d39452c4` (submitted by philote-dev on commit
`0252ee04`), the unscreened peak LR, and the n=1 seed count — all in
[`latent-cot-superposition-prd.md`](latent-cot-superposition-prd.md) §6 and the session context.
Not blockers for the port. Separately unresolved: the `local/` blobs still reachable from
`e0dff82a` on the two old pushed branches, which needs a history rewrite on a branch another person
has pushed to and is therefore not a unilateral call.
