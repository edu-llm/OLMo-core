# Research-image findings, for the platform maintainer

**From:** the latent-CoT superposition experiment (`edullm/latent-cot-superposition-amy`, OLMo-core)
**Date:** 2026-08-07
**Evidence:** run `run_019fde30-1d27-7096-8bd9-3ef9b7748d7b` (gpu-1xa10g, `olmo-core-train`, team
`pre-training`), submitted via workflow run 31221203314

Everything below is either fixed in this repo's `.edullm/Dockerfile` or worked around in
`src/olmo_core/latentcot/`. Nothing here is blocking us. It is written up because **three of the
five items are platform-side and will hit the next repo the same way**, and because two of them
cost a billed GPU allocation to discover, which a build-time check could have caught for free.

---

## 1. The registered image cannot construct the configs the registered repo ships (fixed here)

`run_019fde30` waited 4.5 min for capacity, started, and died **11 seconds later**:

```
RuntimeError: 'FlashAttention2Backend' is missing the flash-attn package or is not
supported on this platform.
  File ".../olmo_core/nn/attention/__init__.py", line 470, in __init__
    backend.assert_supported()
```

Every `olmo3_*` factory in `TransformerConfig` hardcodes `attn_backend=flash_2`, and
`Attention.__init__` calls `assert_supported()` **during construction** — so this is not a
degraded-performance path, it is "the model cannot be instantiated." `olmo2_*` factories are the
same. The image installs `.[wandb]` + `torch` + `boto3` + `edullm-data`; `flash-attn` reaches the
project only through the unused `fa4` extra, and the registered base is a bare python image with
no nvcc, so nothing was going to build it.

**Fixed here** by installing the prebuilt wheel
`flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64` (253 MB, image ~2.94 → ~3.2 GB).
The tuple is not interchangeable and the facts behind it, so nobody has to re-derive them:

| Field | Value | Why |
|---|---|---|
| CUDA major | `cu12` | torch 2.9.0's PyPI wheel pins `nvidia-*-cu12==12.8.x` |
| torch minor | `torch2.9` | the Dockerfile's own `torch==2.9.0` pin |
| python | `cp312` | the registered base is Debian trixie / python 3.12.13 |
| C++ ABI | `abiTRUE` | the only ABI flash-attn publishes for torch 2.9 |

A mismatch on any of them installs cleanly and then fails to import at run time — the same
11-second death by a slower route — so the Dockerfile asserts the CUDA major *before* installing
and re-asserts the torch pin after.

**Platform-side suggestion.** OLMo-core is registered and every training config it offers needs
this package, so a researcher picking `olmo-core-train` with any `olmo3_*` rung currently gets a
guaranteed failure that costs a GPU allocation. Either carry flash-attn in the registered training
base, or add a build-time assertion in the same spirit as
`tools/verify_checkpoint_shape_agreement.py`: construct one config per registered rung and fail
the image build if it raises. That check is cheap, runs where nothing is billed, and would have
turned this into a red build instead of a burnt allocation. It also generalises — the same gate
catches the next dependency a config starts asserting on.

## 2. `tokenizers` is absent, and every encode path needs it (fixed here)

`tokenizers` is **not** in OLMo-core's core dependencies and **not** in the `wandb` extra this
image installs; it arrives only via the `transformers` extra, which nothing here pulls. So
`olmo_core.latentcot.tokens.load_tokenizer()` raises `ImportError`, and it is reached by the first
`LatentCotDataset` access — **the line after** the model construction that already failed. Fixing
item 1 alone would have bought the next 11-second failure rather than a run.

**Fixed here**: `pip install tokenizers huggingface_hub` (the latter named explicitly rather than
relying on `cached-path` pulling it in transitively).

## 3. Hub egress from the training network is undocumented (worked around here)

`load_tokenizer()` calls `hf_hub_download` — a network read at run time, for a ~2 MB
`tokenizer.json`. Whether a training container can reach `huggingface.co` is not something a repo
can determine from its side, and "a run dies reaching for a small json after being handed eight
A100s" is an expensive way to find out.

**Worked around here** in two halves: the image pre-downloads the file at build time (where there
is already network and nothing is billed by the hour) with `HF_HOME=/opt/hf-cache` pinned to an
absolute path so the cache the build writes is the cache the run reads whatever user the container
starts as; and `load_tokenizer()` retries with `local_files_only=True` when the Hub is unreachable.

**Platform-side suggestion.** Worth documenting egress explicitly one way or the other, since the
correct repo-side design differs completely: with egress, nothing needs doing; without it, every
repo must bake its tokenizer/model assets, and should be told so before its first submission.

## 4. `EDULLM_CHECKPOINT_DIR` is an `s3://` URI, and the contract check cannot see misuse

`checkpoints.py` requires an `s3://` prefix, and `require_a_save_folder_a_retry_can_find` checks
that the submitted **command text** expands the variable. That catches a command that ignores it.
It cannot catch the more likely error: a program that accepts it and treats it as a filesystem
path. `pathlib.Path("s3://b/k")` is `PosixPath("s3:/b/k")` — a *relative local* path — so the run
writes checkpoints into a directory named `s3:` beside the process, **raises nothing**, and loses
them when the container exits. It would pass the contract check while satisfying none of its
intent.

Our `train_codi.py` did exactly this before we caught it. The workload catalog records the same
class of failure for `grpo_fast.py:477`, so this is now two repos. **Fixed here** by detecting a
URI, staging locally, and uploading each artifact as it is written; `train_arm` raises if a URI
reaches its save-directory argument.

**Platform-side suggestion.** The variable name reads like a directory and the docs say "prefix";
the sharp edge is real and silent. Consider either naming it `..._URI`, or passing a local path
that the platform syncs on exit, or adding the "is this actually written?" check to the post-run
evidence (a run that declared the contract and left the prefix empty is detectable after the
fact, which is where a text check cannot reach).

## 5. Smaller items

- **The `edullm` CLI is broken.** `.venv/bin/edullm` is a console script importing `edullm.cli`,
  which does not exist; only the `edullm_data` package (dataset publishing) installs. So the
  documented `edullm check` / `edullm submit` / `edullm logs` / `edullm status` path cannot be run
  at all. We used `gh workflow run submit-run.yml` and `cancel-run.yml` (report-only) instead,
  which worked well — `cancel-run.yml` with `stop` unticked was the single most useful tool in this
  whole exercise, since it is the only way to see a run's state without holding an AWS credential.
  Worth either shipping the CLI or pointing the docs at the workflows.
- **`olmo-core-train` declares `maximum_attempts: 2` with `resume_required: true`.** Our driver has
  no resume, so a second attempt would silently restart from the base checkpoint and bill a second
  full run for one result — the finding the catalog already records against
  `open-instruct-scored-rewards-train`. We submit with `--attempts 1`. The declaration is per
  workload, not per repo, so it may be worth splitting or annotating.
- **What worked well, for the record.** The compile job refusing before any credential exists is
  genuinely good: our first submission was validated for free, and every failure we hit was cheap
  and legible. The `waived_launch_check_note` surfacing "bills for 8 devices, starts 5" to the
  approver is exactly the right shape for a waiver. And the queue was 4.5 min, not the hour the
  docs warn about.
