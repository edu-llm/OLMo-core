#!/usr/bin/env python
"""Merge our LoRA checkpoints into standalone HF models + emit a tutor_cat manifest.

The eval team's response generator loads models with ``vllm.LLM(model=model_id)``, which takes an
HF repo id or a directory of full weights. Our post-training checkpoints are PEFT adapters
(``adapter_config.json`` + a few MB of ``adapter_model.safetensors``), so vLLM cannot load them
directly and their pipeline has no LoRA path. Each adapter is therefore folded into the base
weights once, here, and written out as an ordinary HF model directory.

Two settings are written EXPLICITLY into the manifest rather than left to the registry's
heuristics, because both heuristics key off substrings of the model id and our ids are local
paths:

  apply_chat_template  guess_apply_chat_template() looks for "instruct"/"chat" in the id. A path
                       like /.../impl3-b-T2 matches nothing, so it would default to False and feed
                       a raw prompt to a chat-tuned model — every score would be meaningless.
  max_model_len_cap    OLMo-2-1B is a 4096-context model; the manifest default of 32768 would be
                       clamped anyway, but stating it avoids a confusing engine warning.

Usage:
    python eval/merge_for_tutorcat.py --out_dir ~/tutorcat_models --manifest ~/tutorcat_models.yaml
"""
import argparse
import glob
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
# OLMo-2-1B ships a 4096-token context; see config.json max_position_embeddings.
MAX_LEN = 4096
# Tutor turns are short. The manifest default of 4096 exists to bound base-model rambling on a
# 32k-context model; capping lower here saves generation time across ~660 scenarios x 18 models
# without truncating any realistic tutor reply.
MAX_NEW = 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="out/*", help="Glob of run dirs holding checkpoint-*/")
    p.add_argument("--out_dir", default=os.path.expanduser("~/tutorcat_models"))
    p.add_argument("--manifest", default=os.path.expanduser("~/tutorcat_models.yaml"))
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--extra", action="append", default=[], metavar="LABEL=PATH",
                   help="Standalone adapter dir outside the run layout, e.g. poc-c923=checkpoint-923")
    p.add_argument("--require_epoch", type=float, default=0.99)
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to export (default: each run's final).")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def final_ckpt(run_dir, require_epoch, want_step=None):
    """(step, path) of the checkpoint to export, or None if the run is unusable."""
    cks = {}
    for ck in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", ck)
        if m:
            cks[int(m.group(1))] = ck
    if not cks:
        return None
    if want_step is not None:
        return (want_step, cks[want_step]) if want_step in cks else None
    step = max(cks)
    state = os.path.join(cks[step], "trainer_state.json")
    try:
        epoch = json.load(open(state)).get("epoch")
    except Exception:
        return None
    if require_epoch and (epoch is None or epoch < require_epoch):
        print(f"skip {os.path.basename(run_dir)}: epoch={epoch} < {require_epoch}")
        return None
    return step, cks[step]


# Copied verbatim from the base after save_pretrained has run, deliberately overwriting what it
# wrote. A LoRA merge changes tensor values only — never the architecture, never the tokenizer —
# so every one of these files is correct by construction.
BASE_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def copy_base_files(base_model, dest):
    """Restore the base's config/tokenizer instead of keeping what save_pretrained emitted.

    save_pretrained serialises in the schema of whichever transformers is installed in the *merge*
    env, and the merge runs in p7post (transformers 5.x) while generation runs in tutorcat, where
    vLLM 0.10.x pins transformers 4.x. The two schemas are not compatible, and both failures are
    silent-to-catastrophic rather than loud:

    - `torch_dtype` was renamed to `dtype`, and `rope_theta` moved inside a `rope_parameters`
      block. vLLM reads `config.rope_theta` at the top level; when it is missing it falls back to
      a default of 10000 instead of this model's 500000. Positional encoding is then wrong at
      every position and the model emits fluent-looking word salad, with no error raised.
    - The tokenizer is written with `"tokenizer_class": "TokenizersBackend"` and the chat template
      split out into chat_template.jinja, which transformers 4.x rejects outright.

    Copying bytes from the base makes the merged model independent of whatever env merged it.
    """
    import shutil

    src = pathlib.Path(base_model)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = pathlib.Path(snapshot_download(base_model, allow_patterns=list(BASE_FILES)))

    copied = [f for f in BASE_FILES if (src / f).is_file()]
    for f in copied:
        shutil.copyfile(src / f, pathlib.Path(dest) / f)
    if "config.json" not in copied:
        raise RuntimeError(f"no config.json found for {base_model} under {src}")
    if not ({"tokenizer.json", "vocab.json"} & set(copied)):
        raise RuntimeError(f"no usable tokenizer files found for {base_model} under {src}")
    return copied


def merge_one(base_model, adapter_dir, dest):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir, device_map="cpu")
    model = model.merge_and_unload()
    model.save_pretrained(dest, safe_serialization=True)
    copy_base_files(base_model, dest)
    del model


def main():
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    targets = []
    for run_dir in sorted(glob.glob(args.runs)):
        if not os.path.isdir(run_dir):
            continue
        got = final_ckpt(run_dir, args.require_epoch, args.step)
        if got:
            targets.append((os.path.basename(run_dir.rstrip("/")), got[0], got[1]))
    for spec in args.extra:
        label, path = spec.split("=", 1)
        if os.path.isdir(path):
            targets.append((label, 0, path))
        else:
            print(f"skip --extra {label}: {path} not a directory")

    print(f"{len(targets)} adapters to merge -> {out_dir}")
    entries = []
    for i, (label, step, adapter) in enumerate(targets, 1):
        dest = os.path.join(out_dir, label)
        print(f"[{i}/{len(targets)}] {label} (step {step}) <- {adapter}")
        if args.dry_run:
            entries.append((label, dest))
            continue
        if os.path.exists(os.path.join(dest, "config.json")):
            print("   already merged, skipping")
        else:
            merge_one(args.base_model, adapter, dest)
            print(f"   wrote {dest}")
        entries.append((label, dest))

    lines = [
        "# tutor_cat response-generation manifest for the P7 sweep (generated by",
        "# post-training/eval/merge_for_tutorcat.py -- do not hand-edit).",
        "#",
        "# apply_chat_template is pinned true on every row: the registry infers it from substrings",
        "# of the model id, and these ids are local paths with no 'instruct' marker, so the default",
        "# would be False and every model would be prompted as a base LM.",
        "defaults:",
        f"  max_model_len_cap: {MAX_LEN}",
        f"  max_new_tokens: {MAX_NEW}",
        "  tensor_parallel_size: 1",
        "  scoring_method: generate",
        "  temperature: 0.0",
        "  top_p: 1.0",
        "  repetition_penalty: 1.1",
        "  seed: 0",
        "  apply_chat_template: true",
        "  backend: vllm",
        "  architecture: causal",
        "  gated: false",
        "",
        "models:",
        f"  # the untrained reference point every retention number is measured against",
        f"  - id: {args.base_model}",
        f"    params_b: 1",
    ]
    for label, dest in entries:
        lines.append(f"  - id: {dest}")
        lines.append(f"    params_b: 1")
    manifest = os.path.expanduser(args.manifest)
    with open(manifest, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {manifest} with {len(entries) + 1} models")


if __name__ == "__main__":
    main()
