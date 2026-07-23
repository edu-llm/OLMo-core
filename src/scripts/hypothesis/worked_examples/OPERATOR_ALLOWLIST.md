# Operator ask: allowlist Worked-Examples CPT

**From:** submitting team (feature branch `hypothesis/we-metamath-wandb-smoke`)  
**To:** eduLLM ORCD operators (e.g. assignee on connectivity smoke #20)  
**Request:** add a reviewed policy entrypoint so `/submit-edullm-job` can file the 4-arm MetaMath CPT study.  
**Constraint:** please land on `main` via normal review — do **not** ask interns to edit `policy.py` on `main` unsupervised.

## Why

Connectivity smoke (`generic-smoke`, Issue #20) only proves queue + W&B.  
Scientific CPT needs a dedicated profile. Today policy only allows `generic-smoke` and `hypothesis-smoke` (skill-dag/curriculum).

## Proposed profile (draft for operator review)

Suggested name: `worked-examples-cpt`

| Field | Proposed value |
|-------|----------------|
| Script | `src/scripts/hypothesis/worked_examples/train_cpt_arm.py` |
| Launcher | `torchrun` |
| Model | OLMo2-760M (`TransformerConfig.olmo2_760M`) |
| Init ckpt | converted `allenai/OLMo-Ladder-760M-0.5xC` |
| Data kind | new kind e.g. `worked-examples` |
| W&B | entity `eduLLM`, project `pretraining`, callback required |
| GPUs | policy today caps `max_gpu_count` at **2**; request 2× H100 (or ask operators to raise cap for this study) |
| Metrics emitted | `train/PPL`, `eval/pass_at_n`, `eval/pass_ratio_at_n` |

### Data / manifest

HF pack (already built):  
https://huggingface.co/datasets/hiyasvyas/worked-examples-metamath-v0  

Need a policy-controlled ORCD manifest, e.g.:

```text
/orcd/pool/edullm/manifests/worked-examples-metamath-v0.json
```

pointing at a pool copy of:

```text
tokenized/{bare,complete,fade_ordered,fade_shuffled}/shard-00000.npy
tokenized/{arm}/label_mask-00000.npy
eval/holdout_bare.jsonl
```

HF URI alone is **not** accepted as `Data manifest` by the queue validator.

### Arms → 4 job Issues (after allowlist)

Same study, different condition / shard:

1. `bare`
2. `complete`
3. `fade_ordered` (loss mask before `loss_start_char`)
4. `fade_shuffled`

Draft request payloads live in `request_drafts/` on this branch.

## Converted checkpoint note

Operators (or a one-time setup job) should convert HF → OLMo-core:

```bash
python src/examples/huggingface/convert_checkpoint_from_hf.py \
  -i allenai/OLMo-Ladder-760M-0.5xC \
  -m olmo2_760m \
  -t dolma2 \
  -o $EDULLM_SCRATCH/checkpoints/OLMo-Ladder-760M-0.5xC-core
```

## Please reply with

1. Approved profile name + script path  
2. Manifest path + SHA-256  
3. Allowed GPU preference / max count / max runtime  
4. When `main` contains the allowlist (SHA)

Then the submitting team will re-run `/submit-edullm-job` four times on a clean pushed SHA.
