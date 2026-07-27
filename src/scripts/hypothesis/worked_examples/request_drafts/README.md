# Request drafts for 4 CPT arms (MIT / ORCD)

Templates for `/submit-edullm-job`. **Jobs #23–#26 already exist** — prefer
updating those Issues (Commit SHA + real manifest digest) over re-filing.

Profile `worked-examples-cpt` (on branch `hypothesis/we-metamath-wandb-smoke`) locks:

- 2×H100, 360 min, `torchrun --nproc-per-node=2`
- `--model-factory=olmo2_760M`
- `--token-budget=200000000`
- `--pack-dir=/orcd/pool/edullm/data/worked-examples-metamath-v0`
- `--load-path=/orcd/pool/edullm/checkpoints/OLMo-Ladder-760M-0.5xC-core`
- W&B `eduLLM` / `pretraining`

**Operator blockers before train:**

1. Convert Ladder HF → pool `…/OLMo-Ladder-760M-0.5xC-core`
2. Stage HF pack with **shards + label_mask** per arm under pack-dir
3. Write real manifest SHA-256 (Issues currently have placeholder `bbbb…`)
4. Stamp Commit SHA to pushed tip of this branch

Shared study fields:

- Study: `worked-examples-faded-scaffolds`
- Comparison: `bare-vs-complete-vs-fade-ordered-vs-fade-shuffled`
- Success metrics: `eval/pass_at_n,eval/pass_ratio_at_n`
- W&B project: `pretraining`

See `../SUBMIT.md` for the full MIT copy/paste handoff.
