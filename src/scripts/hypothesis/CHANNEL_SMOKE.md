# Paste for training Slack channel

```text
Skill-DAG + Curriculum smoke scripts are ready (additive only — no edits to existing OLMo-core / not on main).

Repo: edu-llm/OLMo-core
Branch to use: hypothesis/smoke-skilldag-cl  (from temp/370M-training-template)
Path: src/scripts/hypothesis/

What we added:
1) shared/prepare_smoke_dolma.py — stream a tiny multi-domain pack from HF allenai/dolma → raw jsonl + uint32 .npy + manifests
2) skill_dag/ — mix configs (natural / fixed_uniform / skillit_init), Dirichlet sampler for RegMix-style proxy grid, run_smoke.sh|.ps1
3) curriculum/ — difficulty scoring (compression / flesch / lexical / random), pacing orders (random / vanilla / linear / warmup), run_smoke.sh|.ps1

Smoke steps:
  python src/scripts/hypothesis/shared/prepare_smoke_dolma.py --out-dir ./data/smoke_dolma_v0 --docs-per-domain 200
  # Skill-DAG
  pwsh src/scripts/hypothesis/skill_dag/run_smoke.ps1 -DataDir ./data/smoke_dolma_v0 -MixName natural
  # CL
  pwsh src/scripts/hypothesis/curriculum/run_smoke.ps1 -DataDir ./data/smoke_dolma_v0 -Pacing linear -Metric compression_ratio

Uses existing OLMo2-190M.py / 370M.py unchanged. Full custom mix/order wiring into dataset.mix may need a finalize pass with whoever owns data loading — smoke pack + configs are ready either way.

Dolma: https://huggingface.co/datasets/allenai/dolma (smoke uses a streamed subset, not full 3T)
```
