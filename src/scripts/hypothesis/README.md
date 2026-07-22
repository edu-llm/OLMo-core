# Hypothesis smoke scripts (Skill-DAG + Curriculum)

**Additive only.** New files under `src/scripts/hypothesis/`. Does **not** modify upstream OLMo-core training code or `main`.

Branch workflow (team policy):

```bash
git fetch origin
git checkout temp/370M-training-template   # or main, then:
git checkout -b hypothesis/smoke-skilldag-cl
# add only files under src/scripts/hypothesis/
git push -u origin hypothesis/smoke-skilldag-cl   # NOT main
```

Smoke target model: existing `src/scripts/train/OLMo2/OLMo2-190M.py` or `OLMo2-370M.py` (unchanged).

Data: tiny sample carved from [allenai/dolma](https://huggingface.co/datasets/allenai/dolma) (prefer `v1_6-sample` exploration / streamed subset — **not** full 3T).

## Quick start (smoke)

```bash
# 0) env
cd /path/to/OLMo-core
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[all]" datasets transformers numpy tqdm textstat  # textstat optional (Flesch)

# 1) carve a tiny multi-domain Dolma smoke pack + tokenize
python src/scripts/hypothesis/shared/prepare_smoke_dolma.py \
  --out-dir ./data/smoke_dolma_v0 \
  --docs-per-domain 200 \
  --tokenizer allenai/dolma2-tokenizer

# 2a) Skill-DAG smoke (natural mix, short run)
bash src/scripts/hypothesis/skill_dag/run_smoke.sh ./data/smoke_dolma_v0 natural

# 2b) Curriculum smoke (random vs linear+compression, short run)
bash src/scripts/hypothesis/curriculum/run_smoke.sh ./data/smoke_dolma_v0 random
bash src/scripts/hypothesis/curriculum/run_smoke.sh ./data/smoke_dolma_v0 linear_compression
```

Windows PowerShell equivalents are in each folder’s `run_smoke.ps1`.

## Layout

```text
hypothesis/
  README.md
  shared/
    prepare_smoke_dolma.py    # HF Dolma → domain jsonl + uint32 .npy shards + manifests
    domains.json              # domain ↔ Dolma source mapping
  skill_dag/
    configs/                  # mix weight JSONs for natural / fixed / skillit init
    sample_dirichlet_mixes.py # RegMix-style Dirichlet weight samples (proxy grid)
    run_smoke.sh / .ps1
  curriculum/
    score_difficulty.py       # compression / flesch / lexical diversity / random
    build_pacing_order.py     # random | vanilla | linear | warmup orders
    run_smoke.sh / .ps1
```

## Smoke success criteria

- Tokenize finishes; `manifests/docs.jsonl` + per-domain `.npy` exist
- `dry_run` (or ~50–100 step train) of 190M/370M starts without path errors
- Skill-DAG: can swap mix JSON without changing model script
- CL: can swap pacing order file without changing model script

Full experiments (RegMix 64× proxies, Skill-it *Aᵢⱼ*, full CL matrix) come **after** smoke + shared substrate on S3/TACC.
