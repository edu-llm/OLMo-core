# Formal Proof Premises v3 evaluator corpus

Hardlink projection for `run_eval.py`; no row is transformed.

- sealed root: `60bb1867feb9c02ffe05e5ab388f6509fd785c77228010f676a1795375835148`
- train rows: 181,652
- eval rows: 4,191

```bash
python scripts/assemble_v3_evaluator_root.py --out corpus-v3 --check-only
HF_DIR=/path/to/exported/hf
ARM=dense
SMOKE_JSON=/path/to/smoke.json
python ../eduLLM/OLMo-core/src/scripts/train/p3_math_split/evals/run_eval.py --model "$HF_DIR" --arm "$ARM" --corpus corpus-v3 --conditions facts_present --limit 1 --out "$SMOKE_JSON"
```

Never use the legacy `corpus/` root for v3 evaluation.
