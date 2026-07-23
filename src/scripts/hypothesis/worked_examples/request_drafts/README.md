# Request drafts for 4 CPT arms

These JSON files are **templates** for `/submit-edullm-job` after operators
allowlist `worked-examples-cpt` (see `../OPERATOR_ALLOWLIST.md`).

**Do not submit until:**

1. Policy on `main` includes the new entrypoint  
2. ORCD manifest path + SHA-256 are filled in  
3. Feature-branch commit with `train_cpt_arm.py` + Pass@N metrics is pushed  
4. `Commit SHA` in each JSON is replaced with that exact 40-char SHA  

Then run (from repo root, clean tree)::

```bash
python src/scripts/hypothesis/worked_examples/request_drafts/fill_sha.py
# then /submit-edullm-job once per arm JSON (or use prepare_submit.py)
```

Shared study fields:

- Study: `worked-examples-faded-scaffolds`
- Comparison: `bare-vs-complete-vs-fade_ordered-vs-fade_shuffled`
- Success metrics: `eval/pass_at_n,eval/pass_ratio_at_n`
- W&B project: `pretraining`
