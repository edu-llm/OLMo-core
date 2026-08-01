# Platform run: olmo2_190M train

Branch: `edullm/190m-platform-train`

Uses the stock entrypoint `.edullm/train_on_corpus.py` (no library changes).
Pick a published corpus on the submission form; do not hard-code a path.

## Command (profile: `olmo-core-train-1gpu`)

```
bash -lc 'python .edullm/train_on_corpus.py "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR" --model-factory olmo2_190M --steps 4000 --save-interval 200 --warmup-steps 200'
```

## Notes

- `bash -lc` is required so `$EDULLM_*` expands.
- `--save-folder "$EDULLM_CHECKPOINT_DIR"` must appear on the command line for the checkpoint contract.
- Default model factory is already `olmo2_190M`; it is repeated here for clarity.
- After push, wait for **Build eduLLM research image** on this commit before submitting.
