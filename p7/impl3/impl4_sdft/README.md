# Implementation 4 — Self-distillation SFT (SDFT)

Keep Impl 2's pedagogy gains while cutting math/logic forgetting, by reducing the
fine-tune's new-task KL from the base — here via a **data change** (self-distilled
targets). Where Impl 3 shaves the incidental/stylistic KL *inside existing targets*,
SDFT manufactures **new, near-base correct targets** the model rewrote into its own
distribution (attacking the behavioral KL reweighting can't remove). SDFT composes
with Impl 3.

## Procedure (PRD §4.2 / SDFT §5.2)
1. **Self-distill targets** (`self_distill.py`):
   - `--mode rewrite`: base rewrites each gold tutor turn in its own words (gold as
     reference), pedagogy SI in context.
   - `--mode domains`: base generates SI-free outputs across general domains (pulls
     the fine-tune toward the base distribution broadly).
2. **Optional pedagogy quality-gate** (`--quality_gate`): keep a rewrite only if it
   still hides the answer / stays one step; else fall back to gold. PRD §4.2 says
   *test whether gating even helps* — no gating matches the base distribution more
   closely.
3. **Train** on the gold/distilled mix with the Impl-2 recipe otherwise unchanged
   (`train_sdft.py`), keeping ≥10 checkpoints.

## Files
| File | What it does |
|---|---|
| `self_distill.py` | Generate self-distilled targets (rewrite / domains). |
| `train_sdft.py`   | Mix gold vs distilled by fraction, then run the SFT recipe. |
| `config.yaml`     | Impl-2 recipe + the distilled-fraction sweep grid. |

## Sweep (comparison knob)
`distilled_frac ∈ {0, 0.25, 0.5, 0.75, 1.0}` (0 = vanilla Impl 2, 1 = full SDFT):
```bash
# 1. make targets (data is blank; supply your own gold train + domain prompts)
python self_distill.py --mode rewrite  --in_file ../data/socrateach_sft_train.jsonl \
    --out_file distilled/pedagogy_rewrite.jsonl --quality_gate
python self_distill.py --mode domains --in_file domain_prompts.jsonl \
    --out_file distilled/general_domains.jsonl
# 2. sweep the fraction
for f in 0 0.25 0.5 0.75 1.0; do
  python train_sdft.py --distilled_frac $f --gold_dir ../data \
      --distilled_pedagogy distilled/pedagogy_rewrite.jsonl \
      --distilled_general distilled/general_domains.jsonl --config config.yaml
done
```

## Definition of done (§4.4)
Vs vanilla Impl 2 at matched pedagogy (CIs): reduced math/logic forgetting; lower
new-task KL (down-left on the plane); SI-gating preserved (no-SI behavior + no-SI KL
stay close to base). Reuse `common/kl.py` + the eval suite. Report `kl_new_SI` and
`kl_ped_noSI` per checkpoint.

> Note: PRD Implementation 5 is a second SDFT variant and is **not to be implemented
> yet**; Implementation 6 (RLHF) is **do-not-start**. This folder covers Impl 4 only.
