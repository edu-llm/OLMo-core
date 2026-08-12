# memsplit-v4

Does a small language model reason better when arbitrary fact *values* are kept
out of its weights and supplied by an exact external store?

This is a **methodology reset** of the MemorySplit line, not a continuation. The
previous generation's science was well-aimed; its instrumentation was not. Two
defects, each independently sufficient, invalidated every dense-vs-split
generative comparison it produced, and both are fixed here as the first commits.

Full audit: `../memory-split/docs/2026-08-12-n-hop-design-and-evidence-audit.md`.

## Why a new tree

| | previous | here |
|---|---|---|
| Decoder | left-pads to batch max with `EOT`, attends over pads | length-grouped, **no padding exists** |
| Loss | `mean` over *surviving* targets -> arm-asymmetric | `sum / fixed divisor`, identical across arms |
| Arms | rendered as **separate token streams** | one stream, four **loss-weight sidecars** |
| Depth | hardcoded 2 at 13 sites | first-class parameter, 1..N |
| Compose templates | exactly **1** phrasing | >= 10 per slot |
| "Novel" entities | new names, **same values** | disjoint train/novel value pools |
| Equal-mass control | tab runs, never run | two controls (contiguous + scattered), bracketed |
| Endpoint check | after 32 cells trained | before any training, refuses endpoints |
| FLOPs | `6ND` | `6ND + 12*L*d*ctx`, both `N` conventions |
| KL | never computed | JSD primary, both KL directions, per-role tails |

Measured consequence of the first two, from the previous project's own records:
the decoder scored the *same checkpoint* at **4.7% batched against 93.8% one
prompt at a time**, and the loss bug meant "every split/masked arm in this
repository was trained under a different effective objective than its dense twin
... no dense-minus-split difference may be reported as a treatment effect."

## Layout

```
memsplit/
  tokenizer.py    frozen special ids; byte-level fallback so tests run offline
  records.py      role-tagged segments; the one lookup-wrapping function
  bios.py         entities; pools large enough to split train/novel
  nhop.py         depth-parameterised composition + the p**n null
  masking.py      four conditions over one stream; two equal-mass controls
  store.py        exact-match dictionary (deliberately the dumbest retriever)
  model.py        transformer; fixed-divisor loss; honest FLOP accounting
  generate.py     length-grouped decoding + store interception
  scorers.py      one stated mode per scorer; chance and floors always reported
  calibration.py  pre-training endpoint gate
  metrics.py      JSD/KL, rank shift, noise floor, compute-to-threshold
tests/            67 tests, CPU-only, no GPU and no tiktoken required
```

## Running

```bash
cd /workspace/edullm/memsplit-v4
MEMSPLIT_TOKENIZER=byte python -m pytest tests/ -q     # offline, ~25s
python scripts/calibrate_nhop.py --depths 1 2 3 4 5    # gate, no training
```

`tiktoken` is unavailable in this environment, so the suite runs on a byte-level
fallback tokenizer. Being byte-level it is a *stricter* test of mask-boundary
logic than BPE, not a weaker one. Anything that writes a trainable corpus calls
`require_production_tokenizer` and will refuse the fallback.

## Status

Landed, with tests:

- fixed decoder + the batching-invariance assertion the old suite lacked
- fixed-divisor loss + a test reproducing the old bug's 1.333x inflation
- depth-parameterised n-hop generator with shortcut/cycle/prefix/leak gates
- four-condition mask ledger with collision-free equal-mass controls
- endpoint calibration gate that refuses no-range, leaky and flat endpoints
- metrics: bounded JSD, rank shift, seed floor, bracketed compute-to-threshold

Not yet written: the corpus builder that materialises `.bin` + sidecars, the
training loop wrapper, the eval driver, and the mid-training/replay arm. Design
for all four is in the audit document, sections 4 and 5.

## The one thing to keep in mind

If per-hop reliability is *p*, an n-hop chain succeeds at ~*p^n*. So an arm at
p=0.999 beats one at p=0.93 by a gap that **grows with depth for purely
arithmetic reasons** -- 13.3pp at depth 1 rising to 34.7pp at depth 5, which
`nhop.pn_table` will print for you. The previous two-hop result is quantitatively
that and nothing more (7.81pp predicted, 7.4pp observed). Overlay the *p^n* curve
on every depth plot; the reasoning quantity is **conditional per-hop accuracy**,
not end-to-end accuracy.
