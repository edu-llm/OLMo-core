# MixLaw 370M on FarmShare (8 × L40S)

FarmShare bootstrap for the four approved MixLaw arms (`ARM_INDEX` 0–3):
`olmo-mix-1124`, `mix01`, `ML-pilot_caps`, `LGB-min1pct`.

```bash
cd /mnt/c/alpha_ai/OLMo-core
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh
```

Staging downloads only the deterministic shard prefixes needed by the selected
arm's domain weights and token budget, with 10% headroom. The manifest is bound
to `ARM_INDEX`; use a separate run directory or restage when changing arms.

Uses PyTorch SDPA (`OLMO_FLASH_ATTENTION=0`). Push AWS and W&B sessions via the
`edullm` FarmShare helpers before submit.
