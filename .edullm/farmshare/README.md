# Token selection 370M on FarmShare (8 × L40S)

Five approved arms via `ARM`:
`rho-1`, `rel-ema-exp`, `middle-ppl-token`, `attention`, `blade`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-token-selection-370m
ARM=attention bash .edullm/farmshare/submit_from_laptop.sh
```

BLADE requires RefHQ stream staging (`REFHQ_VERSION=v1` by default). Reference
checkpoints are downloaded only for arms that need them.

Restage before switching arms. Use `SKIP_TRAIN=1` to stage without launching
training.
