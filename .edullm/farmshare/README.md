# Token selection 370M on FarmShare (8 × L40S)

Approved arms via `ARM`:
`rho-1`, `rel-ema-exp`, `middle-ppl-token`, `attention`, `blade`, `random-control`.

```bash
cd /mnt/c/alpha_ai/OLMo-core-token-selection-370m
ARM=attention bash .edullm/farmshare/submit_from_laptop.sh
```

BLADE requires RefHQ stream staging (`REFHQ_VERSION=v1` by default). Reference
checkpoints are downloaded only for arms that need them.

Restage before switching arms. Use `SKIP_TRAIN=1` to stage without launching
training.

## Control arm on 4 × L40S

`random-control` masks a random 60% of tokens per row from the loss instead of
selecting them by any score, at the same keep rate as the other arms. It needs
no reference checkpoints. Run it on a 4-GPU allocation instead of the default
8-GPU node by overriding `TRAIN_GPUS`:

```bash
cd /mnt/c/alpha_ai/OLMo-core-token-selection-370m
ARM=random-control TRAIN_GPUS=4 bash .edullm/farmshare/submit_from_laptop.sh
```

`launch.sh` automatically passes `--local` to the training entrypoint whenever
`TRAIN_GPUS` isn't 8, since the production entrypoint otherwise refuses any
topology other than one 8-GPU node. Everything else — model, optimizer,
dataset, checkpoint ladder, and task-loss evaluation — is unchanged.
