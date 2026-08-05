# P3 dense vs. split training

P3 compares two Qwen2.5-0.5B continual-pretraining arms on identical packed proof
documents. Dense supervises supplied premise tokens; split can attend to those tokens
but masks their loss. Goal and derivation tokens are supervised in both arms.

The final scientific controls are the paired `configs/dense.yaml` and
`configs/split.yaml`. Their parsed `shared` blocks are identical and `arm` is their only
top-level difference. Arbitrary flags and dotlist config overrides are refused.

## Immutable inputs

- Base model: `Qwen/Qwen2.5-0.5B`
- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- `model.safetensors`: 988,097,824 bytes
- Weight SHA-256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`
- Tokenizer artifact: `tokenizer/qwen25-vendored/v1`
- `tokenizer.json` SHA-256: `3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8`
- `tokenizer_config.json` SHA-256: `ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09`
- Tokenizer behavior SHA-256: `aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a`
- Tokenizers implementation: `0.22.2`
- EOS and pad token ID: 151643

Training downloads both tokenizer files from the exact published artifact, verifies
the complete four-part seal, and derives the split separator from those same bytes.
The checkpoint config and runtime summary retain model, tokenizer, dataset, launch,
world-size, and source-commit provenance.

The dataset ID is fixed to `pretrain/formal-proof-premises-500m`. The version remains
an explicit platform submission value because the repaired final release is not yet
selected. Never use the known-blocked v2 release for final training or conclusions.

## Fresh v3 tokenization

Packed output uses `p3-packed-group-v3` / `p3-packed-corpus-v3` with writer
`tokenize-corpus-v4`. The v4 writer is the accepted implementation of the v3 artifact
shape; rejected pre-v4 v3 markers are legacy and cannot be resumed. Every group binds:

- the exact accepted atomic transaction v2 contract: `corpus-generation-current/v2`,
  `corpus-generation-manifest/v2`, `corpus-generation-plan/v2`,
  `physical-occurrence-routes/v2`, `logical-generation-root/v1`, and
  `generation-transaction-state/v1`;
- its immutable generation ID and logical root, exact files/directories and read-only
  modes, reconstructible validators and plan root, occurrence accounting/routes,
  committed journal state, path/link rules, and every output JSONL digest;
- the approved local `tokenizer.json` and `tokenizer_config.json` hashes, tokenizer
  behavior composite, `tokenizers==0.22.2`, EOS/pad IDs, and separator IDs;
- the encoding-cache build fingerprint and cache root, packing configuration,
  dtype/byte order, sequence length, code/schema version, and every shard digest.

The corpus consumer's **internal transaction input** is selected from transaction
`OutputSpec` entries by the exact `TRAIN`/`EVAL` role plus sibling; it does not infer
private producer paths. Therefore the current producer's `shards/<family>.jsonl` train
layout is accepted. That internal source path is not the published dataset layout. The six
source schemas are pinned exactly:
`metamath-proof-v2`, `mizar-proof-v2` for both Mizar siblings, `atp-v2` for both ATP
siblings, and `isabelle-transition-v2`. Missing/duplicate role+sibling pairs, duplicate
paths, or schema drift reject.

Completed and partial encoding caches use `p3-encoding-cache-v3`. Both token and
offset payloads carry byte counts, SHA-256, dtype/endianness, token/document ranges,
source identity, tokenizer seal, and build fingerprint. Every committed batch also
records contiguous source/token/offset ranges and three digests. Resume rehashes the
whole payload, every batch range, and the exact source-row sequence before appending.
Fingerprint mismatch, same-size mutation, stale temp files, orphan shards, or unknown
control files are preserved and refused.

Use a new output prefix for a repaired corpus. Do not point v3 at the preserved
`artifacts/public` tree: its legacy size-only markers are intentionally refused, not
upgraded in place. Keep the old tree unchanged for audit comparison.

```bash
CORPUS_TRANSACTION=/absolute/path/to/repaired/corpus-transaction
TOKENIZER=/absolute/path/to/tokenizers/qwen25-vendored
OUT=/absolute/path/to/new/p3-tokenized-v3
CACHE=/absolute/path/to/new/p3-token-cache-v3
python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus-contract-root "$CORPUS_TRANSACTION" --out "$OUT" --cache-dir "$CACHE" --split train --sequence-length 16384 --shard-tokens 250000000 --tokenizer "$TOKENIZER" --pack --jobs 2 --batch-size 256
python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus-contract-root "$CORPUS_TRANSACTION" --out "$OUT" --cache-dir "$CACHE" --split val --sequence-length 16384 --shard-tokens 250000000 --tokenizer "$TOKENIZER" --pack --jobs 2 --batch-size 256
```

Token corpus preparation has no evaluator input, profile, dependency record, path, or
pin. The old evaluator-related flags are unknown CLI arguments and reject rather than
being ignored.

There is no production `--corpus` fallback and no mutable default tokenizer ID.
`--test-only-corpus-dir` is an explicit fixture seam and is rejected unless paired with
`--test-only-allow-unsealed-inputs`; it must never appear in an artifact build command.

After both splits finish, require `train_meta.json` and `val_meta.json` to declare the
v3/v4 schema, generation roots, and nested shard/cache seals. The writer enforces
`p3-token-cross-split-binding-v1`: train and val must have identical corpus generation
ID/logical root, tokenizer seal, packing/build contract, and family/schema inventory.
Source paths, source hashes, shard hashes, and counts remain split-specific. This check
runs before final control-file replacement and again during exact group inventory
validation, so independently valid train/val outputs cannot be spliced across corpus
generations.
Then run:

```bash
.venv/bin/python -m pytest -q src/test/scripts/p3_math_split/tokenize_corpus_test.py src/test/scripts/p3_math_split/tokenize_corpus_contract_test.py
TOKENIZED_DIR="$OUT" TOKEN_CACHE_DIR="$CACHE" .venv/bin/python -m pytest -q src/test/scripts/p3_math_split/packed_artifact_test.py
```

The package-shape test uses the `edullm-data` source checkout pinned by the image build,
not an ambient installed package. The same local Git object database must contain the
deployed-policy commit so the test can materialize its exact `pretrain.json` into a
temporary fixture without network access:

```bash
export P3_EDULLM_DATA_SOURCE=/tmp/edullm-data-pinned
test "$(git -C "$P3_EDULLM_DATA_SOURCE" rev-parse HEAD)" = 38bf831a6c3f445e394784018441fd59288b876c
test "$(sha256sum "$P3_EDULLM_DATA_SOURCE/families/pretrain.json" | cut -d" " -f1)" = 2d507a1b8b9a5ce6c361b3e2731c12678cb9f3fc3e24c87aa6dc4b75100f0fd5
test "$(git -C "$P3_EDULLM_DATA_SOURCE" rev-parse 'e0984c88b7c5^{commit}')" = e0984c88b7c5d3d927bda227af4f47e2014dd257
test "$(git -C "$P3_EDULLM_DATA_SOURCE" show e0984c88b7c5:families/pretrain.json | sha256sum | cut -d" " -f1)" = 4128a90ba8ed8bb167180a2a19a4cbfc4788d5f14413dbff5e184745253bfbf3
PYTHONPATH="$P3_EDULLM_DATA_SOURCE/src" .venv/bin/python -c 'import edullm_data; assert edullm_data.__version__ == "0.5.0"'
P3_EDULLM_DATA_SOURCE="$P3_EDULLM_DATA_SOURCE" .venv/bin/python -m pytest -q src/test/scripts/p3_math_split/tokenize_corpus_test.py -k 'synthetic_profile_gate or staged_publication_gate'
```

These are deliberately **synthetic tests**, not a claim about repaired shards. They
exercise the complete network-free FakeS3 publisher/validator harness and prove that it
rejects an omitted family or partition, extra done controls, bad-endian bytes, explicit
out-of-vocabulary IDs, sampled zero runs, excessive EOS, and inadequate diversity. A
separate fixture proves that 200 sampled IDs reports the local-256/deployed-128 policy
delta. Synthetic PASS never authorizes publication.

Copy only the audited payload group into a fresh staging directory; tokenization control
manifests and encoding caches are not publishable payloads. Validate that explicitly
supplied real stage against the external train/val control manifests:

```bash
export PUBLISH_ROOT=/absolute/path/to/new/p3-pretrain-publish
mkdir "$PUBLISH_ROOT"
OUT="$OUT" PUBLISH_ROOT="$PUBLISH_ROOT" .venv/bin/python - <<'PY'
import json
import os
import shutil
from pathlib import Path

out = Path(os.environ["OUT"])
publish_root = Path(os.environ["PUBLISH_ROOT"])
for split in ("train", "val"):
    manifest = json.loads((out / f"{split}_meta.json").read_text())
    for group in manifest["groups"].values():
        for shard in group["shards"]:
            relative = Path(shard["path"])
            if not relative.name.endswith(".u32le.bin"):
                raise RuntimeError(f"refusing non-token payload: {relative}")
            destination = publish_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out / relative, destination)
PY
P3_EDULLM_DATA_SOURCE="$P3_EDULLM_DATA_SOURCE" P3_REAL_STAGED_PAYLOAD="$PUBLISH_ROOT" P3_REAL_TOKENIZED_DIR="$OUT" P3_FIXED_TOKENIZER_DIR="$TOKENIZER" .venv/bin/python -m pytest -q -s src/test/scripts/p3_math_split/tokenize_corpus_test.py::test_real_staged_payload_gate_uses_explicit_paths_and_reports_policy_delta
```

The final physical line above is the exact one-command real gate. It never generates,
changes, or substitutes token bytes and never contacts S3. A file-backed subclass of the
pinned package's `FakeS3` streams publisher hashes from the actual shard-only tree and
services the validator's seeded range reads without retaining the multi-gigabyte payload
in RAM. It seeds `tokenizer/qwen25-vendored/v1` from the exact two files under
`P3_FIXED_TOKENIZER_DIR`, after checking their fixed SHA-256 values.
The pinned tokenizer profile derives `vocab_size=151665` and EOS `151643` from those
bytes. The model's `151936` embedding width is padded capacity, not the validator's
maximum legal emitted token ID.

The test prints one machine-readable line beginning
`P3_REAL_PRETRAIN_PROFILE_GATE_REPORT=`. Interpret its terminal state exactly:

- `PASS`: structure/manifests/hashes are exact; pinned `publish()` generated the expected
  six labels, train+val partitions, uint32 little-endian entries, counts, and sole
  tokenizer dependency; every shard received all five profile range reads under both
  policies; and both policy validations returned zero violations.
- `REPORT`: at least one policy rejected, the policy outcomes differed, or their violation
  sets differed. The complete per-policy codes/paths and delta are printed and pytest exits
  nonzero. This is deliberately not auto-authorized, including the known shape
  local-256 `distinct-too-few` versus deployed-128 PASS.
- `SKIP`: one or more real paths were absent. A skip is honest but proves no real readiness
  and cannot authorize publication. Supplying only some required paths is an error, not a
  skip.

Even `PASS` requires manual review before S3 upload. Record from the report: both control
manifest file/declared hashes, every staged path/byte count/token count/SHA-256, the
FakeS3-generated dataset and token-manifest hashes, tokenizer file/manifest hashes, each
policy result, sampled path/range counts, and the policy delta.

The image-pinned publisher package and deployed validator are intentionally recorded as
two different facts:

- publisher/client: `edullm-data` package `0.5.0`, commit
  `38bf831a6c3f445e394784018441fd59288b876c`; pretrain policy SHA-256
  `2d507a1b8b9a5ce6c361b3e2731c12678cb9f3fc3e24c87aa6dc4b75100f0fd5`,
  `distinct_ids_min=256`;
- deployed validator: job-definition revision 12, full image/source commit
  `e0984c88b7c5d3d927bda227af4f47e2014dd257`; pretrain policy SHA-256
  `4128a90ba8ed8bb167180a2a19a4cbfc4788d5f14413dbff5e184745253bfbf3`,
  `distinct_ids_min=128`.

Neither source is edited. Both validations run the pinned package's registered
`pretrain-tokens/v1` checks (`token-count`, decode smoke, NumPy-magic, and sequence
alignment) over the same actual FakeS3 publication. The deployed fixture changes only
which verified family-policy bytes the validator reads. This offline gate does not replace
the later live landing validation; it prevents reaching upload without first exposing the
same policy delta locally.

After the real-stage gate passes, use the pinned client to upload through
`edullm-landing`:

```bash
PYTHONPATH="$P3_EDULLM_DATA_SOURCE/src" .venv/bin/python - <<'PY'
import datetime
import os

from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

publish(
    os.environ["PUBLISH_ROOT"],
    dataset_id="pretrain/formal-proof-premises-500m",
    purpose="Packed formal-proof premise tokens for the original full 13-epoch P3 pretraining run",
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/qwen25-vendored/v1",
    group_meta={"tokens": {"seq_len": 16384}},
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
)
PY
```

The tokenizer dependency must already exist in `edullm-data`; the producer never
retrains, republishes, mutates, or falls back from it. The client writes to
`edullm-landing`; validation and promotion are platform-owned, as are release selection,
reading, images, runs, and checkpoints.

Only a newly audited v3 payload tree should be copied into a release staging directory.
Tokenization itself does not modify the preserved artifacts or publish anything.
Compatibility is locked to the canonical semantic transaction descriptor and its hash,
not to producer Python bytes. The observed
`scripts/corpus_generation_transaction.py` SHA-256 is retained only as audit metadata;
equivalent producer source remains compatible, while any schema, inventory, mode,
route/accounting, plan/root, validator, journal, or path/link semantic change rejects.
Cross-repository tests create real publications with `GenerationCoordinator`. Published
`pretrain-tokens/v1` output remains
`tokens/<family>/<split>-NNNNN.u32le.bin`: `tokens` is the payload group, `<family>` is
derived as the source label, and the train/val partition is derived from each shard
basename. The exact gate is pinned to `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c` / package `0.5.0`. Synthetic `FakeS3`
coverage verifies publisher structure only; the explicit real-stage gate verifies exact
local bytes and manifests, and the deployed validator remains the final diversity gate.

Production workers leave completed shard/cache bytes unlabelled until finalization. The
finalizer opens every component of `.generation.lock` no-follow, validates regular-file
metadata, and holds `fcntl.LOCK_SH` while re-resolving `CURRENT`, rehashing sources,
atomically replacing every new group done marker and train/val meta manifest, and
fsyncing their directories. The publisher uses `LOCK_EX`, so a switch before the shared
lock rejects with no final token manifests, while a switch during commit blocks until
the coherent token commit is durable. Failed commits remove pending manifests but retain
shard/cache bytes for diagnosis. If replacement happened before a write/fsync failure,
the final manifest is retained and the command reports commit-uncertain state; it never
deletes a potentially committed control file. A clean retry restages and atomically
replaces the manifests. Never add a legacy-path fallback.

## Canonical entrypoint

Only `train_platform.py` may train P3. The old executable is a fail-fast deprecation
stub because it used legacy local arrays and the unsafe pre-initialization weight
lifecycle.

Commands below are intentionally one physical line. The platform command field does
not accept shell continuation formatting.

## No-cost configuration preflight

This unit-level preflight builds the complete platform config with mocked published
artifacts. It uses no AWS credentials or GPU:

```bash
.venv/bin/python -m pytest -q src/test/scripts/p3_math_split/train_platform_integration_test.py -k no_cost_a10g_config_preflight
```

## A10G configuration-build dry run

Use workload profile `olmo-core-check-gpu`, compute profile `gpu-1xa10g`, team
`memory-split`, and the explicit repaired dataset-release dropdown value. This stage
proves GPU visibility and configuration/artifact resolution; it does not train:

```bash
bash -lc 'python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" && python src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --save-folder "$EDULLM_CHECKPOINT_DIR" --dry-run'
```

One A10G and `WORLD_SIZE=1` are valid at this configuration stage. The script records
that observed declaration but does not pretend it is final training hardware.

## 8xH100 runtime smoke

Use workload profile `olmo-core-train-4gpu` for its checkpoint/retry contract and
compute profile `gpu-8xh100` for the actual machine. The workload profile does not
select hardware. The closed `--runtime-smoke` mode reads its 100-step, 10-step-warmup,
and 50-step-checkpoint values from the paired YAML rather than accepting dotlist
overrides:

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --save-folder "$EDULLM_CHECKPOINT_DIR" --runtime-smoke'
```

This remains a terminal live gate. Local tests do not establish FlashAttention2,
eight-rank FSDP, distributed checkpointing, W&B, or S3 round-trip behavior. Confirm all
eight ranks, strict pretrained loading, finite loss, one W&B run, and complete step 50
and step 100 checkpoints before final submission. Every non-dry invocation refuses before
training setup unless torchrun declares a compatible single-node `WORLD_SIZE=8` process set.

## Final runs

Submit two independent forms with the same built commit, workload profile
`olmo-core-train-4gpu`, compute profile `gpu-8xh100`, team, experiment, W&B project,
and repaired dataset release. Only arm/config differ.

Dense:

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

Split:

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm split --config src/scripts/train/p3_math_split/configs/split.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

The platform's hashed run manifest already pins the resolved image digest, workload
and compute profiles, and exact command. Do not duplicate that record with unreliable
in-container image detection. Leave the advanced image field blank so the platform
resolves the built digest from the submitted commit.

## Export/evaluator handoff

Checkpoint export writes `model_provenance.json`. Stable evaluator-facing names are
`checkpoint_step`, `arm`, `base_model_id`, `base_model_revision`,
`initial_weights_sha256`, `trained_weight_files`,
`trained_weights_root_sha256`, `source_commit`, `tokenizer_artifact_id`,
`tokenizer_artifact_version`, `tokenizer_file_sha256`,
`tokenizer_composite_sha256`, `dataset_id`, and `dataset_version`. Semantic hashes
never include local cache or output paths. Reportable exports require a positive checkpoint
step, a non-dry saved config, a nonempty `EDULLM_COMMIT_SHA`, and one coherent BF16
floating-parameter dtype. The trained-weight root canonically binds the exact single or
sharded safetensors payload and its index; evaluation rehashes that complete inventory
before loading the tokenizer or model. `initial_weights_sha256` remains the separate pin
for the starting Qwen weights.

When the platform exposes `EDULLM_RUN_MANIFEST_ID` and
`EDULLM_RUN_MANIFEST_SHA256`, the saved config and export bind them as
`platform_run_manifest_id` and `platform_run_manifest_sha256`. Current platform jobs
still retain their independently hashed manifest even when those optional in-container
variables are unavailable. A `--dry-run` may show `source_commit` as unavailable, but its
config is marked `run_mode=dry-run` and cannot be exported as a reportable model.
