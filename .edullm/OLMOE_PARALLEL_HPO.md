# OLMoE parallel-node HPO

The OLMoE study has three additive probe specifications:

- `hpo-olmoe-no-proxy.json`: stock OLMoE-1B-7B with the 30% Centaur policy.
- `hpo-olmoe-no-centaur.json`: the identical stock OLMoE-1B-7B study with Centaur disabled.
- `hpo-olmoe-curriculum-no-proxy.json`: OLMoE-1B-7B with the warmup quadratic
  MTLD curriculum and the 30% Centaur policy. This is the curriculum/no-proxy capacity run.

These files do not replace the historical dense `hpo-no-proxy.json` or
`hpo-no-centaur.json` arms. The two stock OLMoE arms use
`olmo_core.hpo.comparison:build_olmoe_hpo_experiment`, exact fidelity, and an eight-dimensional
search space. The curriculum arm composes that same model recipe with the existing arm 9
token-progress loader. The global batch is fixed rather than searched.

## Fixed batch and fidelity contract

- Global batch: 262,144 tokens per optimizer step.
- Rank microbatch: 8,192 tokens. Live one-step probes OOMed at both 32,768 and 16,384 tokens
  per rank on an 80 GiB H100, so this is the next-largest divisor of the per-rank global batch.
- Sequence length: 2,048 tokens.
- Worker world size: 8 ranks on one node.
- Gradient accumulation: `262144 / (8 * 8192) = 4`.
- Capacity-block data source: the read-only `edullm-data-us-east-2` corpus mirror.
- Stock-arm quantum/target/budget: 49,807,360 / 499,908,608 / 2,000,158,720 tokens.
- Curriculum-arm quantum/target/budget: 50,331,648 / 503,316,480 / 2,013,265,920 tokens.

BTT minimum fidelity and the initial IPBT update interval equal one quantum. The batch,
microbatch, sequence length, topology, quantum, target, and budget are frozen study inputs, not
search dimensions.

## Parallel capacity-block dispatch

The CPU-only controller discovers currently IDLE capacity-block nodes before every round. It sets
the usable worker count to the smaller of the number of IDLE nodes and `max_workers=6`, then makes
one `block-run.yml` dispatch per IDLE node selected for that round. Each dispatch binds one trial
segment to one full node with `processes=all`; different node IDs can therefore run independent
trials concurrently.

Do not use `block-run-distributed.yml` for this study. That workflow joins several nodes into one
distributed job, whereas this study requires one independent eight-rank trial per node. Re-probe
capacity before the next wave so newly IDLE nodes can be used. Controller observations,
checkpoints, snapshots, and the final study result must remain under the arm's durable
`EDULLM_CHECKPOINT_DIR` root.

The dispatch shape is:

```text
workflow: block-run.yml
branch: edullm/hpo-complex
node: one IDLE node ID
processes: all
command: python .edullm/hpo_on_corpus.py RUN_ID --run-segment --worker-world-size=8 --segment-spec-payload=... --param-dtype=bfloat16 ...
```

This is a shape reference, not an instruction to dispatch before the checks below pass.

## Hard launch safety rules

1. **No launcher in `command`.** `processes=all` prepends `torchrun`; the command starts with
   `python`, never `torchrun`, `torch.distributed.run`, or another launcher.
2. **No expert-mesh CLI flags.** Do not add `--moe-shard-degree` or `--moe-num-replicas`.
   The additive OLMoE factory derives EP/HSDP from the launched process group. If this entrypoint
   is ever placed on a distributed workflow, set `mesh_flags=false`.
3. **No bare trailing override.** The first bare word after the program can be consumed as the
   positional run name. Every configuration override must use explicit `--flag=value` syntax.
4. **Never claim capacity without preflight.** Perform the dry-run and refuse-busy check first,
   and dispatch only node IDs that the immediately preceding capacity probe reports IDLE.

The controller stays outside the worker `torchrun` process group. One node is one trial; never
combine several IDLE nodes into one trial for these probe specs.
