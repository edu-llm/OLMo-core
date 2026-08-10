"""Train Maple M20 (or any ladder rung) on YOUR OWN token shards. No eduLLM infrastructure.

=============================================================================================
UNTESTED. THIS SCRIPT HAS NEVER BEEN EXECUTED -- NOT ONCE, NOT EVEN `--dry-run`.
=============================================================================================

It was written by reading the library rather than by running it, because the environment it was
authored in is not permitted to execute anything computational. Every API call below was checked
against the signature in the source tree at this commit, and the numbers it prints come from
`TransformerConfig.MAPLE_EXPECTED_PARAMS` rather than from anything typed here. That is the most
that can honestly be claimed for it.

Treat the first `--dry-run` as the test. It needs no GPU, no data and no network: it builds the
config, runs every assertion in `_maple_assert_ladder`, and prints the parameter ledger. If it
prints matching counts, both this script and the M20 parameter prediction have been confirmed for
the first time. If it raises, read the traceback before trusting anything else here.

WHY THIS FILE EXISTS
--------------------
`.edullm/train_on_corpus.py` is the platform entrypoint and it cannot run outside our AWS
account: `resolve_corpus()` imports the private `edullm_data` package, whose `DATA_BUCKET` is
hard-coded with no environment override, and it is called as the first statement of
`build_config()` so no flag or override reaches past it. It also derives `vocab_size` from that
private manifest, so there is no `--vocab-size` to set.

This file is the same training setup with the corpus layer replaced by a list of paths. It
imports `olmo_core` only. See `docs/maple/PORTING-M20.md`, which you should read first -- in
particular the sections on `accumulate_grads_without_comm` (mandatory, and M20 does not fit on
40GB cards either way) and on the MoE defaults that are silently wrong upstream.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No held-out evaluation ladder. The platform version localises S3 validation shards through
boto3, and reproducing that here would reintroduce the coupling this file exists to remove.
Point `--val-data` at local shards if you want validation, or run this and evaluate separately.

USAGE
-----
    # Free. No GPU, no data. Do this first.
    python src/scripts/train/maple_m20_local.py --dry-run

    # Real run, 80GB cards. See PORTING-M20.md 3.1 -- 40GB will not fit M20.
    torchrun --nproc-per-node=8 src/scripts/train/maple_m20_local.py \
        --data /data/tokens/*.npy --save-folder /scratch/m20

    # A rung that fits a single small card, for code-path validation.
    python src/scripts/train/maple_m20_local.py --factory maple_r0 --dry-run
"""

import argparse
import logging
import sys

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, LinearWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    SteadyStateThroughputCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Maple M20 on your own token shards. UNTESTED -- read the module "
        "docstring and docs/maple/PORTING-M20.md first.",
    )

    parser.add_argument(
        "--factory",
        default="maple_m20",
        help="Any TransformerConfig classmethod: maple_m20, maple_r0..maple_r3. Dispatched by "
        "getattr, exactly as the platform entrypoint does it.",
    )
    parser.add_argument(
        "--data",
        nargs="*",
        default=[],
        help="Paths or URLs to .npy token-ID shards. Headerless flat integers -- see "
        "--data-dtype. Not required with --dry-run.",
    )
    parser.add_argument(
        "--val-data",
        nargs="*",
        default=[],
        help="Optional held-out shards. Accepted and recorded, but this script runs no eval "
        "ladder; see the module docstring.",
    )
    parser.add_argument(
        "--data-dtype",
        default="uint32",
        choices=[d.value for d in NumpyDatasetDType],
        help="Element type of the token shards. dolma2 shards are headerless uint32. Getting "
        "this wrong does not crash -- it silently reinterprets your corpus.",
    )
    parser.add_argument("--save-folder", default=None, help="Checkpoint dir. Local path is fine.")
    parser.add_argument("--work-dir", default="/tmp/maple-dataset-cache")

    # Tokenizer. Defaults are dolma2, which is what the ladder's param counts assume.
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="UNPADDED vocab size; padded to a multiple of 128 before it reaches the factory. "
        "Default is dolma2's 100,278 -> 100,352. Changing this invalidates the ladder's "
        "expected param counts, and the factory will refuse rather than silently accept.",
    )
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=None)

    # Shapes and schedule. Defaults mirror the platform entrypoint so numbers stay comparable.
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=786432)
    parser.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=4096,
        help="Tokens per rank per micro-step. NOTE this default is 4096, deliberately NOT the "
        "platform's 16384, which OOMs. It is also not numerics-neutral for MoE -- expert "
        "capacity derives from it -- so hold it fixed across arms you compare.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.4e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.033)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--z-loss-multiplier", type=float, default=1e-5)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=1337)
    parser.add_argument(
        "--param-dtype",
        default=DType.bfloat16.value,
        choices=[DType.bfloat16.value, DType.float32.value],
    )

    parser.add_argument(
        "--lm-loss-implementation",
        default="chunked_linear",
        choices=[i.value for i in LMLossImplementation],
        help="Default is chunked_linear, NOT the library default. There are five simultaneously "
        "live fp32 (N,V) logit buffers, not one. 'fused_linear' needs liger-kernel.",
    )
    parser.add_argument("--lm-loss-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--quantize",
        default="off",
        choices=["off", "control", "ternary"],
        help="Ternary QAT. NO training speedup -- see PORTING-M20.md 3.8. The win is inference.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model. Adds minutes to step 1."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the config, assert the ladder, print the parameter ledger, exit. No GPU, no "
        "data, no network. START HERE.",
    )
    return parser


def build_tokenizer(opts) -> TokenizerConfig:
    """dolma2 unless overridden. Overriding it moves the param counts, deliberately loudly."""
    base = TokenizerConfig.dolma2()
    return TokenizerConfig(
        vocab_size=opts.vocab_size if opts.vocab_size is not None else base.vocab_size,
        eos_token_id=opts.eos_token_id if opts.eos_token_id is not None else base.eos_token_id,
        pad_token_id=opts.pad_token_id if opts.pad_token_id is not None else base.pad_token_id,
        # identifier is left unset: nothing here loads a HF tokenizer, and claiming dolma2's
        # identifier while the caller supplied a different vocab size would be a lie in the
        # saved config.
        identifier=base.identifier if opts.vocab_size is None else None,
    )


def build_model_config(opts, tokenizer: TokenizerConfig) -> TransformerConfig:
    """Dispatch by name, exactly as `.edullm/train_on_corpus.py:856` does it."""
    factory = getattr(TransformerConfig, opts.factory, None)
    if factory is None:
        raise SystemExit(f"unknown model factory: {opts.factory}")

    kwargs = {}
    if opts.quantize != "off":
        # The factory takes tri-state `quantize`: None=off, False=control, True=ternary.
        kwargs["quantize"] = opts.quantize == "ternary"

    # PADDED vocab, matching the platform entrypoint. dolma2's 100,278 -> 100,352, which is the
    # key MAPLE_EXPECTED_PARAMS is filed under.
    return factory(vocab_size=tokenizer.padded_vocab_size(), **kwargs)


def print_param_ledger(opts, model_config: TransformerConfig, padded_vocab: int) -> None:
    """Print measured-vs-expected params, reading the table rather than restating it.

    This is the point of `--dry-run`. The M20 row of MAPLE_EXPECTED_PARAMS is the only DERIVED
    row in that table -- closed form plus an independent walk of the config tree, never a built
    model. If the two columns below agree for maple_m20, that prediction has just been confirmed
    for the first time.
    """
    total = model_config.num_params
    active = model_config.num_active_params

    print("\n=== PARAM LEDGER ===")
    print(f"factory          : {opts.factory}")
    print(f"padded vocab     : {padded_vocab:,}")
    print(f"total params     : {total:,}")
    print(f"active params    : {active:,}")

    rung = opts.factory.replace("maple_", "").upper()
    expected = TransformerConfig.MAPLE_EXPECTED_PARAMS.get(padded_vocab, {}).get(rung)
    if expected is None:
        print(f"expected         : no row for rung {rung!r} at vocab {padded_vocab:,}")
        print("                   (nothing to compare against -- not necessarily a problem)")
        return

    exp_total, exp_active = expected
    print(f"expected total   : {exp_total:,}   delta {total - exp_total:+,}")
    print(f"expected active  : {exp_active:,}   delta {active - exp_active:+,}")

    exp_amr = TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS.get(padded_vocab, {}).get(rung)
    if exp_amr is not None:
        print(f"expected active-minus-routers: {exp_amr:,}")
        print("  (this, not plain active, is the quantity that is exactly invariant across the")
        print("   E-sweep -- routers are L*d*E and every token traverses all of them)")

    if rung == "M20":
        print("\nM20 is the only DERIVED row in that table. It has never been built.")
        print("Matching numbers above are a first confirmation, not a re-check.")


def build_configs(opts):
    tokenizer = build_tokenizer(opts)
    padded_vocab = tokenizer.padded_vocab_size()
    model_config = build_model_config(opts, tokenizer)

    if model_config.lm_head is not None:
        model_config.lm_head.loss_implementation = LMLossImplementation(opts.lm_loss_implementation)
        model_config.lm_head.loss_chunk_size = opts.lm_loss_chunk_size

    dataset_config = NumpyFSLDatasetConfig(
        # `or None` because the base config treats an empty list and None differently, and a
        # --dry-run legitimately has no paths. Validation only runs at .build(), which --dry-run
        # never reaches.
        paths=list(opts.data) or None,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType(opts.data_dtype),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.learning_rate,
            betas=(opts.beta1, opts.beta2),
            eps=opts.adam_eps,
            weight_decay=opts.weight_decay,
            group_overrides=[
                # Exempts the embedding matrix from weight decay. Keep this.
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType(opts.param_dtype),
            reduce_dtype=DType.float32,
            # ================== DO NOT REMOVE THIS LINE ==================
            # The library default is True, which keeps an UNSHARDED fp32 gradient accumulator:
            # 74.51 GiB/rank at 20B, which fits nothing. False brings it to 40.30 GiB/rank --
            # still 106.1% of an A100-40GB, about 51% of an H100-80GB.
            # The platform entrypoint does NOT set this and therefore inherits True.
            accumulate_grads_without_comm=False,
        ),
        max_grad_norm=1.0,
        scheduler=LinearWithWarmup(alpha_f=0.0, warmup_fraction=opts.warmup_fraction),
        # `or None` so that 0 means OFF rather than on-with-a-zero-coefficient, which would pay
        # for the computation and then divide by it.
        z_loss_multiplier=opts.z_loss_multiplier or None,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder or opts.work_dir,
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        # Median step time and MFU over a window that opens AFTER warmup. Neither stock MFU
        # metric excludes torch.compile's first step or cold-shard stalls.
        .with_callback("steady_state", SteadyStateThroughputCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                # The platform sets None because its IAM role cannot delete .metadata.json. On
                # local disk the library default of 3 is probably what you want -- None keeps
                # every checkpoint and a 20B model will fill a disk quickly.
                max_checkpoints=3,
                save_async=True,
            ),
        )
    )

    return (
        model_config,
        dataset_config,
        data_loader_config,
        train_module_config,
        trainer_config,
        padded_vocab,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts = build_parser().parse_args()

    (
        model_config,
        dataset_config,
        data_loader_config,
        train_module_config,
        trainer_config,
        padded_vocab,
    ) = build_configs(opts)

    if opts.dry_run:
        print_param_ledger(opts, model_config, padded_vocab)
        return

    if not opts.data:
        raise SystemExit("--data is required for a real run (use --dry-run to build config only)")
    if not opts.save_folder:
        raise SystemExit("--save-folder is required for a real run")

    prepare_training_environment()
    try:
        seed_all(opts.seed)
        if get_rank() == 0:
            print_param_ledger(opts, model_config, padded_vocab)

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        dataset = dataset_config.build()
        data_loader = data_loader_config.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = trainer_config.build(train_module, data_loader)

        # Enables the MoE drop histograms. Left unset both positional histograms come back `nan`
        # rather than erroring, so the failure is silent and looks like the metric is broken.
        wired = 0
        for module in model.modules():
            if hasattr(module, "drop_accounting_seq_len"):
                module.drop_accounting_seq_len = opts.sequence_length
                wired += 1
        if wired == 0:
            log.warning(
                "no module exposed 'drop_accounting_seq_len'; on an MoE run every per-position "
                "drop histogram will be nan"
            )

        if opts.quantize != "off":
            from olmo_core.nn.quantization import audit_quantization

            # Raises if anything in the full-precision carve-out (embeddings, lm_head, router,
            # norms) got quantized. The router carve-out is load-bearing: routing is discrete,
            # so quantizing the router changes which experts fire.
            audit_quantization(model)

        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    sys.exit(main())
