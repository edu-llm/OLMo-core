"""
Train one grid cell on the eduLLM platform.

    python src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" \\
        --cell src/scripts/train/factcrowd/configs/cells/28m_d1p2.yaml \\
        --save-folder "$EDULLM_CHECKPOINT_DIR"

One cell is one run. For a fan-out over a whole row, pass ``--row 28M`` instead of ``--cell`` and the
cell is chosen by ``AWS_BATCH_JOB_ARRAY_INDEX``; the grid is submitted as three jobs, one per row, so
the rows run concurrently and a failure in one does not strand the others.

``--dry-run`` resolves the cell, builds the corpus, prints the plan and trains nothing. It is the
cheapest way to find a bad config, and everything it checks it checks without a GPU.

WHAT THIS FILE DOES NOT DO, AND WHY THAT MATTERS. It builds no model architecture, no optimizer, no
learning-rate schedule, no checkpointer, no data loader and no packing. Every one of those exists in
OLMo-core and is assembled here rather than reimplemented: ``TransformerConfig.llama_like`` for the
model, ``AdamWConfig`` and ``WSD`` for the optimisation, ``ConcatAndChunkInstanceSource`` for packing
a token stream into instances, ``ComposableDataLoaderConfig`` for shuffling and batching,
``ListCheckpointerCallback`` for the log-spaced snapshots. What is ours is the corpus -- the entity
table, the renderer, the token stream -- and the arithmetic that places a cell on the demand axis.

THE PLATFORM CONSTRAINTS THIS FILE SATISFIES. Each was a lost run for somebody.

  --save-folder must be on the command line, not only in this program. The platform reads the command
  text to check that a run promising a checkpoint writes one, and cannot see inside the process. The
  OLMo-core default is /tmp, on a machine that stops existing, so a run that takes it trains for a day,
  writes checkpoints nobody can reach, exits zero and is recorded as a success.

  max_checkpoints must be null. OLMo-core keeps three and prunes the rest, and the prune deletes
  .metadata.json first -- a key the workload role is denied by name -- so the run dies with a network
  error about an hour in. It would also delete seven of the ten snapshots this experiment needs.

  ephemeral_save_interval must be null, or the config is refused in the first seconds.

  lm_evaluator and downstream_evaluator must be off. The first reads a C4 validation shard whose index
  was never published; the second needs a package the training image does not install. Both fail while
  the trainer is being built.

  max_duration must be set explicitly, since the default is one epoch.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Two path fixups, both no-ops on the platform, both needed to run this from a checkout.
# ``factcrowd`` lives in a script directory rather than an installed package, so its parent has to be
# importable; and ``olmo_core`` is installed in the training image but only on the path in a checkout.
if __package__ in (None, ""):  # pragma: no cover - only when run as a script
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[1]))
    try:
        import olmo_core  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_here.parents[3]))

from factcrowd import cells as cell_module  # noqa: E402
from factcrowd.corpus import render as render_module  # noqa: E402
from factcrowd.corpus import stream as stream_module  # noqa: E402
from factcrowd.corpus import values as values_module  # noqa: E402
from factcrowd.corpus import vocab as vocab_module  # noqa: E402
from factcrowd.ladder import sizes as sizes_module  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402

log = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
"""Where the committed cell configs live."""

DOMAIN_TOKENS: Tuple[str, ...] = ("<facts>", "<mano>", "<brevo>", "<related>")
"""One per corpus slice. Prepended to every segment; there is no flag to disable it."""

CHECKPOINT_FRACTIONS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.50,
    0.75,
    1.00,
)
"""
Log-spaced snapshots. Capacity fills fast then asymptotes, so linear spacing puts half its points on
the flat part; this puts seven in the first third, where the bits curve moves and where a
double-descent bump at crossover would appear. 8% is also where a compute-matched comparison against
the lowest-demand cell lands, which is why that reading costs nothing extra.
"""


class BuiltCorpus:
    """
    The corpus objects a cell implies, built once and shared by the dry run and the real run.

    :param resolved: The resolved cell.
    :param work_dir: Local scratch for the entity table and the offset index.
    """

    def __init__(self, resolved: cell_module.ResolvedCell, work_dir: Path) -> None:
        from factcrowd.corpus import entities as entities_module

        spec = resolved.spec
        if spec.sweep == "count":
            self.templates = render_module.BIOS_TEMPLATES
            literals = render_module.literal_words_of(self.templates)
            self.corpus_schema = values_module.bios_schema(
                reserved=tuple(literals) + vocab_module.SPECIALS + DOMAIN_TOKENS
            )
        else:
            assert spec.bits_per_attribute is not None
            self.templates = render_module.entropy_templates(
                values_module.ENTROPY_ATTRIBUTES, values_module.ENTROPY_WORDS_PER_VALUE
            )
            literals = render_module.literal_words_of(self.templates)
            self.corpus_schema = values_module.entropy_schema(
                spec.bits_per_attribute,
                reserved=tuple(literals) + vocab_module.SPECIALS + DOMAIN_TOKENS,
            )

        self.vocabulary = vocab_module.Vocabulary.build(
            self.corpus_schema.schema, literal_words=literals, domain_tokens=DOMAIN_TOKENS
        )
        table_entities = spec.table_entities or resolved.n_entities
        self.table = entities_module.EntityTable.build(
            self.corpus_schema.schema, table_entities, spec.seed
        )
        self.renderer = render_module.Renderer(
            self.table,
            self.corpus_schema,
            self.vocabulary,
            self.templates,
            domain_token=DOMAIN_TOKENS[0],
            seed=spec.seed + 1,
            min_templates=1 if spec.sweep == "entropy" else 20,
        )
        self.stream = stream_module.BioStream(
            self.renderer,
            n_entities=resolved.n_entities,
            exposures=spec.exposures,
            work_dir=work_dir,
            seed=spec.seed + 2,
        )

    def summary(self, resolved: cell_module.ResolvedCell) -> Dict[str, Any]:
        """
        Everything worth printing before a run, with the *measured* token count rather than an estimate.

        :param resolved: The resolved cell.

        :returns: A flat mapping.
        """
        out = resolved.summary(self.renderer.mean_tokens_per_bio)
        out.update(
            {
                "vocab_size": self.vocabulary.size,
                "vocab_size_padded": self.vocabulary.padded_size(),
                "templates": self.renderer.n_templates,
                "tokens_per_bio_min": int(self.renderer.template_lengths.min()),
                "tokens_per_bio_max": self.renderer.max_tokens_per_bio,
                "bits_per_token": round(self.renderer.bits_per_token, 4),
                "fact_tokens_measured": self.stream.num_tokens,
                "table_entities": self.table.n_entities,
                "schema_fingerprint": self.corpus_schema.schema.fingerprint()[:16],
                "stream_fingerprint": self.stream.fingerprint()[:16],
            }
        )
        # The measured count supersedes the estimate; keep both so a drift is visible.
        out["total_tokens"] = self.stream.num_tokens + resolved.spec.reasoning_tokens
        out["steps"] = out["total_tokens"] // resolved.spec.global_batch_size
        return out


def resolve_cell(args: argparse.Namespace) -> cell_module.CellSpec:
    """
    Which cell this process should train.

    Either a named config, or a row plus the fan-out index the platform sets. The row form sorts the
    row's configs by filename so the index-to-cell mapping is a function of the directory and nothing
    else -- a mapping that changed between submission and execution would run the wrong cell under the
    right name.

    :param args: Parsed arguments.

    :returns: The cell.

    :raises OLMoConfigurationError: If neither form is usable, or the index is out of range.
    """
    if args.cell:
        return cell_module.load_cell(args.cell)
    if not args.row:
        raise OLMoConfigurationError("pass either --cell <config> or --row <ladder row>")

    index_text = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", args.cell_index)
    if index_text is None:
        raise OLMoConfigurationError(
            "--row needs a fan-out index: the platform sets AWS_BATCH_JOB_ARRAY_INDEX, or pass "
            "--cell-index for a single run"
        )
    # Each axis has its own directory so a fan-out size is what `ls` says. Sharing one would let
    # --row 28M pick up eleven cells where the submission asked for five, and run the wrong cell under
    # the right name.
    directory = CONFIG_ROOT / "cells" / args.sweep
    row_cells = tuple(
        cell for cell in cell_module.load_cells(directory) if cell.row.lower() == args.row.lower()
    )
    if not row_cells:
        raise OLMoConfigurationError(f"no cells found for row {args.row!r} in {directory}")
    index = int(index_text)
    if not 0 <= index < len(row_cells):
        raise OLMoConfigurationError(
            f"fan-out index {index} is out of range for row {args.row}, which has "
            f"{len(row_cells)} cells: {[c.cell_id for c in row_cells]}"
        )
    return row_cells[index]


def build_trainer(
    resolved: cell_module.ResolvedCell, corpus: BuiltCorpus, args: argparse.Namespace
) -> Any:
    """
    Assemble the OLMo-core objects for a cell and return a trainer ready to ``fit()``. Needs ``torch``.

    Every field named here was read off the class rather than assumed. That is not caution for its own
    sake: an earlier module in this branch passed ``d_model`` to a factory that hardcodes it, which
    raised ``TypeError`` for every input, type-checked clean and was only caught by reading the source.
    ``ListCheckpointerCallback`` takes ``save_steps`` and not ``steps``, which is the same mistake
    waiting to happen.

    :param resolved: The resolved cell.
    :param corpus: The built corpus.
    :param args: Parsed arguments.

    :returns: The trainer.
    """
    import torch
    from factcrowd.corpus.source import BioTokenSource

    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ComposableDataLoaderConfig,
        ConcatAndChunkInstanceSource,
    )
    from olmo_core.optim import WSD, AdamWConfig
    from olmo_core.train import Duration, TrainerConfig
    from olmo_core.train.callbacks import (
        ConfigSaverCallback,
        GPUMemoryMonitorCallback,
        ListCheckpointerCallback,
        SpeedMonitorCallback,
    )
    from olmo_core.train.train_module import TransformerTrainModuleConfig

    spec = resolved.spec
    work_dir = str(args.work_dir)

    # The model. sizes.build asserts the built non-embedding count against the ladder, because rho is
    # computed from it and a 1% error there is a 1% error in every entity count.
    model_config = sizes_module.build(
        spec.ladder_row, corpus.vocabulary.padded_size(), tie_word_embeddings=True
    )

    total_tokens = corpus.stream.num_tokens + spec.reasoning_tokens
    steps = total_tokens // spec.global_batch_size
    checkpoint_steps = sorted(
        {min(steps, max(1, int(fraction * steps))) for fraction in CHECKPOINT_FRACTIONS}
    )
    if len(checkpoint_steps) < 3:
        raise OLMoConfigurationError(
            f"cell '{spec.cell_id}' runs for {steps} steps, which the log-spaced schedule collapses "
            f"to {len(checkpoint_steps)} distinct checkpoints ({checkpoint_steps}). A bits curve "
            f"needs more than that. Lower 'global_batch_size' or raise the demand."
        )
    if len(checkpoint_steps) < len(CHECKPOINT_FRACTIONS):
        # Expected on a smoke cell, never on a real one: at 3,390 steps -- the shortest cell in the
        # grid -- the ten fractions land on ten distinct steps, and a test asserts that for every
        # committed cell.
        log.warning(
            "cell '%s' runs for %d steps, so the ten log-spaced fractions collapse to %d distinct "
            "checkpoints: %s. Fine for a smoke run, wrong for a cell whose bits curve gets read.",
            spec.cell_id,
            steps,
            len(checkpoint_steps),
            checkpoint_steps,
        )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=args.rank_microbatch_size,
        max_sequence_length=spec.sequence_length,
        optim=AdamWConfig(lr=spec.learning_rate, weight_decay=spec.weight_decay),
        # decay_fraction rather than decay: WSD defaults decay_fraction to 0.1 and refuses both,
        # so passing the step count would need decay_fraction=None alongside it.
        scheduler=WSD(warmup=spec.warmup_steps, decay_fraction=spec.decay_fraction),
        compile_model=args.compile_model,
    )

    token_source = BioTokenSource(
        corpus.renderer,
        n_entities=resolved.n_entities,
        exposures=spec.exposures,
        work_dir=work_dir,
        seed=spec.seed + 2,
        label=f"facts:{spec.cell_id}",
    )
    # Packing is OLMo-core's. This turns the token stream into fixed-length instances; the reasoning
    # slices join through MixingInstanceSource once they exist.
    instance_source = ConcatAndChunkInstanceSource(
        token_source, sequence_length=spec.sequence_length, work_dir=work_dir
    )

    trainer_config = (
        TrainerConfig(
            save_folder=args.save_folder,
            work_dir=work_dir,
            save_overwrite=False,
            metrics_collect_interval=10,
            max_duration=Duration.steps(steps),
            # Async bookkeeping runs collectives on a second process group. Under gloo that group is
            # not registered where torch's distributed checkpointing expects it, so a CPU run dies in
            # scatter_object on its first save. Off on CPU, on wherever it works -- and the CPU path
            # matters because cpu-32vcpu is the profile the platform sends every first run to.
            async_bookkeeping=torch.cuda.is_available(),
        )
        .with_callback("speed_monitor", SpeedMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            ListCheckpointerCallback(
                save_steps=checkpoint_steps,
                # Ten snapshots is the point. OLMo-core keeps three by default, and on this platform
                # the prune deletes .metadata.json first -- a key the workload role is denied by name
                # -- so the run would die about an hour in having thrown away seven of them.
                max_checkpoints=None,
                ephemeral_save_interval=None,
            ),
        )
    )

    # The loader needs the special ids to collate: it pads with pad_token_id and counts eos_token_id
    # to find document boundaries. Taking them from our own vocabulary rather than a named tokenizer
    # is what keeps the two from disagreeing -- padding with the wrong id would put a real word in
    # every gap, and sharing pad with eos would inflate the document count until it explodes.
    tokenizer_config = TokenizerConfig(
        vocab_size=corpus.vocabulary.padded_size(),
        eos_token_id=corpus.vocabulary.eos_id,
        pad_token_id=corpus.vocabulary.pad_id,
        bos_token_id=corpus.vocabulary.bos_id,
        identifier=f"factcrowd-words-{corpus.vocabulary.fingerprint()[:12]}",
    )
    # GPUMemoryMonitorCallback calls torch._C._cuda_resetPeakMemoryStats unconditionally, which does
    # not exist in a CPU build -- so on CPU it kills the run in pre_train, before the first step, and
    # the process-group error that follows is only the teardown tripping over the first failure.
    if torch.cuda.is_available():
        trainer_config = trainer_config.with_callback("gpu_monitor", GPUMemoryMonitorCallback())

    data_loader_config = ComposableDataLoaderConfig(
        global_batch_size=spec.global_batch_size, seed=spec.seed + 3, work_dir=work_dir
    )

    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    data_loader = data_loader_config.build(
        instance_source,
        tokenizer=tokenizer_config,
        dp_process_group=train_module.dp_process_group,
    )
    return trainer_config.build(train_module, data_loader)


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    """
    Entry point.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="The run name; on the platform, $EDULLM_RUN_ID")
    parser.add_argument("--cell", help="Path to a cell config")
    parser.add_argument("--row", help="Ladder row, for a fan-out over that row's cells")
    parser.add_argument(
        "--sweep",
        default="count",
        choices=("count", "entropy"),
        help="Which axis --row selects from. Each has its own config directory, so a fan-out size is "
        "what 'ls' says.",
    )
    parser.add_argument("--cell-index", help="Fan-out index, if not set by the platform")
    parser.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR"),
        help="Where checkpoints go; on the platform, $EDULLM_CHECKPOINT_DIR",
    )
    parser.add_argument("--work-dir", default="/tmp/factcrowd", help="Local scratch")
    parser.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="torch.compile the model. Off by default: Inductor needs a C compiler, and "
        "older images do not carry one.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve, build the corpus, print, stop"
    )
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    args = parser.parse_args(argv)

    spec = resolve_cell(args)
    resolved = spec.resolve()
    work_dir = Path(args.work_dir) / spec.cell_id
    work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = work_dir

    corpus = BuiltCorpus(resolved, work_dir)
    summary = corpus.summary(resolved)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        width = max(len(key) for key in summary)
        print(f"factcrowd cell '{spec.cell_id}' -- run {args.run_name}")
        for key, value in summary.items():
            rendered = f"{value:,}" if isinstance(value, int) else str(value)
            print(f"  {key.rjust(width)}  {rendered}")
        print(f"  {'sample'.rjust(width)}  {corpus.renderer.text(0, 0)[:160]}")

    if args.dry_run:
        return 0

    if not args.save_folder:
        raise OLMoConfigurationError(
            "--save-folder is required for a real run. The OLMo-core default is /tmp, on a machine "
            "that stops existing, so a run that takes it exits zero having saved nothing -- and the "
            "platform refuses a submission whose command text does not name it."
        )

    # The distributed environment has to be up before any model is built, and torn down afterwards,
    # even on one device: OLMo-core's train module and data loader both consult the process group.
    # seed_all covers the parts of initialisation the cell's own seeds do not reach -- parameter init
    # and dropout -- so two runs of one cell start from the same weights.
    import torch

    from olmo_core.train import (
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.utils import seed_all

    # The default backend names CUDA, and init_distributed then calls torch.cuda.set_device for it --
    # which fails outright on a CPU-only build. Selecting gloo keeps the cheapest platform profile
    # (cpu-32vcpu, the one its guide sends every first run to) usable for a smoke test.
    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    prepare_training_environment(backend=backend)
    try:
        seed_all(spec.seed)
        build_trainer(resolved, corpus, args).fit()
    finally:
        teardown_training_environment()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
