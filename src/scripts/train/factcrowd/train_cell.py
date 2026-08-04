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
from typing import Any, Dict, List, Optional, Tuple

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
from factcrowd.corpus import tasks as tasks_module  # noqa: E402
from factcrowd.corpus import values as values_module  # noqa: E402
from factcrowd.corpus import vocab as vocab_module  # noqa: E402
from factcrowd.ladder import sizes as sizes_module  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402

log = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
"""Where the committed cell configs live."""

DOMAIN_TOKENS: Tuple[str, ...] = ("<facts>", "<mano>", "<compare>")
"""One per corpus slice. Prepended to every segment; there is no flag to disable it."""

MANO_LENGTH = 10
"""
Expression length for Mano. Ten, not thirteen.

At thirteen the task sits about a point above its own degenerate policy at 13M-28M, which fails
the 20-80% admission band it is meant to pass. At ten Physics 4.1 reports 47.8 to 66.0 from
scratch at our exact twelve layers, moving 18.2 points across the parameter range.
"""

TASK_WORDS: Tuple[str, ...] = tasks_module.all_required_words(
    (tasks_module.ManoTask, tasks_module.CompareTask)
)
"""
Words the reasoning tasks need, taken off the classes before anything is built.

The ordering is forced: a task cannot be constructed without a vocabulary containing its tokens,
and the vocabulary cannot be built without knowing which tokens to reserve -- and the pool
allocator has to avoid them too, or a generated city could collide with an operator.
"""

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
                reserved=tuple(literals) + TASK_WORDS + vocab_module.SPECIALS + DOMAIN_TOKENS
            )
        else:
            assert spec.bits_per_attribute is not None
            self.templates = render_module.entropy_templates(
                values_module.ENTROPY_ATTRIBUTES, values_module.ENTROPY_WORDS_PER_VALUE
            )
            literals = render_module.literal_words_of(self.templates)
            self.corpus_schema = values_module.entropy_schema(
                spec.bits_per_attribute,
                reserved=tuple(literals) + TASK_WORDS + vocab_module.SPECIALS + DOMAIN_TOKENS,
            )

        self.vocabulary = vocab_module.Vocabulary.build(
            self.corpus_schema.schema,
            literal_words=tuple(literals) + TASK_WORDS,
            domain_tokens=DOMAIN_TOKENS,
        )
        # The reasoning-only control has no entities, so it has no table, no renderer and no fact
        # stream. It keeps the schema and the vocabulary above: the model must be architecturally
        # identical to the row's other cells, and dropping ~1,700 unused word embeddings would change
        # the parameter count of the one cell the others are compared against.
        self.table: Optional[Any] = None
        self.renderer: Optional[Any] = None
        self.stream: Optional[Any] = None
        if not spec.is_control:
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

        # The reasoning slices. Each is stated in absolute tokens and is identical in every cell that
        # carries it, which is the invariant that stops reasoning volume moving with fact load -- hold
        # the ratio instead and the result is confounded in both directions.
        #
        # Per slice, not split between them. Splitting a fixed total was the first thing written here
        # and it silently broke the control: the control carries no related slice, so an evenly split
        # total gave it 2x the Mano exposure of every cell it is the reference for, and its Mano score
        # would have been higher for that reason alone. A cell's reasoning total therefore varies with
        # how many slices it carries, and that is correct -- what an endpoint's score must be compared
        # at is exposure to *that endpoint's* data.
        self.task_streams: Tuple[tasks_module.TaskStream, ...] = ()
        if spec.reasoning_tokens > 0:
            built: list = [
                tasks_module.ManoTask(
                    self.vocabulary, domain_token="<mano>", length=MANO_LENGTH, seed=spec.seed + 4
                )
            ]
            # The related-reasoning slice asks about the fact schema, so it exists only where that
            # schema has fields to ask about. The entropy axis's attributes are abstract by
            # construction -- six positional attributes of four words each, with no ordinal field to
            # compare -- so that axis carries unrelated reasoning alone. It costs nothing: the entropy
            # axis's job is the identified demand sweep, and the related-reasoning prediction lives on
            # the count axis where bioS does have a birth year.
            if (
                spec.sweep == "count"
                and self.table is not None
                and spec.related_reasoning_tokens > 0
            ):
                built.append(
                    tasks_module.CompareTask(
                        self.table,
                        self.corpus_schema,
                        self.vocabulary,
                        domain_token="<compare>",
                        probe_ids=self.table.probe_ids,
                        seed=spec.seed + 5,
                    )
                )
            self.task_streams = tuple(
                tasks_module.TaskStream(
                    task, num_tokens=spec.slice_budget(task.name), label=task.name
                )
                for task in built
            )
        # The cell's own arithmetic declares which slices it carries, and every token estimate and cost
        # figure is computed from that. Asserting the built set against it means a divergence fails here
        # rather than showing up as a run whose step count nobody can reproduce.
        names = tuple(stream.task.name for stream in self.task_streams)
        if names != spec.reasoning_slice_names:
            raise OLMoConfigurationError(
                f"cell '{spec.cell_id}' built reasoning slices {names} but its arithmetic declares "
                f"{spec.reasoning_slice_names}. One of the two selection rules has drifted, and the "
                f"cell's token count is computed from the declaration."
            )

    @property
    def fact_tokens(self) -> int:
        """Measured tokens in the fact slice; zero on the reasoning-only control."""
        return 0 if self.stream is None else int(self.stream.num_tokens)

    def summary(self, resolved: cell_module.ResolvedCell) -> Dict[str, Any]:
        """
        Everything worth printing before a run, with the *measured* token count rather than an estimate.

        :param resolved: The resolved cell.

        :returns: A flat mapping.
        """
        out = resolved.summary(0.0 if self.renderer is None else self.renderer.mean_tokens_per_bio)
        out.update(
            {
                "vocab_size": self.vocabulary.size,
                "vocab_size_padded": self.vocabulary.padded_size(),
                "schema_fingerprint": self.corpus_schema.schema.fingerprint()[:16],
                "reasoning_slices": ", ".join(
                    f"{s.task.name}:{s.n_items:,} items/{s.num_tokens:,} tokens"
                    for s in self.task_streams
                )
                or "none",
            }
        )
        if self.renderer is not None and self.stream is not None and self.table is not None:
            out.update(
                {
                    "templates": self.renderer.n_templates,
                    "tokens_per_bio_min": int(self.renderer.template_lengths.min()),
                    "tokens_per_bio_max": self.renderer.max_tokens_per_bio,
                    "bits_per_token": round(self.renderer.bits_per_token, 4),
                    "fact_tokens_measured": self.stream.num_tokens,
                    "table_entities": self.table.n_entities,
                    "stream_fingerprint": self.stream.fingerprint()[:16],
                }
            )
        else:
            out["fact_tokens_measured"] = 0
        # The measured count supersedes the estimate; keep both so a drift is visible.
        out["total_tokens"] = self.fact_tokens + sum(
            stream.num_tokens for stream in self.task_streams
        )
        out["steps"] = out["total_tokens"] // resolved.spec.global_batch_size
        return out


def resolve_platform_value(name: str, value: Optional[str]) -> Optional[str]:
    """
    Refuse a value that is still the literal text of an environment variable.

    **This exists because it already cost three runs.** The platform's ``command`` field is exec'd
    with no shell, so ``$EDULLM_CHECKPOINT_DIR`` only becomes a path if the command starts with
    ``bash -lc``. Without it the text arrives verbatim, and OLMo-core creates a *directory named*
    ``$EDULLM_CHECKPOINT_DIR`` rather than failing -- so a run trains, writes checkpoints into a
    container-local path, exits zero, and is recorded as a success with nothing recoverable. The
    platform's own checkpoint guard catches this at submission, but only under a profile that
    declares a checkpoint contract; ``olmo-core-check`` does not, so a smoke run sails through.

    Falling back to ``os.environ`` would be the friendlier choice and is the wrong one: the platform
    reads the *command text* to decide whether a run promises a checkpoint, so quietly making an
    unexpanded command work would leave the manifest disagreeing with the run. Refusing names the
    fix instead.

    :param name: Which argument is being checked, for the message.
    :param value: The value as parsed.

    :returns: The value, unchanged, when it is not an unexpanded variable.

    :raises OLMoConfigurationError: If it is.
    """
    if value is None or not value.startswith("$"):
        return value
    variable = value.split("/")[0].lstrip("$").strip("{}")
    resolved = os.environ.get(variable)
    raise OLMoConfigurationError(
        f"{name} is the literal text {value!r}, so nothing expanded it. The platform exec's the "
        f"command with no shell, so it has to start with `bash -lc` for a variable to become a "
        f"value:\n"
        f"  bash -lc 'python src/scripts/train/factcrowd/train_cell.py \"$EDULLM_RUN_ID\" ...'\n"
        f"{variable} is currently "
        + (f"set to {resolved!r} in the environment" if resolved else "not set in the environment")
        + ". Without the shell, OLMo-core would create a directory by that literal name and this "
        "run would finish, exit zero, and leave nothing anybody can read."
    )


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
    from factcrowd.corpus.source import BioTokenSource, TaskTokenSource

    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import (
        ComposableDataLoaderConfig,
        ConcatAndChunkInstanceSource,
        MixingInstanceSource,
        MixingInstanceSourceSpec,
    )
    from olmo_core.optim import WSD, AdamWConfig
    from olmo_core.train import Duration, TrainerConfig
    from olmo_core.train.callbacks import (
        ConfigSaverCallback,
        GPUMemoryMonitorCallback,
        ListCheckpointerCallback,
        SpeedMonitorCallback,
        WandBCallback,
    )
    from olmo_core.train.train_module import TransformerTrainModuleConfig

    spec = resolved.spec
    work_dir = str(args.work_dir)

    # The model. sizes.build asserts the built non-embedding count against the ladder, because rho is
    # computed from it and a 1% error there is a 1% error in every entity count.
    model_config = sizes_module.build(
        spec.ladder_row, corpus.vocabulary.padded_size(), tie_word_embeddings=True
    )

    total_tokens = corpus.fact_tokens + sum(stream.num_tokens for stream in corpus.task_streams)
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
        scheduler=WSD(warmup=resolved.warmup(steps), decay_fraction=spec.decay_fraction),
        compile_model=args.compile_model,
    )

    fact_target: List[Tuple[Any, int, str]] = []
    if corpus.renderer is not None:
        token_source = BioTokenSource(
            corpus.renderer,
            n_entities=resolved.n_entities,
            exposures=spec.exposures,
            work_dir=work_dir,
            seed=spec.seed + 2,
            label=f"facts:{spec.cell_id}",
        )
        # Packing is OLMo-core's: this turns a token stream into fixed-length instances.
        fact_target = [
            (
                ConcatAndChunkInstanceSource(
                    token_source, sequence_length=spec.sequence_length, work_dir=work_dir
                ),
                token_source.num_tokens,
                "facts",
            )
        ]

    if not fact_target and not corpus.task_streams:
        raise OLMoConfigurationError(
            f"cell '{spec.cell_id}' has neither facts nor reasoning, so there is nothing to train on"
        )
    if not corpus.task_streams:
        instance_source = fact_target[0][0]
    else:
        # The mixture, at ABSOLUTE token counts. MixingInstanceSource takes ratios, so the ratios are
        # derived from the absolute targets and the total is pinned with num_tokens -- which gives
        # absolute counts as long as no source runs short.
        #
        # If one does, composable/utils.py scales *every* source down by the same factor to preserve
        # the ratios. That is the right behaviour for a corpus mixture and the wrong one here: it would
        # keep the ratios the design does not care about and silently move the absolute volumes it
        # does. So each source is checked against its target first, and a shortfall is a refusal.
        # Counted in *instances*, not tokens. Chunking drops whatever is left over after the last
        # whole instance, so a token-denominated target is always a few tokens above what the source
        # can offer and the check below would fire on its own rounding. Instances are exact on both
        # sides, and the mixer takes num_instances directly.
        targets = fact_target + [
            (
                ConcatAndChunkInstanceSource(
                    TaskTokenSource(stream, work_dir=work_dir),
                    sequence_length=spec.sequence_length,
                    work_dir=work_dir,
                ),
                stream.num_tokens,
                stream.task.name,
            )
            for stream in corpus.task_streams
        ]
        wanted: List[Tuple[Any, int, str]] = []
        for source, target_tokens, name in targets:
            target_instances = target_tokens // spec.sequence_length
            if source.num_instances < target_instances:
                raise OLMoConfigurationError(
                    f"the '{name}' slice wants {target_instances:,} instances "
                    f"({target_tokens:,} tokens) but its source holds only "
                    f"{source.num_instances:,}. The mixer would rescale every slice down to keep the "
                    f"ratios, leaving each cell with a different absolute volume than its config "
                    f"states -- which is the one thing the mixture rule exists to prevent."
                )
            if target_instances < 1:
                raise OLMoConfigurationError(
                    f"the '{name}' slice wants {target_tokens:,} tokens, under one instance of "
                    f"{spec.sequence_length}. Raise the slice or lower the sequence length."
                )
            wanted.append((source, target_instances, name))

        total_instances = sum(count for _, count, _ in wanted)
        instance_source = MixingInstanceSource(
            *[
                MixingInstanceSourceSpec(source=source, ratio=count / total_instances, label=name)
                for source, count, name in wanted
            ],
            num_instances=total_instances,
            seed=spec.seed + 6,
            work_dir=work_dir,
            label=f"mixture:{spec.cell_id}",
        )
        log.info(
            "mixture for '%s': %s",
            spec.cell_id,
            ", ".join(
                f"{name} {count:,} instances ({100 * count / total_instances:.1f}%)"
                for _, count, name in wanted
            ),
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
            "wandb",
            WandBCallback(
                name=args.run_name,
                # EDULLM_WANDB_PROJECT, not WANDB_PROJECT: the platform sets both, and the reference
                # script reads the former. No `group` -- the platform puts the experiment in
                # WANDB_RUN_GROUP and the client reads that itself.
                project=os.environ.get("EDULLM_WANDB_PROJECT"),
                cancel_check_interval=10,
                # Enabled only when the platform named a project, so a local run does not fail on a
                # missing WANDB_API_KEY. Without this callback a nine-hour cell reports nothing until
                # it finishes, which is the difference between noticing a diverged run at step 500 and
                # at the end.
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
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
        global_batch_size=spec.global_batch_size,
        seed=spec.seed + 3,
        work_dir=work_dir,
        # The corpus is generated on demand rather than read from disk, so the loader's default of zero
        # workers puts that generation in the training process. The reasoning tasks are the slow ones --
        # 0.59M tok/s for mano and 0.39M for compare against the bio renderer's 10.1M, since each item
        # costs a mix and a scatter rather than a slice of a precompiled skeleton. A control is 100%
        # reasoning and the smallest demand cells are around two-thirds, so those runs would be
        # data-bound on a GPU. Workers cost nothing here.
        num_workers=args.num_workers,
    )

    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)
    data_loader = data_loader_config.build(
        instance_source,
        tokenizer=tokenizer_config,
        dp_process_group=train_module.dp_process_group,
    )
    trainer = trainer_config.build(train_module, data_loader)

    # Record the cell alongside every checkpoint. This is what makes a checkpoint scoreable later: the
    # corpus is generated rather than stored, so the only way to rebuild the exact vocabulary, entity
    # table and reasoning items a checkpoint was trained on is to replay the cell that produced it. The
    # fingerprints let a scorer prove it rebuilt the right one instead of assuming. Without this a
    # finished run is a directory of weights nobody can attach to a demand.
    #
    # Set after the trainer is built and every callback attached, which is what the callback's own
    # docstring requires -- it forwards the config to W&B and the others at assignment time.
    trainer.callbacks["config_saver"].config = {
        "factcrowd": {
            "cell": spec.to_dict(),
            "resolved": resolved.summary(
                0.0 if corpus.renderer is None else corpus.renderer.mean_tokens_per_bio
            ),
            "fingerprints": {
                "schema": corpus.corpus_schema.schema.fingerprint(),
                "vocabulary": corpus.vocabulary.fingerprint(),
                "stream": None if corpus.stream is None else corpus.stream.fingerprint(),
                "reasoning": {
                    stream.task.name: stream.task.fingerprint() for stream in corpus.task_streams
                },
            },
            "checkpoint_steps": checkpoint_steps,
        }
    }
    return trainer


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
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers. The corpus is generated in-process, so zero makes a reasoning-heavy "
        "cell data-bound.",
    )
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

    # Checked before anything expensive: an unexpanded variable is a submission defect, and finding
    # it after the corpus is built wastes the queue slot it took to get here.
    args.run_name = resolve_platform_value("run_name", args.run_name)
    args.save_folder = resolve_platform_value("--save-folder", args.save_folder)

    spec = resolve_cell(args)
    resolved = spec.resolve()
    work_dir = Path(args.work_dir) / spec.cell_id
    work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = work_dir

    corpus = BuiltCorpus(resolved, work_dir)
    # Re-resolved with the vocabulary, which only the corpus knows. The entity count and the demand are
    # unchanged -- the vocabulary feeds the *total*-parameter basis alone, which is otherwise a copy of
    # the non-embedding one and makes 3's dual-basis reporting one basis printed twice.
    resolved = spec.resolve(vocab_size=corpus.vocabulary.padded_size())
    summary = corpus.summary(resolved)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        width = max(len(key) for key in summary)
        print(f"factcrowd cell '{spec.cell_id}' -- run {args.run_name}")
        for key, value in summary.items():
            rendered = f"{value:,}" if isinstance(value, int) else str(value)
            print(f"  {key.rjust(width)}  {rendered}")
        if corpus.renderer is not None:
            print(f"  {'sample'.rjust(width)}  {corpus.renderer.text(0, 0)[:160]}")
        for stream in corpus.task_streams:
            sample = corpus.vocabulary.decode(stream.task.item(0).tokens)
            print(f"  {stream.task.name.rjust(width)}  {' '.join(sample)[:160]}")

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
