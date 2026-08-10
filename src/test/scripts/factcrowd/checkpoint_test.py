"""
Opening a checkpoint, and consolidating what was measured on it.

The corpus is generated rather than stored, so a checkpoint is only scoreable if the run recorded how to
rebuild it. These tests cover the record, the rebuild, and the fingerprint check that stands between a
wrong rebuild and a plausible-looking score -- a guard that has already fired for real, on a checkpoint
written before the union-vocabulary change, and correctly refused to score it.
"""

import json
from pathlib import Path

import pytest
from factcrowd import cells as C
from factcrowd.measure import checkpoint as CK
from factcrowd.measure import collect
from factcrowd.measure import collect as CO
from factcrowd.measure.bits import achieved_bits
from factcrowd.measure.collect import ScoredCheckpoint
from factcrowd.measure.endpoints import EndpointResult

from olmo_core.exceptions import OLMoConfigurationError

CONFIG_ROOT = Path("src/scripts/train/factcrowd/configs/cells")


def fake_checkpoint(root: Path, step: int, cell: C.CellSpec, *, fingerprints=None, complete=True):
    """
    Write the file shape `Checkpointer.find_checkpoints` validates, so discovery can be tested without
    training. `complete=False` omits one required file, which is what an interrupted upload looks like.
    """
    step_dir = root / f"step{step}"
    (step_dir / "model_and_optim").mkdir(parents=True, exist_ok=True)
    (step_dir / "train").mkdir(parents=True, exist_ok=True)
    (step_dir / "model_and_optim" / ".metadata").write_bytes(b"")
    (step_dir / "train" / "rank0.pt").write_bytes(b"")
    if complete:
        (step_dir / ".metadata.json").write_text("{}")
    record = {
        "cell": cell.to_dict(),
        "resolved": {},
        "fingerprints": fingerprints or {},
        "checkpoint_steps": [step],
    }
    (step_dir / "config.json").write_text(json.dumps({CK.RECORD_KEY: record}))
    return step_dir


def smoke_cell():
    return C.load_cell(CONFIG_ROOT / "smoke" / "smoke_13m_ctrl.yaml")


# --- discovery and the record ------------------------------------------------------------------------


def test_discovery_finds_complete_checkpoints_in_step_order_and_skips_torn_ones(tmp_path):
    """
    A checkpoint still uploading when the job died is not a checkpoint.

    Globbing `step*` cannot tell the difference; `Checkpointer.find_checkpoints` validates that all three
    required files exist, which is why discovery delegates to it rather than listing directories.
    """
    cell = smoke_cell()
    for step in (10, 2, 100):
        fake_checkpoint(tmp_path, step, cell)
    fake_checkpoint(tmp_path, 55, cell, complete=False)

    refs = CK.find_checkpoints(str(tmp_path))
    assert [ref.step for ref in refs] == [2, 10, 100]
    assert refs[0].model_dir.endswith("step2/model_and_optim")


def test_a_checkpoint_without_a_cell_record_cannot_be_scored(tmp_path):
    """
    The corpus is generated, so without the cell there is nothing to rebuild -- and the refusal says so
    rather than failing later with a missing key.
    """
    step_dir = tmp_path / "step1"
    step_dir.mkdir()
    (step_dir / "config.json").write_text(json.dumps({"model": {}}))
    with pytest.raises(OLMoConfigurationError, match="has no 'factcrowd' block"):
        CK.read_record(str(step_dir))

    with pytest.raises(OLMoConfigurationError, match="no config.json"):
        CK.read_record(str(tmp_path / "nope"))


def test_the_record_round_trips_into_the_cell_that_wrote_it(tmp_path):
    cell = smoke_cell()
    fake_checkpoint(tmp_path, 7, cell)
    record = CK.read_record(str(tmp_path / "step7"))
    assert C.CellSpec.from_dict(dict(record["cell"])) == cell


# --- the fingerprint guard ---------------------------------------------------------------------------


def test_a_corpus_that_rebuilds_differently_is_refused(tmp_path):
    """
    The guard that already earned its place.

    A checkpoint written before the entropy axis moved to a union vocabulary rebuilds to a different
    schema today. Scoring it would have produced entirely reasonable-looking numbers about a corpus the
    model never saw. The refusal names which fingerprint disagreed.
    """
    cell = smoke_cell()
    resolved = cell.resolve()
    corpus = CK.BuiltCorpus(resolved, tmp_path / "wd", split="eval", with_streams=False)

    CK.verify_fingerprints(corpus, {"fingerprints": {}})  # nothing recorded, nothing to contradict
    CK.verify_fingerprints(
        corpus,
        {"fingerprints": {"schema": corpus.corpus_schema.schema.fingerprint()}},
    )
    with pytest.raises(OLMoConfigurationError, match="rebuilt schema does not match"):
        CK.verify_fingerprints(corpus, {"fingerprints": {"schema": "deadbeef" * 8}})
    with pytest.raises(OLMoConfigurationError, match="rebuilt vocabulary does not match"):
        CK.verify_fingerprints(corpus, {"fingerprints": {"vocabulary": "deadbeef" * 8}})


def test_the_rebuild_is_on_the_eval_split_and_skips_the_packed_streams(tmp_path):
    """
    Measurement must not score the split the model trained on, and must not spend an offset index over a
    billion tokens to ask thirty thousand questions.
    """
    cell = smoke_cell()
    corpus = CK.BuiltCorpus(cell.resolve(), tmp_path / "wd", split="eval", with_streams=False)
    assert corpus.split == "eval"
    assert corpus.task_streams == ()
    assert corpus.tasks and all(task.split == "eval" for task in corpus.tasks)


# --- consolidation -----------------------------------------------------------------------------------


def scored(step, cell, endpoints=(), achieved=None, recall=None):
    return CO.ScoredCheckpoint(
        ref=CK.CheckpointRef(step=step, path=f"/tmp/step{step}"),
        cell=cell.to_dict(),
        endpoints=endpoints,
        achieved=achieved,
        recall=recall or {},
    )


def endpoint(name, accuracy):
    return EndpointResult(
        name=name,
        n_total=100,
        n_correct=int(accuracy * 100),
        n_degenerate=5,
        n_unparseable=0,
        answer_ce_bits=3.0,
        floor=0.05,
    )


def test_the_table_is_long_so_a_new_endpoint_needs_no_schema_change():
    """
    One row per (checkpoint, endpoint) rather than one column per endpoint.

    PRD 8.3 names two endpoints that are not built yet. Wide would need a migration for each; long does
    not.
    """
    cell = smoke_cell()
    rows = CO.collect(
        [scored(5, cell, endpoints=(endpoint("mano", 0.3), endpoint("compare", 0.6)))]
    )
    assert len(rows) == 2
    assert {row["endpoint"] for row in rows} == {"mano", "compare"}
    assert all(row["step"] == 5 and row["cell_id"] == cell.cell_id for row in rows)


def test_a_checkpoint_with_no_endpoints_still_emits_its_bit_row():
    """The bit curve is collectable before the reasoning half is."""
    cell = smoke_cell()
    a = achieved_bits(
        [20.0], n_entities_total=100, prior_bits_per_entity=47.59, non_embedding_params=12_595_456
    )
    rows = CO.collect([scored(3, cell, achieved=a)])
    assert len(rows) == 1
    assert rows[0]["stored_bits_per_entity"] > 0
    assert "endpoint" not in rows[0]


def test_rows_sort_by_cell_then_replicate_then_step_then_endpoint():
    """A trend has to read in order, and replicate is an axis -- leaving it implicit analyses a paired
    design as an unpaired one."""
    cell = smoke_cell()
    rows = CO.collect(
        [
            scored(20, cell, endpoints=(endpoint("mano", 0.1),)),
            scored(3, cell, endpoints=(endpoint("mano", 0.2),)),
            scored(3, cell, endpoints=(endpoint("compare", 0.3),)),
        ]
    )
    assert [(row["step"], row["endpoint"]) for row in rows] == [
        (3, "compare"),
        (3, "mano"),
        (20, "mano"),
    ]


def test_the_csv_puts_identity_first_and_leaves_missing_measurements_blank(tmp_path):
    """
    A missing measurement must be an empty cell, not a column shift -- otherwise one checkpoint that
    skipped the bit count silently misaligns every row after it.
    """
    cell = smoke_cell()
    a = achieved_bits(
        [20.0], n_entities_total=100, prior_bits_per_entity=47.59, non_embedding_params=12_595_456
    )
    rows = CO.collect(
        [scored(1, cell, endpoints=(endpoint("mano", 0.1),)), scored(2, cell, achieved=a)]
    )
    path = CO.write_csv(rows, tmp_path / "out" / "scores.csv")
    header = path.read_text().splitlines()[0].split(",")
    # The identity columns that are *present* come first, in declared order. `confirmatory` and
    # `admission` are added by score_run rather than by collect, so they are absent here -- and the
    # ordering has to survive a missing column rather than shifting everything after it.
    present = [column for column in CO.IDENTITY_COLUMNS if column in header]
    assert header[: len(present)] == present

    import csv

    back = list(csv.DictReader(path.open()))
    assert len(back) == 2
    assert back[0]["stored_bits_per_entity"] == ""  # step 1 had no bit count
    assert back[1]["endpoint"] == ""  # step 2 had no endpoints
    assert all(len(row) == len(header) for row in back)


def test_writing_an_empty_table_is_refused():
    """
    An empty table is nearly always a prefix that pointed at nothing, and a header-only file hides it.
    """
    with pytest.raises(OLMoConfigurationError, match="no rows to write"):
        CO.write_csv([], Path("/tmp/never-written.csv"))


def test_a_value_containing_a_comma_survives_the_round_trip():
    """
    The csv module quotes; a hand-rolled join would not.

    Cell notes and checkpoint paths are free text, and one comma in either would shift every column to its
    right for that row only -- which is the kind of corruption that shows up as an outlier rather than as
    an error.
    """
    import csv
    import tempfile

    cell = smoke_cell()
    item = scored(1, cell, endpoints=(endpoint("mano", 0.4),))
    item.ref = CK.CheckpointRef(step=1, path='/tmp/a,b/"quoted"/step1')
    rows = CO.collect([item])
    with tempfile.TemporaryDirectory() as raw:
        path = CO.write_csv(rows, Path(raw) / "scores.csv")
        with path.open() as handle:
            back = list(csv.DictReader(handle))
    assert len(back) == 1
    assert back[0]["checkpoint_path"] == '/tmp/a,b/"quoted"/step1'
    assert back[0]["endpoint"] == "mano"


def test_load_rebuilds_on_the_eval_split_and_verifies_what_it_rebuilt(tmp_path):
    """
    Through `load` itself, not through `BuiltCorpus` directly.

    An adversarial pass changed `load`'s `split="eval"` to `"train"` and the whole suite passed, because
    every split test built the corpus directly and never exercised the loader. Scoring the training split
    measures memorisation and would look like a strong result.

    `with_model=False` keeps this fast: the rebuild and the verification are what matter here, and loading
    weights is covered by the end-to-end test.
    """
    cell = smoke_cell()
    resolved = cell.resolve()
    reference = CK.BuiltCorpus(resolved, tmp_path / "ref", split="eval", with_streams=False)
    fake_checkpoint(
        tmp_path,
        11,
        cell,
        fingerprints={
            "schema": reference.corpus_schema.schema.fingerprint(),
            "vocabulary": reference.vocabulary.fingerprint(),
            "reasoning_structure": {
                task.name: task.structure_fingerprint() for task in reference.tasks
            },
        },
    )

    loaded = CK.load(
        CK.CheckpointRef(step=11, path=str(tmp_path / "step11")),
        work_dir=tmp_path / "wd",
        with_model=False,
    )
    assert loaded.corpus.split == "eval"
    assert loaded.corpus.tasks and all(task.split == "eval" for task in loaded.corpus.tasks)
    assert loaded.cell == cell
    # The resolved cell knows the padded vocabulary, so both parameter bases are real rather than equal.
    assert loaded.resolved.total_params > loaded.resolved.non_embedding_params


def test_load_refuses_a_checkpoint_whose_endpoint_shape_changed(tmp_path):
    """
    The structural task digest, which is the one a split-baked fingerprint could not provide.

    A changed expression length passes the schema and vocabulary digests untouched while altering every
    item scored -- demonstrated by an adversarial pass with MANO_LENGTH=13.
    """
    cell = smoke_cell()
    reference = CK.BuiltCorpus(cell.resolve(), tmp_path / "ref", split="eval", with_streams=False)
    fake_checkpoint(
        tmp_path,
        12,
        cell,
        fingerprints={
            "schema": reference.corpus_schema.schema.fingerprint(),
            "vocabulary": reference.vocabulary.fingerprint(),
            "reasoning_structure": {task.name: "deadbeef" * 8 for task in reference.tasks},
        },
    )
    with pytest.raises(OLMoConfigurationError, match="item shape has changed"):
        CK.load(
            CK.CheckpointRef(step=12, path=str(tmp_path / "step12")),
            work_dir=tmp_path / "wd",
            with_model=False,
        )


def test_load_refuses_a_checkpoint_that_carried_an_endpoint_the_rebuild_does_not(tmp_path):
    """An endpoint that has vanished cannot be scored, and its absence may make the others incomparable."""
    cell = smoke_cell()
    reference = CK.BuiltCorpus(cell.resolve(), tmp_path / "ref", split="eval", with_streams=False)
    fake_checkpoint(
        tmp_path,
        13,
        cell,
        fingerprints={
            "schema": reference.corpus_schema.schema.fingerprint(),
            "reasoning_structure": {
                **{task.name: task.structure_fingerprint() for task in reference.tasks},
                "brevo": "f" * 64,
            },
        },
    )
    with pytest.raises(OLMoConfigurationError, match="the rebuild does not carry"):
        CK.load(
            CK.CheckpointRef(step=13, path=str(tmp_path / "step13")),
            work_dir=tmp_path / "wd",
            with_model=False,
        )


def test_scoring_a_cell_builds_its_corpus_once_and_never_its_fact_stream():
    """
    One switch that covers the expensive half, which it did not.

    `with_streams=False` still built the fact stream and its token-offset index -- 4.7s against 0.3s on a
    127k-entity cell, paid on every one of ten checkpoints, because scoring also used a fresh work
    directory per step. Scoring reads `tasks` and `renderer`, built either way; the renderer alone is
    enough for bits and template reconstruction.
    """
    import tempfile
    from pathlib import Path

    from factcrowd import cells as C
    from factcrowd.corpus.build import BuiltCorpus

    cell = C.load_cell("src/scripts/train/factcrowd/configs/cells/smoke/smoke_13m_reason.yaml")
    resolved = cell.resolve()
    with tempfile.TemporaryDirectory() as raw:
        scoring = BuiltCorpus(resolved, Path(raw), split="eval", with_streams=False)
        assert scoring.stream is None and scoring.task_streams == ()
        assert scoring.renderer is not None  # enough for bits and template reconstruction
        # Training still gets both, so the default is unchanged where it matters.
        training = BuiltCorpus(resolved, Path(raw) / "t", split="train")
        assert training.stream is not None and training.task_streams


def test_a_corpus_may_only_be_reused_for_the_cell_it_was_built_for(tmp_path):
    """
    Reuse is checked, not assumed.

    Scoring reuses one corpus across a cell's checkpoints because only the weights differ. Handing it the
    wrong cell's corpus would score every checkpoint against a corpus the model never saw -- and would
    look entirely reasonable.
    """
    from factcrowd import cells as C
    from factcrowd.corpus.build import BuiltCorpus

    cell = smoke_cell()
    other = C.load_cell("src/scripts/train/factcrowd/configs/cells/smoke/smoke_13m_reason.yaml")
    reference = BuiltCorpus(cell.resolve(), tmp_path / "ref", split="eval", with_streams=False)
    fake_checkpoint(
        tmp_path,
        21,
        cell,
        fingerprints={"schema": reference.corpus_schema.schema.fingerprint()},
    )
    wrong = BuiltCorpus(other.resolve(), tmp_path / "wrong", split="eval", with_streams=False)
    with pytest.raises(OLMoConfigurationError, match="was offered for a checkpoint of"):
        CK.load(
            CK.CheckpointRef(step=21, path=str(tmp_path / "step21")),
            work_dir=tmp_path / "wd",
            with_model=False,
            corpus=wrong,
        )


# --- completeness and double counting ---------------------------------------------------------------


def _cp(cell_id, replicate, step, planned, endpoints=("mano",), run="a"):
    from factcrowd.measure.checkpoint import CheckpointRef
    from factcrowd.measure.endpoints import EndpointResult

    return ScoredCheckpoint(
        ref=CheckpointRef(step=step, path=f"/p/{run}/{cell_id}/checkpoints/step{step}"),
        cell={"cell_id": cell_id, "row": "13M", "replicate": replicate},
        endpoints=tuple(
            EndpointResult(
                name=n,
                n_total=100,
                n_correct=5,
                n_degenerate=0,
                n_unparseable=0,
                answer_ce_bits=1.0,
                floor=0.05,
            )
            for n in endpoints
        ),
        extra={"checkpoint_steps": list(planned), "achieved_bits_per_param": 1.23},
    )


def test_a_crashed_run_and_its_rerun_are_one_cell_and_the_partial_one_loses():
    """
    `13m_d0p6` died at 3,441 and was re-run to 10,732. Both wrote checkpoints, both sit under prefixes the
    scorer is pointed at, and without this the cell appears twice with different numbers -- one of them a
    partially-trained model in the same column as fully-trained ones.

    `--last-only` does not solve it: it means "highest checkpoint in *this* prefix", and the crashed
    prefix's highest is the one before it died.
    """
    planned = [53, 107, 214, 429, 858, 1717, 3434, 5366, 8049, 10732]
    crashed = [_cp("13m_d0p6", 0, s, planned, run="crash") for s in (1717, 3434)]
    finished = [_cp("13m_d0p6", 0, s, planned, run="rerun") for s in (1717, 3434, 10732)]
    kept, notes = collect.select_complete(crashed + finished)
    # The whole trajectory of the finished run, and nothing from the crashed one.
    assert [c.ref.step for c in kept] == [1717, 3434, 10732]
    assert all("rerun" in str(c.ref.path) for c in kept)
    assert any("2 runs found" in n for n in notes)


def test_a_cell_that_never_reached_its_planned_final_step_is_dropped_and_named():
    """
    Completion is judged against the cell's *own* recorded plan, because each cell has a different one.
    Dropped rather than flagged: a partially-trained model in a confirmatory table is not a weaker
    measurement of the same thing, it is a measurement of something else.
    """
    planned = [53, 107, 3434, 10732]
    kept, notes = collect.select_complete([_cp("13m_d0p6", 0, s, planned) for s in (53, 107, 3434)])
    assert kept == []
    assert any("no run of it finished" in n and "3,434" in n and "10,732" in n for n in notes)


def test_replicates_of_one_cell_are_kept_apart():
    """The inferential unit is the replicate, so two of them are two rows and not a duplicate."""
    planned = [19, 3814]
    kept, _ = collect.select_complete(
        [_cp("13m_ctrl", 0, 3814, planned), _cp("13m_ctrl", 1, 3814, planned)]
    )
    assert sorted(c.stated("replicate") for c in kept) == [0, 1]


def test_the_storage_fields_are_flagged_on_one_row_per_checkpoint():
    """
    Storage is per checkpoint; the table is per endpoint. So a two-endpoint cell repeats its
    `achieved_bits_per_param`, and averaging that column over rows double-weights exactly the
    fact-bearing count cells and none of the controls -- a systematic bias, not noise.
    """
    rows = _cp("13m_d1p2", 0, 100, [100], endpoints=("mano", "compare")).rows()
    assert len(rows) == 2
    assert sum(1 for r in rows if r["storage_row"]) == 1
    assert len({r["achieved_bits_per_param"] for r in rows}) == 1  # repeated, hence the flag
