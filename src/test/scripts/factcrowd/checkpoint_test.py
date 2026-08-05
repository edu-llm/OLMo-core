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
from factcrowd.measure import collect as CO
from factcrowd.measure.bits import achieved_bits
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
    assert header[: len(CO.IDENTITY_COLUMNS)] == list(CO.IDENTITY_COLUMNS)

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
