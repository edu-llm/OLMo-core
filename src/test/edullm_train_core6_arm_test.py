"""What ``.edullm/train_core6_arm.py`` computes for its held-out endpoint, and what it refuses.

The subject here is ``evaluate_val_aggregate`` and the accounting around it. That function is
the experiment's endpoint -- every contrast the study reports is a difference of it between two
arms -- and the version it replaced could not produce a number at all: it was gated on
``get_rank() == 0`` inside a swallowing ``except``, which under FSDP is a hang rather than an
exception, so the run exited 0 with a null field and a checkpoint nobody could use.

WHAT IS AND IS NOT TESTABLE ON A LAPTOP. The multi-rank behaviour is not: there is no GPU here
and no process group. What IS testable, and is what these tests cover, is everything that
decides whether the number is right once the collectives are correct -- the token accounting,
the refusals, the window arithmetic, and the fact that a rank-count of one is not a special
case in the code. The all-reduce path itself goes through ``all_reduce_value``, which is a
documented no-op off a process group, so the single-rank runs below execute the same lines an
eight-rank run does.

The entry point is not importable as a package -- it lives in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path, the same way
``edullm_train_on_corpus_test.py`` loads its subject.
"""

import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_core6_arm.py"
    spec = importlib.util.spec_from_file_location("edullm_train_core6_arm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()

#: Shard sizes chosen so the window arithmetic is OBSERVABLE, which required getting the
#: reasoning the right way round on the second attempt.
#:
#: The count is ``(n - 1) // seq_len`` -- the minus one is the target shift, since the last
#: window needs one token past its inputs. The wrong version, ``n // seq_len``, differs from it
#: ONLY when ``n`` is an exact multiple of ``seq_len``: at n=96, seq=32 the right answer is 2
#: and the wrong one is 3; at n=100 both say 3. An earlier version of this fixture was all
#: non-multiples, on the stated theory that a clean division "hides every off-by-one" -- which
#: is exactly backwards for this off-by-one, and the fixture was blind to it. ``96`` is in the
#: list for that reason and must stay.
SHARD_TOKENS = (100, 96, 71, 33)
SEQ_LEN = 32
VOCAB = 512


class TinyLM(torch.nn.Module):
    """A model small enough to run on a laptop that still produces real logits.

    Untrained, so its cross-entropy sits near ``ln(vocab)`` -- which is the magnitude assertion
    below, and the only check in this project's history that ever caught uninitialised weights.
    """

    def __init__(self, vocab: int = VOCAB, d: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.out = torch.nn.Linear(d, vocab)

    def forward(self, x):
        return self.out(self.embed(x))


def write_shard(path, n_tokens: int, *, start: int = 0) -> str:
    """A headerless little-endian uint32 shard holding ``start..start+n`` as its tokens.

    Consecutive ids rather than random ones: a test that needs to know WHICH tokens were scored
    can then read the answer off the values, which is what the window and shift tests below do.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.arange(start, start + n_tokens, dtype=np.uint32).astype("<u4").tofile(path)
    return str(path)


@pytest.fixture
def shards(tmp_path) -> List[str]:
    """Four little-endian uint32 shards, under two different topic directories.

    THE TOPIC DIRECTORY IS PART OF THE FIXTURE, NOT SCENERY. In olmo-150b-dolma2 a val shard
    named ``val-00212.u32le.bin`` exists under 24 different topics, so a basename is not a
    unique key. Two of the shards below share a basename for exactly that reason.
    """
    made = []
    for i, n in enumerate(SHARD_TOKENS):
        topic = "art_and_design" if i < 2 else "mathematics"
        # Indices 0 and 2 share a basename across two topics; the others differ.
        name = "val-00212.u32le.bin" if i % 2 == 0 else f"val-{213 + i:05}.u32le.bin"
        made.append(write_shard(tmp_path / "corpus" / topic / name, n))
    return made


#: "Caller did not say", as distinct from ``None``, which is a meaningful value here -- it is
#: the manifest declaring no row count, and the case one test below is entirely about.
_UNSET = object()


def evaluate(shards, tmp_path, *, declared=_UNSET, seq_len=SEQ_LEN, micro=2, model=None):
    return entry.evaluate_val_aggregate(
        model=model or TinyLM(),
        vocab_size=VOCAB,
        val_paths=list(shards),
        work_dir=str(tmp_path / f"wd-{len(os.listdir(tmp_path))}"),
        seq_len=seq_len,
        dtype=np.dtype("<u4"),
        declared_tokens=sum(SHARD_TOKENS) if declared is _UNSET else declared,
        micro=micro,
    )


# --- the endpoint produces a number, and it is the right size -------------------------------


def test_the_endpoint_produces_a_cross_entropy_near_ln_vocab(shards, tmp_path):
    """
    A MAGNITUDE CHECK, NOT AN EXISTENCE ONE. ``result["ce"] is not None`` passes for a model
    whose weights were never initialised, for a CE computed over the wrong tokens, and for a
    zero. An untrained model over any token set has to score near ``ln(vocab)`` -- that is what
    "uniform over the vocabulary" means -- and this project's documented history is five green
    harness results where the only check that caught anything was this one.
    """
    result = evaluate(shards, tmp_path)
    assert abs(result["ce"] - math.log(VOCAB)) < 1.0, (
        f"CE {result['ce']:.4f} is nowhere near ln({VOCAB}) = {math.log(VOCAB):.4f}; an "
        "untrained model cannot score this"
    )


def test_it_returns_sums_and_counts_and_not_only_a_mean(shards, tmp_path):
    """
    Two arms get differenced. A difference of two means computed over different token counts is
    not a paired difference, so the denominator has to ship with the number rather than be
    assumed equal between runs.
    """
    result = evaluate(shards, tmp_path)
    assert result["sum"] > 0 and result["tokens"] > 0
    assert result["ce"] == pytest.approx(result["sum"] / result["tokens"])


def test_every_token_is_accounted_for(shards, tmp_path):
    """
    ``present == declared`` exactly, and ``scored`` short of it by only the window remainder.

    THE EXPECTED SCORED COUNT IS A LITERAL, NOT A FORMULA. An earlier version of this test
    computed ``sum(((n - 1) // SEQ_LEN) * SEQ_LEN for n in SHARD_TOKENS)`` -- which is a
    reimplementation of the exact line under test, so it moved whenever the code moved and
    passed against a broken window count. The numbers below are worked out by hand from the
    fixture and written down:

        n=100 -> (100-1)//32 = 3 windows -> 96 scored, 4 unscored
        n= 96 -> ( 96-1)//32 = 2 windows -> 64 scored, 32 unscored   <-- the multiple-of-32 case
        n= 71 -> ( 71-1)//32 = 2 windows -> 64 scored, 7 unscored
        n= 33 -> ( 33-1)//32 = 1 window  -> 32 scored, 1 unscored
                                            --------- ------------
                                            256        44   = 300 present
    """
    result = evaluate(shards, tmp_path)

    assert sum(SHARD_TOKENS) == 300, "the fixture changed; rework the arithmetic above"
    assert result["tokens"] == 256
    assert result["tokens_present"] == 300
    assert result["declared_tokens"] == 300
    assert result["unscored"] == 44
    assert result["shards"] == len(SHARD_TOKENS)

    entry.assert_val_tokens_account_for_the_corpus(result)


# --- the window arithmetic, the target shift, and the mask alignment --------------------------
#
# These pin the three off-by-ones that produce a plausible number from the wrong tokens. Before
# they existed, mutating `(n-1)//seq_len` to `n//seq_len`, or making the targets equal the
# inputs, left the whole suite green -- the second of those scores the model on predicting the
# token it just read, which makes the CE LOOK BETTER and means nothing.


def test_a_shard_that_is_an_exact_multiple_of_the_sequence_length_loses_its_last_window():
    """
    THE OFF-BY-ONE THAT ONLY SHOWS UP ON A CLEAN DIVISION. A window needs ``seq_len`` inputs
    AND one more token to be the last target, so 96 tokens at seq_len 32 is 2 windows, not 3.
    ``n // seq_len`` gives 3 and would read one token past the end of the shard.
    """
    assert entry._shard_microbatch_count is not None  # imported name, not a typo'd attribute
    for n_tokens, expected in ((96, 2), (97, 3), (95, 2), (32, 0), (33, 1), (64, 1), (65, 2)):
        windows = [
            offsets
            for offsets, _, _ in entry._shard_windows(
                _one_shard(n_tokens), seq_len=32, micro=1, dtype=np.dtype("<u4")
            )
        ]
        assert len(windows) == expected, f"{n_tokens} tokens gave {len(windows)} windows"


_SHARD_CACHE: Dict[int, str] = {}


def _one_shard(n_tokens: int) -> str:
    """A throwaway shard of ``n_tokens`` consecutive ids, cached per size within a session."""
    import tempfile

    if n_tokens not in _SHARD_CACHE:
        directory = Path(tempfile.mkdtemp())
        _SHARD_CACHE[n_tokens] = write_shard(directory / f"s{n_tokens}.u32le.bin", n_tokens)
    return _SHARD_CACHE[n_tokens]


def test_the_targets_are_the_inputs_shifted_by_exactly_one():
    """
    THE SHIFT, ASSERTED ON VALUES RATHER THAN SHAPES. The shard holds consecutive ids, so the
    correct answer is readable: window ``w`` has inputs ``[w*S .. w*S+S-1]`` and targets
    ``[w*S+1 .. w*S+S]``. Targets equal to inputs -- the model predicting the token it just read
    -- has the same shape, produces a LOWER cross-entropy, and is undetectable from the number.
    """
    seq_len = 8
    batches = list(
        entry._shard_windows(_one_shard(40), seq_len=seq_len, micro=2, dtype=np.dtype("<u4"))
    )
    offsets = [off for batch in batches for off in batch[0]]
    assert offsets == [0, 8, 16, 24]  # (40-1)//8 = 4 windows

    for batch_offsets, xs, ys in batches:
        for row, off in enumerate(batch_offsets):
            assert list(xs[row]) == list(range(off, off + seq_len)), "inputs are wrong"
            assert list(ys[row]) == list(range(off + 1, off + seq_len + 1)), "shift is wrong"
            # Stated the other way round too, since this is the whole property.
            assert list(ys[row][:-1]) == list(xs[row][1:])
            assert list(ys[row]) != list(xs[row])


def test_the_band_mask_lines_up_with_the_targets_and_not_the_inputs(tmp_path):
    """
    THE MASK ALIGNMENT IN ``evaluate_sliced``, which the module docstring warns "scores the
    wrong positions and still produces plausible numbers".

    Built so the answer is unambiguous: exactly one mask bit is set, at corpus position ``p``.
    The CE it selects must be the one for the TARGET at ``p`` -- and a target at position ``p``
    is predicted from the input at ``p-1``. Reading ``mask[off:off+seq]`` instead of
    ``mask[off+1:off+seq+1]`` selects the neighbouring position, which this catches.
    """
    seq_len, n_tokens, marked = 8, 40, 11
    shard = write_shard(tmp_path / "s.u32le.bin", n_tokens)
    mask = np.zeros(n_tokens, dtype=np.uint8)
    mask[marked] = 1  # band 0's bit
    mask_path = tmp_path / "s.mask.u8"
    mask.tofile(mask_path)

    # A model whose logits depend on the input token, so per-position CEs differ and selecting
    # the wrong position gives a different number rather than the same one by luck.
    torch.manual_seed(0)
    model = TinyLM()
    out = entry.evaluate_sliced(
        model=model,
        vocab_size=VOCAB,
        val_paths=[shard],
        mask_paths=[str(mask_path)],
        seq_len=seq_len,
        micro=2,
    )
    assert out["bands"]["0"]["n"] == 1, "exactly one position was marked"

    # The independent answer: CE of the target at `marked`, which is predicted from `marked-1`.
    with torch.no_grad():
        logits = model(torch.arange(n_tokens - 1, dtype=torch.int64).unsqueeze(0))[0]
        expected = float(
            torch.nn.functional.cross_entropy(
                logits[marked - 1].unsqueeze(0),
                torch.tensor([marked]),
                reduction="none",
            )
        )
    assert out["bands"]["0"]["sum"] == pytest.approx(expected, rel=1e-5), (
        "the mask selected a different position than the target it indexes"
    )


def test_two_shards_with_the_same_basename_are_both_counted(shards, tmp_path):
    """
    THE DOCUMENTED TRAP, AS A TEST. ``all-dressed-snazzy2__val-00212`` corresponds to
    ``all-dressed-snazzy2/art_and_design/val-00212.u32le.bin``: the topic directory is dropped
    from the name and 24 topics exist. Two of this fixture's shards share a basename, so a
    download that used the basename as its local filename would overwrite one with the other,
    silently drop a third of the val set, and produce a completely normal-looking CE.

    The token count is what catches it, which is the whole reason the count is asserted.
    """
    basenames = [os.path.basename(p) for p in shards]
    assert len(set(basenames)) < len(basenames), "the fixture must contain a repeated basename"

    result = evaluate(shards, tmp_path)
    assert result["tokens_present"] == sum(SHARD_TOKENS)
    entry.assert_val_tokens_account_for_the_corpus(result)


# --- the refusals, which are the point --------------------------------------------------------


def test_a_corpus_with_no_held_out_split_is_refused_rather_than_skipped(tmp_path):
    """
    Not ``return None``. A run that cannot produce the endpoint has produced a checkpoint that
    cannot answer the question, and reporting that as a null field beside exit 0 is what the
    whole of this function exists to stop.
    """
    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=[],
            work_dir=str(tmp_path),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=1000,
        )
    assert refusal.value.stage is entry.Stage.THE_CORPUS_DECLARES_NO_HELD_OUT_SPLIT
    assert int(refusal.value.stage) == 73


def test_a_missing_shard_is_caught_by_the_token_count(shards, tmp_path):
    """
    THE FAILURE THE COUNT ASSERTION IS FOR. Evaluating two of three shards produces a perfectly
    ordinary cross-entropy -- the model is the same, the tokens are real, the number is in
    range. Nothing about the CE says a third of the val set was never read. Only the count does.
    """
    partial = evaluate(shards[:2], tmp_path, declared=sum(SHARD_TOKENS))

    # The CE itself is unremarkable, which is precisely the problem being solved.
    assert abs(partial["ce"] - math.log(VOCAB)) < 1.0

    with pytest.raises(SystemExit) as refusal:
        entry.assert_val_tokens_account_for_the_corpus(partial)
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING
    assert "declares" in refusal.value.explanation


def test_the_unscored_slack_bound_is_tight_enough_to_catch_a_lost_shard():
    """
    THE BOUND'S VALUE, PINNED. ``present - scored`` is allowed up to ``n_shards * (seq_len + 1)``
    -- the tail of each shard that cannot fill a whole window. If that bound were loosened (or
    multiplied by a constant that "makes the flakiness go away"), the check would still exist,
    still be called, and catch nothing. Nothing else in this file asserts its size.

    Checked against the REAL geometry, because that is the only place the tightness matters:
    60 shards at seq_len 2048 gives a 122,940-token allowance against a 229,894,171-token
    partition -- 0.05% -- while ONE lost shard of a 60-shard corpus is ~3.8M tokens, 31x the
    whole allowance. The margin between those two numbers is what makes the assertion do work.
    """
    shards_n, seq_len, declared = 60, 2048, 229_894_171
    slack = shards_n * (seq_len + 1)
    assert slack == 122_940

    def check(scored):
        entry.assert_val_tokens_account_for_the_corpus(
            {
                "declared_tokens": declared,
                "tokens_present": declared,
                "tokens": scored,
                "shards": shards_n,
                "seq_len": seq_len,
            }
        )

    # A correct run: the realized window remainder sits inside the bound with room to spare.
    check(declared - 108_571)
    check(declared)  # a corpus that happens to window exactly
    check(declared - slack)  # the worst legitimate case, exactly on the boundary

    # One shard's worth of tokens missing from the scoring is 31x the allowance.
    for lost in (slack + 1, declared // shards_n, declared // 4):
        with pytest.raises(SystemExit):
            check(declared - lost)

    # And scoring MORE than was present is not "extra credit", it is a bug.
    with pytest.raises(SystemExit):
        check(declared + 1)


def test_a_corpus_that_declares_no_row_count_is_refused_rather_than_unchecked(shards, tmp_path):
    """
    An unchecked token count is how a CE over a quarter of the val set gets recorded as the
    endpoint. If there is nothing to check against, that is a refusal and not a pass.
    """
    result = evaluate(shards, tmp_path, declared=None)
    assert result["declared_tokens"] is None

    with pytest.raises(SystemExit) as refusal:
        entry.assert_val_tokens_account_for_the_corpus(result)
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING


def test_shards_too_short_to_fill_a_window_are_refused(tmp_path):
    """
    A sequence length larger than the shards yields zero windows, and zero windows is zero
    tokens. Returning a CE of 0.0 over 0 tokens would be a number, would serialise, and would
    be differenced against another arm as though it meant something.
    """
    directory = tmp_path / "tiny"
    directory.mkdir()
    path = directory / "val-00000.u32le.bin"
    np.arange(8, dtype=np.uint32).astype("<u4").tofile(path)

    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=[str(path)],
            work_dir=str(tmp_path / "wd"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=8,
        )
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING


def test_a_shard_that_cannot_be_read_fails_rather_than_being_skipped(shards, tmp_path):
    """
    A fetch failure must not become a shorter val set. It is raised -- and in the distributed
    case raised on every rank, so a single rank's bad download does not leave its peers waiting
    on a collective it will never enter.
    """
    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=list(shards) + [str(tmp_path / "does-not-exist.u32le.bin")],
            work_dir=str(tmp_path / "wd"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=sum(SHARD_TOKENS),
        )
    # Whichever stage read_failure assigned, it must be a refusal that names the rank rather
    # than a silently shorter evaluation.
    assert "held-out" in refusal.value.explanation


# --- every rank participates, run as actual ranks ---------------------------------------------
#
# WHY THREADS AND A FAKE COLLECTIVE RATHER THAN AN AST WALK. The first version of these tests
# parsed the function looking for `get_rank() == 0` and for an `except` with no `raise`. Both
# were theatre: a gate spelled `if rank == 0:` -- using the local the function already assigns
# -- passed, and a handler that merely assigned something passed. They checked the ONE spelling
# that had been removed.
#
# What follows runs `evaluate_val_aggregate` on N threads with `get_rank`, `get_world_size`,
# `barrier` and `all_reduce_value` monkeypatched to a real rendezvous. That executes the actual
# control flow of every rank and MEASURES what the AST could only guess at: whether every rank
# reaches every collective, whether the number is the union rather than one rank's share, and
# whether one rank failing takes the others down instead of leaving them waiting. It is not
# NCCL and it is not FSDP -- it cannot prove the all-gathers inside the forward line up -- but
# it does prove the participation structure this function is responsible for.


class FakeGroup:
    """A rendezvous standing in for a process group, with a deadlock detector.

    Every collective is a barrier with a timeout. A rank that never arrives -- because it
    returned early, or raised, or hit a different branch than its peers -- leaves the others
    waiting, and the timeout turns that into a test failure with the collective's name on it
    instead of a hung suite. That is the failure mode this whole rewrite exists to prevent, so
    the harness is built to report it rather than reproduce it.
    """

    def __init__(self, world_size: int, timeout: float = 10.0) -> None:
        import threading

        self.world_size = world_size
        self.timeout = timeout
        self.lock = threading.Condition()
        self.arrived: List = []
        self.generation = 0
        self.results: Dict[int, object] = {}
        #: Name of every collective in the order it was entered, per rank. Compared across ranks
        #: so "they all made N calls" is checked as "they all made THE SAME calls in order".
        self.trace: Dict[int, List[str]] = {}
        self.abandoned = False

    def rendezvous(self, rank: int, name: str, value=None, op=None):
        self.trace.setdefault(rank, []).append(name)
        with self.lock:
            generation = self.generation
            self.arrived.append((rank, value))
            if len(self.arrived) == self.world_size:
                values = [v for _, v in self.arrived]
                if op == "max":
                    self.results[generation] = max(values)
                elif op == "sum":
                    self.results[generation] = sum(values)
                else:
                    self.results[generation] = None
                self.arrived = []
                self.generation += 1
                self.lock.notify_all()
            else:
                if not self.lock.wait_for(
                    lambda: self.generation != generation or self.abandoned,
                    timeout=self.timeout,
                ):
                    self.abandoned = True
                    self.lock.notify_all()
                    raise AssertionError(
                        f"rank {rank} waited {self.timeout}s at collective {name!r} and not "
                        f"every rank arrived -- this is the hang the endpoint must not have"
                    )
                # ORDER MATTERS: a completed generation wins over `abandoned`. Once every rank
                # has arrived the collective SUCCEEDED, and a peer that abandons a moment later
                # (because the reduced value told it to raise, which is the correct behaviour)
                # must not retroactively turn this rank's completed call into a failure.
                if self.generation == generation and self.abandoned:
                    raise AssertionError(f"rank {rank}: another rank abandoned {name!r}")
            return self.results[generation]

    def abandon(self):
        """Called when a rank leaves early, so its peers fail fast instead of timing out."""
        with self.lock:
            self.abandoned = True
            self.lock.notify_all()


def run_on_ranks(monkeypatch, *, world_size: int, val_paths, tmp_path, declared, seq_len=SEQ_LEN):
    """Run the evaluator on ``world_size`` threads against a shared FakeGroup.

    Returns ``(results, errors)``, both indexed by rank.
    """
    import threading

    group = FakeGroup(world_size)
    local = threading.local()

    def fake_all_reduce(value, device, op=None, group_=None):
        name = "max" if op is torch.distributed.ReduceOp.MAX else "sum"
        return group.rendezvous(local.rank, f"all_reduce:{name}", value, op=name)

    monkeypatch.setattr(entry, "get_rank", lambda *a, **k: local.rank)
    monkeypatch.setattr(entry, "get_world_size", lambda *a, **k: world_size)
    monkeypatch.setattr(entry, "barrier", lambda *a, **k: group.rendezvous(local.rank, "barrier"))
    monkeypatch.setattr(entry, "all_reduce_value", fake_all_reduce)

    results: Dict[int, object] = {}
    errors: Dict[int, BaseException] = {}

    def body(rank: int):
        local.rank = rank
        try:
            results[rank] = entry.evaluate_val_aggregate(
                model=TinyLM(),
                vocab_size=VOCAB,
                val_paths=list(val_paths),
                work_dir=str(tmp_path / f"rank{rank}"),
                seq_len=seq_len,
                dtype=np.dtype("<u4"),
                declared_tokens=declared,
                micro=2,
            )
        except BaseException as exc:  # noqa: BLE001 -- reported per rank below
            errors[rank] = exc
            group.abandon()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "a rank never finished -- the endpoint hung"
    return results, errors, group


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_every_rank_reaches_every_collective(monkeypatch, tmp_path, world_size):
    """
    THE CORE PROPERTY. Four world sizes, chosen to cover the cases the shard assignment treats
    differently: one rank; a size that divides the shard count; a size that does NOT (so some
    ranks get fewer shards and must run filler passes); and a size LARGER than the shard count
    (so some ranks get NO shards at all and have no batch of their own to replay).

    The last of those is the one that would have hung. An earlier draft asserted "every rank has
    at least one object" and padded by replaying a rank's last real batch -- which a rank with
    zero objects does not have. 8 ranks over 4 shards is not exotic; it is the default GPU shape.
    """
    paths = [
        write_shard(tmp_path / "corpus" / f"t{i}" / f"val-{i:05}.u32le.bin", n)
        for i, n in enumerate(SHARD_TOKENS)
    ]
    results, errors, group = run_on_ranks(
        monkeypatch,
        world_size=world_size,
        val_paths=paths,
        tmp_path=tmp_path,
        declared=sum(SHARD_TOKENS),
    )
    assert not errors, f"ranks failed: { {r: repr(e) for r, e in errors.items()} }"
    assert set(results) == set(range(world_size)), "not every rank produced a result"

    # Every rank entered the SAME collectives in the SAME order. A rank that skipped one -- or
    # entered a different one -- is the hang, and this is what detects it rather than a count.
    traces = [group.trace[rank] for rank in range(world_size)]
    assert all(t == traces[0] for t in traces), f"ranks diverged: {traces}"
    assert traces[0].count("barrier") >= 2
    assert traces[0].count("all_reduce:sum") >= 3


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_the_number_is_the_whole_runs_and_not_one_ranks_share(
    monkeypatch, tmp_path, world_size
):
    """
    Every rank must come back with the SAME totals, and those totals must be the union over all
    shards -- not the local share, and not the same shard counted world_size times.

    This is what the rank-zero version could not do even if it had not hung: it would have
    reported whatever rank zero happened to hold.
    """
    paths = [
        write_shard(tmp_path / "corpus" / f"t{i}" / f"val-{i:05}.u32le.bin", n)
        for i, n in enumerate(SHARD_TOKENS)
    ]
    results, errors, _ = run_on_ranks(
        monkeypatch,
        world_size=world_size,
        val_paths=paths,
        tmp_path=tmp_path,
        declared=sum(SHARD_TOKENS),
    )
    assert not errors, f"ranks failed: { {r: repr(e) for r, e in errors.items()} }"

    for rank, result in results.items():
        assert result["tokens_present"] == 300, f"rank {rank} saw {result['tokens_present']}"
        assert result["tokens"] == 256, f"rank {rank} scored {result['tokens']}"
        entry.assert_val_tokens_account_for_the_corpus(result)

    # Identical on every rank, which is what makes it safe for rank zero alone to print it.
    first = results[0]
    for rank, result in results.items():
        assert result["tokens"] == first["tokens"]
        assert result["tokens_present"] == first["tokens_present"]
        assert result["ce"] == pytest.approx(first["ce"], rel=1e-9), f"rank {rank} disagrees"


def test_one_rank_failing_its_download_fails_every_rank_rather_than_hanging(
    monkeypatch, tmp_path
):
    """
    THE DEADLOCK CASE, EXERCISED. Rank 1's fetch raises. Without the all-reduced failure flag,
    rank 1 unwinds while ranks 0 and 2 walk into an all_reduce it will never enter -- and a hang
    at the end of a paid run is the worst available outcome, because the money is already spent
    and there is no error to read.

    The FakeGroup times out rather than hanging the suite, so a regression here shows up as a
    failure naming the collective.
    """
    paths = [
        write_shard(tmp_path / "corpus" / f"t{i}" / f"val-{i:05}.u32le.bin", n)
        for i, n in enumerate(SHARD_TOKENS)
    ]
    real_fetch = entry.fetch_val_shards

    def flaky(*, val_paths, work_dir, rank, world_size):
        if rank == 1:
            raise OSError("simulated: this rank's download died")
        return real_fetch(
            val_paths=val_paths, work_dir=work_dir, rank=rank, world_size=world_size
        )

    monkeypatch.setattr(entry, "fetch_val_shards", flaky)
    results, errors, _ = run_on_ranks(
        monkeypatch,
        world_size=3,
        val_paths=paths,
        tmp_path=tmp_path,
        declared=sum(SHARD_TOKENS),
    )

    assert not results, "a rank returned a result while another rank's shards were missing"
    assert set(errors) == {0, 1, 2}, f"only ranks {sorted(errors)} failed; the rest hung"
    for rank, error in errors.items():
        assert isinstance(error, SystemExit), f"rank {rank} raised {error!r}, not a Refusal"
        assert "held-out" in str(error)


def test_one_rank_failing_mid_forward_fails_every_rank_rather_than_hanging(
    monkeypatch, tmp_path
):
    """
    THE SECOND DEADLOCK, WHICH THE FIRST FIX MISSED. The download got a failure-is-a-value
    guard; the forward loop did not -- so a corrupt memmap or a single-card OOM on one rank
    unwound it while its peers stayed inside the loop, and they then walked into an all_reduce
    it would never enter. The same hang, one code block later, at the end of a paid run.

    Rank 2's third forward raises. Every rank must come back with a refusal, and none may hang.
    """
    paths = [
        write_shard(tmp_path / "corpus" / f"t{i}" / f"val-{i:05}.u32le.bin", n)
        for i, n in enumerate(SHARD_TOKENS)
    ]
    real_forward = entry._forward_ce
    calls = {"n": 0}

    def flaky_forward(model, xs, ys, *, vocab_size, device):
        # Not rank-conditioned: whichever rank happens to make the third call in this process
        # trips, which is enough to prove the group is not left waiting on it.
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated: CUDA OOM on this rank")
        return real_forward(model, xs, ys, vocab_size=vocab_size, device=device)

    monkeypatch.setattr(entry, "_forward_ce", flaky_forward)
    results, errors, _ = run_on_ranks(
        monkeypatch,
        world_size=3,
        val_paths=paths,
        tmp_path=tmp_path,
        declared=sum(SHARD_TOKENS),
    )

    assert not results, "a rank returned a CE while another rank's forward pass had failed"
    assert set(errors) == {0, 1, 2}, f"only ranks {sorted(errors)} failed; the rest hung"
    for rank, error in errors.items():
        assert isinstance(error, SystemExit), f"rank {rank} raised {error!r}, not a Refusal"
        assert "forward pass" in str(error)


def test_every_forward_pass_is_the_same_shape(shards, tmp_path, monkeypatch):
    """
    RAGGED BATCHES WERE A REAL RISK AND ARE NOW STRUCTURALLY IMPOSSIBLE.
    ``_shard_windows`` yields a short final micro-batch whenever a shard's window count is odd,
    so rank A's k-th pass could be ``(1, seq)`` while rank B's k-th was ``(2, seq)``. FSDP2
    happens to survive that -- its all-gathers are over parameter shards, not activations -- but
    "survives under the parallelism currently configured" is not a property to rest a run on,
    and it forces a torch.compile recompilation per distinct shape.

    So every pass is padded to ``(micro, seq_len)``, and the padding must NOT reach the sums.
    Both halves are checked here: the shapes, and that the token count is still the unpadded one.
    """
    seen = []
    real_forward = entry._forward_ce

    def recording(model, xs, ys, *, vocab_size, device):
        seen.append((tuple(xs.shape), tuple(ys.shape)))
        return real_forward(model, xs, ys, vocab_size=vocab_size, device=device)

    monkeypatch.setattr(entry, "_forward_ce", recording)
    result = evaluate(shards, tmp_path, micro=2)

    assert seen, "no forward pass happened"
    assert all(
        shapes == ((2, SEQ_LEN), (2, SEQ_LEN)) for shapes in seen
    ), f"ragged batches reached the model: {sorted(set(seen))}"

    # A shard with an odd window count exists in the fixture, so padding really was exercised --
    # 100 tokens gives 3 windows at micro=2, whose second batch has one row.
    assert any((n - 1) // SEQ_LEN % 2 == 1 for n in SHARD_TOKENS), "the fixture pads nothing"

    # And the padded rows contributed nothing: the count is still the unpadded 256.
    assert result["tokens"] == 256


def test_the_model_is_left_in_the_mode_it_was_found_in(shards, tmp_path):
    """
    ``model.eval()`` without a restore leaves dropout and norm statistics off for whatever runs
    next. Checked on the way out through a failure as well, since that is the path a ``finally``
    exists for and the easy one to write without.
    """
    model = TinyLM()
    model.train()
    evaluate(shards, tmp_path, model=model)
    assert model.training

    model.train()
    with pytest.raises(SystemExit):
        entry.evaluate_val_aggregate(
            model=model,
            vocab_size=VOCAB,
            val_paths=[],
            work_dir=str(tmp_path / "x"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=1,
        )
    assert model.training


# --- the corpus's own val partition, resolved rather than reconstructed -----------------------


@dataclass
class ManifestWithSplits:
    """The shape ``edullm_data.read.dataset_paths`` returns for olmo-150b-dolma2-v1.

    ``val`` is a property over ``splits`` filtered by the reader's ``is_trainable``, which is
    what the entry point reads. The val URIs carry a TOPIC DIRECTORY that their filenames do
    not, which is the trap this shape exists to reproduce.
    """

    paths: List[str] = field(
        default_factory=lambda: [
            f"s3://edullm-data/pretrain/olmo-150b-dolma2/v1/t/train-{i:05}.u32le.bin"
            for i in range(4)
        ]
    )
    dtype: Optional[str] = "uint32"
    byte_order: Optional[str] = "little"
    header_bytes: int = 0
    rows: Optional[int] = 157_237_308_712
    splits: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "train": [
                f"s3://edullm-data/pretrain/olmo-150b-dolma2/v1/t/train-{i:05}.u32le.bin"
                for i in range(4)
            ],
            "val": [
                "s3://edullm-data/pretrain/olmo-150b-dolma2/v1/all-dressed-snazzy2/"
                f"{'art_and_design' if i % 2 else 'mathematics'}/val-{i:05}.u32le.bin"
                for i in range(60)
            ],
        }
    )
    split_rows: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"train": 157_237_308_712, "val": 229_894_171}
    )

    @property
    def val(self):
        held = [p for name, ps in self.splits.items() if name != "train" for p in ps]
        return held or None


def resolve(manifest):
    return entry.corpus_from_manifest(
        manifest,
        dataset_id="pretrain/olmo-150b-dolma2",
        version="v1",
        tokenizer_id="tokenizer/dolma2-bpe",
    )


def test_the_val_partition_comes_from_the_manifest_with_its_topic_directory_intact():
    """
    The keys are the reader's, not rebuilt from filenames. ``val-00212.u32le.bin`` lives under
    24 topics; a key reconstructed from the name fetches a real, readable shard of the wrong
    topic, and every number downstream is plausible and wrong.
    """
    corpus = resolve(ManifestWithSplits())

    assert len(corpus.val_paths) == 60
    assert corpus.val_rows == 229_894_171
    # Every URI keeps the directory between the release prefix and the filename.
    for uri in corpus.val_paths:
        assert "/art_and_design/" in uri or "/mathematics/" in uri


def test_held_out_objects_never_appear_in_the_training_paths():
    """Training on the set the arm is scored against would make every contrast meaningless."""
    corpus = resolve(ManifestWithSplits())
    assert not set(corpus.val_paths) & set(corpus.paths)


def test_a_manifest_predating_the_split_api_degrades_to_no_val_rather_than_raising():
    """
    ``val`` and ``split_rows`` arrived in the reader after ``paths`` did. A corpus or a fake
    that predates them should report "no held-out split" -- which the endpoint then refuses
    loudly -- rather than raising an AttributeError in the middle of a config build.
    """

    @dataclass
    class Old:
        paths: List[str] = field(default_factory=lambda: ["s3://edullm-data/x/v1/a.u32le.bin"])
        dtype: Optional[str] = "uint32"
        byte_order: Optional[str] = "little"
        header_bytes: int = 0
        rows: Optional[int] = 1000

    corpus = entry.corpus_from_manifest(
        Old(), dataset_id="x", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )
    assert corpus.val_paths == []
    assert corpus.val_rows is None


def test_a_second_held_out_partition_is_refused_rather_than_folded_in():
    """
    THE CONTAMINATION THE TOKEN CHECK CANNOT SEE. The reader's ``is_trainable`` excludes
    everything that is not ``train``, and the dataset standard's vocabulary is
    ``{train, val, test}`` -- so a corpus declaring a ``test`` partition hands back val AND test
    concatenated in ``.val``.

    An earlier version of this code summed both declarations. The endpoint would then have
    scored val+test and compared it against val+test rows, so the exact token check BALANCED and
    reported a contaminated number as the endpoint. Both sides of an equality moving together is
    precisely the class of failure an equality cannot catch, which is why this is refused at
    resolution time instead.
    """
    manifest = ManifestWithSplits()
    manifest.splits["test"] = [
        "s3://edullm-data/pretrain/olmo-150b-dolma2/v1/held/test-00000.u32le.bin"
    ]
    manifest.split_rows["test"] = 5_000_000

    with pytest.raises(SystemExit) as refusal:
        resolve(manifest)
    assert "held-out partitions" in refusal.value.explanation
    assert "test" in refusal.value.explanation and "val" in refusal.value.explanation


def test_a_held_out_object_listed_twice_is_refused():
    """
    ``.val`` concatenates over partitions, so two overlapping held-out partitions list a shard
    twice -- and a shard scored twice is weighted twice in what is supposed to be a plain
    per-token mean.
    """
    manifest = ManifestWithSplits()
    # A partition that is a strict subset of `val`, which is how this arises in practice.
    manifest.splits["val-small"] = manifest.splits["val"][:3]
    manifest.split_rows["val-small"] = 1000

    with pytest.raises(SystemExit) as refusal:
        resolve(manifest)
    # Either guard is a correct refusal here; what must not happen is a silently doubled shard.
    assert "more than once" in refusal.value.explanation or "held-out partitions" in (
        refusal.value.explanation
    )


def test_the_val_partition_reaches_the_config_the_run_evaluates_from(monkeypatch):
    """
    ``train()`` receives the config, not the ``Corpus``, so the endpoint can only run if the val
    objects were carried across. This is the join between the two halves of the fix.
    """
    monkeypatch.setattr(entry, "resolve_corpus", lambda **kwargs: resolve(ManifestWithSplits()))
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/olmo-150b-dolma2",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            "--arm=G2R0",
            "--steps=10",
        ]
    )
    config = entry.build_config(opts, overrides)

    assert len(config.val_paths) == 60
    assert config.val_rows == 229_894_171
    # And it survives serialisation, because that record is what lands beside the checkpoint.
    assert config.as_config_dict()["val_rows"] == 229_894_171


# --- the init seed, at the entry point --------------------------------------------------------


def test_the_init_seed_flag_reaches_the_model_config(monkeypatch):
    """
    THE OTHER HALF OF THE FIX-1 REGRESSION TEST. ``core6_arms_test`` proves the seed reaches the
    tensors; this proves the flag reaches ``build_arm``. Both are needed: the plumbing was
    correct on one side of that call and absent on the other, and each half passes on its own
    while ``--init-seed`` does nothing.
    """
    monkeypatch.setattr(entry, "resolve_corpus", lambda **kwargs: resolve(ManifestWithSplits()))

    def config_for(seed):
        opts, overrides = entry.build_parser().parse_known_args(
            [
                "a-run-id",
                "--dataset-id=pretrain/olmo-150b-dolma2",
                "--dataset-version=v1",
                "--dataset-tokenizer=tokenizer/dolma2-bpe",
                "--save-folder=/tmp/x",
                f"--init-seed={seed}",
            ]
        )
        return entry.build_config(opts, overrides)

    for seed in (0, 1, 12536):
        config = config_for(seed)
        assert config.model.init_seed == seed, (
            f"--init-seed {seed} did not reach the model config, so the weights are drawn from "
            f"{config.model.init_seed} while the summary reports {seed}"
        )
        # The summary reads config.init_seed, so the two must not be able to disagree.
        assert config.init_seed == config.model.init_seed


@pytest.mark.parametrize("override", ["init_seed=99", "model.init_seed=42"])
def test_an_override_that_moves_only_one_seed_is_refused(monkeypatch, override):
    """
    THE FIX-1 BUG IN ITS SECOND FORM, WHICH THE FIX ITSELF OPENED. ``build_config`` sets
    ``ExperimentConfig.init_seed`` and ``model.init_seed`` together from ``--init-seed``, and
    THEN calls ``config.merge(overrides)``. ``main()`` uses ``parse_known_args``, so any bare
    token on the command line becomes a dotlist override -- and ``init_seed=99`` is a natural
    thing to type.

    That moves the reported field without moving the drawn-from field (or the reverse), and
    ``summarise()`` prints the reported one. The JSON would say 99 while the tensors came from
    whatever ``--init-seed`` was: exactly the false record the flag was just fixed for, arriving
    by a different door. Both directions are checked, since both are typeable.
    """
    monkeypatch.setattr(entry, "resolve_corpus", lambda **kwargs: resolve(ManifestWithSplits()))
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/olmo-150b-dolma2",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            "--init-seed=777",
            override,
        ]
    )
    config = entry.build_config(opts, overrides)

    # The override really does split them -- if this ever stops being true, the guard below is
    # no longer testing anything and this test should be re-derived rather than deleted.
    assert config.init_seed != config.model.init_seed, (
        f"{override!r} no longer splits the two seeds; the guard is untested"
    )

    with pytest.raises(SystemExit) as refusal:
        entry.train(config, opts)
    assert "init_seed" in refusal.value.explanation
    assert str(config.init_seed) in refusal.value.explanation
    assert str(config.model.init_seed) in refusal.value.explanation


def test_train_requires_opts_rather_than_defaulting_to_a_silent_run():
    """
    ``train(config, opts=None)`` used to be the signature, and everything after ``fit()`` -- the
    endpoint, the token assertion and ``summarise()`` -- sat behind ``if opts is not None``. A
    caller that omitted it got a run that trained, wrote a checkpoint, and printed NO JSON at
    all. The JSON is the only channel the platform reads results back through, so that is a
    larger failure than the null endpoint being fixed, not a smaller one.

    Asserted on the signature because the alternative is standing up a real trainer. A default
    reappearing here is the regression.
    """
    import inspect

    parameter = inspect.signature(entry.train).parameters["opts"]
    assert parameter.default is inspect.Parameter.empty, (
        "train() has a default for `opts` again; a caller that omits it will train, checkpoint "
        "and report nothing"
    )
