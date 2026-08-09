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
import textwrap
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

    IT PLACES ITSELF ON THE EVALUATOR'S DEVICE, and that is not cosmetic. Both evaluators pick
    ``cuda`` whenever one is visible (``train_core6_arm.py:1325``, ``:1438``, ``:1712``) and move
    every batch there. A model left on the CPU then meets CUDA token ids inside ``F.embedding``
    and dies with "index is on cuda:0, different from other tensors on cpu". On a laptop the two
    agree because both are CPU, so this suite was green locally and failed 44 ways on the first
    GPU host it ever ran on -- a whole class of test that had never actually executed against the
    device it ships to. Resolved here, once, rather than at the eight construction sites, because
    a per-site fix is one forgotten call away from the same failure.
    """

    def __init__(self, vocab: int = VOCAB, d: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.out = torch.nn.Linear(d, vocab)
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

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
    # Built on the MODEL's device, not the default one: `TinyLM` now places itself on cuda when a
    # GPU is visible (see its docstring), so a CPU `arange` here reproduces the same mismatch in
    # reverse -- "index is on cpu, different from other tensors on cuda:0" inside F.embedding.
    dev = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.arange(n_tokens - 1, dtype=torch.int64, device=dev).unsqueeze(0))[0]
        expected = float(
            torch.nn.functional.cross_entropy(
                logits[marked - 1].unsqueeze(0),
                torch.tensor([marked], device=dev),
                reduction="none",
            )
        )
    # rel=1e-3, AND THE NUMBER IS SIZED AGAINST BOTH SIDES RATHER THAN WIDENED UNTIL GREEN.
    # The evaluator runs the model under CUDA autocast while this independent answer runs it in
    # plain float32, so the two agree only to bf16 accumulation noise: measured 5.733294 against
    # 5.731090, a relative 3.85e-4 that a rel=1e-5 tolerance rejects. That is precision, not
    # position. The defect this test exists to catch -- reading `mask[off:off+seq]` instead of
    # `mask[off+1:off+seq+1]`, i.e. scoring the NEIGHBOURING token -- moves the CE by O(0.1-1)
    # nats, because per-position CEs of an untrained model are independent draws around
    # ln(512) = 6.24. So rel=1e-3 (tolerance 5.7e-3) sits 17x above the precision noise and still
    # 17x below the smallest defect signal; the guard keeps its teeth on both sides.
    assert out["bands"]["0"]["sum"] == pytest.approx(expected, rel=1e-3), (
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


# ---------------------------------------------------------------------------------------------
# THROUGHPUT AND MEMORY REPORTING
#
# These are CO-PRIMARY endpoints alongside `val_ce`, not diagnostics. At the token budget this
# study runs, the CE differences between arms may be under what three seeds can resolve, in which
# case throughput and peak memory are what the production choice actually rests on.
#
# All of it is pure arithmetic over recorded samples, so all of it runs on a CPU with no model,
# no CUDA and no trainer. Each test CALLS the function under test rather than re-deriving its
# formula: a test that recomputes `sum(tokens)/sum(seconds)` in its own body passes when the
# source changes to a mean-of-rates, which is the specific bug it would exist to catch.
# ---------------------------------------------------------------------------------------------


def samples(*triples):
    """`(step, seconds, tokens)` triples as StepSamples, so the tests read as data."""
    return [entry.StepSample(step=s, seconds=sec, tokens=tok) for s, sec, tok in triples]


def test_the_steady_figure_is_absent_rather_than_zero_when_nothing_survives_the_cutoff():
    """
    THE FAILURE THIS EXISTS FOR: a cell whose steps were all inside the warmup window reports
    `throughput_tok_s_steady: 0.0`, that arm sorts last in the ranking table, and a reader has no
    way to tell "this mixer is slow" from "this figure was never measured". Zero is not a
    conservative default for a speed -- it is a wrong claim in the direction that changes the
    recommendation.

    A short cell is not hypothetical: a crashed or cancelled attempt, a resumed run that stopped
    early, or a smoke test at 20 steps all produce fewer completed steps than the 50-step cutoff.

    The whole-run figure is checked in the same breath because it comes from a DIFFERENT
    computation -- wall clock rather than summed step time -- and a null on one does not imply a
    null on the other.
    """
    watcher = entry.LossWatcher()
    watcher.steps = samples(*((i, 0.5, 1000) for i in range(1, entry.WARMUP_STEPS_EXCLUDED + 1)))

    report = entry.throughput_report(watcher, world_size=8, wall_clock_seconds=100.0)

    assert report["throughput_tok_s_steady"] is None, (
        "no step survived the warmup cutoff, so there is no steady-state measurement; a number "
        "here is a fabricated one and 0.0 would rank this arm last"
    )
    assert report["throughput_tok_s_steady_per_device"] is None
    assert report["steady_state_steps"] == 0
    assert report["step_time_s_p50"] is None
    assert report["step_time_s_p90"] is None
    # The whole-run figure IS measurable here -- there was a wall clock and there were tokens --
    # so this also proves the null above is specific rather than the function failing wholesale.
    assert report["throughput_tok_s_whole_run"] is not None


def test_the_steady_figure_excludes_the_warmup_steps_it_says_it_excludes():
    """
    The cutoff is only worth having if the slow warmup steps are genuinely outside the average.
    Here the first 50 steps are 10x slower than the rest, so a figure that included them would be
    dragged far below the steady one -- and the two figures must therefore DIFFER.

    Asserted as a relationship between the two reported numbers rather than against a literal:
    a hard-coded expected value is a second implementation of the formula and goes stale silently.
    """
    slow = [(i, 5.0, 1000) for i in range(1, entry.WARMUP_STEPS_EXCLUDED + 1)]
    fast = [(i, 0.5, 1000) for i in range(entry.WARMUP_STEPS_EXCLUDED + 1, 151)]
    watcher = entry.LossWatcher()
    watcher.steps = samples(*(slow + fast))

    report = entry.throughput_report(watcher, world_size=4, wall_clock_seconds=1000.0)

    assert report["steady_state_steps"] == 100
    assert report["warmup_steps_excluded"] == entry.WARMUP_STEPS_EXCLUDED
    # The steady window is the fast steps only: 100 steps x 1000 tokens / (100 x 0.5s).
    assert report["throughput_tok_s_steady"] == pytest.approx(2000.0)
    # Including the warmup would give 150,000 tokens over 300s = 500 tok/s, a 4x understatement.
    assert report["throughput_tok_s_all_steps"] == pytest.approx(500.0)
    assert report["throughput_tok_s_steady"] > 3 * report["throughput_tok_s_all_steps"], (
        "the warmup steps are 10x slower and are supposed to be outside the steady figure; if "
        "these two are close, the cutoff is not being applied"
    )
    # Per-device is the total divided by the world size that ships beside it.
    assert report["throughput_tok_s_steady_per_device"] == pytest.approx(500.0)


def test_the_cutoff_filters_on_the_step_index_so_a_resumed_run_is_not_over_trimmed():
    """
    A second Batch attempt resumes at step 1,201, so its FIRST recorded sample is step 1,201 --
    already long past any warmup. Dropping "the first 50 entries of the list" would discard 50
    perfectly good steady-state steps and exclude nothing that needed excluding, and the
    resulting figure would still look completely normal.
    """
    resumed = samples(*((i, 0.5, 1000) for i in range(1201, 1301)))
    kept = entry.steps_after_warmup(resumed)
    assert len(kept) == 100, "a resumed run has no warmup steps in its samples to discard"
    assert kept[0].step == 1201


def test_the_cutoff_is_strict_so_exactly_warmup_steps_are_discarded():
    """
    `>` versus `>=` on the cutoff is a one-step difference that no reported number would ever
    reveal. With `warmup_steps=50`, step 50 is discarded and step 51 is the first kept.
    """
    kept = entry.steps_after_warmup(samples(*((i, 1.0, 10) for i in range(1, 101))), warmup_steps=50)
    assert len(kept) == 50
    assert kept[0].step == 51


def test_throughput_is_total_tokens_over_total_seconds_and_not_a_mean_of_rates():
    """
    THE TWO ARE DIFFERENT QUANTITIES AND ONLY ONE IS THROUGHPUT. One slow step among fast ones --
    a checkpoint step, an allocator stall -- is what separates them: the ratio of sums charges
    that step its full duration, while a mean of per-step rates gives it the same weight as a
    fast one and reports a speed the run never achieved.

    Constructed so the two answers are far apart: nine steps at 1000 tok/s and one at 100 tok/s.
    Ratio of sums = 10,000 tokens / 19s = 526. Mean of rates = (9x1000 + 100)/10 = 910.
    """
    fast_then_one_slow = samples(*([(i, 1.0, 1000) for i in range(1, 10)] + [(10, 10.0, 1000)]))
    measured = entry.throughput_tokens_per_second(fast_then_one_slow)
    assert measured == pytest.approx(10_000 / 19.0)
    assert measured < 600, (
        "a mean of per-step rates would report ~910 tok/s here; throughput is the ratio of sums "
        "and must charge the slow step its full duration"
    )


def test_throughput_is_none_rather_than_zero_when_no_tokens_were_counted():
    """
    A data loader that reports no token count leaves every sample at zero tokens. Dividing that
    by real seconds is 0.0 tok/s -- a claim that the arm produced nothing, rather than the truth,
    which is that nothing was counted.
    """
    assert entry.throughput_tokens_per_second(samples((51, 1.0, 0), (52, 1.0, 0))) is None
    assert entry.throughput_tokens_per_second([]) is None
    # And a zero-duration sample cannot become a division by zero or an infinity.
    assert entry.throughput_tokens_per_second(samples((51, 0.0, 1000))) is None


@pytest.mark.parametrize(
    "values,q,expected",
    [
        # Nearest rank: index ceil(q*n)-1 into the sorted list, so the answer is always an
        # OBSERVED value rather than an interpolation between two that were not.
        ([1.0, 2.0, 3.0], 0.5, 2.0),
        # Even n: the lower of the two middle values, not their average (which would be 2.5).
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
        # p90 of ten sorted values is the 9th.
        ([float(i) for i in range(1, 11)], 0.9, 9.0),
        # Unsorted input must be sorted first; the max must be reachable at q=1.
        ([5.0, 1.0, 3.0], 1.0, 5.0),
        # A single sample is its own every quantile.
        ([7.0], 0.9, 7.0),
    ],
)
def test_the_quantile_is_nearest_rank_over_the_sorted_values(values, q, expected):
    assert entry.quantile_nearest_rank(values, q) == expected


def test_the_quantile_of_nothing_is_none_rather_than_zero():
    """A p90 step time of 0.0 seconds is an infinitely fast arm. It must be absent instead."""
    assert entry.quantile_nearest_rank([], 0.5) is None
    assert entry.quantile_nearest_rank([], 0.9) is None


def test_p90_is_at_or_above_p50_on_every_shape_of_input():
    """
    An ordering violation between the two would be invisible in a results table and would
    silently invert what a scheduler thinks the tail costs. Checked over several lengths because
    the nearest-rank index is where an off-by-one would hide.
    """
    for n in range(1, 40):
        values = [float((i * 7) % n) for i in range(n)]
        p50 = entry.quantile_nearest_rank(values, 0.5)
        p90 = entry.quantile_nearest_rank(values, 0.9)
        assert p50 is not None and p90 is not None
        assert p90 >= p50, f"p90 {p90} below p50 {p50} at n={n}"


def test_mfu_is_none_rather_than_wrong_when_the_card_is_not_in_the_table():
    """
    UPSTREAM'S TABLE ENDS IN `else: assume A100`, SO EVERY UNRECOGNISED CARD GETS 312 TFLOP/s.
    On an L40S -- 181 TFLOP/s dense -- that denominator is 1.72x too high and the MFU printed
    from it is 1.72x too low, with nothing in the record saying it was a guess. A wrong number
    that looks authoritative is worse here than no number.
    """
    assert entry.device_peak_bf16_flops("NVIDIA GeForce RTX 4090") is None
    assert entry.device_peak_bf16_flops(None) is None
    assert entry.device_peak_bf16_flops("") is None
    assert (
        entry.model_flops_utilisation(
            tokens_per_second_per_device=5000.0,
            flops_per_token=2_000_000_000,
            device_peak_flops=None,
        )
        is None
    )


def test_the_a100_is_in_the_table_because_that_is_the_card_this_study_runs_on():
    """
    The bake-off runs on gpu-8xa100 -- the only 80GB shape this account can get -- so an absent
    A100 entry would make MFU null on every cell of the actual study. Named cards are checked in
    the forms torch reports them in.
    """
    assert entry.device_peak_bf16_flops("NVIDIA A100-SXM4-80GB") == 312_000_000_000_000
    assert entry.device_peak_bf16_flops("NVIDIA A100 80GB PCIe") == 312_000_000_000_000
    # L40S is the FarmShare card and must NOT resolve to the A100's peak.
    assert entry.device_peak_bf16_flops("NVIDIA L40S") == 181_000_000_000_000


def test_the_longest_device_key_wins_so_h100_nvl_is_not_shadowed():
    """
    `H100` is a substring of `NVIDIA H100 NVL`, so a first-match-wins scan over an unordered
    table would give the NVL card the SXM peak -- 989 against 835 TFLOP/s, an 18% error in the
    MFU denominator that no reader could see.
    """
    assert entry.device_peak_bf16_flops("NVIDIA H100 NVL") == 835_500_000_000_000
    assert entry.device_peak_bf16_flops("NVIDIA H100 80GB HBM3") == 989_500_000_000_000


def test_mfu_uses_the_per_device_throughput_and_not_the_total():
    """
    MFU is a per-CARD utilisation. Feeding it the whole machine's tokens/second would multiply
    the answer by the world size -- 800% on an 8-GPU node, which at least looks wrong; at world
    size 1 it is silently identical, which is how the bug ships.
    """
    mfu = entry.model_flops_utilisation(
        tokens_per_second_per_device=1000.0,
        flops_per_token=312_000_000_000,
        device_peak_flops=312_000_000_000_000,
    )
    assert mfu == pytest.approx(100.0)


def test_mfu_is_none_when_the_model_reports_no_flops_per_token():
    assert (
        entry.model_flops_utilisation(
            tokens_per_second_per_device=5000.0,
            flops_per_token=None,
            device_peak_flops=312_000_000_000_000,
        )
        is None
    )


def test_an_unmeasurable_mfu_says_why_rather_than_leaving_a_bare_null():
    """
    A null with no reason is indistinguishable from a bug in the reporting code, and the causes
    want different fixes: an unlisted card needs a table entry, a missing flops/token needs the
    model to implement it.
    """
    watcher = entry.LossWatcher()
    watcher.steps = samples(*((i, 0.5, 1000) for i in range(1, 151)))

    unknown_card = entry.throughput_report(
        watcher, world_size=1, wall_clock_seconds=100.0, flops_per_token=10**9, device_name="TPUv5"
    )
    assert unknown_card["mfu_pct"] is None
    assert "TPUv5" in unknown_card["mfu_basis"]

    no_flops = entry.throughput_report(
        watcher, world_size=1, wall_clock_seconds=100.0, flops_per_token=None, device_name="A100"
    )
    assert no_flops["mfu_pct"] is None
    assert "FLOPs per token" in no_flops["mfu_basis"]


def test_peak_memory_is_null_and_labelled_rather_than_zero_when_there_is_no_cuda():
    """
    0.0 GiB of peak memory is a claim that the arm is free. On a co-primary memory endpoint that
    would rank the unmeasured arm FIRST, which is the worst possible direction for a missing
    measurement to fail in.
    """
    watcher = entry.LossWatcher()  # never sampled: no CUDA, no steps
    report = entry.memory_report(watcher)
    if torch.cuda.is_available():
        pytest.skip("this asserts the no-CUDA branch and there is a CUDA device here")
    assert report["peak_memory_gib"] is None
    assert report["peak_memory_reserved_gib"] is None
    assert report["peak_memory_source"] == "unavailable"


def test_peak_memory_prefers_the_running_maximum_over_the_truncated_read():
    """
    `GPUMemoryMonitorCallback.post_step` calls `reset_peak_memory_stats()` on EVERY step, so a
    `max_memory_allocated()` read after `fit()` is the LAST STEP's peak. It is a lower bound
    wearing the name of a whole-run peak, and it is the field somebody sizes a card with.

    This matters most for the R=2 Householder arm, whose Triton backward allocates an
    O(B*T*H*K*V) fp32 workspace -- a within-step transient that only a per-step maximum sees.
    """
    watcher = entry.LossWatcher()
    watcher.peak_allocated_bytes = 40 * 1024**3
    watcher.peak_reserved_bytes = 44 * 1024**3
    watcher.memory_samples = 1900

    report = entry.memory_report(watcher)
    assert report["peak_memory_source"] == "per_step_running_max", (
        "the sampled running maximum was available and must be preferred to the truncated "
        "post-fit read"
    )
    assert report["peak_memory_gib"] == pytest.approx(40.0)
    assert report["peak_memory_reserved_gib"] == pytest.approx(44.0)
    assert report["peak_memory_samples"] == 1900


def test_the_watcher_samples_memory_before_the_gpu_monitor_resets_it():
    """
    THE ORDERING IS LOAD-BEARING AND IS OTHERWISE HELD ONLY BY LUCK. `LossWatcher.post_step`
    reads the CUDA peak counters; `GPUMemoryMonitorCallback.post_step` resets them. The trainer
    runs callbacks in descending priority, so the watcher must have the HIGHER priority of the
    two or every reading it takes is of a counter that was just zeroed -- and the resulting
    figure would look entirely plausible.
    """
    from olmo_core.train.callbacks import GPUMemoryMonitorCallback

    assert entry.LossWatcher.priority > GPUMemoryMonitorCallback.priority, (
        "LossWatcher must run before GPUMemoryMonitorCallback in post_step, or it samples peak "
        "memory counters that the monitor has already reset"
    )


def test_the_two_throughput_figures_have_names_that_cannot_be_confused():
    """
    The scar this guards: a 200-step run reported 455,789 tok/s where a 20-step run at the same
    microbatch reported 303,072, and the higher figure was a run-length artifact. Two numbers
    that differ by 1.5x for methodological reasons must not share a key, and must not be
    distinguished only by a suffix somebody can drop when copying into a table.
    """
    watcher = entry.LossWatcher()
    watcher.steps = samples(*((i, 0.5, 1000) for i in range(1, 151)))
    report = entry.throughput_report(watcher, world_size=8, wall_clock_seconds=200.0)

    assert "throughput_tok_s_steady" in report
    assert "throughput_tok_s_whole_run" in report
    assert report["throughput_tok_s_steady"] != report["throughput_tok_s_whole_run"]
    # And the steady figure must be the HIGHER of the two: the whole-run one carries startup.
    assert report["throughput_tok_s_steady"] > report["throughput_tok_s_whole_run"]


def test_the_report_never_emits_a_zero_for_a_figure_it_could_not_measure():
    """
    A sweep of the whole emitted object on the worst input there is -- no steps, no wall clock.
    Every numeric field must be null; a 0.0 anywhere in here is a measurement claim.
    """
    report = entry.throughput_report(entry.LossWatcher(), world_size=8, wall_clock_seconds=0.0)
    for key, value in report.items():
        if key in ("steps_measured", "steady_state_steps", "warmup_steps_excluded"):
            continue  # honest counts of nothing
        if isinstance(value, str):
            continue  # mfu_basis, which explains the nulls
        assert value is None, f"{key} is {value!r}; an unmeasured figure must be null, not zero"


# --- the decode / inference measurement -------------------------------------------------------
#
# WHAT THESE CAN AND CANNOT COVER. The kernel timing needs a GPU and `fla`, and neither exists on
# a laptop, so none of it is tested here -- see the module note at the top of this file. What IS
# tested is every number that is COMPUTED rather than timed: the state footprint that decides
# serving batch size, the KV-cache contrast that gives it meaning, the crossover, the
# tokens/sec arithmetic, and the basis string that stops an operator microbenchmark being quoted
# as a serving figure. Those are the fields a reader acts on, and they are pure functions
# precisely so that a CPU test can call them.
#
# EVERY EXPECTED VALUE BELOW IS COMPUTED INDEPENDENTLY AND WRITTEN DOWN, not re-derived from the
# code's own expression. A test that recomputes `n * k * v * layers * bytes` passes whatever that
# line becomes, which is the documented way this project has shipped green nothing before.


def test_the_recurrent_state_size_is_the_hand_computed_number():
    """
    THE FIELD THAT DECIDES SERVING BATCH SIZE, PINNED TO ARITHMETIC DONE BY HAND.

    At the frozen geometry -- 16 value heads, head_k_dim 64, head_v_dim 64, 2 mixer layers, fp32
    state -- one head's state matrix is 64x64 = 4,096 elements. Sixteen heads is 65,536; that is
    262,144 bytes at 4 bytes each; two layers is 524,288 bytes = exactly 512 KiB per sequence.

    MUTATIONS THIS CATCHES: dropping `n_layers` from the product (reports 256 KiB, half the real
    footprint, on the field somebody sizes a fleet with); using `head_k_dim` twice instead of
    `head_k_dim * head_v_dim`; and defaulting `bytes_per_element` to 2 on the theory that "the run
    is bf16", which halves the answer -- fla keeps the state in fp32 regardless.
    """
    assert (
        entry.recurrent_state_bytes(
            n_heads=16, head_k_dim=64, head_v_dim=64, n_layers=2, bytes_per_element=4
        )
        == 524_288
    )
    assert 524_288 == 512 * 1024, "the hand arithmetic above must equal 512 KiB"

    # One layer is exactly half, which pins that `n_layers` is a factor rather than ignored.
    assert (
        entry.recurrent_state_bytes(
            n_heads=16, head_k_dim=64, head_v_dim=64, n_layers=1, bytes_per_element=4
        )
        == 262_144
    )
    # An asymmetric shape, so a `head_k_dim ** 2` bug cannot pass: 8 * 32 * 128 * 1 * 4 = 131,072.
    assert (
        entry.recurrent_state_bytes(
            n_heads=8, head_k_dim=32, head_v_dim=128, n_layers=1, bytes_per_element=4
        )
        == 131_072
    )
    # And the fp32 default is the DEFAULT, not something the caller must remember.
    assert entry.recurrent_state_bytes(
        n_heads=16, head_k_dim=64, head_v_dim=64, n_layers=2
    ) == 524_288


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_heads=0, head_k_dim=64, head_v_dim=64, n_layers=2),
        dict(n_heads=16, head_k_dim=64, head_v_dim=64, n_layers=0),
        dict(n_heads=16, head_k_dim=0, head_v_dim=64, n_layers=2),
        dict(n_heads=16, head_k_dim=64, head_v_dim=64, n_layers=2, bytes_per_element=0),
    ],
)
def test_a_zero_dimension_is_refused_rather_than_reporting_a_free_state(kwargs):
    """
    A ZERO ANYWHERE IN THE PRODUCT REPORTS A STATE OF 0 BYTES, WHICH IS A CLAIM THAT THE ARM IS
    FREE -- the exact direction a missing measurement must never fail in. `n_layers=0` is the
    reachable one: an arm whose `kda_layers` is empty would produce it, and the answer would be a
    linear-attention mixer with no memory cost at all.

    MUTATION THIS CATCHES: deleting the guard, which makes every case here return 0 and pass any
    test that only checks the happy path.
    """
    with pytest.raises(ValueError):
        entry.recurrent_state_bytes(**kwargs)


def test_the_kv_cache_contrast_is_the_hand_computed_number():
    """
    THE COMPARISON THAT MAKES THE STATE FIGURE A DECISION. Six global-attention layers, 8 KV
    heads, head_dim 64, bf16, both K and V:

        per layer per token = 2 * 8 * 64 * 2 bytes = 2,048
        six layers          = 12,288 bytes per token
        at 4,096 tokens     = 50,331,648 bytes = exactly 48 MiB

    MUTATIONS THIS CATCHES: dropping the factor of 2 for K-and-V (halves it); using `n_heads=16`
    instead of `n_kv_heads=8` (doubles it -- these arms are GQA, so the cache is over KV heads);
    and forgetting `seq_len`, which turns a per-token cost into a per-sequence constant and
    destroys the entire crossover argument.
    """
    assert (
        entry.kv_cache_bytes(n_kv_heads=8, head_dim=64, n_layers=6, seq_len=1) == 12_288
    )
    at_4k = entry.kv_cache_bytes(n_kv_heads=8, head_dim=64, n_layers=6, seq_len=4096)
    assert at_4k == 50_331_648
    assert at_4k == 48 * 1024 * 1024, "the hand arithmetic above must equal 48 MiB"

    # Linear in seq_len -- which is the whole property that distinguishes it from the fixed state.
    doubled = entry.kv_cache_bytes(n_kv_heads=8, head_dim=64, n_layers=6, seq_len=8192)
    assert doubled == 2 * at_4k


def test_the_crossover_is_where_the_kv_cache_passes_the_fixed_state():
    """
    512 KiB of fixed state against 12,288 bytes per token is 524,288 / 12,288 = 42.67 tokens.

    THE NUMBER IS SMALL AND THAT IS THE FINDING, not a bug in the test: past ~43 tokens of context
    the mixer's fixed state is already cheaper than the KV cache it replaces, and the gap widens
    without bound. Pinned so that a sign error or an inverted ratio -- which would put the
    crossover at 0.023 tokens, or in the thousands -- is caught rather than believed.

    MUTATION THIS CATCHES: inverting the division to `kv_bytes_per_token / state_bytes`.
    """
    crossover = entry.decode_state_crossover_tokens(
        state_bytes=524_288, kv_bytes_per_token=12_288
    )
    assert crossover == pytest.approx(42.666, abs=0.01)

    # No attention layers means no KV cost, and a length cannot be divided out of that -- null
    # rather than an infinity or a zero.
    assert entry.decode_state_crossover_tokens(state_bytes=524_288, kv_bytes_per_token=0) is None


def test_decode_throughput_is_batch_over_latency_and_null_when_unmeasurable():
    """
    Every sequence in the batch emits one token per step, so 32 sequences at 2 ms per step is
    16,000 tok/s -- computed by hand here, not from the code's expression.

    MUTATIONS THIS CATCHES: returning `1 / seconds` and ignoring the batch (reports 500 tok/s at
    batch 32, understating by 32x and ranking every arm on a single-stream figure); and returning
    0.0 instead of None for a missing latency, which is a claim that decode produced no tokens
    rather than that it was never timed.
    """
    assert entry.decode_tokens_per_second(seconds_per_token=0.002, batch_size=32) == 16_000.0
    assert entry.decode_tokens_per_second(seconds_per_token=0.002, batch_size=1) == 500.0

    for bad in (None, 0.0, -1.0):
        assert entry.decode_tokens_per_second(seconds_per_token=bad, batch_size=32) is None
    assert entry.decode_tokens_per_second(seconds_per_token=0.002, batch_size=0) is None


def test_the_decode_basis_says_it_is_not_a_whole_model_serving_number():
    """
    THE FIELD THAT STOPS THE MISQUOTE. These latencies are one fused operator on 2 of 16 layers,
    and "3,000 tokens/sec" reads exactly like a serving figure. The exclusions therefore travel
    INSIDE the value, because whoever reads the JSON will not read this file.

    MUTATION THIS CATCHES: shortening the string to something that names only what was measured.
    The assertions below require the words that mark the LIMITS, so a basis that describes only
    the covered part fails.
    """
    basis = entry.decode_basis_string(
        measured=True,
        operator="kda",
        kernel="fla.ops.kda.fused_recurrent.fused_recurrent_kda",
        n_heads=16,
        head_k_dim=64,
        head_v_dim=64,
        mixer_layers=2,
        total_layers=16,
    )
    lowered = basis.lower()
    assert "excludes" in lowered
    assert "not a whole-model serving" in lowered
    # It must name the layer accounting, since 2-of-16 is the single most misreadable part.
    assert "2 mixer layer(s) of 16" in basis
    # And it must name the attention layers whose cache DOES grow, or the summary implies a pure
    # linear-attention model this study is not testing.
    assert "grow with context" in lowered

    # An unmeasured basis states a CAUSE rather than being empty or absent.
    unmeasured = entry.decode_basis_string(measured=False, reason="no CUDA device")
    assert "not measured" in unmeasured.lower() and "no CUDA device" in unmeasured
    # And never silently claims a measurement it did not make.
    assert "tokens/sec" not in unmeasured


def test_an_unmeasured_decode_probe_is_never_reported_as_a_fast_one(monkeypatch):
    """
    THE RECEIPT, ON THE PATH A LAPTOP AND A CUDA-LESS CONTAINER BOTH TAKE. This is the property
    the whole audit finding rests on: a field that records the REQUEST while something else ran.
    With no CUDA there is no kernel, so `decode_fast_path_taken` must be False, every latency must
    be absent, and the basis must say why.

    It must ALSO still report the state footprint, because that number is arithmetic on the
    geometry and does not need a GPU -- a probe that returned nothing at all would drop the one
    field that decides serving batch size.

    MUTATIONS THIS CATCHES: initialising `decode_fast_path_taken` to True and only setting it
    False on an exception (the no-CUDA path returns early and would stay True); and setting it
    from `torch.cuda.is_available()` or from the requested kernel name rather than from the
    observed execution.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    probe = entry.decode_probe(arm_name="KDA_BASE")

    assert probe["decode_fast_path_taken"] is False
    assert probe["decode_batches"] == {}
    assert "not measured" in probe["decode_basis"].lower()
    # The receipt fields are absent rather than optimistic.
    assert probe["decode_kernel_resolved"] is None
    assert probe["decode_state_advanced"] is None
    assert probe["decode_state_dtype_realised"] is None

    # But the computed footprint IS there, and it is the hand-computed 512 KiB.
    assert probe["decode_state_bytes_per_seq"] == 524_288
    assert probe["decode_mixer_layers"] == 2
    # And the requested kernel is recorded as a request, under a name that says so.
    assert probe["decode_kernel_requested"] == (
        "fla.ops.kda.fused_recurrent.fused_recurrent_kda"
    )


def test_every_run_2_arm_resolves_to_a_decode_kernel(monkeypatch):
    """
    THE COORDINATION CHECK, AND IT IS THE ONE MOST LIKELY TO FAIL SILENTLY. The kernel table is
    keyed on the mixer's CONFIG CLASS rather than on `core6_arms.MIXERS`' registry strings,
    exactly so that `KDA_NEGEIG` -- new this wave, and being added by another agent -- works by
    construction instead of falling through to "no kernel known" and dropping a fifth of the
    study's new measurement.

    So: every arm this run measures must resolve to a kernel. Arms absent from `ARMS` are skipped
    rather than failed, because this test must not break while another agent's arm is mid-landing;
    what it will not tolerate is an arm that EXISTS and resolves to nothing.

    MUTATION THIS CATCHES: re-keying `DECODE_KERNELS` on the registry string (`"kda"`,
    `"gdn2"`, ...), which makes any newly-named KDA variant unresolvable.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from olmo_core.nn.transformer.core6_arms import ARMS

    run_2_arms = ["KDA_BASE", "KDA_NOACT", "KDA_NEGEIG", "KDA_GCONV", "GDN2"]
    checked = 0
    for name in run_2_arms:
        if name not in ARMS:
            continue  # not landed yet; another agent owns it
        probe = entry.decode_probe(arm_name=name)
        assert probe.get("decode_kernel_requested"), (
            f"arm {name} resolved to no decode kernel; its mixer config class "
            f"{probe.get('decode_config_class')!r} is missing from DECODE_KERNELS"
        )
        # The state footprint is the headline field and must be real for every arm.
        assert probe["decode_state_bytes_per_seq"] == 524_288, (
            f"arm {name} reports a different state size; every run-2 arm shares the frozen "
            "geometry, so a difference here is a config change rather than a mixer property"
        )
        checked += 1

    assert checked >= 4, (
        f"only {checked} of the run-2 arms were found in ARMS; this test is meant to cover the "
        "four that already exist"
    )


def test_the_householder_arms_report_an_absence_rather_than_a_kda_number(monkeypatch):
    """
    THE HONEST GAP. The R>1 Householder operator is a custom in-tree kernel with no fused
    recurrent form in `fla`, so there is nothing to time. The wrong fix is to fall back to KDA's
    kernel, which would report a number for a DIFFERENT operator under the Householder arm's
    name -- and since R=1 is documented to be parameter-identical to KDA, the number would look
    entirely plausible.

    MUTATION THIS CATCHES: adding `KimiDeltaHouseholderConfig` to `DECODE_KERNELS` pointed at
    `fused_recurrent_kda`.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from olmo_core.nn.transformer.core6_arms import ARMS

    if "KDA_R2" not in ARMS:
        pytest.skip("KDA_R2 is not declared in this build")
    probe = entry.decode_probe(arm_name="KDA_R2")

    assert probe["decode_fast_path_taken"] is False
    assert probe.get("decode_kernel_requested") is None
    assert "nothing to time" in probe["decode_basis"]
    # The state size is still computable and still reported: the operator has a fixed state even
    # though this harness cannot time its decode.
    assert probe["decode_state_bytes_per_seq"] > 0


def test_the_decode_probe_never_raises_even_on_an_unknown_arm():
    """
    A benchmark bolted onto a paid-for training run must not be able to cost it the CE endpoint.
    An arm name that is not in the registry is the cheapest way to reach the failure path.

    MUTATION THIS CATCHES: letting `_decode_geometry`'s KeyError propagate, which would take out
    `summarise()` -- and `summarise()` is the only channel the platform reads results through, so
    a decode bug would turn an eleven-hour run into a run that reported nothing.
    """
    probe = entry.decode_probe(arm_name="NO_SUCH_ARM")
    assert probe["decode_fast_path_taken"] is False
    assert "not measured" in probe["decode_basis"].lower()


# --- the sliced eval's band reduction ---------------------------------------------------------


def test_a_band_is_a_token_weighted_mean_and_not_a_mean_of_rank_means():
    """
    THE SILENT BIAS THIS FUNCTION EXISTS TO PREVENT, WITH THE TWO ANSWERS FAR ENOUGH APART TO
    TELL THEM APART.

    Two ranks, one band. Rank A scored 1,000 tokens at CE 2.0 (sum 2,000); rank B scored 10
    tokens at CE 10.0 (sum 100). Reduced correctly the band CE is

        (2000 + 100) / (1000 + 10) = 2100 / 1010 = 2.0792...

    A mean of the two ranks' means would be (2.0 + 10.0) / 2 = 6.0 -- almost 3x higher, entirely
    in range, and no assertion anywhere would fire. That is the shape of error that puts a wrong
    number into a production decision, and bands are where it bites hardest because a rare band
    like `gap>4096` can sit almost entirely on one rank.

    MUTATION THIS CATCHES: dividing per-rank and averaging, or reducing an already-divided CE.
    The expected value is worked out by hand above and written down, so it does not move when the
    code does.
    """
    reduced = entry.band_ce_from_totals({4096: 2000.0 + 100.0}, {4096: 1000 + 10})
    assert reduced["4096"]["ce"] == pytest.approx(2100.0 / 1010.0)
    assert reduced["4096"]["ce"] == pytest.approx(2.079207920792079)
    # The wrong answer, named so a future reader sees what is being excluded.
    assert reduced["4096"]["ce"] != pytest.approx(6.0)
    # Sums and counts survive into the record, so a reader can re-derive the mean or difference
    # two arms without re-weighting.
    assert reduced["4096"]["sum"] == 2100.0
    assert reduced["4096"]["n"] == 1010


def test_an_empty_band_is_null_rather_than_a_perfect_score():
    """
    THE BUG THE OLD CODE HAD. It divided by `max(count, 1)`, so a band with no tokens reported a
    cross-entropy of 0.0 -- the BEST possible score, the top of the table, for a band that was
    never measured at all. On a small slice `gap>4096` is easily empty, and arms are ranked on
    these numbers.

    MUTATION THIS CATCHES: restoring `total / max(n, 1)`, which turns every empty band back into
    a perfect 0.0.
    """
    reduced = entry.band_ce_from_totals(
        {0: 100.0, 4096: 0.0}, {0: 50, 4096: 0}
    )
    assert reduced["0"]["ce"] == pytest.approx(2.0)
    assert reduced["4096"]["ce"] is None, "an unmeasured band must not report a score"
    assert reduced["4096"]["n"] == 0
    # Null and zero are opposite claims; assert the distinction explicitly.
    assert reduced["4096"]["ce"] != 0.0


def test_the_bands_come_back_in_a_fixed_order_with_string_keys():
    """
    Keys are strings because they go through JSON, and the order is sorted numerically rather
    than by dict insertion -- every rank enters the reduction collectives in this order, and
    "whatever order the dict happened to have" is the wrong thing to rest that on.
    """
    reduced = entry.band_ce_from_totals(
        {4096: 1.0, 0: 1.0, 256: 1.0, 32: 1.0, 1024: 1.0},
        {4096: 1, 0: 1, 256: 1, 32: 1, 1024: 1},
    )
    assert list(reduced) == ["0", "32", "256", "1024", "4096"]
    assert set(reduced) == {str(b) for b in entry.BAND_BIT}


# --- the sliced eval runs on every rank, run as actual ranks -----------------------------------
#
# The same thread harness the aggregate endpoint uses, for the same reason: an AST walk looking
# for `get_rank() == 0` is theatre that passes for `if rank == 0:`. These execute the real control
# flow of every rank against a rendezvous with a deadlock detector, so a rank that skips a
# collective its peers enter is reported as a failure rather than reproduced as a hung suite.


def write_mask(path, n_tokens: int, *, bit: int = 1, every: int = 1) -> str:
    """A uint8 band mask, one byte per token, with ``bit`` set on every ``every``-th position."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(n_tokens, dtype=np.uint8)
    mask[::every] = bit
    mask.tofile(path)
    return str(path)


def run_sliced_on_ranks(monkeypatch, *, world_size: int, pairs, seq_len=SEQ_LEN, micro=2):
    """Run ``evaluate_sliced`` on ``world_size`` threads, sharding ``pairs`` the way train() does.

    ``pairs`` is the WHOLE ``(shard, mask)`` set; each rank takes ``i % world_size == rank``,
    which is what `fetch_slice_inputs` does with the manifest.
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
    torch.manual_seed(0)
    model = TinyLM()

    def body(rank: int):
        local.rank = rank
        mine = [p for i, p in enumerate(pairs) if i % world_size == rank]
        try:
            results[rank] = entry.evaluate_sliced(
                model=model,
                vocab_size=VOCAB,
                val_paths=[s for s, _ in mine],
                mask_paths=[m for _, m in mine],
                seq_len=seq_len,
                micro=micro,
            )
        except BaseException as exc:  # noqa: BLE001 -- reported per rank below
            errors[rank] = exc
            group.abandon()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive(), "a rank never finished -- the sliced eval hung"
    return results, errors, group


@pytest.fixture
def slice_pairs(tmp_path):
    """Four ``(shard, mask)`` pairs at the aggregate fixture's sizes, every token in band 0."""
    pairs = []
    for i, n in enumerate(SHARD_TOKENS):
        shard = write_shard(tmp_path / "slice" / f"s{i}.u32le.bin", n)
        mask = write_mask(tmp_path / "slice" / f"s{i}.mask.u8", n, bit=1)
        pairs.append((shard, mask))
    return pairs


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_the_sliced_eval_reaches_every_collective_on_every_rank(
    monkeypatch, slice_pairs, world_size
):
    """
    THE FIX, MEASURED. This path was `if get_rank() == 0:` and run 2 passes `--slice-mask-uri` on
    8 GPUs, so it WOULD have hung: under FSDP rank zero's forward issues all-gathers the other
    ranks never enter, and the caller's `except` cannot catch a hang.

    Four world sizes for the cases the shard assignment treats differently: one rank; a size that
    divides the shard count; a size that does NOT (so some ranks run filler passes); and a size
    LARGER than the shard count, where some ranks hold NO shards at all. That last is 8 GPUs over
    4 mask shards -- not exotic, it is the default shape this run submits on, and it is the one a
    naive implementation deadlocks on because a rank with nothing to score leaves the loop
    immediately.

    MUTATION THIS CATCHES: restoring the rank gate, or dropping the filler passes -- either makes
    the ranks' collective traces diverge and the FakeGroup times out naming the collective.
    """
    results, errors, group = run_sliced_on_ranks(
        monkeypatch, world_size=world_size, pairs=slice_pairs
    )
    assert not errors, f"ranks failed: { {r: repr(e) for r, e in errors.items()} }"
    assert set(results) == set(range(world_size)), "not every rank produced a result"

    traces = [group.trace[rank] for rank in range(world_size)]
    assert all(t == traces[0] for t in traces), f"ranks diverged: {traces}"
    # The step budget, the failure flag, the aggregate pair, and two per band.
    assert traces[0].count("all_reduce:max") >= 1
    assert traces[0].count("all_reduce:sum") >= 2 + 2 * len(entry.BAND_BIT)
    assert traces[0].count("barrier") >= 1


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_the_sliced_number_is_the_whole_runs_and_not_one_ranks_share(
    monkeypatch, slice_pairs, world_size
):
    """
    Every rank must come back with the SAME totals, and those totals must be the union over all
    shards. The rank-zero version could not do this even if it had not hung -- it would have
    reported whatever rank zero happened to hold.

    THE EXPECTED COUNT IS THE HAND-COMPUTED ONE from the aggregate fixture's arithmetic: 256
    scored tokens over the four shards (96 + 64 + 64 + 32). Every token carries band 0's bit, so
    band 0's count must equal the aggregate count exactly -- which is also what proves the padding
    rows were excluded from the band tallies rather than counted into them.

    MUTATION THIS CATCHES: reducing sums but not counts (or vice versa); and counting padded rows,
    which would push both counts above 256 at world sizes that require filler.
    """
    results, errors, _ = run_sliced_on_ranks(
        monkeypatch, world_size=world_size, pairs=slice_pairs
    )
    assert not errors, f"ranks failed: { {r: repr(e) for r, e in errors.items()} }"

    first = results[0]
    assert first["aggregate"]["n"] == 256, (
        f"scored {first['aggregate']['n']} tokens, expected the fixture's 256 -- a larger number "
        "means padded rows were counted"
    )
    assert first["bands"]["0"]["n"] == 256, "every token carries band 0's bit"
    assert first["bands"]["0"]["sum"] == pytest.approx(first["aggregate"]["sum"], rel=1e-6)

    for rank, result in results.items():
        assert result["aggregate"]["n"] == first["aggregate"]["n"], f"rank {rank} disagrees"
        assert result["aggregate"]["ce"] == pytest.approx(first["aggregate"]["ce"], rel=1e-9)
        assert result["world_size"] == world_size
        # An untrained model over any token set scores near ln(vocab). The magnitude check that
        # has historically been the only one to catch anything.
        assert abs(result["aggregate"]["ce"] - math.log(VOCAB)) < 1.0


def test_a_band_nobody_holds_is_null_across_the_whole_world(monkeypatch, tmp_path):
    """
    An empty band must survive the REDUCTION as null, not just the local computation. With bit 1
    set everywhere and no other bit set anywhere, bands 32/256/1024/4096 are empty on every rank,
    so their reduced count is 0 and their CE must be null rather than a perfect 0.0.

    MUTATION THIS CATCHES: `total / max(n, 1)` after the reduction, which gives four bands a
    0.0 CE -- and 0.0 sorts to the top of a table arms are ranked in.
    """
    pairs = []
    for i, n in enumerate(SHARD_TOKENS):
        shard = write_shard(tmp_path / "b" / f"s{i}.u32le.bin", n)
        mask = write_mask(tmp_path / "b" / f"s{i}.mask.u8", n, bit=1)
        pairs.append((shard, mask))

    results, errors, _ = run_sliced_on_ranks(monkeypatch, world_size=3, pairs=pairs)
    assert not errors, f"ranks failed: { {r: repr(e) for r, e in errors.items()} }"

    for rank, result in results.items():
        assert result["bands"]["0"]["ce"] is not None, f"rank {rank} lost the measured band"
        for band in ("32", "256", "1024", "4096"):
            assert result["bands"][band]["n"] == 0
            assert result["bands"][band]["ce"] is None, (
                f"rank {rank} band {band} reported {result['bands'][band]['ce']!r} for a band "
                "with no tokens; an unmeasured band must not score"
            )


def test_one_rank_with_a_bad_mask_length_fails_every_rank_rather_than_hanging(
    monkeypatch, tmp_path
):
    """
    THE SPLIT THAT WOULD DEADLOCK. The mask/shard length check compares two files ONE rank holds,
    so it can be true on that rank alone. A bare `raise` there unwinds that rank while its peers
    enter the step-budget all-reduce -- a mismatched collective, which is a hang or an NCCL abort
    rather than an error anyone can read.

    So the flag is all-reduced and every rank refuses together. Rank 1 gets the short mask here.

    MUTATION THIS CATCHES: turning the reduced flag back into a direct `raise` inside the loop.
    The FakeGroup then reports that the other ranks waited at a collective rank 1 never entered.
    """
    pairs = []
    for i, n in enumerate(SHARD_TOKENS):
        shard = write_shard(tmp_path / "m" / f"s{i}.u32le.bin", n)
        # Index 1 goes to rank 1 at world_size 2; give it a mask one byte short.
        length = n - 1 if i == 1 else n
        mask = write_mask(tmp_path / "m" / f"s{i}.mask.u8", length, bit=1)
        pairs.append((shard, mask))

    results, errors, _ = run_sliced_on_ranks(monkeypatch, world_size=2, pairs=pairs)

    assert not results, "no rank may return a number when the token set is mislabelled"
    assert set(errors) == {0, 1}, f"both ranks must refuse, got {sorted(errors)}"
    for rank, error in errors.items():
        assert isinstance(error, SystemExit), f"rank {rank} raised {error!r}"
    # The rank that OWNS the bad mask says which file; the other says a peer did.
    assert "mask/shard length mismatch" in str(errors[1])
    assert "another rank" in str(errors[0])


def test_shards_too_short_for_a_window_are_refused_rather_than_scoring_nothing(
    monkeypatch, tmp_path
):
    """
    Zero agreed steps means nothing to score, and the old code would have returned an aggregate of
    0.0/0 through `max(n, 1)`. A refusal instead, on every rank, so a run cannot report a sliced
    CE of zero.

    MUTATION THIS CATCHES: removing the `steps == 0` refusal, or restoring `max(agg_n, 1)` on the
    aggregate divide.
    """
    pairs = []
    for i in range(2):
        # 10 tokens at seq_len 32 yields (10-1)//32 = 0 windows.
        shard = write_shard(tmp_path / "s" / f"s{i}.u32le.bin", 10)
        mask = write_mask(tmp_path / "s" / f"s{i}.mask.u8", 10, bit=1)
        pairs.append((shard, mask))

    results, errors, _ = run_sliced_on_ranks(monkeypatch, world_size=2, pairs=pairs)
    assert not results
    assert set(errors) == {0, 1}
    for error in errors.values():
        assert isinstance(error, SystemExit)
        assert "no window" in str(error)


def test_the_sliced_eval_has_no_rank_gate_left_in_it():
    """
    THE REGRESSION GUARD, ON THE SOURCE, AND IT IS DELIBERATELY NARROW. The thread tests above
    measure the participation structure and are the real check; this one exists because the defect
    being fixed was a specific two-line shape that a reviewer can reintroduce while reading the
    function as "secondary, so rank zero is fine".

    It reads the source of `evaluate_sliced` and of the block in `train()` that calls it. Neither
    may contain a rank comparison that guards work; the only permitted `get_rank()` use in the
    caller is the LOGGING gate, which formats already-reduced numbers.

    MUTATION THIS CATCHES: wrapping either the fetch or the evaluation in `if get_rank() == 0:`.
    """
    import ast
    import inspect
    import re

    # THE DOCSTRING IS STRIPPED BEFORE SCANNING, AND THAT IS NOT A LOOPHOLE. `evaluate_sliced`
    # DOCUMENTS the removed gate verbatim -- "It used to be `if get_rank() == 0:`" -- so a naive
    # grep over the source matches the explanation of the fix and fails on correct code. The
    # first version of this test did exactly that. Parsing and dropping the docstring scans the
    # CODE, which is what the property is about; deleting the sentence to make a grep pass would
    # have removed the best explanation in the function.
    source = inspect.getsource(entry.evaluate_sliced)
    tree = ast.parse(textwrap.dedent(source))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop the docstring, keep every statement
    code = "\n".join(ast.unparse(node) for node in body)

    # `rank` is assigned and used in messages, which is fine; a COMPARISON against 0 is not.
    found = re.search(r"(get_rank\(\)|\brank\b)\s*[=!]=\s*0", code)
    assert not found, (
        f"evaluate_sliced's body contains the rank comparison {found.group(0)!r}; under FSDP a "
        "rank-gated forward waits on all-gathers the other ranks never enter, which is the hang "
        "this rewrite removed"
    )
    # The collectives must not be inside a rank-dependent branch either, so assert positively
    # that the reduction and the barrier are present in the body that was just scanned.
    assert "all_reduce_value" in code and "barrier()" in code

    caller = inspect.getsource(entry.train)
    block = caller[caller.index("if opts.slice_mask_uri:") :]
    block = block[: block.index("summarise(")]
    gates = re.findall(r"if get_rank\(\) == 0:", block)
    assert len(gates) == 1, (
        f"expected exactly one rank gate in the sliced-eval block (the log formatting), found "
        f"{len(gates)}; a gate around the fetch or the evaluation is the hang"
    )
    # And the one that exists must guard logging, not computation.
    after = block[block.index("if get_rank() == 0:") :]
    assert "log.info" in after.split("\n")[1] or "log.info" in after[:400]
    assert "evaluate_sliced(" not in after, "the rank gate must not contain the evaluation"
    assert "fetch_slice_inputs" not in after, "the rank gate must not contain the fetch"
