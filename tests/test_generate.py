"""The regression tests the previous generation lacked.

`test_batching_never_changes_a_generation` is the load-bearing one. The old
harness left-padded prompts to the batch maximum with EOT and attended over the
pads; the same checkpoint then scored 4.7% batched against 93.8% unbatched. The
old test suite could not catch it because its stubs ignored the prompt, and its
docstring described "the full (left-padded) prompt" as the contract.

`PrefixSensitiveStub` below is sensitive to exactly what padding destroys: the
identity of the *first* prompt token. Any decoder that pads on the left will fail
these tests.
"""

import torch

from memsplit.generate import generate_batch_with_stats
from memsplit.store import Organizer
from memsplit.tokenizer import get_tok

TOK = get_tok("byte")
CPU = torch.device("cpu")


class PrefixSensitiveStub:
    """Emits a continuation determined by the prompt's FIRST token.

    On prefill it reads `idx[:, 0]` and derives a per-row program from it. A
    left-padded batch would present EOT as the first token for every short row,
    collapsing all their programs together -- which is precisely the corruption
    mode being guarded against.
    """

    def __init__(self, table: dict[int, list[int]], vocab: int = TOK.VOCAB_SIZE):
        self.table = table
        self.V = vocab
        self.device = CPU

    def forward_step(self, idx, cache):
        B, T = idx.shape
        if cache is None:
            first = idx[:, 0].tolist()
            cache = {"progs": [list(self.table.get(f, [TOK.EOT])) for f in first], "step": 0}
        else:
            cache["step"] += 1
        logits = torch.zeros(B, T, self.V)
        for b, prog in enumerate(cache["progs"]):
            i = cache["step"]
            fav = prog[i] if i < len(prog) else TOK.EOT
            logits[b, -1, fav] = 1.0
        return logits, cache


class EchoLastStub:
    """Closed loop: favours `table[last_input_token]`.

    Proves forced store tokens really pass through `forward_step` -- a
    continuation reachable only via a `<|db_end|>` input cannot be produced
    unless the forced tokens were actually fed back.
    """

    def __init__(self, table: dict[int, int], vocab: int = TOK.VOCAB_SIZE):
        self.table = table
        self.V = vocab
        self.device = CPU

    def forward_step(self, idx, cache):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.V)
        last = idx[:, -1].tolist()
        for b, tid in enumerate(last):
            logits[b, -1, self.table.get(tid, TOK.EOT)] = 1.0
        return logits, cache or {"seen": True}


def _prog(text: str) -> list[int]:
    return TOK.encode(text) + [TOK.EOT]


def test_batching_never_changes_a_generation():
    """Mixed-length prompts batched together == each decoded alone."""
    prompts = ["a", "bb", "ccc", "d", "eeeeeeee", "ff"]
    table = {TOK.encode(p)[0]: _prog(f"->{p}") for p in prompts}

    batched, _ = generate_batch_with_stats(
        PrefixSensitiveStub(table), TOK, prompts, 24, None, CPU
    )
    singles = [
        generate_batch_with_stats(PrefixSensitiveStub(table), TOK, [p], 24, None, CPU)[0][0]
        for p in prompts
    ]
    assert batched == singles, f"batching changed generations:\n{batched}\n{singles}"


def test_generation_is_order_invariant():
    """Permuting the input must permute the output and nothing else."""
    prompts = ["a", "bb", "ccc", "dddd"]
    table = {TOK.encode(p)[0]: _prog(f"={p}") for p in prompts}
    fwd, _ = generate_batch_with_stats(
        PrefixSensitiveStub(table), TOK, prompts, 24, None, CPU
    )
    rev, _ = generate_batch_with_stats(
        PrefixSensitiveStub(table), TOK, prompts[::-1], 24, None, CPU
    )
    assert fwd == rev[::-1]


def test_group_size_does_not_change_output():
    prompts = ["aa", "bb", "cc", "dd", "ee"]
    table = {TOK.encode(p)[0]: _prog(f"#{p}") for p in prompts}
    whole, _ = generate_batch_with_stats(
        PrefixSensitiveStub(table), TOK, prompts, 24, None, CPU
    )
    split, _ = generate_batch_with_stats(
        PrefixSensitiveStub(table), TOK, prompts, 24, None, CPU, max_group_size=2
    )
    assert whole == split


def test_forced_value_is_fed_back_through_the_model():
    """A continuation reachable only after <|db_end|> proves feedback."""
    org = Organizer()
    org.add("Ada", "city", "Paris")

    # Model programme: emit <|db_start|>, the key, <|db_retrieve|>, then whatever
    # the store forces. After <|db_end|> arrives as *input*, emit '!'.
    key_ids = TOK.encode("Ada, city")
    chain = {TOK.encode("q")[0]: [TOK.DB_START, *key_ids, TOK.DB_RETRIEVE]}

    class Prog(PrefixSensitiveStub):
        def forward_step(self, idx, cache):
            logits, cache = super().forward_step(idx, cache)
            last = idx[:, -1].tolist()
            for b, tid in enumerate(last):
                if tid == TOK.DB_END:
                    logits[b, -1, :] = 0.0
                    logits[b, -1, TOK.encode("!")[0]] = 1.0
            return logits, cache

    texts, stats = generate_batch_with_stats(Prog(chain), TOK, ["q"], 48, org, CPU)
    assert stats["n_query_spans"] == 1 and stats["n_hits"] == 1, stats
    assert "Paris" in texts[0], texts[0]
    assert texts[0].rstrip().endswith("!"), texts[0]


def test_store_miss_emits_nothing_and_returns_to_free():
    org = Organizer()
    org.add("Ada", "city", "Paris")
    key_ids = TOK.encode("Nobody, city")
    chain = {TOK.encode("q")[0]: [TOK.DB_START, *key_ids, TOK.DB_RETRIEVE, *TOK.encode("Z")]}
    texts, stats = generate_batch_with_stats(
        PrefixSensitiveStub(chain), TOK, ["q"], 48, org, CPU
    )
    assert stats["n_query_spans"] == 1 and stats["n_misses"] == 1 and stats["n_hits"] == 0, stats
    assert "Paris" not in texts[0]


def test_runaway_query_is_flagged_malformed():
    long_key = TOK.encode("x" * 200)
    chain = {TOK.encode("q")[0]: [TOK.DB_START, *long_key]}
    _, stats = generate_batch_with_stats(
        PrefixSensitiveStub(chain), TOK, ["q"], 220, Organizer(), CPU
    )
    assert stats["n_malformed"] >= 1, stats
    assert stats["n_lookups"] == 0, stats


def test_organizer_disabled_is_closed_book():
    org = Organizer()
    org.add("Ada", "city", "Paris")
    key_ids = TOK.encode("Ada, city")
    chain = {TOK.encode("q")[0]: [TOK.DB_START, *key_ids, TOK.DB_RETRIEVE]}
    texts, stats = generate_batch_with_stats(
        PrefixSensitiveStub(chain), TOK, ["q"], 48, None, CPU
    )
    # The query span is still counted -- that is the addressing rate -- but
    # nothing is resolved and nothing is injected.
    assert stats["n_query_spans"] == 1, stats
    assert stats["n_lookups"] == 0 and stats["n_hits"] == 0 and stats["n_misses"] == 0, stats
    assert "Paris" not in texts[0]
