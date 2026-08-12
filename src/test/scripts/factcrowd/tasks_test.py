"""
What the reasoning slices guarantee: correct answers, a fixed width, and a floor that was measured.

The arithmetic is re-derived from the rendered tokens rather than compared against the generator's own
bookkeeping. That is the point: iGSM produced four nulls in this programme because the eval graded a
single integer while discarding the derivation, so here the test decodes the expression the model
would actually see and evaluates it independently.

The degenerate baseline is measured, not assumed, and measuring it already changed the task. With a
free choice of operand, multiplication by zero is an absorbing state, so ``<n0>`` came out 8.3% of the
time -- a worse floor than the 6.8% Physics 4.1 reports, meaning a weaker instrument. Excluding zero
as a multiplicand brings it to the uniform 4.6%.
"""

import numpy as np
import pytest
from factcrowd.corpus import entities as E
from factcrowd.corpus import render as Rn
from factcrowd.corpus import tasks as T
from factcrowd.corpus import values as V
from factcrowd.corpus import vocab as Vo

from olmo_core.exceptions import OLMoConfigurationError

DOMAIN_TOKENS = ("<facts>", "<mano>", "<compare>", "<ctxmano>")
SEED = 1234


def assemble(n_entities: int = 3_000):
    """A vocabulary and table that the reasoning tasks can be built against."""
    literals = Rn.literal_words_of(Rn.BIOS_TEMPLATES)
    task_words = T.all_required_words((T.ManoTask, T.CompareTask, T.InContextManoTask))
    combined = tuple(literals) + tuple(task_words)
    schema = V.bios_schema(reserved=combined + Vo.SPECIALS + DOMAIN_TOKENS)
    vocabulary = Vo.Vocabulary.build(
        schema.schema, literal_words=combined, domain_tokens=DOMAIN_TOKENS
    )
    table = E.EntityTable.build(schema.schema, n_entities, SEED)
    return schema, vocabulary, table


def evaluate_expression(words) -> int:
    """Re-derive a Mano answer from the tokens the model sees, independently of the generator."""
    body = words[2 : words.index("<equals>")]
    residues = [int(word[2:-1]) for word in body if word.startswith("<n")]
    operators = [word for word in body if word in ("<plus>", "<times>")]
    total = residues[0]
    for operator, residue in zip(operators, residues[1:]):
        total = (
            (total + residue) % T.MANO_MODULUS
            if operator == "<plus>"
            else (total * residue) % T.MANO_MODULUS
        )
    return total


# --- Mano -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [2, 10, 13])
def test_every_mano_answer_survives_independent_re_evaluation(length):
    """
    The answer is checked by decoding the expression and evaluating it, not by trusting the generator.

    Run at the length the grid uses and at the extremes, because an off-by-one in the operator
    interleaving would be invisible at one length and wrong at another.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=length, seed=7)

    for index in range(1_000):
        item = task.item(index)
        words = vocabulary.decode(item.tokens)
        assert words[item.answer_start] == f"<n{evaluate_expression(words)}>"
        assert item.answer == (f"<n{evaluate_expression(words)}>",)


@pytest.mark.parametrize("length", [2, 5, 10, 13])
def test_mano_items_are_a_fixed_width_with_the_right_shape(length):
    """
    Fixed width is what lets the stream locate an item by division rather than by an offset index.

    Shape too: the expression must hold exactly ``length`` residues and ``length - 1`` operators, or
    the task is not the one the difficulty figures were measured on.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=length, seed=7)

    for index in range(200):
        item = task.item(index)
        words = vocabulary.decode(item.tokens)
        assert item.tokens.size == task.tokens_per_item
        body = words[2 : words.index("<equals>")]
        assert sum(1 for word in body if word.startswith("<n")) == length
        assert sum(1 for word in body if word in ("<plus>", "<times>")) == length - 1
        assert words[0] == "<mano>" and words[1] == Vo.BOS and words[-1] == Vo.EOS


def test_mano_never_multiplies_by_zero():
    """
    Multiplication by zero is an absorbing state, and it cost the endpoint its floor.

    With a free operand the answer was zero 8.3% of the time -- above the 6.8% Physics 4.1 reports, so
    a weaker instrument than the paper's. Excluding zero as a multiplicand is the whole fix.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)

    for index in range(2_000):
        words = vocabulary.decode(task.item(index).tokens)
        body = words[2 : words.index("<equals>")]
        for position, word in enumerate(body):
            if word == "<times>":
                assert body[position + 1] != "<n0>", (index, body)


def test_manos_degenerate_policy_is_near_the_uniform_floor():
    """
    Measured, because an endpoint whose score matches its baseline carries no signal.

    1/23 is 4.35%. Anything much above that is headroom the endpoint has lost, and the check that
    catches it has to be a measurement -- assuming a floor is how a previous eval came to report a
    number below its own.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)

    _, frequency = task.degenerate_answer(20_000)
    assert frequency < 0.055, frequency
    assert frequency > 1 / T.MANO_MODULUS - 0.01, frequency


def test_mano_answers_spread_over_the_residues():
    """A task answering only a few residues would be easier than its floor suggests."""
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)

    answers = {task.item(index).answer[0] for index in range(5_000)}
    assert len(answers) == T.MANO_MODULUS


def test_mano_items_do_not_repeat_and_depend_on_the_seed():
    """
    Per-example regeneration is why the slice carries nothing memorizable.

    A slice whose items recurred would offer facts to store, competing for the capacity the experiment
    measures -- which is the reason FLD's regeneration was called a feature rather than a limitation.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)
    other = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=8)

    rendered = {task.item(index).tokens.tobytes() for index in range(3_000)}
    assert len(rendered) > 2_900  # a handful of collisions is chance, not structure
    assert task.item(0).tokens.tobytes() != other.item(0).tokens.tobytes()
    assert task.fingerprint() != other.fingerprint()


def test_a_degenerate_mano_length_is_refused():
    """One residue is not an expression."""
    _, vocabulary, _ = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="at least 2"):
        T.ManoTask(vocabulary, domain_token="<mano>", length=1)


# --- the related-reasoning slice ---------------------------------------------------------------------


def test_compare_answers_the_lower_of_the_two_birth_years():
    """
    Two facts and an ordering, checked against the table rather than against the generator.

    This is the related-reasoning slice: it needs both birth years, so it is where crowding would show
    first if the mechanism is fact access rather than capacity.

    The answer is the earlier person's *year*, not their name, and that is the whole point. A name
    answer is a span of the prompt, so "always name the first person" scored 50.2% -- half the range of a
    binary endpoint, for a policy needing no facts at all. The year never appears in the prompt.
    """
    schema, vocabulary, table = assemble()
    column = schema.pool_index["birth_year"]
    task = T.CompareTask(
        table,
        schema,
        vocabulary,
        domain_token="<compare>",
        probe_ids=table.probe_ids,
        seed=3,
    )

    for index in range(1_000):
        item = task.item(index)
        words = vocabulary.decode(item.tokens)
        assert tuple(words[item.answer_start : item.answer_end]) == item.answer

        # Recover both names from the prompt, look up both years in the table, and confirm the answer
        # is the earlier year -- never reading the generator's own bookkeeping.
        name_width = len(schema.schema.names)
        first = tuple(words[3 : 3 + name_width])
        second = tuple(words[4 + name_width : 4 + 2 * name_width])
        pool = {p.name: p for p in schema.schema.attributes}[
            {v.name: v for v in schema.values}["birth_year"].pool_names[0]
        ]
        indices = []
        for entity_id in table.probe_ids:
            if table.name_parts(int(entity_id)) in (first, second):
                indices.append(int(table.attributes[int(entity_id)][column]))
        assert len(indices) == 2, (first, second)
        assert item.answer == (pool.values[min(indices)],)
        # One token wide, and it is not anywhere in the prompt -- so no copy policy can reach it.
        assert item.answer_end - item.answer_start == 1
        assert item.answer[0] not in words[: item.answer_start]


def test_compare_items_are_a_fixed_width_and_ask_about_two_distinct_people():
    """A comparison against oneself has no answer, so the pair must always be distinct."""
    schema, vocabulary, table = assemble()
    task = T.CompareTask(
        table, schema, vocabulary, domain_token="<compare>", probe_ids=table.probe_ids, seed=3
    )
    name_width = len(schema.schema.names)

    for index in range(1_000):
        item = task.item(index)
        words = vocabulary.decode(item.tokens)
        assert item.tokens.size == task.tokens_per_item
        first = tuple(words[3 : 3 + name_width])
        second = tuple(words[4 + name_width : 4 + 2 * name_width])
        assert first != second, index


def test_compare_only_asks_about_the_probe_subset():
    """
    Restricted so the population is the same 25k people in every cell.

    If it ranged over all entities, coverage per entity would swing 20x across the ladder and P4 would
    be comparing different populations.
    """
    schema, vocabulary, table = assemble()
    probe = table.probe_ids[:50]
    task = T.CompareTask(
        table, schema, vocabulary, domain_token="<compare>", probe_ids=probe, seed=3
    )
    allowed = {table.name_parts(int(entity_id)) for entity_id in probe}
    name_width = len(schema.schema.names)

    for index in range(500):
        words = vocabulary.decode(task.item(index).tokens)
        assert tuple(words[3 : 3 + name_width]) in allowed
        assert tuple(words[4 + name_width : 4 + 2 * name_width]) in allowed


def test_compares_floor_is_low_because_no_copy_policy_can_reach_the_answer():
    """
    The floor searched over both families -- constant answers *and* copies of prompt spans.

    Searching constants alone is how an endpoint loses its floor. When the answer was the earlier
    person's name it was a span of the prompt, so "copy the first name" was right 50.2% of the time
    while the best constant name managed 0.02%: a factor of 1,400, and half the range of a binary
    endpoint available to a policy that reads no facts. Any score under 50% would then have been below
    the endpoint's own floor, which is the reasoning-gym failure the module docstring cites.
    """
    schema, vocabulary, table = assemble()
    task = T.CompareTask(
        table, schema, vocabulary, domain_token="<compare>", probe_ids=table.probe_ids, seed=3
    )
    label, frequency = task.degenerate_baseline(3_000)
    assert frequency < 0.05, (label, frequency)
    # A copy policy must not be the winner, and no offset may beat the best constant by much: the
    # answer word is absent from the prompt by construction.
    assert label.startswith("constant:"), label


def test_mano_and_compare_floors_are_the_best_of_both_policy_families():
    """
    Whichever family wins, the reported floor is the maximum -- that is what a floor means.

    For Mano the two coincide near the uniform 1/23, which is exactly why searching constants alone
    looked adequate until a task arrived whose answer sat in its own prompt.
    """
    schema, vocabulary, table = assemble()
    mano = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)
    label, frequency = mano.degenerate_baseline(5_000)
    # A constant, because no offset in the expression predicts the answer: the best of ~20 offsets sits
    # at the uniform rate and does not clear the noise margin the estimator requires.
    assert label.startswith("constant:"), label
    assert 1 / T.MANO_MODULUS - 0.01 < frequency < 0.06, (label, frequency)
    # The constant-only figure can never exceed the full search.
    assert mano.degenerate_answer(5_000)[1] <= frequency + 1e-9

    compare = T.CompareTask(
        table, schema, vocabulary, domain_token="<compare>", probe_ids=table.probe_ids, seed=3
    )
    assert compare.degenerate_answer(2_000)[1] <= compare.degenerate_baseline(2_000)[1] + 1e-9


def in_context(length: int = 10, alphabet: int = 10, split: str = "train", seed: int = 0):
    """An in-context Mano task and the vocabulary it was built against."""
    _, vocabulary, _ = assemble()
    task = T.InContextManoTask(
        vocabulary,
        domain_token="<ctxmano>",
        length=length,
        alphabet=alphabet,
        seed=seed,
        split=split,
    )
    return task, vocabulary


def replay_in_context(words, alphabet: int, length: int) -> str:
    """
    Re-derive the answer from the tables **as rendered**, independently of the generator.

    The whole point of the in-context variant is that the answer is a function of the prompt, so the test
    that matters is the one that reads the prompt back and composes it by hand. A generator comparing
    against its own bookkeeping would pass even if it laid the table out in an order nothing could read.
    """
    symbol = {f"<n{value}>": value for value in range(alphabet)}
    cursor, cells = 2, {}
    for operator in range(2):
        assert words[cursor] == ("<plus>", "<times>")[operator]
        cursor += 1
        for left in range(alphabet):
            assert symbol[words[cursor]] == left, "each row must be labelled with its left operand"
            assert words[cursor + 1] == "<equals>"
            cursor += 2
            for right in range(alphabet):
                cells[(operator, left, right)] = symbol[words[cursor]]
                cursor += 1
    total = symbol[words[cursor]]
    cursor += 1
    for _ in range(length - 1):
        operator = 0 if words[cursor] == "<plus>" else 1
        total = cells[(operator, total, symbol[words[cursor + 1]])]
        cursor += 2
    assert words[cursor] == "<equals>"
    return f"<n{total}>"


@pytest.mark.parametrize("length", [2, 4, 10])
def test_in_context_mano_answers_are_composed_from_the_prompts_own_table(length):
    """The answer is a function of the tokens the model sees, which is the whole claim of this variant."""
    task, vocabulary = in_context(length=length)
    for index in range(120):
        item = task.item(index)
        words = vocabulary.decode(item.tokens)
        assert replay_in_context(words, task.alphabet, length) == item.answer[0]
        assert item.answer_end - item.answer_start == 1
        assert len(item.tokens) == task.tokens_per_item


def test_in_context_mano_draws_a_fresh_table_every_item():
    """
    Nothing about the mapping is memorisable, which is what removes the table-eviction confound.

    :class:`ManoTask` needs its mod-23 tables in the weights, so a decline under fact load is equally
    explained by the facts having evicted them -- knowledge versus knowledge, not reasoning. Here a model
    that had stored every table it ever saw still could not answer.
    """
    task, vocabulary = in_context()
    tables = {
        tuple(vocabulary.decode(task.item(index).tokens)[2 : 2 + task.table_tokens])
        for index in range(600)
    }
    assert len(tables) == 600


def test_in_context_mano_splits_are_content_disjoint_without_rejection_sampling():
    """
    The fresh table buys disjointness that :class:`ManoTask` needs 64 redraws to approximate.

    Two items sharing an expression do not share an answer unless they also share ``2 * k**2`` cells, so
    there is nothing to reject.
    """
    train, vocabulary = in_context(split="train")
    evaluation, _ = in_context(split="eval")
    seen = {tuple(vocabulary.decode(train.item(index).tokens)[2:]) for index in range(3_000)}
    other = {tuple(vocabulary.decode(evaluation.item(index).tokens)[2:]) for index in range(3_000)}
    assert not seen & other


@pytest.mark.parametrize("length", [2, 4, 10])
def test_in_context_mano_answers_are_uniform_so_the_constant_floor_is_one_over_k(length):
    """
    Composition off a uniformly random table is uniform at every depth, so no constant beats ``1 / k``.

    A row of a uniform table is k iid uniform values, and each operand is an independent uniform draw, so
    ``total <- table[op][total][operand]`` is uniform whatever ``total`` was. This is what makes the floor
    a property of the construction rather than a fact about one seed.
    """
    task, _ = in_context(length=length)
    counts: dict = {}
    for index in range(6_000):
        answer = task.item(index).answer[0]
        counts[answer] = counts.get(answer, 0) + 1
    assert len(counts) == task.alphabet
    share = [count / 6_000 for count in counts.values()]
    assert max(share) - min(share) < 0.02, counts


def test_in_context_mano_floor_exceeds_one_over_k_because_a_copy_can_land_on_the_answer_cell():
    """
    The measured floor is **10.45%** at k=10, not 10.0%, and an earlier docstring here claimed 10.0%.

    A fixed-offset copy is right whenever the cell it reads equals the answer, and once in ``2 * k**2``
    items that offset *is* the answer's own cell: ``1/(2k**2) + (1 - 1/(2k**2))/k``. Small, real, and the
    reason the floor is measured rather than argued.
    """
    task, _ = in_context()
    predicted = 1 / (2 * task.alphabet**2) + (1 - 1 / (2 * task.alphabet**2)) / task.alphabet
    assert abs(predicted - 0.1045) < 1e-9
    _, floor = task.degenerate_baseline(12_000)
    # Three standard errors of the held-out half at this rate is about 0.8pp.
    assert abs(floor - predicted) < 0.012, floor
    assert (
        floor < 0.115
    ), "a floor drifting with the offsets searched is selection bias, not a floor"


def test_in_context_mano_fits_the_instance_and_refuses_an_alphabet_that_does_not():
    """
    ``k`` is bounded by the prompt: the tables cost ``2 * k * (2 + k)`` tokens against 2 per unit of depth.

    k=10 fits the whole calibration ladder at 266 tokens of 512; k=16 needs 602 and is refused rather than
    silently truncated, which is what an instance-length overflow would otherwise become.
    """
    _, vocabulary, _ = assemble()
    widths = {}
    for alphabet in (8, 10, 12):
        task = T.InContextManoTask(
            vocabulary, domain_token="<ctxmano>", length=10, alphabet=alphabet
        )
        widths[alphabet] = task.tokens_per_item
        assert task.tokens_per_item <= 512
        assert task.table_tokens == 2 * (1 + alphabet * (2 + alphabet))
    assert widths == {8: 186, 10: 266, 12: 362}

    with pytest.raises(OLMoConfigurationError, match="over the 512-token instance"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=10, alphabet=16)
    with pytest.raises(OLMoConfigurationError, match="'alphabet' must be between 2 and 23"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=4, alphabet=24)
    with pytest.raises(OLMoConfigurationError, match="'length' must be at least 2"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=1)


def test_in_context_items_are_padded_to_tile_the_instance():
    """
    **Half the in-context items would be cut by an instance boundary without this**, and a cut item is
    worse than a missing one: the instance opens mid-table or loses the answer.

    The trainer concatenates a slice and chunks it into 512-token windows, so an item of width ``w`` is cut
    unless ``w`` divides 512 -- 3.1% of 24-token ``<mano>`` items, the figure recorded on
    ``TaskStream.num_tokens``, but 266/512 = **52%** of in-context ones. At k=10 the natural widths at
    lengths 2 to 5 are 250, 252, 254 and 256, so padding to 256 aligns the whole ladder for at most 2.3% of
    tokens.
    """
    _, vocabulary, _ = assemble()
    pad_id = vocabulary.id_of(Vo.PAD)
    for length, natural, padding in ((2, 250, 6), (3, 252, 4), (4, 254, 2), (5, 256, 0)):
        task = T.InContextManoTask(
            vocabulary, domain_token="<ctxmano>", length=length, alphabet=10, pad_to=256
        )
        assert (task.natural_width, task.padding) == (natural, padding)
        assert task.tokens_per_item == 256
        assert 512 % task.tokens_per_item == 0, "an unaligned width is the whole defect"
        item = task.item(11)
        # Padding sits after the eos, so the answer's offset is untouched and every scorer keeps working.
        assert item.answer_start == natural - 2
        assert replay_in_context(vocabulary.decode(item.tokens), 10, length) == item.answer[0]
        if padding:
            assert list(item.tokens[natural:]) == [pad_id] * padding
            assert pad_id not in item.tokens[:natural]

    # Padding must not move the floor: a pad token never equals a residue answer.
    unpadded = T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=4, alphabet=10)
    padded = T.InContextManoTask(
        vocabulary, domain_token="<ctxmano>", length=4, alphabet=10, pad_to=256
    )
    assert (
        abs(unpadded.degenerate_baseline(8_000)[1] - padded.degenerate_baseline(8_000)[1]) < 0.015
    )

    with pytest.raises(OLMoConfigurationError, match="already 254 tokens"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=4, alphabet=10, pad_to=200)
    with pytest.raises(OLMoConfigurationError, match="must divide the 512-token instance"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=4, alphabet=10, pad_to=300)


def test_in_context_mano_fingerprint_moves_with_alphabet_and_length_and_split():
    """A digest that ignores the alphabet would call two different endpoints the same corpus."""
    base, _ = in_context()
    variants = [
        in_context(length=4)[0],
        in_context(alphabet=8)[0],
        in_context(split="eval")[0],
        in_context(seed=1)[0],
        T.InContextManoTask(assemble()[1], domain_token="<ctxmano>", length=10, pad_to=512),
    ]
    digests = {base.fingerprint()} | {task.fingerprint() for task in variants}
    assert len(digests) == 1 + len(variants)
    # Structure excludes seed and split, so those two must collide there and the other two must not.
    assert in_context(split="eval")[0].structure_fingerprint() == base.structure_fingerprint()
    assert in_context(seed=1)[0].structure_fingerprint() == base.structure_fingerprint()
    assert in_context(alphabet=8)[0].structure_fingerprint() != base.structure_fingerprint()


def test_in_context_mano_keeps_manos_vocabulary_so_the_two_endpoints_are_comparable():
    """
    Same words, so same softmax width and same parameter count.

    The plan runs in-context as the confirmatory endpoint and memorised Mano as a secondary. If the two
    needed different vocabularies they would be different architectures, and the comparison between them
    would carry a width confound rather than the reading it is there to provide.
    """
    assert T.InContextManoTask.required_words() == T.ManoTask.required_words() == T.mano_words()


def test_compare_refuses_a_schema_without_an_ordinal_field():
    """
    The entropy axis has six positional attributes of four words each and no field to order.

    So that axis carries unrelated reasoning alone, and the refusal here is what makes that explicit
    rather than silently comparing on an arbitrary sub-pool.
    """
    entropy = V.entropy_schema(8, reserved=Vo.SPECIALS + DOMAIN_TOKENS)
    vocabulary = Vo.Vocabulary.build(entropy.schema, domain_tokens=DOMAIN_TOKENS)
    table = E.EntityTable.build(entropy.schema, 100, SEED)
    with pytest.raises(OLMoConfigurationError, match="not an attribute of this schema"):
        T.CompareTask(
            table,
            entropy,
            vocabulary,
            domain_token="<compare>",
            probe_ids=table.probe_ids,
            seed=3,
        )


def test_compare_refuses_a_probe_subset_too_small_to_pair():
    """One entity cannot be compared with anything."""
    schema, vocabulary, table = assemble(100)
    with pytest.raises(OLMoConfigurationError, match="at least two entities"):
        T.CompareTask(
            table,
            schema,
            vocabulary,
            domain_token="<compare>",
            probe_ids=table.probe_ids[:1],
            seed=3,
        )


# --- the stream --------------------------------------------------------------------------------------


def test_the_stream_holds_a_whole_number_of_items():
    """
    Rounded down, because a truncated item has a truncated answer.

    That is exactly the failure that made a previous deduction eval score below its own floor: a
    derivation cut short parsed as a wrong answer rather than as an unparseable one.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)
    stream = T.TaskStream(task, num_tokens=10_000)

    assert stream.num_tokens % task.tokens_per_item == 0
    assert stream.num_tokens <= 10_000
    assert stream.n_items == 10_000 // task.tokens_per_item


@pytest.mark.parametrize("width", [1, 7, 24, 512])
def test_consecutive_ranges_reconstruct_the_stream(width):
    """
    Chunking at widths shorter and longer than an item, against a brute-force render.

    This is what ConcatAndChunkInstanceSource does, so a range that began mid-item incorrectly would
    corrupt exactly the instances the trainer sees.
    """
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)
    stream = T.TaskStream(task, num_tokens=24 * 60)

    expected = np.concatenate([task.item(index).tokens for index in range(stream.n_items)])
    pieces = []
    for start in range(0, stream.num_tokens, width):
        end = min(start + width, stream.num_tokens)
        piece = stream.tokens(start, end)
        assert piece.size == end - start
        pieces.append(piece)
    np.testing.assert_array_equal(np.concatenate(pieces), expected)


def test_the_stream_refuses_an_out_of_range_or_empty_request():
    """Reading past the end would silently repeat the last item."""
    _, vocabulary, _ = assemble(50)
    stream = T.TaskStream(
        T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7), num_tokens=2_400
    )
    with pytest.raises(OLMoConfigurationError, match="empty or inverted"):
        stream.tokens(5, 5)
    with pytest.raises(OLMoConfigurationError, match="out of bounds"):
        stream.tokens(0, stream.num_tokens + 1)


def test_a_slice_too_small_for_one_item_is_refused():
    """A slice of no items is a config error, not an empty mixture component."""
    _, vocabulary, _ = assemble(50)
    task = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=7)
    with pytest.raises(OLMoConfigurationError, match="cannot hold one"):
        T.TaskStream(task, num_tokens=task.tokens_per_item - 1)


def test_task_words_are_taken_from_the_classes_before_anything_is_built():
    """
    The ordering the vocabulary forces: reserve first, construct second.

    A task cannot be built without a vocabulary holding its tokens, and the pool allocator has to avoid
    them too -- otherwise a generated city name could collide with an operator.
    """
    words = T.all_required_words((T.ManoTask, T.CompareTask))
    assert "<n0>" in words and "<plus>" in words and "<equals>" in words
    assert "Between" in words and "earlier" in words
    assert len(words) == len(set(words))


def test_the_same_reasoning_items_appear_in_every_cell():
    """
    Section 3.4's invariant: the reasoning slice is identical across cells, not merely equal in volume.

    Generated from a fixed seed rather than materialised, so identity holds by construction -- but only
    as *items*. Token ids necessarily differ wherever the schema differs, since the vocabulary does, so
    the invariant is checked on the decoded words. A difference in reasoning score then cannot be a
    difference in which problems were asked.
    """
    bios_schema, bios_vocabulary, _ = assemble(50)
    entropy = V.entropy_schema(
        8, reserved=tuple(T.all_required_words((T.ManoTask,))) + Vo.SPECIALS + DOMAIN_TOKENS
    )
    entropy_vocabulary = Vo.Vocabulary.build(
        entropy.schema,
        literal_words=tuple(T.all_required_words((T.ManoTask,))),
        domain_tokens=DOMAIN_TOKENS,
    )

    # Two cells on the count axis share a schema, so even the ids match.
    left = T.ManoTask(bios_vocabulary, domain_token="<mano>", length=10, seed=1234 + 4)
    right = T.ManoTask(bios_vocabulary, domain_token="<mano>", length=10, seed=1234 + 4)
    for index in range(200):
        np.testing.assert_array_equal(left.item(index).tokens, right.item(index).tokens)

    # A cell on the entropy axis has a different vocabulary, so only the words match.
    across = T.ManoTask(entropy_vocabulary, domain_token="<mano>", length=10, seed=1234 + 4)
    assert entropy_vocabulary.size != bios_vocabulary.size  # otherwise this proves nothing
    for index in range(200):
        assert bios_vocabulary.decode(left.item(index).tokens) == entropy_vocabulary.decode(
            across.item(index).tokens
        )


def test_the_eval_split_shares_no_item_with_training():
    """
    The defect this keying exists to prevent, and it was total rather than partial.

    The first version mixed ``index ^ seed``, which makes the item set independent of the seed: seeds
    1238 and 1241 differ by 15, so ``item(i)`` under one equalled ``item(i ^ 15)`` under the other and
    every one of 2,000 checked items matched. An evaluation set drawn that way is 100% leaked however
    the seed is chosen, and it looks completely ordinary.

    Now the split is part of the key, so disjointness does not depend on choosing a lucky seed -- it
    holds even when the seeds are identical, which is what this checks.
    """
    _, vocabulary, table = assemble(3_000)
    train = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=1238, split="train")
    other_seed = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=1241, split="eval")
    same_seed = T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=1238, split="eval")

    seen = {train.item(i).tokens.tobytes() for i in range(60_000)}
    for held_out in (other_seed, same_seed):
        overlap = sum(1 for i in range(4_000) if held_out.item(i).tokens.tobytes() in seen)
        assert overlap == 0, overlap
    # The old translation attack, explicitly.
    assert all(
        other_seed.item(i).tokens.tobytes() != train.item(i ^ 15).tokens.tobytes()
        for i in range(500)
    )
    # And the split reaches the fingerprint, so a checkpoint cannot claim the wrong one.
    assert train.fingerprint() != same_seed.fingerprint()


def test_the_compare_eval_split_repeats_a_pair_only_by_chance():
    """
    The related slice cannot be *keyed* into the training set, but it can collide with it.

    Mano draws from ~2^54 expressions, so a held-out item is new with probability ~1. Compare draws
    pairs from a finite probe subset, so some eval pair will have been asked in training however the
    keying works -- that is a property of the population, not of the generator, and it is the 0.84%
    contamination the grid's real budget implies over 25,000 probe entities.

    What the keying has to guarantee is that the overlap is *chance*, not structure. Here 3,000 probe
    entities give 4.5M unordered pairs against 40,000 training items, so the birthday expectation is
    about 0.9% and anything near 100% would mean the splits share a stream.
    """
    schema, vocabulary, table = assemble(3_000)
    kwargs = dict(domain_token="<compare>", probe_ids=table.probe_ids, seed=11)
    train = T.CompareTask(table, schema, vocabulary, split="train", **kwargs)
    held = T.CompareTask(table, schema, vocabulary, split="eval", **kwargs)

    seen = {train.item(i).tokens.tobytes() for i in range(40_000)}
    overlap = sum(1 for i in range(3_000) if held.item(i).tokens.tobytes() in seen) / 3_000
    assert overlap < 0.05, overlap  # chance is ~0.9%; the old keying gave 1.0

    # The draws themselves share nothing, which is the part the keying is responsible for.
    train_keys = {
        T.item_key(class_tag=0x434D5052, split="train", seed=11, index=i) for i in range(40_000)
    }
    assert not any(
        T.item_key(class_tag=0x434D5052, split="eval", seed=11, index=i) in train_keys
        for i in range(3_000)
    )


def test_an_unknown_split_is_refused():
    """A typo'd split would otherwise generate a third, silently-overlapping stream."""
    _, vocabulary, _ = assemble(50)
    with pytest.raises(OLMoConfigurationError, match="unknown split"):
        T.ManoTask(vocabulary, domain_token="<mano>", length=10, split="dev")


def test_the_two_tasks_do_not_share_a_stream_even_at_one_seed():
    """
    The class tag separates them, so Mano and Compare at the same seed and split are unrelated.

    Without it two tasks would walk the same 64-bit sequence, which is not wrong so much as an
    unnecessary coupling between two endpoints that are meant to be independent.
    """
    keys = {
        T.item_key(class_tag=tag, split="train", seed=7, index=i)
        for tag in (0x4D414E4F, 0x434D5052)
        for i in range(200)
    }
    assert len(keys) == 400


def test_an_item_wider_than_the_instance_is_refused_for_either_task():
    """
    Every item would be cut by a boundary, so the slice would be entirely truncated expressions.

    ``InContextManoTask`` refused this from the start, because its width is dominated by a table whose size
    is a config knob. ``ManoTask`` did not, so a length above 254 produced 514-token items the chunker cut in
    half -- unnoticed precisely because nobody would ask for it on purpose.
    """
    _, vocabulary, _ = assemble()
    widest = T.ManoTask(vocabulary, domain_token="<mano>", length=254)
    assert widest.tokens_per_item == 512
    with pytest.raises(OLMoConfigurationError, match="over the 512-token instance"):
        T.ManoTask(vocabulary, domain_token="<mano>", length=255)
    with pytest.raises(OLMoConfigurationError, match="over the 512-token instance"):
        T.InContextManoTask(vocabulary, domain_token="<ctxmano>", length=4, alphabet=16)


def test_padding_leaves_the_answer_where_every_scorer_looks_for_it():
    """
    Padding goes after the eos, so ``answer_start`` is a function of the natural width and nothing else.
    A padded and an unpadded task must agree on the answer *and* its offset, or the label-shift rule that
    charges ``ce_loss[p - 1]`` for token ``p`` would read the wrong position.
    """
    _, vocabulary, _ = assemble()
    for length in (2, 6, 10):
        bare = T.ManoTask(vocabulary, domain_token="<mano>", length=length, seed=5)
        padded = T.ManoTask(vocabulary, domain_token="<mano>", length=length, pad_to=32, seed=5)
        for index in range(60):
            one, two = bare.item(index), padded.item(index)
            assert one.answer == two.answer
            assert one.answer_start == two.answer_start
            assert list(one.tokens) == list(two.tokens[: bare.tokens_per_item])
