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

DOMAIN_TOKENS = ("<facts>", "<mano>", "<compare>")
SEED = 1234


def assemble(n_entities: int = 3_000):
    """A vocabulary and table that the reasoning tasks can be built against."""
    literals = Rn.literal_words_of(Rn.BIOS_TEMPLATES)
    task_words = T.all_required_words((T.ManoTask, T.CompareTask))
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
