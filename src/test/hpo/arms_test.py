import pytest

from olmo_core.hpo.arms import (
    EQUAL_BUDGET_CATEGORIES,
    Arm,
    BudgetLedger,
    ablation_matrix,
    equal_budget,
)


def test_all_preregistered_arms_are_registered():
    names = {a.value for a in Arm}
    for expected in (
        "random_fixed_recipe",
        "random_dynamic_schedule",
        "random_token_screen",
        "ifbo_official",
        "ipbt_reference",
        "ipbt_ifbo",
        "novel_aggregate_btt",
        "cma_only",
        "config_only_centaur",
        "multi_action_advisor",
        "frozen_layer_proxy",
        "umup_proxy",
    ):
        assert expected in names


def test_llm_arms_are_identifiable_in_the_matrix():
    matrix = ablation_matrix()
    assert matrix[Arm.CMA_ONLY]["llm_ratio"] == 0.0
    # config-only vs multi-action differ in the action scope the LLM may take.
    assert matrix[Arm.CONFIG_ONLY_CENTAUR]["llm_scope"] == "config_only"
    assert matrix[Arm.MULTI_ACTION_ADVISOR]["llm_scope"] == "multi_action"


def test_budget_ledger_counts_every_category_and_totals():
    ledger = BudgetLedger()
    for cat in EQUAL_BUDGET_CATEGORIES:
        ledger.charge(cat, accelerator_seconds=10.0, tokens=1000)
    ledger.assert_all_categories_present()
    assert ledger.total_accelerator_seconds == 10.0 * len(EQUAL_BUDGET_CATEGORIES)
    assert ledger.total_tokens == 1000 * len(EQUAL_BUDGET_CATEGORIES)


def test_budget_ledger_fails_if_a_category_is_uncounted():
    ledger = BudgetLedger()
    ledger.charge("main", accelerator_seconds=1.0, tokens=1)  # only one category
    with pytest.raises(ValueError):
        ledger.assert_all_categories_present()


def test_budget_ledger_rejects_unknown_category():
    ledger = BudgetLedger()
    with pytest.raises(ValueError):
        ledger.charge("not_a_category", accelerator_seconds=1.0, tokens=1)


def test_budget_ledger_rejects_invalid_resources_and_incomplete_token_categories():
    ledger = BudgetLedger()
    for seconds, tokens in ((-1.0, 1), (float("nan"), 1), (1.0, -1), (1.0, 1.5)):
        with pytest.raises(ValueError):
            ledger.charge("main", accelerator_seconds=seconds, tokens=tokens)

    for category in EQUAL_BUDGET_CATEGORIES:
        ledger.accelerator_seconds[category] = 0.0
    with pytest.raises(ValueError):
        ledger.assert_all_categories_present()


def test_equal_budget_comparison_within_tolerance():
    a = BudgetLedger()
    b = BudgetLedger()
    for cat in EQUAL_BUDGET_CATEGORIES:
        a.charge(cat, accelerator_seconds=100.0, tokens=1000)
        b.charge(cat, accelerator_seconds=102.0, tokens=1000)
    assert equal_budget(a, b, rel_tol=0.05) is True
    assert equal_budget(a, b, rel_tol=0.001) is False


def test_equal_budget_rejects_incomplete_or_token_unequal_ledgers():
    with pytest.raises(ValueError):
        equal_budget(BudgetLedger(), BudgetLedger())

    a = BudgetLedger()
    b = BudgetLedger()
    for category in EQUAL_BUDGET_CATEGORIES:
        a.charge(category, accelerator_seconds=1.0, tokens=1)
        b.charge(category, accelerator_seconds=1.0, tokens=1000)
    assert equal_budget(a, b) is False


def test_prepopulated_invalid_ledgers_cannot_compare_equal():
    invalid = BudgetLedger(
        accelerator_seconds={category: -1.0 for category in EQUAL_BUDGET_CATEGORIES},
        tokens={category: -1 for category in EQUAL_BUDGET_CATEGORIES},
    )
    with pytest.raises(ValueError):
        invalid.assert_all_categories_present()
    with pytest.raises(ValueError):
        equal_budget(invalid, invalid)
