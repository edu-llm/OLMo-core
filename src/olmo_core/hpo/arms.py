"""Preregistered three-arm HPO study and complete resource accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

__all__ = ["Arm", "EQUAL_BUDGET_CATEGORIES", "BudgetLedger", "equal_budget", "ablation_matrix"]


class Arm(str, Enum):
    """Registered HPO arms, including the curriculum extension."""

    FULL_ACRONYM_SOUP = "full_acronym_soup"
    NO_CENTAUR = "no_centaur"
    NO_PROXY = "no_proxy"
    CURRICULUM_QUADRATIC_MTLD = "curriculum_quadratic_mtld"
    CURRICULUM_QUADRATIC_MTLD_NO_CENTAUR = "curriculum_quadratic_mtld_no_centaur"
    OLMOE_NO_PROXY = "olmoe_no_proxy"
    OLMOE_NO_CENTAUR = "olmoe_no_centaur"


# Every category of compute that must be counted toward an arm's budget.
EQUAL_BUDGET_CATEGORIES = (
    "main",
    "failed",
    "screened",
    "reset",
    "retrained",
    "llm_controller",
    "proxy_selection",
)


@dataclass
class BudgetLedger:
    """Accumulates accelerator-seconds and tokens by category."""

    accelerator_seconds: Dict[str, float] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)

    def charge(self, category: str, *, accelerator_seconds: float, tokens: int) -> None:
        if category not in EQUAL_BUDGET_CATEGORIES:
            raise ValueError(
                f"unknown budget category {category!r}; must be one of {EQUAL_BUDGET_CATEGORIES}"
            )
        if not math.isfinite(accelerator_seconds) or accelerator_seconds < 0.0:
            raise ValueError("accelerator_seconds must be finite and non-negative")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        self.accelerator_seconds[category] = self.accelerator_seconds.get(category, 0.0) + float(
            accelerator_seconds
        )
        self.tokens[category] = self.tokens.get(category, 0) + tokens

    def assert_all_categories_present(self) -> None:
        missing = [
            category
            for category in EQUAL_BUDGET_CATEGORIES
            if category not in self.accelerator_seconds or category not in self.tokens
        ]
        if missing:
            raise ValueError(
                f"budget is missing categories {missing}; equal-budget comparisons must count "
                "failed, screened, reset, retrained, LLM-controller, and proxy-selection work"
            )
        unknown = (set(self.accelerator_seconds) | set(self.tokens)) - set(EQUAL_BUDGET_CATEGORIES)
        if unknown:
            raise ValueError(f"budget has unknown categories: {sorted(unknown)}")
        for category in EQUAL_BUDGET_CATEGORIES:
            seconds = self.accelerator_seconds[category]
            tokens = self.tokens[category]
            if not math.isfinite(seconds) or seconds < 0.0:
                raise ValueError(f"budget category {category} has invalid accelerator seconds")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise ValueError(f"budget category {category} has invalid tokens")

    @property
    def total_accelerator_seconds(self) -> float:
        return sum(self.accelerator_seconds.values())

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens.values())


def equal_budget(a: BudgetLedger, b: BudgetLedger, *, rel_tol: float = 0.05) -> bool:
    """Whether two complete arms match in accelerator-seconds and tokens."""
    if not math.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("rel_tol must be finite and non-negative")
    a.assert_all_categories_present()
    b.assert_all_categories_present()
    seconds_equal = math.isclose(
        a.total_accelerator_seconds, b.total_accelerator_seconds, rel_tol=rel_tol, abs_tol=0.0
    )
    tokens_equal = math.isclose(a.total_tokens, b.total_tokens, rel_tol=rel_tol, abs_tol=0.0)
    return seconds_equal and tokens_equal


def ablation_matrix() -> Dict[Arm, Dict[str, object]]:
    """Return the exact controller and model contract for each study arm."""

    shared = {
        "ftpfn": True,
        "ifbo": True,
        "ipbt": True,
        "btt_aggregate_restarts": True,
    }
    return {
        Arm.FULL_ACRONYM_SOUP: {
            **shared,
            "model_parameterization": "umup_190m_same_depth",
            "target_depth": 16,
            "freeze_first_n_blocks": 8,
            "llm_ratio": 0.3,
            "llm_scope": "multi_action",
        },
        Arm.NO_CENTAUR: {
            **shared,
            "model_parameterization": "umup_190m_same_depth",
            "target_depth": 16,
            "freeze_first_n_blocks": 8,
            "llm_ratio": 0.0,
            "llm_scope": "none",
        },
        Arm.NO_PROXY: {
            **shared,
            "model_parameterization": "stock_olmo2_190m",
            "target_depth": 12,
            "freeze_first_n_blocks": 0,
            "llm_ratio": 0.3,
            "llm_scope": "multi_action",
        },
        Arm.CURRICULUM_QUADRATIC_MTLD: {
            **shared,
            "model_parameterization": "stock_olmo2_190m",
            "target_depth": 12,
            "freeze_first_n_blocks": 0,
            "llm_ratio": 0.3,
            "llm_scope": "multi_action",
        },
        Arm.CURRICULUM_QUADRATIC_MTLD_NO_CENTAUR: {
            **shared,
            "model_parameterization": "stock_olmo2_190m",
            "target_depth": 12,
            "freeze_first_n_blocks": 0,
            "llm_ratio": 0.0,
            "llm_scope": "none",
        },
        Arm.OLMOE_NO_PROXY: {
            **shared,
            "model_parameterization": "stock_olmoe_1b_7b",
            "target_depth": 16,
            "freeze_first_n_blocks": 0,
            "llm_ratio": 0.3,
            "llm_scope": "multi_action",
        },
        Arm.OLMOE_NO_CENTAUR: {
            **shared,
            "model_parameterization": "stock_olmoe_1b_7b",
            "target_depth": 16,
            "freeze_first_n_blocks": 0,
            "llm_ratio": 0.0,
            "llm_scope": "none",
        },
    }
