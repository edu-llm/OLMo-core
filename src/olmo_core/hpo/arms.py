"""
Preregistered ablation arms and equal-budget accounting.

The plan's evidence rests on comparing arms *at equal total budget*, where "budget" is charged
accelerator-seconds and training tokens across **every** category of work -- including failed,
screened, reset, retrained, LLM-controller, and proxy-selection compute, not just the main
trials. :class:`BudgetLedger` makes that accounting explicit and refuses a comparison until every
category has been counted; :func:`equal_budget` compares two ledgers within a tolerance.

:class:`Arm` enumerates the controls and staged ablations so an experiment driver can iterate
them, and :func:`ablation_matrix` records the LLM ratio/scope that makes the incremental LLM
effect identifiable (CMA-only ``r=0`` vs configuration-only Centaur vs the broader multi-action
advisor).

Pure ``math`` + standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

__all__ = ["Arm", "EQUAL_BUDGET_CATEGORIES", "BudgetLedger", "equal_budget", "ablation_matrix"]


class Arm(str, Enum):
    RANDOM_FIXED_RECIPE = "random_fixed_recipe"
    RANDOM_DYNAMIC_SCHEDULE = "random_dynamic_schedule"
    RANDOM_TOKEN_SCREEN = "random_token_screen"
    IFBO_OFFICIAL = "ifbo_official"
    IPBT_REFERENCE = "ipbt_reference"
    IPBT_RESTART_BTT_2X2 = "ipbt_restart_x_btt_2x2"
    IPBT_IFBO = "ipbt_ifbo"
    NOVEL_AGGREGATE_BTT = "novel_aggregate_btt"
    CMA_ONLY = "cma_only"
    CONFIG_ONLY_CENTAUR = "config_only_centaur"
    MULTI_ACTION_ADVISOR = "multi_action_advisor"
    FROZEN_LAYER_PROXY = "frozen_layer_proxy"
    UMUP_PROXY = "umup_proxy"


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
    """The preregistered arm settings that make the LLM contribution identifiable."""
    return {
        Arm.RANDOM_FIXED_RECIPE: {"dynamic_schedule": False, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.RANDOM_DYNAMIC_SCHEDULE: {
            "dynamic_schedule": True,
            "llm_ratio": 0.0,
            "llm_scope": "none",
        },
        Arm.RANDOM_TOKEN_SCREEN: {"dynamic_schedule": False, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.IFBO_OFFICIAL: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.IPBT_REFERENCE: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.IPBT_RESTART_BTT_2X2: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.IPBT_IFBO: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.NOVEL_AGGREGATE_BTT: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.CMA_ONLY: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.CONFIG_ONLY_CENTAUR: {
            "dynamic_schedule": True,
            "llm_ratio": 0.3,
            "llm_scope": "config_only",
        },
        Arm.MULTI_ACTION_ADVISOR: {
            "dynamic_schedule": True,
            "llm_ratio": 0.3,
            "llm_scope": "multi_action",
        },
        Arm.FROZEN_LAYER_PROXY: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
        Arm.UMUP_PROXY: {"dynamic_schedule": True, "llm_ratio": 0.0, "llm_scope": "none"},
    }
