"""
Serializable configuration for the HPO controller.

These are :class:`~olmo_core.config.Config` dataclasses so they inherit the repository's
YAML/JSON serialization and dotlist ``merge()`` behavior, matching how the rest of OLMo-core
is configured. They ``build()`` into the pure runtime objects from :mod:`olmo_core.hpo.types`.

The default search space is the plan's shared <=10-dimensional optimization space; every
experimental arm draws from the *same* space so method quality is not confounded with
search-space quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..config import Config
from ..exceptions import OLMoConfigurationError
from .types import SearchDim, SearchSpace

__all__ = [
    "SearchDimConfig",
    "SearchSpaceConfig",
    "FidelityConfig",
    "HpoControllerConfig",
]


@dataclass
class SearchDimConfig(Config):
    """Config for a single :class:`~olmo_core.hpo.types.SearchDim`."""

    name: str
    low: float
    high: float
    log: bool = False

    def build(self) -> SearchDim:
        return SearchDim(name=self.name, low=self.low, high=self.high, log=self.log)


@dataclass
class SearchSpaceConfig(Config):
    """An ordered set of :class:`SearchDimConfig`."""

    dims: List[SearchDimConfig] = field(default_factory=list)

    def build(self) -> SearchSpace:
        return SearchSpace(tuple(d.build() for d in self.dims))

    @classmethod
    def default_transformer_space(cls) -> "SearchSpaceConfig":
        """The plan's shared 9-dimensional optimization space.

        ``beta2_gap`` and ``eps`` are searched in a log space (``log10(1 - beta2)`` bounds and
        the raw epsilon respectively); ``global_batch_mult`` multiplies the baseline global
        batch size; all others are direct.
        """
        return cls(
            dims=[
                SearchDimConfig(name="lr", low=2.5e-4, high=4e-3, log=True),
                SearchDimConfig(name="weight_decay", low=0.01, high=0.3, log=True),
                SearchDimConfig(name="beta2_gap", low=1e-3, high=1e-1, log=True),
                SearchDimConfig(name="eps", low=1e-12, high=1e-6, log=True),
                SearchDimConfig(name="warmup_fraction", low=0.005, high=0.08, log=False),
                SearchDimConfig(name="decay_fraction", low=0.05, high=0.3, log=False),
                SearchDimConfig(name="terminal_lr_ratio", low=0.0, high=0.2, log=False),
                SearchDimConfig(name="global_batch_mult", low=0.5, high=2.0, log=True),
                SearchDimConfig(name="max_grad_norm", low=0.3, high=3.0, log=True),
            ]
        )


@dataclass
class FidelityConfig(Config):
    """Absolute token rungs the allocator promotes trials through.

    ``rungs`` must be strictly increasing. The first rung is the minimum fidelity a new
    ``START`` receives; the last is the final per-lineage token horizon.
    """

    rungs: List[int] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.rungs) < 1:
            raise OLMoConfigurationError("FidelityConfig requires at least one rung")
        if any(
            isinstance(rung, bool) or not isinstance(rung, int) or rung <= 0 for rung in self.rungs
        ):
            raise OLMoConfigurationError(
                f"FidelityConfig.rungs must be positive integer token counts, got {self.rungs}"
            )
        for a, b in zip(self.rungs, self.rungs[1:]):
            if b <= a:
                raise OLMoConfigurationError(
                    f"FidelityConfig.rungs must be strictly increasing, got {self.rungs}"
                )

    def __post_init__(self) -> None:
        # Fail fast on obviously malformed rungs, but keep validate() callable for tests
        # that construct-then-validate.
        if self.rungs:
            self.validate()

    @property
    def min_fidelity(self) -> int:
        return self.rungs[0]

    @property
    def target_fidelity(self) -> int:
        return self.rungs[-1]


@dataclass
class HpoControllerConfig(Config):
    """Top-level controller knobs shared across arms.

    :param worker_count: Number of concurrent one-trial-per-GPU workers.
    :param population_size: Logical IPBT population (>= worker_count).
    :param llm_ratio: Fraction of ifBO decisions 5.6 Sol may override in ``[0, 1]``.
    :param llm_warmup: Number of pure-ifBO decisions before Sol may intervene.
    :param seed: Controller RNG seed (search reproducibility, distinct from model seed).
    """

    worker_count: int = 8
    population_size: int = 16
    llm_ratio: float = 0.3
    llm_warmup: int = 0
    seed: int = 0

    def validate(self) -> None:
        if self.worker_count < 1:
            raise OLMoConfigurationError("worker_count must be >= 1")
        if self.population_size < self.worker_count:
            raise OLMoConfigurationError(
                f"population_size ({self.population_size}) must be >= worker_count "
                f"({self.worker_count})"
            )
        if not (0.0 <= self.llm_ratio <= 1.0):
            raise OLMoConfigurationError(f"llm_ratio must be in [0, 1], got {self.llm_ratio}")
        if self.llm_warmup < 0:
            raise OLMoConfigurationError("llm_warmup must be >= 0")
