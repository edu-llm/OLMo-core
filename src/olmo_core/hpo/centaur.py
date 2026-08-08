"""
Centaur-style LLM overlay on top of a CMA-ES new-configuration proposer.

The Brainlift's Centaur pattern has a frontier LLM act on *structured, deterministic* optimizer
state and occasionally override the classical proposal. This module keeps CMA-ES as the only
new-configuration proposal state and adds a typed, auditable LLM overlay around it. It fixes the
issues the plan flags in the upstream implementation:

- **Vector identity.** :class:`AskLedger` assigns monotonic ask ids, persists every pending
  candidate, and refuses to ``tell`` CMA a score unless the *evaluated* vector is byte-for-byte
  the *asked* vector.
- **Comparable-fidelity tells.** Only resolved anchors from a declared from-scratch stratum are
  told to CMA; BTT cuts/pauses are *censored* and numeric/OOM failures get a *preregistered
  penalty* -- neither masquerades as a bad full-fidelity objective.
- **Resume is free.** Continuing an existing trial does not consume a CMA ask
  (:func:`consumes_cma_ask`).
- **Deterministic, preregistered intervention.** :func:`should_llm_intervene` picks LLM turns by
  monotonic proposal id after a warmup, hitting an exact ratio with even spacing.
- **Fail loud.** On advisor error the overlay raises :class:`AdvisorUnavailable` so the controller
  checkpoints and pauses; it never silently falls back to CMA while still calling the arm Centaur.
- **Prompt the mean.** :func:`build_advisor_state` includes the CMA mean, which the pinned
  upstream extracts but never actually prompts.

Pure ``numpy`` + standard library; ``cmaes`` is imported lazily only by :class:`CMAESProposer`.
"""

from __future__ import annotations

import base64
import copy
import math
import pickle
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from .types import ProposalSource

__all__ = [
    "LegalAction",
    "AskStatus",
    "PendingAsk",
    "AskLedger",
    "AdvisorResponse",
    "AdvisorRecord",
    "AdvisorUnavailable",
    "LLMAdvisor",
    "RequiredModelAdvisor",
    "CentaurOverlay",
    "CMAESProposer",
    "consumes_cma_ask",
    "should_llm_intervene",
    "validate_start_config",
    "validate_action",
    "build_advisor_state",
]


class LegalAction(str, Enum):
    START = "start"
    RESUME = "resume"
    IPBT_EXPLOIT = "ipbt_exploit"
    IPBT_RESTART = "ipbt_restart"


def consumes_cma_ask(action: LegalAction) -> bool:
    """Only a from-scratch ``START`` consumes a CMA ask; resumes/IPBT actions do not."""
    return action is LegalAction.START


class AskStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    CENSORED = "censored"
    FAILED = "failed"
    TOLD = "told"
    REPLACED = "replaced"


@dataclass
class PendingAsk:
    ask_id: int
    unit_config: Tuple[float, ...]
    stratum: str
    generation_id: int
    vector_dtype: str
    vector_shape: Tuple[int, ...]
    vector_bytes: bytes
    replaces_ask_id: Optional[int] = None
    status: AskStatus = AskStatus.PENDING
    score: Optional[float] = None


class AskLedger:
    """Tracks CMA asks by monotonic id and enforces asked/evaluated vector identity."""

    def __init__(self) -> None:
        self._asks: Dict[int, PendingAsk] = {}
        self._counter = 0
        self._generation_counter = 0

    def register(self, configs: Sequence[Sequence[float]], *, stratum: str) -> List[PendingAsk]:
        if not configs:
            raise ValueError("a CMA generation must contain at least one config")
        generation_id = self._generation_counter
        self._generation_counter += 1
        out: List[PendingAsk] = []
        for cfg in configs:
            vector = np.asarray(cfg)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError("CMA ask vectors must be non-empty and one-dimensional")
            if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
                raise ValueError("CMA ask vectors must lie in the finite unit cube")
            ask = PendingAsk(
                ask_id=self._counter,
                unit_config=tuple(float(x) for x in vector),
                stratum=stratum,
                generation_id=generation_id,
                vector_dtype=vector.dtype.str,
                vector_shape=vector.shape,
                vector_bytes=vector.tobytes(),
            )
            self._asks[ask.ask_id] = ask
            self._counter += 1
            out.append(ask)
        return out

    def get(self, ask_id: int) -> PendingAsk:
        return self._asks[ask_id]

    def _pending(self, ask_id: int) -> PendingAsk:
        ask = self._asks[ask_id]
        if ask.status is not AskStatus.PENDING:
            raise ValueError(f"ask {ask_id} is already terminal ({ask.status.value})")
        return ask

    def resolve(self, ask_id: int, *, score: float, evaluated_config: Sequence[float]) -> None:
        ask = self._pending(ask_id)
        evaluated = np.asarray(evaluated_config)
        if (
            evaluated.dtype.str != ask.vector_dtype
            or evaluated.shape != ask.vector_shape
            or evaluated.tobytes() != ask.vector_bytes
        ):
            raise ValueError(
                f"ask {ask_id}: evaluated vector is not byte-identical to the asked vector; "
                "refusing to tell CMA a score for a different configuration"
            )
        if not math.isfinite(score):
            raise ValueError(f"ask {ask_id}: score must be finite, got {score}")
        ask.status = AskStatus.RESOLVED
        ask.score = float(score)

    def censor(self, ask_id: int) -> None:
        """Mark a BTT-cut/paused ask as censored (never a full-fidelity objective)."""
        self._pending(ask_id).status = AskStatus.CENSORED

    def replace_censored(self, ask_id: int, config: Sequence[float]) -> PendingAsk:
        censored = self._asks[ask_id]
        if censored.status is not AskStatus.CENSORED:
            raise ValueError(f"ask {ask_id} is not censored")
        return self._replace(ask_id, config)

    def replace_pending(self, ask_id: int, config: Sequence[float]) -> PendingAsk:
        self._pending(ask_id)
        return self._replace(ask_id, config)

    def _replace(self, ask_id: int, config: Sequence[float]) -> PendingAsk:
        original = self._asks[ask_id]
        vector = np.asarray(config)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("replacement vector must be non-empty and one-dimensional")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("replacement vector must lie in the finite unit cube")
        original.status = AskStatus.REPLACED
        replacement = PendingAsk(
            ask_id=self._counter,
            unit_config=tuple(float(x) for x in vector),
            stratum=original.stratum,
            generation_id=original.generation_id,
            vector_dtype=vector.dtype.str,
            vector_shape=vector.shape,
            vector_bytes=vector.tobytes(),
            replaces_ask_id=ask_id,
        )
        self._asks[replacement.ask_id] = replacement
        self._counter += 1
        return replacement

    def fail(self, ask_id: int, *, penalty: float) -> None:
        """Mark a genuine numeric/OOM failure with a preregistered penalty objective."""
        ask = self._pending(ask_id)
        if not math.isfinite(penalty):
            raise ValueError("failure penalty must be finite")
        ask.status = AskStatus.FAILED
        ask.score = float(penalty)

    def collect_tell(
        self,
        *,
        stratum: str,
        allow_inherited: bool,
        consume: bool = True,
    ) -> List[Tuple[Tuple[float, ...], float]]:
        """Return each complete generation once; censored generations require replacement."""
        out: List[Tuple[Tuple[float, ...], float]] = []
        eligible = [
            ask
            for ask in self._asks.values()
            if ask.status is not AskStatus.REPLACED
            and (ask.stratum == stratum or (allow_inherited and ask.stratum == "inherited"))
        ]
        generations = sorted({ask.generation_id for ask in eligible})
        for generation_id in generations:
            generation = [ask for ask in eligible if ask.generation_id == generation_id]
            if all(ask.status is AskStatus.TOLD for ask in generation):
                continue
            if any(ask.status is AskStatus.PENDING for ask in generation):
                raise ValueError(f"CMA generation {generation_id} is incomplete")
            if any(ask.status is AskStatus.CENSORED for ask in generation):
                raise ValueError(
                    f"CMA generation {generation_id} contains censored asks; replace them "
                    "under the declared policy before tell"
                )
            if any(ask.status not in (AskStatus.RESOLVED, AskStatus.FAILED) for ask in generation):
                raise ValueError(f"CMA generation {generation_id} has an invalid lifecycle")
            for ask in sorted(generation, key=lambda item: item.ask_id):
                assert ask.score is not None
                out.append((ask.unit_config, ask.score))
                if consume:
                    ask.status = AskStatus.TOLD
        return out

    def mark_told(self, *, stratum: str, allow_inherited: bool) -> None:
        eligible = [
            ask
            for ask in self._asks.values()
            if ask.status is not AskStatus.REPLACED
            and (ask.stratum == stratum or (allow_inherited and ask.stratum == "inherited"))
        ]
        for ask in eligible:
            if ask.status in (AskStatus.RESOLVED, AskStatus.FAILED):
                ask.status = AskStatus.TOLD

    def has_open_generation(self, *, stratum: str) -> bool:
        return any(
            ask.stratum == stratum and ask.status not in (AskStatus.TOLD, AskStatus.REPLACED)
            for ask in self._asks.values()
        )

    def pending_asks(self, *, stratum: str) -> List[PendingAsk]:
        return sorted(
            (
                ask
                for ask in self._asks.values()
                if ask.stratum == stratum and ask.status is AskStatus.PENDING
            ),
            key=lambda ask: ask.ask_id,
        )

    def can_collect_tell(self, *, stratum: str, allow_inherited: bool) -> bool:
        eligible = [
            ask
            for ask in self._asks.values()
            if ask.status is not AskStatus.REPLACED
            and (ask.stratum == stratum or (allow_inherited and ask.stratum == "inherited"))
        ]
        for generation_id in {ask.generation_id for ask in eligible}:
            generation = [ask for ask in eligible if ask.generation_id == generation_id]
            if generation and all(
                ask.status in (AskStatus.RESOLVED, AskStatus.FAILED) for ask in generation
            ):
                return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "counter": self._counter,
            "generation_counter": self._generation_counter,
            "asks": [
                {
                    "ask_id": ask.ask_id,
                    "unit_config": list(ask.unit_config),
                    "stratum": ask.stratum,
                    "generation_id": ask.generation_id,
                    "vector_dtype": ask.vector_dtype,
                    "vector_shape": list(ask.vector_shape),
                    "vector_bytes": ask.vector_bytes.hex(),
                    "replaces_ask_id": ask.replaces_ask_id,
                    "status": ask.status.value,
                    "score": ask.score,
                }
                for ask in sorted(self._asks.values(), key=lambda item: item.ask_id)
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._counter = int(state["counter"])
        self._generation_counter = int(state["generation_counter"])
        self._asks = {}
        for value in state["asks"]:
            ask = PendingAsk(
                ask_id=int(value["ask_id"]),
                unit_config=tuple(float(x) for x in value["unit_config"]),
                stratum=str(value["stratum"]),
                generation_id=int(value["generation_id"]),
                vector_dtype=str(value["vector_dtype"]),
                vector_shape=tuple(int(x) for x in value["vector_shape"]),
                vector_bytes=bytes.fromhex(value["vector_bytes"]),
                replaces_ask_id=(
                    None if value.get("replaces_ask_id") is None else int(value["replaces_ask_id"])
                ),
                status=AskStatus(value["status"]),
                score=None if value["score"] is None else float(value["score"]),
            )
            self._asks[ask.ask_id] = ask


@dataclass(frozen=True)
class AdvisorResponse:
    action: Dict[str, Any]
    raw_text: str
    model: str
    version: str
    latency_ms: float


@dataclass(frozen=True)
class AdvisorRecord:
    """A complete, auditable log of one advisor turn."""

    prompt_state: Dict[str, Any]
    response: AdvisorResponse

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def version(self) -> str:
        return self.response.version

    @property
    def latency_ms(self) -> float:
        return self.response.latency_ms


class AdvisorUnavailable(Exception):
    """Raised when the LLM advisor cannot be reached; the controller must pause, not fall back."""

    def __init__(self, message: str, *, record: Optional[AdvisorRecord] = None) -> None:
        super().__init__(message)
        self.record = record


@runtime_checkable
class LLMAdvisor(Protocol):
    def advise(self, state: Dict[str, Any]) -> AdvisorResponse:
        ...


class RequiredModelAdvisor:
    """Fail closed unless the delegated advisor reports the preregistered model."""

    def __init__(self, delegate: LLMAdvisor, required_model: str) -> None:
        if not required_model:
            raise ValueError("required_model must be non-empty")
        self.delegate = delegate
        self.required_model = required_model

    def advise(self, state: Dict[str, Any]) -> AdvisorResponse:
        response = self.delegate.advise(state)
        if response.model != self.required_model:
            raise AdvisorUnavailable(
                f"advisor reported model {response.model!r}, expected {self.required_model!r}"
            )
        return response


def should_llm_intervene(proposal_id: int, *, warmup: int, ratio: float) -> bool:
    """Deterministically decide whether decision ``proposal_id`` is an LLM turn.

    Uses an evenly spaced cumulative rule so that over ``N`` post-warmup decisions exactly
    ``floor(N * ratio)`` are LLM turns, with no dependence on completion order.
    """
    if proposal_id < 0 or warmup < 0:
        raise ValueError("proposal_id and warmup must be non-negative")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be finite and in [0, 1], got {ratio}")
    if ratio == 0.0:
        return False
    if proposal_id < warmup:
        return False
    j = proposal_id - warmup
    exact_ratio = Decimal(str(ratio))
    previous = int((Decimal(j) * exact_ratio).to_integral_value(rounding=ROUND_FLOOR))
    current = int((Decimal(j + 1) * exact_ratio).to_integral_value(rounding=ROUND_FLOOR))
    return current > previous


def validate_start_config(unit_config: Sequence[float]) -> None:
    arr = [float(x) for x in unit_config]
    if not all(math.isfinite(x) for x in arr):
        raise ValueError(f"start config has non-finite values: {arr}")
    if any(x < 0.0 or x > 1.0 for x in arr):
        raise ValueError(f"start config outside unit cube: {arr}")


def validate_action(
    action: Dict[str, Any], *, expected_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Validate an LLM-proposed action against the legal schema; raise on anything illegal."""
    kind = action.get("kind")
    try:
        legal = LegalAction(kind)
    except ValueError as e:
        raise ValueError(f"illegal action kind: {kind!r}") from e
    if legal is LegalAction.START:
        cfg = action.get("unit_config")
        if cfg is None:
            raise ValueError("start action requires 'unit_config'")
        validate_start_config(cfg)
        if expected_dim is not None and len(cfg) != expected_dim:
            raise ValueError(f"start action requires {expected_dim} dimensions")
    elif legal is LegalAction.RESUME:
        if not action.get("trial_id"):
            raise ValueError("resume action requires 'trial_id'")
        if "unit_config" in action:
            raise ValueError("resume action must not supply a new unit_config")
    elif legal is LegalAction.IPBT_EXPLOIT:
        if not action.get("donor_id") or not action.get("target_slot_id"):
            raise ValueError("ipbt_exploit requires donor_id and target_slot_id")
    elif legal is LegalAction.IPBT_RESTART:
        if not action.get("restart_id") or not action.get("target_slot_id"):
            raise ValueError("ipbt_restart requires restart_id and target_slot_id")
    return copy.deepcopy(action)


def build_advisor_state(
    *,
    cma_mean: Sequence[float],
    cma_sigma: float,
    cma_cov: Sequence[Sequence[float]],
    cma_proposal: Sequence[float],
    ifbo_action: Dict[str, Any],
    ifbo_alternatives: List[Dict[str, Any]],
    population_lineages: List[Any],
    btt_evidence: List[Any],
    incumbent: Dict[str, Any],
    top_five: List[Any],
    recent_decisions: List[Any],
    bounds: Sequence[Sequence[float]],
    remaining_budget: int,
    action_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the structured, deterministic state handed to the LLM advisor."""
    return copy.deepcopy(
        {
            "cma_mean": list(cma_mean),
            "cma_sigma": float(cma_sigma),
            "cma_cov": [list(row) for row in cma_cov],
            "cma_proposal": list(cma_proposal),
            "ifbo_action": ifbo_action,
            "ifbo_alternatives": ifbo_alternatives,
            "population_lineages": population_lineages,
            "btt_evidence": btt_evidence,
            "incumbent": incumbent,
            "top_five": top_five,
            "recent_decisions": recent_decisions,
            "bounds": [list(b) for b in bounds],
            "remaining_budget": int(remaining_budget),
            "action_schema": action_schema,
        }
    )


class CentaurOverlay:
    """Decides, per monotonic proposal id, whether the LLM overrides the CMA proposal."""

    def __init__(self, *, warmup: int, ratio: float) -> None:
        should_llm_intervene(warmup, warmup=warmup, ratio=ratio)
        self.warmup = warmup
        self.ratio = ratio

    def propose(
        self,
        *,
        proposal_id: int,
        cma_config: Tuple[float, ...],
        advisor: LLMAdvisor,
        state: Dict[str, Any],
    ) -> Tuple[Tuple[float, ...], ProposalSource, Optional[AdvisorRecord]]:
        if not should_llm_intervene(proposal_id, warmup=self.warmup, ratio=self.ratio):
            return cma_config, ProposalSource.CMA, None

        prompt_state = copy.deepcopy(state)
        try:
            response = advisor.advise(prompt_state)
        except Exception as e:  # fail loud: pause + checkpoint upstream, never silent CMA
            raise AdvisorUnavailable(f"LLM advisor failed for proposal {proposal_id}: {e!r}") from e

        recorded_response = AdvisorResponse(
            action=copy.deepcopy(response.action),
            raw_text=response.raw_text,
            model=response.model,
            version=response.version,
            latency_ms=response.latency_ms,
        )
        record = AdvisorRecord(prompt_state=prompt_state, response=recorded_response)
        try:
            action = validate_action(recorded_response.action, expected_dim=len(cma_config))
            if LegalAction(action["kind"]) is not LegalAction.START:
                raise ValueError("configuration-only Centaur may emit only START actions")
        except Exception as exc:
            raise AdvisorUnavailable(
                f"LLM advisor returned an invalid action for proposal {proposal_id}: {exc}",
                record=record,
            ) from exc
        cfg = action["unit_config"]
        return tuple(float(x) for x in cfg), ProposalSource.LLM, record

    def decide(
        self,
        *,
        proposal_id: int,
        default_action: Dict[str, Any],
        advisor: LLMAdvisor,
        state: Dict[str, Any],
        expected_dim: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Optional[AdvisorRecord]]:
        """Return a validated broader action override, or the deterministic default."""
        if not should_llm_intervene(proposal_id, warmup=self.warmup, ratio=self.ratio):
            return copy.deepcopy(default_action), None
        prompt_state = copy.deepcopy(state)
        try:
            response = advisor.advise(prompt_state)
        except Exception as exc:
            raise AdvisorUnavailable(
                f"LLM advisor failed for action proposal {proposal_id}: {exc!r}"
            ) from exc
        recorded_response = AdvisorResponse(
            action=copy.deepcopy(response.action),
            raw_text=response.raw_text,
            model=response.model,
            version=response.version,
            latency_ms=response.latency_ms,
        )
        record = AdvisorRecord(prompt_state=prompt_state, response=recorded_response)
        try:
            action = validate_action(recorded_response.action, expected_dim=expected_dim)
        except Exception as exc:
            raise AdvisorUnavailable(
                f"LLM advisor returned an invalid action for proposal {proposal_id}: {exc}",
                record=record,
            ) from exc
        return action, record


class CMAESProposer:
    """Thin wrapper over the public ``cmaes.CMA`` implementation (no Optuna internals)."""

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 0,
        sigma: float = 0.2,
        mean: Optional[Sequence[float]] = None,
        population_size: Optional[int] = None,
    ) -> None:
        from cmaes import CMA  # lazy optional dependency

        mean_arr = np.full(dim, 0.5) if mean is None else np.asarray(mean, dtype=np.float64)
        bounds = np.tile(np.array([[0.0, 1.0]]), (dim, 1))
        self._cma = CMA(
            mean=mean_arr,
            sigma=sigma,
            seed=seed,
            bounds=bounds,
            population_size=population_size,
        )
        self._dim = dim

    def ask(self, n: int) -> List[Tuple[float, ...]]:
        if n != self.population_size:
            raise ValueError(
                f"CMA ask batch must equal population_size={self.population_size}, got {n}"
            )
        configs = [tuple(float(v) for v in self._cma.ask()) for _ in range(n)]
        for config in configs:
            validate_start_config(config)
        return configs

    def tell(self, solutions: Sequence[Tuple[Sequence[float], float]]) -> None:
        if len(solutions) != self.population_size:
            raise ValueError("CMA tell requires one complete generation")
        converted = []
        for config, score in solutions:
            validate_start_config(config)
            if not math.isfinite(score):
                raise ValueError("CMA tell scores must be finite")
            converted.append((np.asarray(config, dtype=np.float64), -float(score)))
        self._cma.tell(converted)

    @property
    def mean(self) -> List[float]:
        return [float(v) for v in self._cma.mean]

    @property
    def population_size(self) -> int:
        return int(self._cma.population_size)

    def state(self) -> Dict[str, Any]:
        """Serializable public wrapper state for prompts and controller snapshots."""
        return {
            "mean": self.mean,
            "sigma": float(self._cma._sigma),
            "covariance": np.asarray(self._cma._C, dtype=np.float64).tolist(),
            "generation": int(self._cma.generation),
            "population_size": self.population_size,
        }

    def state_dict(self) -> Dict[str, str]:
        rng_state = self._cma._rng.get_state()
        return {
            "optimizer": base64.b64encode(
                pickle.dumps(self._cma, protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii"),
            "rng_state": base64.b64encode(
                pickle.dumps(rng_state, protocol=pickle.HIGHEST_PROTOCOL)
            ).decode("ascii"),
        }

    def load_state_dict(self, state: Dict[str, str]) -> None:
        self._cma = pickle.loads(base64.b64decode(state["optimizer"]))
        rng_state = pickle.loads(base64.b64decode(state["rng_state"]))
        self._cma._rng.set_state(rng_state)
        self._dim = int(self._cma.dim)
