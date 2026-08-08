"""
BTTackler: the diagnosis-based hard-cutoff layer, re-implemented natively from the paper.

BTTackler classifies a trial from *training diagnostics* rather than validation performance
alone. This module implements the paper's seven indicator families over the bounded telemetry
:class:`~olmo_core.hpo.worker.HpoDiagnosticsCallback` collects, and returns a versioned
:class:`~olmo_core.hpo.types.Verdict` bound to the exact observation it saw:

===== ==================================== ==================
Code  Meaning                              Verdict
===== ==================================== ==================
AGV   abnormal gradient values             ``FATAL``
EAG   exponentially amplified gradients    ``DEGRADED``
ERG   exponentially reduced gradients      ``DEGRADED``
PLC   passive loss changes (never learned) ``DEGRADED``
LAR   low activation ratio                 ``DEGRADED``
ULC   unexpected loss changes (spike)      ``DEGRADED``
NMG   no more gain (converged/plateaued)   ``SATURATED``
===== ==================================== ==================

BTTackler is the *sole* per-trial eligibility/censoring layer. It never chooses
hyperparameters, donors, or allocation size -- it only emits evidence-bound verdicts. Two arms
exist: a paper-faithful binary arm (any positive indicator stops the trial; NMG stops but keeps
candidacy) and the explicitly novel Transformer-adapted recycle arm with four verdicts. Neither
attributes the adapted behavior to the validated BTTackler result.

Pure ``math`` + ``hashlib``; no ``torch``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .types import BTTDisposition, BTTVerdictKind, Verdict

__all__ = [
    "BTTMode",
    "BTTObservation",
    "BTTCalibrationProfile",
    "BTTConfig",
    "BTTDiagnoser",
]


class BTTMode(str, Enum):
    ADAPTED_RECYCLE = "adapted_recycle"
    """Novel Transformer-adapted arm: four verdicts, DEGRADED requests a recycle."""

    PAPER_BINARY = "paper_binary"
    """Paper-faithful arm: any positive indicator stops the trial (NMG keeps candidacy)."""


@dataclass(frozen=True)
class BTTObservation:
    """Bounded telemetry for one trial at a fidelity boundary."""

    trial_id: str
    completed_fidelity: int
    observation_hash: str
    grad_norm_history: Tuple[float, ...]
    loss_history: Tuple[float, ...]
    activation_ratio: Optional[float] = None
    non_finite: bool = False


@dataclass(frozen=True)
class BTTCalibrationProfile:
    """Frozen thresholds with provenance from completed calibration runs."""

    profile_version: str
    completed_run_ids: Tuple[str, ...]
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_run_ids",
            tuple(self.completed_run_ids),
        )
        if not self.profile_version or not self.completed_run_ids:
            raise ValueError("calibration profile requires a version and completed run ids")
        if len(set(self.completed_run_ids)) != len(self.completed_run_ids):
            raise ValueError("calibration completed_run_ids must be unique")
        if any(not math.isfinite(value) for value in self.thresholds.values()):
            raise ValueError("calibration thresholds must be finite")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))


@dataclass(frozen=True)
class BTTConfig:
    """Thresholds and gates. Freeze these before target HPO."""

    profile_version: str = "btt-v1"
    mode: BTTMode = BTTMode.ADAPTED_RECYCLE
    min_fidelity: int = 0
    window: int = 4
    agv_max_grad_norm: float = 1e4
    eag_ratio: float = 5.0
    erg_ratio: float = 0.2
    plc_min_rel_improve: float = 1e-3
    ulc_spike_ratio: float = 1.2
    lar_min_active: float = 0.1
    nmg_min_rel_improve: float = 5e-3
    late_bloomer_reserve: float = 0.0
    reserve_seed: int = 0
    same_fidelity_top_fraction: float = 0.0
    require_calibration: bool = False

    def __post_init__(self) -> None:
        if self.min_fidelity < 0 or self.window < 1:
            raise ValueError("min_fidelity must be non-negative and window must be positive")
        if not 0.0 <= self.late_bloomer_reserve <= 1.0:
            raise ValueError("late_bloomer_reserve must be in [0, 1]")
        if not 0.0 <= self.same_fidelity_top_fraction <= 1.0:
            raise ValueError("same_fidelity_top_fraction must be in [0, 1]")
        if self.agv_max_grad_norm <= 0.0:
            raise ValueError("agv_max_grad_norm must be positive")
        if self.eag_ratio <= 1.0:
            raise ValueError("eag_ratio must be greater than 1")
        if not 0.0 < self.erg_ratio < 1.0:
            raise ValueError("erg_ratio must be in (0, 1)")
        if not 0.0 <= self.plc_min_rel_improve <= 1.0:
            raise ValueError("plc_min_rel_improve must be in [0, 1]")
        if self.ulc_spike_ratio <= 1.0:
            raise ValueError("ulc_spike_ratio must be greater than 1")
        if not 0.0 <= self.lar_min_active <= 1.0:
            raise ValueError("lar_min_active must be in [0, 1]")
        if not 0.0 <= self.nmg_min_rel_improve <= 1.0:
            raise ValueError("nmg_min_rel_improve must be in [0, 1]")


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _uniform_draw(trial_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{trial_id}:{seed}".encode("utf-8")).hexdigest()
    return int(digest, 16) / float(1 << 256)


class BTTDiagnoser:
    """Applies :class:`BTTConfig` to a :class:`BTTObservation`."""

    def __init__(
        self,
        config: BTTConfig,
        *,
        calibration: Optional[BTTCalibrationProfile] = None,
    ) -> None:
        if config.require_calibration and calibration is None:
            raise ValueError("BTT calibration profile is required")
        if calibration is not None:
            allowed = {
                "agv_max_grad_norm",
                "eag_ratio",
                "erg_ratio",
                "plc_min_rel_improve",
                "ulc_spike_ratio",
                "lar_min_active",
                "nmg_min_rel_improve",
            }
            unknown = set(calibration.thresholds) - allowed
            if unknown:
                raise ValueError(f"unknown calibrated BTT thresholds: {sorted(unknown)}")
            config = replace(
                config,
                profile_version=calibration.profile_version,
                **dict(calibration.thresholds),
            )
        self.config = config

    def diagnose_cohort(
        self,
        observations: Sequence[BTTObservation],
        *,
        scores: Mapping[str, float],
    ) -> Dict[str, Verdict]:
        """Diagnose a cohort, then protect top same-fidelity degraded members."""
        if {observation.trial_id for observation in observations} != set(scores):
            raise ValueError("scores must identify exactly the diagnosed observations")
        if any(not math.isfinite(score) for score in scores.values()):
            raise ValueError("same-fidelity protection scores must be finite")
        verdicts = {
            observation.trial_id: self.diagnose(observation) for observation in observations
        }
        fraction = self.config.same_fidelity_top_fraction
        if fraction <= 0.0:
            return verdicts
        by_fidelity: Dict[int, List[BTTObservation]] = {}
        for observation in observations:
            by_fidelity.setdefault(observation.completed_fidelity, []).append(observation)
        for peers in by_fidelity.values():
            protect_count = max(1, math.ceil(fraction * len(peers)))
            protected_ids = {
                observation.trial_id
                for observation in sorted(
                    peers,
                    key=lambda item: (-scores[item.trial_id], item.trial_id),
                )[:protect_count]
            }
            for trial_id in protected_ids:
                verdict = verdicts[trial_id]
                if verdict.kind is BTTVerdictKind.DEGRADED:
                    verdicts[trial_id] = replace(
                        verdict,
                        kind=BTTVerdictKind.HEALTHY,
                        disposition=BTTDisposition.CONTINUE,
                        protected_by_peer_rank=True,
                    )
        return verdicts

    def _fatal_indicators(self, obs: BTTObservation) -> List[str]:
        indicators: List[str] = []
        all_values = list(obs.grad_norm_history) + list(obs.loss_history)
        non_finite = obs.non_finite or any(not math.isfinite(v) for v in all_values)
        too_large = any(
            math.isfinite(g) and abs(g) > self.config.agv_max_grad_norm
            for g in obs.grad_norm_history
        )
        if non_finite or too_large:
            indicators.append("AGV")
        return indicators

    def _degraded_indicators(self, obs: BTTObservation) -> List[str]:
        cfg = self.config
        w = min(cfg.window, len(obs.grad_norm_history) // 2)
        indicators: List[str] = []

        # Gradient dynamics.
        if w >= 1:
            early = _mean(obs.grad_norm_history[:w])
            late = _mean(obs.grad_norm_history[-w:])
            if early > 0:
                ratio = late / early
                if ratio > cfg.eag_ratio:
                    indicators.append("EAG")
                elif ratio < cfg.erg_ratio:
                    indicators.append("ERG")

        # Loss dynamics.
        loss = obs.loss_history
        if len(loss) >= 2:
            first, last = loss[0], loss[-1]
            denom = abs(first) if first != 0 else 1.0
            overall_rel = (first - last) / denom
            if 0.0 <= overall_rel < cfg.plc_min_rel_improve:
                indicators.append("PLC")
            lw = min(cfg.window, len(loss))
            recent_before_last = loss[-lw:-1]
            best = min(recent_before_last) if recent_before_last else last
            if last > best * cfg.ulc_spike_ratio:
                indicators.append("ULC")

        # Activation support (only when telemetry is present).
        if obs.activation_ratio is not None and obs.activation_ratio < cfg.lar_min_active:
            indicators.append("LAR")
        return indicators

    def _nmg_indicator(self, obs: BTTObservation) -> List[str]:
        cfg = self.config
        loss = obs.loss_history
        if len(loss) < 2:
            return []
        w = min(cfg.window, len(loss))
        window_start = loss[-w]
        last = loss[-1]
        denom = abs(window_start) if window_start != 0 else 1.0
        recent_rel = (window_start - last) / denom
        if recent_rel < cfg.nmg_min_rel_improve:
            return ["NMG"]
        return []

    def diagnose(self, obs: BTTObservation) -> Verdict:
        cfg = self.config

        def make(
            kind: BTTVerdictKind,
            indicators: Sequence[str],
            disposition: BTTDisposition,
            spared: bool = False,
        ) -> Verdict:
            return Verdict(
                kind=kind,
                indicators=tuple(indicators),
                trial_id=obs.trial_id,
                completed_fidelity=obs.completed_fidelity,
                observation_hash=obs.observation_hash,
                profile_version=cfg.profile_version,
                spared_by_reserve=spared,
                disposition=disposition,
            )

        # FATAL bypasses every gate: non-finite training is never salvageable.
        fatal = self._fatal_indicators(obs)
        if fatal:
            return make(BTTVerdictKind.FATAL, fatal, BTTDisposition.RETIRE)

        # Below the minimum-fidelity gate there is not enough evidence to cut.
        if obs.completed_fidelity < cfg.min_fidelity:
            return make(BTTVerdictKind.HEALTHY, (), BTTDisposition.CONTINUE)

        degraded = self._degraded_indicators(obs)
        if degraded:
            if (
                cfg.late_bloomer_reserve > 0.0
                and _uniform_draw(obs.trial_id, cfg.reserve_seed) < cfg.late_bloomer_reserve
            ):
                # Spared: continue this would-be termination to measure false-kill rate.
                return make(
                    BTTVerdictKind.HEALTHY,
                    degraded,
                    BTTDisposition.CONTINUE,
                    spared=True,
                )
            disposition = (
                BTTDisposition.STOP if cfg.mode is BTTMode.PAPER_BINARY else BTTDisposition.RECYCLE
            )
            return make(BTTVerdictKind.DEGRADED, degraded, disposition)

        nmg = self._nmg_indicator(obs)
        if nmg:
            return make(BTTVerdictKind.SATURATED, nmg, BTTDisposition.COMPLETE)

        return make(BTTVerdictKind.HEALTHY, (), BTTDisposition.CONTINUE)
