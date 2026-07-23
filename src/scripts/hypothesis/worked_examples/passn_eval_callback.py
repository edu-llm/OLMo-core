"""Trainer callback: record eval/pass_at_n and eval/pass_ratio_at_n for W&B.

Heavy generation is optional. By default this callback records placeholder-ready
metric slots at ``post_train`` from an offline predictions file, or runs a
lightweight exact-match probe when ``predictions_path`` / generator is supplied.

Metric names are whitespace-free so `/submit-edullm-job` Success metrics accept them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks import Callback

try:
    from holdout_passn import load_holdout, log_metrics_to_wandb, score_generations
except ImportError:  # package-style
    from scripts.hypothesis.worked_examples.holdout_passn import (  # type: ignore
        load_holdout,
        log_metrics_to_wandb,
        score_generations,
    )

log = logging.getLogger(__name__)


@dataclass
class PassNEvalCallback(Callback):
    """Log Pass@N / PassRatio@N into the trainer metric stream (→ WandBCallback)."""

    holdout_path: Optional[str] = None
    predictions_path: Optional[str] = None
    n_samples: int = 8
    max_items: int = 0
    eval_every_steps: int = 0  # 0 = only post_train
    wandb_entity: str = "eduLLM"
    wandb_project: str = "pretraining"
    wandb_group: Optional[str] = None
    enabled: bool = True

    def _maybe_score(self) -> None:
        if not self.enabled or get_rank() != 0:
            return
        if not self.holdout_path or not self.predictions_path:
            log.info(
                "PassNEvalCallback: skip scoring (set holdout_path + predictions_path). "
                "Expected metrics: eval/pass_at_n, eval/pass_ratio_at_n"
            )
            return
        holdout = Path(self.holdout_path)
        preds = Path(self.predictions_path)
        if not holdout.exists() or not preds.exists():
            log.warning("PassNEvalCallback: missing holdout or predictions file")
            return

        items = load_holdout(holdout, max_items=self.max_items)
        generations: List[List[str]] = []
        with preds.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                gens = [str(x) for x in row["gens"]]
                if self.n_samples and len(gens) > self.n_samples:
                    gens = gens[: self.n_samples]
                generations.append(gens)
        metrics = score_generations(items[: len(generations)], generations)
        # Trainer metric stream (picked up by WandBCallback.log_metrics).
        self.trainer.record_metric("pass_at_n", metrics["eval/pass_at_n"], namespace="eval")
        self.trainer.record_metric(
            "pass_ratio_at_n", metrics["eval/pass_ratio_at_n"], namespace="eval"
        )
        log.info(
            "PassNEvalCallback: eval/pass_at_n=%.4f eval/pass_ratio_at_n=%.4f",
            metrics["eval/pass_at_n"],
            metrics["eval/pass_ratio_at_n"],
        )
        # Also push directly in case collect interval skips the last step.
        log_metrics_to_wandb(
            metrics,
            entity=self.wandb_entity,
            project=self.wandb_project,
            group=self.wandb_group,
        )

    def post_step(self):
        if self.eval_every_steps and self.step > 0 and self.step % self.eval_every_steps == 0:
            self._maybe_score()

    def post_train(self):
        self._maybe_score()
