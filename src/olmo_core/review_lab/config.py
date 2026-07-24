from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping

Condition = Literal[
    "no_review",
    "uniform",
    "expanding",
    "cramming",
    "adaptive_mix",
    "adaptive_due",
    "forever_full",
]


@dataclass
class ModelConfig:
    """Model-loading settings.

    ``allenai/OLMo-1B-hf`` is AI2's official Transformers conversion of the original
    ``allenai/OLMo-1B`` base checkpoint. It is not an instruct model.
    """

    name: str = "allenai/OLMo-1B-hf"
    revision: str = "aee7752d9c08ee4775e9b0091426d8410e8f6a89"
    backend: Literal["auto", "hf_olmo"] = "auto"
    trust_remote_code: bool = False
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    attention: Literal["auto", "sdpa", "flash_attention_2", "eager"] = "auto"
    gradient_checkpointing: bool = True
    adaptation: Literal["full", "lora"] = "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class DataConfig:
    dataset: Literal["micro_world", "fictionalqa", "fictionalqa_interference"] = "micro_world"
    path: str = "data/review_lab/micro_world.jsonl"
    seed: int = 20260721
    source_dataset: str = "jwkirchenbauer/fictionalqa"
    source_revision: str = "131cb74fdc3e601b5e896ed768ad9852ea35a8f9"
    source_config: str = "fict_qa"
    old_events: int = 20
    new_events: int = 40
    facts_per_skill: int = 64
    new_facts_per_skill: int = 96
    train_paraphrases_per_fact: int = 2
    eval_paraphrases_per_fact: int = 2
    eval_facts_per_skill: int = 16


@dataclass
class TrainingConfig:
    output_dir: str = "runs/review_lab"
    sequence_length: int = 256
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 8
    stage1_steps: int = 60
    stage2_steps: int = 180
    buffer_steps: int = 0
    buffer_eval_delays: List[int] = field(default_factory=list)
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    eval_interval: int = 15
    eval_examples_per_skill: int = 16
    final_eval_examples_per_skill: int = 0
    minimum_mastered_facts: int = 100
    regression_loss_margin: float = 0.5
    log_interval: int = 5
    save_stage1: bool = True
    save_final_model: bool = False
    compile_model: bool = False
    num_workers: int = 0


@dataclass
class ReviewConfig:
    events: int = 12
    first_step: int = 8
    last_step: int = 165
    expansion_ratio: float = 2.0
    adaptive_opportunity_interval: int = 10
    loss_delta_threshold: float = 0.08
    min_gap: int = 3


@dataclass
class ExperimentConfig:
    experiment_name: str = "olmo1b-micro-world"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    conditions: List[Condition] = field(
        default_factory=lambda: [
            "no_review",
            "uniform",
            "expanding",
            "cramming",
            "adaptive_mix",
            "adaptive_due",
        ]
    )
    seeds: List[int] = field(default_factory=lambda: [17])

    def validate(self) -> None:
        if not self.conditions:
            raise ValueError("At least one condition is required")
        unknown = set(self.conditions) - {
            "no_review",
            "uniform",
            "expanding",
            "cramming",
            "adaptive_mix",
            "adaptive_due",
            "forever_full",
        }
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        if not self.seeds:
            raise ValueError("At least one paired seed is required")
        if self.training.stage1_steps < 1 or self.training.stage2_steps < 1:
            raise ValueError("Both training stages need at least one step")
        if self.training.buffer_steps < 0:
            raise ValueError("buffer_steps must be non-negative")
        if any(delay < 1 or delay > self.training.buffer_steps for delay in self.training.buffer_eval_delays):
            raise ValueError("buffer_eval_delays must fall inside the post-review buffer")
        if len(set(self.training.buffer_eval_delays)) != len(self.training.buffer_eval_delays):
            raise ValueError("buffer_eval_delays must be unique")
        if self.review.events < 0:
            raise ValueError("review.events must be non-negative")
        if self.review.events:
            if not 0 <= self.review.first_step <= self.review.last_step:
                raise ValueError("Review first_step must be <= last_step")
            if self.review.last_step >= self.training.stage2_steps:
                raise ValueError("Review last_step must be inside stage 2")
            available = self.review.last_step - self.review.first_step + 1
            if self.review.events > available:
                raise ValueError("More review events requested than available steps")
        if self.training.sequence_length < 32:
            raise ValueError("sequence_length is unexpectedly small")
        if self.training.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be positive")
        if self.training.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be positive")
        if self.training.minimum_mastered_facts < 0:
            raise ValueError("minimum_mastered_facts must be non-negative")
        if self.training.regression_loss_margin <= 0:
            raise ValueError("regression_loss_margin must be positive")
        if not self.training.save_stage1:
            raise ValueError(
                "save_stage1 must be true because all conditions reuse that checkpoint"
            )
        if self.model.adaptation == "lora" and self.model.lora_rank < 1:
            raise ValueError("lora_rank must be positive")
        if self.data.dataset in {"fictionalqa", "fictionalqa_interference"}:
            if self.data.source_config != "fict_qa":
                raise ValueError("The FictionalQA adapter currently requires source_config=fict_qa")
            if self.data.old_events < 1:
                raise ValueError("FictionalQA needs positive old_events")
            if self.data.dataset == "fictionalqa" and self.data.new_events < 1:
                raise ValueError("FictionalQA needs positive old_events and new_events")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        config = self.as_dict()
        # Execution selection and filesystem location do not change a run's scientific recipe.
        config.pop("conditions", None)
        config.pop("seeds", None)
        config["data"].pop("path", None)
        config["training"].pop("output_dir", None)
        config["training"].pop("save_final_model", None)
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def resolve_paths(self, repo_root: Path) -> "ExperimentConfig":
        data_path = Path(self.data.path)
        output_path = Path(self.training.output_dir)
        if not data_path.is_absolute():
            self.data.path = str(repo_root / data_path)
        if not output_path.is_absolute():
            self.training.output_dir = str(repo_root / output_path)
        return self


def _construct_dataclass(cls, raw: Mapping[str, Any]):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(raw))


def load_experiment_config(
    path: str | Path, repo_root: str | Path | None = None
) -> ExperimentConfig:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a YAML mapping")

    allowed = {"experiment_name", "model", "data", "training", "review", "conditions", "seeds"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config fields: {sorted(unknown)}")

    config = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "olmo1b-micro-world"),
        model=_construct_dataclass(ModelConfig, raw.get("model", {})),
        data=_construct_dataclass(DataConfig, raw.get("data", {})),
        training=_construct_dataclass(TrainingConfig, raw.get("training", {})),
        review=_construct_dataclass(ReviewConfig, raw.get("review", {})),
        conditions=list(raw.get("conditions", ExperimentConfig().conditions)),
        seeds=list(raw.get("seeds", [17])),
    )
    root = Path(repo_root) if repo_root is not None else config_path.resolve().parents[2]
    config.resolve_paths(root)
    config.validate()
    return config
