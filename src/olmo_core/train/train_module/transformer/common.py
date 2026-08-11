import logging
from typing import List, Optional, TypeVar, cast

import torch
from torch.distributed import DeviceMesh

from olmo_core.distributed.parallel import (
    DataParallelType,
    get_cp_mesh,
    get_device_mesh_info,
    get_dp_model_mesh,
    get_ep_mesh,
    get_pp_mesh,
    get_tp_mesh,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import MoETransformer, Transformer

from .config import (
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTensorParallelConfig,
)

log = logging.getLogger(__name__)


M = TypeVar("M", Transformer, List[Transformer])


def parallelize_model(
    model: M,
    *,
    world_mesh: Optional[DeviceMesh],
    device: torch.device,
    max_sequence_length: Optional[int] = None,
    rank_microbatch_size: Optional[int] = None,
    compile_model: bool = False,
    float8_config: Optional[Float8Config] = None,
    dp_config: Optional[TransformerDataParallelConfig] = None,
    tp_config: Optional[TransformerTensorParallelConfig] = None,
    cp_config: Optional[TransformerContextParallelConfig] = None,
    ep_config: Optional[TransformerExpertParallelConfig] = None,
    ac_config: Optional[TransformerActivationCheckpointingConfig] = None,
    pp_enabled: bool = False,
) -> M:
    model_parts: List[Transformer] = [model] if isinstance(model, Transformer) else model

    pp_mesh: Optional[DeviceMesh] = None
    if pp_enabled:
        assert world_mesh is not None
        pp_mesh = get_pp_mesh(world_mesh)
        for m in model_parts:
            m.apply_pp(pp_mesh)

    # Maybe apply FP8 training.
    if float8_config is not None and float8_config.enabled:
        for m in model_parts:
            m.apply_fp8(float8_config)
            log.info("Swapped linear layers to Float8 linear layers\n%s", m)

    # Maybe apply context parallelism.
    if cp_config is not None:
        assert world_mesh is not None
        cp_mesh = get_cp_mesh(world_mesh)
        for m in model_parts:
            m.apply_cp(cp_mesh, ring=cp_config.ring, uly=cp_config.uly)
        log.info(f"Applied context parallelism to the model with {get_device_mesh_info(cp_mesh)}")

    # Maybe apply tensor.
    if tp_config is not None:
        if ep_config is not None:
            raise NotImplementedError("TP + EP is not implemented yet")
        assert world_mesh is not None
        tp_mesh = get_tp_mesh(world_mesh)
        for m in model_parts:
            m.apply_tp(tp_mesh)
        tp_config.maybe_enable_async_tp(tp_mesh)
        log.info(f"Applied tensor parallelism to the model with {get_device_mesh_info(tp_mesh)}")

    # Maybe apply expert parallelism.
    if ep_config is not None:
        assert world_mesh is not None
        ep_mesh = get_ep_mesh(world_mesh)
        for m in model_parts:
            if not m.is_moe:
                raise OLMoConfigurationError("Expert parallelism is only valid for MoE models")
            cast(MoETransformer, m).apply_ep(ep_mesh)
        log.info(f"Applied expert parallelism to the model with {get_device_mesh_info(ep_mesh)}")

    # Maybe apply activation checkpointing.
    if ac_config is not None:
        for m in model_parts:
            m.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
                activation_memory_budget=ac_config.activation_memory_budget,
                determinism_check=ac_config.determinism_check,
            )
        log.info(f"Applied '{ac_config.mode}' activation checkpointing to the model")

    # Maybe compile.
    if compile_model:
        if torch.cuda.is_available():
            for m in model_parts:
                m.apply_compile()
            log.info("Applied torch.compile() to the model")
        else:
            log.warning("Skipping model compilation since CUDA is not available")

    # Maybe convert quantized weights to all-gather as packed ternary. This MUST happen before
    # `fully_shard` and before `prepare_experts_for_fsdp`: FSDP2 discovers the extension hooks by
    # `hasattr` on the sharded local tensor it constructs at `fully_shard` time
    # (torch 2.9.0 `_fsdp_param.py:678`), so a swap afterwards is silently invisible and the run
    # would all-gather bf16 while reporting itself compressed. Same ordering constraint torchao's
    # float8 conversion documents at `olmo_core/float8/__init__.py:73-76`.
    #
    # Gated on a per-model-part flag so the default path is untouched: with `ternary_comm=False`
    # nothing below runs and behaviour is bitwise what it was.
    if dp_config is not None and any(
        getattr(m, "ternary_comm_enabled", False) for m in model_parts
    ):
        from olmo_core.nn.quantization_comm import apply_ternary_comm

        for m in model_parts:
            if not getattr(m, "ternary_comm_enabled", False):
                continue
            converted = apply_ternary_comm(m)
            if not converted:
                raise OLMoConfigurationError(
                    "ternary_comm=True converted no parameters. That means the model has no "
                    "enabled quantized weights, so the compressed all-gather would be a no-op "
                    "while the config records it as active. Set quantize=True or ternary_comm="
                    "False."
                )
            log.info(
                "Ternary-compressed all-gather enabled on %d parameter(s)", len(converted)
            )

    # Maybe shard/replicate according to data parallel config.
    if dp_config is not None:
        assert world_mesh is not None
        dp_mesh = get_dp_model_mesh(world_mesh)
        param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
        if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
            for m in model_parts:
                if m.is_moe:
                    cast(MoETransformer, m).prepare_experts_for_fsdp(
                        world_mesh,
                        param_dtype=param_dtype,
                        reduce_dtype=dp_config.reduce_dtype.as_pt(),
                        pp_enabled=pp_enabled,
                    )
                m.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=pp_enabled,
                    prefetch_factor=dp_config.prefetch_factor,
                )
            log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
        elif dp_config.name == DataParallelType.ddp:
            for m in model_parts:
                if m.is_moe:
                    cast(MoETransformer, m).prepare_experts_for_ddp(world_mesh)
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
        else:
            raise NotImplementedError(dp_config.name)

    # Materialize and init parameters.
    log.info("Initializing model weights...")
    for model_part_idx, m in enumerate(model_parts):
        m.init_weights(
            max_seq_len=max_sequence_length,
            max_local_microbatch_size=rank_microbatch_size,
            device=device,
            world_mesh=world_mesh,
            model_part_idx=model_part_idx,
        )

    return model
