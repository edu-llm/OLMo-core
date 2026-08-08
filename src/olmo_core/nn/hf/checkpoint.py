import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed as dist
from huggingface_hub import repo_exists
from torch.distributed.tensor import DTensor, distribute_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_full_tensor
from olmo_core.doc_utils import beta_feature
from olmo_core.io import clear_directory, copy_dir, file_exists, is_url
from olmo_core.nn.hf.config import (
    get_hf_config,
    get_hybrid_hf_config,
    get_hybrid_layer_types,
)
from olmo_core.nn.hf.convert import (
    convert_hybrid_state_to_hf,
    convert_state_from_hf,
    convert_state_to_hf,
)
from olmo_core.nn.transformer.model import Transformer

try:
    from accelerate import init_empty_weights  # type: ignore
except ImportError:

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        del include_buffers
        log.warning("accelerate not installed, will initialize weights.")
        yield None


log = logging.getLogger(__name__)

#: The per-expert weights this repository's MoE mapping emits, paired with the fused
#: parameter each contributes to and the order it is concatenated in. ``gate`` before ``up``
#: because ``FlexOlmoExperts.forward`` splits the projection's output with
#: ``.chunk(2, dim=-1)`` and reads the first half as the gate.
_FUSED_MOE_EXPERT_PARTS: Dict[str, tuple[str, ...]] = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "down_proj": ("down_proj",),
}


def _fuse_moe_expert_weights(
    hf_state_dict: Dict[str, torch.Tensor], expected_keys: set
) -> Dict[str, torch.Tensor]:
    """Stack per-expert MoE weights into the single tensor per projection HF now wants.

    THE LAYOUT MOVED UNDER US AND NOTHING IN THIS REPOSITORY NOTICED. ``transformers``
    stores a MoE layer's experts as one 3D parameter per projection --
    ``mlp.experts.gate_up_proj`` of ``(experts, 2 x intermediate, hidden)`` and
    ``mlp.experts.down_proj`` of ``(experts, hidden, intermediate)`` -- and has done since
    before ``5.4.0``, which is the floor this project declares. The mapping in
    :mod:`olmo_core.nn.hf.convert` still emits the ``nn.ModuleList`` form that preceded it,
    one ``mlp.experts.{i}.gate_proj.weight`` per expert. ``load_state_dict`` then reports
    every expert weight as both missing and unexpected and the conversion dies.

    It survived because no test loads a converted MoE state dict into an HF model. The two
    that mention MoE check the generated config and the state mapping, and those agree with
    each other; what they never ask is whether ``transformers`` agrees with either.

    KEYED ON WHAT THE TARGET MODEL ASKS FOR RATHER THAN ON A VERSION TEST. ``expected_keys``
    is the instantiated HF model's own ``state_dict`` keys, so this fuses exactly when the
    model in front of it wants fused parameters and is a no-op when it wants the module
    list. A ``transformers`` that moves back, or a model type that never moved, needs no
    change here.
    """
    fused: Dict[str, torch.Tensor] = {}
    consumed: set = set()
    for expected in expected_keys:
        prefix, _, projection = expected.rpartition(".")
        parts = _FUSED_MOE_EXPERT_PARTS.get(projection)
        if parts is None or not prefix.endswith(".experts"):
            continue

        stacked = []
        expert = 0
        while True:
            keys = [f"{prefix}.{expert}.{part}.weight" for part in parts]
            if not all(key in hf_state_dict for key in keys):
                break
            stacked.append(torch.cat([hf_state_dict[key] for key in keys], dim=0))
            consumed.update(keys)
            expert += 1
        if not stacked:
            continue
        fused[expected] = torch.stack(stacked)

    if not fused:
        return hf_state_dict

    log.info(
        "Fused %d per-expert weights into %d MoE parameters, which is the layout the "
        "installed transformers expects",
        len(consumed),
        len(fused),
    )
    remaining = {key: value for key, value in hf_state_dict.items() if key not in consumed}
    remaining.update(fused)
    return remaining


def _split_moe_expert_weights(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Unstack fused MoE parameters back into the per-expert weights the mapping reads.

    THE FIX FOR THE OUTBOUND LAYOUT LEFT THE INBOUND ONE BROKEN, AND THE INBOUND ONE IS
    THE POST-TRAINING PATH. :func:`_fuse_moe_expert_weights` stacks per-expert tensors on
    the way out because that is what ``transformers`` wants; nothing does the reverse, so
    :func:`convert_state_from_hf` -- which still reads
    ``mlp.experts.{i}.gate_proj.weight`` -- meets a 3D ``mlp.experts.gate_up_proj`` it has
    no template for and raises ``Some state keys were not converted``. Every exported MoE
    checkpoint is therefore write-only, which matters because ``open_instruct``'s
    ``olmo_core_finetune`` reloads weights through :func:`load_hf_model` and nothing else.
    Measured on a 32-expert export on 2026-08-08: eight unconverted keys, four layers,
    both projections.

    KEYED ON THE TENSOR'S RANK RATHER THAN ON A VERSION TEST, for the same reason its
    outbound twin keys on the target model's own parameter names: a ``transformers`` that
    moves back, or a model type that never moved, presents 2D per-expert weights and this
    is then a no-op. The expert count comes from the leading dimension, so nothing here
    needs the config.

    ``gate_up_proj`` splits in half along the fused axis, gate first, which is the order
    the outbound concatenation used and the order ``FlexOlmoExperts.forward`` reads with
    ``.chunk(2, dim=-1)``. Getting it backwards produces a model that loads clean and
    computes the gate from the up projection, which no shape check can catch.
    """
    split: Dict[str, torch.Tensor] = {}
    consumed: set = set()
    for key, value in hf_state_dict.items():
        prefix, _, projection = key.rpartition(".")
        parts = _FUSED_MOE_EXPERT_PARTS.get(projection)
        if parts is None or not prefix.endswith(".experts"):
            continue
        if not isinstance(value, torch.Tensor) or value.dim() != 3:
            continue

        for expert, weights in enumerate(value):
            for part, chunk in zip(parts, weights.chunk(len(parts), dim=0)):
                split[f"{prefix}.{expert}.{part}.weight"] = chunk.contiguous()
        consumed.add(key)

    if not consumed:
        return hf_state_dict

    log.info(
        "Split %d fused MoE parameters into %d per-expert weights, which is the layout "
        "the OLMo-core state mapping reads",
        len(consumed),
        len(split),
    )
    remaining = {key: value for key, value in hf_state_dict.items() if key not in consumed}
    remaining.update(split)
    return remaining


@beta_feature
def undo_dropless_moe_reshape(
    model_state_dict: Dict[str, torch.Tensor], *, num_experts: int
) -> Dict[str, torch.Tensor]:
    """Invert the reshape ``convert_checkpoint_to_hf`` applies to a dropless MoE.

    ``DroplessMoEMLP`` holds ``w1`` and ``w3`` as ``(experts x intermediate, d_model)``
    while the regular MoE holds them transposed, and the state mapping cannot tell the two
    apart, so ``convert_checkpoint_to_hf`` permutes them into the regular layout on its way
    to :func:`save_hf_model`. Nothing undoes that, and :func:`load_hf_model` cannot: the
    permutation is a fact about the OLMo-core module and the only things in scope there are
    a directory and a state dict.

    SO IT LIVES BESIDE THE EXPORT RATHER THAN INSIDE THE IMPORT, WHICH IS WHERE ITS TWIN
    LIVES. A caller reloading a dropless MoE calls this on the state dict
    :func:`load_hf_model` filled, exactly as the exporter calls the forward reshape on the
    state dict it is about to save. The asymmetry of doing it in one place and not the
    other is what left every exported MoE readable only as noise, which is worse than
    unreadable because the shapes match.

    :param model_state_dict: An OLMo-core state dict as :func:`load_hf_model` left it.
    :param num_experts: The routed expert count of the model being loaded into.
    """
    for key, value in list(model_state_dict.items()):
        if not (key.endswith(".experts.mlp.w1") or key.endswith(".experts.mlp.w3")):
            continue
        assert isinstance(value, torch.Tensor), (key, value)
        d_model = value.shape[0] // num_experts
        model_state_dict[key] = (
            value.reshape(num_experts, d_model, -1).permute(0, 2, 1).reshape(-1, d_model)
        )
    return model_state_dict


@beta_feature
def load_hf_model(
    model_name_or_path: PathOrStr,
    model_state_dict: Dict[str, Any],
    *,
    revision: str = "main",
    model_id: Optional[str] = None,
    num_embeddings: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
):
    """
    Loads an OLMo Core model state dict using a model in Hugging Face transformers format.

    :param model_name_or_path: The name of a model in HF Hub or the path to a model saved in HF format.
    :param model_state_dict: The OLMo Core model state dict in which to load HF state.
    :param revision: If ``model_name_or_path`` is the id of a model in HF Hub, then this is the revision
        (branch) of that model. Defaults to "main".
    :param model_id: Deprecated, model-specific mappings are now determined by the model architecture,
        in :mod:`olmo_core.nn.hf.convert`
    :param num_embeddings: The number of embeddings in the OLMo Core model being loaded into,
        defaults to the number of embeddings in the HF model.
    :param process_group: The process group to use for distributed communication.
    :param work_dir: A local directory that can be used for holding temporary state. Required when
        downloading a model from a cloud directory.
    """
    del model_id

    work_dir = f"{work_dir}/hf-tmp" if work_dir is not None else None

    if is_url(model_name_or_path):
        log.warning(
            "Model id or path provided is a remote Hugging Face directory. This may not be suitable for unshared file systems."
        )
        assert work_dir is not None
        assert (
            file_exists(f"{model_name_or_path}/generation_config.json")
            or file_exists(f"{model_name_or_path}/model.safetensors.index.json")
            or file_exists(f"{model_name_or_path}/pytorch_model.bin")
        )

        # Download model to local FS
        if get_fs_local_rank() == 0:
            copy_dir(model_name_or_path, work_dir)
        barrier(group=process_group)
    elif Path(model_name_or_path).is_dir():
        assert (
            file_exists(f"{model_name_or_path}/generation_config.json")
            or file_exists(f"{model_name_or_path}/model.safetensors.index.json")
            or file_exists(f"{model_name_or_path}/pytorch_model.bin")
        )
    elif repo_exists(str(model_name_or_path)):
        log.warning(
            "Model id or path provided is a Hugging Face model id. This may not be suitable for unshared file systems."
        )
    else:
        raise NotImplementedError

    # Warm up the HF local cache by downloading the model on just local rank 0
    if get_fs_local_rank() == 0:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision)
        del hf_model
    barrier(group=process_group)

    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision)
    log.info(f"Loaded hf model: {hf_model}")
    hf_model.resize_token_embeddings(num_embeddings)

    converted_state_dict: Dict[str, torch.Tensor] = convert_state_from_hf(
        hf_model.config,
        _split_moe_expert_weights(hf_model.state_dict()),
        model_type=getattr(hf_model.config, "model_type", None),
    )

    for key in sorted(converted_state_dict.keys()):
        state = converted_state_dict[key]
        olmo_core_state = model_state_dict[key]
        if isinstance(olmo_core_state, DTensor):
            olmo_core_state = distribute_tensor(
                state, olmo_core_state.device_mesh, olmo_core_state.placements
            )
        else:
            olmo_core_state = state

        model_state_dict[key] = olmo_core_state

    if work_dir:
        clear_directory(work_dir)


@beta_feature
def save_hf_model(
    save_dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    huggingface_tokenizer: Optional[AutoTokenizer] = None,
    *,
    dtype: Optional[DType] = None,
    vocab_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    work_dir: Optional[PathOrStr] = None,
    save_overwrite: bool = False,
):
    """
    Saves an OLMo Core model state dict in Hugging Face transformers format.

    :param save_dir: Directory in which to save model.
    :param model_state_dict: The OLMo Core model state dict being saved in HF format.
    :param dtype: The torch dtype that model weights should be saved as.
    :param vocab_size: The size of the vocab, defaults to the number of embeddings in the OLMo Core model.
    :param process_group: The process group to use for distributed communication.
    :param work_dir: A local directory that can be used for holding temporary state. Required when
        downloading a model from a cloud directory.
    :param save_overwrite: Overwrite existing files in ``save_dir``.
    """

    hf_config = get_hf_config(model)

    model_state_dict = {key: get_full_tensor(state) for key, state in model_state_dict.items()}
    if dtype is not None:
        model_state_dict = {
            key: state.to(dtype=dtype.as_pt()) for key, state in model_state_dict.items()
        }

    hf_state_dict: Dict[str, torch.Tensor] = convert_state_to_hf(hf_config, model_state_dict)

    # model.save_pretrained fails says `tensor.reshape()` should be used instead of `tensor.view()`
    # if we do not make the state contiguous. Unfortunately this is bad for perf.
    hf_state_dict = {key: state.contiguous() for key, state in hf_state_dict.items()}

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        hf_model = AutoModelForCausalLM.from_config(hf_config)
        del hf_config

    hf_state_dict = _fuse_moe_expert_weights(hf_state_dict, set(hf_model.state_dict()))

    hf_model.load_state_dict(hf_state_dict, assign=True)

    hf_model.config.vocab_size = vocab_size or model.vocab_size
    hf_model.resize_token_embeddings(hf_model.config.vocab_size)
    hf_model.generation_config.do_sample = True

    if huggingface_tokenizer is not None:
        hf_model.generation_config.eos_token_id = huggingface_tokenizer.convert_tokens_to_ids(
            ["<|im_end|>", "<|endoftext|>"]
        )
        hf_model.generation_config.pad_token = huggingface_tokenizer.pad_token_id

    if get_fs_local_rank(process_group) == 0:
        if is_url(save_dir):
            assert work_dir is not None
            hf_model.save_pretrained(work_dir)

            copy_dir(work_dir, save_dir, save_overwrite=save_overwrite)
        else:
            target = Path(save_dir)
            if target.is_dir() and not save_overwrite:
                raise FileExistsError(target)
            target.parent.mkdir(exist_ok=True, parents=True)
            hf_model.save_pretrained(target)


@beta_feature
def save_hf_hybrid_model(
    save_dir: PathOrStr,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    *,
    dtype: Optional[DType] = None,
    vocab_size: Optional[int] = None,
    max_sequence_length: int = 65536,
) -> None:
    """
    Save a hybrid (GDN + attention) model as ``config.json`` + ``model.safetensors``.

    Unlike :func:`save_hf_model`, this writes files directly to avoid a hard dependency
    on a specific ``transformers`` version.

    :param save_dir: Directory in which to save the model.
    :param model_state_dict: The OLMo-core model state dict.
    :param model: The OLMo-core hybrid transformer model.
    :param dtype: Optional dtype to cast weights to.
    :param vocab_size: If set, truncate embeddings/lm_head to this size.
    :param max_sequence_length: Maximum sequence length for ``max_position_embeddings``.
    """
    import json

    from safetensors.torch import save_file

    layer_types = get_hybrid_layer_types(model)
    hf_config = get_hybrid_hf_config(model, layer_types, max_seq_len=max_sequence_length)

    model_state_dict = {key: get_full_tensor(state) for key, state in model_state_dict.items()}
    hf_state = convert_hybrid_state_to_hf(model_state_dict, layer_types)

    if dtype is not None:
        hf_state = {
            k: v.to(dtype.as_pt()) if torch.is_tensor(v) else v for k, v in hf_state.items()
        }

    if vocab_size is not None:
        hf_config["vocab_size"] = vocab_size
        if "model.embed_tokens.weight" in hf_state:
            hf_state["model.embed_tokens.weight"] = hf_state["model.embed_tokens.weight"][
                :vocab_size
            ]
        if "lm_head.weight" in hf_state:
            hf_state["lm_head.weight"] = hf_state["lm_head.weight"][:vocab_size]

    log.info(f"Converted state dict has {len(hf_state)} keys")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config_path = save_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    log.info(f"Saved config to {config_path}")

    save_file(hf_state, save_path / "model.safetensors")
    log.info(f"Saved weights to {save_path / 'model.safetensors'}")
