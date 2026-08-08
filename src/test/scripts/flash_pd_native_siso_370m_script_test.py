import importlib.util
from pathlib import Path

from olmo_core.config import DType
from olmo_core.nn.flash_pd_native import (
    NativeFlashPDMamba3SISOMixer,
    NativeFlashPDMamba3SISOMixerConfig,
)
from olmo_core.nn.mamba3 import Mamba3Config

SCRIPT = Path("src/scripts/train/OLMo3/OLMo3-370M-flash-pd-native-siso.py")


def _load_script():
    assert SCRIPT.exists()
    spec = importlib.util.spec_from_file_location("flash_pd_native_siso_370m", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_siso_factory_replaces_every_recurrent_slot_without_mimo_or_convolution():
    module = _load_script()
    baseline = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    before = baseline.as_config_dict()

    integrated = module.integrate_native_siso(baseline)

    assert baseline.as_config_dict() == before
    assert integrated.block_pattern == [
        "flash_pd_native_mamba3_siso",
        "flash_pd_native_mamba3_siso",
        "flash_pd_native_mamba3_siso",
        "attn",
    ]
    assert isinstance(integrated.block, dict)
    block = integrated.block["flash_pd_native_mamba3_siso"]
    mixer = block.sequence_mixer
    assert isinstance(mixer, NativeFlashPDMamba3SISOMixerConfig)
    assert (mixer.n_heads, mixer.d_state, mixer.dictionary_size) == (16, 64, 16)
    assert mixer.chunk_size == 64
    assert mixer.backend.value == "cuda"
    assert mixer.mode.value == "general_scatter"
    assert mixer.dtype == DType.bfloat16
    assert mixer.fuse_input_projections is True
    assert block.feed_forward.hidden_size == module.PARAMETER_MATCHED_RECURRENT_FF_DIM
    built = mixer.build(1024, layer_idx=0, n_layers=16, init_device="meta")
    assert isinstance(built, NativeFlashPDMamba3SISOMixer)
    assert not hasattr(built, "mimo_rank")
    assert built.conv_kernel_size is None


def test_370m_factory_is_parameter_matched_and_config_count_matches_meta_model():
    module = _load_script()
    baseline = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    integrated = module.integrate_native_siso(baseline)
    report = module.parameter_match_report(baseline, integrated)

    assert report["target_non_embedding_parameters"] == baseline.num_non_embedding_params
    assert report["actual_non_embedding_parameters"] == integrated.num_non_embedding_params
    assert report["difference"] == (
        integrated.num_non_embedding_params - baseline.num_non_embedding_params
    )
    assert abs(report["relative_difference"]) < 0.001
    meta_model = integrated.build(init_device="meta")
    assert integrated.num_params == sum(parameter.numel() for parameter in meta_model.parameters())


def test_entrypoint_is_native_only_and_fp8_excludes_the_fused_sensitive_projection():
    module = _load_script()
    source = SCRIPT.read_text()
    config = module.integrate_native_siso(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))
    ignored = module.native_fp8_modules_to_ignore(config)

    assert "flash_pd_ssm" not in source
    assert "mamba3_flash" not in source
    assert len(ignored) == 12
    assert all(name.endswith(".in_proj") for name in ignored)
