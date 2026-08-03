"""Runtime dependencies required by the platform-specific training path."""

from pathlib import Path


DOCKERFILE = Path(".edullm/Dockerfile")
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
FLASH_ATTN_SHA256 = "4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810"
TRANSFORMERS_VERSION = "5.14.1"
TOKENIZERS_VERSION = "0.22.2"


def test_research_image_installs_matching_binary_flash_attention():
    """The registered base has no nvcc, so an sdist can never be a fallback."""
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert FLASH_ATTN_WHEEL in source
    assert f"#sha256={FLASH_ATTN_SHA256}" in source
    assert '"einops==0.8.2"' in source
    assert "--no-deps" in source
    assert source.index('"torch==2.9.0"') < source.index(FLASH_ATTN_WHEEL)


def test_research_image_asserts_the_varlen_api_used_by_packed_rows():
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert "flash_attn_varlen_func" in source
    assert "flash_attn_varlen_qkvpacked_func" in source


def test_research_image_installs_and_imports_qwen_runtime_dependencies():
    """Separator resolution and pretrained loading use different HF packages."""
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert f'"transformers=={TRANSFORMERS_VERSION}"' in source
    assert f'"tokenizers=={TOKENIZERS_VERSION}"' in source
    assert "from tokenizers import Tokenizer" in source
    assert "from transformers import AutoModelForCausalLM" in source
    assert source.index('"torch==2.9.0"') < source.index(
        f'"transformers=={TRANSFORMERS_VERSION}"'
    )
