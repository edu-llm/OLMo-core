import olmo_core.nn.transformer as transformer
from olmo_core.nn.transformer import (
    MoEEngramReorderedNormTransformerBlock,
    MoELngramReorderedNormTransformerBlock,
)
from olmo_core.nn.transformer.block import (
    MoEEngramReorderedNormTransformerBlock as BlockMoEEngramReorderedNormTransformerBlock,
)
from olmo_core.nn.transformer.block import (
    MoELngramReorderedNormTransformerBlock as BlockMoELngramReorderedNormTransformerBlock,
)


def test_memory_blocks_are_publicly_exported():
    assert MoEEngramReorderedNormTransformerBlock is BlockMoEEngramReorderedNormTransformerBlock
    assert MoELngramReorderedNormTransformerBlock is BlockMoELngramReorderedNormTransformerBlock
    assert "MoEEngramReorderedNormTransformerBlock" in transformer.__all__
    assert "MoELngramReorderedNormTransformerBlock" in transformer.__all__
