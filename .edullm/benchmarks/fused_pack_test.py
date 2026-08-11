#!/usr/bin/env python
"""GPU unittest for the fused native-pack BF16 materialization."""

import os
import sys
import unittest

_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if os.path.isdir(_REPO_SRC):
    sys.path.insert(0, _REPO_SRC)

import torch

from olmo_core.kernels import ternary as ternary_kernels
from olmo_core.ops.ternary import PackedTWNCache, dequantize_packed_twn, pack_twn_reference


class FusedPackTest(unittest.TestCase):
    """Verify that forward-only packing emits the exact logical BF16 weight."""

    def test_forward_only_pack_fuses_exact_bf16_materialization(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.manual_seed(21)
        weight = torch.randn(5, 37, 65, device="cuda", dtype=torch.bfloat16)
        expected = dequantize_packed_twn(pack_twn_reference(weight, 2))

        actual = ternary_kernels.pack_twn_forward_only(weight, 2)

        self.assertTrue(torch.equal(actual.materialized, expected))
        self.assertEqual(actual.codes_t.numel(), 0)

    def test_cache_reuses_fused_pack_storage_after_invalidation(self) -> None:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        torch.manual_seed(22)
        weight = torch.randn(5, 37, 65, device="cuda", dtype=torch.bfloat16)
        cache = PackedTWNCache()
        first = cache.get_or_pack(
            weight,
            in_dim=2,
            orientation="reuse",
            packer=ternary_kernels.pack_twn_forward_only,
        )
        assert first.materialized is not None
        pointers = (first.codes.data_ptr(), first.alpha.data_ptr(), first.materialized.data_ptr())

        with torch.no_grad():
            weight.add_(0.5)
        cache.clear()
        second = cache.get_or_pack(
            weight,
            in_dim=2,
            orientation="reuse",
            packer=ternary_kernels.pack_twn_forward_only,
        )
        expected = dequantize_packed_twn(pack_twn_reference(weight, 2))

        assert second.materialized is not None
        self.assertEqual(
            (second.codes.data_ptr(), second.alpha.data_ptr(), second.materialized.data_ptr()),
            pointers,
        )
        self.assertTrue(torch.equal(second.materialized, expected))


if __name__ == "__main__":
    unittest.main()
