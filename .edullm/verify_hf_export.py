"""Decide whether a HuggingFace export gathered the experts, from the export alone.

    python .edullm/verify_hf_export.py s3://.../runs/<run>/hf/step4

WHAT THIS IS FOR, AND WHY IT IS NOT A UNIT TEST. ``HFConverterCallback`` gathers the model with
``get_model_state_dict(full_state_dict=True)``. On a dense FSDP model that is a well-travelled
path. On this one the routed experts are DTensors sharded over a SEPARATE expert-parallel mesh
from the one FSDP shards the rest of the model over, and nothing in this repository has ever
checked that the gather understands the second mesh. Every test that touches the MoE mapping
runs in one process, where there is no mesh to misread.

THE FAILURE IS SILENT AND THE FILE IS THE RIGHT SIZE EITHER WAY. ``save_hf_model`` builds an HF
model from the config and calls ``load_state_dict`` on it, so the tensors are whatever shape
``FlexOlmoExperts`` declares no matter what was gathered into them. A gather that returned one
rank's local experts, or the same shard broadcast to every slot, produces a checkpoint that
loads, generates text, and is wrong in a way no downstream consumer can see.

WHAT THIS CHECKS, WHICH NEEDS NEITHER A GPU NOR THE ORIGINAL CHECKPOINT. Two properties of the
exported tensors:

  1. The expert dimension is the full expert count, not the count divided by the mesh degree.
  2. The experts are all DIFFERENT. Two experts of a trained MoE agree to floating-point
     equality only if the same bytes were written twice, and the shape of a mis-gathered
     expert-parallel tensor is exactly that -- one rank's shard repeated with period
     ``num_experts / expert_parallel_degree``, or zeros where a rank's contribution never
     arrived. A repeat count that divides the mesh degree names the bug directly.

Both are read off the safetensors index without instantiating a model, so this runs on a laptop
against the run's own output prefix in the minutes after the smoke test finishes.
"""

import argparse
import collections
import hashlib
import json
import sys
from typing import Dict, List

import torch

from olmo_core.io import file_exists, resource_path


def _local(uri: str, name: str):
    """Fetch one object of the export to somewhere safetensors can mmap it."""
    return resource_path(uri.rstrip("/"), name)


def expert_tensors(directory: str) -> Dict[str, torch.Tensor]:
    """Every stacked per-expert parameter in the export, keyed by its HF name."""
    from safetensors.torch import load_file

    directory = directory.rstrip("/")
    if file_exists(f"{directory}/model.safetensors.index.json"):
        index = json.loads(_local(directory, "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    found: Dict[str, torch.Tensor] = {}
    for shard in shards:
        for name, tensor in load_file(str(_local(directory, shard))).items():
            # The fused layout: one 3D parameter per projection, experts on dimension zero.
            if ".experts." in name and tensor.ndim == 3:
                found[name] = tensor
    return found


def duplicate_groups(tensor: torch.Tensor) -> List[List[int]]:
    """Which experts of one parameter are byte-identical to each other.

    Digested rather than compared pairwise: 32 experts of 8.4M parameters is a 4-billion-element
    comparison per layer done the obvious way, and equality is all that is being asked.
    """
    by_digest: Dict[bytes, List[int]] = collections.defaultdict(list)
    for expert in range(tensor.shape[0]):
        block = tensor[expert].contiguous()
        digest = hashlib.blake2b(block.view(torch.uint8).numpy().tobytes(), digest_size=16).digest()
        by_digest[digest].append(expert)
    return [sorted(group) for group in by_digest.values() if len(group) > 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="The hf/step{N} directory the run wrote.")
    parser.add_argument("--experts", type=int, default=32)
    options = parser.parse_args()

    tensors = expert_tensors(options.export)
    if not tensors:
        print(f"FAIL: {options.export} holds no stacked expert parameters at all")
        return 1

    failures = 0
    zeroed = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.shape[0] != options.experts:
            print(f"FAIL {name}: {tensor.shape[0]} experts on dimension zero, not {options.experts}")
            failures += 1
            continue
        repeats = duplicate_groups(tensor)
        if repeats:
            period = len(repeats[0])
            print(
                f"FAIL {name}: experts repeat -- {repeats[:3]}{' ...' if len(repeats) > 3 else ''}. "
                f"A period of {period} means each rank's shard was written to "
                f"{period} slots, so the gather read one mesh and the experts live on another."
            )
            failures += 1
        if not tensor.any():
            zeroed += 1

    print()
    print(f"{len(tensors)} stacked expert parameters, {options.experts} experts each")
    print(f"all-zero parameters: {zeroed}")
    if zeroed:
        print("FAIL: a parameter of all zeros is a slot no rank's shard ever reached")
        failures += 1
    if failures:
        print(f"FAIL: {failures} checks failed. Do not build the downstream lane on this export.")
        return 1
    print("PASS: every expert slot is filled and no two experts are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
