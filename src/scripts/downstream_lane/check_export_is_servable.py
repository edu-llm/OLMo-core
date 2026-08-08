"""Decide whether a checkpoint can be served before a machine is paid for to find out.

THE ARCHITECTURE STRING IS THE WHOLE QUESTION AND IT IS ANSWERABLE WITHOUT WEIGHTS.
vLLM dispatches on the single name in ``config.json``'s ``architectures``, matched
exactly against its own registry. An architecture it implements under a different
spelling fails at load exactly as one it never implemented, and the failure arrives after
the image has been pulled and the GPU allocated. This reads the name off a build on a
meta device -- no weights, no CUDA, a fraction of a second -- and compares it against the
registry of the installed vLLM.

Run it against the recipe before a training run rather than against the export after one.
An export that cannot be served is a discovery worth making on the day the recipe is
chosen; discovered on the export it is a model definition to write with the presentation
already scheduled.

    python src/scripts/downstream_lane/check_export_is_servable.py --factory olmoe_7b_32x4
    python src/scripts/downstream_lane/check_export_is_servable.py --exported-dir /path/to/hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def architecture_of_an_exported_directory(directory: Path) -> str:
    written = json.loads((directory / "config.json").read_text())
    architectures = written.get("architectures") or []
    if len(architectures) != 1:
        raise SystemExit(
            f"{directory}/config.json names {architectures}, and vLLM dispatches on exactly one"
        )
    return str(architectures[0])


def architecture_of_a_model_factory(factory: str) -> str:
    """What ``save_pretrained`` would write for a model this repository can build.

    Resolved through ``MODEL_FOR_CAUSAL_LM_MAPPING`` rather than by string surgery on the
    config class name, because that mapping is what ``AutoModelForCausalLM.from_config``
    consults and therefore what decides the class name ``save_pretrained`` records.
    """
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    from olmo_core.nn.hf.config import get_hf_config
    from olmo_core.nn.transformer.config import TransformerConfig

    if not hasattr(TransformerConfig, factory):
        raise SystemExit(
            f"TransformerConfig has no {factory!r}. A recipe that lives as a function inside "
            "a platform entry point is not reachable by name, which is also why open-instruct "
            "cannot post-train from it -- see guides/the-downstream-lane.md."
        )
    # 256 is a vocabulary, not the vocabulary. Nothing about the architecture name depends
    # on it and a real one would only slow the build down.
    model = getattr(TransformerConfig, factory)(256).build(init_device="meta")
    return MODEL_FOR_CAUSAL_LM_MAPPING[type(get_hf_config(model))].__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--exported-dir", type=Path)
    source.add_argument("--factory")
    arguments = parser.parse_args()

    if arguments.exported_dir is not None:
        architecture = architecture_of_an_exported_directory(arguments.exported_dir)
    else:
        architecture = architecture_of_a_model_factory(arguments.factory)

    from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS

    registered = _TEXT_GENERATION_MODELS.get(architecture)
    print(f"architectures = {architecture}")
    if registered is None:
        print(f"vLLM does not register {architecture}. This checkpoint cannot be served.")
        sys.exit(1)
    print(f"vLLM serves it natively from vllm.model_executor.models.{registered[0]}")


if __name__ == "__main__":
    main()
