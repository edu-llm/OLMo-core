"""Refuse this image if it cannot supply an attention backend the model ladder names.

Run by the platform's ``tools/verify_image_self_check.py`` inside the assembled image, after
the last layer and before the push, with no network and no GPU. The platform supplies the
place and those guarantees; this file is the question, because which factories exist and what
they ask for are facts about this repository.

THE FAILURE IT EXISTS FOR. Run ``run_019fde30-1d27-7096-8bd9-3ef9b7748d7b`` waited four and a
half minutes for a ``gpu-1xa10g``, started, and died eleven seconds later:

    RuntimeError: 'FlashAttention2Backend' is missing the flash-attn package or is not
    supported on this platform.
      File ".../olmo_core/nn/attention/__init__.py", line 470, in __init__
        backend.assert_supported()

Every ``olmo3_*`` factory pins ``attn_backend=flash_2``, ``Attention.__init__`` calls
``assert_supported()`` while the model is being constructed, and the image had no flash-attn.
Not a slow path -- the model could not be instantiated. This file makes that a red build.

WHY IT ASKS THE BACKENDS AND NOT THE MODELS, WHICH IS THE WHOLE SUBTLETY OF CHECKING THIS
WITHOUT A CARD. The obvious check is to build every rung on the meta device and let the
constructor raise. It does not work here, and it is worse than not checking, because it goes
green either way. Four lines above the assertion that killed that run:

    if not torch.cuda.is_available() and backend != AttentionBackendName.torch:
        warnings.warn(f"Backend is set to {backend}, but GPUs are not available. ...")
        backend = AttentionBackendName.torch

    backend.assert_supported()

A build runner has no device, so a constructed ``olmo3_*`` model is quietly downgraded to the
torch backend and never asks whether flash-attn is installed. That is not a hypothetical: the
Dockerfile's own assertion block constructs ``olmo2_190M`` and stayed green for the entire
period in which no ``olmo3_*`` model could be built at all.

So this reads the backend each factory *names* -- calling a factory returns a config dataclass
and never reaches ``Attention.__init__``, so nothing is downgraded -- and then calls that
backend's own ``assert_supported()`` directly. That predicate is device-free for the backends
asserted below, so the answer here is the answer on the GPU.

WHAT IT CANNOT SEE, which is worth as much as what it can.

- **That the kernel runs.** ``assert_supported()`` for flash-attn is ``flash_attn_2 is not
  None``: the package imported. A wheel built for the wrong compute capability satisfies that
  and still fails on the first forward pass. Only a GPU can answer that, which is what the
  ``olmo-core-check`` workload profile is for.
- **Anything past construction.** Data loading, checkpoint writes, the optimizer, distributed
  init. This file is about the eleven seconds before step 1.
- **Backends whose support test reads the device**, listed in ``DEVICE_DEPENDENT`` below and
  skipped by name.
- **Factories outside the prefixes below.** ``.edullm/train_on_corpus.py`` resolves
  ``--model-factory`` with ``getattr(TransformerConfig, ...)``, so ``llama_*``, ``qwen3_*`` and
  the rest are reachable by a submission and are not covered here. They are excluded because
  several are hybrids whose backends are not the ones this repository's rungs use, not because
  they are safe. Widening the prefixes is the fix if somebody trains one.
- **A rung that names no backend.** ``olmo2_*`` leaves ``attn_backend`` at ``None``, which
  ``Attention.__init__`` resolves to ``torch``, and torch is supported everywhere. Those rungs
  work today and this file has nothing to prove about them.
"""

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from typing import Any

from olmo_core.data import TokenizerConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

#: The model ladders this repository is registered to train. A prefix rather than a list of
#: names, because a list is correct on the day it is typed and silently short by one the day
#: somebody adds a rung -- which is failing open, and failing open is the thing this file
#: exists to stop happening one level up.
FACTORY_PREFIXES = ("olmo2_", "olmo3_")

#: Backends whose ``assert_supported()`` is answerable on a machine with no GPU, so that
#: asking here gives the same answer the training node would give. Each was read out of
#: ``olmo_core/nn/attention/`` rather than assumed:
#:
#: - ``torch`` asserts nothing; SDPA is always present.
#: - ``flash_2`` is ``has_flash_attn_2()``, which is ``flash_attn_2 is not None`` against a
#:   module-level import and nothing else. Loading its compiled extension needs no driver:
#:   the extension links the CUDA *runtime* libraries out of the torch wheel and never
#:   ``libcuda.so.1``.
#: - ``te`` is ``has_te_attn()``, which is ``te is not None``. Same shape.
DEVICE_FREE = frozenset(
    {
        AttentionBackendName.torch,
        AttentionBackendName.flash_2,
        AttentionBackendName.te,
    }
)

#: Backends this file refuses to judge, because their support test reads the device.
#: ``has_flash_attn_3`` and ``has_flash_attn_4`` call ``torch.cuda.get_device_capability()``
#: and gate on Hopper and Blackwell respectively. On this runner there is no device, so they
#: fall back to a bare import check and answer a *different question* than the one that
#: matters -- an image that passes here can still be wrong on the node, and the reverse.
#: Asserting them would also redden every build of a correct image the moment a factory named
#: one, which is the failure this check must never introduce. The right place for them is the
#: ``olmo-core-check`` workload profile: a short run on the smallest GPU shape, where there is
#: a real card to ask.
DEVICE_DEPENDENT = frozenset(
    {
        AttentionBackendName.flash_3,
        AttentionBackendName.flash_4,
    }
)


def registered_factories() -> list[str]:
    """The rung names, read off the class rather than written down here."""
    return sorted(name for name in dir(TransformerConfig) if name.startswith(FACTORY_PREFIXES))


def named_backends(config: Any) -> set[AttentionBackendName]:
    """Every backend reachable from a config, wherever in it the attention lives.

    A walk rather than ``config.block.attention.backend`` because ``block`` is a
    ``TransformerBlockConfig`` *or* a dict of them, ``block_overrides`` is another dict of
    them, and a hybrid model holds several with different mixers. Anything that reads one
    path is a check that quietly stops finding the second one.
    """
    found: set[AttentionBackendName] = set()
    seen: set[int] = set()
    stack: list[Any] = [config]

    while stack:
        item = stack.pop()

        if isinstance(item, AttentionBackendName):
            found.add(item)
            continue

        # `use_flash=True` is the deprecated spelling of `backend=flash_2` and is resolved as
        # such by `Attention.__init__` unconditionally, with no device in the question, so it
        # names a backend as surely as the field does.
        #
        # The sliding-window rule beside it is deliberately NOT reproduced. That one reads
        # `if backend is None and has_flash_attn_2()`, so it can only ever opt into a backend
        # that is already installed -- it can never be the reason an image is short one.
        if is_dataclass(item) and not isinstance(item, type):
            if getattr(item, "use_flash", None) is True:
                found.add(AttentionBackendName.flash_2)

        if id(item) in seen:
            continue
        seen.add(id(item))

        if is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, field.name, None) for field in fields(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)

    return found


def refuse(message: str) -> None:
    """Fail the build with something a reader can act on.

    The message is printed rather than handed to ``SystemExit``, because the platform's probe
    reports the exit *code* and does not reproduce the object -- a string given to SystemExit
    would be swallowed. Streams are echoed into the build log on a failure.
    """
    print(f"verify_image: REFUSED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    # A real tokenizer's width rather than a round number, so the factories are called the way
    # `.edullm/train_on_corpus.py` calls them. It is arithmetic over a dataclass of constants:
    # no file is opened and no host is contacted, which matters because this runs with
    # `--network none`.
    vocab_size = TokenizerConfig.dolma2().padded_vocab_size()

    factories = registered_factories()
    if not factories:
        refuse(
            f"no factories on TransformerConfig start with {FACTORY_PREFIXES}. Either the "
            "ladder was renamed and this file's prefixes are stale, or olmo_core is not the "
            "package that got imported. Either way this check is asserting nothing, which is "
            "not a state it may pass in."
        )

    print(f"verify_image: {len(factories)} registered factories, vocab_size={vocab_size}")

    required: dict[AttentionBackendName, list[str]] = {}
    for factory in factories:
        config = getattr(TransformerConfig, factory)(vocab_size=vocab_size)
        for backend in named_backends(config):
            required.setdefault(backend, []).append(factory)

    if not required:
        refuse(
            f"none of the {len(factories)} factories names an attention backend. Every one of "
            "them leaving the field at None is possible and would be fine, but it is also what "
            "a walk that stopped finding them would look like, and the two are worth telling "
            "apart by hand before this passes."
        )

    for backend in sorted(required, key=str):
        rungs = sorted(required[backend])
        witness = f"{len(rungs)} rungs, e.g. {rungs[0]}"

        if backend in DEVICE_DEPENDENT:
            print(
                f"verify_image: SKIPPED {backend} ({witness}): its support test reads the "
                "device, so this runner cannot answer it. Covered by olmo-core-check, not here."
            )
            continue

        if backend not in DEVICE_FREE:
            refuse(
                f"{backend} is named by {witness} and this file does not know whether its "
                "assert_supported() needs a GPU. Read it in olmo_core/nn/attention/backend.py "
                "and add it to DEVICE_FREE or to DEVICE_DEPENDENT. Refusing rather than "
                "guessing: guessing device-free reddens every build of a correct image, and "
                "guessing device-dependent is how this check silently stops covering a rung."
            )

        # The real assertion, and the line the failing run died on. Left to raise on its own
        # rather than wrapped, so the build log carries olmo_core's own message about its own
        # backend instead of a paraphrase.
        print(f"verify_image: asserting {backend} ({witness})")
        backend.assert_supported()

    print(f"verify_image: OK, {len(required)} backend(s) named by the registered ladder")


if __name__ == "__main__":
    main()
