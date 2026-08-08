"""One command from a checkpoint URI to a URL somebody can type at.

Point it at what you have. A run's checkpoint directory, one step inside one, a
HuggingFace export this repository already wrote, or a hub id -- it works out which of
those it was given, does only the steps that shape needs, and does not return until an
endpoint answers.

    python src/scripts/downstream_lane/serve_a_checkpoint.py s3://bucket/run/checkpoints
    python src/scripts/downstream_lane/serve_a_checkpoint.py /work/hf/step10000
    python src/scripts/downstream_lane/serve_a_checkpoint.py allenai/OLMoE-1B-7B-0924-Instruct

THE HIGHEST COMPLETE STEP AND NOT THE HIGHEST STEP, WHICH IS THE WHOLE REASON THIS
POINTS AT A DIRECTORY RATHER THAN A CHECKPOINT. A training job that is still running has
a directory for the step it is writing right now, and that directory is listed, is named
with the largest number, and is not loadable. Sorting a listing therefore picks the one
checkpoint guaranteed to fail, and it fails after the weights have been read for several
minutes. ``Checkpointer.find_checkpoints`` yields only directories that pass
``dir_is_checkpoint``, which is the same test the loader applies, so the step this picks
is a step that loads. Demonstrating against a live run is the ordinary case here rather
than a corner of it: the model somebody wants to show on a Thursday is being trained on
the Thursday.

CONVERSION HAPPENS ON THE GPU BECAUSE IT CANNOT HAPPEN ANYWHERE ELSE. A dropless MoE
gathers and scatters through Triton kernels, and Triton refuses a CPU pointer on a host
where CUDA is present, so the export step of this lane needs the accelerator even though
the tensors would fit in host memory. The default here is ``cuda`` when torch reports a
device and the conversion is skipped entirely when it is not needed, rather than a flag
somebody has to know to pass.

See ``guides/the-downstream-lane.md`` in the platform repository for the chain this
automates and for what each link was measured at.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The template installed into an export that carries none. Read from a file beside this
#: one rather than embedded as a string, because vLLM's ``--chat-template`` takes a path
#: and a template that exists twice diverges the first time one copy is edited.
TEMPLATE_FILE = HERE / "base_model_chat_template.jinja"

#: What ends a turn in the template above, given to the server as its default so that a
#: client which knows nothing about any of this still gets a reply that stops.
#:
#: A BASE MODEL EMITS NO END-OF-TURN TOKEN, SO A STOP STRING IS NOT A REFINEMENT. Nothing
#: in pretraining teaches a model to close a turn, so a reply relying on ``eos`` runs to
#: ``max_tokens`` every time -- forty seconds of a demonstration spent watching a model
#: answer a question it already answered. What a base model does do, reliably, is start
#: the next speaker's turn, because that is how the transcripts it read continue. These
#: catch that. Both spellings of the newline appear because a model that has been reading
#: its own output decides the blank line for itself.
TURN_ENDINGS = ("\nUser:", "\n\nUser:", "\nuser:")

#: Where the front end binds. **8888 IS THE ONE PORT ON A LANE MACHINE THAT A LAPTOP CAN
#: REACH, AND IT IS NOT A COINCIDENCE.** The lane's security group holds zero ingress
#: rules on purpose, so nothing on the machine is addressable from anywhere; the single
#: route in is the Systems Manager forward that ``edullm shell --notebook`` opens, and
#: that forward is wired to this port. Binding the chat page here means the reachability
#: story is a verb that already exists rather than a security-group amendment and a
#: review. Jupyter wants the same port and the two cannot both have it, which is the
#: cost, and for a machine whose job is to serve a demonstration it is not a cost.
DEFAULT_CHAT_PAGE_PORT = 8888


class Shape(Enum):
    """What the URI turned out to name. Each one skips a different amount of work."""

    #: A directory of ``stepNNNN`` directories, at most one of which is being written now.
    RUN_CHECKPOINTS = "a run's checkpoint directory"
    #: One OLMo-core checkpoint, sharded, needing conversion before anything can serve it.
    OLMO_CORE_CHECKPOINT = "an OLMo-core checkpoint"
    #: Weights and a ``config.json`` that names an architecture. Ready to serve.
    HUGGINGFACE_EXPORT = "a HuggingFace export"
    #: An ``org/name`` the hub resolves. vLLM fetches it; nothing here touches it.
    HUB_ID = "a HuggingFace hub id"


@dataclass(frozen=True)
class Resolved:
    shape: Shape
    #: What to hand the next step. For a run directory this is the step that was chosen,
    #: not the directory that was named.
    location: str
    #: Said out loud on the way past, because a person who asked for a run directory and
    #: got step 4000 when step 5000 exists needs to be told why before they wonder.
    note: str = ""


def _is_remote(uri: str) -> bool:
    return "://" in uri


def _looks_like_a_hub_id(uri: str) -> bool:
    """``org/name`` and nothing else.

    Checked before the filesystem rather than after, so that a hub id is never mistaken
    for a relative path that happens not to exist and reported as a missing directory.
    """
    if _is_remote(uri) or uri.startswith((".", "/", "~")):
        return False
    parts = uri.split("/")
    return len(parts) == 2 and all(parts) and not Path(uri).exists()


def _read_json(uri: str) -> dict | None:
    from olmo_core.io import file_exists, resource_path

    if not file_exists(uri):
        return None
    if _is_remote(uri):
        parent, _, name = uri.rpartition("/")
        return json.loads(resource_path(parent, name).read_text())
    return json.loads(Path(uri).read_text())


def latest_servable_checkpoint(directory: str) -> tuple[int, str] | None:
    """The highest step in ``directory`` that would actually load, or None.

    ``find_checkpoints`` applies ``dir_is_checkpoint`` to every ``stepNNNN`` it lists and
    drops the ones that fail, which is what makes this safe against a run that is mid
    write. Taking the max of what survives rather than the max of the listing is the
    entire difference between serving a live run and serving a traceback.
    """
    from olmo_core.train.checkpoint import Checkpointer

    found = list(Checkpointer.find_checkpoints(directory))
    if not found:
        return None
    return max(found, key=lambda pair: pair[0])


def _all_step_directories(directory: str) -> list[int]:
    """Every ``stepNNNN`` in the listing, complete or not. Used only to say what was skipped."""
    import re

    from olmo_core.io import list_directory

    steps = []
    for path in list_directory(directory):
        match = re.match(r"^step(\d+)$", os.path.basename(path.rstrip("/")))
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def resolve(uri: str) -> Resolved:
    """Decide what was handed in, cheaply, before anything is downloaded or allocated."""
    from olmo_core.train.checkpoint import Checkpointer

    uri = uri.rstrip("/")

    if _looks_like_a_hub_id(uri):
        return Resolved(Shape.HUB_ID, uri)

    written = _read_json(f"{uri}/config.json")
    if written is not None and written.get("architectures"):
        return Resolved(Shape.HUGGINGFACE_EXPORT, uri)

    if Checkpointer.dir_is_checkpoint(uri):
        return Resolved(Shape.OLMO_CORE_CHECKPOINT, uri)

    latest = latest_servable_checkpoint(uri)
    if latest is not None:
        step, path = latest
        listed = _all_step_directories(uri)
        skipped = [other for other in listed if other > step]
        note = f"step {step} of {len(listed)} in the listing"
        if skipped:
            # NAMED RATHER THAN SWALLOWED. A higher step that failed the completeness test
            # is the normal state of a directory a job is still writing to, and somebody
            # who is told the number they can see was passed over stops worrying about it.
            note += f"; {', '.join(f'step {n}' for n in skipped)} not complete yet"
        return Resolved(Shape.RUN_CHECKPOINTS, path, note)

    raise SystemExit(
        f"{uri} is not a HuggingFace export, not an OLMo-core checkpoint, and holds no "
        "complete stepNNNN directory. If a job has only just started, no step has been "
        "written whole yet and there is nothing here that can be served."
    )


def convert(source: str, destination: Path, *, max_sequence_length: int, device: str) -> None:
    """OLMo-core checkpoint to HuggingFace, through the repository's own converter.

    Shelled out to rather than imported, because the converter is a script with a
    ``main`` that arranges logging and CLI state, and reproducing that arrangement here
    would be a second copy of it that drifts. What this adds is the two flags nobody
    should have to remember, and the reasons are in the guide: ``--max-sequence-length``
    because ``get_hf_config`` writes -1 and vLLM reads that as the context window, and
    ``--device cuda`` because a MoE conversion routes through Triton.

    Validation stays on. It runs both models over the same tokens and compares logits,
    and without it this step produces a directory rather than a proof.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(HERE.parents[1] / "examples" / "huggingface" / "convert_checkpoint_to_hf.py"),
        "-i",
        source,
        "-o",
        str(destination),
        "--max-sequence-length",
        str(max_sequence_length),
        "--device",
        device,
        "--validation-device",
        device,
    ]
    print(f"converting {source} -> {destination} on {device}", flush=True)
    subprocess.run(argv, check=True)


def stage(source: str, destination: Path) -> Path:
    """Bring a remote export down, because vLLM opens files and not URIs."""
    if not _is_remote(source):
        return Path(source)
    from olmo_core.io import copy_dir

    destination.mkdir(parents=True, exist_ok=True)
    print(f"staging {source} -> {destination}", flush=True)
    copy_dir(source, destination)
    return destination


def install_chat_template(directory: Path) -> str:
    """Give an export a chat template if it has none, and leave one that has one alone.

    **THIS IS WHAT MAKES ``/v1/chat/completions`` ANSWER 200 INSTEAD OF 400.** A base
    export carries no ``chat_template`` key, transformers has had no default since 4.44,
    and the endpoint every chat client in the world speaks refuses the request before the
    model is reached. The template is not a claim that the model was post-trained; it is
    the missing half of a protocol, and the demonstration should not be waiting on an SFT
    run to have one.

    A post-trained checkpoint arrives with its own and keeps it. Overwriting that would
    replace turn markers the model was actually tuned on with markers it has never seen,
    which is a way to make a good model look like a bad one.
    """
    config_path = directory / "tokenizer_config.json"
    if not config_path.exists():
        return "no tokenizer_config.json in the export; passing the template to the server only"

    config = json.loads(config_path.read_text())
    if config.get("chat_template"):
        return "the export carries its own chat template, which is left alone"

    config["chat_template"] = TEMPLATE_FILE.read_text()
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return f"installed {TEMPLATE_FILE.name} into tokenizer_config.json"


def context_window(directory: Path, requested: int | None) -> int:
    """What to tell vLLM the model's context is, refusing the value that means "unset".

    ``get_hf_config`` builds every config with ``max_position_embeddings=-1`` and the
    converter overwrites it only from ``--max-sequence-length`` or from a tokenizer
    carrying ``model_max_length``. An export that got neither hands vLLM -1, which fails
    on the way up in a manner that reads as a broken model rather than as missing
    metadata. Catching it here costs nothing and names the actual problem.
    """
    if requested is not None:
        return requested
    config = json.loads((directory / "config.json").read_text())
    written = int(config.get("max_position_embeddings", -1))
    if written <= 0:
        raise SystemExit(
            f"{directory}/config.json says max_position_embeddings={written}, which is what "
            "an export written without --max-sequence-length carries. vLLM reads it as the "
            "context window. Re-export with a length, or pass --max-model-len here."
        )
    return written


def serve(
    model: str,
    *,
    port: int,
    served_name: str,
    max_model_len: int,
    gpu_fraction: float,
    log: Path,
) -> None:
    """Hand the launch to the script that already knows how to wait for a port.

    ``serve_exported_checkpoint.sh`` polls ``/v1/models``, gives up on a process that
    died rather than on a clock, and prints the tail of the log when it does. Calling it
    keeps exactly one thing in this repository that starts a vLLM, which matters more
    than the awkwardness of a Python script shelling out to a shell script: two launchers
    diverge, and the one that diverges is always the one nobody ran this week.
    """
    environment = dict(os.environ)
    environment["MAX_MODEL_LEN"] = str(max_model_len)
    environment["GPU_FRACTION"] = str(gpu_fraction)
    environment["SERVE_LOG"] = str(log)
    environment["EXTRA_SERVE_ARGS"] = " ".join(
        (
            environment.get("EXTRA_SERVE_ARGS", ""),
            f"--chat-template {TEMPLATE_FILE}",
        )
    ).strip()
    subprocess.run(
        [str(HERE / "serve_exported_checkpoint.sh"), model, str(port), served_name],
        env=environment,
        check=True,
    )


def start_chat_page(*, endpoint: str, model: str, port: int, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("wb")
    return subprocess.Popen(
        [
            sys.executable,
            str(HERE / "chat_page.py"),
            "--endpoint",
            endpoint,
            "--model",
            model,
            "--port",
            str(port),
            "--stop",
            *TURN_ENDINGS,
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def _wait_for(url: str, *, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="a run's checkpoint directory, one step, an export, or a hub id")
    parser.add_argument("--port", type=int, default=8000, help="where vLLM answers")
    parser.add_argument("--served-name", default="edullm", help="the name the endpoint reports and clients ask for")
    parser.add_argument("--max-model-len", type=int, help="context window; read off the export when absent")
    parser.add_argument("--work-dir", type=Path, default=Path("/work/serve"), help="where a conversion or a staged copy lands")
    parser.add_argument("--gpu-fraction", type=float, default=0.90, help="fraction of the card vLLM may hold")
    parser.add_argument("--device", default=None, help="where a conversion runs; cuda when a card is present")
    parser.add_argument("--chat-page-port", type=int, default=DEFAULT_CHAT_PAGE_PORT, help="where the front end binds")
    parser.add_argument("--no-chat-page", action="store_true", help="stand up the API only")
    parser.add_argument("--reuse", action="store_true", help="keep an existing conversion in --work-dir instead of redoing it")
    parser.add_argument("--dry-run", action="store_true", help="say what the URI resolved to and stop, without a GPU")
    arguments = parser.parse_args()

    resolved = resolve(arguments.checkpoint)
    print(f"{arguments.checkpoint} is {resolved.shape.value}", flush=True)
    if resolved.note:
        print(f"  serving {resolved.note}", flush=True)
    if resolved.location != arguments.checkpoint:
        print(f"  -> {resolved.location}", flush=True)
    if arguments.dry_run:
        return 0

    if resolved.shape is Shape.HUB_ID:
        # Nothing to stage, nothing to convert, and nothing to install: a hub model that
        # is worth serving is post-trained and carries its own template. vLLM fetches it.
        model_for_vllm = resolved.location
        max_model_len = arguments.max_model_len or 4096
        print("hub model: no conversion, no template installed", flush=True)
    else:
        if resolved.shape is Shape.HUGGINGFACE_EXPORT:
            directory = stage(resolved.location, arguments.work_dir / "hf")
        else:
            directory = arguments.work_dir / "hf"
            if arguments.reuse and (directory / "config.json").exists():
                print(f"reusing the conversion already in {directory}", flush=True)
            else:
                device = arguments.device or _default_device()
                convert(
                    resolved.location,
                    directory,
                    max_sequence_length=arguments.max_model_len or 4096,
                    device=device,
                )
        print(install_chat_template(directory), flush=True)
        max_model_len = context_window(directory, arguments.max_model_len)
        model_for_vllm = str(directory)

    started = time.time()
    serve(
        model_for_vllm,
        port=arguments.port,
        served_name=arguments.served_name,
        max_model_len=max_model_len,
        gpu_fraction=arguments.gpu_fraction,
        log=Path(f"/tmp/vllm-{arguments.port}.log"),
    )
    endpoint = f"http://127.0.0.1:{arguments.port}/v1"
    print(f"endpoint up in {time.time() - started:.0f}s: {endpoint}, model {arguments.served_name}", flush=True)

    if arguments.no_chat_page:
        return 0

    start_chat_page(
        endpoint=endpoint,
        model=arguments.served_name,
        port=arguments.chat_page_port,
        log=Path(f"/tmp/chat-page-{arguments.chat_page_port}.log"),
    )
    if not _wait_for(f"http://127.0.0.1:{arguments.chat_page_port}/healthz", seconds=30):
        print(
            f"the chat page did not answer on {arguments.chat_page_port}; see "
            f"/tmp/chat-page-{arguments.chat_page_port}.log. The API above is unaffected.",
            file=sys.stderr,
        )
        return 1

    print(
        f"chat page on {arguments.chat_page_port}. From a laptop: edullm shell "
        f"--project <this project> --notebook, then http://localhost:8890/",
        flush=True,
    )
    return 0


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
