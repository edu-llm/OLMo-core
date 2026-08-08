"""What the serving lane must not get wrong when nobody is looking.

Three behaviours are covered and they are the three that fail silently. Picking the wrong
step produces a traceback minutes into a load, on a machine that is already paid for.
Overwriting a post-trained model's chat template produces a model that answers, badly, in
a way that reads as the model being poor rather than the template being wrong. And a proxy
that drops the stop strings produces replies that run to the token cap, which during a
demonstration is indistinguishable from a hung endpoint.

The scripts live in ``src/scripts/downstream_lane`` and are not importable as a package,
so they are loaded by path -- the same arrangement, and for the same reason, as
``src/test/edullm_train_on_corpus_test.py``.
"""

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

LANE = Path(__file__).parent.parent.parent / "scripts" / "downstream_lane"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"downstream_lane_{name}", LANE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


serve = _load("serve_a_checkpoint")
page = _load("chat_page")


def _write_checkpoint(directory: Path, *, whole: bool) -> None:
    """A directory shaped the way ``Checkpointer.dir_is_checkpoint`` reads it.

    The incomplete one is missing ``train/rank0.pt`` and nothing else, which is what a
    directory a running job is part way through writing actually looks like -- not an
    empty directory, which would be caught by anything. The three names come out of
    ``Checkpointer`` rather than being spelled here, so a rename upstream fails this test
    instead of quietly making it test nothing.
    """
    from olmo_core.train.checkpoint import Checkpointer

    (directory / "model_and_optim").mkdir(parents=True)
    (directory / "model_and_optim" / ".metadata").write_bytes(b"")
    (directory / Checkpointer.METADATA_FNAME).write_text("{}")
    (directory / "config.json").write_text("{}")
    if whole:
        (directory / "train").mkdir()
        (directory / "train" / "rank0.pt").write_bytes(b"")


def test_the_highest_complete_step_beats_the_highest_step(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "step1000", whole=True)
    _write_checkpoint(tmp_path / "step2000", whole=True)
    # The step a live job is writing right now: listed, largest, and not loadable.
    _write_checkpoint(tmp_path / "step3000", whole=False)

    resolved = serve.resolve(str(tmp_path))
    assert resolved.shape is serve.Shape.RUN_CHECKPOINTS
    assert resolved.location.endswith("step2000")
    assert "step 3000 not complete yet" in resolved.note


def test_a_directory_with_no_complete_step_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "step500", whole=False)
    with pytest.raises(SystemExit, match="no step has been written whole"):
        serve.resolve(str(tmp_path))


def test_an_export_is_recognised_without_being_loaded(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["FlexOlmoForCausalLM"]}))
    assert serve.resolve(str(tmp_path)).shape is serve.Shape.HUGGINGFACE_EXPORT


def test_a_hub_id_is_not_mistaken_for_a_missing_directory() -> None:
    assert serve.resolve("allenai/OLMoE-1B-7B-0924-Instruct").shape is serve.Shape.HUB_ID


def test_a_template_is_installed_only_where_there_is_none(tmp_path: Path) -> None:
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"eos_token": "<|endoftext|>"}))
    serve.install_chat_template(tmp_path)
    installed = json.loads(config.read_text())["chat_template"]
    assert "User:" in installed

    config.write_text(json.dumps({"chat_template": "{{ 'mine' }}"}))
    serve.install_chat_template(tmp_path)
    assert json.loads(config.read_text())["chat_template"] == "{{ 'mine' }}"


def test_a_context_window_of_minus_one_is_named_rather_than_passed_on(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": -1}))
    with pytest.raises(SystemExit, match="max_position_embeddings=-1"):
        serve.context_window(tmp_path, None)
    assert serve.context_window(tmp_path, 4096) == 4096


def test_the_template_renders_a_two_turn_exchange() -> None:
    jinja2 = pytest.importorskip("jinja2")
    template = jinja2.Environment().from_string(serve.TEMPLATE_FILE.read_text())
    rendered = template.render(
        messages=[
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "content": "Tokyo."},
            {"role": "user", "content": "And of France?"},
        ],
        add_generation_prompt=True,
    )
    assert rendered.endswith("Assistant:")
    assert rendered.count("User: ") == 2
    # The turn ending the server stops on has to be a string this template actually
    # produces, or every reply runs to the token cap.
    assert any(ending.strip() in rendered for ending in serve.TURN_ENDINGS)


class _Upstream(BaseHTTPRequestHandler):
    """Records what the proxy sent it, and can be told to hang up mid-reply."""

    seen: dict = {}
    die_after: int = 0

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        type(self).seen = json.loads(self.rfile.read(length))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for index, word in enumerate(("one", "two", "three", "four")):
            if self.die_after and index >= self.die_after:
                self.connection.close()
                return
            body = f"data: {json.dumps({'choices': [{'delta': {'content': word}}]})}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(body), body))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def wired(request):
    """A chat page in front of a recording upstream, both on ephemeral ports."""
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    upstream.daemon_threads = True
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    page.Handler.endpoint = f"http://127.0.0.1:{upstream.server_port}/v1"
    page.Handler.model = "edullm"
    page.Handler.stop = ("\nUser:",)
    page.Handler.max_tokens = 64
    front = page.Server(("127.0.0.1", 0), page.Handler)
    threading.Thread(target=front.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{front.server_port}"
    finally:
        front.shutdown()
        upstream.shutdown()


def _ask(base: str, **extra) -> str:
    body = json.dumps({"model": "edullm", "messages": [{"role": "user", "content": "hi"}], **extra})
    request = urllib.request.Request(
        f"{base}/v1/chat/completions", data=body.encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def test_the_proxy_supplies_the_stops_a_base_model_needs(wired: str) -> None:
    _ask(wired)
    assert _Upstream.seen["stop"] == ["\nUser:"]
    assert _Upstream.seen["max_tokens"] == 64


def test_a_caller_that_names_its_own_limits_keeps_them(wired: str) -> None:
    _ask(wired, stop=["###"], max_tokens=8)
    assert _Upstream.seen["stop"] == ["###"]
    assert _Upstream.seen["max_tokens"] == 8


def test_an_endpoint_that_dies_mid_reply_ends_the_stream_instead_of_the_thread(wired: str) -> None:
    """The tokens already sent survive, a sentence explains the rest, and the page lives.

    Left uncaught this arrives as ``IncompleteRead``, takes the handler thread with it and
    closes the socket on an unfinished chunk -- which a browser reports as a bare network
    error with no partial answer kept. Measured on 2026-08-08 before this was handled.
    """
    _Upstream.die_after = 2
    try:
        received = _ask(wired)
    finally:
        _Upstream.die_after = 0
    assert '"content": "one"' in received
    assert "edullm_note" in received

    # The server is still answering afterwards, which is the half that matters.
    with urllib.request.urlopen(f"{wired}/healthz", timeout=10) as response:
        assert response.read() == b"ok\n"


def test_the_page_carries_the_model_name_it_was_started_with(wired: str) -> None:
    with urllib.request.urlopen(f"{wired}/", timeout=10) as response:
        html = response.read().decode()
    assert "__MODEL__" not in html
    assert 'id="model">edullm<' in html


def test_an_unknown_path_is_a_404_and_not_a_proxy_hole(wired: str) -> None:
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(f"{wired}/../etc/passwd", timeout=10)
    assert raised.value.code == 404
