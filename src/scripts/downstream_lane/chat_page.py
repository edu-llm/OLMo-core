"""A chat page for the endpoint next door, in one file and with no dependencies.

    python src/scripts/downstream_lane/chat_page.py --endpoint http://127.0.0.1:8000/v1 --model edullm

NOTHING IS IMPORTED THAT vLLM DOES NOT ALREADY HAVE, AND THAT IS THE ENTIRE ARGUMENT FOR
WRITING IT RATHER THAN INSTALLING ONE. Gradio is the obvious choice and there is no
precedent for it anywhere in either repository, so choosing it means putting a package
nobody here has ever run into the same interpreter as the server holding the weights,
hours before a presentation. That package pins fastapi, pydantic, httpx and starlette,
and so does vLLM; the resolver reconciles them on the demonstration machine or it does
not, and finding out costs the machine. A file that imports only the standard library
cannot lose that argument because it never has it. The second reason is the share link:
Gradio's runs through a third party's relay, and a demonstration whose reachability
depends on somebody else's uptime has a failure mode nobody in the room can fix.

IT PROXIES RATHER THAN LETTING THE BROWSER TALK TO vLLM, FOR THREE REASONS THAT ARE ALL
ABOUT THE ROOM. One origin means no CORS to configure on a server whose flags nobody
wants to be editing on the day. One port means one Systems Manager forward, and the lane
opens exactly one. And the stop strings a base model needs live here, server side, so a
reply stops even for a client that knows nothing about how this model was or was not
post-trained.

The page itself is deliberately dull. A dropped connection leaves the transcript intact
and says what happened in the transcript rather than in an alert; a slow first token
shows that the request is alive rather than looking hung; typing during a reply is
allowed and the message waits its turn instead of being lost or interleaved. Those three
are what a live demonstration actually breaks on, and each of them is a few lines.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: How long to wait on the upstream before deciding it is gone. A cold vLLM answering its
#: first request after a model load can sit for a while, and a proxy that gives up sooner
#: turns a slow first token into an error the audience sees. Nothing here is a request
#: anybody wants to abandon early.
UPSTREAM_TIMEOUT_SECONDS = 600


def _stream_note(message: str) -> bytes:
    """One more server-sent event, carrying a sentence rather than a token.

    Under its own key and not as ``delta.content``, so that the note is never mistaken for
    something the model said and never ends up in the transcript sent back on the next
    turn. Any client that does not know the key ignores it and sees a stream that ended,
    which is the correct degradation.
    """
    return f"data: {json.dumps({'edullm_note': message})}\n\n".encode()

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__MODEL__</title>
<!-- An empty data URI, so the browser stops asking for a file this server does not have.
     A 404 in the console of a machine somebody is presenting from is a question from the
     audience that costs a minute and is about nothing. -->
<link rel="icon" href="data:,">
<style>
  :root { color-scheme: light dark; --edge:#d8d8de; --dim:#6b6b76; --me:#eceef4; --bad:#b4232c; }
  @media (prefers-color-scheme: dark) {
    :root { --edge:#33333c; --dim:#9a9aa6; --me:#26262e; --bad:#ff7a82; }
  }
  * { box-sizing: border-box; }
  body { margin:0; height:100dvh; display:flex; flex-direction:column;
         font:16px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:.75rem; padding:.7rem 1rem;
           border-bottom:1px solid var(--edge); flex:0 0 auto; }
  header h1 { font-size:1rem; margin:0; font-weight:600; }
  header .model { font:13px/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--dim);
                  border:1px solid var(--edge); border-radius:999px; padding:.3rem .6rem; }
  header .spacer { flex:1; }
  button { font:inherit; padding:.45rem .8rem; border:1px solid var(--edge);
           border-radius:.5rem; background:transparent; color:inherit; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  #log { flex:1 1 auto; overflow-y:auto; padding:1.25rem; display:flex;
         flex-direction:column; gap:1rem; }
  .turn { max-width:52rem; width:100%; margin:0 auto; }
  .who { font-size:.72rem; letter-spacing:.09em; text-transform:uppercase;
         color:var(--dim); margin-bottom:.28rem; }
  .body { white-space:pre-wrap; overflow-wrap:anywhere; }
  .turn.user .body { background:var(--me); padding:.6rem .8rem; border-radius:.6rem; }
  .turn.note .body { color:var(--bad); font-size:.9rem; }
  .cursor::after { content:"\\258B"; animation:blink 1.05s steps(2,start) infinite; }
  @keyframes blink { to { visibility:hidden; } }
  form { flex:0 0 auto; border-top:1px solid var(--edge); padding:.85rem 1rem; }
  .row { max-width:52rem; margin:0 auto; display:flex; gap:.6rem; align-items:flex-end; }
  textarea { flex:1; resize:none; font:inherit; padding:.6rem .7rem; border-radius:.6rem;
             border:1px solid var(--edge); background:transparent; color:inherit;
             min-height:2.7rem; max-height:11rem; }
  .hint { max-width:52rem; margin:.4rem auto 0; font-size:.75rem; color:var(--dim); }
</style>
</head>
<body>
<header>
  <h1>eduLLM</h1>
  <span class="model" id="model">__MODEL__</span>
  <span class="spacer"></span>
  <button id="stop" disabled>Stop</button>
  <button id="clear">Clear</button>
</header>

<div id="log"></div>

<form id="form">
  <div class="row">
    <textarea id="input" rows="1" placeholder="Ask it something. Enter sends, Shift+Enter is a newline."
              autocomplete="off" autofocus></textarea>
    <button id="send" type="submit">Send</button>
  </div>
  <div class="hint" id="hint"></div>
</form>

<script>
// The transcript is the only state, and it lives here rather than on the server. A server
// that remembers the conversation has to be told when a browser reloads, when a second
// person opens the page, and when a stream dies half way; a server that remembers nothing
// is correct in all three without being told anything.
const history = [];
const log = document.getElementById("log");
const input = document.getElementById("input");
const send = document.getElementById("send");
const stopButton = document.getElementById("stop");
const hint = document.getElementById("hint");
let inFlight = null;

const MODEL = document.getElementById("model").textContent;

function turn(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "turn " + role;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "You" : role === "note" ? "" : MODEL;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  wrap.append(who, body);
  log.append(wrap);
  return body;
}

// Only follow the bottom when the reader is already there. Scrolling somebody back to the
// newest token while they are reading an earlier answer is the single most irritating thing
// a streaming transcript can do, and during a demonstration it happens on the projector.
function follow() {
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 120;
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function busy(on) {
  send.disabled = on;
  stopButton.disabled = !on;
  hint.textContent = on ? "Generating. You can keep typing; Send waits." : "";
}

async function ask(text) {
  history.push({ role: "user", content: text });
  const asked = turn("user", text);
  const body = turn("assistant", "");
  body.classList.add("cursor");
  follow();
  busy(true);

  const controller = new AbortController();
  inFlight = controller;
  let answer = "";
  let failed = false;
  let interrupted = "";
  try {
    const response = await fetch("v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages: history, stream: true }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("the endpoint answered " + response.status + " " + (await response.text()).slice(0, 400));

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      // Server-sent events are separated by a blank line and a chunk boundary lands in the
      // middle of one often enough to matter, so the tail is kept rather than parsed.
      const events = buffered.split("\\n\\n");
      buffered = events.pop();
      for (const event of events) {
        const line = event.split("\\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let piece;
        try { piece = JSON.parse(payload); } catch { continue; }
        if (piece.edullm_note) { interrupted = piece.edullm_note; continue; }
        const delta = piece.choices?.[0]?.delta?.content;
        if (delta) { answer += delta; body.textContent = answer; follow(); }
      }
    }
    if (!answer) answer = "(the model returned nothing)";
    history.push({ role: "assistant", content: answer });
    // A reply that was cut short is still a reply and stays in the transcript, so the next
    // question follows on from what the audience actually read rather than from a gap.
    if (interrupted) turn("note", interrupted + " \\u2014 the answer above is what arrived.");
  } catch (error) {
    if (error.name === "AbortError") {
      // A stopped reply is still a turn the model took, and dropping it would leave the
      // next request with a user message following a user message.
      history.push({ role: "assistant", content: answer || "(stopped)" });
    } else {
      // THE FAILED EXCHANGE COMES BACK OUT OF BOTH THE TRANSCRIPT AND THE PAGE, WHICH IS
      // NOT TIDINESS. Popping the array and leaving the two bubbles on screen puts the
      // page and the request that goes out next into disagreement about what was said,
      // and the version the audience is reading is the wrong one. The question returns to
      // the box instead, so recovering from a dropped endpoint is pressing Send again --
      // unless something newer is already typed there, which is not worth overwriting.
      history.pop();
      asked.parentElement.remove();
      body.parentElement.remove();
      if (!input.value.trim()) input.value = text;
      turn("note", String(error.message || error) + " \\u2014 nothing was lost; send again when it is back.");
      failed = true;
    }
  } finally {
    body.classList.remove("cursor");
    if (!failed) body.textContent = answer || body.textContent;
    inFlight = null;
    busy(false);
    follow();
    input.focus();
  }
}

document.getElementById("form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || inFlight) return;
  input.value = "";
  input.style.height = "auto";
  ask(text);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("form").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 176) + "px";
});

stopButton.addEventListener("click", () => inFlight && inFlight.abort());
document.getElementById("clear").addEventListener("click", () => {
  if (inFlight) inFlight.abort();
  history.length = 0;
  log.textContent = "";
  input.focus();
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so that a streamed reply can be chunked. Under 1.0 the only way to delimit a
    # response of unknown length is to close the socket, which costs a fresh connection per
    # turn and reads to the browser as a dropped stream whenever a proxy buffers.
    protocol_version = "HTTP/1.1"

    endpoint = ""
    model = ""
    stop: tuple[str, ...] = ()
    max_tokens = 512

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102 - quieter than the default
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.replace("__MODEL__", self.model).encode())
        elif path == "/healthz":
            self._send(200, "text/plain; charset=utf-8", b"ok\n")
        elif path.startswith("/v1/"):
            self._relay(path, None)
        else:
            self._send(404, "text/plain; charset=utf-8", b"no\n")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if not path.startswith("/v1/"):
            self._send(404, "text/plain; charset=utf-8", b"no\n")
            return
        length = int(self.headers.get("Content-Length") or 0)
        self._relay(path, self.rfile.read(length))

    def _relay(self, path: str, body: bytes | None) -> None:
        """Pass the request upstream and the bytes back as they arrive.

        The defaults are applied here and not in the page, so that a request arriving from
        anything else -- curl during a rehearsal, a second front end, somebody's script --
        gets the same stopping behaviour as the page does. A caller that names its own
        ``stop`` or ``max_tokens`` keeps them.
        """
        if body and path.endswith(("/chat/completions", "/completions")):
            try:
                parsed = json.loads(body)
                parsed.setdefault("stop", list(self.stop))
                parsed.setdefault("max_tokens", self.max_tokens)
                body = json.dumps(parsed).encode()
            except (ValueError, AttributeError):
                pass  # not JSON we understand; upstream will say so better than we can

        request = urllib.request.Request(
            self.endpoint.rstrip("/") + path[len("/v1") :],
            data=body,
            method="POST" if body is not None else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            upstream = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self._send(error.code, error.headers.get("Content-Type", "application/json"), payload)
            return
        except OSError as error:
            self._send(
                502,
                "application/json",
                json.dumps({"error": f"the model server did not answer: {error}"}).encode(),
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                try:
                    chunk = upstream.read1(4096)
                except (http.client.HTTPException, OSError) as error:
                    # THE MODEL SERVER DYING HALF WAY THROUGH A REPLY IS THE FAILURE THIS
                    # WHOLE BRANCH EXISTS FOR, AND IT ARRIVES AS AN EXCEPTION RATHER THAN AS
                    # AN EMPTY READ. Uncaught it takes the handler thread with it and the
                    # socket closes on an unfinished chunk, which the browser reports as a
                    # bare "network error" -- no partial answer kept and nothing said about
                    # why. Measured against a server killed mid-stream on 2026-08-08. Saying
                    # so inside the stream instead lets the page keep the tokens it already
                    # has and put a sentence under them.
                    self._chunk(_stream_note(f"the model server stopped mid-reply: {error}"))
                    break
                if not chunk:
                    break
                self._chunk(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # THE BROWSER LEAVING IS NOT AN ERROR AND MUST NOT TAKE THE SERVER WITH IT. A
            # laptop that closed its lid, a tab that was refreshed and a tunnel that dropped
            # all arrive here, and all three happen during a demonstration. Closing the
            # upstream on the way out is what tells vLLM to abandon the generation, so a
            # reader who walks away stops costing the card immediately.
            pass
        finally:
            upstream.close()

    def _chunk(self, payload: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
        self.wfile.flush()

    def _send(self, code: int, content_type: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    #: A thread per connection, and the process exits without waiting for any of them. Three
    #: people typing at once is a stated requirement and a single-threaded handler serves the
    #: second person's page only after the first person's reply has finished streaming.
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1", help="the OpenAI-compatible base URL")
    parser.add_argument("--model", default="edullm", help="what the page asks for and shows")
    parser.add_argument("--host", default="0.0.0.0", help="what to bind; the lane has no ingress rules so this is not exposure")
    parser.add_argument("--port", type=int, default=8888, help="the port the lane's notebook forward reaches")
    parser.add_argument("--stop", nargs="*", default=[], help="stop strings applied to a request that names none")
    parser.add_argument("--max-tokens", type=int, default=512, help="cap for a request that names none")
    arguments = parser.parse_args()

    Handler.endpoint = arguments.endpoint
    Handler.model = arguments.model
    Handler.stop = tuple(arguments.stop)
    Handler.max_tokens = arguments.max_tokens

    server = Server((arguments.host, arguments.port), Handler)
    print(f"chat page on http://{arguments.host}:{arguments.port}/ for {arguments.model} at {arguments.endpoint}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
