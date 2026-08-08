#!/usr/bin/env bash
#
# Everything the serving lane claims, run against a real model on a real card, in one
# pass. Meant for `edullm run`, where the tree is already on the machine and the output
# streams back to a laptop.
#
#   edullm run --project chat-endpoint-test --compute gpu-1xl40s --hours 2 \
#     -- bash src/scripts/downstream_lane/prove_the_lane.sh
#
# A BASE MODEL AND NOT AN INSTRUCT ONE, WHICH IS THE ONLY CHOICE THAT PROVES ANYTHING
# HERE. An instruction-tuned checkpoint arrives carrying its own chat template, so serving
# it demonstrates that vLLM works and says nothing at all about the case this lane exists
# for -- a base export with no template, which is what the capacity block will hand over.
# allenai/OLMoE-1B-7B-0924 is that case in the same architecture family: a pretrained MoE,
# no template, no post-training. If it holds a conversation, the block's own checkpoint
# will.
#
# It is pulled to a directory rather than named as a hub id on purpose. A hub id goes
# straight to vLLM untouched, and the template installation this is testing happens on a
# directory.
set -euo pipefail

MODEL_ID=${MODEL_ID:-allenai/OLMoE-1B-7B-0924}
SERVED=${SERVED:-edullm}
PORT=${PORT:-8000}
PAGE_PORT=${PAGE_PORT:-8888}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# THE LANE IMAGE HAS python3 AND NO python, AND pip PUTS CONSOLE SCRIPTS SOMEWHERE THAT IS
# NOT ON PATH. Both were measured here on 2026-08-08: `python` exited 127 twice, and after
# a clean `pip install vllm` the `vllm` command did not exist. There is no virtualenv on
# this image by design -- a framework the platform chose would be a version no repository
# declared -- so a user install is the only kind there is, and ~/.local/bin is where it
# lands.
export PATH="${HOME}/.local/bin:${PATH}"

say() { printf '\n=== %s ===\n' "$*"; }

say "the machine"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' || true

say "vllm"
# Installed if absent rather than assumed. The deep-learning image carries torch and not
# a server, and pinning here would fight the image's own torch on the next AMI refresh --
# which is a worse failure than a resolver taking four minutes.
if ! python3 -c 'import vllm' 2>/dev/null; then
  pip install --quiet vllm
fi
python3 -c 'import vllm; print("vllm", vllm.__version__)'

say "pulling ${MODEL_ID}"
# OUTSIDE THE DIRECTORY `edullm run` CARRIES BACK, WHICH IS NOT A DETAIL. The verb syncs
# the working tree to the scratch bucket when the command returns, and a thirteen-gigabyte
# checkpoint dropped inside it turns a two-second sync into a long upload of something
# already on the hub. The home directory is the sibling to use rather than /work: the lane
# gives a machine one writable directory under /work, the project's own, and /work itself
# refuses a mkdir.
MODEL_DIR=${MODEL_DIR:-${HOME}/model}
mkdir -p "$MODEL_DIR"
python3 - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=[
    "*.json", "*.safetensors", "*.txt", "*.model",
]))
PY

say "what the export carries before anything touches it"
python3 - "$MODEL_DIR" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
config = json.loads((d / "config.json").read_text())
tokenizer = json.loads((d / "tokenizer_config.json").read_text())
print("architectures         ", config.get("architectures"))
print("model_type            ", config.get("model_type"))
print("max_position_embeddings", config.get("max_position_embeddings"))
print("chat_template present ", bool(tokenizer.get("chat_template")))
PY

say "does vLLM register the name this export writes"
python3 "${HERE}/check_export_is_servable.py" --exported-dir "$MODEL_DIR"

say "the endpoint that exists before a template is installed"
# Started by hand for this one check, because serve_a_checkpoint.py installs the template
# and the point is to see the 400 it fixes. Everything after this goes through the script.
MAX_MODEL_LEN=4096 GPU_FRACTION=0.90 SERVE_LOG=/tmp/vllm-bare.log \
  bash "${HERE}/serve_exported_checkpoint.sh" "$MODEL_DIR" 8001 bare
echo "--- /v1/models ---"
curl -s localhost:8001/v1/models | head -c 400; echo
echo "--- /v1/completions ---"
curl -s localhost:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"bare","prompt":"The capital of Japan is","max_tokens":12}' | head -c 400; echo
echo "--- /v1/chat/completions ---"
curl -s -o /tmp/bare-chat.json -w 'HTTP %{http_code}\n' localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bare","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
head -c 400 /tmp/bare-chat.json; echo
pkill -f 'vllm serve' || true
sleep 20

say "one command, cold, from the directory to a URL"
COLD_START=$(date +%s)
python3 "${HERE}/serve_a_checkpoint.py" "$MODEL_DIR" \
  --port "$PORT" --served-name "$SERVED" --max-model-len 4096 \
  --chat-page-port "$PAGE_PORT"
echo "cold start, command to answering endpoint: $(( $(date +%s) - COLD_START ))s"

say "the same request that was a 400"
curl -s -w '\nHTTP %{http_code}\n' localhost:${PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$SERVED"'","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' | head -c 600

say "a two-turn conversation, second turn carrying the first"
python3 - "$PORT" "$SERVED" <<'PY'
import json, sys, urllib.request

port, model = sys.argv[1], sys.argv[2]

def ask(messages):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 120,
                       "temperature": 0.0, "stop": ["\nUser:", "\n\nUser:"]}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        answer = json.load(response)
    choice = answer["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason")

history = [{"role": "user", "content": "What is the capital of Japan?"}]
first, why = ask(history)
print("USER : What is the capital of Japan?")
print("MODEL:", first.strip(), f"   [finish_reason={why}]")

history += [{"role": "assistant", "content": first},
            {"role": "user", "content": "What is the population of that city?"}]
second, why = ask(history)
print("USER : What is the population of that city?")
print("MODEL:", second.strip(), f"   [finish_reason={why}]")
PY

say "the page the audience sees"
curl -s "localhost:${PAGE_PORT}/healthz"
curl -s "localhost:${PAGE_PORT}/" | grep -o '<title>[^<]*</title>'
echo "--- a streamed turn straight through the proxy ---"
curl -sN --max-time 120 "localhost:${PAGE_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$SERVED"'","messages":[{"role":"user","content":"Name three cities in Japan."}],"stream":true}' \
  | tail -6

say "measurements, against the endpoint"
python3 "${HERE}/measure_the_endpoint.py" --endpoint "http://127.0.0.1:${PORT}/v1" --model "$SERVED"

say "measurements, through the page's proxy, which is what the room goes through"
python3 "${HERE}/measure_the_endpoint.py" --endpoint "http://127.0.0.1:${PAGE_PORT}/v1" \
  --model "$SERVED" --repeats 2

say "shutting the machine's servers down"
pkill -f 'vllm serve' || true
pkill -f chat_page.py || true
echo "done"
