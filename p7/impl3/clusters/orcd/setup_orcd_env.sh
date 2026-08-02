#!/bin/bash
# One-time environment setup for P7 post-training on MIT ORCD (Engaging).
# Run ONCE on a login node (installs software only; no GPU needed):
#     bash setup_orcd_env.sh
# Then submit training with run.sbatch / submit_sweep.sh.
set -euo pipefail

ENV_NAME="${ENV_NAME:-p7post}"

# Resolve project root from THIS script's absolute location FIRST, before any `cd` below can
# change the working directory (that CWD change was the original requirements.txt-not-found bug).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQ="$PROJECT_ROOT/requirements.txt"

# --- 1. Miniforge in your home dir (if absent) ---
if [ ! -d "$HOME/miniforge3" ]; then
    echo "Installing Miniforge into $HOME/miniforge3 ..."
    cd "$HOME"
    curl -L -o Miniforge3.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash Miniforge3.sh -b -p "$HOME/miniforge3"
    rm -f Miniforge3.sh
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# --- 2. Env (Python 3.11) ---
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# --- 3. Deps ---
pip install --upgrade pip
# PyTorch (cu121 wheels work on L40S / H100 / H200 nodes). Pinned to match requirements.txt.
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
# The post-training project deps (REQ resolved at the top, before any `cd`). Fail clearly
# if it's not synced.
if [ ! -f "$REQ" ]; then
    echo "FATAL: requirements.txt not found at $REQ" >&2
    echo "The project isn't fully synced. Re-run the step-0 rsync from your Mac, then rerun this." >&2
    exit 1
fi
pip install -r "$REQ"

# IMPORTANT (gotcha): an old torchao (0.10) breaks peft.get_peft_model. We don't use
# torchao — make sure it's not installed.
pip uninstall -y torchao 2>/dev/null || true

echo
echo "Environment '$ENV_NAME' ready."
python - <<'PY'
import torch, transformers, peft
import sentencepiece, tiktoken  # eval tokenizer backends; missing -> eval stage fails silently
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("sentencepiece", sentencepiece.__version__, "| tiktoken", tiktoken.__version__)
print("CUDA available (expected False on a login node):", torch.cuda.is_available())
PY
echo
echo "W&B logging is ON by default. Authenticate once: 'wandb login' (or export WANDB_API_KEY)."
echo "Next: export WANDB_API_KEY=... ; then bash submit_sweep.sh impl2  (or sbatch run.sbatch)"
