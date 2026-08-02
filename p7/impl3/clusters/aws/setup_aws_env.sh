#!/bin/bash
# One-time environment setup for P7 post-training on an AWS GPU box (via SSM).
# Run ON THE BOX as the training user (see AWS_RUN.md for how to connect):
#     bash setup_aws_env.sh
set -euo pipefail

VENV="${VENV:-$HOME/p7post-venv}"

# The box authenticates to S3 via its EC2 INSTANCE ROLE. A laptop AWS_PROFILE here
# causes ProfileNotFound — make sure it is unset on the box.
unset AWS_PROFILE || true

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip wheel

# Reuse a preinstalled CUDA torch if the DLAMI already has a working one; otherwise
# install a CUDA build. (Blackwell/B200 needs a recent CUDA/torch — see the pretrain
# runbook's B200 notes.)
python -c "import torch; print('found torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r "$(dirname "$0")/../../requirements.txt"
pip uninstall -y torchao 2>/dev/null || true   # old torchao breaks peft.get_peft_model

python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
PY
echo "Env ready at $VENV. Next: bash run_aws.sh (inside tmux for long runs)."
