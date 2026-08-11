param(
  [int]$IntervalSeconds = 30,
  [string]$SshHost = "157.157.221.201",
  [int]$SshPort = 15268
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_arm9_known_hosts"

$remote = @'
set +e
date -u
python3 <<'PY'
from pathlib import Path
import json, re, subprocess

ARM = Path("/workspace/edullm-runs/hpo-moe/warmup-quadratic10-mtld-256ki")
LOG = Path("/workspace/hpo-moe-arm9-256ki.log")
CKPT = ARM / "checkpoints"
PROG = ARM / "progress"

lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
step = None
for line in reversed(lines):
    m = re.search(r"\[step=(\d+)/38144", line)
    if m:
        step = int(m.group(1))
        break

durable = {}
marker = PROG / "last_durable_step.json"
if marker.exists():
    durable = json.loads(marker.read_text())

ckpts = sorted(
    [p.name for p in CKPT.iterdir() if p.is_dir() and p.name.startswith("step")],
    key=lambda n: int(n[4:].split("-")[0]),
)
task_loss = sorted(PROG.glob("task_loss_results/step*_task_loss.json"))
last_task = task_loss[-1].name if task_loss else "none"

workers = subprocess.check_output("pgrep -c -f '[h]po_moe_arm9_entrypoint.py' || echo 0", shell=True).decode().strip()
launcher = subprocess.check_output("pgrep -c -f 'launch_hpo_moe_arm9' || echo 0", shell=True).decode().strip()
restart = (PROG / "restart_after_checkpoint.json").exists()
gpu = subprocess.check_output(
    "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1",
    shell=True,
).decode().strip()

events = []
for needle in (
    "Saving checkpoint for step 38000",
    "Saving checkpoint for step 38144",
    "task_loss",
    "Resuming arm 9 from durable step",
    "hard_stop",
):
    hits = [i for i, l in enumerate(lines) if needle in l]
    if hits:
        events.append(f"{needle}@{hits[-1]}")

print(
    "step={step} durable={durable_step} task_loss_complete={tlc} last_task_loss={lt} "
    "ckpts={ckpts} workers={workers} launcher={launcher} restart={restart} gpu={gpu}".format(
        step=step,
        durable_step=durable.get("last_durable_step"),
        tlc=durable.get("task_loss_complete"),
        lt=last_task,
        ckpts=",".join(ckpts[-3:]),
        workers=workers,
        launcher=launcher,
        restart=restart,
        gpu=gpu,
    )
)
if events:
    print("events=" + ";".join(events[-6:]))
PY
tail -8 /workspace/hpo-moe-arm9-256ki.log 2>/dev/null
'@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

while ($true) {
  $out = & $sshExe -i $sshKey -p $SshPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
    "root@$SshHost" "echo $encoded | base64 -d | bash" 2>&1
  Write-Output "AGENT_LOOP_TICK_ARM9_38K $($out -join ' | ')"
  Start-Sleep -Seconds $IntervalSeconds
}
