param(
  [int]$IntervalSeconds = 300,
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
printf "workers="; pgrep -c -f "[h]po_moe_arm9_entrypoint.py" || echo 0
python3 <<'PY'
from pathlib import Path
import json, re

TOTAL_STEPS = 38144
LOG = Path("/workspace/hpo-moe-arm9-256ki.log")
ARM = "/workspace/edullm-runs/hpo-moe/warmup-quadratic10-mtld-256ki"

lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
errs = [
    line
    for line in lines
    if any(token in line for token in ("OutOfMemory", "Traceback", "ChildFailedError"))
]

step = eta = mfu = tps = None
for line in reversed(lines):
    if step is None:
        match = re.search(r"\[step=(\d+)/%d" % TOTAL_STEPS, line)
        if match:
            step = int(match.group(1))
    if eta is None:
        match = re.search(r"eta=([^,\]]+)", line)
        if match:
            eta = match.group(1)
    if mfu is None and "throughput/device/MFU (actual avg)=" in line:
        mfu = line.split("=", 1)[1].strip()
    if tps is None and "throughput/device/TPS (actual avg)=" in line:
        tps = line.split("=", 1)[1].strip()
    if step is not None and eta is not None and mfu is not None and tps is not None:
        break

marker = Path(ARM) / "progress/last_durable_step.json"
durable = json.loads(marker.read_text())["last_durable_step"] if marker.exists() else None
pct = None if step is None else round(100 * step / TOTAL_STEPS, 2)
wandb = None if step is None else step * 2
print(
    "step={step}/{total} wandb~={wandb} pct={pct} eta={eta} mfu={mfu} tps={tps} "
    "durable={durable} errors={errors}".format(
        step=step,
        total=TOTAL_STEPS,
        wandb=wandb,
        pct=pct,
        eta=eta or "unknown",
        mfu=mfu,
        tps=tps,
        durable=durable,
        errors=len(errs),
    )
)
PY
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1
'@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

while ($true) {
  $out = & $sshExe -i $sshKey -p $SshPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
    "root@$SshHost" "echo $encoded | base64 -d | bash" 2>&1
  Write-Output "AGENT_LOOP_TICK_HPO_MOE_ARM9_256KI $($out -join ' | ')"
  Start-Sleep -Seconds $IntervalSeconds
}
