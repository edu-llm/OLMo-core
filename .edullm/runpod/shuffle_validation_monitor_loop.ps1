param(
  [int]$IntervalSeconds = 300,
  [string]$SshHost = "216.249.100.66",
  [int]$SshPort = 22687,
  [string]$RunSlot = "shuffle-mtld-370m-mb32k-v1",
  [string]$WandbRunId = "7d40f1bd16928595949f674123405573"
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_shuffle_validation_known_hosts"

$prompt = @"
Inspect the dense OLMo2-370M shuffle-baseline validation (control pacing, no curriculum learning) on pod root@$SshHost`:$SshPort, job /workspace/edullm-runs/hpo-validation/$RunSlot, W&B hpo-validation run $WandbRunId. Verify process/GPU health, current step/38146, CE/PPL/MFU/TPS/ETA, exact shuffle/control pacing identity (curriculum_learning=false, pacing=control), global batch 262144, rank microbatch 32768, checkpoint and 20-label eval durability, disk headroom, and errors. Automatically diagnose and repair confirmed recoverable faults from the latest checkpoint without changing scientific identity; do not merely report faults. If 32768 OOMs, use 16384 as the performance-only fallback. Stop monitoring after a clean final checkpoint/eval/artifact and exit 0. Report briefly.
"@

$promptJson = ($prompt -replace '\\', '\\\\' -replace '"', '\"')

$remote = @"
set +e
date -u
printf "workers="; pgrep -c -f "[s]huffle_validation_entrypoint.py" || echo 0
python3 <<'PY'
from pathlib import Path
import json, re

TOTAL_STEPS = 38146
LOG = Path("/workspace/edullm-runs/hpo-validation/$RunSlot/run.log")
JOB = Path("/workspace/edullm-runs/hpo-validation/$RunSlot")

lines = LOG.read_text(errors="replace").splitlines() if LOG.exists() else []
errs = [
    line
    for line in lines
    if any(token in line for token in ("OutOfMemory", "Traceback", "ChildFailedError"))
]

step = eta = mfu = tps = ce = ppl = None
for line in reversed(lines):
    if step is None:
        match = re.search(r"\[step=(\d+)/%d" % TOTAL_STEPS, line)
        if match:
            step = int(match.group(1))
    if eta is None:
        match = re.search(r"eta=([^,\]]+)", line)
        if match:
            eta = match.group(1)
    if ce is None and "train/CE loss=" in line:
        match = re.search(r"train/CE loss=([0-9.]+)", line)
        if match:
            ce = match.group(1)
    if ppl is None and "train/PPL=" in line:
        match = re.search(r"train/PPL=([0-9.]+)", line)
        if match:
            ppl = match.group(1)
    if mfu is None and "throughput/device/MFU (actual avg)=" in line:
        mfu = line.split("=", 1)[1].strip()
    if tps is None and "throughput/device/TPS (actual avg)=" in line:
        tps = line.split("=", 1)[1].strip()
    if step is not None and eta is not None and mfu is not None and tps is not None:
        break

exit_code = None
exit_path = JOB / "last-exit-code"
if exit_path.exists():
    exit_code = exit_path.read_text().strip()

checkpoints = sorted(
    int(path.name.replace("step", ""))
    for path in (JOB / "checkpoints").glob("step*")
    if path.is_dir() and path.name[4:].isdigit()
)
latest_ckpt = checkpoints[-1] if checkpoints else None
pct = None if step is None else round(100 * step / TOTAL_STEPS, 2)
print(
    "step={step}/{total} pct={pct} eta={eta} ce={ce} ppl={ppl} mfu={mfu} tps={tps} "
    "latest_ckpt={latest_ckpt} exit={exit_code} errors={errors}".format(
        step=step,
        total=TOTAL_STEPS,
        pct=pct,
        eta=eta or "unknown",
        ce=ce,
        ppl=ppl,
        mfu=mfu,
        tps=tps,
        latest_ckpt=latest_ckpt,
        exit_code=exit_code,
        errors=len(errs),
    )
)
PY
df -h /workspace | awk 'NR==2 {print "disk_free=" $4}'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1 | awk '{print "gpu0=" $0}'
"@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

while ($true) {
  $out = & $sshExe -i $sshKey -p $SshPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
    "root@$SshHost" "echo $encoded | base64 -d | bash" 2>&1
  $status = ($out | Where-Object { $_ -match "step=" }) -join " | "
  if (-not $status) {
    $status = ($out | Select-Object -Last 3) -join " | "
  }
  Write-Output "AGENT_LOOP_TICK_SHUFFLE_VALIDATION_MB32K $status | {`"prompt`":`"$promptJson`"}"
  Start-Sleep -Seconds $IntervalSeconds
}
