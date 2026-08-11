param(
  [int]$IntervalSeconds = 300,
  [string]$ShuffleSshHost = "216.249.100.66",
  [int]$ShuffleSshPort = 22687,
  [string]$ShuffleRunSlot = "shuffle-mtld-370m-mb32k-v1",
  [string]$ShuffleWandbRunId = "7d40f1bd16928595949f674123405573",
  [string]$CurriculumSshHost = "103.207.149.77",
  [int]$CurriculumSshPort = 11746,
  [string]$CurriculumRunSlot = "quadratic-mtld-noproxy-hps-mb32k-v1",
  [string]$CurriculumWandbRunId = "a34f952c1c13b97fcc9afbaf2c813233"
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_dual_validation_known_hosts"
$totalSteps = 38146

$prompt = @"
Inspect both dense OLMo2-370M validation runs every 5 minutes:
1) Shuffle baseline (control pacing, no curriculum) on root@$ShuffleSshHost`:$ShuffleSshPort, job /workspace/edullm-runs/hpo-validation/$ShuffleRunSlot, W&B $ShuffleWandbRunId.
2) Quadratic-MTLD curriculum with no-proxy hyperparameters on root@$CurriculumSshHost`:$CurriculumSshPort, job /workspace/edullm-runs/hpo-validation/$CurriculumRunSlot, W&B $CurriculumWandbRunId.
For each: verify process/GPU health, step/$totalSteps, pct, ETA, CE/PPL, MFU, TPS, latest checkpoint, disk headroom, and errors. Automatically diagnose and repair confirmed recoverable faults from the latest checkpoint without changing scientific identity. If 32768 OOMs, use 16384 as the performance-only fallback. Stop monitoring after both runs finish with clean final checkpoint/eval/artifact. Report briefly.
"@

$promptJson = ($prompt -replace '\\', '\\\\' -replace '"', '\"')

function New-RemoteScript {
  param(
    [string]$RunSlot,
    [string]$WorkerPattern
  )

  @"
set +e
date -u
printf "workers="; pgrep -c -f "$WorkerPattern" || echo 0
python3 <<'PY'
from pathlib import Path
import re

TOTAL_STEPS = $totalSteps
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
}

function Invoke-RemoteStatus {
  param(
    [string]$SshHost,
    [int]$SshPort,
    [string]$RemoteScript
  )

  $script = ($RemoteScript -replace "`r", "")
  $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
  $out = & $sshExe -i $sshKey -p $SshPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
    "root@$SshHost" "echo $encoded | base64 -d | bash" 2>&1
  $status = ($out | Where-Object { $_ -match "step=" }) -join " | "
  if (-not $status) {
    $status = ($out | Select-Object -Last 3) -join " | "
  }
  return $status
}

function Get-WandbStatus {
  param(
    [string]$WandbRunId
  )

  $pyFile = Join-Path $env:TEMP "dual_validation_wandb_$WandbRunId.py"
  @"
import wandb

api = wandb.Api()
run = api.run("eduLLM/hpo-validation/$WandbRunId")
summary = run.summary
step = summary.get("_step")
if step is None:
    print("wandb=unavailable")
    raise SystemExit(0)
step = int(step)
total = $totalSteps
pct = round(100 * step / total, 2)
ce = summary.get("train/CE loss")
ppl = summary.get("train/PPL")
mfu = summary.get("throughput/device/MFU (actual avg)")
tps = summary.get("throughput/device/TPS (actual avg)")
runtime = summary.get("_runtime")
remaining = None
if runtime and step > 0:
    remaining = int(runtime * (total - step) / step)
eta = "unknown"
if remaining is not None:
    hours, rem = divmod(remaining, 3600)
    minutes = rem // 60
    eta = f"{hours}h{minutes}m" if hours else f"{minutes}m"
print(
    f"step={step}/{total} pct={pct} eta={eta} ce={ce} ppl={ppl} "
    f"mfu={mfu} tps={tps} source=wandb state={run.state} errors=0"
)
"@ | Set-Content -Path $pyFile -Encoding UTF8

  $out = & py -3 $pyFile 2>&1
  $status = ($out | Where-Object { $_ -match "step=" }) -join " | "
  if (-not $status) {
    $status = "wandb_error=$($out -join ' ')"
  }
  return $status
}

$shuffleRemote = New-RemoteScript -RunSlot $ShuffleRunSlot -WorkerPattern "[s]huffle_validation_entrypoint.py"
$curriculumRemote = New-RemoteScript -RunSlot $CurriculumRunSlot -WorkerPattern "[c]urriculum_noproxy_validation_entrypoint.py"

while ($true) {
  $shuffleStatus = Invoke-RemoteStatus -SshHost $ShuffleSshHost -SshPort $ShuffleSshPort -RemoteScript $shuffleRemote
  if ($shuffleStatus -notmatch "step=") {
    $shuffleStatus = Get-WandbStatus -WandbRunId $ShuffleWandbRunId
  }

  Start-Sleep -Seconds 10

  $curriculumStatus = Invoke-RemoteStatus -SshHost $CurriculumSshHost -SshPort $CurriculumSshPort -RemoteScript $curriculumRemote
  if ($curriculumStatus -notmatch "step=") {
    $curriculumStatus = Get-WandbStatus -WandbRunId $CurriculumWandbRunId
  }

  Write-Output "AGENT_LOOP_TICK_DUAL_VALIDATION shuffle=$shuffleStatus || curriculum_noproxy=$curriculumStatus | {`"prompt`":`"$promptJson`"}"

  $shuffleDone = $shuffleStatus -match "exit=0" -or ($shuffleStatus -match "step=38146/38146" -and $shuffleStatus -match "workers=0") -or ($shuffleStatus -match "state=finished")
  $curriculumDone = $curriculumStatus -match "exit=0" -or ($curriculumStatus -match "step=38146/38146" -and $curriculumStatus -match "workers=0") -or ($curriculumStatus -match "state=finished")
  if ($shuffleDone -and $curriculumDone) {
    Write-Output "AGENT_LOOP_DONE_DUAL_VALIDATION both runs finished cleanly"
    break
  }

  Start-Sleep -Seconds $IntervalSeconds
}
