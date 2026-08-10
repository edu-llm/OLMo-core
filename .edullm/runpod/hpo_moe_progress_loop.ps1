param(
  [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_known_hosts"
$log = Join-Path $env:TEMP "hpo_moe_progress.log"
$hostName = "103.207.149.77"
$port = 11746

$remote = @'
set -euo pipefail
utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pidfile=/workspace/hpo-moe-no-proxy-control-128ki.pid
log=/workspace/hpo-moe-no-proxy-control-128ki.log
arm=/workspace/edullm-runs/hpo-moe/no-proxy-optimized-control-128ki
alive=no
if [ -f "$pidfile" ] && ps -p "$(cat "$pidfile")" >/dev/null 2>&1; then alive=yes; fi
train=$(pgrep -fc 'torch.distributed.run.*hpo_control_entrypoint' || true)
disk=$(df -P /workspace | awk 'NR==2{gsub("%","",$5); print $5}')
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END {if(n) printf "%.0f", s/n; else print "NA"}')
python3 <<'PY'
import re
from pathlib import Path
log = Path("/workspace/hpo-moe-no-proxy-control-128ki.log")
text = log.read_text(errors="replace") if log.exists() else ""
start = text.rfind("wandb: setting up run ")
seg = text[start:] if start >= 0 else text
steps = re.findall(r"\[step=([0-9,]+)/([0-9,]+)(?:,epoch=[^,\]]+)?(?:,eta=([^\]]+))?\]", seg)
tps = re.findall(r"throughput/device/TPS \(actual avg\)=([0-9,]+)", seg)
ce = re.findall(r"train/CE loss=([0-9.eE+,/-]+)", seg)
run = re.findall(r"View run at (https://\S+)", seg)
step = steps[-1] if steps else ("NA", "NA", "")
print(
    f"step={step[0]}/{step[1]} eta={step[2] or 'NA'} "
    f"tps={tps[-1] if tps else 'NA'} ce={ce[-1] if ce else 'NA'} "
    f"run={run[-1] if run else 'NA'}"
)
print("traceback", "Traceback" in seg)
print("oom", "out of memory" in seg.lower())
print("no_space", "No space left" in seg)
PY
exit_code=$(cat "$arm/last-exit-code" 2>/dev/null || echo NA)
ck=$(ls -1d "$arm/checkpoints"/step* 2>/dev/null | tail -1 || echo none)
echo "meta utc=$utc alive=$alive train_procs=$train disk=${disk}% gpu=${gpu}% exit=$exit_code checkpoint=$ck"
if [ "$alive" != yes ] || [ "$train" -eq 0 ]; then echo "ALERT training_not_running"; fi
if [ "$disk" -ge 85 ]; then echo "ALERT disk_${disk}pct"; fi
if [ "$alive" != yes ] && [ "$train" -eq 0 ] && [ "$exit_code" != NA ] && [ "$exit_code" != 0 ]; then
  echo "ALERT last_exit=$exit_code"
fi
'@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

while ($true) {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $out = & $sshExe -i $sshKey -p $port -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
  "root@$hostName" "echo $encoded | base64 -d | bash" 2>&1
  $summary = ($out | Where-Object { $_ -notmatch '^\s*$' }) -join ' | '
  $line = "[$ts] $summary"
  Add-Content -Path $log -Value $line
  Write-Output "HPO_MOE_PROGRESS $line"
  $prompt = if ($summary -match 'ALERT|traceback True|oom True|no_space True') {
    "hpo-moe 128ki run needs attention: $summary"
  } else {
    "Briefly report hpo-moe 128ki progress with step/tps/CE/ETA from: $summary"
  }
  $payload = @{ prompt = $prompt } | ConvertTo-Json -Compress
  Write-Output "AGENT_LOOP_TICK_hpo_moe $payload"
  Start-Sleep -Seconds $IntervalSeconds
}
