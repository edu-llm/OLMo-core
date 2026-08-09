param(
  [int]$IntervalSeconds = 300
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_known_hosts"
$log = Join-Path $env:TEMP "hpo_5min_reports.log"

$remote = @'
set -euo pipefail
utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 <<'PY'
from pathlib import Path
STEPS = 872

def max_step(trial: Path) -> int:
    mx = 0
    for step_dir in trial.iterdir():
        if not step_dir.is_dir() or not step_dir.name.startswith("step"):
            continue
        name = step_dir.name[4:].split("-")[0]
        try:
            mx = max(mx, int(name))
        except ValueError:
            pass
    return mx

def summarize(root: str, label: str) -> None:
    td = Path(root) / "checkpoints/trials"
    if not td.exists():
        print(f"{label}: missing")
        return
    rows = [max_step(t) for t in sorted(td.iterdir()) if t.is_dir()]
    if not rows:
        print(f"{label}: trials=0")
        return
    avg = sum(rows) / len(rows)
    done = sum(1 for x in rows if x >= STEPS)
    print(
        f"{label}: trials={len(rows)} avg_step={avg:.0f} "
        f"({100 * avg / STEPS:.0f}% of 50M) done50M={done}/{len(rows)}"
    )

for root, label in [
    ("/workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4", "proxy-v4"),
    ("/workspace/edullm-runs/hpo-probe/no_proxy/aligned-v4", "no-proxy-v4"),
    ("/workspace/edullm-runs/hpo-probe/no_centaur/aligned-v4", "no-centaur-v4"),
    ("/workspace/edullm-runs/hpo-probe/full_acronym_soup/aligned-v4", "full-soup-v4"),
]:
    summarize(root, label)
PY
disk=$(df -P /workspace | awk 'NR==2{gsub("%","",$5); print $5}')
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END {if(n) printf "%.0f", s/n; else print "NA"}')
launch=$(pgrep -fc '/\.edullm/runpod/launch\.sh' || true)
segments=$(pgrep -fc 'run-segment' || true)
elapsed=$(ps -o etime= -p $(pgrep -f 'entrypoint.py hpo-' | head -1) 2>/dev/null | tr -d ' ' || true)
evidence=$([ -s /workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json ] && echo yes || echo no)
echo "meta utc=$utc disk=${disk}% gpu=${gpu}% launch=$launch segments=$segments elapsed=${elapsed:-NA} evidence=$evidence"
if [ "$disk" -ge 90 ]; then echo "ALERT disk_${disk}pct"; fi
if [ "$launch" -eq 0 ]; then
  for f in /workspace/edullm-runs/hpo-probe/*/*/last-exit-code; do
    [ -f "$f" ] || continue
    code=$(cat "$f")
    [ "$code" = "0" ] || echo "ALERT ${f#/workspace/edullm-runs/hpo-probe/} exit=$code"
  done
fi
if pgrep -f 'entrypoint.py hpo-proxy-cohort-aligned-v4' >/dev/null; then
  echo "ACTIVE proxy-cohort-v4"
elif pgrep -f 'entrypoint.py hpo-no_proxy-aligned-v4' >/dev/null; then
  echo "ACTIVE no-proxy-v4"
elif pgrep -f 'entrypoint.py hpo-no_centaur-aligned-v4' >/dev/null; then
  echo "ACTIVE no-centaur-v4"
elif pgrep -f 'entrypoint.py hpo-full_acronym_soup-aligned-v4' >/dev/null; then
  echo "ACTIVE full-soup-v4"
else
  echo "ACTIVE none"
fi
grep -R -m1 -E 'Traceback|RuntimeError|No space left on device|OutOfMemoryError|CUDA error|NCCL.*error' \
  /workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4 \
  /workspace/edullm-runs/hpo-probe/no_proxy/aligned-v4 \
  /workspace/hpo-proxy-cohort-aligned-v4.log \
  /workspace/hpo-no-proxy-aligned-v4.log 2>/dev/null || echo "errors=none"
'@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

$endpoints = @(
  @{ Role = "proxy"; Host = "216.249.100.66"; Port = 22414 },
  @{ Role = "no-proxy"; Host = "185.216.23.177"; Port = 33206 },
  @{ Role = "full-soup"; Host = "216.249.100.66"; Port = 22986 }
)

while ($true) {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  foreach ($endpoint in $endpoints) {
    $out = & $sshExe -i $sshKey -p $endpoint.Port -o StrictHostKeyChecking=no `
      -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=25 `
      "root@$($endpoint.Host)" "echo $encoded | base64 -d | bash" 2>&1
    $line = "[$ts] role=$($endpoint.Role) $($out -join ' | ')"
    Add-Content -Path $log -Value $line
    Write-Output "HPO_5MIN $line"
  }
  Start-Sleep -Seconds $IntervalSeconds
}
