param(
  [int]$IntervalSeconds = 120
)

$ErrorActionPreference = "Continue"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_known_hosts"
$log = Join-Path $env:TEMP "hpo_2min_reports.log"
$fixScript = Join-Path $PSScriptRoot "hpo_invoke_sol_fix.ps1"

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
    ("/workspace/edullm-runs/hpo-probe/no_centaur/exact-v1", "no-centaur-exact"),
]:
    summarize(root, label)
PY
disk=$(df -P /workspace | awk 'NR==2{gsub("%","",$5); print $5}')
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END {if(n) printf "%.0f", s/n; else print "NA"}')
launch=$(pgrep -fc 'entrypoint.py hpo-' || true)
segments=$(pgrep -fc 'run-segment' || true)
elapsed=$(ps -o etime= -p $(pgrep -f 'entrypoint.py hpo-' | head -1) 2>/dev/null | tr -d ' ' || true)
evidence=$([ -s /workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json ] && echo yes || echo no)
decision=$(python3 -c 'import json; from pathlib import Path; p=Path("/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json"); print(json.loads(p.read_text()).get("decision", "none") if p.exists() else "none")' 2>/dev/null || echo none)
completed=$(python3 -c 'from pathlib import Path; roots=Path("/workspace/edullm-runs/hpo-probe"); done=[]; [done.append(str(p.parent.relative_to(roots))) for p in roots.glob("*/*/study-result.json") if (p.parent/"last-exit-code").exists() and (p.parent/"last-exit-code").read_text().strip()=="0"]; print(",".join(done) or "none")')
echo "meta utc=$utc disk=${disk}% gpu=${gpu}% launch=$launch segments=$segments elapsed=${elapsed:-NA} evidence=$evidence decision=$decision completed=$completed"
if [ "$disk" -ge 90 ]; then echo "ALERT disk_${disk}pct"; fi
if [ "$launch" -eq 0 ]; then
  for f in /workspace/edullm-runs/hpo-probe/*/*/last-exit-code; do
    [ -f "$f" ] || continue
    code=$(cat "$f")
    if [[ "$f" == *"/proxy-cohort/"* ]] && [ "$decision" = "reporting_only" ]; then
      continue
    fi
    [ "$code" = "0" ] || echo "ALERT ${f#/workspace/edullm-runs/hpo-probe/} exit=$code"
  done
fi
if [ "$completed" != "none" ] && [ "$launch" -eq 0 ]; then
  echo "TERMINAL completed:$completed"
elif pgrep -f 'entrypoint.py hpo-proxy-cohort-aligned-v4' >/dev/null; then
  echo "ACTIVE proxy-cohort-v4"
elif pgrep -f 'entrypoint.py hpo-no_centaur-exact-v1' >/dev/null; then
  echo "ACTIVE no-centaur-exact"
elif [ "$decision" = "reporting_only" ]; then
  echo "TERMINAL proxy-gate-reporting-only"
else
  echo "ACTIVE none"
fi
if [ "$launch" -gt 0 ] || [ "$decision" = "reporting_only" ] || [ "$completed" != "none" ]; then
  echo "errors=none"
else
  tail -n 120 \
    /workspace/hpo-proxy-cohort-aligned-v4.log \
    /workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4/run.log \
    /workspace/edullm-runs/hpo-probe/no_centaur/exact-v1/run.log 2>/dev/null |
    grep -m1 -E 'Traceback|RuntimeError|No space left on device|OutOfMemoryError|CUDA error|NCCL.*error' ||
    echo "errors=none"
fi
'@

$remote = $remote -replace "`r", ""
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))

$endpoints = @(
  @{ Role = "no-centaur"; Host = "216.249.100.66"; Port = 22414 },
  @{ Role = "no-proxy"; Host = "185.216.23.177"; Port = 33206 }
)

function Invoke-PodStatus {
  param(
    [string]$HostName,
    [int]$Port,
    [string]$EncodedRemote,
    [int]$TimeoutSeconds = 60
  )

  $job = Start-Job -ScriptBlock {
    param($SshExe, $SshKey, $KnownHosts, $HostName, $Port, $EncodedRemote)
    & $SshExe -i $SshKey -p $Port `
      -o StrictHostKeyChecking=no `
      -o UserKnownHostsFile=$KnownHosts `
      -o BatchMode=yes `
      -o ConnectTimeout=15 `
      -o ServerAliveInterval=5 `
      -o ServerAliveCountMax=2 `
      "root@$HostName" "echo $EncodedRemote | base64 -d | bash" 2>&1
  } -ArgumentList $sshExe, $sshKey, $knownHosts, $HostName, $Port, $EncodedRemote

  $done = Wait-Job -Job $job -Timeout $TimeoutSeconds
  if (-not $done) {
    Stop-Job -Job $job -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    return @("ssh_timeout after ${TimeoutSeconds}s")
  }

  $out = Receive-Job -Job $job
  Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
  return @($out)
}
function Get-Issues([string]$Role, [string[]]$Lines) {
  $issues = @()
  $text = ($Lines -join "`n")
  if ($text -match 'SyntaxError|API_ERROR|ssh_exit=|ssh_timeout') {
    $issues += @{ Key = "monitor:$Role"; Detail = ($Lines -join ' | ') }
  }
  foreach ($line in $Lines) {
    if ($line -match '^ALERT (.+)$') {
      $issues += @{ Key = "alert:${Role}:$($Matches[1])"; Detail = $line }
    }
    if ($line -notmatch 'errors=none' -and $line -match 'Traceback|RuntimeError|OutOfMemoryError|CUDA error|No space left') {
      $issues += @{ Key = "error:$Role"; Detail = $line }
    }
  }
  if ($Role -in @('no-centaur', 'no-proxy') -and ($text -match 'ACTIVE none') -and ($text -notmatch 'launch=[1-9]') -and ($text -notmatch 'TERMINAL completed:')) {
    $issues += @{ Key = "idle:$Role"; Detail = "expected training arm idle on $Role pod" }
  }
  return $issues
}

while ($true) {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  foreach ($endpoint in $endpoints) {
    $lines = Invoke-PodStatus -HostName $endpoint.Host -Port $endpoint.Port -EncodedRemote $encoded
    $summary = ($lines | Where-Object { $_ -notmatch '^\s*$' } | Select-Object -First 6) -join ' | '
    $line = "[$ts] role=$($endpoint.Role) $summary"
    Add-Content -Path $log -Value $line
    Write-Output "HPO_2MIN $line"

    foreach ($issue in (Get-Issues -Role $endpoint.Role -Lines $lines)) {
      Write-Output "HPO_FIX_REQUIRED role=$($endpoint.Role) key=$($issue.Key)"
      if (Test-Path $fixScript) {
        Start-Process -FilePath "powershell.exe" -WindowStyle Hidden -ArgumentList @(
          "-NoProfile", "-ExecutionPolicy", "Bypass",
          "-File", $fixScript,
          "-IssueKey", $issue.Key,
          "-Role", $endpoint.Role,
          "-Details", $issue.Detail
        ) | Out-Null
      }
    }
  }
  Start-Sleep -Seconds $IntervalSeconds
}
