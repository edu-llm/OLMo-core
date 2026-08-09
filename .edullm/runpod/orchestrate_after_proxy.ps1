param(
  [int]$IntervalSeconds = 15
)

$ErrorActionPreference = "Stop"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$scpExe = Join-Path $env:SystemRoot "System32\OpenSSH\scp.exe"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_known_hosts"
$proxyHost = "216.249.100.66"
$proxyPort = 22414
$fullHost = "216.249.100.66"
$fullPort = 22986
$evidence = "/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json"
$proxyJob = "/workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4"
$tempEvidence = Join-Path $env:TEMP ("proxy-evidence-" + [guid]::NewGuid().ToString("N") + ".json")

function Invoke-Remote {
  param(
    [string]$RemoteHost,
    [int]$RemotePort,
    [string]$Script
  )

  $normalized = $Script -replace "`r", ""
  $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalized))
  $result = & $sshExe -i $sshKey -p $RemotePort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=20 `
    "root@${RemoteHost}" "echo $encoded | base64 -d | bash" 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "SSH command failed for ${RemoteHost}:$RemotePort ($result)"
  }
  return $result
}

try {
  $lastHeartbeat = [DateTime]::MinValue
  while ($true) {
    $status = Invoke-Remote -RemoteHost $proxyHost -RemotePort $proxyPort -Script @'
set -euo pipefail
job=/workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4
evidence=/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json
if pgrep -f '[e]ntrypoint.py hpo-proxy-cohort-aligned-v4' >/dev/null; then
  echo ACTIVE
elif [ -f "$job/last-exit-code" ]; then
  code=$(cat "$job/last-exit-code")
  decision=$(
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("decision", ""))' \
      "$evidence" 2>/dev/null || true
  )
  if [ "$decision" = "reporting_only" ]; then
    echo TERMINAL:reporting_only
  elif [ "$code" = "0" ] && [ "$decision" = "prune_promote" ]; then
    echo READY
  else
    echo "FAILED:$code"
  fi
else
  echo STARTING
fi
'@
    $status = ($status | Select-Object -Last 1).Trim()
    if ($status -eq "READY") {
      break
    }
    if ($status -eq "TERMINAL:reporting_only") {
      Write-Output (
        "PROXY_HANDOFF_TERMINAL decision=reporting_only " +
        "downstream=not-launched checkpoints=preserved"
      )
      return
    }
    if ($status.StartsWith("FAILED:")) {
      throw "proxy cohort did not complete successfully: $status"
    }
    if (((Get-Date) - $lastHeartbeat).TotalSeconds -ge 60) {
      Write-Output "PROXY_HANDOFF_WAIT status=$status"
      $lastHeartbeat = Get-Date
    }
    Start-Sleep -Seconds $IntervalSeconds
  }

  & $scpExe -i $sshKey -P $proxyPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts "root@${proxyHost}:$evidence" $tempEvidence
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tempEvidence)) {
    throw "proxy evidence download failed"
  }
  Invoke-Remote -RemoteHost $fullHost -RemotePort $fullPort -Script @'
set -euo pipefail
mkdir -p /workspace/edullm-runs/hpo-probe/shared
'@
  & $scpExe -i $sshKey -P $fullPort -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts $tempEvidence `
    "root@${fullHost}:/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json"
  if ($LASTEXITCODE -ne 0) {
    throw "proxy evidence upload to full-soup failed"
  }

  Invoke-Remote -RemoteHost $proxyHost -RemotePort $proxyPort -Script @'
set -euo pipefail
source /workspace/wandb-session.env
job=/workspace/edullm-runs/hpo-probe/proxy-cohort/aligned-v4
test "$(cat "$job/last-exit-code")" = "0"
test -s /workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json
! pgrep -f '[e]ntrypoint.py hpo-proxy-cohort-aligned-v4' >/dev/null
python3 - <<'PY'
import json
from pathlib import Path

evidence = json.loads(
    Path("/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json").read_text()
)
if evidence.get("decision") != "prune_promote":
    raise SystemExit("proxy cohort did not pass the admission gate")
PY
# The evidence JSON is now present on both arm pods and mirrored to W&B. Trial
# checkpoints are superseded and cannot coexist with no-centaur on a 250 GB volume.
rm -rf "$job/checkpoints/trials"
setsid -f env MODE=no_centaur RUN_SLOT=aligned-v4 RECOVERY_MODE=fresh \
  bash /workspace/OLMo-core/.edullm/runpod/launch.sh \
  > /workspace/hpo-no-centaur-aligned-v4.log 2>&1
'@
  Invoke-Remote -RemoteHost $fullHost -RemotePort $fullPort -Script @'
set -euo pipefail
source /workspace/wandb-session.env
test -s /workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json
test -s /workspace/edullm-inputs/hpo-probe/ready.json
test ! -e /workspace/aws-session.env
setsid -f env MODE=full_acronym_soup RUN_SLOT=aligned-v4 RECOVERY_MODE=fresh \
  bash /workspace/OLMo-core/.edullm/runpod/launch.sh \
  > /workspace/hpo-full-soup-aligned-v4.log 2>&1
'@

  Start-Sleep -Seconds 5
  $noCentaur = Invoke-Remote -RemoteHost $proxyHost -RemotePort $proxyPort -Script @'
pgrep -f '[e]ntrypoint.py hpo-no_centaur-aligned-v4' >/dev/null && echo RUNNING
'@
  $fullSoup = Invoke-Remote -RemoteHost $fullHost -RemotePort $fullPort -Script @'
pgrep -f '[e]ntrypoint.py hpo-full_acronym_soup-aligned-v4' >/dev/null && echo RUNNING
'@
  if (($noCentaur | Select-Object -Last 1) -ne "RUNNING") {
    throw "no-centaur arm did not stay running after launch"
  }
  if (($fullSoup | Select-Object -Last 1) -ne "RUNNING") {
    throw "full-soup arm did not stay running after launch"
  }
  Write-Output "PROXY_HANDOFF_LAUNCHED no-centaur=RUNNING full-soup=RUNNING"
}
catch {
  Write-Output "PROXY_HANDOFF_FAILED $($_.Exception.Message)"
  exit 1
}
finally {
  Remove-Item -Force $tempEvidence -ErrorAction SilentlyContinue
}
