param(
  [int]$IntervalSeconds = 20
)

$ErrorActionPreference = "Continue"
$createPod = "C:\alpha_ai\edullm\scripts\runpod\smollm2_colmlm\create_idle_pod.js"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$sshExe = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_known_hosts"
$endpoints = @(
  @{ Role = "full-soup"; Host = "216.249.100.66"; Port = 22986 },
  @{ Role = "no-centaur"; Host = "216.249.100.66"; Port = 22414 },
  @{ Role = "no-proxy"; Host = "185.216.23.177"; Port = 33206 }
)

while ($true) {
  $now = (Get-Date).ToUniversalTime().ToString("o")
  try {
    $pods = node $createPod --list "hpo-three-arm" | ConvertFrom-Json
    foreach ($pod in @($pods)) {
      if ($pod.status -ne "RUNNING") {
        Write-Output "HPO_MONITOR_ALERT pod=$($pod.name) status=$($pod.status) at=$now"
      }
    }
  }
  catch {
    Write-Output "HPO_MONITOR_ALERT api_check_failed=$($_.Exception.Message) at=$now"
  }

  foreach ($endpoint in $endpoints) {
    $remote = @'
set -o pipefail
disk=$(df -P /workspace | awk 'NR==2 {gsub("%","",$5); print $5}')
if [ "$disk" -ge 90 ]; then echo "disk_${disk}pct"; fi
if ! pgrep -f '/\.edullm/runpod/launch\.sh' >/dev/null; then
  for f in /workspace/edullm-runs/hpo-probe/*/*/last-exit-code; do
    [ -f "$f" ] || continue
    code=$(cat "$f")
    [ "$code" = "0" ] || echo "$f exit=$code"
  done
fi
'@
    $remote = $remote -replace "`r", ""
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remote))
    $remoteCommand = "echo $encoded | base64 -d | bash"
    $result = & $sshExe -i $sshKey -p $endpoint.Port -o StrictHostKeyChecking=no `
      -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=15 `
      "root@$($endpoint.Host)" $remoteCommand 2>&1
    $sshExit = $LASTEXITCODE
    if ($sshExit -ne 0) {
      $oneLine = ($result -join " ") -replace "\s+", " "
      Write-Output "HPO_MONITOR_ALERT role=$($endpoint.Role) ssh_exit=$sshExit detail=$oneLine at=$now"
    }
    elseif ($result) {
      $oneLine = ($result -join " ") -replace "\s+", " "
      Write-Output "HPO_MONITOR_ALERT role=$($endpoint.Role) detail=$oneLine at=$now"
    }
  }
  Start-Sleep -Seconds $IntervalSeconds
}
