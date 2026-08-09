param(
  [Parameter(Mandatory = $true)][string]$IssueKey,
  [Parameter(Mandatory = $true)][string]$Role,
  [Parameter(Mandatory = $true)][string]$Details,
  [int]$CooldownSeconds = 900
)

$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stateFile = Join-Path $env:TEMP "hpo_sol_fix_state.json"
$logFile = Join-Path $env:TEMP "hpo_sol_fix.log"
$model = "gpt-5.6-sol-high"

function Write-Log([string]$Message) {
  $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
  Add-Content -Path $logFile -Value $line
  Write-Output $line
}

function Get-AgentCli {
  $cmd = Get-Command agent -ErrorAction SilentlyContinue
  $candidates = @(
    $(if ($cmd) { $cmd.Source }),
    (Join-Path $env:USERPROFILE ".local\bin\agent.exe"),
    (Join-Path $env:LOCALAPPDATA "cursor-agent\agent.exe"),
    (Join-Path $env:USERPROFILE "AppData\Local\cursor-agent\agent.exe")
  ) | Where-Object { $_ -and (Test-Path $_) }
  return $candidates | Select-Object -First 1
}

$state = @{}
if (Test-Path $stateFile) {
  try {
    $parsed = Get-Content $stateFile -Raw | ConvertFrom-Json
    $state = @{}
    foreach ($prop in $parsed.PSObject.Properties) { $state[$prop.Name] = $prop.Value }
  }
  catch { $state = @{} }
}
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
if ($state.ContainsKey($IssueKey)) {
  $last = [int64]$state[$IssueKey]
  if (($now - $last) -lt $CooldownSeconds) {
    Write-Log "SKIP cooldown issue=$IssueKey role=$Role"
    exit 0
  }
}

$prompt = @"
HPO RunPod remediation for eduLLM/hpo-probe.

Repo: $repoRoot
Role: $Role
Issue: $Details

Pods:
- proxy / no-centaur: 216.249.100.66:22414
- no-proxy: 185.216.23.177:33206
- full-soup: 216.249.100.66:22986
SSH key: $env:USERPROFILE\.ssh\runpod_ed25519

Active slot is aligned-v4. Fix the issue on the pod (disk cleanup, restart launch.sh, patch adapter on pod and locally if needed). Do not start a 5-minute monitor loop. Keep the 2-minute loop as the only status monitor.
"@

$promptFile = Join-Path $env:TEMP ("hpo_sol_fix_" + [guid]::NewGuid().ToString("N") + ".txt")
Set-Content -Path $promptFile -Value $prompt -Encoding UTF8

$agentCli = Get-AgentCli
if (-not $agentCli) {
  Write-Log "NO_AGENT_CLI issue=$IssueKey role=$Role details=$Details prompt=$promptFile"
  exit 2
}

$args = @(
  "-p",
  "--force",
  "--trust",
  "--model", $model,
  "--workspace", $repoRoot,
  "--output-format", "text",
  (Get-Content $promptFile -Raw)
)

Write-Log "START issue=$IssueKey role=$Role agent=$agentCli"
$proc = Start-Process -FilePath $agentCli -ArgumentList $args -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $env:TEMP "hpo_sol_fix_out_$IssueKey.log") `
  -RedirectStandardError (Join-Path $env:TEMP "hpo_sol_fix_err_$IssueKey.log")
$state[$IssueKey] = $now
($state | ConvertTo-Json -Compress) | Set-Content -Path $stateFile -Encoding UTF8
Write-Log "SPAWNED pid=$($proc.Id) issue=$IssueKey"
