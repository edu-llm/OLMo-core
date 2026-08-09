param(
  [Parameter(Mandatory = $true)][string]$HostName,
  [Parameter(Mandatory = $true)][int]$Port,
  [Parameter(Mandatory = $true)][string]$Role,
  [string]$Commit = "064a5b2ab1b8854b14a3153e7902655c89da3e57",
  [string]$Profile = "sbsandbox"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$adapterRoot = Join-Path $repoRoot ".edullm\runpod"
$mintScript = "C:\alpha_ai\edullm\scripts\farmshare\mint_aws_session_local.ps1"
$sshKey = Join-Path $env:USERPROFILE ".ssh\runpod_ed25519"
$knownHosts = Join-Path $env:TEMP "runpod_hpo_known_hosts"
$token = [guid]::NewGuid().ToString("N")
$awsEnv = Join-Path $env:TEMP "aws-hpo-$Role-$token.env"
$runtimeEnv = Join-Path $env:TEMP "runtime-hpo-$Role-$token.env"

function Invoke-PodSsh {
  param([string]$Command)
  & ssh -i $sshKey -p $Port -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=30 `
    "root@${HostName}" $Command
  if ($LASTEXITCODE -ne 0) { throw "SSH failed for ${Role}: $Command" }
}

function Copy-ToPod {
  param([string]$LocalPath, [string]$RemotePath)
  & scp -i $sshKey -P $Port -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts $LocalPath "root@${HostName}:$RemotePath"
  if ($LASTEXITCODE -ne 0) { throw "SCP failed for ${Role}: $LocalPath" }
}

try {
  Write-Host "[$Role] checking SSH"
  Invoke-PodSsh "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"

  Write-Host "[$Role] transferring local adapter"
  Invoke-PodSsh "rm -rf /workspace/hpo-runpod-adapter; mkdir -p /workspace/hpo-runpod-adapter"
  & scp -i $sshKey -P $Port -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -r (Join-Path $adapterRoot "*") `
    "root@${HostName}:/workspace/hpo-runpod-adapter/"
  if ($LASTEXITCODE -ne 0) { throw "adapter SCP failed for $Role" }

  & ssh -i $sshKey -p $Port -o StrictHostKeyChecking=no `
    -o UserKnownHostsFile=$knownHosts -o BatchMode=yes -o ConnectTimeout=30 `
    "root@${HostName}" "test `"`$(cat /workspace/edullm-bootstrap/hpo-probe.commit 2>/dev/null)`" = '$Commit'"
  $bootstrapReady = $LASTEXITCODE -eq 0
  if ($bootstrapReady) {
    Write-Host "[$Role] bootstrap already complete for commit $Commit"
  }
  else {
    Write-Host "[$Role] bootstrapping commit $Commit"
    Invoke-PodSsh "chmod +x /workspace/hpo-runpod-adapter/bootstrap.sh && REPO_DIR=/workspace/OLMo-core OLMO_CORE_COMMIT_SHA='$Commit' bash /workspace/hpo-runpod-adapter/bootstrap.sh"
  }

  Write-Host "[$Role] minting temporary S3 session"
  & $mintScript -Profile $Profile -OutputPath $awsEnv
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $awsEnv)) {
    throw "AWS session mint failed for $Role"
  }
  Copy-ToPod $awsEnv "/workspace/aws-session.env"
  Invoke-PodSsh "chmod 600 /workspace/aws-session.env; cd /workspace/OLMo-core && PYTHONPATH=/workspace/OLMo-core/src:/workspace/OLMo-core/.edullm python3 .edullm/runpod/stage_inputs.py --credentials-file /workspace/aws-session.env"

  $wandbKey = (Get-Content (Join-Path $env:USERPROFILE ".wandb_api_key") -Raw).Trim()
  $openaiKey = (Get-Content (Join-Path $env:USERPROFILE ".openai_api_key") -Raw).Trim()
  $runtimeContent = @"
# Generated locally for HPO RunPod. Do not commit.
export WANDB_API_KEY='$wandbKey'
export WANDB_ENTITY='eduLLM'
export OPENAI_API_KEY='$openaiKey'
export OPENAI_BASE_URL='https://gateway.truefoundry.ai/v1'
"@
  [IO.File]::WriteAllText(
    $runtimeEnv,
    $runtimeContent.Replace("`r`n", "`n"),
    [Text.ASCIIEncoding]::new()
  )
  Copy-ToPod $runtimeEnv "/workspace/wandb-session.env"
  Invoke-PodSsh 'chmod 600 /workspace/wandb-session.env; test ! -e /workspace/aws-session.env; test -f /workspace/edullm-inputs/hpo-probe/ready.json; test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 8'
  Write-Host "PREPARED role=$Role host=${HostName}:$Port commit=$Commit"
}
finally {
  Remove-Item -Force $awsEnv, $runtimeEnv -ErrorAction SilentlyContinue
}
