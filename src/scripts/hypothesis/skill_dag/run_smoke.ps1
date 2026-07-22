param(
  [Parameter(Mandatory = $true)][string]$DataDir,
  [string]$MixName = "natural",
  [int]$Steps = 50,
  [string]$TrainScript = "OLMo2/OLMo2-190M.py"
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $Root
$RunName = "skilldag-smoke-$MixName"
$MixFile = Join-Path $Root "src\scripts\hypothesis\skill_dag\configs\$MixName.json"
if (-not (Test-Path $MixFile)) {
  $MixFile = Join-Path $DataDir "manifests\mixes\natural.json"
}

Write-Host "Skill-DAG smoke"
Write-Host "  data: $DataDir"
Write-Host "  mix:  $MixFile"
Write-Host "  script: src/scripts/train/$TrainScript"

$scriptPath = "src/scripts/train/$TrainScript"
python $scriptPath dry_run $RunName local `
  --save-folder="./runs/$RunName" `
  --trainer.hard_stop="{value: $Steps, unit: steps}"
if ($LASTEXITCODE -ne 0) {
  python $scriptPath train_single $RunName `
    --save-folder="./runs/$RunName" `
    --trainer.hard_stop="{value: $Steps, unit: steps}"
}

Write-Host "Mix file: $MixFile"
