param(
  [Parameter(Mandatory = $true)][string]$DataDir,
  [string]$Pacing = "random",
  [string]$Metric = "compression_ratio",
  [int]$Steps = 50,
  [string]$TrainScript = "OLMo2/OLMo2-190M.py"
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $Root
$Hyp = Join-Path $Root "src\scripts\hypothesis"
$RunName = "cl-smoke-$Pacing-$Metric"

python (Join-Path $Hyp "curriculum\score_difficulty.py") --data-dir $DataDir
python (Join-Path $Hyp "curriculum\build_pacing_order.py") --data-dir $DataDir --pacing $Pacing --metric $Metric

$OrderFile = if ($Pacing -eq "random") {
  Join-Path $DataDir "orders\random__none.jsonl"
} else {
  Join-Path $DataDir "orders\${Pacing}__${Metric}.jsonl"
}

Write-Host "CL smoke order: $OrderFile"
$scriptPath = "src/scripts/train/$TrainScript"
python $scriptPath dry_run $RunName local `
  --save-folder="./runs/$RunName" `
  --trainer.hard_stop="{value: $Steps, unit: steps}"
if ($LASTEXITCODE -ne 0) {
  python $scriptPath train_single $RunName `
    --save-folder="./runs/$RunName" `
    --trainer.hard_stop="{value: $Steps, unit: steps}"
}

Write-Host "Order file: $OrderFile"
