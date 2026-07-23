param(
  [string]$OutDir = "./data/worked_examples_gsm8k_v0",
  [int]$MaxTrain = 200,
  [int]$MaxTest = 100
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Set-Location $Root
$Here = Join-Path $Root "src\scripts\hypothesis\worked_examples"

python (Join-Path $Here "build_from_gsm8k.py") --out-dir $OutDir --max-train $MaxTrain --max-test $MaxTest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python (Join-Path $Here "validate_pack.py") --pack-dir $OutDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python (Join-Path $Here "export_eval.py") --pack-dir $OutDir
Write-Host "Build+validate OK. Optional next: tokenize_arms.py --pack-dir $OutDir"
