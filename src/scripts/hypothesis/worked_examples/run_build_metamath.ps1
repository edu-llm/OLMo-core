# Build MetaMathQA worked-examples pack (4 arms) + validate + tokenize + eval check.
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $Root
$Out = Join-Path $Root "data\worked_examples_metamath_v0"

$MaxTrain = if ($env:WE_MAX_TRAIN) { $env:WE_MAX_TRAIN } else { "10000" }
$MaxHoldout = if ($env:WE_MAX_HOLDOUT) { $env:WE_MAX_HOLDOUT } else { "1000" }
$Types = if ($env:WE_TYPES) { $env:WE_TYPES } else { "GSM" }

python src/scripts/hypothesis/worked_examples/build_from_metamath.py `
  --out-dir $Out `
  --max-train $MaxTrain `
  --max-holdout $MaxHoldout `
  --types $Types

python src/scripts/hypothesis/worked_examples/validate_pack.py --pack-dir $Out
python src/scripts/hypothesis/worked_examples/tokenize_arms.py --pack-dir $Out --tokenizer dolma2
python src/scripts/hypothesis/worked_examples/export_eval.py --pack-dir $Out
Write-Host "Pack ready at $Out"
