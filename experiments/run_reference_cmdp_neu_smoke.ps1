param(
    [string]$RepoRoot = "D:\drft-v2",
    [string]$DatasetRoot = "D:\drft-v2\data\NEU\split\neu_det_split",
    [string]$OutputRoot = "D:\drft-v2\generation\neu-det-pipeline\outputs",
    [int]$TrainSteps = 1,
    [int]$GenerateSteps = 4,
    [int]$MaxTrainSamples = 1,
    [int]$MaxGenerateSamples = 1
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $RepoRoot "generation\neu-det-pipeline\src"
$env:TMP = Join-Path $RepoRoot ".tmp"
$env:TEMP = Join-Path $RepoRoot ".tmp"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

$ldmOut = Join-Path $OutputRoot "reference_ldm_neu_smoke"
$cmdpOut = Join-Path $OutputRoot "reference_cmdp_neu_smoke"

Set-Location $RepoRoot

python -m neu_det_pipeline.cli train-reference-ldm `
    $DatasetRoot `
    --output-dir $ldmOut `
    --train-split train `
    --per-class `
    --steps $TrainSteps `
    --max-train-samples $MaxTrainSamples `
    --batch-size 1 `
    --resolution 512

python -m neu_det_pipeline.cli generate-reference-cmdp `
    $DatasetRoot `
    $ldmOut `
    --output-dir $cmdpOut `
    --generation-split train `
    --max-samples $MaxGenerateSamples `
    --steps $GenerateSteps `
    --resolution 512 `
    --make-mixed-dataset

