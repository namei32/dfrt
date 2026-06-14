param(
    [int]$Seed = 0,
    [int]$LoraSteps = 1000,
    [int]$GenerateCandidates = 5,
    [int]$GenerateSteps = 24,
    [int]$DetectorEpochs = 200,
    [int]$Workers = 8,
    [int]$Batch = 16,
    [int]$ImgSize = 640,
    [string]$Device = "0",
    [string]$Models = "yolov5s,yolov8n,yolov10n,yolo11n"
)

$ErrorActionPreference = "Continue"

$Root = "D:\drft-v2"
$Pipeline = Join-Path $Root "generation\neu-det-pipeline"
$LogsDir = Join-Path $Root "experiments\logs"
$ResultsDir = Join-Path $Root "experiments\results\four_dataset_lora_native"
$StatusPath = Join-Path $ResultsDir "status.json"
$LoraRoot = Join-Path $Pipeline "outputs\formal_dataset_specific_drft_lora"
$GenRoot = Join-Path $Pipeline "outputs\formal_dataset_specific_drft_v2_native"
$NeuRawTrain = Join-Path $Pipeline "data\raw\NEU-DET_trainonly"
$NeuRawFull = Join-Path $Pipeline "data\raw\NEU-DET"
$SplitManifest = Join-Path $Pipeline "data\processed\split_manifest.json"
$NeuLoraDir = Join-Path $LoraRoot "neu"
$NeuLoraPath = Join-Path $NeuLoraDir "drft_lora.safetensors"
$NeuOutput = Join-Path $GenRoot "neu"
$NeuGuidance = Join-Path $GenRoot "neu_guidance_empty"
$NeuCaptions = Join-Path $NeuOutput "captions_train.json"
$DatasetTemplate = "{dataset}\formal_lora_native\{dataset}_drft_lora_native_ratio_{ratio}_dataset"
$Project = Join-Path $Root "runs\four_dataset_lora_native_yolo_family_${DetectorEpochs}ep"
$SummaryCsv = Join-Path $ResultsDir "yolo_family_seed${Seed}_summary.csv"

New-Item -ItemType Directory -Force -Path $LogsDir, $ResultsDir, $LoraRoot, $GenRoot, $NeuLoraDir, $NeuOutput, $NeuGuidance | Out-Null
$env:PYTHONPATH = Join-Path $Pipeline "src"

function Write-Status {
    param(
        [string]$Stage,
        [string]$Status,
        [string]$Message,
        [int]$ExitCode = 0,
        [string]$RunDir = "",
        [string]$ImagesDir = ""
    )
    $payload = [ordered]@{
        timestamp = (Get-Date).ToString("s")
        stage = $Stage
        status = $Status
        message = $Message
        exit_code = $ExitCode
        seed = $Seed
        lora_steps = $LoraSteps
        generate_candidates = $GenerateCandidates
        generate_steps = $GenerateSteps
        detector_epochs = $DetectorEpochs
        models = $Models
        datasets = "neu,gc10,mt,tilda"
        dataset_template = $DatasetTemplate
        results_dir = $ResultsDir
        project = $Project
        run_dir = $RunDir
        images_dir = $ImagesDir
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Stage,
        [string]$Message,
        [string]$Stdout,
        [string]$Stderr,
        [scriptblock]$Block,
        [string]$RunDir = "",
        [string]$ImagesDir = ""
    )
    Write-Status -Stage $Stage -Status "RUNNING" -Message $Message -RunDir $RunDir -ImagesDir $ImagesDir
    & $Block > $Stdout 2> $Stderr
    $code = $LASTEXITCODE
    if ($null -eq $code) {
        $code = 0
    }
    if ($code -ne 0) {
        Write-Status -Stage $Stage -Status "FAILED" -Message "$Message failed" -ExitCode $code -RunDir $RunDir -ImagesDir $ImagesDir
        throw "$Stage failed with exit code $code"
    }
    Write-Status -Stage $Stage -Status "OK" -Message $Message -RunDir $RunDir -ImagesDir $ImagesDir
}

function Get-LatestRun {
    param([string]$OutputRoot)
    $run = Get-ChildItem -Path $OutputRoot -Directory -Filter "run_drft-v2_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $run) {
        return $null
    }
    $images = Join-Path $run.FullName "images"
    if (-not (Test-Path $images)) {
        return $null
    }
    return @{ Run = $run.FullName; Images = $images }
}

function Count-Images {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    return (Get-ChildItem -Path $Path -File | Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff") } | Measure-Object).Count
}

Push-Location $Root
try {
    if (-not (Test-Path $NeuLoraPath)) {
        Push-Location $Pipeline
        try {
            Invoke-Logged -Stage "neu_train_lora" -Message "training NEU dataset-specific DRFT-LoRA" `
                -Stdout (Join-Path $LogsDir "four_dataset_neu_train_lora.out.log") `
                -Stderr (Join-Path $LogsDir "four_dataset_neu_train_lora.err.log") `
                -Block {
                    python -m neu_det_pipeline.cli train-drft `
                        $NeuRawTrain `
                        --output-dir $NeuLoraDir `
                        --split-manifest $SplitManifest `
                        --train-split train `
                        --steps $LoraSteps `
                        --batch-size 1 `
                        --grad-accum 1 `
                        --learning-rate 2e-5 `
                        --rank 4 `
                        --alpha 8 `
                        --num-experts 3
                }
        }
        finally {
            Pop-Location
        }
    }

    Invoke-Logged -Stage "neu_audit_lora" -Message "auditing NEU LoRA class coverage" `
        -Stdout (Join-Path $LogsDir "four_dataset_neu_audit_lora.out.log") `
        -Stderr (Join-Path $LogsDir "four_dataset_neu_audit_lora.err.log") `
        -Block {
            python experiments\audit_dataset_specific_lora.py `
                --report (Join-Path $ResultsDir "audit_neu_lora.json") `
                lora `
                --raw-root $NeuRawTrain `
                --lora-dir $NeuLoraDir
        }

    $latest = Get-LatestRun -OutputRoot $NeuOutput
    $expectedTrain = (Get-Content -LiteralPath $SplitManifest -Raw | ConvertFrom-Json).train.Count
    $haveGenerated = if ($null -eq $latest) { 0 } else { Count-Images -Path $latest.Images }
    if ($haveGenerated -lt $expectedTrain) {
        Push-Location $Pipeline
        try {
            Invoke-Logged -Stage "neu_generate" -Message "generating NEU DRFT-v2 native images" `
                -Stdout (Join-Path $LogsDir "four_dataset_neu_generate.out.log") `
                -Stderr (Join-Path $LogsDir "four_dataset_neu_generate.err.log") `
                -Block {
                    python -m neu_det_pipeline.cli generate `
                        $NeuRawTrain `
                        $NeuGuidance `
                        $NeuLoraPath `
                        --output-dir $NeuOutput `
                        --caption-file $NeuCaptions `
                        --mode drft-v2 `
                        --generation-split train `
                        --split-manifest $SplitManifest `
                        --drft-candidates $GenerateCandidates `
                        --drft-steps $GenerateSteps `
                        --seed $Seed `
                        --no-mixed-dataset
                }
        }
        finally {
            Pop-Location
        }
        $latest = Get-LatestRun -OutputRoot $NeuOutput
    }

    if ($null -eq $latest) {
        throw "No NEU generation run found under $NeuOutput"
    }
    $scores = Join-Path $latest.Run "artifacts\drft_v2_lora\candidate_scores.jsonl"

    Invoke-Logged -Stage "neu_audit_generation" -Message "auditing NEU generation" `
        -Stdout (Join-Path $LogsDir "four_dataset_neu_audit_generation.out.log") `
        -Stderr (Join-Path $LogsDir "four_dataset_neu_audit_generation.err.log") `
        -RunDir $latest.Run -ImagesDir $latest.Images `
        -Block {
            python experiments\audit_dataset_specific_lora.py `
                --report (Join-Path $ResultsDir "audit_neu_generation.json") `
                generation `
                --raw-root $NeuRawTrain `
                --images-dir $latest.Images `
                --scores-path $scores
        }

    Invoke-Logged -Stage "neu_build_datasets" -Message "building NEU real/mixed formal datasets" `
        -Stdout (Join-Path $LogsDir "four_dataset_neu_build_datasets.out.log") `
        -Stderr (Join-Path $LogsDir "four_dataset_neu_build_datasets.err.log") `
        -RunDir $latest.Run -ImagesDir $latest.Images `
        -Block {
            python experiments\build_neu_lora_native_datasets.py `
                --orig-images-dir (Join-Path $NeuRawFull "IMAGES") `
                --orig-labels-dir (Join-Path $Root "data\NEU\split\neu_det_split\labels") `
                --split-manifest $SplitManifest `
                --generated-images-dir $latest.Images `
                --scores-path $scores `
                --seed $Seed `
                --summary (Join-Path $ResultsDir "neu_dataset_summary.json")
        }

    Invoke-Logged -Stage "yolo_family_seed0" -Message "running four-dataset YOLO-family real-vs-mixed validation" `
        -Stdout (Join-Path $LogsDir "four_dataset_yolo_family_seed${Seed}.out.log") `
        -Stderr (Join-Path $LogsDir "four_dataset_yolo_family_seed${Seed}.err.log") `
        -Block {
            python experiments\run_ratio_and_detector_experiments.py `
                --suite models `
                --datasets neu,gc10,mt,tilda `
                --models $Models `
                --dataset-dir-template $DatasetTemplate `
                --epochs $DetectorEpochs `
                --seed $Seed `
                --workers $Workers `
                --imgsz $ImgSize `
                --device $Device `
                --batch $Batch `
                --exist-ok `
                --results-dir $ResultsDir `
                --project $Project `
                --summary-csv $SummaryCsv
        }

    Write-Status -Stage "complete" -Status "OK" -Message "four-dataset DRFT-LoRA native experiment queue finished"
}
catch {
    Write-Status -Stage "failed" -Status "FAILED" -Message $_.Exception.Message -ExitCode 1
    throw
}
finally {
    Pop-Location
}
