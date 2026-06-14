param(
    [int]$Seed = 0,
    [int]$LoraSteps = 1000,
    [int]$GenerateCandidates = 5,
    [int]$GenerateSteps = 24,
    [int]$YoloEpochs = 200,
    [int]$Workers = 8,
    [int]$Batch = 16,
    [int]$ImgSize = 640,
    [string]$Device = "0"
)

$ErrorActionPreference = "Continue"

$Root = "D:\drft-v2"
$Pipeline = Join-Path $Root "generation\neu-det-pipeline"
$LogsDir = Join-Path $Root "experiments\logs"
$ResultsDir = Join-Path $Root "experiments\results\dataset_specific_lora_yolo_family"
$StatusPath = Join-Path $Root "experiments\results\dataset_specific_lora_yolo_family_status.json"
$LoraRoot = Join-Path $Pipeline "outputs\formal_dataset_specific_drft_lora"
$GenRoot = Join-Path $Pipeline "outputs\formal_dataset_specific_drft_v2_native"
$DatasetTemplate = "{dataset}_drft_lora_native_ratio_{ratio}_dataset"
$Project = Join-Path $Root "runs\dataset_specific_lora_yolo_family_${YoloEpochs}ep"
$SummaryCsv = Join-Path $ResultsDir "summary.csv"
$DatasetSummary = Join-Path $ResultsDir "dataset_summary.json"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LoraRoot | Out-Null
New-Item -ItemType Directory -Force -Path $GenRoot | Out-Null

$env:PYTHONPATH = Join-Path $Pipeline "src"

$datasets = @(
    @{
        Key = "gc10"
        Raw = "data\raw\GC10-DET_trainonly"
    },
    @{
        Key = "mt"
        Raw = "data\raw\MT_trainonly"
    },
    @{
        Key = "tilda"
        Raw = "data\raw\TILDA_trainonly"
    }
)

function Write-Status {
    param(
        [string]$Dataset,
        [string]$Stage,
        [string]$Message,
        [int]$ExitCode = 0,
        [string]$RunDir = "",
        [string]$ImagesDir = "",
        [string]$LoraDir = ""
    )
    $payload = [ordered]@{
        timestamp = (Get-Date).ToString("s")
        dataset = $Dataset
        stage = $Stage
        message = $Message
        exit_code = $ExitCode
        seed = $Seed
        lora_steps = $LoraSteps
        generate_candidates = $GenerateCandidates
        generate_steps = $GenerateSteps
        yolo_epochs = $YoloEpochs
        workers = $Workers
        batch = $Batch
        imgsz = $ImgSize
        device = $Device
        dataset_template = $DatasetTemplate
        results_dir = $ResultsDir
        project = $Project
        run_dir = $RunDir
        images_dir = $ImagesDir
        lora_dir = $LoraDir
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Dataset,
        [string]$Stage,
        [string]$Message,
        [string]$Stdout,
        [string]$Stderr,
        [scriptblock]$Block,
        [string]$RunDir = "",
        [string]$ImagesDir = "",
        [string]$LoraDir = ""
    )
    Write-Status -Dataset $Dataset -Stage $Stage -Message $Message -RunDir $RunDir -ImagesDir $ImagesDir -LoraDir $LoraDir
    & $Block > $Stdout 2> $Stderr
    $code = $LASTEXITCODE
    if ($null -eq $code) {
        $code = 0
    }
    if ($code -ne 0) {
        Write-Status -Dataset $Dataset -Stage "${Stage}_failed" -Message "$Message failed" -ExitCode $code -RunDir $RunDir -ImagesDir $ImagesDir -LoraDir $LoraDir
        throw "$Dataset $Stage failed with exit code $code"
    }
}

function Ensure-TrainOnlyManifest {
    param([string]$RawRoot)
    $manifest = Join-Path $RawRoot "split_manifest.json"
    if (-not (Test-Path $manifest)) {
        $annotationDir = Join-Path $RawRoot "ANNOTATIONS"
        $stems = Get-ChildItem $annotationDir -Filter "*.xml" | Sort-Object Name | ForEach-Object { $_.BaseName }
        $payload = [ordered]@{
            train = @($stems)
            val = @()
            test = @()
        }
        $json = $payload | ConvertTo-Json -Depth 4
        [System.IO.File]::WriteAllText($manifest, $json, [System.Text.UTF8Encoding]::new($false))
    }
    return $manifest
}

function Get-LatestRunImages {
    param([string]$OutputRoot)
    $run = Get-ChildItem -Path $OutputRoot -Directory -Filter "run_drft-v2_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $run) {
        throw "No run_drft-v2_* directory under $OutputRoot"
    }
    $images = Join-Path $run.FullName "images"
    if (-not (Test-Path $images)) {
        throw "Missing images dir: $images"
    }
    return @{ Run = $run.FullName; Images = $images }
}

Push-Location $Root
try {
    foreach ($d in $datasets) {
        $key = $d.Key
        $raw = Join-Path $Pipeline $d.Raw
        $splitManifest = Ensure-TrainOnlyManifest -RawRoot $raw
        $loraDir = Join-Path $LoraRoot $key
        $loraPath = Join-Path $loraDir "drft_lora.safetensors"
        $guidance = Join-Path $GenRoot "${key}_guidance_empty"
        $fullOut = Join-Path $GenRoot $key
        $captionFile = Join-Path $fullOut "captions_train.json"
        New-Item -ItemType Directory -Force -Path $guidance | Out-Null
        New-Item -ItemType Directory -Force -Path $fullOut | Out-Null

        if (-not (Test-Path $loraPath)) {
            Push-Location $Pipeline
            try {
                Invoke-Logged -Dataset $key -Stage "train_lora" -Message "training dataset-specific DRFT-LoRA" `
                    -Stdout (Join-Path $LogsDir "formal_${key}_train_drft_lora.out.log") `
                    -Stderr (Join-Path $LogsDir "formal_${key}_train_drft_lora.err.log") `
                    -LoraDir $loraDir `
                    -Block {
                        python -m neu_det_pipeline.cli train-drft `
                            $raw `
                            --output-dir $loraDir `
                            --split-manifest $splitManifest `
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

        Invoke-Logged -Dataset $key -Stage "audit_lora" -Message "auditing LoRA class_to_id coverage" `
            -Stdout (Join-Path $LogsDir "formal_${key}_audit_lora.out.log") `
            -Stderr (Join-Path $LogsDir "formal_${key}_audit_lora.err.log") `
            -LoraDir $loraDir `
            -Block {
                python experiments\audit_dataset_specific_lora.py `
                    --report (Join-Path $ResultsDir "audit_${key}_lora.json") `
                    lora `
                    --raw-root $raw `
                    --lora-dir $loraDir
            }

        Push-Location $Pipeline
        try {
            Invoke-Logged -Dataset $key -Stage "full_generate" -Message "running full native-resolution generation" `
                -Stdout (Join-Path $LogsDir "formal_${key}_full_generate.out.log") `
                -Stderr (Join-Path $LogsDir "formal_${key}_full_generate.err.log") `
                -LoraDir $loraDir `
                -Block {
                    python -m neu_det_pipeline.cli generate `
                        $raw `
                        $guidance `
                        $loraPath `
                        --output-dir $fullOut `
                        --caption-file $captionFile `
                        --mode drft-v2 `
                        --generation-split train `
                        --split-manifest $splitManifest `
                        --drft-candidates $GenerateCandidates `
                        --drft-steps $GenerateSteps `
                        --seed $Seed `
                        --no-mixed-dataset
                }
        }
        finally {
            Pop-Location
        }

        $fullRun = Get-LatestRunImages -OutputRoot $fullOut
        $fullScores = Join-Path $fullRun.Run "artifacts\drft_v2_lora\candidate_scores.jsonl"
        Invoke-Logged -Dataset $key -Stage "audit_full_generation" -Message "auditing full generation" `
            -Stdout (Join-Path $LogsDir "formal_${key}_audit_full_generation.out.log") `
            -Stderr (Join-Path $LogsDir "formal_${key}_audit_full_generation.err.log") `
            -RunDir $fullRun.Run -ImagesDir $fullRun.Images -LoraDir $loraDir `
            -Block {
                python experiments\audit_dataset_specific_lora.py `
                    --report (Join-Path $ResultsDir "audit_${key}_full_generation.json") `
                    generation `
                    --raw-root $raw `
                    --images-dir $fullRun.Images `
                    --scores-path $fullScores
            }

        Invoke-Logged -Dataset $key -Stage "build_dataset" -Message "building isolated real/mixed datasets" `
            -Stdout (Join-Path $LogsDir "formal_${key}_build_dataset.out.log") `
            -Stderr (Join-Path $LogsDir "formal_${key}_build_dataset.err.log") `
            -RunDir $fullRun.Run -ImagesDir $fullRun.Images -LoraDir $loraDir `
            -Block {
                python experiments\build_dataset_specific_lora_ratio_datasets.py `
                    --dataset $key `
                    --ratios "0,100" `
                    --generated-images-dir $fullRun.Images `
                    --out-template $DatasetTemplate `
                    --seed $Seed `
                    --summary $DatasetSummary
            }
    }

    Invoke-Logged -Dataset "all" -Stage "yolo_family" -Message "running YOLO family real-vs-mixed detector generalization" `
        -Stdout (Join-Path $LogsDir "formal_dataset_specific_lora_yolo_family.out.log") `
        -Stderr (Join-Path $LogsDir "formal_dataset_specific_lora_yolo_family.err.log") `
        -Block {
            python experiments\run_ratio_and_detector_experiments.py `
                --suite models `
                --datasets gc10,mt,tilda `
                --models yolov5s,yolov8n,yolov10n,yolo11n `
                --dataset-dir-template $DatasetTemplate `
                --epochs $YoloEpochs `
                --seed $Seed `
                --workers $Workers `
                --imgsz $ImgSize `
                --device $Device `
                --batch $Batch `
                --exist-ok `
                --results-dir $ResultsDir `
                --project $Project `
                --summary-csv $SummaryCsv `
                --stop-on-failure
        }

    Write-Status -Dataset "all" -Stage "complete" -Message "dataset-specific DRFT-LoRA + YOLO-family formal experiment finished"
}
finally {
    Pop-Location
}

