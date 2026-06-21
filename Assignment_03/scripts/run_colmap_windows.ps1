param(
    [string]$DataDir = "data",
    [string]$Colmap = "colmap",
    [string]$SparseModel = "",
    [switch]$Dense,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$database = Join-Path $DataDir "database.db"
$imageDir = Join-Path $DataDir "images"
$sparseDir = Join-Path $DataDir "colmap\sparse"
$denseDir = Join-Path $DataDir "colmap\dense"

if ($Force -and (Test-Path (Join-Path $DataDir "colmap"))) {
    Remove-Item -LiteralPath (Join-Path $DataDir "colmap") -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $sparseDir | Out-Null

function Invoke-Colmap {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & $Colmap @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "COLMAP command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-Colmap @(
    "feature_extractor",
    "--database_path", $database,
    "--image_path", $imageDir,
    "--ImageReader.single_camera", "1",
    "--FeatureExtraction.use_gpu", "1"
)

Invoke-Colmap @(
    "exhaustive_matcher",
    "--database_path", $database,
    "--FeatureMatching.use_gpu", "1"
)

Invoke-Colmap @(
    "mapper",
    "--database_path", $database,
    "--image_path", $imageDir,
    "--output_path", $sparseDir
)

if ($Dense) {
    if ($SparseModel -eq "") {
        $sparseModels = Get-ChildItem -LiteralPath $sparseDir -Directory | Sort-Object Name
        if ($sparseModels.Count -eq 0) {
            throw "No sparse model found in $sparseDir"
        }
        $selectedSparse = $sparseModels[-1].FullName
    } else {
        $selectedSparse = Join-Path $sparseDir $SparseModel
    }
    if (-not (Test-Path $selectedSparse)) {
        throw "Selected sparse model does not exist: $selectedSparse"
    }
    Write-Host "Dense input sparse model: $selectedSparse"

    New-Item -ItemType Directory -Force -Path $denseDir | Out-Null
    Invoke-Colmap @(
        "image_undistorter",
        "--image_path", $imageDir,
        "--input_path", $selectedSparse,
        "--output_path", $denseDir,
        "--output_type", "COLMAP"
    )
    Invoke-Colmap @(
        "patch_match_stereo",
        "--workspace_path", $denseDir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    )
    Invoke-Colmap @(
        "stereo_fusion",
        "--workspace_path", $denseDir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", (Join-Path $denseDir "fused.ply")
    )
}

Write-Host "COLMAP finished. Sparse output: $sparseDir"
