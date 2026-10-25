param(
    [string]$PhotoSet = "03",
    [string]$FaceParserModel = "D:\AI-cache\face-parsing\resnet18.onnx"
)

$ErrorActionPreference = "Stop"

# Xác định thư mục project từ vị trí script
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputDir = Join-Path $projectRoot "input/$PhotoSet"
$repoDir = Join-Path $projectRoot "vendor/RealDenseFace"

$rdfPython = Join-Path $projectRoot ".venv-rdf/Scripts/python.exe"
$mainPython = Join-Path $projectRoot ".venv/Scripts/python.exe"

function Invoke-Python {
    param(
        [string]$Python,
        [string[]]$Arguments
    )

    & $Python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Lệnh Python thất bại, exit code: $LASTEXITCODE"
    }
}

function Find-SuccessfulRun {
    param(
        [string]$Parent,
        [string]$Pattern,
        [string]$ExpectedStatus
    )

    $folders = Get-ChildItem -LiteralPath $Parent -Directory -Filter $Pattern |
        Sort-Object LastWriteTime -Descending

    foreach ($folder in $folders) {
        $reportPath = Join-Path $folder.FullName "report.json"

        if (Test-Path -LiteralPath $reportPath) {
            $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 |
                ConvertFrom-Json

            if ($report.status -eq $ExpectedStatus) {
                return $folder.FullName
            }
        }
    }

    throw "Không tìm thấy output thành công: $Pattern / $ExpectedStatus"
}

# Kiểm tra môi trường trước khi chạy
$requiredPaths = @(
    $rdfPython,
    $mainPython,
    $inputDir,
    "$repoDir/fit_single_image.py",
    "$repoDir/configs/model/vitb.yaml",
    "$repoDir/weights/realdenseface/vitb.pth",
    "$repoDir/weights/flame/flame2023_Open.pkl",
    "$repoDir/weights/flame/flame_assets.npz",
    "$repoDir/weights/facebox/face_box.pth",
    "$projectRoot/scripts/fit_rdf_three_photos.py",
    "$projectRoot/scripts/preview_rdf_three_textures.py",
    "$projectRoot/scripts/optimize_head.py",
    $FaceParserModel
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Thiếu file hoặc thư mục: $path"
    }
}

foreach ($slot in @("front", "left", "right")) {
    $photos = @(
        Get-ChildItem -LiteralPath $inputDir -File |
            Where-Object {
                $_.BaseName -ieq $slot -and
                $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg", ".png")
            }
    )

    if ($photos.Count -ne 1) {
        throw "Cần đúng một ảnh $slot.jpg/.jpeg/.png trong $inputDir"
    }

    if ($slot -eq "front") {
        $frontPhoto = $photos[0].FullName
    }
}

$runId = "rdf_" + $PhotoSet + "_" +
    (Get-Date -Format "yyyyMMdd_HHmmss") + "_" +
    ([guid]::NewGuid().ToString("N").Substring(0, 6))

$runDir = Join-Path $projectRoot "runs/$runId"
New-Item -ItemType Directory -Path $runDir | Out-Null

Push-Location $projectRoot

try {
    Write-Host "`n[1/4] Dựng đầu từ ảnh chính diện..." -ForegroundColor Cyan

    # Chạy trong repo để các đường dẫn config mặc định được resolve đúng
    Push-Location $repoDir
    try {
        Invoke-Python -Python $rdfPython -Arguments @(
            "fit_single_image.py",
            "--input", $frontPhoto,
            "--output_npz", "$runDir/front.flame.npz",
            "--output_image", "$runDir/front.reconstruction.jpg",
            "--flame_model", "weights/flame/flame2023_Open.pkl",
            "--flame_assets", "weights/flame/flame_assets.npz",
            "--facebox_weights", "weights/facebox/face_box.pth",
            "--num_expressions", "100",
            "--device", "cuda",
            "--no_compile"
        )
    }
    finally {
        Pop-Location
    }

    Write-Host "`n[2/4] Fit chung ba ảnh..." -ForegroundColor Cyan

    Invoke-Python -Python $rdfPython -Arguments @(
        "$projectRoot/scripts/fit_rdf_three_photos.py",
        "--params", "$runDir/front.flame.npz",
        "--input-dir", $inputDir,
        "--repo", $repoDir
    )

    $headRun = Find-SuccessfulRun `
        -Parent $runDir `
        -Pattern "three_photo_*" `
        -ExpectedStatus "JOINT_CANDIDATE_EXPORTED"

    Write-Host "`n[3/4] Tạo texture từ ba ảnh..." -ForegroundColor Cyan

    Invoke-Python -Python $rdfPython -Arguments @(
        "$projectRoot/scripts/preview_rdf_three_textures.py",
        "--run-dir", $headRun,
        "--input-dir", $inputDir,
        "--repo", $repoDir
    )

    $textureRun = Find-SuccessfulRun `
        -Parent $headRun `
        -Pattern "three_textures_*" `
        -ExpectedStatus "THREE_TEXTURES_PREVIEW_EXPORTED"

    Write-Host "`n[4/4] Mask tự động và tối ưu màu..." -ForegroundColor Cyan

    Invoke-Python -Python $mainPython -Arguments @(
        "$projectRoot/scripts/optimize_head.py",
        "--head-run", $headRun,
        "--texture-run", $textureRun,
        "--auto-mask",
        "--face-parser-model", $FaceParserModel
    )

    $workspaceRun = Find-SuccessfulRun `
        -Parent $headRun `
        -Pattern "head_workspace_*" `
        -ExpectedStatus "TEXTURE_REVIEW_READY"

    $viewer = Join-Path $workspaceRun "head.workspace.html"

    if (-not (Test-Path -LiteralPath $viewer)) {
        throw "Không tìm thấy viewer: $viewer"
    }

    $summary = [ordered]@{
        status        = "TEXTURE_REVIEW_READY"
        input_dir     = $inputDir
        run_dir       = $runDir
        head_run      = $headRun
        texture_run   = $textureRun
        workspace     = $viewer
        report        = Join-Path $workspaceRun "report.json"
    }

    $summaryPath = Join-Path $runDir "test.summary.json"
    $summary | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $summaryPath -Encoding UTF8

    Write-Host "`nHOÀN TẤT" -ForegroundColor Green
    $summary | ConvertTo-Json -Depth 5
    Write-Host "`nSummary: $summaryPath"

    Start-Process -FilePath $viewer
}
finally {
    Pop-Location
}