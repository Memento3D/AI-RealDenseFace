param(
    [string]$Workspace = "",
    [string]$ParserModel = "",
    [switch]$InstallDependencies
)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "scripts\run_pipeline_v21.py"
$profilePath = Join-Path $project "configs\pipeline-v21-04.json"
$manifestPath = Join-Path $project "pipeline-v21-code.json"
foreach ($file in @($python,$script,$profilePath,$manifestPath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Incomplete V2.1 code: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "V2.1 code differs from release: $file" }
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Workspace)) { $Workspace = Join-Path $project $profile.workspace }
if ([string]::IsNullOrWhiteSpace($ParserModel)) { $ParserModel = $profile.parser_model }
foreach ($file in @($Workspace,$ParserModel)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
}
& $python -c "import numpy, scipy, PIL, trimesh, manifold3d, onnxruntime, skimage, rtree, networkx; assert hasattr(manifold3d, 'Mesh64'), 'manifold3d 3.1.1 required'"
$missingDependencies = $LASTEXITCODE -ne 0
if ($InstallDependencies -or $missingDependencies) {
    & $python -m pip install -r (Join-Path $project "requirements-pipeline-v2.txt")
    if ($LASTEXITCODE -ne 0) { throw "V2.1 dependency installation failed." }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $project "runs\pipeline_v21_$stamp"
& $python -u $script --profile $profilePath --workspace $Workspace --parser-model $ParserModel --output $output
$exitCode = $LASTEXITCODE
$review = Join-Path $output "review.html"
if (Test-Path -LiteralPath $review -PathType Leaf) { Start-Process $review }
if ($exitCode -ne 0) { throw "V2.1 retained diagnostics. Inspect: $output" }
Write-Host "V2.1 exported: $output" -ForegroundColor Green
Write-Host "Geometry demo only. Machine slicing, resin calibration and a physical test remain unverified."
