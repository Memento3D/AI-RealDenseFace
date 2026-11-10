param(
    [string]$Workspace = "",
    [string]$ParserModel = "",
    [string]$ReferenceRun = "",
    [switch]$InstallDependencies
)
$ErrorActionPreference = "Stop"
$project = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "scripts\run_pipeline_v23.py"
$profilePath = Join-Path $project "configs\pipeline-v23-04.json"
$manifestPath = Join-Path $project "configs/manifests/pipeline-v23-code.json"
foreach ($file in @($python,$script,$profilePath,$manifestPath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Incomplete V2.3 code: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "V2.3 code differs from release: $file" }
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
    if ($LASTEXITCODE -ne 0) { throw "V2.3 dependency installation failed." }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $project "runs\pipeline_v23_$stamp"
$arguments = @("-u",$script,"--profile",$profilePath,"--workspace",$Workspace,"--parser-model",$ParserModel,"--output",$output)
if (![string]::IsNullOrWhiteSpace($ReferenceRun)) {
    if (!(Test-Path -LiteralPath (Join-Path $ReferenceRun "report.json") -PathType Leaf)) { throw "Missing reference report: $ReferenceRun" }
    $arguments += @("--reference-run",$ReferenceRun)
}
& $python @arguments
$exitCode = $LASTEXITCODE
$review = Join-Path $output "review.html"
if (Test-Path -LiteralPath $review -PathType Leaf) { Start-Process $review }
if ($exitCode -ne 0) { throw "V2.3 retained diagnostics. Inspect: $output" }
Write-Host "V2.3 exported: $output" -ForegroundColor Green
Write-Host "Geometry demo only. Machine slicing, resin calibration and a physical test remain unverified."
