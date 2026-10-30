param(
    [string]$Workspace = "",
    [string]$ParserModel = "",
    [string[]]$HairIds = @(),
    [string]$Expression = "",
    [string]$SkinId = "",
    [double]$HeadMm = 22,
    [double]$Volume = 0,
    [double]$FringeLiftMm = 0,
    [double]$NapeLiftMm = 0,
    [double]$HairYawDeg = 0,
    [string]$Teeth = "auto",
    [switch]$Quick,
    [switch]$InstallDependencies
)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "scripts\run_pipeline_v2.py"
$profilePath = Join-Path $project "configs\pipeline-v2-04.json"
$manifestPath = Join-Path $project "pipeline-v2-code.json"
foreach ($file in @($python,$script,$profilePath,$manifestPath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Incomplete V2 code: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "V2 code differs from release: $file" }
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Workspace)) { $Workspace = Join-Path $project $profile.workspace }
if ([string]::IsNullOrWhiteSpace($ParserModel)) { $ParserModel = $profile.parser_model }
foreach ($file in @($Workspace,$ParserModel)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
}
# Use the existing main venv only; leave RDF/WSL environments alone.
& $python -c "import numpy, scipy, PIL, trimesh, manifold3d, onnxruntime, skimage, rtree, networkx"
$missingDependencies = $LASTEXITCODE -ne 0
if ($InstallDependencies -or $missingDependencies) {
    Write-Host "Installing/checking Pipeline V2 CPU dependencies in the main .venv..."
    & $python -m pip install -r (Join-Path $project "requirements-pipeline-v2.txt")
    if ($LASTEXITCODE -ne 0) { throw "Pipeline V2 dependency installation failed." }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $project "runs\pipeline_v2_$stamp"
$arguments = @("-u",$script,"--profile",$profilePath,"--workspace",$Workspace,"--parser-model",$ParserModel,"--output",$output,"--teeth",$Teeth)
if ($HairIds.Count -gt 0) { $arguments += "--hair-id"; $arguments += $HairIds }
if (![string]::IsNullOrWhiteSpace($Expression)) { $arguments += @("--expression",$Expression) }
if (![string]::IsNullOrWhiteSpace($SkinId)) { $arguments += @("--skin-id",$SkinId) }
if ($PSBoundParameters.ContainsKey("HeadMm")) { $arguments += @("--head-mm",$HeadMm.ToString([System.Globalization.CultureInfo]::InvariantCulture)) }
foreach ($pair in @(@("--volume",$Volume),@("--fringe-lift-mm",$FringeLiftMm),@("--nape-lift-mm",$NapeLiftMm),@("--hair-yaw-deg",$HairYawDeg))) {
    $arguments += @([string]$pair[0],([double]$pair[1]).ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($Quick) { $arguments += "--quick" }
& $python @arguments
$exitCode = $LASTEXITCODE
$review = Join-Path $output "review.html"
if (Test-Path -LiteralPath $review -PathType Leaf) { Start-Process $review }
if ($exitCode -ne 0) { throw "V2 retained results and diagnostics; inspect the error/report. Output: $output" }
Write-Host "Pipeline V2 exported: $output" -ForegroundColor Green
Write-Host "Geometry demo exported. Resin calibration, machine slicing and physical printing remain separate checks."
