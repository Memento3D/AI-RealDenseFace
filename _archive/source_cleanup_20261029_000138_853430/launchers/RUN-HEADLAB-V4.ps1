param(
    [string]$Workspace = "",
    [string]$ParserModel = "",
    [int]$Repeats = 0,
    [double[]]$HeightsMm = @(),
    [switch]$InstallDependencies
)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "scripts\run_headlab_v4.py"
$profilePath = Join-Path $project "configs\headlab04.v4.json"
$manifestPath = Join-Path $project "headlab-v4-code.json"
foreach ($file in @($python, $script, $profilePath, $manifestPath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Incomplete HeadLab update: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "HeadLab code differs from update ZIP: $file"
    }
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Workspace)) { $Workspace = Join-Path $project $profile.workspace }
if ([string]::IsNullOrWhiteSpace($ParserModel)) { $ParserModel = $profile.parser_model }
foreach ($file in @($Workspace, $ParserModel)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
}
if ($InstallDependencies) {
    & $python -m pip install -r (Join-Path $project "requirements-headlab-cpu.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $project "runs\headlab_v4_$stamp"
$arguments = @("-u", $script, "--profile", $profilePath, "--workspace", $Workspace, "--parser-model", $ParserModel, "--output", $output)
if ($Repeats -gt 0) { $arguments += @("--repeats", [string]$Repeats) }
if ($HeightsMm.Count -gt 0) {
    $arguments += "--heights-mm"
    foreach ($height in $HeightsMm) { $arguments += $height.ToString([System.Globalization.CultureInfo]::InvariantCulture) }
}
& $python @arguments
$exitCode = $LASTEXITCODE
$review = Join-Path $output "compare.html"
if (Test-Path -LiteralPath $review -PathType Leaf) { Start-Process $review }
if ($exitCode -eq 2) { throw "Some candidates failed. The comparison page lists each failure; other results were retained. Output: $output" }
if ($exitCode -ne 0) { throw "HeadLab failed. Inspect the error above. Output: $output" }
Write-Host "Benchmark complete: $output" -ForegroundColor Green
Write-Host "Research candidates only; manufacturing readiness is checked separately."
