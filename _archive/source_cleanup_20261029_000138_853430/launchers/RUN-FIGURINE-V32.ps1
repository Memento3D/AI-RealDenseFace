param([string]$Workspace = "", [switch]$InstallDependencies)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$profilePath = Join-Path $project "configs\figurine04.v32.json"
$scriptPath = Join-Path $project "scripts\run_figurine_v32.py"
if (!(Test-Path -LiteralPath $profilePath -PathType Leaf)) { throw "Missing profile: $profilePath" }
$manifestPath = Join-Path $project "figurine-v32-code.json"
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Missing update manifest: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $path = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Incomplete V3.2 update. Missing: $path" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.sha256) {
        throw "V3.2 code differs from the update package: $path"
    }
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ([string]::IsNullOrWhiteSpace($Workspace)) { $Workspace = Join-Path $project $profile.workspace }
$model = $profile.parser_model
foreach ($file in @($python, $scriptPath, $Workspace, $model)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if ($InstallDependencies) {
    & $python -m pip install -r (Join-Path $project "requirements-figurine-cpu.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output = Join-Path $project "runs\figurine_v32_$stamp"
& $python -u $scriptPath --profile $profilePath --workspace $Workspace --parser-model $model --output $output
if ($LASTEXITCODE -ne 0) { throw "V3.2 failed; inspect the error above." }
Write-Host "Output: $output" -ForegroundColor Green
Start-Process (Join-Path $output "review.html")
