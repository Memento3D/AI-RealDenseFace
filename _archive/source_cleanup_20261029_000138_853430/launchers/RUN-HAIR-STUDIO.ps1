param(
    [string]$RunDir = "",
    [int]$Port = 8765,
    [switch]$InstallDependencies
)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$script = Join-Path $project "scripts\hair_studio.py"
foreach ($file in @($python,$script,(Join-Path $project "assets\hair-studio\index.html"))) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
& $python -c "import numpy, scipy, PIL, trimesh, manifold3d, skimage, rtree, networkx, psutil; from manifold3d import Mesh64"
if ($InstallDependencies -or $LASTEXITCODE -ne 0) {
    & $python -m pip install -r (Join-Path $project "requirements-pipeline-v2.txt")
    if ($LASTEXITCODE -ne 0) { throw "CPU dependency installation failed." }
}
$manifestPath = Join-Path $project "hair-studio-code.json"
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Missing Hair Studio release manifest: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files) {
    $file = Join-Path $project $entry.path
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Incomplete Hair Studio code: $file" }
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256) { throw "Hair Studio code differs from release: $file" }
}
$arguments = @("-u",$script,"--project",$project,"--port",$Port.ToString(),"--open-browser")
if (![string]::IsNullOrWhiteSpace($RunDir)) { $arguments += @("--run",$RunDir) }
Write-Host "Opening Hair Studio. Keep this PowerShell window open; Ctrl+C stops it."
& $python @arguments
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 130) { throw "Hair Studio stopped with an error; inspect the message above." }
