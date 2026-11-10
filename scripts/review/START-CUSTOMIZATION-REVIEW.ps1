param([ValidateRange(1024,65535)][int]$Port = 8092)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) '.venv\Scripts\python.exe'
$reviewRoot = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'runs\customization-prep-20261105'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing AI .venv Python.' }
if (-not (Test-Path -LiteralPath (Join-Path $reviewRoot 'review-site\manifest.json'))) { throw 'Local review artifacts are missing.' }
Write-Host "Local 3D review: http://127.0.0.1:$Port/review-site/" -ForegroundColor Cyan
Write-Host 'Open this address in your browser. Press Ctrl+C to stop. This viewer is separate from the AI API.'
& $pythonPath -B -m http.server $Port --bind 127.0.0.1 --directory $reviewRoot
exit $LASTEXITCODE
