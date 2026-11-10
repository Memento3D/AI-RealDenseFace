$ErrorActionPreference = "Stop"
$project = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$base = Join-Path $project ".venv\Scripts\python.exe"
$api = Join-Path $project ".venv-headapi\Scripts\python.exe"
Push-Location -LiteralPath $project
try {
 $exclude = Join-Path $project ".git\info\exclude"
 if (Test-Path -LiteralPath $exclude -PathType Leaf) {
  $existing = [string](Get-Content -LiteralPath $exclude -Raw)
  if (!$existing.Contains("/.venv-headapi/")) { Add-Content -LiteralPath $exclude -Value "`n/.venv-headapi/" }
 }
 if (!(Test-Path -LiteralPath $base -PathType Leaf)) { throw "Missing existing CPU .venv" }
 if (!(Test-Path -LiteralPath $api -PathType Leaf)) {
  & $base -m venv (Join-Path $project ".venv-headapi")
  if ($LASTEXITCODE -ne 0) { throw "API environment creation failed" }
 }
 & $api -m pip install --no-cache-dir -r (Join-Path $project "requirements\headapi.txt")
 if ($LASTEXITCODE -ne 0) { throw "API dependency installation failed" }
 & $api -m pip check
 if ($LASTEXITCODE -ne 0) { throw "API environment has conflicting dependencies" }
 Write-Host "Ready: powershell -ExecutionPolicy Bypass -File .\scripts/api/RUN-HEADAPI.ps1"
} finally { Pop-Location }
