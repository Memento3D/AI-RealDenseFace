param([int]$Port=8090,[string]$Config="")
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv-headapi\Scripts\python.exe"
if (!$Config) { $Config = Join-Path $project "configs\headapi.json" }
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Run SETUP-HEADAPI.ps1 first" }
Push-Location -LiteralPath $project
try {
 Write-Host "Head API: http://127.0.0.1:$Port/ | Swagger: http://127.0.0.1:$Port/docs"
 & $python -m tools.headapi.api --config $Config --port $Port
 if ($LASTEXITCODE -ne 0) { throw "Head API stopped with an error" }
} finally { Pop-Location }
