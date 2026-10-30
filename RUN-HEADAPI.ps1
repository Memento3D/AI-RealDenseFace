param([int]$Port=8090,[string]$Config="")
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv-headapi\Scripts\python.exe"
if (!$Config) { $Config = Join-Path $project "configs\headapi.json" }
$envFile = Join-Path $project ".env.local"
if (Test-Path -LiteralPath $envFile -PathType Leaf) {
 foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
  $entry = $line.Trim()
  if (!$entry -or $entry.StartsWith("#")) { continue }
  if ($entry -notmatch '^(HEADAPI_API_KEY|HEADAPI_ALLOWED_IMAGE_HOSTS)\s*=\s*(.*)$') {
   throw "Unsupported setting in .env.local; allowed keys are HEADAPI_API_KEY and HEADAPI_ALLOWED_IMAGE_HOSTS"
  }
  $name = $Matches[1]
  $value = $Matches[2].Trim()
  if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
   $value = $value.Substring(1, $value.Length - 2)
  }
  if (!$value) { throw "$name must not be empty in .env.local" }
  [Environment]::SetEnvironmentVariable($name, $value, "Process")
 }
}
if (!$env:HEADAPI_API_KEY) {
 throw "Set HEADAPI_API_KEY in .env.local or the current process environment before exposing the API"
}
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Run SETUP-HEADAPI.ps1 first" }
Push-Location -LiteralPath $project
try {
 Write-Host "Head API: http://127.0.0.1:$Port/ | Swagger: http://127.0.0.1:$Port/docs"
 & $python -m tools.headapi.api --config $Config --port $Port
 if ($LASTEXITCODE -ne 0) { throw "Head API stopped with an error" }
} finally { Pop-Location }
