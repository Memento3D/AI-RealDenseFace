param(
 [string]$InputDir=".\input\02",
 [int]$Port=8090,
 [switch]$Wait
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
try {
 $envFile = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) ".env.local"
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
  throw "Set HEADAPI_API_KEY in .env.local or the current process environment before testing the API"
 }
 $arguments = @("-m","tools.headapi.client","--input",(Resolve-Path -LiteralPath $InputDir).Path,"--url","http://127.0.0.1:$Port")
 if ($Wait) { $arguments += "--wait" }
 & (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) ".venv-headapi\Scripts\python.exe") @arguments
 if ($LASTEXITCODE -ne 0) { throw "API request or job failed; inspect the reported job" }
} finally { Pop-Location }
