param([Parameter(Mandatory=$true)][string]$JobId,[int]$Port=8090,[switch]$Wait)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
 $arguments = @("-m","tools.headapi.client","--assemble-job",$JobId,"--url","http://127.0.0.1:$Port")
 if ($Wait) { $arguments += "--wait" }
 & (Join-Path $PSScriptRoot ".venv-headapi\Scripts\python.exe") @arguments
 if ($LASTEXITCODE -ne 0) { throw "Assembly request/job failed; inspect reported job" }
} finally { Pop-Location }
