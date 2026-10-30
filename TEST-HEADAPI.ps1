param(
 [string]$InputDir=".\input\02",
 [ValidateSet("mvp","components","raw")][string]$Mode="mvp",
 [ValidateSet("front","front-left","front-right","all")][string]$Views="all",
 [ValidateSet(128,192,256)][int]$Resolution=128,
 [int]$Port=8090,
 [switch]$Wait
)
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
 $arguments = @("-m","tools.headapi.client","--input",(Resolve-Path -LiteralPath $InputDir).Path,"--mode",$Mode,"--views",$Views,"--resolution",[string]$Resolution,"--url","http://127.0.0.1:$Port")
 if ($Wait) { $arguments += "--wait" }
 & (Join-Path $PSScriptRoot ".venv-headapi\Scripts\python.exe") @arguments
 if ($LASTEXITCODE -ne 0) { throw "API request or job failed; inspect the reported job" }
} finally { Pop-Location }
