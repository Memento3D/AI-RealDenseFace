param(
 [Parameter(Mandatory=$true)][string]$ProbeRun,
 [string]$CompareResolutions = "",
 [string]$CacheRoot = "D:\AI-cache\triposr-probe"
)
$ErrorActionPreference = "Stop"
$project = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path $project ".venv-triposr\Scripts\python.exe"
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing .venv-triposr; run scripts/setup/SETUP-TRIPOSR-PROBE.ps1" }
$resolutions = @()
if ($CompareResolutions) {
 foreach ($token in $CompareResolutions.Split(",")) {
  if ($token.Trim() -notin @("128","192","256")) { throw "Supported resolutions: 128,192,256" }
  $resolutions += [int]$token.Trim()
 }
}
$names = @("HF_HOME","HF_HUB_CACHE","HUGGINGFACE_HUB_CACHE","TRANSFORMERS_CACHE","TORCH_HOME","U2NET_HOME","TEMP","TMP")
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,"Process") }
Push-Location -LiteralPath $project
try {
 $source = (Resolve-Path -LiteralPath $ProbeRun).Path
 foreach ($folder in @("hf","hub","transformers","torch","rembg","temp")) { New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot $folder) | Out-Null }
 $env:HF_HOME = Join-Path $CacheRoot "hf"
 $env:HF_HUB_CACHE = Join-Path $CacheRoot "hub"
 $env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
 $env:TRANSFORMERS_CACHE = Join-Path $CacheRoot "transformers"
 $env:TORCH_HOME = Join-Path $CacheRoot "torch"
 $env:U2NET_HOME = Join-Path $CacheRoot "rembg"
 $env:TEMP = Join-Path $CacheRoot "temp"
 $env:TMP = $env:TEMP
 $arguments = @("-m","tools.triposr_probe.review","--run",$source)
 if ($resolutions.Count -gt 0) {
  $arguments += "--compare"
  foreach ($resolution in $resolutions) { $arguments += [string]$resolution }
 }
 & $python @arguments
 if ($LASTEXITCODE -ne 0) { throw "Review/comparison stopped or partially exported; inspect the new runs/triposr_review_* report" }
} finally {
 foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],"Process") }
 Pop-Location
}
