param(
 [string]$InputDir = ".\input\02",
 [string]$Image = "",
 [string]$CacheRoot = "D:\AI-cache\triposr-probe",
 [int]$Resolution = 128,
 [int]$ChunkSize = 2048,
 [switch]$WholeImage
)
$ErrorActionPreference = "Stop"
$project = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$basePython = Join-Path $project ".venv\Scripts\python.exe"
$probePython = Join-Path $project ".venv-triposr\Scripts\python.exe"
if (!(Test-Path -LiteralPath $probePython)) { throw "Run scripts/setup/SETUP-TRIPOSR-PROBE.ps1 first" }
$names = @("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "U2NET_HOME", "TEMP", "TMP")
$previous = @{}
foreach ($name in $names) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,"Process") }
Push-Location -LiteralPath $project
try {
 if (!$Image) {
  $found = @(Get-ChildItem -LiteralPath $InputDir -File | Where-Object { $_.BaseName -ieq "front" -and $_.Extension.ToLowerInvariant() -in @(".jpg",".jpeg",".png",".webp") })
  if ($found.Count -ne 1) { throw "Need exactly one front image, or pass -Image with an explicit source path" }
  $Image = $found[0].FullName
 }
 $source = (Resolve-Path -LiteralPath $Image).Path
 foreach ($folder in @("hf","hub","transformers","torch","rembg","temp")) { New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot $folder) | Out-Null }
 $env:HF_HOME = Join-Path $CacheRoot "hf"
 $env:HF_HUB_CACHE = Join-Path $CacheRoot "hub"
 $env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
 $env:TRANSFORMERS_CACHE = Join-Path $CacheRoot "transformers"
 $env:TORCH_HOME = Join-Path $CacheRoot "torch"
 $env:U2NET_HOME = Join-Path $CacheRoot "rembg"
 $env:TEMP = Join-Path $CacheRoot "temp"
 $env:TMP = $env:TEMP
 $run = Join-Path $project ("runs\triposr_probe_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff") + "_" + [guid]::NewGuid().ToString("N").Substring(0,6))
 $prepareArgs = @("-m","tools.triposr_probe.prepare","--source",$source,"--task","D:\AI-cache\mediapipe\face_landmarker.task","--output",(Join-Path $run "observation"))
 if ($WholeImage) { $prepareArgs += "--whole-image" }
 & $basePython @prepareArgs
 if ($LASTEXITCODE -ne 0) { throw "Head observation preparation failed; no neural inference run" }
 & $probePython -m tools.triposr_probe.worker --image (Join-Path $run "observation\head_crop.png") --output (Join-Path $run "raw") --resolution $Resolution --chunk-size $ChunkSize
 if ($LASTEXITCODE -ne 0) { throw "Probe failed. Read raw\report.json and traceback.txt under $run" }
 Write-Host "Review: $(Join-Path $run 'raw\review.html')"
 Write-Host "Raw generated head only. Hair isolation, identity and physical printing remain unverified."
} finally {
 foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],"Process") }
 Pop-Location
}
