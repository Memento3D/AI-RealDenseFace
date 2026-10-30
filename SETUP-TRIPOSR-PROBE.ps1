param([string]$CacheRoot = "D:\AI-cache\triposr-probe")
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$basePython = Join-Path $project ".venv\Scripts\python.exe"
$probePython = Join-Path $project ".venv-triposr\Scripts\python.exe"
if (!(Test-Path -LiteralPath $basePython)) { throw "Missing project Python: $basePython" }
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
Push-Location -LiteralPath $project
try {
 New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot "temp") | Out-Null
 $env:TEMP = Join-Path $CacheRoot "temp"
 $env:TMP = $env:TEMP
 & $basePython -c "import sys; assert sys.version_info[:2] == (3,11), 'This probe pins Windows Python3.11 wheels'"
 if ($LASTEXITCODE -ne 0) { throw "Python3.11 required" }
 $exclude = Join-Path $project ".git\info\exclude"
 if (Test-Path -LiteralPath $exclude) {
  $existing = [string](Get-Content -LiteralPath $exclude -Raw)
  foreach ($pattern in @("/.venv-triposr/", "/vendor/TripoSR-probe/", "/vendor/TripoSR-probe.download/", "/vendor/TripoSR-probe-source/")) {
   if (!$existing.Contains($pattern)) { Add-Content -LiteralPath $exclude -Value ("`n" + $pattern) }
  }
 }
 & $basePython -m tools.triposr_probe.bootstrap
 if ($LASTEXITCODE -ne 0) { throw "Pinned source preparation failed" }
 if (!(Test-Path -LiteralPath $probePython)) {
  & $basePython -m venv (Join-Path $project ".venv-triposr")
  if ($LASTEXITCODE -ne 0) { throw "Creating separate environment failed" }
 }
 & $probePython -m pip install --no-cache-dir "pip==24.3.1"
 if ($LASTEXITCODE -ne 0) { throw "pip setup failed" }
 & $probePython -m pip install --no-cache-dir "torch==2.2.2" "torchvision==0.17.2" --index-url https://download.pytorch.org/whl/cu121
 if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
 & $probePython -m pip install --no-cache-dir -r (Join-Path $project "requirements\triposr-probe.txt")
 if ($LASTEXITCODE -ne 0) { throw "Probe dependencies failed" }
 & $probePython -m pip check
 if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
 & $probePython -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA not available'"
 if ($LASTEXITCODE -ne 0) { throw "GPU check failed" }
 & $probePython -m pip freeze | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $project ".venv-triposr\probe.freeze.txt")
 Write-Host "Setup complete. No model inference has run. Use RUN-TRIPOSR-PROBE.ps1 next."
} finally { $env:TEMP = $oldTemp; $env:TMP = $oldTmp; Pop-Location }
