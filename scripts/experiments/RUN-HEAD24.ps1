param([string]$InputDir = "", [string]$LandmarkerModel = "D:\AI-cache\mediapipe\face_landmarker.task", [switch]$CheckOnly, [string]$ResumeRun = "")
$ErrorActionPreference = "Stop"
$project = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$python = Join-Path $project ".venv\Scripts\python.exe"
if (!$InputDir) { $InputDir = Join-Path $project "input\05" }
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
$arguments = @("-u", (Join-Path $project "scripts\run_head24.py"), "--input", $InputDir, "--task", $LandmarkerModel)
if ($CheckOnly) { $arguments += "--check-only" }
if ($ResumeRun) { $arguments += @("--resume", $ResumeRun) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Head24 stopped. Read pipeline.report.json and the stage log in the new runs/head24_* directory." }
