param([string]$InputDir = "", [string]$LandmarkerModel = "D:\AI-cache\mediapipe\face_landmarker.task", [switch]$CheckOnly, [string]$ResumeRun = "", [string]$GeometryCache = "", [string]$Profile = "")
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
if (!$InputDir) { $InputDir = Join-Path $project "input\05" }
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
$arguments = @("-u", (Join-Path $project "scripts\run_head245.py"), "--input", $InputDir, "--task", $LandmarkerModel)
if ($CheckOnly) { $arguments += "--check-only" }
if ($ResumeRun) { $arguments += @("--resume", $ResumeRun) }
if ($GeometryCache) { $arguments += @("--geometry-cache", $GeometryCache) }
if ($Profile) { $arguments += @("--profile", $Profile) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Head24.5 stopped. Read pipeline.report.json and the stage log in the new runs/head245_* directory." }
