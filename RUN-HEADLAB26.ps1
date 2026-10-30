param(
 [string]$InputRoot = "", [string]$Datasets = "", [string]$InputDir = "",
 [string]$Backends = "", [string]$Profile = "", [string]$OutputDir = "",
 [string]$ResumeRun = "", [switch]$CheckOnly, [switch]$FaceOnly, [switch]$AllVariants
)
$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
if (!$InputRoot) { $InputRoot = Join-Path $project "input" }
$arguments = @("-u", (Join-Path $project "tools\headlab26\cli.py"), "--input-root", $InputRoot)
if ($Datasets) { $arguments += @("--datasets", $Datasets) }
if ($InputDir) { $arguments += @("--input-dir", $InputDir) }
if ($Backends) { $arguments += @("--backends", $Backends) }
if ($Profile) { $arguments += @("--profile", $Profile) }
if ($OutputDir) { $arguments += @("--output", $OutputDir) }
if ($ResumeRun) { $arguments += @("--resume", $ResumeRun) }
if ($CheckOnly) { $arguments += "--check-only" }
if ($FaceOnly) { $arguments += "--face-only" }
if ($AllVariants) { $arguments += "--all-variants" }
Push-Location -LiteralPath $project
try {
 & $python @arguments
 $code = $LASTEXITCODE
 if ($code -eq 2) { Write-Warning "Batch finished with warnings/rejections. Open index.html and batch.report.json in the new run." }
 elseif ($code -ne 0) { throw "HeadLab26 failed before completing the batch. Read the error above." }
} finally { Pop-Location }
