param([switch]$Apply, [string]$Rollback = "")
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "maintenance\cleanup_project.py"
if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
$arguments = @($script, "--root", $PSScriptRoot)
if ($Apply -and $Rollback) { throw "Choose Apply or Rollback, not both." }
if ($Apply) { $arguments += "--apply" }
if ($Rollback) { $arguments += @("--rollback", $Rollback) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Cleanup failed. Read diagnostics above." }
