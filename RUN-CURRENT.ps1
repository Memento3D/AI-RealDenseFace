param([string]$Workspace = "", [string]$ParserModel = "", [string]$ReferenceRun = "")
$ErrorActionPreference = "Stop"
if (!$Workspace) { $Workspace = Join-Path $PSScriptRoot "hair-bench\input04\head.workspace.html" }
if (!(Test-Path -LiteralPath $Workspace -PathType Leaf)) { throw "Missing workspace: $Workspace" }
$arguments = @("-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "RUN-PIPELINE-V23.ps1"), "-Workspace", $Workspace)
if ($ParserModel) { $arguments += @("-ParserModel", $ParserModel) }
if ($ReferenceRun) { $arguments += @("-ReferenceRun", $ReferenceRun) }
& powershell @arguments
if ($LASTEXITCODE -ne 0) { throw "V2.3 failed. Read diagnostics above." }
