[CmdletBinding()]
param([string]$Distro='Ubuntu-22.04')
$ErrorActionPreference='Stop'
$project=$PSScriptRoot
$workspace=Join-Path $project 'hair-bench\input04\head.workspace.html'
$maskRun=Join-Path $project 'hair-bench\reference04'
$launcher=Join-Path $project 'RUN-HAIR-BENCH.ps1'
foreach ($file in @($workspace,$launcher,(Join-Path $maskRun 'report.json'),(Join-Path $maskRun 'checksums.json'))) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file. Extract the complete workspace-04 ZIP into the project root." }
}
$expected=(Get-Content -LiteralPath (Join-Path $maskRun 'report.json') -Raw | ConvertFrom-Json).face.workspace_sha256
$actual=(Get-FileHash -LiteralPath $workspace -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'Workspace 04 checksum differs from the bundled face/mask reference. No inference started.' }
Write-Host "Verified workspace 04: $workspace"
Write-Host 'Using existing completed TripoSR setup. Run mode only; no Setup/All or Hunyuan.'
$runArgs=@('-ExecutionPolicy','Bypass','-File',$launcher,'-Mode','Run','-Models','triposr','-Distro',$Distro,'-Workspace',$workspace,'-MaskRun',$maskRun)
& powershell.exe @runArgs
if ($LASTEXITCODE -ne 0) { throw 'TripoSR run blocked or failed. Read the error and the newest runs\hair_bench_*\triposr.log. Existing cache/env are retained.' }
