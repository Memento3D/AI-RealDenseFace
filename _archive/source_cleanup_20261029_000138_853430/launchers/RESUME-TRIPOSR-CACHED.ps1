[CmdletBinding()]
param([string]$Distro='Ubuntu-22.04',[string]$SubjectDir)
$ErrorActionPreference='Stop'
$project=$PSScriptRoot
$kit=Join-Path $project 'hair-bench'
$helper=Join-Path $kit 'scripts\finalize_triposr_cache.py'
$launcher=Join-Path $project 'RUN-HAIR-BENCH.ps1'
foreach ($file in @($helper,$launcher,(Join-Path $kit 'models.lock.json'))) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file. Extract the complete cached-resume ZIP into the project root." }
}
$helperLinux=(& wsl.exe -d $Distro --exec wslpath -a -u ($helper.Replace("\", "/")) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot convert helper path for WSL.' }
$kitLinux=(& wsl.exe -d $Distro --exec wslpath -a -u ($kit.Replace("\", "/")) | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot convert kit path for WSL.' }
$python='/home/indows/.local/share/memento-hairbench-v1/envs/triposr/bin/python'
Write-Host 'Completing TripoSR setup using the existing cache and environment...'
& wsl.exe -d $Distro --exec $python -u $helperLinux --kit $kitLinux
if ($LASTEXITCODE -ne 0) { throw 'Cached setup was not completed. Existing cache/env retained. Read the Python error above.' }
if (!$SubjectDir) {
    $runs=Join-Path $project 'runs'
    if (Test-Path -LiteralPath $runs -PathType Container) {
        foreach ($dir in (Get-ChildItem -LiteralPath $runs -Directory -Filter 'hair_bench_*' | Sort-Object LastWriteTime -Descending)) {
            $candidate=Join-Path $dir.FullName 'subject'
            if (Test-Path -LiteralPath (Join-Path $candidate 'subject.json') -PathType Leaf) { $SubjectDir=$candidate; break }
        }
    }
}
$arguments=@('-ExecutionPolicy','Bypass','-File',$launcher,'-Mode','Run','-Models','triposr','-Distro',$Distro)
if ($SubjectDir) { $arguments+=@('-SubjectDir',$SubjectDir) }
Write-Host 'Running TripoSR only in the HairBench Run/offline mode...'
& powershell.exe @arguments
if ($LASTEXITCODE -ne 0) { throw 'TripoSR run blocked or failed. See the new runs\hair_bench_* report and triposr.log.' }
