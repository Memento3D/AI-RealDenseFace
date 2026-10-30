[CmdletBinding()]
param(
    [ValidateSet('Inspect','Setup','Run','All')][string]$Mode='Inspect',
    [string[]]$Models=@('triposr','hunyuan-mini','difflocks'),
    [string]$Workspace,
    [string]$MaskRun,
    [string]$SubjectDir,
    [string]$Distro
)
$ErrorActionPreference='Stop'
$project=$PSScriptRoot
$kit=Join-Path $project 'hair-bench'
$wsl=Get-Command wsl.exe -ErrorAction SilentlyContinue
if (!$wsl) { throw 'WSL is not installed. This kit needs an existing WSL2 Linux environment with NVIDIA GPU access; it will not change Windows features or drivers automatically.' }
$wslArgs=@()
if ($Distro) { $wslArgs+=@('--distribution',$Distro) }
function Invoke-Linux([string[]]$Arguments) {
    & $wsl.Source @wslArgs --exec @Arguments
}
function Linux-Path([string]$Path) {
    $value=Invoke-Linux @('wslpath','-a','-u',$Path.Replace('\','/'))
    if ($LASTEXITCODE -ne 0) { throw "Cannot convert path for WSL: $Path" }
    return ($value -join "`n").Trim()
}
$null=Invoke-Linux @('python3','--version')
if ($LASTEXITCODE -ne 0) { throw 'Existing WSL distribution needs python3. Select it with -Distro Ubuntu if necessary.' }
$output=Join-Path $project ('runs\hair_bench_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $output | Out-Null
$hardware=@{ computer_ram_gib=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2) }
$smi=Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) { $hardware.gpu_csv=(& $smi.Source '--query-gpu=name,memory.total,memory.free,driver_version' '--format=csv,noheader,nounits' | Out-String).Trim() }
$subject=$null
if ($Mode -in @('Run','All')) {
    if ($SubjectDir) {
        $subject=Join-Path (Resolve-Path -LiteralPath $SubjectDir).Path 'subject.json'
        if (!(Test-Path -LiteralPath $subject -PathType Leaf)) { throw "Missing prepared subject: $subject" }
    } else {
        $python=Join-Path $project '.venv\Scripts\python.exe'
        if (!(Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing project CPU Python: $python" }
        if (!$Workspace) { $Workspace=Join-Path $project 'runs\v1_20261025_132459_bbbc53c9\three_photo_346bb6bf\head_workspace_3cdeba95\head.workspace.html' }
        if (!(Test-Path -LiteralPath $Workspace -PathType Leaf)) { throw "Workspace not found: $Workspace . Pass -Workspace with the actual head.workspace.html path." }
        $prepared=Join-Path $output 'subject'
        $prepareArgs=@((Join-Path $kit 'scripts\prepare_subject.py'),'--project',$project,'--workspace',(Resolve-Path -LiteralPath $Workspace).Path,'--output',$prepared)
        if ($MaskRun) { $prepareArgs+=@('--mask-run',(Resolve-Path -LiteralPath $MaskRun).Path) }
        & $python -X utf8 @prepareArgs
        if ($LASTEXITCODE -ne 0) { throw 'Subject preparation failed. Pass -MaskRun to the matching V2.1/V2.2 result containing shared/source_reference.glb, shared/view_0.labels.png, report.json and checksums.json.' }
        $subject=Join-Path $prepared 'subject.json'
    }
}
# Find the physical Windows volume backing the selected distro VHD.
$lxss = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
$distroKey = $null
if ($Distro) {
    $distroKey = Get-ChildItem -LiteralPath $lxss | Get-ItemProperty | Where-Object { $_.DistributionName -eq $Distro } | Select-Object -First 1
} else {
    $defaultId = (Get-ItemProperty -LiteralPath $lxss).DefaultDistribution
    if ($defaultId) { $distroKey = Get-ItemProperty -LiteralPath (Join-Path $lxss $defaultId) }
}
if (!$distroKey -or !$distroKey.BasePath) { throw 'Cannot locate the selected WSL storage. Pass -Distro with its registered name.' }
$storageBase = [Environment]::ExpandEnvironmentVariables($distroKey.BasePath)
if ($storageBase.StartsWith('\\?\')) { $storageBase = $storageBase.Substring(4) }
$stateDrive = [System.IO.Path]::GetPathRoot($storageBase)
$projectDrive = [System.IO.Path]::GetPathRoot($project)
$hostDisks = @(@($stateDrive,$projectDrive) | Select-Object -Unique | ForEach-Object { Linux-Path $_ })
$request=@{mode=$Mode.ToLowerInvariant();models=@($Models);output_linux=(Linux-Path $output);windows_hardware=$hardware;host_disks_linux=$hostDisks}
if ($subject) { $request.subject_linux=Linux-Path $subject }
$requestFile=Join-Path $output 'request.json'
$request | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $requestFile -Encoding UTF8
Write-Host "Mode: $Mode . Model downloads/setup logs: $output"
Write-Host 'Models run sequentially. First setup downloads weights; later runs reuse the isolated cache.'
Invoke-Linux @('python3',(Linux-Path (Join-Path $kit 'scripts\control.py')),'--request',(Linux-Path $requestFile))
$code=$LASTEXITCODE
$report=Join-Path $output 'report.html'
if (Test-Path -LiteralPath $report) { Start-Process $report }
if ($subject) { Write-Host "Compare generated hair: powershell -ExecutionPolicy Bypass -File .\OPEN-HAIR-BENCH.ps1 -SubjectDir `"$(Split-Path $subject)`"" }
if ($code -eq 130) { Write-Host "Cancelled by user. Cached files retained. Report: $report" -ForegroundColor Yellow; exit 130 }
if ($code -ne 0) { throw "Benchmark blocked or failed (code $code). See $report and model .log files. No successful inference is implied." }
Write-Host "Report: $report"
