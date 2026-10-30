[CmdletBinding()]
param([string]$SubjectDir,[int]$Port=8765)
$ErrorActionPreference='Stop'
$project=$PSScriptRoot
if (!$SubjectDir) {
    $candidates=@(Get-ChildItem -LiteralPath (Join-Path $project 'runs') -Filter 'subject.json' -Recurse -File | Where-Object { $_.Directory.Parent.Name -like 'hair_bench_*' } | Sort-Object LastWriteTime -Descending)
    if (!$candidates.Count) { throw 'No benchmark subject yet. Run RUN-HAIR-BENCH.ps1 -Mode All first, or pass -SubjectDir.' }
    $SubjectDir=$candidates[0].Directory.FullName
}
$manifest=Get-Content -LiteralPath (Join-Path $SubjectDir 'subject.json') -Raw | ConvertFrom-Json
$python=Join-Path $project '.venv\Scripts\python.exe'
& $python (Join-Path $project 'hair-bench\scripts\hair_studio.py') --project $manifest.viewer_project --run $manifest.viewer_run --port $Port --open-browser
if ($LASTEXITCODE -ne 0) { throw 'Viewer failed. Read the Python error above; the viewer needs existing project numpy/scipy/trimesh/Pillow dependencies.' }
