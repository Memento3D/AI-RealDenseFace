$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $python -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable installation failed' }
$reports = @(Get-ChildItem .\runs -Filter report.json -Recurse -File | ForEach-Object {
    try { [PSCustomObject]@{ File = $_; Report = (Get-Content $_.FullName -Raw -Encoding UTF8 | ConvertFrom-Json) } } catch { }
})
$candidate = $reports | Where-Object {
    $_.Report.pipeline_type -eq 'bounded_frontal_jaw_proxy_fit' -and
    $_.Report.status -in @('JAW_EXPERIMENT_EXPORTED', 'NO_SHAPE_CHANGE')
} | Sort-Object { $_.File.LastWriteTimeUtc } -Descending | Select-Object -First 1
if (-not $candidate) { throw 'No successful fit-jaw report found in runs' }
$expected = $candidate.Report.input_sha256.registration
$registration = $reports | Where-Object {
    $_.Report.pipeline_type -eq 'three_view_camera_registration' -and
    $_.Report.status -eq 'REGISTRATIONS_EXPORTED' -and
    (Get-FileHash $_.File.FullName -Algorithm SHA256).Hash -eq $expected
} | Select-Object -First 1
if (-not $registration) { throw 'Cannot find the exact registration linked to this candidate' }
$output = Join-Path $projectRoot ('runs\multiview_color_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
Write-Host ('Candidate: ' + $candidate.File.Directory.FullName)
Write-Host ('Registration: ' + $registration.File.Directory.FullName)
& $python -m memento3d_head preview-multiview-color --candidate-run $candidate.File.Directory.FullName --registration-run $registration.File.Directory.FullName --output $output
if ($LASTEXITCODE -ne 0) { throw 'Multiview preview failed; see terminal error' }
Start-Process (Join-Path $output 'multiview_preview.html')
