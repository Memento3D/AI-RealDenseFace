param([string]$Directory = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'assets\custom-library\mvp-selected-preview-v1'))
$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'maintenance\mvp-asset-upload-plan-20261105.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$verifyRoot = (Resolve-Path -LiteralPath $Directory).Path
foreach ($file in $plan.files) {
    $filePath = Join-Path $verifyRoot $file.local_path
    if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw ('Missing file: ' + $file.local_path) }
    $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $file.sha256 -or (Get-Item -LiteralPath $filePath).Length -ne $file.bytes) { throw ('Checksum/size mismatch: ' + $file.local_path) }
}
Write-Output ('PASS: ' + $plan.file_count + ' files match the MVP upload plan. This does not test remote storage unless Directory points to a downloaded copy.')
