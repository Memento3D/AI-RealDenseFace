$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$cleanupStore = Join-Path $cleanupRoot 'maintenance/local-cleanup/20261110'
$cleanupPlan = Get-Content -LiteralPath (Join-Path $cleanupStore 'plan.json') -Raw | ConvertFrom-Json
if ([IO.Path]::GetFullPath($cleanupPlan.root) -ne $cleanupRoot) { throw 'Unexpected workspace in cleanup plan' }
function Get-SafeCleanupPath([string]$relative) {
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $relative))
    $boundary = (Join-Path $cleanupRoot 'runs') + [IO.Path]::DirectorySeparatorChar
    if (-not $target.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) { throw "Outside runs: $relative" }
    $parent = $target
    while ($parent -and $parent -ne $cleanupRoot) {
        if ((Test-Path -LiteralPath $parent) -and ((Get-Item -LiteralPath $parent -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Linked path refused: $relative" }
        $parent = [IO.Path]::GetDirectoryName($parent)
    }
    return $target
}
$removed = [Collections.Generic.List[object]]::new()
$preserved = [Collections.Generic.List[object]]::new()
$failures = [Collections.Generic.List[object]]::new()
foreach ($record in $cleanupPlan.preserve_parameters) {
    try {
        $target = Get-SafeCleanupPath $record.path
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
        $destination = Join-Path (Join-Path $cleanupStore 'parameters') $record.path
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destination)) -Force | Out-Null
        Copy-Item -LiteralPath $target -Destination $destination
        $pin = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $pin) { throw 'Copy checksum mismatch' }
        Remove-Item -LiteralPath $target
        $preserved.Add(@{path=$record.path; sha256=$pin; size=$record.size})
    } catch { $failures.Add(@{path=$record.path; error=$_.Exception.Message}) }
}
$allowed = @('.glb','.blend','.blend1','.stl','.obj','.ply','.fbx','.png','.jpg','.jpeg','.html','.gif','.mp4','.zip','.log','.whl','.before-natural-review-20261105','.before-neck-review-20261105')
foreach ($record in $cleanupPlan.delete_generated) {
    try {
        $target = Get-SafeCleanupPath $record.path
        if (-not ($allowed -contains [IO.Path]::GetExtension($target).ToLowerInvariant())) { continue }
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { continue }
        $size = (Get-Item -LiteralPath $target).Length
        if ($size -ne $record.size) { throw 'File changed since inventory' }
        Remove-Item -LiteralPath $target
        $removed.Add(@{path=$record.path; size=$size})
    } catch { $failures.Add(@{path=$record.path; error=$_.Exception.Message}) }
}
$bytesRemoved = ($removed | ForEach-Object { $_.size } | Measure-Object -Sum).Sum
$result = @{deleted=$removed; preserved_parameters=$preserved; failures=$failures; deleted_bytes=$bytesRemoved}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $cleanupStore 'result.json') -Encoding utf8
Write-Output "Deleted $($removed.Count) generated files; retained $($preserved.Count) parameter records; errors $($failures.Count). Bytes removed: $($result.deleted_bytes)"
