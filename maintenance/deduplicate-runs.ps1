$ErrorActionPreference = 'Stop'
$workspacePath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runsRoot = (Resolve-Path -LiteralPath (Join-Path $workspacePath 'runs')).Path
$inventory = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'run-duplicates-20261104.json') -Raw | ConvertFrom-Json
$records = [Collections.Generic.List[object]]::new()
$logPath = Join-Path $PSScriptRoot 'run-dedup-actions-20261104.json'
foreach ($group in $inventory.duplicates) {
 $sourcePath = (Resolve-Path -LiteralPath (Join-Path $workspacePath $group.paths[0])).Path
 if (-not $sourcePath.StartsWith($runsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside runs' }
 if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $group.sha256) { throw 'Source changed' }
 foreach ($relativePath in $group.paths | Select-Object -Skip 1) {
  $targetPath = (Resolve-Path -LiteralPath (Join-Path $workspacePath $relativePath)).Path
  if (-not $targetPath.StartsWith($runsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside runs' }
  if ((Get-Item -LiteralPath $targetPath -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse target' }
  if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $group.sha256) { throw 'Target changed' }
  $backupPath = $targetPath + '.dedup-backup'
  if (Test-Path -LiteralPath $backupPath) { throw 'Existing backup' }
  Move-Item -LiteralPath $targetPath -Destination $backupPath
  try {
   New-Item -ItemType HardLink -Path $targetPath -Target $sourcePath | Out-Null
   if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $group.sha256) { throw 'Post-link hash mismatch' }
  } catch {
   if (Test-Path -LiteralPath $targetPath) { Remove-Item -LiteralPath $targetPath -Force }
   Move-Item -LiteralPath $backupPath -Destination $targetPath
   throw
  }
  Remove-Item -LiteralPath $backupPath -Force
  $records.Add([ordered]@{path=$relativePath;shares_data_with=$group.paths[0];sha256=$group.sha256;bytes=$group.bytes;status='hardlinked_hash_verified'})
  $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $logPath -Encoding UTF8
 }
}
Write-Output ('Verified hard links: ' + $records.Count)
