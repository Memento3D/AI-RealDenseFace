param([int]$Port = 8090, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
if ($CheckOnly) {
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & .venv/Scripts/python.exe -B -c "from tools.headapi.service import JobService,load_config; s=object.__new__(JobService); s.config=load_config('configs/headapi-customization-base.json'); s.preflight('mvp'); print('Base-only technical prerequisites: PASS. Not a customization release.')"
        if ($LASTEXITCODE -ne 0) { throw 'Base-only preflight failed.' }
    } finally { Pop-Location }
    return
}
Write-Warning 'Technical base-only preview. Local accepted neck/lip baseline and asset placements are not integrated. Existing assembly configuration is unchanged.'
& (Join-Path $PSScriptRoot 'START-HEADAPI-DEMO.ps1') -Port $Port -ConfigPath 'configs/headapi-customization-base.json'
