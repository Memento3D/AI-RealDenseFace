param([int]$Port = 8090, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
if ($CheckOnly) {
    & (Join-Path $PSScriptRoot 'START-HEADAPI-DEMO.ps1') -CheckOnly -ConfigPath 'configs/headapi-assembly.json'
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & .venv/Scripts/python.exe -B -c "from tools.headapi.service import JobService,load_config; s=object.__new__(JobService); s.config=load_config('configs/headapi-assembly.json'); s.preflight('mvp'); print('Assembly prerequisites and asset pins: PASS')"
        if ($LASTEXITCODE -ne 0) { throw 'Assembly preflight failed; inspect missing asset or pin above.' }
    } finally { Pop-Location }
    return
}
& (Join-Path $PSScriptRoot 'START-HEADAPI-DEMO.ps1') -Port $Port -ConfigPath 'configs/headapi-assembly.json'
