param([int]$Port = 8090, [switch]$CheckOnly, [string]$ConfigPath = 'configs/headapi-demo.json')
$ErrorActionPreference = 'Stop'
$demoRoot = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
# Read supported settings as data; never execute dotenv contents or print values.
# Existing process settings win, followed by .env.local, then .env.
foreach ($demoEnvName in @('.env.local', '.env')) {
    $demoEnvPath = Join-Path $demoRoot $demoEnvName
    if (-not (Test-Path -LiteralPath $demoEnvPath -PathType Leaf)) { continue }
    foreach ($demoLine in [System.IO.File]::ReadAllLines($demoEnvPath)) {
        if ($demoLine -notmatch '^\s*(?:export\s+)?(HEADAPI_API_KEY|HEADAPI_ALLOWED_IMAGE_HOSTS)\s*=\s*(.*)$') { continue }
        $demoSetting = $Matches[1]
        $demoValue = $Matches[2].Trim()
        if (($demoValue.StartsWith('"') -and $demoValue.EndsWith('"')) -or
            ($demoValue.StartsWith("'") -and $demoValue.EndsWith("'"))) {
            if ($demoValue.Length -lt 2) { throw "Invalid quoted setting in $demoEnvName" }
            $demoValue = $demoValue.Substring(1, $demoValue.Length - 2)
        } else {
            $demoValue = ($demoValue -replace '\s+#.*$', '').TrimEnd()
        }
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($demoSetting, 'Process')) -and
            -not [string]::IsNullOrWhiteSpace($demoValue)) {
            [Environment]::SetEnvironmentVariable($demoSetting, $demoValue, 'Process')
        }
    }
}
if ([string]::IsNullOrWhiteSpace($env:HEADAPI_API_KEY)) {
    throw 'Set HEADAPI_API_KEY in .env.local, .env, or this terminal to match backend app.ai.api-key.'
}
if ($CheckOnly) {
    Write-Output 'Head API environment check passed. Secret values are not displayed.'
    return
}
$demoPython = Join-Path $demoRoot '.venv/Scripts/python.exe'
$demoConfig = Join-Path $demoRoot $ConfigPath
Push-Location -LiteralPath $demoRoot
try {
    # Uvicorn logs to stderr; treat process exit code as the native failure signal.
    $demoPreviousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $demoPython -m tools.headapi.api --config $demoConfig --port $Port
        $demoExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $demoPreviousErrorAction
    }
    if ($demoExitCode -ne 0) { throw "Head API exited with code $demoExitCode" }
} finally {
    Pop-Location
}
