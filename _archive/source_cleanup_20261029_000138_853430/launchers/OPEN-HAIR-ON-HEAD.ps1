param([string]$FitRun="",[string]$BenchRun="")
$ErrorActionPreference="Stop"
$project=$PSScriptRoot
if(![string]::IsNullOrWhiteSpace($BenchRun)){$requested=(Resolve-Path -LiteralPath $BenchRun).Path;if($requested -eq $project){$BenchRun=""}else{$BenchRun=$requested}}
$python=Join-Path $project ".venv\Scripts\python.exe"
if(!(Test-Path -LiteralPath $python -PathType Leaf)){throw "Missing Python: $python"}
if([string]::IsNullOrWhiteSpace($FitRun)){
    if([string]::IsNullOrWhiteSpace($BenchRun)){
        if(Test-Path -LiteralPath (Join-Path (Get-Location).Path "subject\subject.json")){$BenchRun=(Get-Location).Path}
        else{$candidate=Get-ChildItem -LiteralPath (Join-Path $project "runs") -Directory -Filter "hair_bench_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1;if(!$candidate){throw "No benchmark found"};$BenchRun=$candidate.FullName}
    }
    $latest=Get-ChildItem -LiteralPath $BenchRun -Directory -Filter "fitted_hair_*" | Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName "head_with_hair.glb")} | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if(!$latest){throw "No fitted result yet. Run RUN-HAIR-ON-HEAD.ps1 first."};$FitRun=$latest.FullName
}
$FitRun=(Resolve-Path -LiteralPath $FitRun).Path
$report=Get-Content -LiteralPath (Join-Path $FitRun "report.json") -Raw | ConvertFrom-Json
if($report.status -ne "FITTED_ON_SOURCE_HEAD"){throw "Result did not pass fitting gates. Inspect review.html."}
$ready=Join-Path $FitRun "review_server.json"
if(Test-Path -LiteralPath $ready){
    try{$url=(Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json).url;$check=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2;if($check.StatusCode -eq 200){Start-Process $url;exit 0}}catch{}
    Remove-Item -LiteralPath $ready
}
$server=Join-Path $project "scripts\serve_hair_on_head.py"
$argsForServer=@(('"'+$server+'"'),"--directory",('"'+$FitRun+'"'))
Start-Process -FilePath $python -ArgumentList $argsForServer -WindowStyle Hidden -RedirectStandardOutput (Join-Path $FitRun "viewer.stdout.log") -RedirectStandardError (Join-Path $FitRun "viewer.stderr.log")
for($i=0;$i -lt 80 -and !(Test-Path -LiteralPath $ready);$i++){Start-Sleep -Milliseconds 100}
if(Test-Path -LiteralPath $ready){Start-Process ((Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json).url)}else{Start-Process (Join-Path $FitRun "review.html")}
