param([string]$BenchRun="",[string]$Workspace="",[string]$ParserModel="D:\AI-cache\face-parsing\resnet18.onnx",[string]$BackImage="",[switch]$NoBackImage,[switch]$NoOpen)
$ErrorActionPreference="Stop"
$project=(Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
if(![string]::IsNullOrWhiteSpace($BenchRun)){$requested=(Resolve-Path -LiteralPath $BenchRun).Path;if($requested -eq $project){$BenchRun=""}else{$BenchRun=$requested}}
$python=Join-Path $project ".venv\Scripts\python.exe"
$script=Join-Path $project "scripts\run_on_head.py"
$profile=Join-Path $project "configs\hair-on-head.json"
$manifest=Join-Path $project "configs/manifests/hair-on-head-code.json"
foreach($file in @($python,$script,$profile,$manifest)){if(!(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing file: $file"}}
$checks=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
foreach($entry in $checks.files){$file=Join-Path $project $entry.path;if(!(Test-Path -LiteralPath $file -PathType Leaf)){throw "Incomplete package: $file"};if((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $entry.sha256){throw "Code differs from release: $file"}}
if([string]::IsNullOrWhiteSpace($BenchRun)){
    if(Test-Path -LiteralPath (Join-Path (Get-Location).Path "subject\subject.json")){$BenchRun=(Get-Location).Path}
    else{$latest=Get-ChildItem -LiteralPath (Join-Path $project "runs") -Directory -Filter "hair_bench_*" | Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName "subject\subject.json")} | Sort-Object LastWriteTime -Descending | Select-Object -First 1;if(!$latest){throw "No benchmark subject found. Pass -BenchRun."};$BenchRun=$latest.FullName}
}
$BenchRun=(Resolve-Path -LiteralPath $BenchRun).Path
$subject=Get-Content -LiteralPath (Join-Path $BenchRun "subject\subject.json") -Raw | ConvertFrom-Json
if([string]::IsNullOrWhiteSpace($Workspace)){
    $candidates=@((Join-Path $project "hair-bench\input04\head.workspace.html"),(Join-Path $project "runs\v1_20261025_132459_bbbc53c9\three_photo_346bb6bf\head_workspace_3cdeba95\head.workspace.html"))
    foreach($file in $candidates){if((Test-Path -LiteralPath $file -PathType Leaf) -and (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -eq $subject.workspace_sha256){$Workspace=$file;break}}
    if([string]::IsNullOrWhiteSpace($Workspace)){throw "No workspace with matching subject SHA. Pass -Workspace to the original head.workspace.html."}
}
foreach($file in @($Workspace,$ParserModel)){if(!(Test-Path -LiteralPath $file -PathType Leaf)){throw "Missing input: $file"}}
if([string]::IsNullOrWhiteSpace($BackImage) -and !$NoBackImage){$file=Join-Path $project "input\04\back.jpg";if(Test-Path -LiteralPath $file -PathType Leaf){$BackImage=$file}}
& $python -c "import numpy, scipy, PIL, trimesh, manifold3d, onnxruntime, skimage, rtree, networkx; assert hasattr(manifold3d,'Mesh64')"
if($LASTEXITCODE -ne 0){& $python -m pip install -r (Join-Path $project "requirements\hair-on-head-cpu.txt");if($LASTEXITCODE -ne 0){throw "CPU dependency install failed"}}
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fff"
$output=Join-Path $BenchRun "fitted_hair_$stamp"
$arguments=@("-u",$script,"--bench",$BenchRun,"--workspace",$Workspace,"--parser-model",$ParserModel,"--profile",$profile,"--output",$output)
if(!$NoBackImage -and ![string]::IsNullOrWhiteSpace($BackImage)){if(!(Test-Path -LiteralPath $BackImage -PathType Leaf)){throw "Missing back image: $BackImage"};$arguments+=@("--back-image",$BackImage)}
& $python @arguments
$exitCode=$LASTEXITCODE
Write-Host "Result: $output"
if($exitCode -ne 0){if(!$NoOpen -and (Test-Path -LiteralPath (Join-Path $output "review.html"))){Start-Process (Join-Path $output "review.html")};throw "Fit rejected; diagnostics retained. Inspect review.html."}
if(!$NoOpen){
    $server=Join-Path $project "scripts\serve_hair_on_head.py"
    $serverArgs=@('"'+$server+'"',"--directory",'"'+$output+'"')
    Start-Process -FilePath $python -ArgumentList $serverArgs -WindowStyle Hidden -RedirectStandardOutput (Join-Path $output "viewer.stdout.log") -RedirectStandardError (Join-Path $output "viewer.stderr.log")
    $ready=Join-Path $output "review_server.json"
    for($i=0;$i -lt 80 -and !(Test-Path -LiteralPath $ready);$i++){Start-Sleep -Milliseconds 100}
    if(Test-Path -LiteralPath $ready){$url=(Get-Content -LiteralPath $ready -Raw | ConvertFrom-Json).url;Start-Process $url}else{Start-Process (Join-Path $output "review.html")}
}
Write-Host "Hair fitted on the source head. No GPU inference was rerun. Physical print validation remains pending." -ForegroundColor Green
