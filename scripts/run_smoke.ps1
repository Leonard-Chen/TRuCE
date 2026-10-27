param([string]$Output = "runs/smoke", [string]$Device = "auto")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = "utf-8"
& "$projectRoot\.venv\Scripts\python.exe" -m truce smoke --output $Output --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
