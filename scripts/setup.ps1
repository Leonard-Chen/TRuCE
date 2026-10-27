param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [switch]$ReuseInstalledPackages
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

& $Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "需要 Python 3.9 或更高版本，请通过 -Python 指定解释器。"
    exit 1
}

$venvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $projectRoot $VenvDir }
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Test-Path -LiteralPath $venvPath) {
        Write-Error "目录已存在但没有可用的 Windows Python：$venvPath。请指定新的 -VenvDir，不要复用 Linux 环境。"
        exit 1
    }
    if ($ReuseInstalledPackages) {
        & $Python -m venv --system-site-packages $venvPath
    } else {
        & $Python -m venv $venvPath
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $venvPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "已有环境的 Python 版本低于 3.9。"
    exit 1
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m pip install "torch==2.8.0" --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt") "torch==2.8.0" "transformers==4.57.6"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m truce doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "环境已就绪：$venvPython"
