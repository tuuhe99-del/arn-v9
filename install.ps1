# Windows PowerShell one-command installer.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
  & py -3 "$Root\install.py" @args
  exit $LASTEXITCODE
}
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
  & python "$Root\install.py" @args
  exit $LASTEXITCODE
}
Write-Error "Python 3.10+ is required. Install Python from python.org, then rerun: powershell -ExecutionPolicy Bypass -File .\install.ps1"
