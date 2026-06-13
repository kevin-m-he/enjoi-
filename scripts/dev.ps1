<#
.SYNOPSIS
    Development launcher - backend in a new window + Vite/Electron in this one.

.DESCRIPTION
    1. Opens a new PowerShell window running scripts\run_backend.ps1
       (uvicorn on http://127.0.0.1:8723).
    2. Runs `npm run dev` in frontend\ in the foreground (Vite dev server +
       Electron pointed at it). Ctrl+C here stops the frontend; close the
       backend window separately when done.

    Run from anywhere:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dev.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $RepoRoot 'frontend'
$RunBackend  = Join-Path $PSScriptRoot 'run_backend.ps1'

# --- sanity checks
if (-not (Test-Path $RunBackend)) {
    Write-Host "[FAIL] scripts\run_backend.ps1 is missing." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $FrontendDir 'package.json'))) {
    Write-Host "[FAIL] frontend\package.json not found - run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] npm not found - install Node.js 18+ (winget install OpenJS.NodeJS.LTS)." -ForegroundColor Red
    exit 1
}

# --- backend in its own window (keeps logs separate, survives frontend restarts)
Write-Host "[OK] launching backend window (http://127.0.0.1:8723) ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', $RunBackend
)

# --- frontend dev (foreground)
Write-Host "[OK] starting frontend dev (npm run dev) ..." -ForegroundColor Green
Set-Location $FrontendDir
& npm run dev
exit $LASTEXITCODE
