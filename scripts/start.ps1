<#
.SYNOPSIS
    Production-ish launcher - built frontend + backend window + Electron.

.DESCRIPTION
    1. Ensures frontend\dist exists (runs `npm run build` if it does not).
    2. Starts the backend in a new PowerShell window (uvicorn on
       http://127.0.0.1:8723) and waits for /api/health to answer.
    3. Launches Electron against the built files (`npx electron .` from
       frontend\ with VITE_DEV_SERVER unset, so Electron loads dist\index.html
       instead of the Vite dev server).

    Run from anywhere:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $RepoRoot 'frontend'
$DistIndex   = Join-Path $FrontendDir 'dist\index.html'
$RunBackend  = Join-Path $PSScriptRoot 'run_backend.ps1'
$HealthUrl   = 'http://127.0.0.1:8723/api/health'

# --- sanity checks
if (-not (Test-Path (Join-Path $FrontendDir 'package.json'))) {
    Write-Host "[FAIL] frontend\package.json not found - run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] npm not found - install Node.js 18+ (winget install OpenJS.NodeJS.LTS)." -ForegroundColor Red
    exit 1
}

# --- 1. build frontend if needed
if (Test-Path $DistIndex) {
    Write-Host "[OK] frontend\dist already built" -ForegroundColor Green
} else {
    Write-Host "[..] frontend\dist not found - running npm run build ..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    & npm run build
    $buildExit = $LASTEXITCODE
    Pop-Location
    if ($buildExit -ne 0 -or -not (Test-Path $DistIndex)) {
        Write-Host "[FAIL] frontend build failed (see output above)." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] frontend built" -ForegroundColor Green
}

# --- 2. backend window
if (-not (Test-Path $RunBackend)) {
    Write-Host "[FAIL] scripts\run_backend.ps1 is missing." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] launching backend window (http://127.0.0.1:8723) ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', $RunBackend
)

# --- wait for the backend to answer (max ~30 s)
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if ($ready) {
    Write-Host "[OK] backend is up" -ForegroundColor Green
} else {
    Write-Host "[!] backend did not answer /api/health within 30 s - launching Electron anyway." -ForegroundColor Yellow
    Write-Host "    Check the backend window for errors (missing venv? port 8723 busy?)." -ForegroundColor Yellow
}

# --- 3. Electron against built files (no dev server)
if (Test-Path Env:VITE_DEV_SERVER) {
    Remove-Item Env:VITE_DEV_SERVER -ErrorAction SilentlyContinue
}
Write-Host "[OK] launching Electron (built files) ..." -ForegroundColor Green
Set-Location $FrontendDir
& npx electron .
exit $LASTEXITCODE
