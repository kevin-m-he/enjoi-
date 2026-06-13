<#
.SYNOPSIS
    Idempotent setup for enjoi - backend venv + dependencies + frontend npm install.

.DESCRIPTION
    Run from anywhere:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1 -Full

    Steps:
      1. Locate Python 3.11 (or tell you how to install it).
      2. Create backend\.venv if missing.
      3. pip install -r backend\requirements-core.txt
      4. With -Full: install PyTorch (CUDA cu121 wheels when an NVIDIA GPU is
         detected via nvidia-smi, CPU wheels otherwise), then
         backend\requirements-full.txt.
      5. Locate/verify ffmpeg (winget package probe + install instructions).
      6. npm install in frontend\.

    Safe to re-run; every step skips or fast-forwards when already done.

.PARAMETER Full
    Additionally install the optional ML stack (requirements-full.txt:
    torch/torchaudio, audiocraft/MusicGen, faster-whisper, demucs, torchcrepe,
    pyrubberband). Large download; GPU strongly recommended.
#>
[CmdletBinding()]
param(
    [switch]$Full
)

$ErrorActionPreference = 'Continue'

$OK   = [char]0x2713    # check mark
$BAD  = [char]0x2717    # ballot X

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$BackendDir  = Join-Path $RepoRoot 'backend'
$FrontendDir = Join-Path $RepoRoot 'frontend'
$VenvDir     = Join-Path $BackendDir '.venv'
$VenvPython  = Join-Path $VenvDir 'Scripts\python.exe'

function Write-Ok   ([string]$msg) { Write-Host ("  [{0}] {1}" -f $OK,  $msg) -ForegroundColor Green }
function Write-Bad  ([string]$msg) { Write-Host ("  [{0}] {1}" -f $BAD, $msg) -ForegroundColor Red }
function Write-Warn2([string]$msg) { Write-Host ("  [!] {0}" -f $msg) -ForegroundColor Yellow }
function Write-Step ([string]$msg) { Write-Host ("`n== {0} ==" -f $msg) -ForegroundColor Cyan }

function Stop-Setup ([string]$msg, [string]$hint) {
    Write-Bad $msg
    if ($hint) { Write-Host ("      -> {0}" -f $hint) -ForegroundColor Yellow }
    Write-Host "`nSetup aborted." -ForegroundColor Red
    exit 1
}

Write-Host "enjoi setup" -ForegroundColor Magenta
Write-Host ("repo root : {0}" -f $RepoRoot)
if ($Full) { Write-Host "mode      : FULL (core + ML extras)" } else { Write-Host "mode      : core (re-run with -Full for the ML stack)" }

# ---------------------------------------------------------------- Python 3.11
Write-Step "Python 3.11"

function Find-Python311 {
    $candidates = @()

    # 1. Standard per-user install location
    $localPy = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
    if (Test-Path $localPy) { $candidates += $localPy }

    # 2. The py launcher, asked specifically for 3.11
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $fromPy = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $fromPy) { $candidates += ([string]$fromPy).Trim() }
    }

    # 3. Anything called python on PATH - but never the Microsoft Store alias
    #    (the WindowsApps stub just opens the Store and breaks venv creation)
    $onPath = Get-Command python -All -ErrorAction SilentlyContinue
    foreach ($c in $onPath) {
        if ($c.Source -and ($c.Source -notlike '*WindowsApps*')) { $candidates += $c.Source }
    }

    foreach ($cand in ($candidates | Select-Object -Unique)) {
        if (-not (Test-Path $cand)) { continue }
        $ver = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -and (([string]$ver).Trim() -eq '3.11')) { return $cand }
    }
    return $null
}

$SystemPython = $null
if (Test-Path $VenvPython) {
    # venv already exists - verify it actually runs and is 3.11
    $vver = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $vver -and (([string]$vver).Trim() -eq '3.11')) {
        Write-Ok ("existing venv found: {0} (Python {1})" -f $VenvDir, ([string]$vver).Trim())
    } else {
        Write-Warn2 "backend\.venv exists but is broken or not Python 3.11"
        Write-Warn2 "delete backend\.venv and re-run setup to rebuild it"
    }
} else {
    $SystemPython = Find-Python311
    if ($SystemPython) {
        Write-Ok ("Python 3.11 found: {0}" -f $SystemPython)
    } else {
        Stop-Setup "Python 3.11 not found (PATH, py launcher, and %LOCALAPPDATA%\Programs\Python checked)" `
            "Install it with:  winget install Python.Python.3.11   (then open a NEW terminal and re-run this script)"
    }
}

# ----------------------------------------------------------------- create venv
Write-Step "Virtual environment (backend\.venv)"
if (Test-Path $VenvPython) {
    Write-Ok "already present - skipping creation"
} else {
    & $SystemPython -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Stop-Setup "venv creation failed" "check that the Python above is a real install, not the Microsoft Store alias"
    }
    Write-Ok ("created {0}" -f $VenvDir)
}

# ------------------------------------------------------------------- pip core
Write-Step "Core Python dependencies (requirements-core.txt)"
& $VenvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Warn2 "pip self-upgrade failed - continuing with the bundled pip" }

$CoreReq = Join-Path $BackendDir 'requirements-core.txt'
if (-not (Test-Path $CoreReq)) { Stop-Setup ("missing {0}" -f $CoreReq) "" }

& $VenvPython -m pip install -r $CoreReq
if ($LASTEXITCODE -ne 0) {
    Stop-Setup "core dependency install failed (see pip output above)" "fix the reported package and re-run setup.ps1"
}
Write-Ok "core dependencies installed"

# ------------------------------------------------------------------ pip full
if ($Full) {
    Write-Step "ML extras (requirements-full.txt)"

    # Detect an NVIDIA GPU so we can pick the right torch wheels.
    # cu121 wheels come from the dedicated PyTorch index (see the note at the
    # top of requirements-full.txt); CPU wheels come from PyPI.
    $HasGpu = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -L *> $null
        if ($LASTEXITCODE -eq 0) { $HasGpu = $true }
    }

    if ($HasGpu) {
        Write-Ok "NVIDIA GPU detected - installing CUDA 12.1 (cu121) torch wheels"
        & $VenvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    } else {
        Write-Warn2 "no NVIDIA GPU detected (nvidia-smi missing or failed) - installing CPU torch wheels"
        Write-Warn2 "generation will work but is slow on CPU; see Hardware tiers in README.md"
        & $VenvPython -m pip install torch torchaudio
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup "torch/torchaudio install failed" "check network/disk space; for GPU wheels the index is https://download.pytorch.org/whl/cu121"
    }
    Write-Ok "torch + torchaudio installed"

    $FullReq = Join-Path $BackendDir 'requirements-full.txt'
    if (-not (Test-Path $FullReq)) { Stop-Setup ("missing {0}" -f $FullReq) "" }

    & $VenvPython -m pip install -r $FullReq
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup "ML extras install failed (see pip output above)" "the app still runs core-only; re-run setup.ps1 -Full after fixing"
    }
    Write-Ok "ML extras installed (audiocraft, faster-whisper, demucs, torchcrepe, pyrubberband)"
    Write-Warn2 "pyrubberband also needs the rubberband CLI on PATH for formant-safe stretching (optional; librosa fallback is used otherwise)"
}

# -------------------------------------------------------------------- ffmpeg
Write-Step "FFmpeg"
function Resolve-Ffmpeg {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # winget installs may not be on PATH in this (older) shell yet - probe the
    # package folder directly and graft it onto this session's PATH.
    $pkgRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $pkgRoot) {
        $dirs = Get-ChildItem -Path $pkgRoot -Directory -Filter 'Gyan.FFmpeg*' -ErrorAction SilentlyContinue
        foreach ($d in $dirs) {
            $exe = Get-ChildItem -Path $d.FullName -Recurse -Filter 'ffmpeg.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($exe) {
                $env:Path = $exe.DirectoryName + ';' + $env:Path
                return $exe.FullName
            }
        }
    }
    return $null
}

$Ffmpeg = Resolve-Ffmpeg
if ($Ffmpeg) {
    Write-Ok ("ffmpeg: {0}" -f $Ffmpeg)
} else {
    Write-Warn2 "ffmpeg not found on PATH or in the winget package folder"
    Write-Warn2 "install it with:  winget install Gyan.FFmpeg   (then open a NEW terminal)"
    Write-Warn2 "the backend ships an imageio-ffmpeg fallback binary, but a real ffmpeg is strongly recommended for reference analysis and export"
}

# ------------------------------------------------------------------ frontend
Write-Step "Frontend (npm install)"
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Stop-Setup "npm not found" "install Node.js 18+ with:  winget install OpenJS.NodeJS.LTS   (then open a NEW terminal and re-run)"
}
$PkgJson = Join-Path $FrontendDir 'package.json'
if (-not (Test-Path $PkgJson)) {
    Write-Warn2 "frontend\package.json not present yet - skipping npm install (re-run setup.ps1 once the frontend is scaffolded)"
} else {
    Push-Location $FrontendDir
    & npm install
    $npmExit = $LASTEXITCODE
    Pop-Location
    if ($npmExit -ne 0) {
        Stop-Setup "npm install failed (see output above)" "fix the reported issue and re-run setup.ps1"
    }
    Write-Ok "frontend dependencies installed"
}

# -------------------------------------------------------------------- summary
Write-Step "Done"
Write-Ok "setup complete"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  scripts\dev.ps1              - start backend + Vite/Electron dev mode"
Write-Host "  scripts\run_backend.ps1      - start only the backend (http://127.0.0.1:8723)"
Write-Host "  scripts\download_models.ps1  - optional: prefetch Whisper/MusicGen weights (-Full installs first)"
exit 0
