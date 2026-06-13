<#
.SYNOPSIS
    Start the enjoi backend (FastAPI/uvicorn on http://127.0.0.1:8723).

.DESCRIPTION
    Activates backend\.venv and runs uvicorn from the backend folder.
    Prepends ffmpeg to this session's PATH if it only exists in the winget
    package folder (fresh shells after `winget install Gyan.FFmpeg` often
    have not picked up the PATH change yet).

    Run from anywhere:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_backend.ps1

.PARAMETER Port
    TCP port to bind (default 8723 - keep the default unless you also change
    the frontend's base URL).
#>
[CmdletBinding()]
param(
    [int]$Port = 8723
)

$ErrorActionPreference = 'Continue'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RepoRoot 'backend'
$VenvPython = Join-Path $BackendDir '.venv\Scripts\python.exe'

# --- ffmpeg PATH probe (must run before the server starts so yt-dlp/export see it)
function Resolve-Ffmpeg {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
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
    Write-Host ("[OK] ffmpeg: {0}" -f $Ffmpeg) -ForegroundColor Green
} else {
    Write-Host "[!] ffmpeg not found - reference analysis and MP3 export will degrade." -ForegroundColor Yellow
    Write-Host "    Install with:  winget install Gyan.FFmpeg   (then open a new terminal)" -ForegroundColor Yellow
}

# --- venv check
if (-not (Test-Path $VenvPython)) {
    Write-Host "[FAIL] backend\.venv not found." -ForegroundColor Red
    Write-Host "       Run setup first:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1" -ForegroundColor Yellow
    exit 1
}

# --- start uvicorn from the backend folder (imports are package-relative)
Set-Location $BackendDir
Write-Host ("[OK] starting backend on http://127.0.0.1:{0}  (Ctrl+C to stop)" -f $Port) -ForegroundColor Green
& $VenvPython -m uvicorn main:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
