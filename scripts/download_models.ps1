<#
.SYNOPSIS
    Optional model prefetch - downloads ML weights ahead of first use.

.DESCRIPTION
    Runs the backend venv's Python with a small script that:
      a) downloads the faster-whisper "small" model (lyric transcription)
         if the faster-whisper package is installed, and
      b) downloads MusicGen weights (facebook/musicgen-small) if audiocraft
         is installed.
    Each step skips cleanly when the library is absent (core-only installs).
    Prints download cache sizes at the end.

    Models land in the Hugging Face cache (%USERPROFILE%\.cache\huggingface),
    so the first real song generation does not stall on a multi-GB download.

    Run from anywhere:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\download_models.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RepoRoot 'backend'
$VenvPython = Join-Path $BackendDir '.venv\Scripts\python.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host "[FAIL] backend\.venv not found - run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$PyCode = @'
import importlib.util
import sys

def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False

failures = 0

# --- (a) faster-whisper "small" -------------------------------------------
if have("faster_whisper"):
    print("[..] faster-whisper installed - fetching the 'small' model (~460 MB on first run)...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel("small", device="cpu", compute_type="int8")
        print("[OK] faster-whisper 'small' is ready.")
    except Exception as exc:
        failures += 1
        print("[FAIL] faster-whisper download failed: %s" % exc)
else:
    print("[skip] faster_whisper not installed (core-only setup) - lyrics fall back to energy-only segmentation.")

# --- (b) MusicGen weights ---------------------------------------------------
if have("audiocraft"):
    print("[..] audiocraft installed - fetching MusicGen weights (facebook/musicgen-small, ~1.6 GB on first run)...")
    try:
        from audiocraft.models import MusicGen
        MusicGen.get_pretrained("facebook/musicgen-small")
        print("[OK] musicgen-small is ready.")
    except Exception as exc:
        failures += 1
        print("[FAIL] MusicGen download failed: %s" % exc)
else:
    print("[skip] audiocraft not installed (core-only setup) - generation falls back to the procedural synth engine.")

sys.exit(1 if failures else 0)
'@

$TmpPy = Join-Path $env:TEMP ("enjoi_download_models_{0}.py" -f ([guid]::NewGuid().ToString('N')))
Set-Content -Path $TmpPy -Value $PyCode -Encoding ASCII

Write-Host "[OK] running model prefetch with backend venv python ..." -ForegroundColor Green
& $VenvPython $TmpPy
$PyExit = $LASTEXITCODE
Remove-Item $TmpPy -Force -ErrorAction SilentlyContinue

# --- report cache sizes
function Get-DirSizeMB ([string]$path) {
    if (-not (Test-Path $path)) { return 0 }
    $sum = (Get-ChildItem -Path $path -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    if (-not $sum) { return 0 }
    return [math]::Round($sum / 1MB, 1)
}

Write-Host ""
Write-Host "Model cache sizes:" -ForegroundColor Cyan
$HfCache    = Join-Path $env:USERPROFILE '.cache\huggingface'
$TorchCache = Join-Path $env:USERPROFILE '.cache\torch'
Write-Host ("  {0,-50} {1,10} MB" -f $HfCache,    (Get-DirSizeMB $HfCache))
Write-Host ("  {0,-50} {1,10} MB" -f $TorchCache, (Get-DirSizeMB $TorchCache))

if ($PyExit -ne 0) {
    Write-Host "[!] one or more downloads failed - see messages above (re-run to resume)." -ForegroundColor Yellow
} else {
    Write-Host "[OK] model prefetch finished." -ForegroundColor Green
}
exit $PyExit
