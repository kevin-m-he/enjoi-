<#
.SYNOPSIS
    Install the optional GPU MusicGen + Whisper stack into the backend venv.

.DESCRIPTION
    audiocraft 1.3.0 is fragile to install on Windows: it hard-pins torch==2.1.0
    (which would replace a CUDA build with a CPU one) and av==11.0.0 (no Windows
    wheel -> tries to compile against FFmpeg and fails). This script installs a
    known-good combination instead:

      1. torch/torchaudio 2.1.0 + CUDA 12.1 (cu121) when an NVIDIA GPU is present,
         otherwise the CPU build.
      2. audiocraft's dependency stack with a modern prebuilt `av` wheel and the
         2024-era versions it expects (torch held fixed by a constraints file).
      3. audiocraft itself with --no-deps.
      4. faster-whisper.
      5. (optional) prefetch the MusicGen + Whisper models with -Prefetch.

    Verified on Windows 11 / Python 3.11 / RTX 4080.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-ml.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install-ml.ps1 -Prefetch
#>
[CmdletBinding()]
param(
    [switch]$Prefetch
)

$ErrorActionPreference = 'Stop'
$RepoRoot   = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $RepoRoot 'backend'
$Py         = Join-Path $BackendDir '.venv\Scripts\python.exe'

if (-not (Test-Path $Py)) {
    Write-Host "[FAIL] backend venv not found at $Py - run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

# --- detect NVIDIA GPU --------------------------------------------------------
$gpu = $false
try { & nvidia-smi *> $null; if ($LASTEXITCODE -eq 0) { $gpu = $true } } catch { $gpu = $false }
if ($gpu) {
    Write-Host "[OK] NVIDIA GPU detected - installing CUDA (cu121) build." -ForegroundColor Green
    $index = 'https://download.pytorch.org/whl/cu121'
    $torchPin = 'torch==2.1.0+cu121'
    $taPin = 'torchaudio==2.1.0+cu121'
} else {
    Write-Host "[!] No NVIDIA GPU - installing CPU build (generation will be slow)." -ForegroundColor Yellow
    $index = 'https://download.pytorch.org/whl/cpu'
    $torchPin = 'torch==2.1.0'
    $taPin = 'torchaudio==2.1.0'
}

# --- 1. PyTorch ---------------------------------------------------------------
Write-Host "[..] Installing PyTorch (this is a large download) ..." -ForegroundColor Cyan
& $Py -m pip install torch==2.1.0 torchaudio==2.1.0 --index-url $index
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] torch install failed." -ForegroundColor Red; exit 1 }

# --- 2. audiocraft dependency stack ------------------------------------------
$work = Join-Path $BackendDir '.wheels'
New-Item -ItemType Directory -Force $work | Out-Null
$constraints = Join-Path $work 'constraints.txt'
"$torchPin`n$taPin" | Set-Content $constraints -Encoding Ascii
$deps = Join-Path $work 'ml-deps.txt'
@(
    'av','transformers==4.40.2','huggingface_hub==0.23.4','tokenizers==0.19.1',
    'xformers==0.0.22.post7','encodec==0.1.1','einops','julius','num2words',
    'sentencepiece','omegaconf==2.3.0','hydra-core==1.3.2','hydra_colorlog',
    'demucs==4.0.1','torchmetrics','spacy>=3.6.1,<3.9','flashy'
) | Set-Content $deps -Encoding Ascii

Write-Host "[..] Installing audiocraft dependencies ..." -ForegroundColor Cyan
& $Py -m pip install -r $deps -c $constraints
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] dependency install failed." -ForegroundColor Red; exit 1 }

# --- 3. audiocraft (no deps) --------------------------------------------------
Write-Host "[..] Installing audiocraft (--no-deps) ..." -ForegroundColor Cyan
& $Py -m pip install audiocraft --no-deps
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] audiocraft install failed." -ForegroundColor Red; exit 1 }

# --- 4. faster-whisper --------------------------------------------------------
Write-Host "[..] Installing faster-whisper ..." -ForegroundColor Cyan
& $Py -m pip install faster-whisper -c $constraints
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] faster-whisper install failed." -ForegroundColor Red; exit 1 }

# --- verify -------------------------------------------------------------------
& $Py -c "import torch; from audiocraft.models import MusicGen; from faster_whisper import WhisperModel; print('[OK] imports fine - cuda available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] import check failed." -ForegroundColor Red; exit 1 }

# --- 5. optional model prefetch ----------------------------------------------
if ($Prefetch) {
    Write-Host "[..] Prefetching models (MusicGen medium + Whisper small, ~3.5 GB) ..." -ForegroundColor Cyan
    Push-Location $BackendDir
    & $Py -c "import torch; from enjoi.core import config; from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8', download_root=str(config.models_dir())); from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-medium' if torch.cuda.is_available() else 'facebook/musicgen-small'); print('[OK] models cached')"
    Pop-Location
}

Write-Host ""
Write-Host "[DONE] MusicGen + Whisper installed. Restart the app to pick up the new capabilities." -ForegroundColor Green
