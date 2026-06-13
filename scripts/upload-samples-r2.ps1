<#
.SYNOPSIS
    Upload the local sample library to the Cloudflare R2 bucket via wrangler.

.DESCRIPTION
    Pushes every backend\sample_library\*.wav to the R2 bucket that the
    enjoi-samples Worker serves (privately, token-gated). Run AFTER:
      wrangler login
      wrangler r2 bucket create enjoi-samples
      (cd cloudflare\sample-worker; wrangler secret put SAMPLE_TOKEN; wrangler deploy)

    Requires Node + wrangler (npm i -g wrangler). Does NOT log in for you.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\upload-samples-r2.ps1
#>
[CmdletBinding()]
param(
    [string]$Bucket = "enjoi-samples"
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Lib = Join-Path $RepoRoot 'backend\sample_library'

if (-not (Test-Path $Lib)) {
    Write-Host "[FAIL] $Lib not found." -ForegroundColor Red; exit 1
}
if (-not (Get-Command wrangler -ErrorAction SilentlyContinue)) {
    Write-Host "[FAIL] wrangler not found. Install: npm i -g wrangler, then 'wrangler login'." -ForegroundColor Red
    exit 1
}

$wavs = Get-ChildItem $Lib -Filter *.wav -File
Write-Host "[..] Uploading $($wavs.Count) loops to R2 bucket '$Bucket' ..." -ForegroundColor Cyan
$i = 0
foreach ($f in $wavs) {
    $i++
    Write-Host ("  [{0}/{1}] {2}" -f $i, $wavs.Count, $f.Name)
    & wrangler r2 object put "$Bucket/$($f.Name)" --file "$($f.FullName)" --content-type "audio/wav" --remote
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] upload failed on $($f.Name)" -ForegroundColor Red; exit 1
    }
}
Write-Host "[DONE] Uploaded $($wavs.Count) loops. Set ENJOI_SAMPLE_CDN + ENJOI_SAMPLE_CDN_TOKEN to use them." -ForegroundColor Green
