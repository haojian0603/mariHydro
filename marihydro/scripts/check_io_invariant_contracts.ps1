#!/usr/bin/env pwsh
#
# Blocking guard for IO-layer internal invariants.
# Once non-empty axes or bounded calendar ranges are established, the code must
# fail explicitly instead of using unwrap_or-style synthetic defaults.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Errors = @()

function Add-Failure {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    $script:Errors += $Message
}

function Check-PatternAbsent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required IO invariant file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Add-Failure $Message
        $matches | Select-Object -First 5 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

function Check-PatternPresent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required IO invariant file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking IO invariant contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'lon_range:\s*\(lon\[0\], \*lon\.last\(\)\.unwrap_or\(&lon\[0\]\)\)|lat_range:\s*\(lat\[0\], \*lat\.last\(\)\.unwrap_or\(&lat\[0\]\)\)' `
        -Message "tide grid detection must not fabricate axis endpoints after regular-axis validation"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'fn axis_bounds\(|TidalIoError::FormatError\(format!\(' `
        -Message "tide grid detection must use explicit axis-bound failures"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" `
        -Pattern 'unwrap_or\(12\)' `
        -Message "CF calendar conversion must not collapse missing month lookup to December"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" `
        -Pattern 'find_month_index_or_panic|day-of-year out of range for' `
        -Message "CF calendar conversion must expose an explicit day-of-year invariant failure"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] IO invariant contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] IO invariant contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
