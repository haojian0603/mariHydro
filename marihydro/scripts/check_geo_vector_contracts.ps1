#!/usr/bin/env pwsh
#
# Blocking guard for explicit vector normalization failure semantics.
# Zero-length vectors must not be normalized through synthetic zero outputs
# or Option-based fallbacks on the public mh_geo geometry surface.
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
        Add-Failure "missing required geo file: $RelativePath"
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
        Add-Failure "missing required geo file: $RelativePath"
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
    Write-Host "=== Checking geo vector contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $geometryPath = "crates/mh_geo/src/geometry.rs"

    Check-PatternAbsent `
        -RelativePath $geometryPath `
        -Pattern 'pub fn normalize\(&self\) -> Option<Self>|pub fn normalize_or_zero\(&self\) -> Self' `
        -Message "vector normalization APIs must not expose Option-based or synthetic-zero public semantics"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'pub fn normalize\(&self\) -> GeoResult<Self>' `
        -Message "vector normalization APIs must return GeoResult<Self>"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'GeoError::zero_length_vector\("Point3D"\)|GeoError::zero_length_vector\("Point2D"\)' `
        -Message "vector normalization must emit explicit zero-length vector errors"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'test_point3d_normalize_reports_zero_length_vector|test_point2d_normalize_reports_zero_length_vector' `
        -Message "vector normalization must keep regression coverage for zero-length failures"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo vector contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo vector contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
