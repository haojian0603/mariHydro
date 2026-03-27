#!/usr/bin/env pwsh
#
# Blocking guard for explicit ellipsoid EPSG failure semantics.
# Public ellipsoid EPSG helpers must not hide unsupported codes behind Option.
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
    Write-Host "=== Checking geo ellipsoid EPSG contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $ellipsoidPath = "crates/mh_geo/src/ellipsoid.rs"
    $crsPath = "crates/mh_geo/src/crs.rs"

    Check-PatternAbsent `
        -RelativePath $ellipsoidPath `
        -Pattern 'pub fn from_epsg\(code: u32\) -> Option<Self>' `
        -Message "Ellipsoid::from_epsg must not expose Option-based failure semantics"

    Check-PatternPresent `
        -RelativePath $ellipsoidPath `
        -Pattern 'pub fn from_epsg\(code: u32\) -> GeoResult<Self>' `
        -Message "Ellipsoid::from_epsg must return GeoResult<Self>"

    Check-PatternPresent `
        -RelativePath $ellipsoidPath `
        -Pattern 'GeoError::unsupported_epsg\(' `
        -Message "Ellipsoid::from_epsg must raise an explicit unsupported EPSG error"

    Check-PatternPresent `
        -RelativePath $ellipsoidPath `
        -Pattern 'test_from_epsg' `
        -Message "Ellipsoid::from_epsg must keep EPSG regression coverage"

    Check-PatternPresent `
        -RelativePath $crsPath `
        -Pattern 'if let Ok\(ellipsoid\) = Ellipsoid::from_epsg\(code\)' `
        -Message "CRS ellipsoid detection must consume the explicit Ellipsoid::from_epsg result"

    Check-PatternAbsent `
        -RelativePath $crsPath `
        -Pattern 'if let Some\(ellipsoid\) = Ellipsoid::from_epsg\(code\)' `
        -Message "CRS ellipsoid detection must not consume Ellipsoid::from_epsg as Option"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo ellipsoid EPSG contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo ellipsoid EPSG contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
