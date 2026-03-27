#!/usr/bin/env pwsh
#
# Blocking guard for explicit central meridian semantics.
# Invalid UTM / Gauss-Kruger zones must not fabricate central meridian values
# through raw formulas or Option-based public surfaces.
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
    Write-Host "=== Checking geo central meridian contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $utmPath = "crates/mh_geo/src/projection/utm.rs"
    $gkPath = "crates/mh_geo/src/projection/gauss_kruger.rs"
    $projectionPath = "crates/mh_geo/src/projection/mod.rs"
    $crsPath = "crates/mh_geo/src/crs.rs"

    Check-PatternAbsent `
        -RelativePath $utmPath `
        -Pattern 'pub fn utm_central_meridian\(zone: u8\) -> f64' `
        -Message "UTM central meridian helper must not return raw f64 without validation"

    Check-PatternPresent `
        -RelativePath $utmPath `
        -Pattern 'pub fn utm_central_meridian\(zone: u8\) -> GeoResult<f64>' `
        -Message "UTM central meridian helper must return GeoResult<f64>"

    Check-PatternPresent `
        -RelativePath $utmPath `
        -Pattern 'GeoError::check_utm_zone\(zone\)\?' `
        -Message "UTM central meridian helper must validate the zone explicitly"

    Check-PatternPresent `
        -RelativePath $utmPath `
        -Pattern 'test_utm_central_meridian_rejects_invalid_zone' `
        -Message "UTM central meridian helper must keep invalid-zone regression coverage"

    Check-PatternAbsent `
        -RelativePath $gkPath `
        -Pattern 'pub fn gk3_central_meridian\(zone: u8\) -> f64|pub fn gk6_central_meridian\(zone: u8\) -> f64' `
        -Message "Gauss-Kruger central meridian helpers must not return raw f64 without validation"

    Check-PatternPresent `
        -RelativePath $gkPath `
        -Pattern 'pub fn gk3_central_meridian\(zone: u8\) -> GeoResult<f64>' `
        -Message "GK3 central meridian helper must return GeoResult<f64>"

    Check-PatternPresent `
        -RelativePath $gkPath `
        -Pattern 'pub fn gk6_central_meridian\(zone: u8\) -> GeoResult<f64>' `
        -Message "GK6 central meridian helper must return GeoResult<f64>"

    Check-PatternPresent `
        -RelativePath $gkPath `
        -Pattern 'GeoError::check_gauss_kruger_zone\(zone, 25, 45\)\?' `
        -Message "GK3 central meridian helper must validate the zone range explicitly"

    Check-PatternPresent `
        -RelativePath $gkPath `
        -Pattern 'GeoError::check_gauss_kruger_zone\(zone, 13, 23\)\?' `
        -Message "GK6 central meridian helper must validate the zone range explicitly"

    Check-PatternPresent `
        -RelativePath $gkPath `
        -Pattern 'test_gauss_kruger_central_meridian_rejects_invalid_zone' `
        -Message "Gauss-Kruger central meridian helpers must keep invalid-zone regression coverage"

    Check-PatternAbsent `
        -RelativePath $projectionPath `
        -Pattern 'pub fn central_meridian\(&self\) -> Option<f64>' `
        -Message "ProjectionType::central_meridian must not hide invalid zones behind Option<f64>"

    Check-PatternPresent `
        -RelativePath $projectionPath `
        -Pattern 'pub fn central_meridian\(&self\) -> GeoResult<Option<f64>>' `
        -Message "ProjectionType::central_meridian must return GeoResult<Option<f64>>"

    Check-PatternPresent `
        -RelativePath $projectionPath `
        -Pattern 'utm_central_meridian\(\*zone\)\.map\(Some\)' `
        -Message "ProjectionType::central_meridian must delegate UTM zones to the validated helper"

    Check-PatternPresent `
        -RelativePath $projectionPath `
        -Pattern 'gk3_central_meridian\(\*zone\)\.map\(Some\)' `
        -Message "ProjectionType::central_meridian must delegate GK3 zones to the validated helper"

    Check-PatternPresent `
        -RelativePath $projectionPath `
        -Pattern 'gk6_central_meridian\(\*zone\)\.map\(Some\)' `
        -Message "ProjectionType::central_meridian must delegate GK6 zones to the validated helper"

    Check-PatternPresent `
        -RelativePath $projectionPath `
        -Pattern 'test_projection_type_central_meridian_reports_invalid_zone' `
        -Message "ProjectionType::central_meridian must keep invalid-zone regression coverage"

    Check-PatternAbsent `
        -RelativePath $crsPath `
        -Pattern 'pub fn central_meridian\(&self\) -> Option<f64>' `
        -Message "Crs::central_meridian must not expose Option<f64> fallback semantics"

    Check-PatternPresent `
        -RelativePath $crsPath `
        -Pattern 'pub fn central_meridian\(&self\) -> MhResult<Option<f64>>' `
        -Message "Crs::central_meridian must return MhResult<Option<f64>>"

    Check-PatternPresent `
        -RelativePath $crsPath `
        -Pattern 'projection\.central_meridian\(\)\.map_err\(MhError::from\)' `
        -Message "Crs::central_meridian must propagate projection zone failures explicitly"

    Check-PatternPresent `
        -RelativePath $crsPath `
        -Pattern 'test_crs_central_meridian_reports_invalid_projection_zone' `
        -Message "Crs::central_meridian must keep invalid-zone regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo central meridian contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo central meridian contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
