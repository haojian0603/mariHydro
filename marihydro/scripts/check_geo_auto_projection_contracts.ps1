#!/usr/bin/env pwsh
#
# Guard explicit automatic projection zone selection semantics.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ProjectionMod = Join-Path $ProjectRoot "crates/mh_geo/src/projection/mod.rs"
$GaussKruger = Join-Path $ProjectRoot "crates/mh_geo/src/projection/gauss_kruger.rs"
$Utm = Join-Path $ProjectRoot "crates/mh_geo/src/projection/utm.rs"
$Errors = @()

function Assert-PatternAbsent {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )

    if (Select-String -Path $Path -Pattern $Pattern -Quiet) {
        Write-Host "[FAIL] $Message" -ForegroundColor Red
        $script:Errors += $Message
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

function Assert-PatternPresent {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )

    if (Select-String -Path $Path -Pattern $Pattern -Quiet) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $Message" -ForegroundColor Red
        $script:Errors += $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking geo auto projection contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Assert-PatternPresent -Path $ProjectionMod -Pattern 'pub fn auto_utm\(lon: f64, lat: f64\) -> GeoResult<Self>' -Message "ProjectionType::auto_utm must return GeoResult<Self>"
    Assert-PatternPresent -Path $ProjectionMod -Pattern 'pub fn auto_gk3\(lon: f64\) -> GeoResult<Self>' -Message "ProjectionType::auto_gk3 must return GeoResult<Self>"
    Assert-PatternPresent -Path $ProjectionMod -Pattern 'pub fn auto_gk6\(lon: f64\) -> GeoResult<Self>' -Message "ProjectionType::auto_gk6 must return GeoResult<Self>"
    Assert-PatternAbsent -Path $ProjectionMod -Pattern 'zone\s*=\s*zone\.clamp\(1,\s*60\)|zone\s*=\s*zone\.clamp\(13,\s*23\)' -Message "ProjectionType auto helpers must not clamp fabricated zones"
    Assert-PatternPresent -Path $ProjectionMod -Pattern 'test_auto_projection_helpers_reject_invalid_inputs' -Message "ProjectionType auto helpers must keep invalid-input regression coverage"
    Assert-PatternPresent -Path $ProjectionMod -Pattern 'auto_utm_zone\(lon\)\.map_err\(' -Message "wgs84_to_auto_utm must reuse explicit auto_utm_zone error semantics"

    Assert-PatternPresent -Path $GaussKruger -Pattern 'pub fn auto_gk3_zone\(lon: f64\) -> GeoResult<u8>' -Message "auto_gk3_zone must return GeoResult<u8>"
    Assert-PatternPresent -Path $GaussKruger -Pattern 'pub fn auto_gk6_zone\(lon: f64\) -> GeoResult<u8>' -Message "auto_gk6_zone must return GeoResult<u8>"
    Assert-PatternAbsent -Path $GaussKruger -Pattern 'return 39;|zone\.clamp\(25,\s*45\)|zone\.clamp\(13,\s*23\)' -Message "Gauss-Kruger auto zone helpers must not fabricate default or clamped zones"
    Assert-PatternPresent -Path $GaussKruger -Pattern 'test_gk_zone_functions_reject_invalid_longitude' -Message "Gauss-Kruger auto zone helpers must keep invalid-longitude regression coverage"

    Assert-PatternPresent -Path $Utm -Pattern 'pub fn auto_utm_zone\(lon: f64\) -> GeoResult<u8>' -Message "auto_utm_zone must return GeoResult<u8>"
    Assert-PatternAbsent -Path $Utm -Pattern 'zone\.clamp\(1,\s*60\)' -Message "auto_utm_zone must not clamp fabricated zones"
    Assert-PatternPresent -Path $Utm -Pattern 'test_auto_utm_zone_rejects_invalid_longitude' -Message "auto_utm_zone must keep invalid-longitude regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo auto projection contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo auto projection contracts failed" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
