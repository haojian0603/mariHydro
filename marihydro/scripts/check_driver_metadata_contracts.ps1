#!/usr/bin/env pwsh
#
# Blocking guard for external driver metadata truthfulness.
# Optional metadata may be absent, but read/decode failures must stay explicit.
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

function Check-PatternPresentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required driver metadata contract file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

function Check-PatternAbsentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required driver metadata contract file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Add-Failure $Message
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking driver metadata contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $driversModPath = "crates/mh_io/src/drivers/mod.rs"
    $gdalPath = "crates/mh_io/src/drivers/gdal/driver.rs"
    $netcdfPath = "crates/mh_io/src/drivers/netcdf/driver.rs"

    Check-PatternPresentRaw `
        -RelativePath $driversModPath `
        -Pattern 'DRIVER_METADATA_SCOPE:' `
        -Message "driver module docs must declare explicit metadata truthfulness semantics"

    Check-PatternAbsentRaw `
        -RelativePath $gdalPath `
        -Pattern 'dataset\.projection\(\)\.ok\(\)' `
        -Message "GDAL native driver must not collapse projection query failures into missing metadata"

    Check-PatternAbsentRaw `
        -RelativePath $gdalPath `
        -Pattern 'dataset\.rasterband\(1\)\.ok\(\)' `
        -Message "GDAL native driver must not collapse first-band metadata failures into missing NoData"

    Check-PatternPresentRaw `
        -RelativePath $gdalPath `
        -Pattern 'fn projection_metadata_result\(|fn first_band_nodata_metadata\(' `
        -Message "GDAL native driver must use explicit metadata resolution helpers"

    Check-PatternAbsentRaw `
        -RelativePath $netcdfPath `
        -Pattern 'attribute\("standard_name"\)\s*\.and_then\(\|a\| a\.value\(\)\.ok\(\)\)|attribute\("long_name"\)\s*\.and_then\(\|a\| a\.value\(\)\.ok\(\)\)|attribute\("units"\)\s*\.and_then\(\|a\| a\.value\(\)\.ok\(\)\)' `
        -Message "NetCDF native driver must not collapse attribute decode failures into missing metadata"

    Check-PatternAbsentRaw `
        -RelativePath $netcdfPath `
        -Pattern 'other => Ok\(format!\("\{:\?\}", other\)\)' `
        -Message "NetCDF global attribute access must not stringify wrong-typed values as fake success"

    Check-PatternPresentRaw `
        -RelativePath $netcdfPath `
        -Pattern 'fn resolve_optional_string_metadata\(|fn resolve_required_string_metadata\(' `
        -Message "NetCDF native driver must use explicit string-metadata resolution helpers"

    Check-PatternPresentRaw `
        -RelativePath $gdalPath `
        -Pattern 'test_projection_metadata_result_rejects_projection_query_failure|test_first_band_nodata_metadata_rejects_first_band_failure|test_first_band_nodata_metadata_rejects_non_finite_value' `
        -Message "GDAL native driver must keep regression coverage for metadata read failures"

    Check-PatternPresentRaw `
        -RelativePath $netcdfPath `
        -Pattern 'test_resolve_optional_string_metadata_rejects_wrong_type|test_resolve_required_string_metadata_rejects_missing|test_resolve_required_string_metadata_rejects_read_failure' `
        -Message "NetCDF native driver must keep regression coverage for string metadata failures"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] driver metadata contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] driver metadata contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
