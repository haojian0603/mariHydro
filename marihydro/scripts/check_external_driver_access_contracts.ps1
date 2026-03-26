#!/usr/bin/env pwsh
#
# Blocking guard for explicit external-driver index access semantics.
# Public raster/NetCDF accessors must return explicit errors instead of
# collapsing invalid access into Option::None.
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

function Check-PatternPresent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

function Check-PatternAbsent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required file: $RelativePath"
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

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking external driver access contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/gdal/error.rs" `
        -Pattern 'PixelOutOfBounds' `
        -Message "GDAL error surface must expose PixelOutOfBounds"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/gdal/error.rs" `
        -Pattern 'NoDataPixel' `
        -Message "GDAL error surface must expose NoDataPixel"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'pub fn get\(&self, x: usize, y: usize\) -> Result<f64, GdalError>' `
        -Message "RasterBand::get must return Result<f64, GdalError>"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'pub fn interpolate\(&self, x: f64, y: f64\) -> Result<f64, GdalError>' `
        -Message "RasterBand::interpolate must return Result<f64, GdalError>"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'pub fn get\(&self, x: usize, y: usize\) -> Option<f64>|pub fn interpolate\(&self, x: f64, y: f64\) -> Option<f64>|return None;|Some\(v0 \* \(1\.0 - fy\) \+ v1 \* fy\)' `
        -Message "GDAL raster accessors must not collapse invalid access into Option-based fallbacks"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/error.rs" `
        -Pattern 'InvalidIndices' `
        -Message "NetCDF error surface must expose InvalidIndices"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'fn linear_index\(&self, indices: &\[usize\]\) -> Result<usize, NetCdfError>' `
        -Message "Variable::linear_index must return Result<usize, NetCdfError>"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'pub fn get\(&self, indices: &\[usize\]\) -> Result<f64, NetCdfError>' `
        -Message "Variable::get must return Result<f64, NetCdfError>"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'fn linear_index\(&self, indices: &\[usize\]\) -> Option<usize>|pub fn get\(&self, indices: &\[usize\]\) -> Option<f64>|return None;|Some\(self\.data\[idx\]\)' `
        -Message "NetCDF variable accessors must not collapse invalid indices into Option-based fallbacks"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern '\.map_err\(\|err\|' `
        -Message "tide sampling must propagate driver index failures as explicit format errors"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'value\.ok_or_else\(' `
        -Message "tide sampling must not translate driver access into synthetic Option-based misses"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'test_raster_band_get_reports_out_of_bounds|test_raster_band_get_reports_nodata|test_raster_band_interpolate_reports_out_of_bounds' `
        -Message "GDAL driver tests must cover out-of-bounds and NoData access"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'test_variable_get|InvalidIndices' `
        -Message "NetCDF driver tests must cover invalid index access"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] external driver access contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] external driver access contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
