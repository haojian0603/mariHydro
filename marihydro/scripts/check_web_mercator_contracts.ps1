#!/usr/bin/env pwsh
#
# Blocking guard for Web Mercator public helper semantics.
# Web Mercator helpers must reject out-of-domain inputs explicitly instead of
# clamping latitude, hiding projection extent violations, or using internal
# expect()-based synthetic success paths.
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

function Check-PatternAbsentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required Web Mercator file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Add-Failure $Message
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
        Add-Failure "missing required Web Mercator file: $RelativePath"
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
    Write-Host "=== Checking Web Mercator contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $relativePath = "crates/mh_geo/src/projection/web_mercator.rs"

    Check-PatternAbsentRaw `
        -RelativePath $relativePath `
        -Pattern 'lat\.clamp\(\s*-WEB_MERCATOR_MAX_LAT\s*,\s*WEB_MERCATOR_MAX_LAT\s*\)' `
        -Message "Web Mercator forward helper must not clamp latitude to fabricate success"

    Check-PatternAbsentRaw `
        -RelativePath $relativePath `
        -Pattern 'expect\(\"tile_to_lonlat returns coordinates inside Web Mercator domain\"\)' `
        -Message "Web Mercator tile bbox helper must not rely on internal expect() for domain assumptions"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'out of Web Mercator domain|out of Web Mercator extent|must be finite and greater than zero' `
        -Message "Web Mercator helpers must emit explicit domain and extent failures"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'pub fn web_mercator_resolution\(lat: f64, zoom: u8, tile_size: u32\) -> MhResult<f64>' `
        -Message "Web Mercator resolution helper must return MhResult<f64>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'pub fn web_mercator_scale\(lat: f64, zoom: u8, dpi: f64\) -> MhResult<f64>' `
        -Message "Web Mercator scale helper must return MhResult<f64>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'pub fn lonlat_to_tile\(lon: f64, lat: f64, zoom: u8\) -> MhResult<\(u32, u32\)>' `
        -Message "Web Mercator tile conversion helper must return MhResult<(u32, u32)>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'pub fn tile_to_lonlat\(x: u32, y: u32, zoom: u8\) -> MhResult<\(f64, f64\)>' `
        -Message "Web Mercator tile origin helper must return MhResult<(f64, f64)>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'pub fn tile_to_bbox\(x: u32, y: u32, zoom: u8\) -> MhResult<\(f64, f64, f64, f64\)>' `
        -Message "Web Mercator tile bbox helper must return MhResult<(f64, f64, f64, f64)>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'validate_tile_index|tile_grid_width|zoom .* too large for tile indexing|tile corner x .* out of range|tile y .* out of range' `
        -Message "Web Mercator tile helpers must keep explicit tile-index and zoom-domain checks"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'test_web_mercator_rejects_out_of_range_latitude|test_web_mercator_to_geographic_rejects_out_of_extent|test_web_mercator_resolution_rejects_invalid_latitude|test_web_mercator_scale_rejects_nonpositive_dpi|test_tile_to_lonlat_rejects_out_of_range_tile_index|test_tile_to_bbox_rejects_out_of_range_tile_index|test_lonlat_to_tile_rejects_antimeridian_open_boundary|test_tile_helpers_reject_unsupported_zoom' `
        -Message "Web Mercator helpers must keep regression coverage for domain and tile-index failures"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] Web Mercator contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] Web Mercator contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
