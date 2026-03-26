#!/usr/bin/env pwsh
#
# Blocking guard for spatial index radius-query semantics.
# Radius queries must use explicit candidate envelopes plus exact distance
# filtering, and they must reject invalid radii explicitly.
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
        Add-Failure "missing required spatial index file: $RelativePath"
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
        Add-Failure "missing required spatial index file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

function Check-PatternPresentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required spatial index file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking spatial index contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $relativePath = "crates/mh_geo/src/spatial_index.rs"

    Check-PatternPresentRaw `
        -RelativePath $relativePath `
        -Pattern 'pub fn query_within_distance\(\s*&self,\s*point: &Point2D,\s*distance: f64,\s*\)\s*->\s*GeoResult<Vec<\(&Point2D, &T\)>>' `
        -Message "SpatialIndex::query_within_distance must return GeoResult<Vec<(&Point2D, &T)>>"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'distance must be finite and non-negative' `
        -Message "SpatialIndex::query_within_distance must reject invalid radii explicitly"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'AABB::from_corners\(' `
        -Message "SpatialIndex::query_within_distance must build an explicit candidate envelope"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'point\.x - distance|point\.y - distance|point\.x \+ distance|point\.y \+ distance' `
        -Message "SpatialIndex::query_within_distance candidate envelope must be centered on the query point and radius"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'locate_in_envelope\(&envelope\)|distance_2 <= dist_squared|results\.sort_by\(\|left, right\| left\.0\.total_cmp\(&right\.0\)\)' `
        -Message "SpatialIndex::query_within_distance must filter by exact distance and keep deterministic ordering"

    Check-PatternAbsentRaw `
        -RelativePath $relativePath `
        -Pattern '(?s)pub fn query_within_distance\(.*?nearest_neighbor_iter\(&\[point\.x, point\.y\]\).*?take_while\(\|entry\|' `
        -Message "SpatialIndex::query_within_distance must not rely on nearest-neighbor iteration plus take_while"

    Check-PatternPresent `
        -RelativePath $relativePath `
        -Pattern 'test_spatial_index_within_distance_sorts_by_exact_distance|test_spatial_index_within_distance_filters_bounding_box_false_positive|test_spatial_index_within_distance_rejects_negative_radius' `
        -Message "SpatialIndex radius query must keep regression coverage for exact filtering, ordering, and invalid radii"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] spatial index contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] spatial index contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
