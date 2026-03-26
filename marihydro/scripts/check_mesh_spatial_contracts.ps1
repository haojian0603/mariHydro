#!/usr/bin/env pwsh
#
# Blocking guard for mesh circle-query semantics.
# Mesh circle queries must use true polygon-circle intersection semantics,
# and invalid circle definitions must fail explicitly.
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
        Add-Failure "missing required mesh spatial file: $RelativePath"
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
        Add-Failure "missing required mesh spatial file: $RelativePath"
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
        Add-Failure "missing required mesh spatial file: $RelativePath"
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
    Write-Host "=== Checking mesh spatial contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $spatialPath = "crates/mh_mesh/src/spatial_index.rs"
    $errorPath = "crates/mh_mesh/src/error.rs"

    Check-PatternPresentRaw `
        -RelativePath $spatialPath `
        -Pattern 'pub fn locate_in_circle\(\s*&self,\s*center_x: f64,\s*center_y: f64,\s*radius: f64,\s*\)\s*->\s*MeshResult<Vec<usize>>' `
        -Message "MeshSpatialIndex::locate_in_circle must return MeshResult<Vec<usize>>"

    Check-PatternPresent `
        -RelativePath $spatialPath `
        -Pattern 'circle center and radius must be finite, and radius must be non-negative' `
        -Message "MeshSpatialIndex::locate_in_circle must reject invalid circle definitions explicitly"

    Check-PatternPresent `
        -RelativePath $spatialPath `
        -Pattern 'polygon_intersects_circle|point_in_polygon\(center_x, center_y, vertices\)|point_on_segment_tol\(center_x, center_y, a, b, radius\)' `
        -Message "MeshSpatialIndex::locate_in_circle must use polygon-circle intersection checks"

    Check-PatternPresent `
        -RelativePath $spatialPath `
        -Pattern 'hits\.sort_unstable\(\)' `
        -Message "MeshSpatialIndex::locate_in_circle must keep deterministic hit ordering"

    Check-PatternAbsentRaw `
        -RelativePath $spatialPath `
        -Pattern 'dcx \* dcx \+ dcy \* dcy <= r2' `
        -Message "MeshSpatialIndex::locate_in_circle must not fall back to bbox-center heuristics or weak tests"

    Check-PatternPresent `
        -RelativePath $spatialPath `
        -Pattern 'test_locate_in_circle_detects_edge_only_intersection|test_locate_in_circle_rejects_invalid_radius|test_locate_in_circle_rejects_non_finite_center' `
        -Message "Mesh circle query must keep regression coverage for edge intersections and invalid-circle failures"

    Check-PatternPresent `
        -RelativePath $errorPath `
        -Pattern 'SpatialQueryError|pub fn spatial_query_error|test_spatial_query_error_conversion_to_runtime' `
        -Message "mh_mesh error layer must preserve explicit spatial-query errors and runtime conversion coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] mesh spatial contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] mesh spatial contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
