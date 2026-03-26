#!/usr/bin/env pwsh
#
# Blocking guard for explicit geodesic failure semantics.
# Ensures Vincenty non-convergence is surfaced as an explicit GeoError
# instead of being collapsed into Option::None or other sentinels.
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
    Write-Host "=== Checking geo geodesic contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $geometryPath = "crates/mh_geo/src/geometry.rs"

    Check-PatternAbsent `
        -RelativePath $geometryPath `
        -Pattern 'pub fn vincenty_distance_to\(&self, other: &Self\) -> Option<f64>|pub fn vincenty_distance\(&self, other: &Self, ellipsoid: &Ellipsoid\) -> Option<f64>' `
        -Message "Vincenty APIs must not expose Option-based failure semantics"

    Check-PatternAbsent `
        -RelativePath $geometryPath `
        -Pattern 'return Some\(0\.0\)|Some\(s\)' `
        -Message "Vincenty implementation must not encode success or failure through Option sentinels"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'pub fn vincenty_distance_to\(&self, other: &Self\) -> GeoResult<f64>|pub fn vincenty_distance\(&self, other: &Self, ellipsoid: &Ellipsoid\) -> GeoResult<f64>' `
        -Message "Vincenty APIs must return GeoResult<f64>"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'GeoError::vincenty_not_converged\(\)' `
        -Message "Vincenty implementation must emit GeoError::vincenty_not_converged() on non-convergence"

    Check-PatternPresent `
        -RelativePath $geometryPath `
        -Pattern 'test_vincenty_antipodal_reports_not_converged|test_vincenty_same_point|test_vincenty_distance' `
        -Message "Vincenty implementation must keep explicit convergence and failure regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo geodesic contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo geodesic contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
