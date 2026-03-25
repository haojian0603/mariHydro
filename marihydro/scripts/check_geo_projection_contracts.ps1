#!/usr/bin/env pwsh
#
# Blocking guard for geographic projection helper semantics.
# Projection-derived helpers must fail explicitly when the target CRS does not
# define the requested quantity.
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
        Add-Failure "missing required geo projection file: $RelativePath"
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
        Add-Failure "missing required geo projection file: $RelativePath"
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
    Write-Host "=== Checking geo projection contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsentRaw `
        -RelativePath "crates/mh_geo/src/transform.rs" `
        -Pattern '(?s)if\s+self\.target_crs\.is_geographic\(\)\s*\{.*?return\s+Ok\(0\.0\);' `
        -Message "geo convergence helpers must not fabricate zero angle for geographic target CRS"

    Check-PatternPresent `
        -RelativePath "crates/mh_geo/src/transform.rs" `
        -Pattern 'convergence angle is only defined for projected target CRS' `
        -Message "geo convergence helpers must emit an explicit projected-target error"

    Check-PatternPresent `
        -RelativePath "crates/mh_geo/src/transform.rs" `
        -Pattern 'test_geographic_target_convergence_angle_requires_projected_target|test_geographic_target_rotate_vector_requires_projected_target' `
        -Message "geo convergence helpers must keep regression coverage for geographic target rejection"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo projection contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo projection contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
