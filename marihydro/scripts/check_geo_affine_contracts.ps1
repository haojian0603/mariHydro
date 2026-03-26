#!/usr/bin/env pwsh

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
    Write-Host "=== Checking geo affine contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $transformPath = "crates/mh_geo/src/transform.rs"

    Check-PatternAbsent `
        -RelativePath $transformPath `
        -Pattern 'pub fn inverse\(&self\) -> Option<Self>|pub fn apply_inverse\(&self, x: f64, y: f64\) -> Option<\(f64, f64\)>' `
        -Message "Affine inverse APIs must not expose Option-based failure semantics"

    Check-PatternAbsent `
        -RelativePath $transformPath `
        -Pattern 'return None;' `
        -Message "Affine inverse implementation must not collapse singular transforms into None"

    Check-PatternPresent `
        -RelativePath $transformPath `
        -Pattern 'pub fn inverse\(&self\) -> MhResult<Self>|pub fn apply_inverse\(&self, x: f64, y: f64\) -> MhResult<\(f64, f64\)>' `
        -Message "Affine inverse APIs must return MhResult with explicit failure semantics"

    Check-PatternPresent `
        -RelativePath $transformPath `
        -Pattern 'GeoError::SingularTransform' `
        -Message "Affine inverse must surface GeoError::SingularTransform on singular matrices"

    Check-PatternPresent `
        -RelativePath $transformPath `
        -Pattern 'test_affine_inverse_reports_singular_transform|test_affine_transform' `
        -Message "Affine inverse implementation must keep explicit singular-matrix regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo affine contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo affine contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
