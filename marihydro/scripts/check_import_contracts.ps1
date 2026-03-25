#!/usr/bin/env pwsh
#
# Blocking guard for import-layer structural truthfulness.
# Ensures vector importers reject incomplete geometry structure instead of
# fabricating empty shells or silently collapsing malformed layouts.
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

function Check-TagSet {
    param(
        [string]$RelativePath,
        [string]$Label
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required import file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match 'IO_SOURCE:\s*\S.{20,}') {
        Write-Host "[OK] $Label source tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SOURCE describing the real upstream geometry contract"
    }

    if ($content -match 'IO_SCOPE:\s*\S.{20,}') {
        Write-Host "[OK] $Label scope tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SCOPE describing explicit failure behavior for invalid structure"
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
        Add-Failure "missing required import file: $RelativePath"
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
        Add-Failure "missing required import file: $RelativePath"
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
    Write-Host "=== Checking import contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-TagSet -RelativePath "crates/mh_io/src/import/mod.rs" -Label "import module"
    Check-TagSet -RelativePath "crates/mh_io/src/import/geojson.rs" -Label "GeoJSON importer"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'rings\.first\(\)\.cloned\(\)\.unwrap_or_default\(\)' `
        -Message "GeoJSON Polygon and MultiPolygon import must not synthesize empty exterior rings"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'InvalidStructure\(String\)' `
        -Message "GeoJSON importer must expose an explicit InvalidStructure error for malformed geometry"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'must contain at least one linear ring|must contain at least 4 positions|must be closed' `
        -Message "GeoJSON importer must reject missing, underspecified, or open linear rings explicitly"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] import contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] import contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
