#!/usr/bin/env pwsh
#
# Blocking guard for structured mesh bed-elevation semantics.
# Structured meshes must provide explicit bed data before freeze,
# and flat-bed workflows must opt in explicitly.
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
        Add-Failure "missing required structured-mesh file: $RelativePath"
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
        Add-Failure "missing required structured-mesh file: $RelativePath"
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
        Add-Failure "missing required structured-mesh file: $RelativePath"
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
    Write-Host "=== Checking structured mesh contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $structuredPath = "crates/mh_mesh/src/structured.rs"
    $errorPath = "crates/mh_mesh/src/error.rs"
    $runnerPath = "crates/mh_workflow/src/runner.rs"
    $adapterPath = "crates/mh_physics/src/adapter.rs"
    $dambreakPath = "crates/mh_physics/tests/dambreak.rs"

    Check-PatternPresentRaw `
        -RelativePath $structuredPath `
        -Pattern 'pub fn set_bed_elevation\(&mut self, elevation: Vec<f64>\) -> MeshResult<\(\)>' `
        -Message "StructuredMesh::set_bed_elevation must fail explicitly instead of asserting"

    Check-PatternPresentRaw `
        -RelativePath $structuredPath `
        -Pattern 'pub fn set_uniform_bed_elevation\(&mut self, elevation: f64\) -> MeshResult<\(\)>' `
        -Message "StructuredMesh must expose an explicit flat-bed API"

    Check-PatternPresent `
        -RelativePath $structuredPath `
        -Pattern 'MissingRequiredData|missing_required_data|set_uniform_bed_elevation' `
        -Message "StructuredMesh::freeze must require explicit bed data"

    Check-PatternAbsentRaw `
        -RelativePath $structuredPath `
        -Pattern 'bed_elevation\(i \+ j \* nx\)\.unwrap_or\(0\.0\)' `
        -Message "StructuredMesh::freeze must not synthesize flat bed from missing data"

    Check-PatternPresent `
        -RelativePath $structuredPath `
        -Pattern 'test_freeze_requires_explicit_bed_elevation|test_set_uniform_bed_elevation_allows_flat_bed_freeze|test_set_bed_elevation_rejects_non_finite_value' `
        -Message "structured mesh regressions must cover missing bed, explicit flat bed, and invalid bed values"

    Check-PatternPresent `
        -RelativePath $errorPath `
        -Pattern 'MissingRequiredData|pub fn missing_required_data|test_missing_required_data_conversion_to_runtime' `
        -Message "mh_mesh error layer must preserve explicit missing-data errors"

    Check-PatternPresent `
        -RelativePath $runnerPath `
        -Pattern 'uniform_bed_elevation|必须显式提供 bed_elevation 或 uniform_bed_elevation|不能同时提供' `
        -Message "workflow structured mesh loader must require explicit bed semantics"

    Check-PatternPresent `
        -RelativePath $adapterPath `
        -Pattern 'test_from_structured_requires_explicit_bed_elevation|test_from_structured_accepts_explicit_flat_bed' `
        -Message "PhysicsMesh adapter must keep structured-bed regression coverage"

    Check-PatternPresent `
        -RelativePath $dambreakPath `
        -Pattern 'set_uniform_bed_elevation\(0\.0\)' `
        -Message "structured dambreak test must opt in explicitly to flat bed"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] structured mesh contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] structured mesh contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
