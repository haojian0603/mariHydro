#!/usr/bin/env pwsh
#
# Blocking guard for source-term public API truthfulness.
# Prevents silent index fallback, unused configuration parameters,
# and misleading public signatures in hydraulic source modules.
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
        Add-Failure "missing required source file: $RelativePath"
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
        Add-Failure "missing required source file: $RelativePath"
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
    Write-Host "=== Checking source API semantics contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'pub fn effective_drag\(&self,\s*water_depth: f64,\s*velocity: f64\)' `
        -Message "vegetation drag helper must not expose an unused velocity parameter"

    Check-PatternAbsent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'pub rho_water: f64' `
        -Message "vegetation config must not expose unused rho_water parameters"

    Check-PatternAbsent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'if cell < self\.vegetation\.len\(\)' `
        -Message "vegetation config must not silently ignore out-of-range set_vegetation calls"

    Check-PatternAbsent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'unwrap_or\(VegetationType::None\)|unwrap_or\(B::Scalar::ONE\)' `
        -Message "vegetation source must not collapse missing cells to no-op vegetation or unit decay"

    Check-PatternPresent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'expect\("vegetation cell index must be within configured domain"\)|expect\("vegetation configuration must cover every computed cell"\)|expect\("vegetation decay factor requires a valid cell index"\)' `
        -Message "vegetation source must keep explicit invariant failures for invalid indices"

    Check-PatternPresent `
        -RelativePath "crates/mh_physics/src/sources/vegetation.rs" `
        -Pattern 'test_set_vegetation_rejects_out_of_range_index|test_get_decay_factor_rejects_out_of_range_index' `
        -Message "vegetation source must keep regression tests for explicit index failures"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] source API semantics contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] source API semantics contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
