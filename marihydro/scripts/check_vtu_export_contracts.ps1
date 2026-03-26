#!/usr/bin/env pwsh
#
# Blocking guard for explicit VTU exporter state access semantics.
# Public VTU state accessors must return explicit errors instead of
# collapsing missing fields or invalid indices into Option::None.
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
    Write-Host "=== Checking VTU export contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'MissingScalarField|ScalarIndexOutOfBounds' `
        -Message "VTU exporter must expose explicit scalar access errors"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'fn scalar\(&self, name: &str, _?idx: usize\) -> Result<f64, VtuError>' `
        -Message "VtuState::scalar must return Result<f64, VtuError>"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'fn scalar\(&self, .*?\) -> Option<f64>|return None;|\.is_none\(\)' `
        -Message "VTU scalar access must not use Option-based failure semantics"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'values\.push\(state\.scalar\(name, i\)\?\);' `
        -Message "VTU cell-data export must propagate scalar access failures explicitly"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'ScalarIndexOutOfBounds' `
        -Message "VTU scalar implementations must report index out-of-bounds explicitly"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/exporters/vtu.rs" `
        -Pattern 'test_state_with_scalars|test_state_with_scalars_reports_out_of_bounds' `
        -Message "VTU tests must cover missing scalar fields and out-of-bounds access"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/lib.rs" `
        -Pattern 'pub use exporters::\{VtuError, VtuExporter, VtuMesh, VtuState\};' `
        -Message "mh_io root must re-export VtuError with the public VTU state trait"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] VTU export contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] VTU export contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
