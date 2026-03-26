#!/usr/bin/env pwsh
#
# Blocking guard for snapshot boundary metadata truthfulness.
# Boundary IDs in snapshots must fail explicitly when missing instead of being
# serialized through sentinel values.
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
        Add-Failure "missing required snapshot boundary contract file: $RelativePath"
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
        Add-Failure "missing required snapshot boundary contract file: $RelativePath"
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
    Write-Host "=== Checking snapshot boundary contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/snapshot.rs" `
        -Pattern 'unwrap_or\(u32::MAX\)|boundary_ids:\s*Option<Vec<u32>>.*u32::MAX' `
        -Message "snapshot boundary IDs must not fall back to u32::MAX sentinels"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/snapshot.rs" `
        -Pattern 'pub fn from_frozen<|Result<Self, String>|face_boundary_id|ok_or_else\(' `
        -Message "snapshot construction from frozen meshes must fail explicitly on missing boundary IDs"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/snapshot.rs" `
        -Pattern 'test_mesh_snapshot_from_frozen_rejects_missing_boundary_id|test_mesh_snapshot_from_frozen_preserves_boundary_id' `
        -Message "snapshot boundary truthfulness tests must remain in place"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/pipeline.rs" `
        -Pattern 'test_write_vtu_ascii_rejects_missing_boundary_ids|mesh\.validate\(' `
        -Message "VTU ASCII pipeline must reject snapshots with missing boundary IDs"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/vtu/binary.rs" `
        -Pattern 'test_write_vtu_binary_rejects_missing_boundary_ids|io::ErrorKind::InvalidData|mesh\.validate\(' `
        -Message "VTU binary writer must reject snapshots with missing boundary IDs"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] snapshot boundary contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] snapshot boundary contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
