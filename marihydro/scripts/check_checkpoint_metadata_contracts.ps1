#!/usr/bin/env pwsh
#
# Blocking guard for checkpoint metadata truthfulness.
# Checkpoint hashes and directory scans must not use sentinel values or
# silently skip invalid checkpoint files.
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

    $AbsolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $AbsolutePath)) {
        Add-Failure "missing required checkpoint file: $RelativePath"
        return
    }

    $Matches = Select-String -Path $AbsolutePath -Pattern $Pattern -CaseSensitive
    if ($Matches) {
        Add-Failure $Message
        $Matches | Select-Object -First 5 | ForEach-Object {
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

    $AbsolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $AbsolutePath)) {
        Add-Failure "missing required checkpoint file: $RelativePath"
        return
    }

    $Matches = Select-String -Path $AbsolutePath -Pattern $Pattern -CaseSensitive
    if ($Matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking checkpoint metadata contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'config_hash\.unwrap_or\(0\)|mesh_hash:\s*0\b|config_hash != 0|found:\s*0\s*\}|if let Ok\(header\) = Checkpoint::read_header' `
        -Message "checkpoint metadata must not use sentinel hashes or silently skip invalid checkpoint headers"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'FLAG_CONFIG_HASH_PRESENT|FLAG_MESH_HASH_PRESENT' `
        -Message "checkpoint writer must encode explicit hash-presence flags"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'MissingConfigHash|MissingMeshHash' `
        -Message "checkpoint compatibility checks must report missing hashes explicitly"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'test_checkpoint_strict_mode_rejects_missing_hash_metadata|test_list_checkpoints_rejects_invalid_header' `
        -Message "checkpoint module must keep regression coverage for missing-hash strict mode and invalid catalog entries"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] checkpoint metadata contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] checkpoint metadata contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
