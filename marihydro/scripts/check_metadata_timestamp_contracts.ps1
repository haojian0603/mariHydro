#!/usr/bin/env pwsh
#
# Blocking guard for persisted metadata timestamp truthfulness.
# Snapshot/checkpoint creation must not fabricate Unix-epoch zero timestamps
# when the system clock query fails.
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
        Add-Failure "missing required metadata file: $RelativePath"
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
        Add-Failure "missing required metadata file: $RelativePath"
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
    Write-Host "=== Checking metadata timestamp contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'duration_since\(std::time::UNIX_EPOCH\)\s*\.map\(\|d\| d\.as_secs\(\)\)\s*\.unwrap_or\(0\)' `
        -Message "checkpoint creation must not collapse created_at clock failures to zero"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/snapshot.rs" `
        -Pattern 'duration_since\(std::time::UNIX_EPOCH\)\s*\.map\(\|d\| d\.as_secs\(\)\)\s*\.unwrap_or\(0\)' `
        -Message "snapshot metadata must not collapse created_at clock failures to zero"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/checkpoint.rs" `
        -Pattern 'fn current_unix_timestamp\(|test_current_unix_timestamp_nonzero|created_at > 0' `
        -Message "checkpoint module must keep explicit timestamp helper and regression coverage"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/snapshot.rs" `
        -Pattern 'fn current_unix_timestamp\(|test_current_unix_timestamp_nonzero|created_at > 0' `
        -Message "snapshot module must keep explicit timestamp helper and regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] metadata timestamp contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] metadata timestamp contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
