#!/usr/bin/env pwsh
#
# Blocking guard for export metadata serialization truthfulness.
# Export paths must fail explicitly when metadata serialization fails instead of
# fabricating empty JSON arrays or other placeholder payloads.
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
        Add-Failure "missing required export file: $RelativePath"
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
        Add-Failure "missing required export file: $RelativePath"
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
    Write-Host "=== Checking export metadata contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/pipeline.rs" `
        -Pattern 'serde_json::to_string\(names\)\.unwrap_or_else\(\|_\| "\[\]"\.into\(\)\)' `
        -Message "pipeline VTU ASCII export must not collapse boundary_names serialization to []"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/vtu/binary.rs" `
        -Pattern 'serde_json::to_string\(names\)\.unwrap_or_else\(\|_\| "\[\]"\.into\(\)\)' `
        -Message "binary VTU export must not collapse boundary_names serialization to []"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/pipeline.rs" `
        -Pattern 'fn serialize_boundary_names\(|boundary_names 序列化失败|test_write_vtu_ascii_preserves_boundary_names_json' `
        -Message "pipeline VTU ASCII export must use explicit boundary_names serialization failures and regression coverage"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/vtu/binary.rs" `
        -Pattern 'fn serialize_boundary_names\(|boundary_names 序列化失败|test_serialize_boundary_names_json' `
        -Message "binary VTU export must use explicit boundary_names serialization failures and regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] export metadata contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] export metadata contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
