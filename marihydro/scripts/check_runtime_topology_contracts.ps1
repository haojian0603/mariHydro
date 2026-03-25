#!/usr/bin/env pwsh
#
# Blocking guard for runtime topology defaults.
# NUMA topology and thread-pool configuration must not fabricate a fake
# single-node topology through Default implementations.
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
        Add-Failure "missing required runtime file: $RelativePath"
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
    Write-Host "=== Checking runtime topology contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'impl Default for NumaTopology|impl Default for NumaThreadPoolConfig' `
        -Message "runtime topology types must not expose fake Default implementations"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'NumaTopology::default\(\)|NumaThreadPoolConfig::default\(\)' `
        -Message "runtime topology call sites must use explicit detect() instead of fake defaults"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'unwrap_or_else\(\|_\| Self \{' `
        -Message "runtime topology must not synthesize a fake single-node fallback on detection failure"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] runtime topology contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] runtime topology contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
