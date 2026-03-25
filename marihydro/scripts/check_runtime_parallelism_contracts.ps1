#!/usr/bin/env pwsh
#
# Blocking guard for runtime parallelism probe contracts.
# Prevents thread-count and core-topology detection from silently fabricating
# usable values when the OS probe or parser has already failed.
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
    Write-Host "=== Checking runtime parallelism contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'available_parallelism\(\)[^\r\n;]*unwrap_or\((?:1|1usize)\)' `
        -Message "runtime parallelism probe must not collapse OS query failure into one thread"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'available_parallelism\(\)[^\r\n;]*unwrap_or_default\(\)' `
        -Message "runtime parallelism probe must not collapse OS query failure into default thread counts"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'parse::<usize>\(\)\.ok\(\)|parse\(\)\.ok\(\)' `
        -Message "runtime physical-core probe must not silently discard malformed '/proc/cpuinfo' metadata"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] runtime parallelism contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] runtime parallelism contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
