#!/usr/bin/env pwsh
#
# Blocking guard for runtime system-probe contracts.
# Ensures hardware and OS metadata are queried explicitly rather than
# synthesized from silent defaults.
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
    Write-Host "=== Checking runtime probe contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'read_to_string\(&cpulist_path\)\.unwrap_or_default\(\)|read_to_string\(&meminfo_path\)\.unwrap_or_default\(\)' `
        -Message "NUMA probe must not swallow sysfs read failures into empty text"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern 'parse\(\)\.unwrap_or\(0\)|parse::<u64>\(\)\.unwrap_or\(0\)' `
        -Message "NUMA probe must not collapse malformed memory metadata into zero"

    Check-PatternAbsent `
        -RelativePath "crates/mh_runtime/src/numa.rs" `
        -Pattern '8 \* 1024 \* 1024 \* 1024|4 \* 1024 \* 1024 \* 1024' `
        -Message "NUMA probe must not advertise synthetic fixed memory sizes"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] runtime probe contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] runtime probe contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
