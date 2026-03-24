#!/usr/bin/env pwsh
#
# Fail when temporary workspace artifacts are tracked by git.
#

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$TrackedPatterns = @(
    "tmp_*",
    "tmp_*/",
    "marihydro/tmp_*",
    "marihydro/tmp_*.log"
)

Write-Host "=== Checking tracked temporary artifacts ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

Push-Location $ProjectRoot
try {
    $TrackedFiles = @()
    foreach ($Pattern in $TrackedPatterns) {
        $Matches = & git ls-files $Pattern 2>$null
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        foreach ($Match in $Matches) {
            if (-not [string]::IsNullOrWhiteSpace($Match)) {
                $TrackedFiles += $Match.Trim()
            }
        }
    }

    $TrackedFiles = $TrackedFiles | Sort-Object -Unique

    if ($TrackedFiles.Count -eq 0) {
        Write-Host "[OK] No tracked temporary artifacts found" -ForegroundColor Green
        exit 0
    }

    Write-Host "[FAIL] Tracked temporary artifacts detected:" -ForegroundColor Red
    foreach ($Path in $TrackedFiles) {
        Write-Host "  $Path" -ForegroundColor Red
    }
    exit 1
}
finally {
    Pop-Location
}
