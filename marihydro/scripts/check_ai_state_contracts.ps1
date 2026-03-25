#!/usr/bin/env pwsh
#
# Blocking guard for silent AI state fallbacks in the shipped surrogate path.
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

Write-Host "=== Checking AI state contracts ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$surrogatePath = Join-Path $ProjectRoot "crates/mh_agent/src/surrogate.rs"
if (-not (Test-Path $surrogatePath)) {
    Add-Failure "missing surrogate implementation path"
} else {
    $patternChecks = @(
        @{ Name = "surrogate must not swallow model load errors"; Pattern = 'let _ = model\.load_state'; Message = "surrogate model construction must fail explicitly when persisted state loading fails" },
        @{ Name = "surrogate must not swallow model save errors"; Pattern = 'let _ = self\.save_state'; Message = "surrogate training must fail explicitly when persisted state saving fails" },
        @{ Name = "surrogate must not synthesize optimistic confidence"; Pattern = '\.unwrap_or\(0\.8\)'; Message = "surrogate confidence must not default to a synthetic optimistic value" },
        @{ Name = "surrogate must not zero-fill missing feature slots"; Pattern = 'features\.get\(i\)\.copied\(\)\.unwrap_or\(0\.0\)'; Message = "surrogate linear prediction must reject feature-shape mismatches instead of zero-filling them" },
        @{ Name = "surrogate must not zero-fill missing prediction slots"; Pattern = 'pred\.get\(o\)\.copied\(\)\.unwrap_or\(0\.0\)|target_norm\.get\(o\)\.copied\(\)\.unwrap_or\(0\.0\)'; Message = "surrogate training must reject prediction-shape mismatches instead of zero-filling them" },
        @{ Name = "surrogate normalization must not fabricate defaults"; Pattern = 'norm\.(mean|std|m2)\.get\(.*\)\.copied\(\)\.unwrap_or\((0\.0|1\.0)\)'; Message = "surrogate normalization state must be validated explicitly instead of fabricating missing statistics" },
        @{ Name = "surrogate apply must not partially align by min length"; Pattern = 'pred\.values\.len\(\)\.min\(cell_areas\.len\(\)\)'; Message = "surrogate apply must reject prediction/state size mismatches instead of truncating silently" }
    )

    foreach ($check in $patternChecks) {
        $matches = Select-String -Path $surrogatePath -Pattern $check.Pattern -CaseSensitive
        if ($matches) {
            Add-Failure $check.Message
            $matches | Select-Object -First 5 | ForEach-Object {
                Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
            }
        } else {
            Write-Host "[OK] $($check.Name)" -ForegroundColor Green
        }
    }
}

if ($Errors.Count -eq 0) {
    Write-Host ""
    Write-Host "[OK] AI state contracts passed" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "[FAIL] AI state contracts failed" -ForegroundColor Red
Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
exit 1
