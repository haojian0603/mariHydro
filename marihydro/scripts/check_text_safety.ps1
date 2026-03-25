#!/usr/bin/env pwsh
#
# Text safety guard for local gates.
# - blocks unresolved merge markers in tracked files
# - blocks whitespace errors in staged changes when available
#

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking text safety ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $failed = $false

    $mergeMatches = @(
        & git grep -n -I -E "^(<<<<<<< |=======|>>>>>>> )" -- . 2>$null
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($mergeMatches.Count -gt 0) {
        Write-Host "[FAIL] unresolved merge markers detected:" -ForegroundColor Red
        $mergeMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        if ($mergeMatches.Count -gt 10) {
            Write-Host "  ... and $($mergeMatches.Count - 10) more" -ForegroundColor Red
        }
        $failed = $true
    } else {
        Write-Host "[OK] no unresolved merge markers" -ForegroundColor Green
    }

    $corruptionMatches = @(
        & git grep -n -I -P '[\x{E000}-\x{F8FF}\x{FFFD}]' -- '*.rs' '*.ps1' '*.toml' '*.json' '*.yml' '*.yaml' 'AGENTS.md' 2>$null
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($corruptionMatches.Count -gt 0) {
        Write-Host "[FAIL] text corruption markers detected:" -ForegroundColor Red
        $corruptionMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        if ($corruptionMatches.Count -gt 10) {
            Write-Host "  ... and $($corruptionMatches.Count - 10) more" -ForegroundColor Red
        }
        $failed = $true
    } else {
        Write-Host "[OK] no replacement/private-use Unicode corruption" -ForegroundColor Green
    }

    $questionCorruptionMatches = @(
        & git grep -n -I -P '\?{3,}' -- '*.rs' '*.ps1' '*.toml' '*.json' '*.yml' '*.yaml' 'AGENTS.md' 2>$null
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($questionCorruptionMatches.Count -gt 0) {
        Write-Host "[FAIL] obvious question-mark text corruption detected:" -ForegroundColor Red
        $questionCorruptionMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        if ($questionCorruptionMatches.Count -gt 10) {
            Write-Host "  ... and $($questionCorruptionMatches.Count - 10) more" -ForegroundColor Red
        }
        $failed = $true
    } else {
        Write-Host "[OK] no obvious question-mark text corruption" -ForegroundColor Green
    }

    $mixedScriptMatches = @(
        & git grep -n -I -P '[\x{0400}-\x{04FF}\x{20AC}\x{3220}-\x{3229}\x{FF21}-\x{FF3A}\x{FF41}-\x{FF5A}]' -- '*.rs' '*.ps1' '*.toml' '*.json' '*.yml' '*.yaml' 'AGENTS.md' 2>$null
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($mixedScriptMatches.Count -gt 0) {
        Write-Host "[FAIL] suspicious mixed-script text corruption detected:" -ForegroundColor Red
        $mixedScriptMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor Red
        }
        if ($mixedScriptMatches.Count -gt 10) {
            Write-Host "  ... and $($mixedScriptMatches.Count - 10) more" -ForegroundColor Red
        }
        $failed = $true
    } else {
        Write-Host "[OK] no suspicious mixed-script corruption" -ForegroundColor Green
    }

    $stagedFiles = @(
        & git diff --cached --name-only --diff-filter=ACMR 2>$null
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($stagedFiles.Count -gt 0) {
        if ($Verbose) {
            Write-Host "[INFO] checking staged patch whitespace" -ForegroundColor Yellow
        }
        $diffCheck = & git diff --cached --check 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] staged whitespace or conflict issues detected:" -ForegroundColor Red
            $diffCheck | Select-Object -First 20 | ForEach-Object {
                Write-Host "  $_" -ForegroundColor Red
            }
            $failed = $true
        } else {
            Write-Host "[OK] staged patch text safety" -ForegroundColor Green
        }
    } else {
        Write-Host "[OK] no staged patch to validate" -ForegroundColor Green
    }

    if ($failed) {
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] text safety passed" -ForegroundColor Green
    exit 0
}
finally {
    Pop-Location
}
