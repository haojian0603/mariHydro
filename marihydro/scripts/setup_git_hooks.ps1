#!/usr/bin/env pwsh
#
# Configure the repository to use the tracked hook directory.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $ProjectRoot
$HooksPath = "marihydro/.githooks"

Push-Location $RepoRoot
try {
    Write-Host "Configuring MariHydro git hooks" -ForegroundColor Cyan
    Write-Host "Repository root: $RepoRoot"
    Write-Host "Hooks path: $HooksPath"
    Write-Host ""

    git config --local core.hooksPath $HooksPath
    if ($LASTEXITCODE -ne 0) {
        throw "git config core.hooksPath failed"
    }

    $ConfiguredPath = git config --local --get core.hooksPath
    if ($LASTEXITCODE -ne 0) {
        throw "git config --get core.hooksPath failed"
    }

    Write-Host "[OK] core.hooksPath = $ConfiguredPath" -ForegroundColor Green
    Write-Host "[INFO] pre-commit -> scripts/run_fast_gates.ps1" -ForegroundColor Cyan
    Write-Host "[INFO] pre-push   -> scripts/run_required_gates.ps1" -ForegroundColor Cyan

    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_repo_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "repository collaboration contract check failed"
    }

    exit 0
}
catch {
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
