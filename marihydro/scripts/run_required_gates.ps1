#!/usr/bin/env pwsh
#
# Full gates for push hooks and manual convergence checks.
#

param(
    [string]$HookName = "manual"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host "== $Name ==" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "MariHydro required gates ($HookName)" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Invoke-Step -Name "cargo check --workspace" -Action {
        cargo check --workspace
    }

    Invoke-Step -Name "cargo clippy --workspace --all-targets" -Action {
        cargo clippy --workspace --all-targets
    }

    Invoke-Step -Name "cargo test --workspace" -Action {
        cargo test --workspace
    }

    Invoke-Step -Name "tracked temp artifacts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_tracked_temp_artifacts.ps1"
    }

    Invoke-Step -Name "text safety" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_text_safety.ps1"
    }

    Invoke-Step -Name "repository contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_repo_contracts.ps1"
    }

    Invoke-Step -Name "architecture verification" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/verify_architecture.ps1"
    }

    Invoke-Step -Name "architecture audit" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/architecture_audit.ps1"
    }

    Write-Host ""
    Write-Host "[OK] required gates passed" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ""
    Write-Host "[FAIL] required gates failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
