#!/usr/bin/env pwsh
#
# Repository collaboration contract guard.
# Ensures tracked norms and local hook wiring stay active.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $ProjectRoot
$ExpectedHooksPath = "marihydro/.githooks"
$Errors = @()

Push-Location $RepoRoot
try {
    Write-Host "=== Checking repository collaboration contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Repository root: $RepoRoot"
    Write-Host ""

    $requiredFiles = @(
        "AGENTS.md",
        ".githooks/pre-commit",
        ".githooks/pre-push",
        "scripts/check_real_implementation_contracts.ps1",
        "scripts/run_fast_gates.ps1",
        "scripts/run_required_gates.ps1",
        "scripts/setup_git_hooks.ps1"
    )

    foreach ($relativePath in $requiredFiles) {
        $absolutePath = Join-Path $ProjectRoot $relativePath
        if (Test-Path $absolutePath) {
            Write-Host "[OK] $relativePath" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] missing required contract file: $relativePath" -ForegroundColor Red
            $Errors += "missing required contract file: $relativePath"
        }
    }

    $agentsPath = Join-Path $ProjectRoot "AGENTS.md"
    if (Test-Path $agentsPath) {
        $agentsContent = Get-Content $agentsPath -Raw -Encoding UTF8
        $requiredClauses = @(
            "[RULE_NO_COMPAT_MAINLINE]",
            "[RULE_NO_FAKE_IMPL]",
            "[RULE_PHYSICS_PROVENANCE]",
            "[RULE_FORMULA_NAME_HONEST]",
            "[RULE_PUBLIC_SURFACE_TRUTHFUL]",
            "[RULE_GATES_STRICTER_ONLY]"
        )

        foreach ($clause in $requiredClauses) {
            if ($agentsContent -like "*$clause*") {
                Write-Host "[OK] AGENTS.md contains required clause: $clause" -ForegroundColor Green
            } else {
                Write-Host "[FAIL] AGENTS.md is missing required clause: $clause" -ForegroundColor Red
                $Errors += "AGENTS.md must contain required clause: $clause"
            }
        }
    }

    $configuredHooksPath = git config --local --get core.hooksPath 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($configuredHooksPath)) {
        Write-Host "[FAIL] core.hooksPath is not configured" -ForegroundColor Red
        $Errors += "core.hooksPath must be configured to the tracked hook directory"
    } elseif ($configuredHooksPath.Trim() -ne $ExpectedHooksPath) {
        Write-Host "[FAIL] core.hooksPath drifted: $configuredHooksPath" -ForegroundColor Red
        $Errors += "core.hooksPath must equal $ExpectedHooksPath"
    } else {
        Write-Host "[OK] core.hooksPath = $ExpectedHooksPath" -ForegroundColor Green
    }

    $preCommit = Join-Path $ProjectRoot ".githooks/pre-commit"
    if (Test-Path $preCommit) {
        $preCommitContent = Get-Content $preCommit -Raw
        if ($preCommitContent -match 'run_fast_gates\.ps1') {
            Write-Host "[OK] pre-commit routes to run_fast_gates.ps1" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] pre-commit does not route to run_fast_gates.ps1" -ForegroundColor Red
            $Errors += "pre-commit must route to scripts/run_fast_gates.ps1"
        }
    }

    $prePush = Join-Path $ProjectRoot ".githooks/pre-push"
    if (Test-Path $prePush) {
        $prePushContent = Get-Content $prePush -Raw
        if ($prePushContent -match 'run_required_gates\.ps1') {
            Write-Host "[OK] pre-push routes to run_required_gates.ps1" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] pre-push does not route to run_required_gates.ps1" -ForegroundColor Red
            $Errors += "pre-push must route to scripts/run_required_gates.ps1"
        }
    }

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] repository collaboration contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] repository collaboration contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
