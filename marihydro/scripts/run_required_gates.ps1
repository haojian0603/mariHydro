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

    Invoke-Step -Name "external data contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_external_data_contracts.ps1"
    }

    Invoke-Step -Name "export metadata contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_export_metadata_contracts.ps1"
    }

    Invoke-Step -Name "import contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_import_contracts.ps1"
    }

    Invoke-Step -Name "geo projection contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_projection_contracts.ps1"
    }

    Invoke-Step -Name "Web Mercator contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_web_mercator_contracts.ps1"
    }

    Invoke-Step -Name "IO invariant contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_io_invariant_contracts.ps1"
    }

    Invoke-Step -Name "AI state contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_ai_state_contracts.ps1"
    }

    Invoke-Step -Name "physics provenance" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_physics_provenance.ps1"
    }

    Invoke-Step -Name "real implementation contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_real_implementation_contracts.ps1"
    }

    Invoke-Step -Name "runtime probe contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_probe_contracts.ps1"
    }

    Invoke-Step -Name "runtime parallelism contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_parallelism_contracts.ps1"
    }

    Invoke-Step -Name "runtime topology contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_topology_contracts.ps1"
    }

    Invoke-Step -Name "runtime allocator contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_allocator_contracts.ps1"
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
