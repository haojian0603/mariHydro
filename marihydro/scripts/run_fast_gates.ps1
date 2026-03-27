#!/usr/bin/env pwsh
#
# Fast gates for local commit hooks.
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
    Write-Host "MariHydro fast gates ($HookName)" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Invoke-Step -Name "tracked temp artifacts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_tracked_temp_artifacts.ps1"
    }

    Invoke-Step -Name "text safety" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_text_safety.ps1"
    }

    Invoke-Step -Name "repository contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_repo_contracts.ps1"
    }

    Invoke-Step -Name "external data contracts (layout truth + matched-layout validation + CLI tool context)" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_external_data_contracts.ps1"
    }

    Invoke-Step -Name "driver metadata contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_driver_metadata_contracts.ps1"
    }

    Invoke-Step -Name "external driver access contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_external_driver_access_contracts.ps1"
    }

    Invoke-Step -Name "VTU export contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_vtu_export_contracts.ps1"
    }

    Invoke-Step -Name "export metadata contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_export_metadata_contracts.ps1"
    }

    Invoke-Step -Name "metadata timestamp contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_metadata_timestamp_contracts.ps1"
    }

    Invoke-Step -Name "checkpoint metadata contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_checkpoint_metadata_contracts.ps1"
    }

    Invoke-Step -Name "snapshot boundary contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_snapshot_boundary_contracts.ps1"
    }

    Invoke-Step -Name "IO pipeline shutdown contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_pipeline_shutdown_contracts.ps1"
    }

    Invoke-Step -Name "geo geodesic contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_geodesic_contracts.ps1"
    }

    Invoke-Step -Name "geo affine contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_affine_contracts.ps1"
    }

    Invoke-Step -Name "geo CRS contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_crs_contracts.ps1"
    }

    Invoke-Step -Name "import contracts (geometry + null-feature rejection + semantic metadata + multipart name preservation + feature id semantics + CSV strict default)" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_import_contracts.ps1"
    }

    Invoke-Step -Name "geo projection contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_projection_contracts.ps1"
    }

    Invoke-Step -Name "spatial index contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_spatial_index_contracts.ps1"
    }

    Invoke-Step -Name "mesh spatial contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mesh_spatial_contracts.ps1"
    }

    Invoke-Step -Name "structured mesh contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mesh_structured_contracts.ps1"
    }

    Invoke-Step -Name "MHB binary mesh contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mhb_contracts.ps1"
    }

    Invoke-Step -Name "GMSH parser contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_gmsh_contracts.ps1"
    }

    Invoke-Step -Name "Web Mercator tile/domain contracts" -Action {
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

    Invoke-Step -Name "source API semantics contracts" -Action {
        powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_source_semantics_contracts.ps1"
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

    Invoke-Step -Name "cargo check --workspace" -Action {
        cargo check --workspace
    }

    Write-Host ""
    Write-Host "[OK] fast gates passed" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ""
    Write-Host "[FAIL] fast gates failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
