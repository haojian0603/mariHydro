#!/usr/bin/env pwsh
#
# MariHydro architecture verification script.
# ASCII-only version for reliable execution on Windows PowerShell.
#

param(
    [switch]$Verbose,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Push-Location $ProjectRoot
try {
    function Get-RustFilesFromTargets {
        param(
            [string[]]$Targets
        )

        $files = @()
        foreach ($target in $Targets) {
            $absoluteTarget = Join-Path $ProjectRoot $target
            if (-not (Test-Path $absoluteTarget)) {
                continue
            }

            $item = Get-Item $absoluteTarget
            if ($item.PSIsContainer) {
                $files += Get-ChildItem -Path $item.FullName -Recurse -Filter "*.rs" -File
            } elseif ($item.Extension -eq ".rs") {
                $files += $item
            }
        }

        return $files | Sort-Object FullName -Unique
    }

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " MariHydro architecture verification" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $errors = @()

    # Phase 0: tracked temporary artifact guard
    Write-Host "=== Phase 0: tracked temporary artifact guard ===" -ForegroundColor Cyan

    $trackedTempFiles = @()
    foreach ($pattern in @("tmp_*", "tmp_*/", "marihydro/tmp_*", "marihydro/tmp_*.log")) {
        $matches = & git ls-files $pattern 2>$null
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        foreach ($match in $matches) {
            if (-not [string]::IsNullOrWhiteSpace($match)) {
                $trackedTempFiles += $match.Trim()
            }
        }
    }

    $trackedTempFiles = $trackedTempFiles | Sort-Object -Unique
    if ($trackedTempFiles.Count -gt 0) {
        Write-Host "[FAIL] tracked temporary artifacts detected:" -ForegroundColor Red
        $trackedTempFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        $errors += "temporary workspace artifacts must not be tracked"
    } else {
        Write-Host "[OK] no tracked temporary artifacts" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5: external data contract guard
    Write-Host "=== Phase 0.5: external data contract guard (layout truth + matched-layout validation + CLI tool context) ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_external_data_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "external data contracts must pass"
    } else {
        Write-Host "[OK] external data contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.506: driver metadata contract guard
    Write-Host "=== Phase 0.506: driver metadata contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_driver_metadata_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "driver metadata contracts must pass"
    } else {
        Write-Host "[OK] driver metadata contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5125: external driver access contract guard
    Write-Host "=== Phase 0.5125: external driver access contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_external_driver_access_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "external driver access contracts must pass"
    } else {
        Write-Host "[OK] external driver access contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.518: VTU export contract guard
    Write-Host "=== Phase 0.518: VTU export contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_vtu_export_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "VTU export contracts must pass"
    } else {
        Write-Host "[OK] VTU export contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.525: export metadata contract guard
    Write-Host "=== Phase 0.525: export metadata contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_export_metadata_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "export metadata contracts must pass"
    } else {
        Write-Host "[OK] export metadata contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.53: metadata timestamp contract guard
    Write-Host "=== Phase 0.53: metadata timestamp contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_metadata_timestamp_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "metadata timestamp contracts must pass"
    } else {
        Write-Host "[OK] metadata timestamp contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.54: checkpoint metadata contract guard
    Write-Host "=== Phase 0.54: checkpoint metadata contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_checkpoint_metadata_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "checkpoint metadata contracts must pass"
    } else {
        Write-Host "[OK] checkpoint metadata contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.545: snapshot boundary contract guard
    Write-Host "=== Phase 0.545: snapshot boundary contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_snapshot_boundary_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "snapshot boundary contracts must pass"
    } else {
        Write-Host "[OK] snapshot boundary contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.546: IO snapshot contract guard
    Write-Host "=== Phase 0.546: IO snapshot contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_io_snapshot_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "IO snapshot contracts must pass"
    } else {
        Write-Host "[OK] IO snapshot contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.547: IO pipeline shutdown contract guard
    Write-Host "=== Phase 0.547: IO pipeline shutdown contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_pipeline_shutdown_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "IO pipeline shutdown contracts must pass"
    } else {
        Write-Host "[OK] IO pipeline shutdown contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.548: geo geodesic contract guard
    Write-Host "=== Phase 0.548: geo geodesic contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_geodesic_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo geodesic contracts must pass"
    } else {
        Write-Host "[OK] geo geodesic contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5485: geo vector contract guard
    Write-Host "=== Phase 0.5485: geo vector contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_vector_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo vector contracts must pass"
    } else {
        Write-Host "[OK] geo vector contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5487: geo central meridian contract guard
    Write-Host "=== Phase 0.5487: geo central meridian contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_central_meridian_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo central meridian contracts must pass"
    } else {
        Write-Host "[OK] geo central meridian contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5489: geo ellipsoid EPSG contract guard
    Write-Host "=== Phase 0.5489: geo ellipsoid EPSG contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_ellipsoid_epsg_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo ellipsoid EPSG contracts must pass"
    } else {
        Write-Host "[OK] geo ellipsoid EPSG contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.549: geo affine contract guard
    Write-Host "=== Phase 0.549: geo affine contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_affine_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo affine contracts must pass"
    } else {
        Write-Host "[OK] geo affine contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5495: geo CRS contract guard
    Write-Host "=== Phase 0.5495: geo CRS contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_crs_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo CRS contracts must pass"
    } else {
        Write-Host "[OK] geo CRS contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5497: geo auto projection contract guard
    Write-Host "=== Phase 0.5497: geo auto projection contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_auto_projection_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo auto projection contracts must pass"
    } else {
        Write-Host "[OK] geo auto projection contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.55: import contract guard
    Write-Host "=== Phase 0.55: import contract guard (geometry + null-feature rejection + semantic metadata + multipart name preservation + feature id semantics + CSV strict default) ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_import_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "import contracts must pass"
    } else {
        Write-Host "[OK] import contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.56: geo projection contract guard
    Write-Host "=== Phase 0.56: geo projection contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_geo_projection_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "geo projection contracts must pass"
    } else {
        Write-Host "[OK] geo projection contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.562: spatial index contract guard
    Write-Host "=== Phase 0.562: spatial index contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_spatial_index_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "spatial index contracts must pass"
    } else {
        Write-Host "[OK] spatial index contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.563: mesh spatial contract guard
    Write-Host "=== Phase 0.563: mesh spatial contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mesh_spatial_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "mesh spatial contracts must pass"
    } else {
        Write-Host "[OK] mesh spatial contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.564: structured mesh contract guard
    Write-Host "=== Phase 0.564: structured mesh contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mesh_structured_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "structured mesh contracts must pass"
    } else {
        Write-Host "[OK] structured mesh contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5645: MHB binary mesh contract guard
    Write-Host "=== Phase 0.5645: MHB binary mesh contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_mhb_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "MHB binary mesh contracts must pass"
    } else {
        Write-Host "[OK] MHB binary mesh contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.5647: GMSH parser contract guard
    Write-Host "=== Phase 0.5647: GMSH parser contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_gmsh_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "GMSH parser contracts must pass"
    } else {
        Write-Host "[OK] GMSH parser contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.565: Web Mercator contract guard
    Write-Host "=== Phase 0.565: Web Mercator contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_web_mercator_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "Web Mercator contracts must pass"
    } else {
        Write-Host "[OK] Web Mercator contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.57: IO invariant contract guard
    Write-Host "=== Phase 0.57: IO invariant contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_io_invariant_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "IO invariant contracts must pass"
    } else {
        Write-Host "[OK] IO invariant contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.6: runtime probe contract guard
    Write-Host "=== Phase 0.6: runtime probe contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_probe_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "runtime probe contracts must pass"
    } else {
        Write-Host "[OK] runtime probe contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.65: runtime parallelism contract guard
    Write-Host "=== Phase 0.65: runtime parallelism contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_parallelism_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "runtime parallelism contracts must pass"
    } else {
        Write-Host "[OK] runtime parallelism contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.66: runtime topology contract guard
    Write-Host "=== Phase 0.66: runtime topology contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_topology_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "runtime topology contracts must pass"
    } else {
        Write-Host "[OK] runtime topology contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.67: runtime allocator contract guard
    Write-Host "=== Phase 0.67: runtime allocator contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_runtime_allocator_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "runtime allocator contracts must pass"
    } else {
        Write-Host "[OK] runtime allocator contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 0.7: AI state contract guard
    Write-Host "=== Phase 0.7: AI state contract guard ===" -ForegroundColor Cyan
    powershell -ExecutionPolicy Bypass -File "$ScriptDir/check_ai_state_contracts.ps1"
    if ($LASTEXITCODE -ne 0) {
        $errors += "AI state contracts must pass"
    } else {
        Write-Host "[OK] AI state contracts passed" -ForegroundColor Green
    }

    Write-Host ""

    # Phase 1: layer dependency checks
    Write-Host "=== Phase 1: layer dependency checks ===" -ForegroundColor Cyan

    Write-Host "Checking mh_foundation zero dependencies..." -ForegroundColor Yellow
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $foundationDeps = & cargo tree -p mh_foundation --edges normal 2>$null | Select-String "mh_" | Where-Object { $_ -notmatch "^mh_foundation" }
    $ErrorActionPreference = $oldEap
    if ($foundationDeps) {
        Write-Host "[FAIL] mh_foundation has internal dependencies:" -ForegroundColor Red
        $foundationDeps | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        $errors += "mh_foundation should not depend on other mh_* modules"
    } else {
        Write-Host "[OK] mh_foundation is dependency-free" -ForegroundColor Green
    }

    Write-Host "Checking mh_runtime dependencies..." -ForegroundColor Yellow
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $runtimeResult = & cargo tree -p mh_runtime --depth 1 2>$null
    $ErrorActionPreference = $oldEap
    if ($LASTEXITCODE -eq 0) {
        $runtimeDeps = $runtimeResult | Select-String "mh_" | Where-Object { $_ -notmatch "mh_foundation" -and $_ -notmatch "^mh_runtime" }
        if ($runtimeDeps) {
            Write-Host "[FAIL] mh_runtime has illegal dependencies:" -ForegroundColor Red
            $runtimeDeps | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            $errors += "mh_runtime should only depend on mh_foundation"
        } else {
            Write-Host "[OK] mh_runtime dependencies are valid" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARN] mh_runtime is unavailable or does not build" -ForegroundColor Yellow
    }

    Write-Host "Checking mh_config dependencies..." -ForegroundColor Yellow
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $configResult = & cargo tree -p mh_config --depth 1 2>$null
    $ErrorActionPreference = $oldEap
    if ($LASTEXITCODE -eq 0) {
        $configDeps = $configResult | Select-String "mh_" | Where-Object { $_ -notmatch "mh_runtime" -and $_ -notmatch "mh_foundation" -and $_ -notmatch "^mh_config" }
        if ($configDeps) {
            Write-Host "[FAIL] mh_config has illegal dependencies:" -ForegroundColor Red
            $configDeps | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            $errors += "mh_config should only depend on mh_runtime"
        } else {
            Write-Host "[OK] mh_config dependencies are valid" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARN] mh_config is unavailable or does not build" -ForegroundColor Yellow
    }

    # Phase 2: legacy residue checks
    Write-Host ""
    Write-Host "=== Phase 2: legacy residue checks ===" -ForegroundColor Cyan

    Write-Host "Checking Legacy type aliases..." -ForegroundColor Yellow
    $typeAliases = Get-ChildItem -Path "crates" -Recurse -Filter "*.rs" -File | Select-String -Pattern "pub type \w+ = \w+Generic<f64>;"
    if ($typeAliases) {
        Write-Host "[FAIL] Legacy type aliases exist:" -ForegroundColor Red
        $typeAliases | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red }
        $errors += "Legacy type aliases should be removed"
    } else {
        Write-Host "[OK] No Legacy type aliases" -ForegroundColor Green
    }

    Write-Host "Checking mh_core status..." -ForegroundColor Yellow
    if (Test-Path "crates/mh_core") {
        $coreFiles = Get-ChildItem -Path "crates/mh_core/src" -Filter "*.rs" -ErrorAction SilentlyContinue
        if ($coreFiles) {
            Write-Host "[WARN] mh_core still exists and may need migration cleanup" -ForegroundColor Yellow
        } else {
            Write-Host "[OK] mh_core is empty or absent" -ForegroundColor Green
        }
    } else {
        Write-Host "[OK] mh_core is deleted" -ForegroundColor Green
    }

    # Phase 3: config layer generic check
    Write-Host ""
    Write-Host "=== Phase 3: config layer generic check ===" -ForegroundColor Cyan

    if (Test-Path "crates/mh_config/src") {
        $configGenerics = Get-ChildItem -Path "crates/mh_config/src" -Recurse -Filter "*.rs" -File | Select-String -Pattern "<.*:.*Backend"
        if ($configGenerics) {
            Write-Host "[FAIL] mh_config contains Backend generics:" -ForegroundColor Red
            $configGenerics | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red }
            $errors += "mh_config should not contain Backend generics"
        } else {
            Write-Host "[OK] mh_config has no Backend generics" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARN] mh_config/src is unavailable" -ForegroundColor Yellow
    }

    # Phase 4: index generation check
    Write-Host ""
    Write-Host "=== Phase 4: index generation check ===" -ForegroundColor Cyan

    if (Test-Path "crates/mh_runtime/src/indices.rs") {
        $indicesGen = Select-String -Path "crates/mh_runtime/src/indices.rs" -Pattern "generation"
        if ($indicesGen) {
            Write-Host "[FAIL] indices.rs contains generation fields:" -ForegroundColor Red
            $indicesGen | ForEach-Object { Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red }
            $errors += "mh_runtime/indices.rs should not contain generation fields"
        } else {
            Write-Host "[OK] indices.rs has no generation fields" -ForegroundColor Green
        }
    } else {
        Write-Host "[WARN] mh_runtime/src/indices.rs is unavailable" -ForegroundColor Yellow
    }

    # Phase 5: raw scalar conversion guard
    Write-Host ""
    Write-Host "=== Phase 5: raw scalar conversion guard ===" -ForegroundColor Cyan

    $rawScalarCalls = Get-ChildItem -Path "crates/mh_physics/src" -Recurse -Filter "*.rs" -File |
        Select-String -Pattern '(?<!try_)\bscalar_from_f64\(' -CaseSensitive
    if ($rawScalarCalls) {
        Write-Host "[FAIL] raw scalar_from_f64 call sites exist:" -ForegroundColor Red
        $rawScalarCalls | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($rawScalarCalls.Count -gt 10) {
            Write-Host "  ... and $($rawScalarCalls.Count - 10) more" -ForegroundColor Red
        }
        $errors += "raw scalar_from_f64 call sites must be removed from mh_physics/src"
    } else {
        Write-Host "[OK] no raw scalar_from_f64 call sites in mh_physics/src" -ForegroundColor Green
    }

    $silentConfigFallbackRoots = @(
        "crates/mh_physics/src/numerics/gradient",
        "crates/mh_physics/src/numerics/limiter",
        "crates/mh_physics/src/numerics/reconstruction",
        "crates/mh_physics/src/mesh",
        "crates/mh_physics/src/assimilation/bridge.rs",
        "crates/mh_physics/src/assimilation/conservation.rs",
        "crates/mh_physics/src/assimilation/mod.rs",
        "crates/mh_physics/src/boundary/ghost.rs",
        "crates/mh_physics/src/boundary/manager.rs",
        "crates/mh_physics/src/boundary/types.rs",
        "crates/mh_physics/src/config_bridge.rs",
        "crates/mh_physics/src/engine/timestep.rs",
        "crates/mh_physics/src/engine/time_integrator.rs",
        "crates/mh_physics/src/engine/strategy/explicit.rs",
        "crates/mh_physics/src/engine/strategy/mod.rs",
        "crates/mh_physics/src/numerics/linear_algebra/solver.rs",
        "crates/mh_physics/src/numerics/linear_algebra/csr.rs",
        "crates/mh_physics/src/schemes/riemann/hllc.rs",
        "crates/mh_physics/src/vertical/profile.rs",
        "crates/mh_physics/src/boundary/traits.rs",
        "crates/mh_physics/src/builder/solver_builder.rs",
        "crates/mh_physics/src/engine/parallel.rs",
        "crates/mh_physics/src/engine/strategy/semi_implicit.rs",
        "crates/mh_physics/src/schemes/riemann/traits.rs",
        "crates/mh_physics/src/schemes/wetting_drying/handler.rs",
        "crates/mh_physics/src/sources/coriolis.rs",
        "crates/mh_physics/src/tracer/transport.rs"
    )
    $silentConfigFallbacks = @(
        Get-RustFilesFromTargets -Targets $silentConfigFallbackRoots |
            Select-String -Pattern '\bfrom_config\(.*\)\.unwrap_or\(' -CaseSensitive
    )
    if ($silentConfigFallbacks) {
        Write-Host "[FAIL] silent from_config(...).unwrap_or(...) fallbacks exist in guarded mh_physics roots:" -ForegroundColor Red
        $silentConfigFallbacks | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($silentConfigFallbacks.Count -gt 10) {
            Write-Host "  ... and $($silentConfigFallbacks.Count - 10) more" -ForegroundColor Red
        }
        $errors += "silent from_config(...).unwrap_or(...) fallbacks must be removed from the guarded mh_physics conversion roots"
    } else {
        Write-Host "[OK] no silent from_config(...).unwrap_or(...) fallbacks in guarded mh_physics conversion roots" -ForegroundColor Green
    }

    $rawIfLetConfigOptions = @(
        Get-RustFilesFromTargets -Targets @("crates/mh_physics/src") |
            Select-String -Pattern 'if let Some\([^\)]*\) = .*::from_config\(' -CaseSensitive
    )
    if ($rawIfLetConfigOptions) {
        Write-Host "[FAIL] raw if-let from_config option handling exists in mh_physics/src:" -ForegroundColor Red
        $rawIfLetConfigOptions | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($rawIfLetConfigOptions.Count -gt 10) {
            Write-Host "  ... and $($rawIfLetConfigOptions.Count - 10) more" -ForegroundColor Red
        }
        $errors += "raw if-let from_config option handling must be removed from mh_physics/src"
    } else {
        Write-Host "[OK] no raw if-let from_config option handling in mh_physics/src" -ForegroundColor Green
    }

    $realImplementationGuard = Join-Path $ScriptDir "check_real_implementation_contracts.ps1"
    if (-not (Test-Path $realImplementationGuard)) {
        Write-Host "[FAIL] check_real_implementation_contracts.ps1 is missing" -ForegroundColor Red
        $errors += "check_real_implementation_contracts.ps1 must exist"
    } else {
        Write-Host "Checking real implementation contracts..." -ForegroundColor Yellow
        & powershell -ExecutionPolicy Bypass -File $realImplementationGuard
        if ($LASTEXITCODE -ne 0) {
            $errors += "real implementation contracts failed"
        } else {
            Write-Host "[OK] real implementation contracts passed" -ForegroundColor Green
        }
    }

    $sourceSemanticsGuard = Join-Path $ScriptDir "check_source_semantics_contracts.ps1"
    if (-not (Test-Path $sourceSemanticsGuard)) {
        Write-Host "[FAIL] check_source_semantics_contracts.ps1 is missing" -ForegroundColor Red
        $errors += "check_source_semantics_contracts.ps1 must exist"
    } else {
        Write-Host "Checking source API semantics contracts..." -ForegroundColor Yellow
        & powershell -ExecutionPolicy Bypass -File $sourceSemanticsGuard
        if ($LASTEXITCODE -ne 0) {
            $errors += "source API semantics contracts failed"
        } else {
            Write-Host "[OK] source API semantics contracts passed" -ForegroundColor Green
        }
    }

    $silentScalarFallbacks = @(
        Get-RustFilesFromTargets -Targets @("crates/mh_physics/src") |
            Select-String -Pattern '\bfrom_f(?:64|32)\(.*\)\.unwrap_or\(' -CaseSensitive
    )
    if ($silentScalarFallbacks) {
        Write-Host "[FAIL] silent from_f64/from_f32(...).unwrap_or(...) fallbacks exist in mh_physics/src:" -ForegroundColor Red
        $silentScalarFallbacks | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($silentScalarFallbacks.Count -gt 10) {
            Write-Host "  ... and $($silentScalarFallbacks.Count - 10) more" -ForegroundColor Red
        }
        $errors += "silent from_f64/from_f32(...).unwrap_or(...) fallbacks must be removed from mh_physics/src"
    } else {
        Write-Host "[OK] no silent from_f64/from_f32(...).unwrap_or(...) fallbacks in mh_physics/src" -ForegroundColor Green
    }

    $runtimeSilentScalarFallbacks = @(
        Get-RustFilesFromTargets -Targets @("crates/mh_runtime/src") |
            Select-String -Pattern '\bfrom_f(?:64|32)\(.*\)\.unwrap_or\(' -CaseSensitive
    )
    if ($runtimeSilentScalarFallbacks) {
        Write-Host "[FAIL] silent from_f64/from_f32(...).unwrap_or(...) fallbacks exist in mh_runtime/src:" -ForegroundColor Red
        $runtimeSilentScalarFallbacks | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($runtimeSilentScalarFallbacks.Count -gt 10) {
            Write-Host "  ... and $($runtimeSilentScalarFallbacks.Count - 10) more" -ForegroundColor Red
        }
        $errors += "silent from_f64/from_f32(...).unwrap_or(...) fallbacks must be removed from mh_runtime/src"
    } else {
        Write-Host "[OK] no silent from_f64/from_f32(...).unwrap_or(...) fallbacks in mh_runtime/src" -ForegroundColor Green
    }

    # Phase 6: compile check
    Write-Host ""
    Write-Host "=== Phase 6: compile check ===" -ForegroundColor Cyan

    if ($Strict) {
        Write-Host "Running cargo check..." -ForegroundColor Yellow
        $oldEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $null = & cargo check --workspace 2>$null
        $ErrorActionPreference = $oldEap
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[FAIL] cargo check failed" -ForegroundColor Red
            $errors += "cargo check failed"
        } else {
            Write-Host "[OK] cargo check passed" -ForegroundColor Green
        }
    } else {
        Write-Host "[SKIP] cargo check skipped (use -Strict to enable)" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    if ($errors.Count -eq 0) {
        Write-Host "[OK] architecture verification passed" -ForegroundColor Green
        exit 0
    }

    Write-Host "[FAIL] architecture verification failed" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
finally {
    Pop-Location
}
