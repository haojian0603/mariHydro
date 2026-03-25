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

    $legacySourceImpls = Get-ChildItem -Path "crates/mh_physics/src/sources" -Recurse -Filter "*.rs" -File |
        Where-Object { $_.FullName -notlike "*\sources\legacy.rs" } |
        Select-String -Pattern '\bimpl\s+SourceTerm\s+for\b' -CaseSensitive
    if ($legacySourceImpls) {
        Write-Host "[FAIL] legacy SourceTerm implementations exist outside the bridge definition:" -ForegroundColor Red
        $legacySourceImpls | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($legacySourceImpls.Count -gt 10) {
            Write-Host "  ... and $($legacySourceImpls.Count - 10) more" -ForegroundColor Red
        }
        $errors += "legacy SourceTerm implementations must not be added outside crates/mh_physics/src/sources/legacy.rs"
    } else {
        Write-Host "[OK] no legacy SourceTerm implementations outside the bridge definition" -ForegroundColor Green
    }

    $legacyBridgeLeaks = Get-ChildItem -Path "crates/mh_physics/src/sources" -Recurse -Filter "*.rs" -File |
        Where-Object {
            $_.FullName -notlike "*\sources\legacy.rs" -and
            $_.FullName -notlike "*\sources\mod.rs"
        } |
        Select-String -Pattern '\bSourceTerm\b|\bSourceContext\b|\bSourceContribution\b' -CaseSensitive
    if ($legacyBridgeLeaks) {
        Write-Host "[FAIL] legacy source bridge symbols leaked outside allowed bridge files:" -ForegroundColor Red
        $legacyBridgeLeaks | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($legacyBridgeLeaks.Count -gt 10) {
            Write-Host "  ... and $($legacyBridgeLeaks.Count - 10) more" -ForegroundColor Red
        }
        $errors += "legacy SourceTerm/SourceContext/SourceContribution symbols must remain confined to sources/legacy.rs and sources/mod.rs"
    } else {
        Write-Host "[OK] legacy source bridge symbols are confined to the allowed bridge files" -ForegroundColor Green
    }

    $legacySourceTopLevelExports = @()
    $legacySourceTopLevelExports += Select-String -Path "crates/mh_physics/src/sources/mod.rs" -Pattern 'SourceContribution,\s*SourceContext,\s*SourceTerm,\s*SourceHelpers' -CaseSensitive
    $legacySourceTopLevelExports += Select-String -Path "crates/mh_physics/src/lib.rs" -Pattern 'SourceContribution,\s*SourceContext,\s*SourceTerm,\s*SourceHelpers' -CaseSensitive
    if ($legacySourceTopLevelExports) {
        Write-Host "[FAIL] legacy source bridge items are still exported from top-level modules:" -ForegroundColor Red
        $legacySourceTopLevelExports | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        $errors += "legacy source bridge items must be confined to mh_physics::sources::legacy"
    } else {
        Write-Host "[OK] legacy source bridge exports are confined to mh_physics::sources::legacy" -ForegroundColor Green
    }

    $legacySourceNamespace = Select-String -Path "crates/mh_physics/src/sources/mod.rs" -Pattern '^\s*pub mod legacy;' -CaseSensitive
    if (-not $legacySourceNamespace) {
        Write-Host "[FAIL] sources::legacy namespace is missing" -ForegroundColor Red
        $errors += "crates/mh_physics/src/sources/mod.rs must expose an explicit legacy namespace"
    } else {
        Write-Host "[OK] sources::legacy namespace is present" -ForegroundColor Green
    }

    $sourceCpuBackendLeakage = Get-ChildItem -Path "crates/mh_physics/src/sources" -Recurse -Filter "*.rs" -File |
        Where-Object {
            $_.FullName -notlike "*\sources\legacy.rs"
        } |
        Select-String -Pattern 'ShallowWaterState<CpuBackend<f64>>|SourceTermGeneric::<CpuBackend<f64>>|ShallowWaterState::<CpuBackend<f64>>::new_with_backend' -CaseSensitive
    if ($sourceCpuBackendLeakage) {
        Write-Host "[FAIL] CpuBackend<f64> source residue leaked outside sources/legacy.rs:" -ForegroundColor Red
        $sourceCpuBackendLeakage | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($sourceCpuBackendLeakage.Count -gt 10) {
            Write-Host "  ... and $($sourceCpuBackendLeakage.Count - 10) more" -ForegroundColor Red
        }
        $errors += "CpuBackend<f64> source residue must remain confined to crates/mh_physics/src/sources/legacy.rs"
    } else {
        Write-Host "[OK] CpuBackend<f64> source residue is confined to sources/legacy.rs" -ForegroundColor Green
    }

    $publicLegacyLimiterRoot = Select-String -Path "crates/mh_physics/src/lib.rs" -Pattern '^\s*pub mod limiters;' -CaseSensitive
    if ($publicLegacyLimiterRoot) {
        Write-Host "[FAIL] public root-level limiters shim is still exposed from lib.rs" -ForegroundColor Red
        $publicLegacyLimiterRoot | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        $errors += "public root-level crate::limiters shim must be removed from crates/mh_physics/src/lib.rs"
    } else {
        Write-Host "[OK] public root-level limiters shim is removed" -ForegroundColor Green
    }

    $legacyLimitersNamespace = Select-String -Path "crates/mh_physics/src/lib.rs" -Pattern '^\s*pub mod legacy_limiters \{' -CaseSensitive
    if (-not $legacyLimitersNamespace) {
        Write-Host "[FAIL] legacy_limiters namespace is missing" -ForegroundColor Red
        $errors += "crates/mh_physics/src/lib.rs must expose an explicit legacy_limiters namespace"
    } else {
        Write-Host "[OK] legacy_limiters namespace is present" -ForegroundColor Green
    }

    $legacyLimiterImports = Get-ChildItem -Path "crates/mh_physics/src" -Recurse -Filter "*.rs" -File |
        Where-Object {
            $_.FullName -notlike "*\mh_physics\src\limiters.rs" -and
            $_.FullName -notlike "*\mh_physics\src\lib.rs"
        } |
        Select-String -Pattern 'crate::limiters|limiters::LimiterType|limiters::MusclConfig|limiters::MusclReconstructor' -CaseSensitive
    if ($legacyLimiterImports) {
        Write-Host "[FAIL] legacy limiters bridge is referenced outside the allowed bridge files:" -ForegroundColor Red
        $legacyLimiterImports | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($legacyLimiterImports.Count -gt 10) {
            Write-Host "  ... and $($legacyLimiterImports.Count - 10) more" -ForegroundColor Red
        }
        $errors += "legacy limiters bridge must remain confined to crates/mh_physics/src/lib.rs and crates/mh_physics/src/limiters.rs"
    } else {
        Write-Host "[OK] legacy limiters bridge is confined to the allowed bridge files" -ForegroundColor Green
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
