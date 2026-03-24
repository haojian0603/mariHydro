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

    $silentConfigFallbacks = @(
        Get-ChildItem -Path "crates/mh_physics/src/numerics/gradient", "crates/mh_physics/src/numerics/limiter", "crates/mh_physics/src/numerics/reconstruction", "crates/mh_physics/src/mesh" -Recurse -Filter "*.rs" -File |
            Select-String -Pattern '\bfrom_config\(.*\)\.unwrap_or\(' -CaseSensitive
    )
    if ($silentConfigFallbacks) {
        Write-Host "[FAIL] silent from_config(...).unwrap_or(...) fallbacks exist in numerics gradient/limiter/reconstruction or mesh:" -ForegroundColor Red
        $silentConfigFallbacks | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
        if ($silentConfigFallbacks.Count -gt 10) {
            Write-Host "  ... and $($silentConfigFallbacks.Count - 10) more" -ForegroundColor Red
        }
        $errors += "silent from_config(...).unwrap_or(...) fallbacks must be removed from mh_physics/src/numerics/{gradient,limiter,reconstruction} and mh_physics/src/mesh"
    } else {
        Write-Host "[OK] no silent from_config(...).unwrap_or(...) fallbacks in mh_physics/src/numerics/{gradient,limiter,reconstruction} and mh_physics/src/mesh" -ForegroundColor Green
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
