#!/usr/bin/env pwsh
#
# Blocking guard for fake features, compatibility namespaces and false public promises.
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

Write-Host "=== Checking real implementation contracts ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

Push-Location $ProjectRoot
try {
    $forbiddenPaths = @(
        "crates/mh_physics/src/sources/legacy.rs",
        "crates/mh_physics/src/legacy_limiters",
        "crates/mh_terrain/src/interpolation/natural_neighbor.rs",
        "crates/mh_physics/src/builder/solver_builder.rs",
        "crates/mh_mesh/src/compat.rs"
    )

    foreach ($relativePath in $forbiddenPaths) {
        $absolutePath = Join-Path $ProjectRoot $relativePath
        if (Test-Path $absolutePath) {
            Add-Failure "forbidden compatibility or fake implementation path exists: $relativePath"
        } else {
            Write-Host "[OK] $relativePath is absent" -ForegroundColor Green
        }
    }

    $patternChecks = @(
        @{ Name = "root legacy limiters namespace"; Path = "crates/mh_physics/src/lib.rs"; Pattern = '^\s*pub mod legacy_limiters;'; Message = "mh_physics root must not expose legacy_limiters" },
        @{ Name = "sources legacy namespace"; Path = "crates/mh_physics/src/sources/mod.rs"; Pattern = '^\s*pub mod legacy;'; Message = "sources::legacy must not reappear" },
        @{ Name = "terrain natural neighbor export"; Path = "crates/mh_terrain/src/interpolation/mod.rs"; Pattern = '^\s*pub mod natural_neighbor;|NaturalNeighborConfig|NaturalNeighborInterpolator'; Message = "fake Natural Neighbor exports must stay removed" },
        @{ Name = "friction fake White-Colebrook"; Path = "crates/mh_physics/src/friction.rs"; Pattern = '\bWhiteColebrook\b'; Message = "WhiteColebrook must not exist until a verified formula is implemented" },
        @{ Name = "builder fake solver export"; Path = "crates/mh_physics/src/builder/mod.rs"; Pattern = '\bSolverBuilder\b|\bSolverHandle\b|\bBuildError\b|solver_builder'; Message = "builder fake solver surface must stay removed" },
        @{ Name = "cli dyn-solver promise"; Path = "apps/mh_cli/src/main.rs"; Pattern = 'Box<dyn DynSolver>|SolverBuilder'; Message = "mh_cli main docs must not promise a dyn-solver or SolverBuilder path" },
        @{ Name = "cli run fake builder path"; Path = "apps/mh_cli/src/commands/run.rs"; Pattern = 'Box<dyn DynSolver>|\bSolverBuilder\b'; Message = "run command must use the real solver path" },
        @{ Name = "config dyn-solver false promise"; Path = "crates/mh_config/src/lib.rs"; Pattern = 'impl DynSolver for|Box<dyn DynSolver>'; Message = "mh_config docs must not claim a DynSolver implementation that is absent" },
        @{ Name = "agent canned optical or fake SAR presets"; Path = "crates/mh_agent/src/observation.rs"; Pattern = 'modis_red_band|sentinel2_b4|\bSAROperator\b|\bPolarization\b'; Message = "agent observation module must not ship canned sensor coefficients or default SAR proxy models" },
        @{ Name = "agent hardcoded remote sensing inversion"; Path = "crates/mh_agent/src/remote_sensing.rs"; Pattern = '\bmodel_path\b|\bempirical_inversion\b|SensorType::Optical => .*10\.0|SensorType::SAR => .*5\.0|SensorType::Hyperspectral => .*8\.0'; Message = "remote sensing agent must require explicit inversion calibration instead of hardcoded coefficients" },
        @{ Name = "vegetation preset shortcuts"; Path = "crates/mh_physics/src/sources/vegetation.rs"; Pattern = '\bFlexible\b|\breed\(\)|\bmangrove\(\)|with_reed_zone|with_mangrove_zone'; Message = "vegetation source must not expose unverified preset vegetation shortcuts or unsourced flexible formulas" },
        @{ Name = "wave forcing fake stress helpers"; Path = "crates/mh_physics/src/sources/wave_forcing.rs"; Pattern = 'update_from_radiation_stress|compute_effective_shear\('; Message = "wave forcing must not expose misleading helpers that pretend to compute verified stress products" },
        @{ Name = "wave radiation placeholder gradient"; Path = "crates/mh_physics/src/sources/wave_source.rs"; Pattern = 'compute_gradient_simple'; Message = "wave radiation source must not keep placeholder gradient APIs" },
        @{ Name = "wave bottom friction JONSWAP preset"; Path = "crates/mh_physics/src/waves/bottom_friction.rs"; Pattern = '\bJonswap\b|\bjonswap\('; Message = "wave bottom friction must not expose unsourced JONSWAP preset coefficients" },
        @{ Name = "wave spectral misleading JONSWAP shortcut"; Path = "crates/mh_physics/src/waves/spectral.rs"; Pattern = '\bfrom_jonswap\(|JONSWAP 谱初始化'; Message = "spectral helpers must name JONSWAP frequency spectrum plus directional spreading honestly" },
        @{ Name = "sources bridge narrative"; Path = "crates/mh_physics/src/sources/mod.rs"; Pattern = 'CPU/f64.*retain|retain.*CPU/f64|\bcompat'; Message = "sources module docs must not claim bridge retention" },
        @{ Name = "sources registry bridge narrative"; Path = "crates/mh_physics/src/sources/registry.rs"; Pattern = 'sources/legacy|retain.*bridge|\bcompat'; Message = "sources registry comments must not mention removed bridge paths" },
        @{ Name = "pcg compatibility narrative"; Path = "crates/mh_physics/src/engine/pcg.rs"; Pattern = '\bcompat'; Message = "pcg docs must not describe a compatibility layer" }
    )

    foreach ($check in $patternChecks) {
        $absolutePath = Join-Path $ProjectRoot $check.Path
        if (-not (Test-Path $absolutePath)) {
            continue
        }
        $matches = Select-String -Path $absolutePath -Pattern $check.Pattern -CaseSensitive
        if ($matches) {
            Add-Failure $check.Message
            $matches | Select-Object -First 5 | ForEach-Object {
                Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
            }
        } else {
            Write-Host "[OK] $($check.Name)" -ForegroundColor Green
        }
    }

    $productionFiles = Get-ChildItem -Path (Join-Path $ProjectRoot "crates"), (Join-Path $ProjectRoot "apps") -Recurse -Filter "*.rs" -File |
        Where-Object {
            $_.FullName -notlike "*\tests\*" -and
            $_.FullName -notlike "*\benches\*"
        }

    $placeholderMatches = $productionFiles | Select-String -Pattern '\bplaceholder\b|\bfake implementation\b|\bstub\b' -CaseSensitive:$false
    if ($placeholderMatches) {
        Add-Failure "placeholder wording exists in production source"
        $placeholderMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
    } else {
        Write-Host "[OK] no placeholder wording in production source" -ForegroundColor Green
    }

    $compatibilityMatches = $productionFiles | Select-String -Pattern '\blegacy_limiters\b|\bsources::legacy\b|\bcompatibility shim\b|\bcompatibility namespace\b|\bhistorical compatibility\b' -CaseSensitive:$false
    if ($compatibilityMatches) {
        Add-Failure "compatibility residue exists in production source"
        $compatibilityMatches | Select-Object -First 10 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
    } else {
        Write-Host "[OK] no compatibility residue in production source" -ForegroundColor Green
    }

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] real implementation contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] real implementation contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
