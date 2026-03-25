#!/usr/bin/env pwsh
#
# MariHydro T06 unified audit entrypoint.
# It preserves the legacy script name referenced by docs and task files.
# The script chains the existing guard scripts from the repo root.
#

param(
    [switch]$Verbose,
    [switch]$Strict,
    [switch]$Deep
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MariHydro unified audit (T06)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

function Invoke-GuardStep {
    param(
        [string]$Name,
        [string]$Path,
        [hashtable]$Arguments
    )

    if (-not (Test-Path $Path)) {
        Write-Host "[SKIP] $Name missing" -ForegroundColor Yellow
        return $false
    }

    Write-Host ""
    Write-Host "=== Running: $Name ===" -ForegroundColor Cyan
    & $Path @Arguments
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        Write-Host "[FAIL] $Name (exit code: $ExitCode)" -ForegroundColor Red
        return $false
    }

    Write-Host "[OK] $Name" -ForegroundColor Green
    return $true
}

function Invoke-InformationalScan {
    param(
        [string]$Name,
        [string[]]$Roots,
        [string]$Pattern
    )

    $Findings = @()
    foreach ($File in (Get-RustFilesFromTargets -Targets $Roots)) {
            $Matches = Select-String -Path $File.FullName -Pattern $Pattern -CaseSensitive:$false
            foreach ($Match in $Matches) {
                $Findings += [pscustomobject]@{
                    Path = $File.FullName.Replace($ProjectRoot + "\", "")
                    Line = $Match.LineNumber
                    Text = $Match.Line.Trim()
                }
            }
    }

    if ($Findings.Count -eq 0) {
        Write-Host "[OK] $Name" -ForegroundColor Green
        return
    }

    Write-Host "[WARN] $Name findings: $($Findings.Count)" -ForegroundColor Yellow
    $Findings | Select-Object -First 5 | ForEach-Object {
        Write-Host ("  " + $_.Path + ":" + $_.Line + ": " + $_.Text) -ForegroundColor Yellow
    }
    if ($Findings.Count -gt 5) {
        Write-Host "  ... and $($Findings.Count - 5) more" -ForegroundColor Yellow
    }
}

function Invoke-FailingScan {
    param(
        [string]$Name,
        [string[]]$Roots,
        [string]$Pattern
    )

    $Findings = @()
    foreach ($File in (Get-RustFilesFromTargets -Targets $Roots)) {
            $Matches = Select-String -Path $File.FullName -Pattern $Pattern -CaseSensitive
            foreach ($Match in $Matches) {
                $Findings += [pscustomobject]@{
                    Path = $File.FullName.Replace($ProjectRoot + "\", "")
                    Line = $Match.LineNumber
                    Text = $Match.Line.Trim()
                }
            }
    }

    if ($Findings.Count -eq 0) {
        Write-Host "[OK] $Name" -ForegroundColor Green
        return $true
    }

    Write-Host "[FAIL] $Name findings: $($Findings.Count)" -ForegroundColor Red
    $Findings | Select-Object -First 10 | ForEach-Object {
        Write-Host ("  " + $_.Path + ":" + $_.Line + ": " + $_.Text) -ForegroundColor Red
    }
    if ($Findings.Count -gt 10) {
        Write-Host "  ... and $($Findings.Count - 10) more" -ForegroundColor Red
    }
    return $false
}

function Get-RustFilesFromTargets {
    param(
        [string[]]$Targets
    )

    $Files = @()
    foreach ($Target in $Targets) {
        $AbsoluteTarget = Join-Path $ProjectRoot $Target
        if (-not (Test-Path $AbsoluteTarget)) {
            continue
        }

        $Item = Get-Item $AbsoluteTarget
        if ($Item.PSIsContainer) {
            $Files += Get-ChildItem -Path $Item.FullName -Recurse -Filter "*.rs" -File
        } elseif ($Item.Extension -eq ".rs") {
            $Files += $Item
        }
    }

    return $Files | Sort-Object FullName -Unique
}

$Failed = @()

Push-Location $ProjectRoot
try {
    $VerifyArgs = @{}
    if ($Verbose) {
        $VerifyArgs.Verbose = $true
    }
    if ($Strict) {
        $VerifyArgs.Strict = $true
    }
    if (-not (Invoke-GuardStep -Name "verify_architecture.ps1" -Path (Join-Path $ScriptDir "verify_architecture.ps1") -Arguments $VerifyArgs)) {
        $Failed += "verify_architecture.ps1"
    }

    $GlobalStateArgs = @{}
    if ($Verbose) {
        $GlobalStateArgs.Verbose = $true
    }
    if (-not (Invoke-GuardStep -Name "check_global_state.ps1" -Path (Join-Path $ScriptDir "check_global_state.ps1") -Arguments $GlobalStateArgs)) {
        $Failed += "check_global_state.ps1"
    }

    $IndexArgs = @{}
    if ($Verbose) {
        $IndexArgs.Verbose = $true
    }
    if (-not (Invoke-GuardStep -Name "check_index_uniqueness.ps1" -Path (Join-Path $ScriptDir "check_index_uniqueness.ps1") -Arguments $IndexArgs)) {
        $Failed += "check_index_uniqueness.ps1"
    }

    $TrackedTempArgs = @{}
    if ($Verbose) {
        $TrackedTempArgs.Verbose = $true
    }
    if (-not (Invoke-GuardStep -Name "check_tracked_temp_artifacts.ps1" -Path (Join-Path $ScriptDir "check_tracked_temp_artifacts.ps1") -Arguments $TrackedTempArgs)) {
        $Failed += "check_tracked_temp_artifacts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_repo_contracts.ps1" -Path (Join-Path $ScriptDir "check_repo_contracts.ps1") -Arguments @{})) {
        $Failed += "check_repo_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_physics_provenance.ps1" -Path (Join-Path $ScriptDir "check_physics_provenance.ps1") -Arguments @{})) {
        $Failed += "check_physics_provenance.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_real_implementation_contracts.ps1" -Path (Join-Path $ScriptDir "check_real_implementation_contracts.ps1") -Arguments @{})) {
        $Failed += "check_real_implementation_contracts.ps1"
    }

    if ($Deep) {
        $HardcodedArgs = @{}
        if ($Verbose) {
            $HardcodedArgs.Verbose = $true
        }
        if (-not (Invoke-GuardStep -Name "check_hardcoded_f64.ps1" -Path (Join-Path $ScriptDir "check_hardcoded_f64.ps1") -Arguments $HardcodedArgs)) {
            $Failed += "check_hardcoded_f64.ps1"
        }
    }

    Write-Host ""
    Write-Host "=== Blocking scalar conversion scan ===" -ForegroundColor Cyan
    if (-not (Invoke-FailingScan -Name "raw scalar_from_f64 call usage" -Roots @("crates/mh_physics/src") -Pattern '(?<!try_)\bscalar_from_f64\(')) {
        $Failed += "raw scalar_from_f64 call usage"
    }
    if (-not (Invoke-FailingScan -Name "silent from_config unwrap_or usage (guarded conversion roots)" -Roots @("crates/mh_physics/src/numerics/gradient", "crates/mh_physics/src/numerics/limiter", "crates/mh_physics/src/numerics/reconstruction", "crates/mh_physics/src/mesh", "crates/mh_physics/src/assimilation/bridge.rs", "crates/mh_physics/src/assimilation/conservation.rs", "crates/mh_physics/src/assimilation/mod.rs", "crates/mh_physics/src/boundary/ghost.rs", "crates/mh_physics/src/boundary/manager.rs", "crates/mh_physics/src/boundary/types.rs", "crates/mh_physics/src/config_bridge.rs", "crates/mh_physics/src/engine/timestep.rs", "crates/mh_physics/src/engine/time_integrator.rs", "crates/mh_physics/src/engine/strategy/explicit.rs", "crates/mh_physics/src/engine/strategy/mod.rs", "crates/mh_physics/src/numerics/linear_algebra/solver.rs", "crates/mh_physics/src/numerics/linear_algebra/csr.rs", "crates/mh_physics/src/schemes/riemann/hllc.rs", "crates/mh_physics/src/vertical/profile.rs", "crates/mh_physics/src/boundary/traits.rs", "crates/mh_physics/src/builder/solver_builder.rs", "crates/mh_physics/src/engine/parallel.rs", "crates/mh_physics/src/engine/strategy/semi_implicit.rs", "crates/mh_physics/src/schemes/riemann/traits.rs", "crates/mh_physics/src/schemes/wetting_drying/handler.rs", "crates/mh_physics/src/sources/coriolis.rs", "crates/mh_physics/src/tracer/transport.rs") -Pattern '\bfrom_config\(.*\)\.unwrap_or\(')) {
        $Failed += "silent from_config unwrap_or usage (guarded conversion roots)"
    }
    if (-not (Invoke-FailingScan -Name "raw if-let from_config option usage" -Roots @("crates/mh_physics/src") -Pattern 'if let Some\([^\)]*\) = .*::from_config\(')) {
        $Failed += "raw if-let from_config option usage"
    }
    if (-not (Invoke-FailingScan -Name "silent from_f64/from_f32 unwrap_or usage" -Roots @("crates/mh_physics/src") -Pattern '\bfrom_f(?:64|32)\(.*\)\.unwrap_or\(')) {
        $Failed += "silent from_f64/from_f32 unwrap_or usage"
    }
    if (-not (Invoke-FailingScan -Name "silent from_f64/from_f32 unwrap_or usage (mh_runtime/src)" -Roots @("crates/mh_runtime/src") -Pattern '\bfrom_f(?:64|32)\(.*\)\.unwrap_or\(')) {
        $Failed += "silent from_f64/from_f32 unwrap_or usage (mh_runtime/src)"
    }

    Write-Host ""
    Write-Host "=== Advisory scans ===" -ForegroundColor Cyan
    Invoke-InformationalScan -Name "placeholder wording residue" -Roots @("crates", "apps") -Pattern "\bplaceholder\b|\bfake implementation\b|\bstub\b"
    Invoke-InformationalScan -Name "compatibility residue" -Roots @("crates", "apps") -Pattern "\blegacy_limiters\b|\bsources::legacy\b|\bcompatibility shim\b|\bcompatibility namespace\b|\bhistorical compatibility\b"
    Invoke-InformationalScan -Name "fake model naming residue" -Roots @("crates", "apps") -Pattern "\bWhiteColebrook\b|\bNaturalNeighborInterpolator\b|\bNaturalNeighborConfig\b|\bSolverBuilder\b|\bSimpleSolver\b|Box<dyn DynSolver>"
    Invoke-InformationalScan -Name "physical formula risk wording" -Roots @("crates/mh_physics", "crates/mh_agent", "apps") -Pattern "^(?!.*PHYSICS_(?:SOURCE|SCOPE):).*\b(?:empirical|simplified|approximate|experimental|uncalibrated|unverified|temporary)\b"
    Invoke-InformationalScan -Name "remote sensing hardcoded calibration residue" -Roots @("crates/mh_agent") -Pattern "modis_red_band|sentinel2_b4|empirical_inversion|SensorType::Optical => .*10\.0|SensorType::SAR => .*5\.0|SensorType::Hyperspectral => .*8\.0"
    Invoke-InformationalScan -Name "surrogate fake model surface residue" -Roots @("crates/mh_agent") -Pattern "\bSurrogateType\b|\bReducedOrder\b|\bGaussianProcess\b|\bPolynomialChaos\b|\bUnsupportedModelType\b|only LinearRegression is implemented"
    Invoke-InformationalScan -Name "gpu placeholder surface residue" -Roots @("crates/mh_physics") -Pattern "\bCudaBackendPlaceholder\b|\bGpuStatus\b|\bGpuCapabilities\b|pub mod gpu;|\bhas_cuda\b|\bavailable_gpus\b"
    Invoke-InformationalScan -Name "source hydraulic preset residue" -Roots @("crates/mh_physics/src/sources") -Pattern "\bFlexible\b|\breed\(\)|\bmangrove\(\)|with_reed_zone|with_mangrove_zone|update_from_radiation_stress|compute_effective_shear\(|compute_gradient_simple"
    Invoke-InformationalScan -Name "wave bottom friction preset residue" -Roots @("crates/mh_physics/src/waves/bottom_friction.rs") -Pattern "\bJonswap\b|\bjonswap\("
    Invoke-InformationalScan -Name "spectral misleading naming residue" -Roots @("crates/mh_physics/src/waves/spectral.rs") -Pattern "\bfrom_jonswap\(|JONSWAP 谱初始化"
    Invoke-InformationalScan -Name "try_scalar_from_f64 explicit-path usage" -Roots @("crates/mh_physics") -Pattern "\btry_scalar_from_f64\("
    Invoke-InformationalScan -Name "scalar_from_f64 symbol residue" -Roots @("crates/mh_physics") -Pattern "\bscalar_from_f64\b"
    Invoke-InformationalScan -Name "T06 unimplemented residue" -Roots @("crates/mh_geo", "crates/mh_io", "crates/mh_mesh", "crates/mh_terrain", "apps", "tests") -Pattern "unimplemented!"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($Failed.Count -eq 0) {
    Write-Host "[OK] unified audit passed" -ForegroundColor Green
    exit 0
}

Write-Host "[FAIL] unified audit failed" -ForegroundColor Red
Write-Host "Failed items: $($Failed -join ', ')" -ForegroundColor Red
exit 1
