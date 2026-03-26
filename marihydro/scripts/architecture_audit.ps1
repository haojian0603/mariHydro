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

    if (-not (Invoke-GuardStep -Name "check_external_data_contracts.ps1" -Path (Join-Path $ScriptDir "check_external_data_contracts.ps1") -Arguments @{})) {
        $Failed += "check_external_data_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_external_driver_access_contracts.ps1" -Path (Join-Path $ScriptDir "check_external_driver_access_contracts.ps1") -Arguments @{})) {
        $Failed += "check_external_driver_access_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_vtu_export_contracts.ps1" -Path (Join-Path $ScriptDir "check_vtu_export_contracts.ps1") -Arguments @{})) {
        $Failed += "check_vtu_export_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_export_metadata_contracts.ps1" -Path (Join-Path $ScriptDir "check_export_metadata_contracts.ps1") -Arguments @{})) {
        $Failed += "check_export_metadata_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_metadata_timestamp_contracts.ps1" -Path (Join-Path $ScriptDir "check_metadata_timestamp_contracts.ps1") -Arguments @{})) {
        $Failed += "check_metadata_timestamp_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_pipeline_shutdown_contracts.ps1" -Path (Join-Path $ScriptDir "check_pipeline_shutdown_contracts.ps1") -Arguments @{})) {
        $Failed += "check_pipeline_shutdown_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_geo_geodesic_contracts.ps1" -Path (Join-Path $ScriptDir "check_geo_geodesic_contracts.ps1") -Arguments @{})) {
        $Failed += "check_geo_geodesic_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_geo_affine_contracts.ps1" -Path (Join-Path $ScriptDir "check_geo_affine_contracts.ps1") -Arguments @{})) {
        $Failed += "check_geo_affine_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_import_contracts.ps1" -Path (Join-Path $ScriptDir "check_import_contracts.ps1") -Arguments @{})) {
        $Failed += "check_import_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_geo_projection_contracts.ps1" -Path (Join-Path $ScriptDir "check_geo_projection_contracts.ps1") -Arguments @{})) {
        $Failed += "check_geo_projection_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_spatial_index_contracts.ps1" -Path (Join-Path $ScriptDir "check_spatial_index_contracts.ps1") -Arguments @{})) {
        $Failed += "check_spatial_index_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_web_mercator_contracts.ps1" -Path (Join-Path $ScriptDir "check_web_mercator_contracts.ps1") -Arguments @{})) {
        $Failed += "check_web_mercator_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_io_invariant_contracts.ps1" -Path (Join-Path $ScriptDir "check_io_invariant_contracts.ps1") -Arguments @{})) {
        $Failed += "check_io_invariant_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_ai_state_contracts.ps1" -Path (Join-Path $ScriptDir "check_ai_state_contracts.ps1") -Arguments @{})) {
        $Failed += "check_ai_state_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_runtime_probe_contracts.ps1" -Path (Join-Path $ScriptDir "check_runtime_probe_contracts.ps1") -Arguments @{})) {
        $Failed += "check_runtime_probe_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_runtime_parallelism_contracts.ps1" -Path (Join-Path $ScriptDir "check_runtime_parallelism_contracts.ps1") -Arguments @{})) {
        $Failed += "check_runtime_parallelism_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_runtime_topology_contracts.ps1" -Path (Join-Path $ScriptDir "check_runtime_topology_contracts.ps1") -Arguments @{})) {
        $Failed += "check_runtime_topology_contracts.ps1"
    }

    if (-not (Invoke-GuardStep -Name "check_runtime_allocator_contracts.ps1" -Path (Join-Path $ScriptDir "check_runtime_allocator_contracts.ps1") -Arguments @{})) {
        $Failed += "check_runtime_allocator_contracts.ps1"
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
    if (-not (Invoke-FailingScan -Name "source text corruption residue" -Roots @("crates", "apps") -Pattern '[\uE000-\uF8FF\uFFFD]')) {
        $Failed += "source text corruption residue"
    }
    if (-not (Invoke-FailingScan -Name "question-mark text corruption residue" -Roots @("crates", "apps") -Pattern '\?{3,}')) {
        $Failed += "question-mark text corruption residue"
    }

    if (-not (Invoke-FailingScan -Name "mixed-script text corruption residue" -Roots @("crates", "apps") -Pattern '[\u0400-\u04FF\u20AC\u3220-\u3229\uFF21-\uFF3A\uFF41-\uFF5A]')) {
        $Failed += "mixed-script text corruption residue"
    }

    Write-Host ""
    Write-Host "=== Advisory scans ===" -ForegroundColor Cyan
    Invoke-InformationalScan -Name 'placeholder wording residue' -Roots @("crates", "apps") -Pattern '\bplaceholder\b|\bfake implementation\b|\bstub\b'
    Invoke-InformationalScan -Name 'compatibility residue' -Roots @("crates", "apps") -Pattern '\blegacy_limiters\b|\bsources::legacy\b|\bcompatibility shim\b|\bcompatibility namespace\b|\bhistorical compatibility\b'
    Invoke-InformationalScan -Name 'fake model naming residue' -Roots @("crates", "apps") -Pattern '\bWhiteColebrook\b|\bNaturalNeighborInterpolator\b|\bNaturalNeighborConfig\b|\bSolverBuilder\b|\bSimpleSolver\b|Box<dyn DynSolver>|\bReflectanceOperator\b|\bReflectanceCalibration\b|\bInversionModel\b|\bDietrichSettling\b'
    Invoke-InformationalScan -Name 'physical formula risk wording' -Roots @("crates/mh_physics", "crates/mh_agent", "apps") -Pattern '^(?!.*PHYSICS_(?:SOURCE|SCOPE):).*\b(?:empirical|simplified|approximate|experimental|uncalibrated|unverified|temporary)\b'
    Invoke-InformationalScan -Name 'formula regression residue' -Roots @("crates/mh_physics") -Pattern 'let f = \(d_star - 1\.0\) / 99\.0|let f = \(d_star - d_star_1\) / cfg\(99\.0\)|self\.longitudinal \* cos_theta\.abs\(\) \+ self\.transverse \* sin_theta|\bDietrichSettling\b'
    Invoke-InformationalScan -Name 'silent numeric fallback residue' -Roots @("crates/mh_geo", "crates/mh_io/src/drivers") -Pattern 'unwrap_or\(0\.0\)|unwrap_or\(0\)|unwrap_or\(Self::ZERO\)|compute_convergence_angle_checked\(x, y\)\.unwrap_or\(0\.0\)'
    Invoke-InformationalScan -Name 'geo projection sentinel residue' -Roots @("crates/mh_geo/src/projection") -Pattern 'unwrap_or\(f64::NAN\)|pub fn utm_scale_factor\(.*\) -> f64|pub fn utm_convergence_angle\(.*\) -> f64|unwrap_or\(\(0\.0, 0\.0\)\)'
    Invoke-InformationalScan -Name 'geo geodesic Option-failure residue' -Roots @("crates/mh_geo/src/geometry.rs") -Pattern 'vincenty_distance_to\(&self, other: &Self\) -> Option<f64>|vincenty_distance\(&self, other: &Self, ellipsoid: &Ellipsoid\) -> Option<f64>|return Some\(0\.0\)|Some\(s\)'
    Invoke-InformationalScan -Name 'geo affine Option-failure residue' -Roots @("crates/mh_geo/src/transform.rs") -Pattern 'pub fn inverse\(&self\) -> Option<Self>|pub fn apply_inverse\(&self, x: f64, y: f64\) -> Option<\(f64, f64\)>|return None;'
    Invoke-InformationalScan -Name 'geo finite-difference convergence residue' -Roots @("crates/mh_geo/src") -Pattern 'delta_lat\s*=|lat\s*\+\s*delta_lat|dy\.atan2\(dx\)'
    Invoke-InformationalScan -Name 'spatial index radius-query shortcut residue' -Roots @("crates/mh_geo/src/spatial_index.rs") -Pattern 'take_while\(\|entry\|'
    Invoke-InformationalScan -Name 'Web Mercator domain fallback residue' -Roots @("crates/mh_geo/src/projection/web_mercator.rs") -Pattern 'lat\.clamp\(\s*-WEB_MERCATOR_MAX_LAT\s*,\s*WEB_MERCATOR_MAX_LAT\s*\)|pub fn web_mercator_resolution\(.*\) -> f64|pub fn web_mercator_scale\(.*\) -> f64|pub fn lonlat_to_tile\(.*\) -> \(u32, u32\)|pub fn tile_to_lonlat\(.*\) -> \(f64, f64\)|pub fn tile_to_bbox\(.*\) -> \(f64, f64, f64, f64\)|expect\("tile_to_lonlat returns coordinates inside Web Mercator domain"\)'
    Invoke-InformationalScan -Name 'external data partial-parse residue' -Roots @("crates/mh_io/src/drivers") -Pattern 'token\.parse::<f64>\(\)\.ok\(\)|and_then\(\|v\| v\.parse\(\)\.ok\(\)|_ => 30|unwrap_or\(&empty_bands\)|bands\.len\(\)\.max\(1\)|parts\.next\(\)\.unwrap_or\(\"\"\)|if let Some\(space\) = cleaned\.find\('
    Invoke-InformationalScan -Name 'external CLI context-loss residue' -Roots @("crates/mh_io/src/drivers") -Pattern 'map_err\(\|_\|\s*(?:GdalError|NetCdfError)::NotAvailable|(?:OpenFailed|ReadFailed)\(\s*String::from_utf8_lossy\(&output\.stderr\)\.to_string\(\)\s*\)|NotAvailable,\s*$'
    Invoke-InformationalScan -Name 'export metadata fallback residue' -Roots @("crates/mh_io/src") -Pattern 'serde_json::to_string\(names\)\.unwrap_or_else\(\|_\| "\[\]"\.into\(\)\)|boundary_names.*\[\]|field names.*\[\]'
    Invoke-InformationalScan -Name 'metadata timestamp fallback residue' -Roots @("crates/mh_io/src") -Pattern 'duration_since\(std::time::UNIX_EPOCH\)\s*\.map\(\|d\| d\.as_secs\(\)\)\s*\.unwrap_or\(0\)|created_at:\s*0\b'
    Invoke-InformationalScan -Name 'checkpoint sentinel metadata residue' -Roots @("crates/mh_io/src/checkpoint.rs") -Pattern 'config_hash\.unwrap_or\(0\)|mesh_hash:\s*0\b|config_hash != 0|found:\s*0\s*\}|if let Ok\(header\) = Checkpoint::read_header'
    Invoke-InformationalScan -Name 'snapshot boundary sentinel residue' -Roots @("crates/mh_io/src/snapshot.rs") -Pattern 'unwrap_or\(u32::MAX\)|-1\b'
    Invoke-InformationalScan -Name 'pipeline shutdown swallow residue' -Roots @("crates/mh_io/src/pipeline.rs") -Pattern 'pub fn wait_for_completion\(&self, timeout: Duration\) -> bool|let _ = self\.flush\(\)|let _ = self\.wait_for_completion\(timeout\)|let _ = self\.sender\.send\(OutputRequest::Shutdown\)|if let Ok\(mut stats\) = self\.stats\.lock\(\)|写入超时警告|PipelineError::Timeout\(Duration::from_millis\(write_timeout_ms\)\)\s*\.into_io_error\("process_request"\)'
    Invoke-InformationalScan -Name 'GeoJSON semantic-name fallback residue' -Roots @("crates/mh_io/src/import") -Pattern 'get_string\("name"\)\.unwrap_or\("unnamed"\)|get_string\("name"\)\.unwrap_or\("zone"\)|pub fn boundary_conditions\(&self\) -> Vec<BoundaryConditionLocation>|pub fn zone_properties\(&self\) -> Vec<ZoneProperties>'
    Invoke-InformationalScan -Name 'GeoJSON multipart semantic-name synthesis residue' -Roots @("crates/mh_io/src/import") -Pattern 'format!\(\"\\{\\}_\\{\\}\", name, idx \+ 1\)'
    Invoke-InformationalScan -Name 'GeoJSON feature-id collapse residue' -Roots @("crates/mh_io/src/import") -Pattern 'rf\.id\.map\(\|v\| match v|_ => String::new\(\)|id:\s*None,'
    Invoke-InformationalScan -Name 'GeoJSON null-geometry drop residue' -Roots @("crates/mh_io/src/import/geojson.rs") -Pattern 'None => return Ok\(None\)'
    Invoke-InformationalScan -Name 'CSV tolerant-default residue' -Roots @("crates/mh_io/src/import/timeseries_csv.rs") -Pattern 'skip_invalid:\s*true,|unwrap_or_default\(\)|parts\.len\(\)\.min\(n_cols \+ 1\)'
    Invoke-InformationalScan -Name 'external shape fallback residue' -Roots @("crates/mh_io/src") -Pattern 'dims\.first\(\)\.copied\(\)\.unwrap_or_default\(\)\s*==\s*1'
    Invoke-InformationalScan -Name 'external data filename heuristic residue' -Roots @("crates/mh_io/src") -Pattern 'let model = TidalModel::detect\(path\)'
    Invoke-InformationalScan -Name 'matched layout skip residue' -Roots @("crates/mh_io/src/netcdf_tide.rs") -Pattern 'let driver = match NetCdfDriver::open\(&path\)|Err\(_\)\s*=>\s*continue|files\.entry\(constituent\)\.or_insert\(path\)|match TidalModel::infer_from_path_hint\(path\)'
    Invoke-InformationalScan -Name 'public tolerant parser residue' -Roots @("crates/mh_io/src") -Pattern 'pub fn parse_calendar_or_default\(|pub fn detect\(path: &Path\)'
    Invoke-InformationalScan -Name 'ncdump empty-header residue' -Roots @("crates/mh_io/src/drivers/netcdf/driver.rs") -Pattern 'CliHeader::default\('
    Invoke-InformationalScan -Name 'AI state fallback residue' -Roots @("crates/mh_agent") -Pattern 'let _ = model\.load_state|let _ = self\.save_state|\.unwrap_or\(0\.8\)|features\.get\(i\)\.copied\(\)\.unwrap_or\(0\.0\)|pred\.get\(o\)\.copied\(\)\.unwrap_or\(0\.0\)|target_norm\.get\(o\)\.copied\(\)\.unwrap_or\(0\.0\)|norm\.(mean|std|m2)\.get\(.*\)\.copied\(\)\.unwrap_or\((0\.0|1\.0)\)|pred\.values\.len\(\)\.min\(cell_areas\.len\(\)\)'
    Invoke-InformationalScan -Name 'remote sensing hardcoded calibration residue' -Roots @("crates/mh_agent") -Pattern 'modis_red_band|sentinel2_b4|empirical_inversion|SensorType::Optical => .*10\.0|SensorType::SAR => .*5\.0|SensorType::Hyperspectral => .*8\.0'
    Invoke-InformationalScan -Name 'surrogate fake model surface residue' -Roots @("crates/mh_agent") -Pattern '\bSurrogateType\b|\bReducedOrder\b|\bGaussianProcess\b|\bPolynomialChaos\b|\bUnsupportedModelType\b|only LinearRegression is implemented'
    Invoke-InformationalScan -Name 'gpu placeholder surface residue' -Roots @("crates/mh_physics") -Pattern '\bCudaBackendPlaceholder\b|\bGpuStatus\b|\bGpuCapabilities\b|pub mod gpu;|\bhas_cuda\b|\bavailable_gpus\b'
    Invoke-InformationalScan -Name 'source hydraulic preset residue' -Roots @("crates/mh_physics/src/sources") -Pattern '\bFlexible\b|\breed\(\)|\bmangrove\(\)|with_reed_zone|with_mangrove_zone|update_from_radiation_stress|compute_effective_shear\(|compute_gradient_simple'
    Invoke-InformationalScan -Name 'source semantic fallback residue' -Roots @("crates/mh_physics/src/sources/vegetation.rs") -Pattern 'effective_drag\(&self,\s*water_depth: f64,\s*velocity: f64\)|pub rho_water: f64|if cell < self\.vegetation\.len\(\)|unwrap_or\(VegetationType::None\)|unwrap_or\(B::Scalar::ONE\)'
    Invoke-InformationalScan -Name 'wave bottom friction preset residue' -Roots @("crates/mh_physics/src/waves/bottom_friction.rs") -Pattern '\bJonswap\b|\bjonswap\('
    Invoke-InformationalScan -Name 'spectral misleading naming residue' -Roots @("crates/mh_physics/src/waves/spectral.rs") -Pattern '\bfrom_jonswap\(|JONSWAP 谱初始化'
    Invoke-InformationalScan -Name 'runtime probe fallback residue' -Roots @("crates/mh_runtime/src") -Pattern 'read_to_string\(&cpulist_path\)\.unwrap_or_default\(\)|read_to_string\(&meminfo_path\)\.unwrap_or_default\(\)|parse\(\)\.unwrap_or\(0\)|parse::<u64>\(\)\.unwrap_or\(0\)|8 \* 1024 \* 1024 \* 1024|4 \* 1024 \* 1024 \* 1024'
    Invoke-InformationalScan -Name 'runtime parallelism fallback residue' -Roots @("crates/mh_runtime/src") -Pattern 'available_parallelism\(\)[^\r\n;]*unwrap_or\((?:1|1usize)\)|available_parallelism\(\)[^\r\n;]*unwrap_or_default\(\)|parse::<usize>\(\)\.ok\(\)'
    Invoke-InformationalScan -Name 'runtime topology default residue' -Roots @("crates/mh_runtime/src") -Pattern 'impl Default for NumaTopology|impl Default for NumaThreadPoolConfig|NumaTopology::default\(\)|NumaThreadPoolConfig::default\(\)|unwrap_or_else\(\|_\| Self \{'
    Invoke-InformationalScan -Name 'runtime allocator node fabrication residue' -Roots @("crates/mh_runtime/src") -Pattern 'fn get_node\(&self, _ptr: \*const u8\) -> Option<usize> \{\s*Some\(0\)'
    Invoke-InformationalScan -Name 'try_scalar_from_f64 explicit-path usage' -Roots @("crates/mh_physics") -Pattern '\btry_scalar_from_f64\('
    Invoke-InformationalScan -Name 'scalar_from_f64 symbol residue' -Roots @("crates/mh_physics") -Pattern '\bscalar_from_f64\b'
    Invoke-InformationalScan -Name 'T06 unimplemented residue' -Roots @("crates/mh_geo", "crates/mh_io", "crates/mh_mesh", "crates/mh_terrain", "apps", "tests") -Pattern 'unimplemented!'
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
