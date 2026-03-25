#!/usr/bin/env pwsh
#
# Blocking guard for external-data readers and driver fallbacks.
# Ensures supported layouts are stated explicitly and parsing failures do not
# collapse into synthetic fills or silent numeric defaults.
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

function Check-TagSet {
    param(
        [string]$RelativePath,
        [string]$Label
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required external-data file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match 'IO_SOURCE:\s*\S.{20,}') {
        Write-Host "[OK] $Label source tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SOURCE describing the real upstream layout or tool contract"
    }

    if ($content -match 'IO_SCOPE:\s*\S.{20,}') {
        Write-Host "[OK] $Label scope tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SCOPE describing supported layouts and explicit failure behavior"
    }
}

function Check-PatternAbsent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required external-data file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Add-Failure $Message
        $matches | Select-Object -First 5 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking external data contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-TagSet -RelativePath "crates/mh_io/src/netcdf_tide.rs" -Label "netcdf_tide reader"
    Check-TagSet -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" -Label "GDAL CLI driver fallback"
    Check-TagSet -RelativePath "crates/mh_io/src/drivers/netcdf/mod.rs" -Label "NetCDF driver module"
    Check-TagSet -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" -Label "NetCDF CLI driver fallback"
    Check-TagSet -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" -Label "CF time parser"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern '模拟实现|vec!\[vec!\[0\.0; n_lon\]; n_lat\]|Ok\(\(0\.0, 0\.0\)\)|Ok\(\(\(0\.0, 0\.0\), \(0\.0, 0\.0\)\)\)|尝试作为 TPXO 格式打开' `
        -Message "tide reader must not synthesize zero-valued constituents or fallback layouts"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'let model = TidalModel::detect\(path\)' `
        -Message "tide reader dispatch must not rely on filename heuristics when a real path is available"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'dims\.first\(\)\.copied\(\)\.unwrap_or_default\(\)\s*==\s*1' `
        -Message "tide reader must not guess leading singleton axes from synthetic default dimensions"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/netcdf_tide.rs" `
        -Pattern 'pub fn detect\(path: &Path\)' `
        -Message "tide reader must not expose a public filename-heuristic model detector"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'and_then\(\|v\| v\.parse\(\)\.ok\(\)\)' `
        -Message "GDAL CLI fallback must not silently discard invalid NoData metadata"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'if let Ok\(v\) = token\.parse::<f64>\(\)' `
        -Message "GDAL CLI fallback must not silently drop invalid raster payload tokens"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/gdal/driver.rs" `
        -Pattern 'unwrap_or\(&empty_bands\)|bands\.len\(\)\.max\(1\)' `
        -Message "GDAL metadata parsing must not invent a synthetic single-band layout when bands are missing"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'parse::<usize>\(\)\.ok\(\)\.unwrap_or\(0\)' `
        -Message "NetCDF header parsing must not collapse invalid dimensions to zero"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'token\.parse::<f64>\(\)\.ok\(\)' `
        -Message "NetCDF CLI fallback must not silently skip invalid numeric payload tokens"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/driver.rs" `
        -Pattern 'parts\.next\(\)\.unwrap_or\(\"\"\)|if let Some\(space\) = cleaned\.find\(' `
        -Message "NetCDF header parsing must not skip malformed variable declarations or attributes by filling empty placeholders"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" `
        -Pattern 'parse\(\)\.ok\(\)\.unwrap_or\((?:0|0\.0)\)' `
        -Message "CF time parsing must not collapse malformed components to zero"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" `
        -Pattern '_ => 30' `
        -Message "CF calendar month length must not fall back to a synthetic default month"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/drivers/netcdf/time.rs" `
        -Pattern 'pub fn parse_calendar_or_default\(' `
        -Message "CF calendar parser must not expose a public default-on-failure entrypoint"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] external data contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] external data contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
