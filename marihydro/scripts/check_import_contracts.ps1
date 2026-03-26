#!/usr/bin/env pwsh
#
# Blocking guard for import-layer structural truthfulness.
# Ensures vector importers reject incomplete geometry structure instead of
# fabricating empty shells or silently collapsing malformed layouts.
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
        Add-Failure "missing required import file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match 'IO_SOURCE:\s*\S.{20,}') {
        Write-Host "[OK] $Label source tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SOURCE describing the real upstream geometry contract"
    }

    if ($content -match 'IO_SCOPE:\s*\S.{20,}') {
        Write-Host "[OK] $Label scope tag" -ForegroundColor Green
    } else {
        Add-Failure "$RelativePath must include IO_SCOPE describing explicit failure behavior for invalid structure"
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
        Add-Failure "missing required import file: $RelativePath"
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

function Check-PatternPresent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required import file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking import contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Check-TagSet -RelativePath "crates/mh_io/src/import/mod.rs" -Label "import module"
    Check-TagSet -RelativePath "crates/mh_io/src/import/geojson.rs" -Label "GeoJSON importer"
    Check-TagSet -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" -Label "CSV time-series importer"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'rings\.first\(\)\.cloned\(\)\.unwrap_or_default\(\)' `
        -Message "GeoJSON Polygon and MultiPolygon import must not synthesize empty exterior rings"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'get_string\("name"\)\.unwrap_or\("unnamed"\)|get_string\("name"\)\.unwrap_or\("zone"\)' `
        -Message "GeoJSON semantic boundary/zone names must not be synthesized with fallback strings"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'format!\(\"\\{\\}_\\{\\}\", name, idx \+ 1\)' `
        -Message "GeoJSON multipart semantic features must not fabricate suffixed names like name_1/name_2"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'rf\.id\.map\(\|v\| match v|_ => String::new\(\)' `
        -Message "GeoJSON Feature.id must not collapse invalid ids into empty strings or ad-hoc string synthesis"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'id:\s*None,' `
        -Message "Top-level GeoJSON Feature parsing must not drop an explicit Feature.id"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'None => return Ok\(None\)' `
        -Message "GeoJSON importer must not silently drop null Feature geometry"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'InvalidStructure\(String\)' `
        -Message "GeoJSON importer must expose an explicit InvalidStructure error for malformed geometry"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'pub fn boundary_conditions\(&self\) -> Result<Vec<BoundaryConditionLocation>, GeoJsonError>|pub fn zone_properties\(&self\) -> Result<Vec<ZoneProperties>, GeoJsonError>' `
        -Message "GeoJSON semantic feature extraction must return explicit Result contracts"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'test_boundary_conditions_require_name|test_zone_properties_require_name' `
        -Message "GeoJSON importer must keep regression tests for missing semantic names"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'part_index: Option<usize>' `
        -Message "GeoJSON multipart semantic features must expose an explicit part_index field"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'test_multipolygon_preserves_name_and_tracks_part_index' `
        -Message "GeoJSON multipart semantic features must keep regression coverage for name preservation and part indices"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'fn parse_feature_id\(value: serde_json::Value\) -> Result<String, GeoJsonError>' `
        -Message "GeoJSON importer must use an explicit Feature.id parser instead of ad-hoc fallback conversion"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'let id = rf\.id\.map\(Self::parse_feature_id\)\.transpose\(\)\?;' `
        -Message "GeoJSON FeatureCollection path must preserve Feature.id through the explicit parser"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'id: doc\.id\.clone\(\)\.map\(Self::parse_feature_id\)\.transpose\(\)\?,' `
        -Message "Top-level GeoJSON Feature path must preserve Feature.id through the explicit parser"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'test_feature_id_numeric_is_preserved' `
        -Message "GeoJSON importer must keep regression coverage for numeric Feature.id preservation"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'test_feature_id_rejects_invalid_type' `
        -Message "GeoJSON importer must keep regression coverage for invalid Feature.id rejection"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'test_top_level_feature_id_is_preserved' `
        -Message "GeoJSON importer must keep regression coverage for top-level Feature.id preservation"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'Feature geometry must not be null|test_feature_collection_rejects_null_geometry|test_top_level_feature_rejects_null_geometry' `
        -Message "GeoJSON importer must reject null Feature geometry explicitly"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/geojson.rs" `
        -Pattern 'must contain at least one linear ring|must contain at least 4 positions|must be closed' `
        -Message "GeoJSON importer must reject missing, underspecified, or open linear rings explicitly"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'skip_invalid:\s*true,' `
        -Message "CSV import defaults must stay strict; skip_invalid cannot default to true"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'unwrap_or_default\(\)' `
        -Message "CSV import parse errors must not collapse source labels into empty strings"

    Check-PatternAbsent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'parts\.len\(\)\.min\(n_cols \+ 1\)' `
        -Message "Multi-column CSV import must not silently truncate extra columns"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'pub fn with_skip_invalid\(mut self, skip_invalid: bool\) -> Self' `
        -Message "CSV import must require an explicit opt-in builder for skip_invalid"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'fn source_label\(path: Option<&Path>\) -> String' `
        -Message "CSV import must keep an explicit source-label helper for parse errors"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'test_default_rejects_invalid_lines' `
        -Message "CSV import must keep regression coverage for strict-by-default row parsing"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'test_parse_error_reports_string_source_label' `
        -Message "CSV import must keep regression coverage for explicit <string> source labels"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'test_multi_column_default_rejects_inconsistent_columns' `
        -Message "Multi-column CSV import must keep regression coverage for inconsistent-column rejection"

    Check-PatternPresent `
        -RelativePath "crates/mh_io/src/import/timeseries_csv.rs" `
        -Pattern 'test_multi_column_skip_invalid_requires_explicit_opt_in' `
        -Message "Multi-column CSV import must keep regression coverage for explicit skip_invalid opt-in"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] import contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] import contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
