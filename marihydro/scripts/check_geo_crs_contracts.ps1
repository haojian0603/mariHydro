#!/usr/bin/env pwsh
#
# Guard explicit CRS ellipsoid detection semantics.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Target = Join-Path $ProjectRoot "crates/mh_geo/src/crs.rs"
$Errors = @()

function Assert-PatternAbsent {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )

    if (Select-String -Path $Path -Pattern $Pattern -Quiet) {
        Write-Host "[FAIL] $Message" -ForegroundColor Red
        $script:Errors += $Message
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

function Assert-PatternPresent {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Message
    )

    if (Select-String -Path $Path -Pattern $Pattern -Quiet) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $Message" -ForegroundColor Red
        $script:Errors += $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking geo CRS contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Assert-PatternPresent -Path $Target -Pattern 'fn detect_ellipsoid\(def: &str, epsg: Option<u32>\) -> MhResult<Ellipsoid>' -Message "CRS ellipsoid detection must return MhResult<Ellipsoid>"
    Assert-PatternPresent -Path $Target -Pattern 'fn normalized_ellipsoid_label\(' -Message "CRS ellipsoid detection must normalize explicit ellipsoid identifiers"
    Assert-PatternPresent -Path $Target -Pattern 'fn detect_proj_ellipsoid\(' -Message "CRS ellipsoid detection must inspect explicit PROJ parameter values"
    Assert-PatternPresent -Path $Target -Pattern 'fn detect_wkt_ellipsoid\(' -Message "CRS ellipsoid detection must inspect explicit WKT quoted identifiers"
    Assert-PatternAbsent -Path $Target -Pattern 'contains\("wgs84"\)|contains\("wgs 84"\)|contains\("cgcs2000"\)|contains\("grs80"\)|contains\("grs 80"\)|contains\("krassovsky"\)|contains\("krasovsky"\)' -Message "CRS ellipsoid detection must not use raw substring heuristics"
    Assert-PatternAbsent -Path $Target -Pattern '默认 WGS84|Ellipsoid::WGS84\s*$' -Message "CRS ellipsoid detection must not fall back to synthetic default WGS84"
    Assert-PatternPresent -Path $Target -Pattern 'test_ellipsoid_detection_rejects_custom_identifier_suffix' -Message "CRS ellipsoid detection must keep regression coverage for custom identifier rejection"
    Assert-PatternPresent -Path $Target -Pattern 'assert_eq!\(\*grs80\.ellipsoid\(\), Ellipsoid::GRS80\);' -Message "CRS ellipsoid detection must keep exact GRS80 regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] geo CRS contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] geo CRS contracts failed" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
