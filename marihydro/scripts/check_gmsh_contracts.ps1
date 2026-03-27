#!/usr/bin/env pwsh
#
# Blocking guard for GMSH parser truthfulness.
# Supported GMSH sections must reject malformed structural fields explicitly.
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

function Check-PatternPresentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required GMSH contract file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

function Check-PatternAbsentRaw {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required GMSH contract file: $RelativePath"
        return
    }

    $content = Get-Content $absolutePath -Raw -Encoding UTF8
    if ($content -match $Pattern) {
        Add-Failure $Message
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking GMSH parser contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $gmshPath = "crates/mh_mesh/src/io/gmsh.rs"
    $modPath = "crates/mh_mesh/src/io/mod.rs"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'GMSH_SOURCE:' `
        -Message "gmsh.rs must declare the real GMSH source contract"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'GMSH_SCOPE:' `
        -Message "gmsh.rs must declare explicit GMSH failure semantics"

    Check-PatternPresentRaw `
        -RelativePath $modPath `
        -Pattern 'GMSH_MAINLINE:' `
        -Message "mh_mesh::io module docs must preserve explicit GMSH semantics"

    Check-PatternAbsentRaw `
        -RelativePath $gmshPath `
        -Pattern 'parse::<usize>\(\)\.unwrap_or\(0\)|parts\[3\]\.parse\(\)\.unwrap_or\(0\)' `
        -Message "GMSH parser must not collapse malformed structural integers to zero"

    Check-PatternAbsentRaw `
        -RelativePath $gmshPath `
        -Pattern 'filter_map\(\|s\| s\.parse\(\)\.ok\(\)\)' `
        -Message "GMSH structural parsing must not silently drop malformed numeric tokens"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'fn parse_usize_token\(' `
        -Message "GMSH parser must use explicit usize token parsing helpers"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'fn parse_node_index\(' `
        -Message "GMSH parser must use explicit node-reference validation"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'node block parametric flag|must contain exactly 2 node tags' `
        -Message "GMSH parser must validate node-block flags and supported element arity explicitly"

    Check-PatternPresentRaw `
        -RelativePath $gmshPath `
        -Pattern 'test_load_v2_rejects_invalid_tag_count_token|test_load_v4_rejects_invalid_node_block_header|test_load_v4_rejects_unknown_node_reference|test_load_v4_rejects_extra_edge_node_tags' `
        -Message "GMSH parser must keep regression coverage for malformed structural fields"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] GMSH parser contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] GMSH parser contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
