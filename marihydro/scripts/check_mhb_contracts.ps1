#!/usr/bin/env pwsh
#
# Blocking guard for MHB binary mesh format contracts.
# Counts, offsets and scalar payloads must fail explicitly instead of collapsing
# to synthetic defaults.
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
        Add-Failure "missing required MHB contract file: $RelativePath"
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
        Add-Failure "missing required MHB contract file: $RelativePath"
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
    Write-Host "=== Checking MHB binary mesh contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $mhbPath = "crates/mh_mesh/src/io/mhb.rs"
    $modPath = "crates/mh_mesh/src/io/mod.rs"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'MHB_SOURCE:' `
        -Message "mhb.rs must declare the real MHB source contract"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'MHB_SCOPE:' `
        -Message "mhb.rs must expose an MHB scope tag"

    Check-PatternPresentRaw `
        -RelativePath $modPath `
        -Pattern 'MHB_SCOPE:' `
        -Message "mh_mesh::io module docs must preserve an explicit MHB scope tag"

    Check-PatternAbsentRaw `
        -RelativePath $mhbPath `
        -Pattern 'to_f32\(\)\.unwrap_or\(0\.0\)|to_f64\(\)\.unwrap_or\(0\.0\)' `
        -Message "MHB scalar export must not silently collapse failed conversions to zero"

    Check-PatternAbsentRaw `
        -RelativePath $mhbPath `
        -Pattern 'values\.first\(\)\.copied\(\)\.unwrap_or\(0\)' `
        -Message "MHB count fields must not treat empty payloads as zero"

    Check-PatternAbsentRaw `
        -RelativePath $mhbPath `
        -Pattern '\.map\(\|v\| v as usize\)' `
        -Message "MHB offsets must not use unchecked usize casts"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'count field must contain exactly one value' `
        -Message "MHB count reader must reject empty or multi-value payloads explicitly"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'offset at index \{index\} with value \{value\} does not fit usize' `
        -Message "MHB offsets must preserve an explicit overflow error message"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'scalar value at index \{index\} is not finite' `
        -Message "MHB scalar writer must reject non-finite values explicitly"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'test_read_count_rejects_empty_payload' `
        -Message "MHB regressions must cover empty-count rejection"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'test_read_count_rejects_multi_value_payload' `
        -Message "MHB regressions must cover multi-count rejection"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'test_write_scalar_field_rejects_non_finite_f64' `
        -Message "MHB regressions must cover non-finite scalar export rejection"

    Check-PatternPresentRaw `
        -RelativePath $mhbPath `
        -Pattern 'test_read_offsets_accept_u64_payload' `
        -Message "MHB regressions must cover offset decoding"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] MHB binary mesh contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] MHB binary mesh contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
