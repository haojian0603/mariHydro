#!/usr/bin/env pwsh
#
# Ensures high-risk physical-model files carry explicit provenance and scope tags.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Errors = @()

$RequiredFiles = @(
    @{ Path = "crates/mh_physics/src/waves/spectral.rs"; Description = "wave spectral model" },
    @{ Path = "crates/mh_physics/src/sediment/formulas.rs"; Description = "sediment transport formulas" },
    @{ Path = "crates/mh_physics/src/sediment/suspended/settling.rs"; Description = "settling velocity formulas" },
    @{ Path = "crates/mh_physics/src/tracer/diffusion.rs"; Description = "anisotropic tracer diffusion" },
    @{ Path = "crates/mh_physics/src/sources/turbulence/smagorinsky.rs"; Description = "Smagorinsky turbulence closure" }
)

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking physical formula provenance ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    foreach ($entry in $RequiredFiles) {
        $absolutePath = Join-Path $ProjectRoot $entry.Path
        if (-not (Test-Path $absolutePath)) {
            Write-Host "[FAIL] missing required physics file: $($entry.Path)" -ForegroundColor Red
            $Errors += "missing required physics file: $($entry.Path)"
            continue
        }

        $content = Get-Content $absolutePath -Raw -Encoding UTF8

        if ($content -match 'PHYSICS_SOURCE:\s*.+(19|20)\d{2}.+') {
            Write-Host "[OK] $($entry.Description) source tag" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $($entry.Description) is missing a verifiable PHYSICS_SOURCE tag" -ForegroundColor Red
            $Errors += "$($entry.Path) must include PHYSICS_SOURCE with verifiable citation data"
        }

        if ($content -match 'PHYSICS_SCOPE:\s*\S.{20,}') {
            Write-Host "[OK] $($entry.Description) scope tag" -ForegroundColor Green
        } else {
            Write-Host "[FAIL] $($entry.Description) is missing a concrete PHYSICS_SCOPE tag" -ForegroundColor Red
            $Errors += "$($entry.Path) must include PHYSICS_SCOPE describing scope and exclusions"
        }
    }

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] physical formula provenance passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] physical formula provenance failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
