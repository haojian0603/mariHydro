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
    foreach ($Root in $Roots) {
        $AbsoluteRoot = Join-Path $ProjectRoot $Root
        if (-not (Test-Path $AbsoluteRoot)) {
            continue
        }

        $Files = Get-ChildItem -Path $AbsoluteRoot -Recurse -Filter "*.rs" -File
        foreach ($File in $Files) {
            $Matches = Select-String -Path $File.FullName -Pattern $Pattern -CaseSensitive:$false
            foreach ($Match in $Matches) {
                $Findings += [pscustomobject]@{
                    Path = $File.FullName.Replace($ProjectRoot + "\", "")
                    Line = $Match.LineNumber
                    Text = $Match.Line.Trim()
                }
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
    foreach ($Root in $Roots) {
        $AbsoluteRoot = Join-Path $ProjectRoot $Root
        if (-not (Test-Path $AbsoluteRoot)) {
            continue
        }

        $Files = Get-ChildItem -Path $AbsoluteRoot -Recurse -Filter "*.rs" -File
        foreach ($File in $Files) {
            $Matches = Select-String -Path $File.FullName -Pattern $Pattern -CaseSensitive
            foreach ($Match in $Matches) {
                $Findings += [pscustomobject]@{
                    Path = $File.FullName.Replace($ProjectRoot + "\", "")
                    Line = $Match.LineNumber
                    Text = $Match.Line.Trim()
                }
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

    Write-Host ""
    Write-Host "=== Advisory scans ===" -ForegroundColor Cyan
    Invoke-InformationalScan -Name "legacy SourceTrait residue" -Roots @("crates/mh_physics") -Pattern "SourceTrait"
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
