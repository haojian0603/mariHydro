#!/usr/bin/env pwsh
#
# Guard explicit snapshot statistics semantics.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Snapshot = Join-Path $ProjectRoot "crates/mh_io/src/snapshot.rs"
$ErrorFile = Join-Path $ProjectRoot "crates/mh_io/src/error.rs"
$LibFile = Join-Path $ProjectRoot "crates/mh_io/src/lib.rs"
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
    Write-Host "=== Checking IO snapshot contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    Assert-PatternPresent -Path $Snapshot -Pattern 'pub fn statistics\(&self\) -> IoResult<StateStatistics>' -Message "StateSnapshot::statistics must return IoResult<StateStatistics>"
    Assert-PatternAbsent -Path $Snapshot -Pattern 'return StateStatistics::default\(\);' -Message "StateSnapshot::statistics must not fabricate zero statistics for empty snapshots"
    Assert-PatternPresent -Path $Snapshot -Pattern 'IoError::MissingRequiredData \{' -Message "StateSnapshot::statistics must reject empty snapshots explicitly"
    Assert-PatternPresent -Path $Snapshot -Pattern 'test_state_statistics_rejects_empty_snapshot' -Message "StateSnapshot::statistics must keep empty-snapshot regression coverage"
    Assert-PatternAbsent -Path $Snapshot -Pattern '#\[derive\(Debug, Clone, Default\)\]\s*pub struct StateStatistics' -Message "StateStatistics must not derive Default for empty-snapshot fallback semantics"

    Assert-PatternPresent -Path $ErrorFile -Pattern 'MissingRequiredData \{ context: String \}' -Message "mh_io error layer must expose MissingRequiredData"
    Assert-PatternPresent -Path $ErrorFile -Pattern 'IoError::MissingRequiredData \{ context \} => \{' -Message "mh_io error layer must preserve MissingRequiredData conversion"

    Assert-PatternPresent -Path $LibFile -Pattern 'Empty snapshot statistics must fail explicitly\.' -Message "mh_io module docs must declare explicit empty-snapshot statistics semantics"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] IO snapshot contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] IO snapshot contracts failed" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
