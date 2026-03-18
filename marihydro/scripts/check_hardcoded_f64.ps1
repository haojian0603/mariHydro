param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "=== Layer 3 hardcoded f64 guard ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$ScanDirs = @(
    "crates\mh_physics\src\engine",
    "crates\mh_physics\src\flux",
    "crates\mh_physics\src\boundary",
    "crates\mh_physics\src\numerics\linear_algebra",
    "crates\mh_physics\src\numerics\gradient",
    "crates\mh_physics\src\numerics\reconstruction",
    "crates\mh_physics\src\numerics\limiter",
    "crates\mh_physics\src\numerics\operators",
    "crates\mh_physics\src\sources",
    "crates\mh_physics\src\time_integrator",
    "crates\mh_physics\src\timestep",
    "crates\mh_physics\src\riemann",
    "crates\mh_physics\src\wetting_drying"
)

$ExcludeDirNames = @(
    "\tests\",
    "\benches\"
)

$ExcludeFileNames = @(
    "*_test.rs",
    "test_*.rs",
    "*_tests.rs",
    "scalar.rs",
    "precision.rs",
    "constants.rs",
    "physical_constants.rs",
    "numerical_params.rs",
    "properties.rs",
    "morphology.rs",
    "atmosphere.rs",
    "field.rs",
    "config.rs"
)

$Whitelist = @(
    "std::f64::consts::",
    "PI",
    "GRAVITY",
    "EARTH_ANGULAR_VELOCITY"
)

$issues = @()

function Test-ExcludedFile {
    param(
        [string]$FullName,
        [string]$Name
    )

    foreach ($pattern in $ExcludeFileNames) {
        if ($Name -like $pattern -or $FullName -like "*$pattern") {
            return $true
        }
    }

    foreach ($dirPattern in $ExcludeDirNames) {
        if ($FullName -like "*$dirPattern*") {
            return $true
        }
    }

    return $false
}

foreach ($dir in $ScanDirs) {
    $fullDir = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullDir)) {
        if ($Verbose) {
            Write-Host "[SKIP] missing: $fullDir" -ForegroundColor Yellow
        }
        continue
    }

    $files = Get-ChildItem -Path $fullDir -Recurse -Filter "*.rs" -File
    foreach ($file in $files) {
        if (Test-ExcludedFile -FullName $file.FullName -Name $file.Name) {
            continue
        }

        $lineNo = 0
        $lines = Get-Content -Path $file.FullName
        foreach ($line in $lines) {
            $lineNo++
            $trim = $line.Trim()
            if ($trim.StartsWith("//") -or $trim.StartsWith("/*") -or $trim.StartsWith("*")) {
                continue
            }

            if ($trim -match "\bf64\b") {
                $allowed = $false
                foreach ($ok in $Whitelist) {
                    if ($trim -like "*$ok*") {
                        $allowed = $true
                        break
                    }
                }

                if (-not $allowed) {
                    $issues += [pscustomobject]@{
                        Path = $file.FullName.Replace($ProjectRoot + "\", "")
                        Line = $lineNo
                        Text = $trim
                    }
                }
            }
        }
    }
}

if ($issues.Count -eq 0) {
    Write-Host "[OK] no hardcoded f64 found in Layer 3 scan" -ForegroundColor Green
    exit 0
}

Write-Host "[FAIL] hardcoded f64 findings:" -ForegroundColor Red
foreach ($issue in $issues) {
    Write-Host ("  " + $issue.Path + ":" + $issue.Line + ": " + $issue.Text) -ForegroundColor Red
}
Write-Host "Hint: use backend scalar conversion or add an explicit allowlist comment." -ForegroundColor Yellow
exit 1
