param(
    [string]$Root = ".",
    [string]$Out = "docs/reports/architecture_audit_latest.md"
)

$ErrorActionPreference = "Stop"
$rootPath = Resolve-Path $Root

$rules = @(
    @{ Name = "legacy_marker"; Pattern = "legacy"; Severity = "P1" },
    @{ Name = "deprecated_compat"; Pattern = "deprecated\("; Severity = "P1" },
    @{ Name = "unimplemented_placeholder"; Pattern = "unimplemented!"; Severity = "P0" },
    @{ Name = "todo_placeholder"; Pattern = "TODO"; Severity = "P2" },
    @{ Name = "cpu_f64_specialization"; Pattern = "CpuBackend<f64>"; Severity = "P1" },
    @{ Name = "mh_agent_dyn_assimilable"; Pattern = "dyn Assimilable"; Severity = "P0" }
)

$files = Get-ChildItem "$rootPath/crates" -Recurse -File -Include *.rs
$rows = New-Object System.Collections.Generic.List[object]

foreach ($f in $files) {
    foreach ($r in $rules) {
        $matches = Select-String -Path $f.FullName -Pattern $r.Pattern
        if ($matches) {
            foreach ($m in $matches) {
                $rows.Add([pscustomobject]@{
                    Rule = $r.Name
                    Severity = $r.Severity
                    File = $m.Path.Replace($rootPath.Path + "\\", "")
                    Line = $m.LineNumber
                    Snippet = ($m.Line.Trim())
                })
            }
        }
    }
}

$summary = $rows | Group-Object Rule, Severity | Sort-Object Count -Descending
$total = $rows.Count
$now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("# Architecture Audit Baseline")
[void]$md.AppendLine("")
[void]$md.AppendLine("- generated_at: $now")
[void]$md.AppendLine("- root: $($rootPath.Path)")
[void]$md.AppendLine("- scanned_files: $($files.Count)")
[void]$md.AppendLine("- total_hits: $total")
[void]$md.AppendLine("")
[void]$md.AppendLine("## Rule Summary")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Rule | Severity | Hits |")
[void]$md.AppendLine("|---|---:|---:|")
foreach ($s in $summary) {
    $parts = $s.Name -split ", "
    [void]$md.AppendLine("| $($parts[0]) | $($parts[1]) | $($s.Count) |")
}
[void]$md.AppendLine("")

[void]$md.AppendLine("## Top Files")
[void]$md.AppendLine("")
[void]$md.AppendLine("| File | Hits |")
[void]$md.AppendLine("|---|---:|")
$topFiles = $rows | Group-Object File | Sort-Object Count -Descending | Select-Object -First 30
foreach ($tf in $topFiles) {
    [void]$md.AppendLine("| $($tf.Name) | $($tf.Count) |")
}
[void]$md.AppendLine("")

[void]$md.AppendLine("## Details (first 200)")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Rule | Sev | File:Line | Snippet |")
[void]$md.AppendLine("|---|---:|---|---|")
foreach ($r in $rows | Select-Object -First 200) {
    $snippet = $r.Snippet.Replace("|", "\\|")
    [void]$md.AppendLine("| $($r.Rule) | $($r.Severity) | $($r.File):$($r.Line) | $snippet |")
}

$dir = Split-Path -Parent $Out
if (!(Test-Path $dir)) { New-Item -ItemType Directory $dir -Force | Out-Null }
$md.ToString() | Set-Content -Path $Out -Encoding UTF8
Write-Output "[OK] wrote report: $Out"
