# marihydro/scripts/check_global_state.ps1
#
# CI 守护脚本：检测全局可变状态 (Windows PowerShell 版本)
#
# 本脚本扫描代码中的全局可变状态（static mut, lazy_static, thread_local!）
# 以确保项目遵循无全局状态的设计原则。

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# 定义扫描的核心目录
$ScanDirs = @(
    "crates\mh_core\src",
    "crates\mh_physics\src",
    "crates\mh_mesh\src",
    "crates\mh_geo\src",
    "crates\mh_foundation\src",
    "crates\mh_workflow\src",
    "crates\mh_io\src",
    "crates\mh_terrain\src"
)

Write-Host "=== Checking for global mutable state ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$FoundIssues = 0

# 检测模式
$Patterns = @(
    "static\s+mut",           # static mut 变量
    "lazy_static!",           # lazy_static 宏
    "thread_local!",          # thread_local 宏
    "OnceCell::new\(\)",      # 全局 OnceCell
    "OnceLock::new\(\)",      # 全局 OnceLock
    "Lazy::new"               # once_cell::Lazy
)

$CombinedPattern = $Patterns -join "|"

foreach ($dir in $ScanDirs) {
    $FullDir = Join-Path $ProjectRoot $dir
    
    if (-not (Test-Path $FullDir)) {
        continue
    }
    
    $RsFiles = Get-ChildItem -Path $FullDir -Filter "*.rs" -Recurse -File
    
    foreach ($file in $RsFiles) {
        # 跳过测试文件
        if ($file.Name -like "*_test.rs" -or $file.FullName -like "*\tests\*") {
            continue
        }
        
        $Lines = Get-Content $file.FullName
        $LineNum = 0
        
        foreach ($line in $Lines) {
            $LineNum++
            
            if ($line -match $CombinedPattern) {
                # 检查是否有例外标注
                if ($line -match '@global-state-ok') { continue }
                
                # 跳过注释行
                $TrimmedLine = $line.Trim()
                if ($TrimmedLine.StartsWith("//") -or $TrimmedLine.StartsWith("/*") -or $TrimmedLine.StartsWith("*")) {
                    continue
                }
                
                $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                Write-Host "ISSUE: $RelPath`:$LineNum" -ForegroundColor Red
                Write-Host "  $line"
                Write-Host ""
                $FoundIssues++
            }
        }
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
if ($FoundIssues -eq 0) {
    Write-Host "✅ No global mutable state issues found!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Found $FoundIssues global mutable state issues" -ForegroundColor Red
    Write-Host "Consider using dependency injection or explicit parameter passing."
    Write-Host "If intentional, add // @global-state-ok comment."
    exit 1
}
