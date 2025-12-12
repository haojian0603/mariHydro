# marihydro/scripts/check_hardcoded_f64.ps1
#
# CI 守护脚本：检测硬编码 f64 类型 (Windows PowerShell 版本)
#
# 本脚本扫描核心计算代码中的硬编码 f64，确保所有数值类型都通过 Scalar 泛型
# 或 Precision 枚举进行管理。

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# 定义扫描的核心目录
$ScanDirs = @(
    "crates\mh_physics\src\engine",
    "crates\mh_physics\src\schemes",
    "crates\mh_physics\src\sources",
    "crates\mh_physics\src\state",
    "crates\mh_mesh\src",
    "crates\mh_geo\src"
)

# 定义排除的文件模式
$ExcludePatterns = @(
    "scalar.rs",
    "precision.rs",
    "builder",
    "_test.rs",
    "_tests.rs",
    "test_",
    "mod.rs"
)

Write-Host "=== Checking for hardcoded f64 types ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$FoundIssues = 0

foreach ($dir in $ScanDirs) {
    $FullDir = Join-Path $ProjectRoot $dir
    
    if (-not (Test-Path $FullDir)) {
        if ($Verbose) {
            Write-Host "Warning: Directory not found: $FullDir" -ForegroundColor Yellow
        }
        continue
    }
    
    $RsFiles = Get-ChildItem -Path $FullDir -Filter "*.rs" -Recurse -File
    
    foreach ($file in $RsFiles) {
        # 检查是否在排除列表中
        $Skip = $false
        foreach ($pattern in $ExcludePatterns) {
            if ($file.FullName -like "*$pattern*") {
                $Skip = $true
                break
            }
        }
        
        if ($Skip) { continue }
        
        # 读取文件内容并搜索硬编码 f64 模式
        $Content = Get-Content $file.FullName -Raw
        $Lines = Get-Content $file.FullName
        
        $LineNum = 0
        foreach ($line in $Lines) {
            $LineNum++
            
            # 匹配: `: f64`, `f64,`, `f64>`, `f64)`, `as f64`, `[f64;`, `Vec<f64>`
            if ($line -match '(:\s*f64\b|f64[,)>\]]|as\s+f64\b|\[f64;|Vec<f64>)') {
                # 跳过 Scalar trait bound
                if ($line -match 'Scalar|Float') { continue }
                
                # 跳过纯注释行
                $TrimmedLine = $line.Trim()
                if ($TrimmedLine.StartsWith("//") -or $TrimmedLine.StartsWith("/*") -or $TrimmedLine.StartsWith("*")) {
                    continue
                }
                
                # 检查 f64 是否在注释中
                $CommentPos = $line.IndexOf("//")
                $F64Pos = $line.IndexOf("f64")
                if ($CommentPos -ge 0 -and $F64Pos -gt $CommentPos) {
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
    Write-Host "✅ No hardcoded f64 issues found!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Found $FoundIssues potential hardcoded f64 issues" -ForegroundColor Red
    Write-Host "Please use Scalar<S> generic type or ensure these are intentional."
    exit 1
}
