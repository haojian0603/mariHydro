# CI 守护脚本：检测硬编码 f64 类型 (Windows PowerShell 版本)

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# === 排除规则（添加在这里）===
# 排除的目录（底层库，坐标和几何需要f64精度）
$ExcludeDirs = @(
    "crates\mh_mesh",              # 几何库（坐标存储）- Layer 1允许f64
    "crates\mh_geo"                # 地理库（坐标转换）- Layer 1允许f64
)

# 排除的文件模式（物理常数、材料属性、配置参数）
# 注意：PowerShell -like 使用通配符，* 匹配任意字符，? 匹配单个字符
$ExcludeFilePatterns = @(
    "scalar.rs",                   # Scalar trait定义 - 基础类型定义
    "precision.rs",                # Precision枚举 - 配置层
    "constants.rs",                # 物理常数 - 明确允许
    "physical_constants.rs",       # 物理常数 - 明确允许
    "numerical_params.rs",         # 数值参数配置 - Layer 4配置层
    "properties.rs",               # 沉积物材料属性 - Layer 4配置层
    "morphology.rs",               # 地形几何数据 - Layer 1几何层
    "atmosphere.rs",               # 大气物理常数 - Layer 1
    "field.rs",                    # 地基参数 - Layer 4配置层
    "reconstruction\config.rs",    # 重构配置 - Layer 4配置层
    "limiter\config.rs",           # 限制器配置 - Layer 4配置层
    "diffusion.rs",                # 扩散算子配置 - Layer 4配置层
    "*_test.rs",                   # 测试文件
    "test_*.rs",                   # 测试文件
    "_test.rs",                    # 测试文件
    "_tests.rs",                   # 测试文件
    "tests\"                       # 测试模块
)

# 需要严格扫描的核心目录（Layer 3 引擎层）
$ScanDirs = @(
    "crates\mh_physics\src\engine"
    "crates\mh_physics\src\flux"
    "crates\mh_physics\src\boundary"
    "crates\mh_physics\src\numerics\linear_algebra"
    "crates\mh_physics\src\numerics\gradient"
    "crates\mh_physics\src\numerics\reconstruction"
    "crates\mh_physics\src\numerics\limiter"
    "crates\mh_physics\src\sources"
)

Write-Host "=== Checking for hardcoded f64 types ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

$FoundIssues = 0
$IssueDetails = @()  # 用于收集详细信息以便导出

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
        # 检查是否在排除目录中
        $InExcludeDir = $false
        foreach ($excludeDir in $ExcludeDirs) {
            if ($file.FullName -like "*$excludeDir*") {
                $InExcludeDir = $true
                break
            }
        }
        if ($InExcludeDir) { continue }
        
        # 检查文件名是否在排除列表中
        $Skip = $false
        foreach ($pattern in $ExcludeFilePatterns) {
            if ($file.Name -like $pattern -or $file.FullName -like "*$pattern") {
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
            
            # 匹配: `: f64`, `as f64`, `[f64;`, `Vec<f64>` 等
            if ($line -match '(:\s*f64\b|as\s+f64\b|\[f64;|Vec<f64>)') {
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
                
                # === 新增：检查是否包含 ALLOW_F64 注释 ===
                if ($line -match '//\s*ALLOW_F64:') {
                    continue
                }
                # 也检查上一行是否有 ALLOW_F64
                if ($LineNum -gt 1) {
                    $PrevLine = $Lines[$LineNum-2]
                    if ($PrevLine -match '//\s*ALLOW_F64:') {
                        continue
                    }
                }
                
                $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                $IssueLine = "ISSUE: $RelPath`:$LineNum"
                Write-Host $IssueLine -ForegroundColor Red
                Write-Host "  $line"
                Write-Host ""
                $FoundIssues++
                
                # 收集详细信息用于导出
                $IssueDetails += @{
                    File = $RelPath
                    Line = $LineNum
                    Code = $line.Trim()
                }
            }
        }
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan

# === 导出结果到文件 ===
$OutputFile = Join-Path $ProjectRoot "f64_check_results.txt"
$OutputContent = @"
=== MariHydro Hardcoded f64 Check Results ===
Date: $(Get-Date)
Project: $ProjectRoot
Found Issues: $FoundIssues
"@
if ($FoundIssues -eq 0) {
    $OutputContent += "`n✅ No hardcoded f64 issues found in Layer 3 Engine!"
} else {
    $OutputContent += "`n`n=== Detailed Issues ===`n"
    foreach ($issue in $IssueDetails) {
        $OutputContent += "File: $($issue.File):$($issue.Line)`n"
        $OutputContent += "Code: $($issue.Code)`n`n"
    }
    $OutputContent += "Please use Scalar<S> generic type or add // ALLOW_F64: <原因> comment`n"
}
$OutputContent | Out-File -FilePath $OutputFile -Encoding UTF8
Write-Host "📄 Results exported to: $OutputFile" -ForegroundColor Cyan

if ($FoundIssues -eq 0) {
    Write-Host "✅ No hardcoded f64 issues found in Layer 3 Engine!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Found $FoundIssues hardcoded f64 issues" -ForegroundColor Red
    Write-Host "Please use Scalar<S> generic type or add // ALLOW_F64: <原因> comment"
    exit 1
}