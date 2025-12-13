# crates/marihydro/scripts/check_hardcoded_f64.ps1
param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# === 排除规则（基于架构五层设计原则） ===
# 排除的目录（Layer 1: 基础几何库，坐标存储天然需要f64精度）
$ExcludeDirs = @(
    "crates\mh_mesh",              # 几何网格库 - 坐标几何计算允许f64
    "crates\mh_geo"                # 地理坐标库 - 大地坐标转换需要f64
)

# 排除的文件模式（Layer 4/5: 配置参数、物理常数、材料属性）
# 支持 // ALLOW_F64: <原因> 注释的排除机制
$ExcludeFilePatterns = @(
    "scalar.rs",                   # RuntimeScalar trait定义 - 基础抽象层
    "precision.rs",                # Precision枚举 - 运行时精度选择
    "constants.rs",                # 物理常数文件 - 全局常数明确允许f64
    "physical_constants.rs",       # 物理常数文件 - 全局常数明确允许f64
    "numerical_params.rs",         # 数值参数配置 - Layer 4配置层允许f64
    "properties.rs",               # 材料属性配置 - Layer 4配置层允许f64
    "morphology.rs",               # 地形几何数据 - Layer 1几何层允许f64
    "atmosphere.rs",               # 大气物理常数 - Layer 1物理常数层
    "field.rs",                    # 地基参数配置 - Layer 4配置层
    "reconstruction\config.rs",    # 重构配置 - Layer 4配置层
    "limiter\config.rs",           # 限制器配置 - Layer 4配置层
    "diffusion.rs",                # 扩散算子配置 - Layer 4配置层
    "*_test.rs",                   # 单元测试文件 - 测试逻辑不受限
    "test_*.rs",                   # 集成测试文件 - 测试逻辑不受限
    "_test.rs",                    # 测试模块文件 - 测试逻辑不受限
    "_tests.rs",                   # 测试模块文件 - 测试逻辑不受限
    "tests\"                       # 测试目录 - 测试逻辑不受限
)

# 需要严格扫描的核心目录（Layer 3: 引擎计算核心层，禁止硬编码f64）
# 包括：求解器、通量计算、边界处理、数值算子、时间积分等
$ScanDirs = @(
    "crates\mh_physics\src\engine",           # 求解器核心 - 必须泛型化
    "crates\mh_physics\src\flux",             # 通量计算 - 必须泛型化
    "crates\mh_physics\src\boundary",         # 边界处理 - 必须泛型化
    "crates\mh_physics\src\numerics\linear_algebra", # 线性代数 - 必须泛型化
    "crates\mh_physics\src\numerics\gradient",       # 梯度计算 - 必须泛型化
    "crates\mh_physics\src\numerics\reconstruction", # 重构 - 必须泛型化
    "crates\mh_physics\src\numerics\limiter",        # 限制器 - 必须泛型化
    "crates\mh_physics\src\numerics\operators",      # 算子 - 必须泛型化
    "crates\mh_physics\src\sources",      # 源项 - 必须泛型化
    "crates\mh_physics\src\time_integrator", # 时间积分 - 必须泛型化
    "crates\mh_physics\src\timestep",       # 时间步控制 - 必须泛型化
    "crates\mh_physics\src\riemann",        # 黎曼求解器 - 必须泛型化
    "crates\mh_physics\src\wetting_drying"  # 干湿处理 - 必须泛型化
)

Write-Host "=== MariHydro Layer 3 Engine F64 Guardian ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host ""

# === 新增：标准库常数白名单 ===
$WhitelistPatterns = @(
    'std::f64::consts::',
    'EARTH_ANGULAR_VELOCITY',
    'GRAVITY',
    'PI'
)

# 用于缓存 ALLOW_F64 注释的影响范围
$AllowF64Scopes = @{}

function Get-TrimmedLine($line) {
    # 去除注释后的代码部分
    $commentPos = $line.IndexOf("//")
    if ($commentPos -ge 0) {
        return $line.Substring(0, $commentPos).Trim()
    }
    return $line.Trim()
}

function Update-BraceDepth($line, [ref]$depth) {
    # 跳过字符串中的大括号
    $processedLine = $line -replace '"[^"]*"', '' -replace "'[^']*'", ''
    
    $openBraces = ([regex]::Matches($processedLine, '\{')).Count
    $closeBraces = ([regex]::Matches($processedLine, '\}')).Count
    $depth.Value += $openBraces - $closeBraces
    return $depth.Value
}

function Is-WhiteListLine($line) {
    foreach ($pattern in $WhitelistPatterns) {
        if ($line -match $pattern) {
            return $true
        }
    }
    return $false
}

$FoundIssues = 0
$IssueDetails = @()

foreach ($dir in $ScanDirs) {
    $FullDir = Join-Path $ProjectRoot $dir
    
    if (-not (Test-Path $FullDir)) {
        if ($Verbose) { Write-Host "跳过: $FullDir 不存在" -ForegroundColor Yellow }
        continue
    }
    
    $RsFiles = Get-ChildItem -Path $FullDir -Filter "*.rs" -Recurse -File
    
    foreach ($file in $RsFiles) {
        # 排除目录检查
        if ($ExcludeDirs | Where-Object { $file.FullName -like "*$_*" }) { continue }
        
        # 排除文件模式检查
        if ($ExcludeFilePatterns | Where-Object { $file.Name -like $_ -or $file.FullName -like "*$_" }) { continue }
        
        $Lines = Get-Content $file.FullName
        
        # === 状态追踪 ===
        $InTestCfg = $false          # 是否在 #[cfg(test)] 影响范围内
        $TestBraceDepth = 0
        $TestScopeStartLine = 0
        
        $InAllowF64Block = $false    # 是否在 // ALLOW_F64_BEGIN 块注释影响范围内
        $InAllowF64LineScope = $false # 是否在 // ALLOW_F64: 行注释影响范围内
        $AllowF64StartLine = 0
        
        $InStructBlock = $false      # 是否在 struct { ... } 块内
        $StructBraceDepth = 0
        
        $InTraitOrImpl = $false      # 是否在 trait/impl 块内
        $BlockBraceDepth = 0
        
        $InWhereClause = $false      # 是否在 where 子句内
        
        $LineNum = 0
        
        foreach ($line in $Lines) {
            $LineNum++
            $TrimmedLine = Get-TrimmedLine $line
            
            # === 1. 追踪 #[cfg(test)] 范围 ===
            # 检测 #[cfg(test)] 属性
            if ($TrimmedLine -match '#\[cfg\(test\)\]') {
                $InTestCfg = $true
                $TestBraceDepth = 0
                $TestScopeStartLine = $LineNum
                continue
            }
            
            if ($InTestCfg) {
                # 更新测试模块的大括号深度
                $prevDepth = $TestBraceDepth
                $TestBraceDepth = Update-BraceDepth $line ([ref]$TestBraceDepth)
                
                # 如果离开测试模块（深度回到0且之前进入过）
                if ($prevDepth -gt 0 -and $TestBraceDepth -le 0) {
                    $InTestCfg = $false
                }
                
                # 跳过测试模块内的所有行
                continue
            }
            
            # === 2. 追踪 // ALLOW_F64: 行注释范围 ===
            # 检测 ALLOW_F64 行注释
            if ($TrimmedLine -match '//\s*ALLOW_F64:') {
                $InAllowF64LineScope = $true
                $AllowF64StartLine = $LineNum
                # 记录影响范围：注释应用于下一个顶层项
                continue
            }
            
            # === 3. 追踪 // ALLOW_F64_BEGIN/END: 块注释范围 ===
            # 检测 ALLOW_F64_BEGIN 块注释开始
            if ($TrimmedLine -match '//\s*ALLOW_F64_BEGIN:') {
                $InAllowF64Block = $true
                continue
            }
            
            # 检测 ALLOW_F64_END 块注释结束
            if ($TrimmedLine -match '//\s*ALLOW_F64_END') {
                $InAllowF64Block = $false
                continue
            }
            
            # 如果在块注释范围内，跳过检测
            if ($InAllowF64Block) { continue }
            
            # === 4. 检测 where 子句并跳过 ===
            if ($TrimmedLine -match '\bwhere\b') {
                $InWhereClause = $true
            }
            
            if ($InWhereClause) {
                # 追踪 where 子句的范围（通常在同一行或下一行的 { 开始）
                if ($TrimmedLine -match '\{$') {
                    $InWhereClause = $false
                }
                # 跳过 where 子句内的检查
                continue
            }
            
            # === 5. 追踪 struct 块范围 ===
            # 检测 struct 开始
            if ($TrimmedLine -match '^(pub\s+)?struct\b.*\{$') {
                $InStructBlock = $true
                $StructBraceDepth = 0
            }
            
            if ($InStructBlock) {
                $StructBraceDepth = Update-BraceDepth $line ([ref]$StructBraceDepth)
                
                # 在 struct 块内，每行都应检查（用于检测字段类型）
                if ($StructBraceDepth -gt 0) {
                    # 如果离开 struct 块
                    if ($StructBraceDepth -le 0) {
                        $InStructBlock = $false
                    }
                    
                    # 在 struct 块内，即使不在下一行，也要检查
                    # 这将允许更灵活的字段定义
                }
            }
            
            # === 6. 追踪 trait/impl 块深度（用于顶层项识别） ===
            if ($TrimmedLine -match '^(pub\s+)?(impl|trait)\b.*\{') {
                $InTraitOrImpl = $true
                $BlockBraceDepth = 0
            }
            if ($TrimmedLine -match '^(pub\s+)?(struct|enum)\b') {
                # 标记为顶层项，结束行注释作用域
                if ($InAllowF64LineScope -and $LineNum -gt $AllowF64StartLine) {
                    $InAllowF64LineScope = $false
                }
            }
            
            if ($InTraitOrImpl) {
                $BlockBraceDepth = Update-BraceDepth $line ([ref]$BlockBraceDepth)
                if ($BlockBraceDepth -le 0) {
                    $InTraitOrImpl = $false
                    # 结束 trait/impl 块时，也结束行注释作用域
                    if ($InAllowF64LineScope) {
                        $InAllowF64LineScope = $false
                    }
                }
            }
            
            # === 7. 检测硬编码 f64 ===
            # 如果在行注释或块注释范围内，跳过检测
            if ($InAllowF64LineScope -or $InAllowF64Block) { continue }
            
            # 如果是白名单行，跳过检测
            if (Is-WhiteListLine $TrimmedLine) { continue }
            
            # 匹配各种 f64 硬编码模式
            $Patterns = @(
                ':\s*f64\b',           # 类型注解
                'as\s+f64\b',          # 类型转换
                '\[f64\b',             # 数组类型
                'Vec<f64>'             # 泛型容器
            )
            
            foreach ($pattern in $Patterns) {
                if ($TrimmedLine -match $pattern) {
                    # 排除 Scalar trait bound
                    if ($TrimmedLine -match 'Scalar|Float') { break }
                    
                    # 排除空行和注释行
                    if ([string]::IsNullOrWhiteSpace($TrimmedLine)) { break }
                    if ($TrimmedLine -match '^(pub\s+)?(use|mod)\b') { break }
                    
                    $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                    $IssueLine = "ISSUE: $RelPath`:$LineNum"
                    Write-Host $IssueLine -ForegroundColor Red
                    Write-Host "  $line" -ForegroundColor Gray
                    $FoundIssues++
                    
                    $IssueDetails += @{File=$RelPath; Line=$LineNum; Code=$line.Trim()}
                    break
                }
            }
        }
    }
}

# === 结果输出 ===
Write-Host ""
Write-Host "=== Guardian Check Completed ===" -ForegroundColor Cyan

$OutputFile = Join-Path $ProjectRoot "f64_check_results.txt"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$Report = @"
================================================================
MariHydro Layer 3 Engine F64 Guardian Report
================================================================
Timestamp: $Timestamp
Total Issues: $FoundIssues
================================================================
"@

if ($FoundIssues -eq 0) {
    $Report += "`n[OK] SUCCESS: No hardcoded f64 found in Layer 3 Engine!`n"
    $Report += "================================================================"
    $Report | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host $Report -ForegroundColor Green
    exit 0
} else {
    $Report += "`n[FAIL] Found $FoundIssues hardcoded f64 violations in Layer 3`n`n"
    $Report += "在以下位置发现 f64 硬编码类型:`n"
    
    foreach ($issue in $IssueDetails) {
        $Report += "  -> $($issue.File):$($issue.Line)`n"
    }
    
    $Report += "`n修复要求:`n"
    $Report += "  1. Vec<f64> → B::Buffer<S> (Backend 泛型缓冲区)`n"
    $Report += "  2. Type: f64 → Type: S (RuntimeScalar)`n"
    $Report += "  3. as f64 → Scalar::from_f64() 或直接修改`n"
    $Report += "  4. 在 Layer 3 中移除 // ALLOW_F64: 注释（配置层除外）`n"
    $Report += "  5. 使用 // ALLOW_F64_BEGIN: 和 // ALLOW_F64_END: 进行块排除`n"
    $Report += "  6. where 子句中的 f64 属于 trait bound，已自动排除`n"
    $Report += "`n================================================================"
    
    $Report | Out-File -FilePath $OutputFile -Encoding UTF8
    Write-Host $Report -ForegroundColor Red
    
    Write-Host "详细报告: $OutputFile" -ForegroundColor Cyan
    Write-Host "用法: ./scripts/check_hardcoded_f64.ps1 -Verbose" -ForegroundColor Yellow
    
    exit 1
}