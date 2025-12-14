#!/usr/bin/env pwsh

<#
.SYNOPSIS
交互式提取指定 crate 的源代码到文本文件

.DESCRIPTION
扫描指定 crate 目录，收集所有 .rs 文件内容，输出到单个 txt 文件
便于 AI 检查代码结构

.PARAMETER ProjectRoot
项目根目录路径（默认为脚本所在目录的父目录，即 workspace 根目录）

.EXAMPLE
.\extract_crates.ps1
.\extract_crates.ps1 -ProjectRoot "E:\Documents\mariHydro\marihydro"
#>

param(
    [Parameter()]
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
)

# ANSI 颜色
$colorCyan = "`e[36m"
$colorYellow = "`e[33m"
$colorGreen = "`e[32m"
$colorRed = "`e[31m"
$colorReset = "`e[0m"

Write-Host "${colorCyan}=== MariHydro 代码提取工具 ===${colorReset}"
Write-Host "项目路径: $ProjectRoot"
Write-Host "Crates 路径: $(Join-Path $ProjectRoot 'crates')`n"

# 验证路径
$cratesDir = Join-Path $ProjectRoot "crates"
if (!(Test-Path $cratesDir)) {
    Write-Host "${colorRed}错误: Crates 目录不存在 - $cratesDir${colorReset}"
    Write-Host "${colorYellow}提示: 请确保在 workspace 根目录或 crates 目录下运行此脚本${colorReset}"
    exit 1
}

# 所有候选 crate
$crates = @(
    "mh_runtime",
    "mh_config",
    "mh_physics",
    "mh_mesh",
    "mh_geo",
    "mh_io",
    "mh_agent",
    "mh_foundation",
    "mh_terrain",
    "mh_workflow",
    "mh_editor"
)

Write-Host "${colorYellow}请选择要检查的 crate (可多选，用空格分隔):${colorReset}"
for ($i = 0; $i -lt $crates.Count; $i++) {
    Write-Host "  [$i] $($crates[$i])"
}
Write-Host "  [A] 选择全部"
Write-Host "  [Q] 退出`n"

$selection = Read-Host "输入选择"

if ($selection -eq 'Q' -or $selection -eq 'q') {
    Write-Host "`n${colorGreen}退出。${colorReset}"
    exit 0
}

# 解析选择
if ($selection -eq 'A' -or $selection -eq 'a') {
    $selected = $crates
} else {
    $indices = $selection -split ' ' | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ }
    $selected = $indices | Where-Object { $_ -ge 0 -and $_ -lt $crates.Count } | ForEach-Object { $crates[$_] }
    
    if ($selected.Count -eq 0) {
        Write-Host "${colorRed}错误: 未选择有效的 crate${colorReset}"
        exit 1
    }
}

Write-Host "`n${colorGreen}已选择: $($selected -join ', ')${colorReset}`n"

# 排除目录和文件类型
$excludeDirs = @(
    ".git", "target", "node_modules", "venv", "__pycache__",
    "build", "dist", "output", "vendor", ".cache", ".vscode",
    ".vs", ".idea", ".gradle", ".svn", ".hg", ".cargo"
)
$excludeExts = @(
    ".7z", ".avi", ".bin", ".bmp", ".bz2", ".class", ".db", ".dll", ".doc",
    ".docx", ".exe", ".flac", ".gif", ".gz", ".ico", ".jpeg", ".jpg", ".lock",
    ".mov", ".mp3", ".mp4", ".o", ".obj", ".pdf", ".png", ".ppt", ".pptx",
    ".pyc", ".pyo", ".rar", ".so", ".sqlite", ".svg", ".tar", ".wav", ".webp",
    ".xls", ".xlsx", ".xz", ".zip", ".toml", ".md", ".txt", ".log"
)

# 处理每个 crate
foreach ($crate in $selected) {
    # 正确的路径：$ProjectRoot/crates/$crate
    $cratePath = Join-Path $cratesDir $crate
    $outputFile = Join-Path $ProjectRoot "$($crate)_code_review.txt"
    
    Write-Host "${colorCyan}正在提取: $crate${colorReset}"
    
    if (!(Test-Path $cratePath)) {
        Write-Host "${colorRed}  错误: 路径不存在 - $cratePath${colorReset}"
        continue
    }
    
    if (!(Test-Path $cratePath -PathType Container)) {
        Write-Host "${colorRed}  错误: $crate 不是目录${colorReset}"
        continue
    }

    # 收集所有 .rs 文件
    $rsFiles = Get-ChildItem -Path $cratePath -Recurse -File -Filter "*.rs" |
        Where-Object {
            $file = $_
            # 排除目录
            $exclude = $false
            foreach ($dir in $excludeDirs) {
                if ($file.FullName -match [regex]::Escape("\$dir\")) {
                    $exclude = $true
                    break
                }
            }
            # 排除隐藏文件和备份文件
            if (!$exclude) {
                $exclude = $file.Name -match '^\.' -or $file.Name -match '\.bak$' -or $file.Name -match '~$'
            }
            !$exclude
        } |
        Sort-Object FullName
    
    if ($rsFiles.Count -eq 0) {
        Write-Host "${colorYellow}  警告: 未找到 .rs 文件${colorReset}"
        continue
    }
    
    Write-Host "  找到 $($rsFiles.Count) 个 Rust 文件"
    
    # 开始写入
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $header = @"
================================================================================
代码提取日志
生成时间: $timestamp
Crate: $crate
项目路径: $ProjectRoot
================================================================================

"@
    $header | Out-File -FilePath $outputFile -Encoding utf8 -Force
    
    # 写入文件树
    $fileTree = @"
文件树：
==================================================
"@
    $fileTree | Out-File -FilePath $outputFile -Encoding utf8 -Append
    
    # 生成树形结构
    $tree = @{}
    foreach ($file in $rsFiles) {
        $relPath = $file.FullName.Substring($cratePath.FullName.Length + 1)
        $parts = $relPath -split '\\'
        $current = $tree
        for ($i = 0; $i -lt $parts.Count - 1; $i++) {
            $part = $parts[$i]
            if ($current[$part] -isnot [hashtable]) {
                $current[$part] = @{}
            }
            $current = $current[$part]
        }
        $current[$parts[-1]] = $null
    }
    
    # 递归打印树
    function Write-Tree($node, $prefix = "") {
        $keys = $node.Keys | Sort-Object
        for ($i = 0; $i -lt $keys.Count; $i++) {
            $key = $keys[$i]
            $last = ($i -eq $keys.Count - 1)
            if ($last) {
                Write-Output "$prefix└── $key"
            } else {
                Write-Output "$prefix├── $key"
            }
            if ($node[$key] -is [hashtable]) {
                $newPrefix = if ($last) { "$prefix    " } else { "$prefix│   " }
                Write-Tree $node[$key] $newPrefix
            }
        }
    }
    
    Write-Tree $tree | Out-File -FilePath $outputFile -Encoding utf8 -Append
    "==================================================`n" | Out-File -FilePath $outputFile -Encoding utf8 -Append
    
    # 写入每个文件内容
    "================================================================================
# RS 文件
================================================================================
" | Out-File -FilePath $outputFile -Encoding utf8 -Append
    
    foreach ($file in $rsFiles) {
        $relPath = $file.FullName.Substring($cratePath.FullName.Length)
        $dir = Split-Path -Path $relPath -Parent
        if ($dir) {
            $dir = $dir.TrimStart('\')
            $dir = "$dir\"
        } else {
            $dir = ""
        }
        $fileName = $file.Name
        
        # 读取文件内容（处理可能的编码问题）
        try {
            $content = Get-Content -Path $file.FullName -Raw -ErrorAction Stop
            
            $header = @"

# File: $dir$fileName

```rust
// $($file.FullName.Replace('\', '/').Replace($ProjectRoot.Replace('\', '/'), ''))

$content
```"@

            $header | Out-File -FilePath $outputFile -Encoding utf8 -Append
        } catch {
            Write-Host "${colorRed}    错误: 无法读取文件 $($file.Name): $($_.Exception.Message)${colorReset}"
        }
    }
    
    # 写入 Cargo.toml
    $cargoToml = Join-Path $cratePath "Cargo.toml"
    if (Test-Path $cargoToml) {
        "`n================================================================================
# TOML 文件
================================================================================
" | Out-File -FilePath $outputFile -Encoding utf8 -Append
        
        try {
            $tomlContent = Get-Content -Path $cargoToml -Raw -ErrorAction Stop
            
            $tomlHeader = @"

# File: Cargo.toml

```toml
$tomlContent
```"@
            $tomlHeader | Out-File -FilePath $outputFile -Encoding utf8 -Append
        } catch {
            Write-Host "${colorRed}    错误: 无法读取 Cargo.toml: $($_.Exception.Message)${colorReset}"
        }
    }
    
    Write-Host "  ${colorGreen}✓ 完成: $outputFile${colorReset}"
    Write-Host ""
}

Write-Host "${colorGreen}=== 提取完成！${colorReset}`n"
Write-Host "生成的文件位于: $ProjectRoot"
Write-Host "接下来操作建议："
Write-Host "1. 检查 ${colorYellow}mh_runtime_code_review.txt${colorReset} - 核心抽象层实现"
Write-Host "2. 检查 ${colorYellow}mh_config_code_review.txt${colorReset} - 无泛型配置层"
Write-Host "3. 检查 ${colorYellow}mh_physics_code_review.txt${colorReset} - 验证 builder 迁移和硬编码 f64"
Write-Host "4. 检查 ${colorYellow}mh_mesh_code_review.txt${colorReset} - 索引类型使用"
Write-Host "5. 检查 ${colorYellow}mh_geo_code_review.txt${colorReset} - 错误类型定义位置"
Write-Host "6. 检查 ${colorYellow}mh_io_code_review.txt${colorReset} - 配置相关代码`n"