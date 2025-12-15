<#
.SYNOPSIS
    自动添加缺失的 Vector2D/RuntimeScalar/DeviceBuffer 等 trait 导入
.DESCRIPTION
    在 mh_physics crate 的源文件中添加必要的 use 语句
    特性：跳过已存在的导入，支持预览模式
#>

param(
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf  # 预览模式，不实际修改文件
)

$cratePath = "crates\mh_physics\src"
$traitImports = @(
    "use mh_runtime::{Vector2D, RuntimeScalar, DeviceBuffer};",
    "use num_traits::{Float, FromPrimitive};"
)

# 获取所有 .rs 文件（排除测试和 mod.rs）
$rsFiles = Get-ChildItem -Path $cratePath -Recurse -Filter "*.rs" | 
           Where-Object { $_.Name -ne "mod.rs" -and $_.Directory.Name -ne "tests" }

foreach ($file in $rsFiles) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    
    # 检查是否已存在导入
    $needsImport = $false
    foreach ($import in $traitImports) {
        $importName = ($import -split '{')[1] -split '}')[0] -split ',' | ForEach-Object { $_.Trim() }
        foreach ($trait in $importName) {
            if ($content -notmatch "use.*$trait") {
                $needsImport = $true
                break
            }
        }
        if ($needsImport) { break }
    }
    
    if (-not $needsImport) { continue }

    Write-Host "[$($file.Name)] 添加 trait 导入" -ForegroundColor Cyan
    
    if ($WhatIn) {
        Write-Host "  [预览] 将添加: $($traitImports -join '`n  ')" -ForegroundColor Yellow
        continue
    }

    # 创建备份
    $backupPath = "$($file.FullName).backup"
    if (-not (Test-Path $backupPath)) {
        Copy-Item -Path $file.FullName -Destination $backupPath -Force
    }

    # 在第一个非注释行后插入导入
    $lines = $content -split "`n"
    $insertIndex = 0
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i].Trim()
        if ($line -and -not $line.StartsWith("//") -and -not $line.StartsWith("//!")) {
            $insertIndex = $i
            break
        }
    }

    # 插入导入语句
    $newContent = @()
    $newContent += $lines[0..$insertIndex]
    $newContent += ""
    $newContent += $traitImports
    $newContent += $lines[($insertIndex + 1)..($lines.Count - 1)]
    
    # 去除可能的重复空行
    $finalContent = ($newContent | Out-String) -replace "`n{3,}", "`n`n"
    
    Set-Content -Path $file.FullName -Value $finalContent -Encoding UTF8 -NoNewline
    Write-Host "  ✓ 已修复" -ForegroundColor Green
}

Write-Host "`n完成！检查了 $($rsFiles.Count) 个文件" -ForegroundColor Magenta