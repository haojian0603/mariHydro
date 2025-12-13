# marihydro/scripts/check_index_uniqueness.ps1
#
# CI 守护脚本：检测重复索引类型定义 (Windows PowerShell 版本)
#
# 本脚本确保 CellIndex, FaceIndex, NodeIndex 等索引类型仅在 mh_core 中定义，
# 防止在其他 crate 中出现重复定义。
#
# 统一定义位置：mh_core/src/indices.rs

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "=== Checking for duplicate index type definitions ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# 需要检查的索引类型
$IndexTypes = @(
    "CellIndex",
    "FaceIndex",
    "NodeIndex",
    "VertexIndex",
    "HalfEdgeIndex",
    "BoundaryIndex"
)

# 唯一允许定义的文件
$AllowedFile = "crates\mh_core\src\indices.rs"

$FoundIssues = 0

$CratesDir = Join-Path $ProjectRoot "crates"
$AllRsFiles = Get-ChildItem -Path $CratesDir -Filter "*.rs" -Recurse -File

foreach ($idxType in $IndexTypes) {
    Write-Host "Checking: $idxType" -ForegroundColor Yellow
    
    foreach ($file in $AllRsFiles) {
        $Content = Get-Content $file.FullName -Raw
        
        # 搜索 struct 定义模式
        if ($Content -match "^\s*(pub\s+)?struct\s+$idxType\b") {
            $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
            
            if ($RelPath -eq $AllowedFile) {
                Write-Host "  [OK] Canonical definition in $RelPath" -ForegroundColor Green
            }
            else {
                Write-Host "  [FAIL] Duplicate definition in $RelPath" -ForegroundColor Red
                $FoundIssues++
            }
        }
    }
    Write-Host ""
}

# 额外检查 type alias 重定义
Write-Host "Checking for conflicting type aliases..." -ForegroundColor Yellow

foreach ($idxType in $IndexTypes) {
    foreach ($file in $AllRsFiles) {
        $Lines = Get-Content $file.FullName
        $LineNum = 0
        
        foreach ($line in $Lines) {
            $LineNum++
            
            # 搜索 type alias
            if ($line -match "^\s*(pub\s+)?type\s+$idxType\s*=") {
                $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                
                if ($RelPath -ne $AllowedFile) {
                    Write-Host "  Warning: Type alias for $idxType in $RelPath`:$LineNum" -ForegroundColor Yellow
                    Write-Host "    Consider migrating to mh_core::$idxType"
                }
            }
        }
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
if ($FoundIssues -eq 0) {
    Write-Host "[OK] No duplicate index type definitions found!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "[FAIL] Found $FoundIssues duplicate index type definitions" -ForegroundColor Red
    Write-Host "Please remove duplicates and import from mh_core"
    exit 1
}
