# marihydro/scripts/check_index_uniqueness.ps1
#
# CI 守护脚本：检测重复索引类型定义 (Windows PowerShell 版本)
#
# 本脚本确保 CellIndex, FaceIndex, NodeIndex 等索引类型仅在 mh_core 中定义，
# 防止在其他 crate 中出现重复定义。
#
# T06 scope only: geo / io / mesh / terrain / apps / tests

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

$FoundIssues = 0
$CanonicalIndexFile = Join-Path $ProjectRoot "crates\mh_mesh\src\halfedge\mesh.rs"

$ScanDirs = @(
    "crates\mh_geo\src",
    "crates\mh_io\src",
    "crates\mh_mesh\src",
    "crates\mh_terrain\src",
    "apps",
    "tests"
)

$AllRsFiles = @()
foreach ($dir in $ScanDirs) {
    $fullDir = Join-Path $ProjectRoot $dir
    if (Test-Path $fullDir) {
        $AllRsFiles += Get-ChildItem -Path $fullDir -Filter "*.rs" -Recurse -File
    }
}

foreach ($idxType in $IndexTypes) {
    Write-Host "Checking: $idxType" -ForegroundColor Yellow
    
    foreach ($file in $AllRsFiles) {
        $Lines = Get-Content $file.FullName
        $LineNum = 0
        
        foreach ($line in $Lines) {
            $LineNum++
            
            $TrimmedLine = $line.Trim()
            if ($TrimmedLine.StartsWith("//") -or $TrimmedLine.StartsWith("/*") -or $TrimmedLine.StartsWith("*")) {
                continue
            }
            
            # 搜索 struct 定义模式
            if ($line -match "^\s*(pub\s+)?struct\s+$idxType\b") {
                $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                if ($file.FullName -eq $CanonicalIndexFile) {
                    Write-Host "  [INFO] Canonical index struct for ${idxType}: $RelPath`:$LineNum" -ForegroundColor DarkGray
                } else {
                    Write-Host "  [WARN] Index struct found in T06 scope: $RelPath`:$LineNum" -ForegroundColor Yellow
                    $FoundIssues++
                }
            }
            
            # 搜索 type alias
            if ($line -match "^\s*(pub\s+)?type\s+$idxType\s*=") {
                $RelPath = $file.FullName.Replace($ProjectRoot + "\", "")
                if ($file.FullName -eq $CanonicalIndexFile) {
                    Write-Host "  [INFO] Canonical index alias for ${idxType}: $RelPath`:$LineNum" -ForegroundColor DarkGray
                } else {
                    Write-Host "  Warning: Type alias for $idxType in $RelPath`:$LineNum" -ForegroundColor Yellow
                    $FoundIssues++
                }
            }
        }
    }
    Write-Host ""
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
