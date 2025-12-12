# refactor_scalar.ps1
# 完整的Scalar重构脚本：
# 1. 更新lib.rs导出RuntimeScalar
# 2. 替换所有 from_f64_lossless 调用为 from_f64().unwrap_or(S::ZERO)
# 3. 替换所有 use mh_core::Scalar 为 use mh_core::RuntimeScalar

param(
    [string]$ProjectRoot = "E:\Documents\mariHydro\marihydro"
)

Set-Location $ProjectRoot

Write-Host "=== MariHydro Scalar 重构脚本 ===" -ForegroundColor Cyan
Write-Host "项目根目录: $ProjectRoot" -ForegroundColor Gray

# ============================================================================
# Step 1: 更新 mh_core/src/lib.rs 导出
# ============================================================================
Write-Host "`n[Step 1] 更新 mh_core/src/lib.rs 导出..." -ForegroundColor Yellow

$libFile = "crates\mh_core\src\lib.rs"
if (Test-Path $libFile) {
    $content = Get-Content $libFile -Raw -Encoding UTF8
    
    # 替换 pub use scalar::Scalar 为 RuntimeScalar
    $content = $content -replace 'pub use scalar::Scalar;', 'pub use scalar::RuntimeScalar;'
    
    # 替换 prelude 中的 Scalar
    $content = $content -replace 'pub use crate::scalar::Scalar;', 'pub use crate::scalar::RuntimeScalar;'
    
    # 添加向后兼容别名（如果不存在）
    if ($content -notmatch 'RuntimeScalar as Scalar') {
        $content = $content -replace '(pub use scalar::RuntimeScalar;)', @"
`$1
// TODO(重构-Phase5): 删除此别名，统一使用RuntimeScalar
// 临时向后兼容，确保迁移期间代码可编译
#[allow(deprecated)]
#[deprecated(since = "0.2.0", note = "Use RuntimeScalar instead")]
pub use scalar::RuntimeScalar as Scalar;
"@
    }
    
    Set-Content $libFile -Value $content -NoNewline -Encoding UTF8
    Write-Host "  ✅ 更新 $libFile" -ForegroundColor Green
} else {
    Write-Host "  ❌ 找不到 $libFile" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 2: 更新 backend.rs 中的导入
# ============================================================================
Write-Host "`n[Step 2] 更新 backend.rs 导入..." -ForegroundColor Yellow

$backendFile = "crates\mh_core\src\backend.rs"
if (Test-Path $backendFile) {
    $content = Get-Content $backendFile -Raw -Encoding UTF8
    $content = $content -replace 'use crate::scalar::Scalar;', 'use crate::scalar::RuntimeScalar;'
    $content = $content -replace 'Scalar \+ Pod', 'RuntimeScalar + Pod'
    Set-Content $backendFile -Value $content -NoNewline -Encoding UTF8
    Write-Host "  ✅ 更新 $backendFile" -ForegroundColor Green
}

# ============================================================================
# Step 3: 全局替换 from_f64_lossless 调用
# ============================================================================
Write-Host "`n[Step 3] 替换 from_f64_lossless 调用..." -ForegroundColor Yellow

$files = Get-ChildItem -Path "crates" -Filter "*.rs" -Recurse
$replacementCount = 0
$fileCount = 0

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    if ($null -eq $content) { continue }
    
    $original = $content
    
    # 模式1: <B::Scalar as Scalar>::from_f64_lossless(value) 
    #     -> B::Scalar::from_f64(value).unwrap_or(B::Scalar::ZERO)
    $content = $content -replace '<B::Scalar as Scalar>::from_f64_lossless\(([^)]+)\)', 'B::Scalar::from_f64($1).unwrap_or(B::Scalar::ZERO)'
    
    # 模式2: <S as Scalar>::from_f64_lossless(value)
    #     -> S::from_f64($1).unwrap_or(S::ZERO)
    $content = $content -replace '<S as Scalar>::from_f64_lossless\(([^)]+)\)', 'S::from_f64($1).unwrap_or(S::ZERO)'
    
    # 模式3: S::from_f64_lossless(value)
    #     -> S::from_f64($1).unwrap_or(S::ZERO)
    $content = $content -replace '([A-Z][a-zA-Z0-9_]*)::from_f64_lossless\(([^)]+)\)', '$1::from_f64($2).unwrap_or($1::ZERO)'
    
    # 模式4: B::Scalar::from_f64_lossless(value)
    #     -> B::Scalar::from_f64($1).unwrap_or(B::Scalar::ZERO)
    $content = $content -replace 'B::Scalar::from_f64_lossless\(([^)]+)\)', 'B::Scalar::from_f64($1).unwrap_or(B::Scalar::ZERO)'
    
    if ($content -ne $original) {
        Set-Content $file.FullName -Value $content -NoNewline -Encoding UTF8
        $fileCount++
        # 统计替换次数
        $matches = [regex]::Matches($original, 'from_f64_lossless')
        $replacementCount += $matches.Count
    }
}

Write-Host "  ✅ 在 $fileCount 个文件中替换了 $replacementCount 处 from_f64_lossless" -ForegroundColor Green

# ============================================================================
# Step 4: 更新 use mh_core::Scalar 导入（可选，保留兼容别名后不需要）
# ============================================================================
Write-Host "`n[Step 4] 检查 Scalar 导入状态..." -ForegroundColor Yellow

$scalarImports = Get-ChildItem -Path "crates" -Filter "*.rs" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw -Encoding UTF8
    if ($content -match 'use mh_core::Scalar;') {
        $_.FullName
    }
}

if ($scalarImports) {
    Write-Host "  ℹ️ 以下文件使用 'use mh_core::Scalar'（向后兼容别名已启用）:" -ForegroundColor Cyan
    $scalarImports | ForEach-Object { Write-Host "    - $_" -ForegroundColor Gray }
} else {
    Write-Host "  ✅ 无需更新导入语句" -ForegroundColor Green
}

# ============================================================================
# Step 5: 验证 scalar.rs 中定义
# ============================================================================
Write-Host "`n[Step 5] 验证 scalar.rs 定义..." -ForegroundColor Yellow

$scalarFile = "crates\mh_core\src\scalar.rs"
$content = Get-Content $scalarFile -Raw -Encoding UTF8

if ($content -match 'pub trait RuntimeScalar') {
    Write-Host "  ✅ RuntimeScalar trait 已定义" -ForegroundColor Green
} else {
    Write-Host "  ❌ RuntimeScalar trait 未定义!" -ForegroundColor Red
    exit 1
}

if ($content -match 'fn from_f64_lossless') {
    Write-Host "  ⚠️ 警告: scalar.rs 仍包含 from_f64_lossless 定义，请手动移除" -ForegroundColor Yellow
}

if ($content -match 'fn from_config') {
    Write-Host "  ✅ from_config 方法已定义" -ForegroundColor Green
} else {
    Write-Host "  ❌ from_config 方法未定义!" -ForegroundColor Red
}

# ============================================================================
# 完成
# ============================================================================
Write-Host "`n=== 重构脚本执行完成 ===" -ForegroundColor Cyan
Write-Host "下一步: 运行 'cargo check' 验证编译" -ForegroundColor Yellow
